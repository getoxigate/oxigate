// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Gemini/Vertex AI authentication.
//!
//! API key (Gemini) and OAuth (Vertex) auth.

use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::config::SecretString;
use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
use reqwest::Url;
use secrecy::ExposeSecret;
use serde::Deserialize;
use thiserror::Error;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;

/// Auth error.
#[derive(Debug, Error)]
pub enum AuthError {
    #[error("missing or invalid service account JSON: {0}")]
    InvalidServiceAccount(String),
    #[error("token exchange failed: {0}")]
    TokenExchange(String),
    #[error("unauthorized: {0}")]
    Unauthorized(String),
    #[error("forbidden: {0}")]
    Forbidden(String),
}

/// Gemini API key — appends ?key= to URL.
#[derive(Clone)]
pub struct GeminiApiKey(pub SecretString);

impl GeminiApiKey {
    /// Appends `?key=<api_key>` to the request URL.
    pub fn apply_to_url(&self, url: &mut Url) {
        url.query_pairs_mut()
            .append_pair("key", self.0.expose_secret());
    }
}

/// Service account private key JSON (subset we need).
#[derive(Debug, Deserialize)]
struct ServiceAccountKeys {
    client_email: String,
    private_key: String,
    token_uri: String,
}

struct CachedToken {
    access_token: SecretString,
    expires_at: Instant,
}

/// Vertex AI OAuth token manager with background refresh.
pub struct VertexOAuthTokens {
    project: String,
    location: String,
    client_email: String,
    private_key: String,
    token_uri: String,
    token: Arc<RwLock<CachedToken>>,
    refresh_mutex: std::sync::Arc<tokio::sync::Mutex<()>>,
    http: reqwest::Client,
    cancel: CancellationToken,
}

impl VertexOAuthTokens {
    /// Creates Vertex OAuth tokens, fetches initial token, and spawns refresh task.
    pub async fn new(
        project: String,
        location: String,
        service_account_json: SecretString,
    ) -> Result<Self, AuthError> {
        let json_str = service_account_json.expose_secret();
        let keys: ServiceAccountKeys = serde_json::from_str(json_str)
            .map_err(|e| AuthError::InvalidServiceAccount(e.to_string()))?;

        let http = reqwest::Client::new();
        let token = Arc::new(RwLock::new(CachedToken {
            access_token: SecretString::new(String::new()),
            expires_at: Instant::now(),
        }));

        let tokens = Self {
            project: project.clone(),
            location: location.clone(),
            client_email: keys.client_email.clone(),
            private_key: keys.private_key.clone(),
            token_uri: keys.token_uri.clone(),
            token: Arc::clone(&token),
            refresh_mutex: std::sync::Arc::new(tokio::sync::Mutex::new(())),
            http: http.clone(),
            cancel: CancellationToken::new(),
        };

        tokens.refresh_token().await?;

        let cancel = tokens.cancel.clone();
        let project_clone = tokens.project.clone();
        let location_clone = tokens.location.clone();
        let client_email_clone = tokens.client_email.clone();
        let private_key_clone = tokens.private_key.clone();
        let token_uri_clone = tokens.token_uri.clone();
        let token_clone = Arc::clone(&tokens.token);
        let refresh_mutex_clone = std::sync::Arc::clone(&tokens.refresh_mutex);
        let http_clone = tokens.http.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(45));
            loop {
                tokio::select! {
                    _ = cancel.cancelled() => break,
                    _ = interval.tick() => {}
                }

                let guard = token_clone.read().await;
                let expires_at = guard.expires_at;
                drop(guard);

                if expires_at <= Instant::now() + Duration::from_secs(120) {
                    let _refresh_guard = refresh_mutex_clone.lock().await;
                    let guard = token_clone.read().await;
                    if guard.expires_at <= Instant::now() + Duration::from_secs(120) {
                        drop(guard);
                        let inner = VertexOAuthTokens {
                            project: project_clone.clone(),
                            location: location_clone.clone(),
                            client_email: client_email_clone.clone(),
                            private_key: private_key_clone.clone(),
                            token_uri: token_uri_clone.clone(),
                            token: Arc::clone(&token_clone),
                            refresh_mutex: refresh_mutex_clone.clone(),
                            http: http_clone.clone(),
                            cancel: CancellationToken::new(),
                        };
                        if let Err(e) = inner.refresh_token().await {
                            tracing::warn!(error = %e, "Vertex OAuth token refresh failed, will retry");
                        }
                    }
                }
            }
        });

        Ok(tokens)
    }

    /// Returns a valid token, refreshing if necessary.
    /// Uses a mutex to prevent thundering herd when multiple callers refresh concurrently.
    pub async fn get_token(&self) -> Result<SecretString, AuthError> {
        {
            let guard = self.token.read().await;
            if guard.expires_at > Instant::now() + Duration::from_secs(30) {
                return Ok(guard.access_token.clone());
            }
        }
        let _refresh_guard = self.refresh_mutex.lock().await;
        // Re-check after acquiring lock — another caller may have refreshed
        {
            let guard = self.token.read().await;
            if guard.expires_at > Instant::now() + Duration::from_secs(30) {
                return Ok(guard.access_token.clone());
            }
        }
        self.refresh_token().await?;
        let guard = self.token.read().await;
        Ok(guard.access_token.clone())
    }

    fn build_jwt(&self) -> Result<String, AuthError> {
        let now = std::time::SystemTime::now();
        let exp = now + Duration::from_secs(3600);
        let claims = serde_json::json!({
            "iss": self.client_email,
            "sub": self.client_email,
            "aud": self.token_uri,
            "iat": now.duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
            "exp": exp.duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
        });

        let key = EncodingKey::from_rsa_pem(self.private_key.as_bytes())
            .map_err(|e| AuthError::InvalidServiceAccount(format!("invalid private key: {e}")))?;

        let header = Header::new(Algorithm::RS256);
        encode(&header, &claims, &key).map_err(|e| AuthError::InvalidServiceAccount(e.to_string()))
    }

    async fn refresh_token(&self) -> Result<(), AuthError> {
        let jwt = self.build_jwt()?;

        let resp = self
            .http
            .post(&self.token_uri)
            .form(&[
                ("grant_type", "urn:ietf:params:oauth:grant-type:jwt-bearer"),
                ("assertion", &jwt),
            ])
            .send()
            .await
            .map_err(|e| AuthError::TokenExchange(e.to_string()))?;

        let status = resp.status();
        let body = resp
            .text()
            .await
            .map_err(|e| AuthError::TokenExchange(e.to_string()))?;

        if !status.is_success() {
            return Err(AuthError::TokenExchange(format!("{status}: {body}")));
        }

        let json: serde_json::Value =
            serde_json::from_str(&body).map_err(|e| AuthError::TokenExchange(e.to_string()))?;
        let access_token = json
            .get("access_token")
            .and_then(|v| v.as_str())
            .ok_or_else(|| AuthError::TokenExchange("no access_token in response".into()))?
            .to_string();

        let expires_in = json
            .get("expires_in")
            .and_then(|v| v.as_u64())
            .unwrap_or(3600);

        let mut guard = self.token.write().await;
        *guard = CachedToken {
            access_token: SecretString::new(access_token),
            expires_at: Instant::now() + Duration::from_secs(expires_in.saturating_sub(60)),
        };
        Ok(())
    }

    /// Cancels the background refresh task. Call on drop for graceful shutdown.
    pub fn cancel(&self) {
        self.cancel.cancel();
    }

    /// Test-only stub with a far-future token. Never touches the network.
    #[cfg(test)]
    pub fn new_stub() -> Self {
        Self {
            project: "test-project".into(),
            location: "us-central1".into(),
            client_email: "test@test.iam.gserviceaccount.com".into(),
            private_key: String::new(),
            token_uri: "http://unused".into(),
            token: Arc::new(RwLock::new(CachedToken {
                access_token: SecretString::new("stub-token".to_string()),
                expires_at: Instant::now() + Duration::from_secs(3600),
            })),
            refresh_mutex: Arc::new(tokio::sync::Mutex::new(())),
            http: reqwest::Client::new(),
            cancel: CancellationToken::new(),
        }
    }

    /// GCP project ID.
    #[must_use]
    pub fn project(&self) -> &str {
        &self.project
    }

    /// Vertex AI location/region.
    #[must_use]
    pub fn location(&self) -> &str {
        &self.location
    }
}

impl Drop for VertexOAuthTokens {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rsa::RsaPrivateKey;
    use rsa::pkcs1::{EncodeRsaPrivateKey, LineEnding};
    use rsa::rand_core::OsRng;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn test_rsa_key_pem() -> String {
        let mut rng = OsRng;
        let key = RsaPrivateKey::new(&mut rng, 2048).expect("rsa key generation must succeed");
        key.to_pkcs1_pem(LineEnding::LF)
            .expect("pkcs1 pem conversion must succeed")
            .to_string()
    }

    async fn make_tokens(
        mock: &MockServer,
        token_response: serde_json::Value,
    ) -> Result<VertexOAuthTokens, AuthError> {
        let token_uri = format!("{}/token", mock.uri().trim_end_matches('/'));
        Mock::given(method("POST"))
            .and(path("/token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(token_response))
            .mount(mock)
            .await;

        let sa_json = serde_json::json!({
            "client_email": "test@test.iam.gserviceaccount.com",
            "private_key": test_rsa_key_pem(),
            "token_uri": token_uri
        });

        VertexOAuthTokens::new(
            "proj".into(),
            "us-central1".into(),
            SecretString::new(sa_json.to_string()),
        )
        .await
    }

    /// get_token() returns cached token when expires_at > now + 30s.
    #[tokio::test]
    async fn test_cached_token_not_refreshed_before_expiry() {
        let mock = MockServer::start().await;
        let token_resp = serde_json::json!({
            "access_token": "cached-token",
            "expires_in": 3600
        });

        let tokens = make_tokens(&mock, token_resp)
            .await
            .expect("tokens must create");

        let t1 = tokens
            .get_token()
            .await
            .expect("first get_token must succeed");
        let t2 = tokens
            .get_token()
            .await
            .expect("second get_token must succeed");

        assert_eq!(t1.expose_secret(), t2.expose_secret());
        assert_eq!(t1.expose_secret(), "cached-token");

        tokens.cancel();
        let received = mock.received_requests().await.unwrap_or_default();
        assert_eq!(
            received.len(),
            1,
            "only initial refresh in new(), no refresh on get_token when cached valid"
        );
    }

    /// get_token() triggers refresh when expires_at - now < 60s.
    #[tokio::test]
    async fn test_cached_token_refreshed_near_expiry() {
        let mock = MockServer::start().await;
        let short_lived = serde_json::json!({
            "access_token": "short-token",
            "expires_in": 1
        });
        let refreshed = serde_json::json!({
            "access_token": "refreshed-token",
            "expires_in": 3600
        });

        Mock::given(method("POST"))
            .and(path("/token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(short_lived))
            .up_to_n_times(1)
            .mount(&mock)
            .await;
        Mock::given(method("POST"))
            .and(path("/token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(refreshed))
            .mount(&mock)
            .await;

        let token_uri = format!("{}/token", mock.uri().trim_end_matches('/'));
        let sa_json = serde_json::json!({
            "client_email": "test@test.iam.gserviceaccount.com",
            "private_key": test_rsa_key_pem(),
            "token_uri": token_uri
        });

        let tokens = VertexOAuthTokens::new(
            "proj".into(),
            "us-central1".into(),
            SecretString::new(sa_json.to_string()),
        )
        .await
        .expect("tokens must create");

        let t = tokens.get_token().await.expect("get_token must succeed");
        assert_eq!(
            t.expose_secret(),
            "refreshed-token",
            "must have refreshed to new token"
        );

        tokens.cancel();
        let received = mock.received_requests().await.unwrap_or_default();
        assert_eq!(
            received.len(),
            2,
            "initial refresh in new() + refresh in get_token when near expiry"
        );
    }
}
