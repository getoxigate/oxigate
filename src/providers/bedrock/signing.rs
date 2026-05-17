// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! SigV4 request signing for AWS Bedrock .
//!
//! Service name is "bedrock" (NOT "bedrock-runtime"). The "bedrock-runtime" label is
//! boto3's client identifier; the SigV4 credential scope component is "bedrock".
//!
//! Each request gets a fresh x-amz-date — signed headers are never cached.

use std::time::SystemTime;

use aws_credential_types::Credentials;
use aws_sigv4::http_request::{
    PayloadChecksumKind, SignableBody, SignableRequest, SigningSettings, sign,
};
use aws_sigv4::sign::v4;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use secrecy::{ExposeSecret, SecretString};

use crate::domain::ports::ProviderError;

const BEDROCK_SERVICE: &str = "bedrock";

/// Holds resolved credentials for signing Bedrock requests.
///
/// Credentials are resolved once at adapter construction (startup) — not per-request.
/// `secret_access_key` and `session_token` are kept as `SecretString` so they are
/// zeroed on drop and never appear in `Debug` output.
pub struct BedrockSigner {
    access_key_id: String, // intentionally plain: key IDs are semi-public identifiers per AWS model
    secret_access_key: SecretString,
    session_token: Option<SecretString>,
    region: String,
}

impl BedrockSigner {
    pub fn new(
        access_key_id: String,
        secret_access_key: SecretString,
        session_token: Option<SecretString>,
        region: String,
    ) -> Self {
        Self {
            access_key_id,
            secret_access_key,
            session_token,
            region,
        }
    }

    /// Signs an HTTP request and returns the headers to add (Authorization, x-amz-date,
    /// x-amz-content-sha256, and optionally x-amz-security-token).
    ///
    /// `url` must be the full URL including scheme and path.
    /// `body` is the serialized request body (required to compute x-amz-content-sha256).
    pub fn sign_request(
        &self,
        method: &str,
        url: &str,
        body: &[u8],
    ) -> Result<HeaderMap, ProviderError> {
        // Build credentials and convert to Identity (required by aws-sigv4 1.x builder).
        let credentials = Credentials::new(
            &self.access_key_id,
            self.secret_access_key.expose_secret().as_str(),
            self.session_token
                .as_ref()
                .map(|t| t.expose_secret().clone()),
            None,
            "bedrock-adapter",
        );
        let identity = credentials.into();

        let mut settings = SigningSettings::default();
        settings.payload_checksum_kind = PayloadChecksumKind::XAmzSha256;

        let params = v4::SigningParams::builder()
            .identity(&identity)
            .region(self.region.as_str())
            .name(BEDROCK_SERVICE)
            .time(SystemTime::now())
            .settings(settings)
            .build()
            .map_err(|e| ProviderError::Auth(format!("sigv4 params: {e}")))?;

        // http_request::SigningParams<'_> wraps v4::SigningParams
        let signing_params: aws_sigv4::http_request::SigningParams<'_> = params.into();

        let headers_to_sign: Vec<(&str, &str)> = vec![("content-type", "application/json")];

        let signable = SignableRequest::new(
            method,
            url,
            headers_to_sign.into_iter(),
            SignableBody::Bytes(body),
        )
        .map_err(|e| ProviderError::Auth(format!("signable request: {e}")))?;

        // sign() returns SigningOutput<SigningInstructions>; .into_parts() yields (T, signature_string)
        let (instructions, _signature) = sign(signable, &signing_params)
            .map_err(|e| ProviderError::Auth(format!("sigv4 sign: {e}")))?
            .into_parts();

        // SigningInstructions::into_parts() yields (Vec<Header>, query_params)
        let (new_headers, _new_query_params) = instructions.into_parts();

        let mut header_map = HeaderMap::new();
        for header in new_headers {
            // Header::name() -> &'static str, Header::value() -> &str
            header_map.insert(
                HeaderName::from_static(header.name()),
                HeaderValue::from_str(header.value())
                    .map_err(|e| ProviderError::Auth(format!("invalid header value: {e}")))?,
            );
        }
        // Ensure content-type is always present (may already be in new_headers)
        header_map
            .entry(reqwest::header::CONTENT_TYPE)
            .or_insert(HeaderValue::from_static("application/json"));

        Ok(header_map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_signer() -> BedrockSigner {
        BedrockSigner::new(
            "AKIDEXAMPLE".to_string(),
            SecretString::new("wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY".to_string()),
            None,
            "us-east-1".to_string(),
        )
    }

    fn test_signer_with_token() -> BedrockSigner {
        BedrockSigner::new(
            "AKIDEXAMPLE".to_string(),
            SecretString::new("wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY".to_string()),
            Some(SecretString::new("session-token-xyz".to_string())),
            "us-east-1".to_string(),
        )
    }

    const TEST_URL: &str = "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-5-sonnet-20241022-v2:0/converse";
    const TEST_BODY: &[u8] =
        b"{\"messages\":[{\"role\":\"user\",\"content\":[{\"text\":\"hi\"}]}]}";

    #[test]
    fn test_sigv4_authorization_header_present() {
        let headers = test_signer()
            .sign_request("POST", TEST_URL, TEST_BODY)
            .unwrap();
        assert!(
            headers.contains_key("authorization"),
            "Authorization header must be present"
        );
        let auth = headers["authorization"].to_str().unwrap();
        assert!(
            auth.starts_with("AWS4-HMAC-SHA256 "),
            "Authorization must use AWS4-HMAC-SHA256 algorithm"
        );
    }

    #[test]
    fn test_sigv4_date_header_present() {
        let headers = test_signer()
            .sign_request("POST", TEST_URL, TEST_BODY)
            .unwrap();
        assert!(
            headers.contains_key("x-amz-date"),
            "x-amz-date header must be present"
        );
        let date = headers["x-amz-date"].to_str().unwrap();
        // x-amz-date must be 16 chars: YYYYMMDDTHHmmssZ
        assert!(
            date.len() == 16 && date.ends_with('Z') && date.contains('T'),
            "x-amz-date must be ISO 8601 basic format (YYYYMMDDTHHmmssZ), got: {date}"
        );
    }

    #[test]
    fn test_sigv4_content_sha256_header() {
        let headers = test_signer()
            .sign_request("POST", TEST_URL, TEST_BODY)
            .unwrap();
        assert!(
            headers.contains_key("x-amz-content-sha256"),
            "x-amz-content-sha256 must be present (Bedrock requires explicit hash)"
        );
        let sha = headers["x-amz-content-sha256"].to_str().unwrap();
        assert_eq!(sha.len(), 64, "x-amz-content-sha256 must be 64-char hex");
        assert!(
            sha.chars().all(|c| c.is_ascii_hexdigit()),
            "x-amz-content-sha256 must be hex"
        );
        assert_ne!(sha, "UNSIGNED-PAYLOAD");
    }

    #[test]
    fn test_sigv4_service_name_is_bedrock() {
        let headers = test_signer()
            .sign_request("POST", TEST_URL, TEST_BODY)
            .unwrap();
        let auth = headers["authorization"].to_str().unwrap();
        // Credential scope: YYYYMMDD/region/service/aws4_request
        assert!(
            auth.contains("/bedrock/aws4_request"),
            "credential scope must contain 'bedrock', not 'bedrock-runtime'. Got: {auth}"
        );
    }

    #[test]
    fn test_sigv4_session_token_header_when_set() {
        let headers = test_signer_with_token()
            .sign_request("POST", TEST_URL, TEST_BODY)
            .unwrap();
        assert!(
            headers.contains_key("x-amz-security-token"),
            "x-amz-security-token must be present when session token is configured"
        );
        assert_eq!(
            headers["x-amz-security-token"].to_str().unwrap(),
            "session-token-xyz"
        );
    }

    #[test]
    fn test_sigv4_no_session_token_header_when_absent() {
        let headers = test_signer()
            .sign_request("POST", TEST_URL, TEST_BODY)
            .unwrap();
        assert!(
            !headers.contains_key("x-amz-security-token"),
            "x-amz-security-token must be absent when no session token is configured"
        );
    }
}
