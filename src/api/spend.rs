// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Spend query handlers: GET /v1/spend/daily, /providers, /models .
//!
//! All endpoints are auth-gated (existing middleware stack). Tenant isolation
//! is enforced by extracting `org_id` from `RequestIdentity` — callers cannot
//! query another org's spend.

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::{Extension, Json};
use chrono::{Duration, NaiveDate, NaiveTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::api::AppState;
use crate::db::spend_reader::{query_daily_spend, query_spend_by_model, query_spend_by_provider};
use crate::domain::auth::RequestIdentity;

// --- Query params ---

/// Optional date-range query parameters for spend endpoints.
///
/// Both parameters are independently optional. Missing parameters default to a
/// 30-day window ending today. Dates must be in `YYYY-MM-DD` format (UTC).
#[derive(Debug, Deserialize)]
pub struct SpendQuery {
    from: Option<String>,
    to: Option<String>,
}

// --- Error type ---

#[derive(Debug, Error)]
pub enum SpendError {
    #[error("invalid date format: {0}")]
    InvalidDate(String),
    #[error("invalid date range: {0}")]
    InvalidRange(String),
    #[error("database error")]
    Db(#[from] sqlx::Error),
}

impl IntoResponse for SpendError {
    fn into_response(self) -> Response {
        let (status, msg) = match &self {
            SpendError::InvalidDate(s) => {
                (StatusCode::BAD_REQUEST, format!("invalid date format: {s}"))
            }
            SpendError::InvalidRange(s) => {
                (StatusCode::BAD_REQUEST, format!("invalid date range: {s}"))
            }
            SpendError::Db(err) => {
                tracing::error!(error = ?err, "spend handler: database error");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "internal error".to_string(),
                )
            }
        };
        (status, Json(serde_json::json!({ "error": msg }))).into_response()
    }
}

// --- Response DTOs ---

/// Response entry for `GET /v1/spend/daily`.
#[derive(Debug, Serialize)]
pub struct SpendDailyEntry {
    /// UTC calendar date in `YYYY-MM-DD` format.
    pub date: NaiveDate,
    /// Total cost in nano-USD (1 USD = 1 000 000 000 nano-USD).
    pub cost_nano_usd: i64,
}

/// Response entry for `GET /v1/spend/providers` and `GET /v1/spend/models`.
#[derive(Debug, Serialize)]
pub struct SpendDimensionEntry {
    /// Dimension value: provider name or model name depending on the endpoint.
    pub dimension: String,
    /// Total cost in nano-USD (1 USD = 1 000 000 000 nano-USD).
    pub cost_nano_usd: i64,
}

/// Response body for `GET /v1/spend/daily`.
#[derive(Debug, Serialize)]
pub struct SpendDailyResponse {
    pub data: Vec<SpendDailyEntry>,
}

/// Response body for `GET /v1/spend/providers` and `GET /v1/spend/models`.
#[derive(Debug, Serialize)]
pub struct SpendByDimensionResponse {
    pub data: Vec<SpendDimensionEntry>,
}

// --- Date range helper ---

/// Resolve user-supplied `from`/`to` strings into `(from_dt, to_exclusive_dt)` UTC bounds.
///
/// | `from` | `to` | window |
/// |--------|------|--------|
/// | absent | absent | `(today − 30 d, today)` |
/// | present | absent | `(parsed_from, today)` |
/// | absent | present | `(parsed_to − 30 d, parsed_to)` |
/// | present | present | `(parsed_from, parsed_to)` |
///
/// `to_exclusive_dt` is `to + 1 day` midnight UTC so the full `to` calendar day is included.
///
/// Validation: range ≤ 365 days; `from` ≤ `to`; `to` must not be `NaiveDate::MAX`
/// (which would make the exclusive upper bound unrepresentable).
fn resolve_date_range(
    q: SpendQuery,
) -> Result<(chrono::DateTime<Utc>, chrono::DateTime<Utc>), SpendError> {
    let today = Utc::now().date_naive();
    let fmt = "%Y-%m-%d";

    let from = match q.from {
        None => None,
        Some(s) => Some(
            NaiveDate::parse_from_str(&s, fmt).map_err(|_| SpendError::InvalidDate(s.clone()))?,
        ),
    };

    let to = match q.to {
        None => None,
        Some(s) => Some(
            NaiveDate::parse_from_str(&s, fmt).map_err(|_| SpendError::InvalidDate(s.clone()))?,
        ),
    };

    let (from_date, to_date) = match (from, to) {
        (None, None) => (today - Duration::days(30), today),
        (Some(f), None) => (f, today),
        (None, Some(t)) => (t - Duration::days(30), t),
        (Some(f), Some(t)) => (f, t),
    };

    if to_date < from_date {
        return Err(SpendError::InvalidRange(
            "'to' must not be before 'from'".to_string(),
        ));
    }
    if (to_date - from_date).num_days() > 365 {
        return Err(SpendError::InvalidRange(
            "range must not exceed 365 days".to_string(),
        ));
    }

    let to_exclusive = to_date
        .succ_opt()
        .ok_or_else(|| SpendError::InvalidDate(format!("{to_date} is too far in the future")))?;

    let from_dt = from_date.and_time(NaiveTime::MIN).and_utc();
    let to_dt = to_exclusive.and_time(NaiveTime::MIN).and_utc();

    Ok((from_dt, to_dt))
}

// --- Handlers ---

/// GET /v1/spend/daily — daily aggregated spend for the authenticated org.
#[tracing::instrument(skip_all, fields(
    org_id      = %identity.org_id,
    identity_id = %identity.id,
    from_dt     = tracing::field::Empty,
    to_dt       = tracing::field::Empty,
))]
pub async fn daily(
    State(state): State<AppState>,
    Extension(identity): Extension<RequestIdentity>,
    Query(q): Query<SpendQuery>,
) -> Result<Json<SpendDailyResponse>, SpendError> {
    let (from_dt, to_dt) = resolve_date_range(q)?;
    let span = tracing::Span::current();
    span.record("from_dt", from_dt.date_naive().to_string());
    span.record("to_dt", to_dt.date_naive().to_string());

    let pool = state.pool.read().await.clone();
    let rows = query_daily_spend(&pool, &identity.org_id, from_dt, to_dt).await?;
    let data = rows
        .into_iter()
        .map(|r| SpendDailyEntry {
            date: r.date,
            cost_nano_usd: r.cost_nano_usd,
        })
        .collect();
    Ok(Json(SpendDailyResponse { data }))
}

/// GET /v1/spend/providers — spend grouped by provider for the authenticated org.
#[tracing::instrument(skip_all, fields(
    org_id      = %identity.org_id,
    identity_id = %identity.id,
    from_dt     = tracing::field::Empty,
    to_dt       = tracing::field::Empty,
))]
pub async fn by_provider(
    State(state): State<AppState>,
    Extension(identity): Extension<RequestIdentity>,
    Query(q): Query<SpendQuery>,
) -> Result<Json<SpendByDimensionResponse>, SpendError> {
    let (from_dt, to_dt) = resolve_date_range(q)?;
    let span = tracing::Span::current();
    span.record("from_dt", from_dt.date_naive().to_string());
    span.record("to_dt", to_dt.date_naive().to_string());

    let pool = state.pool.read().await.clone();
    let rows = query_spend_by_provider(&pool, &identity.org_id, from_dt, to_dt).await?;
    let data = rows
        .into_iter()
        .map(|r| SpendDimensionEntry {
            dimension: r.dimension,
            cost_nano_usd: r.cost_nano_usd,
        })
        .collect();
    Ok(Json(SpendByDimensionResponse { data }))
}

/// GET /v1/spend/models — spend grouped by model for the authenticated org.
#[tracing::instrument(skip_all, fields(
    org_id      = %identity.org_id,
    identity_id = %identity.id,
    from_dt     = tracing::field::Empty,
    to_dt       = tracing::field::Empty,
))]
pub async fn by_model(
    State(state): State<AppState>,
    Extension(identity): Extension<RequestIdentity>,
    Query(q): Query<SpendQuery>,
) -> Result<Json<SpendByDimensionResponse>, SpendError> {
    let (from_dt, to_dt) = resolve_date_range(q)?;
    let span = tracing::Span::current();
    span.record("from_dt", from_dt.date_naive().to_string());
    span.record("to_dt", to_dt.date_naive().to_string());

    let pool = state.pool.read().await.clone();
    let rows = query_spend_by_model(&pool, &identity.org_id, from_dt, to_dt).await?;
    let data = rows
        .into_iter()
        .map(|r| SpendDimensionEntry {
            dimension: r.dimension,
            cost_nano_usd: r.cost_nano_usd,
        })
        .collect();
    Ok(Json(SpendByDimensionResponse { data }))
}

// --- Unit tests for resolve_date_range ---

#[cfg(test)]
mod tests {
    use super::*;

    fn q(from: Option<&str>, to: Option<&str>) -> SpendQuery {
        SpendQuery {
            from: from.map(str::to_owned),
            to: to.map(str::to_owned),
        }
    }

    #[test]
    fn both_absent_gives_30_day_window() {
        let today = Utc::now().date_naive();
        let (from_dt, to_dt) = resolve_date_range(q(None, None)).unwrap();
        let expected_from = (today - Duration::days(30))
            .and_time(NaiveTime::MIN)
            .and_utc();
        let expected_to = today.succ_opt().unwrap().and_time(NaiveTime::MIN).and_utc();
        assert_eq!(from_dt, expected_from);
        assert_eq!(to_dt, expected_to);
    }

    #[test]
    fn only_from_gives_from_to_today() {
        // Use a recent `from` to avoid tripping the 365-day cap.
        let today = Utc::now().date_naive();
        let from = today - Duration::days(10);
        let from_str = from.format("%Y-%m-%d").to_string();
        let (from_dt, to_dt) = resolve_date_range(q(Some(&from_str), None)).unwrap();
        assert_eq!(from_dt.date_naive(), from);
        assert_eq!(to_dt.date_naive(), today.succ_opt().unwrap());
    }

    #[test]
    fn only_to_gives_30_days_before_to() {
        let (from_dt, to_dt) = resolve_date_range(q(None, Some("2025-06-30"))).unwrap();
        assert_eq!(
            from_dt.date_naive(),
            NaiveDate::from_ymd_opt(2025, 5, 31).unwrap()
        );
        assert_eq!(
            to_dt.date_naive(),
            NaiveDate::from_ymd_opt(2025, 7, 1).unwrap()
        );
    }

    #[test]
    fn invalid_from_returns_invalid_date() {
        let err = resolve_date_range(q(Some("not-a-date"), None)).unwrap_err();
        assert!(matches!(err, SpendError::InvalidDate(_)));
    }

    #[test]
    fn invalid_to_returns_invalid_date() {
        let err = resolve_date_range(q(None, Some("2025-13-01"))).unwrap_err();
        assert!(matches!(err, SpendError::InvalidDate(_)));
    }

    #[test]
    fn from_after_to_returns_invalid_range() {
        let err = resolve_date_range(q(Some("2025-06-01"), Some("2025-01-01"))).unwrap_err();
        assert!(matches!(err, SpendError::InvalidRange(_)));
    }

    #[test]
    fn exactly_365_days_is_ok() {
        // 2025-01-01 to 2026-01-01 = 365 days difference.
        let result = resolve_date_range(q(Some("2025-01-01"), Some("2026-01-01")));
        assert!(result.is_ok(), "365-day range must be accepted");
    }

    #[test]
    fn range_366_days_returns_invalid_range() {
        let err = resolve_date_range(q(Some("2025-01-01"), Some("2026-01-02"))).unwrap_err();
        assert!(
            matches!(err, SpendError::InvalidRange(_)),
            "366-day range must be rejected"
        );
    }
}
