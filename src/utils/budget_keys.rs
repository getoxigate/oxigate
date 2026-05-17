// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Pure helpers for budget period keys and Redis key builders .
//!
//! Shared by middleware, db, and config validation. No I/O.

use chrono::{DateTime, Datelike, Duration, NaiveDate, NaiveDateTime, TimeZone, Utc, Weekday};
use chrono_tz::Tz;

use crate::config::BudgetDuration;
use crate::domain::ports::NanoUsd;

// Re-export so existing call sites (`use crate::utils::budget_keys::BudgetScope`) compile
// without change until they are updated to `crate::domain::ports::BudgetScope`.
pub use crate::domain::ports::BudgetScope;

/// Convert a intended local wall-clock instant to UTC without panicking on DST edge cases.
///
/// - **Ambiguous** local time (fall-back): `chrono::LocalResult::earliest` picks the first occurrence.
/// - **Nonexistent** local time (spring-forward gap, including some “midnight” dates): try the
///   same calendar intent at `naive + 1h`, `+ 2h`, … up to 47h, then anchor from local noon.
fn local_wall_clock_to_utc(tz: Tz, naive: NaiveDateTime) -> DateTime<Utc> {
    for offset_hours in 0_i64..48 {
        let cand = naive + Duration::hours(offset_hours);
        if let Some(dt) = tz.from_local_datetime(&cand).earliest() {
            return dt.to_utc();
        }
    }
    let date = naive.date();
    // Fallback: anchor from local noon if midnight is unresolvable.
    // UNREACHABLE IN PRACTICE: no active IANA timezone has a DST gap > 1 hour,
    // so the first loop (midnight + 0–47h) always resolves. This is defensive
    // coding for hypothetical future timezone anomalies or corrupted TZ database.
    let noon = date
        .and_hms_opt(12, 0, 0)
        .expect("local noon exists for budget reset fallback");
    for offset_hours in 0_i64..48 {
        let cand = noon + Duration::hours(offset_hours);
        if let Some(dt) = tz.from_local_datetime(&cand).earliest() {
            return dt.to_utc();
        }
    }
    tracing::error!(
        ?naive,
        "budget reset: could not map local wall clock to UTC; using UTC interpretation of naive instant"
    );
    Utc.from_utc_datetime(&naive)
}

/// TTL for legacy (non-period) per-identity spend keys — unchanged from pre-period-keyed behaviour.
pub const LEGACY_SPEND_KEY_TTL_SECS: u64 = 60 * 24 * 3600;

/// Parse duration strings into `BudgetDuration`.
///
/// Valid: `"1d"`, `"7d"`, `"30d"`, `"1mo"`. Anything else is an error.
pub fn parse_budget_duration(s: &str) -> Result<BudgetDuration, String> {
    match s.trim() {
        "1d" => Ok(BudgetDuration::Daily),
        "7d" => Ok(BudgetDuration::Weekly),
        "30d" | "1mo" => Ok(BudgetDuration::Monthly),
        other => Err(format!("invalid budget duration: '{other}'")),
    }
}

/// Next standardized reset instant in UTC.
///
/// - `None` → far future (no automatic reset)
/// - Daily → next local midnight in `tz`
/// - Weekly → next Monday 00:00 local in `tz`
/// - Monthly → first day of next calendar month 00:00 local in `tz`
#[must_use]
pub fn get_next_standardized_reset_time(
    duration: BudgetDuration,
    now: DateTime<Utc>,
    tz: Tz,
) -> DateTime<Utc> {
    let local_now = now.with_timezone(&tz);
    match duration {
        BudgetDuration::None => DateTime::<Utc>::MAX_UTC,
        BudgetDuration::Daily => {
            let tomorrow = local_now.date_naive() + Duration::days(1);
            let midnight = tomorrow
                .and_hms_opt(0, 0, 0)
                .expect("local midnight exists as naive time");
            local_wall_clock_to_utc(tz, midnight)
        }
        BudgetDuration::Weekly => {
            let days_until_monday: i64 = match local_now.weekday() {
                Weekday::Mon => 7,
                w => (7 - w.num_days_from_monday()) as i64,
            };
            let next_monday = local_now.date_naive() + Duration::days(days_until_monday);
            let midnight = next_monday
                .and_hms_opt(0, 0, 0)
                .expect("local midnight exists as naive time");
            local_wall_clock_to_utc(tz, midnight)
        }
        BudgetDuration::Monthly => {
            let (y, m) = if local_now.month() == 12 {
                (local_now.year() + 1, 1)
            } else {
                (local_now.year(), local_now.month() + 1)
            };
            let first_of_next = NaiveDate::from_ymd_opt(y, m, 1).expect("valid month start");
            let midnight = first_of_next
                .and_hms_opt(0, 0, 0)
                .expect("local midnight exists as naive time");
            local_wall_clock_to_utc(tz, midnight)
        }
    }
}

/// Period suffix for Redis keys (`""` when `duration` is `None`).
#[must_use]
pub fn period_key(duration: BudgetDuration, now: DateTime<Utc>, tz: Tz) -> String {
    let local = now.with_timezone(&tz);
    match duration {
        BudgetDuration::None => String::new(),
        BudgetDuration::Daily => local.format("%Y-%m-%d").to_string(),
        BudgetDuration::Weekly => {
            let days_since_monday = local.weekday().num_days_from_monday() as i64;
            let monday = local.date_naive() - Duration::days(days_since_monday);
            monday.format("%Y-%m-%d").to_string()
        }
        BudgetDuration::Monthly => local.format("%Y-%m").to_string(),
    }
}

/// Redis TTL for a per-identity spend key (2× period length, or 60 days when no period).
#[must_use]
pub fn spend_key_ttl_secs(duration: BudgetDuration) -> u64 {
    match duration {
        BudgetDuration::None => LEGACY_SPEND_KEY_TTL_SECS,
        BudgetDuration::Daily => 2 * 24 * 3600,
        BudgetDuration::Weekly => 14 * 24 * 3600,
        BudgetDuration::Monthly => 62 * 24 * 3600,
    }
}

/// Redis key for a per-identity spend counter.
///
/// When `period` is empty, format matches pre-period-keyed: `oxigate:org:{org_id}:spend:{identity_id}`.
#[must_use]
pub fn identity_spend_key(org_id: &str, identity_id: &str, period: &str) -> String {
    if period.is_empty() {
        format!("oxigate:org:{org_id}:spend:{identity_id}")
    } else {
        format!("oxigate:org:{org_id}:spend:{identity_id}:{period}")
    }
}

/// Redis key for a per-team spend counter.
#[must_use]
pub fn team_spend_key(org_id: &str, team: &str, period: &str) -> String {
    if period.is_empty() {
        format!("oxigate:org:{org_id}:team:{team}:spend")
    } else {
        format!("oxigate:org:{org_id}:team:{team}:spend:{period}")
    }
}

/// Redis key for a per-tag spend counter. `tag_kv` is the full `"key:value"` string,
/// e.g. `"project:chat-bot"` (produced by `format!("{k}:{v}")` from `RequestIdentity.tags`).
#[must_use]
pub fn tag_spend_key(org_id: &str, tag_kv: &str, period: &str) -> String {
    if period.is_empty() {
        format!("oxigate:org:{org_id}:tag:{tag_kv}:spend")
    } else {
        format!("oxigate:org:{org_id}:tag:{tag_kv}:spend:{period}")
    }
}

/// Format a `NanoUsd` spend value as a USD string with 6 decimal places.
///
/// Delegates to `NanoUsd::to_display_string` (domain/ports.rs, always compiled).
#[must_use]
#[inline]
pub fn nanos_to_usd_display(n: NanoUsd) -> String {
    n.to_display_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn parse_budget_duration_ok() {
        assert_eq!(parse_budget_duration("1d").unwrap(), BudgetDuration::Daily);
        assert_eq!(parse_budget_duration("7d").unwrap(), BudgetDuration::Weekly);
        assert_eq!(
            parse_budget_duration("30d").unwrap(),
            BudgetDuration::Monthly
        );
        assert_eq!(
            parse_budget_duration("1mo").unwrap(),
            BudgetDuration::Monthly
        );
    }

    #[test]
    fn parse_budget_duration_err() {
        assert!(parse_budget_duration("bad").is_err());
    }

    #[test]
    fn period_key_none_empty() {
        assert_eq!(
            period_key(BudgetDuration::None, Utc::now(), chrono_tz::UTC),
            ""
        );
    }

    #[test]
    fn period_key_daily_utc_boundary() {
        let t1 = Utc.with_ymd_and_hms(2026, 3, 19, 23, 59, 59).unwrap();
        assert_eq!(
            period_key(BudgetDuration::Daily, t1, chrono_tz::UTC),
            "2026-03-19"
        );
        let t2 = Utc.with_ymd_and_hms(2026, 3, 20, 0, 0, 0).unwrap();
        assert_eq!(
            period_key(BudgetDuration::Daily, t2, chrono_tz::UTC),
            "2026-03-20"
        );
    }

    #[test]
    fn period_key_daily_us_eastern() {
        let tz = chrono_tz::America::New_York;
        // November — New York is EST (UTC-5): 04:59 UTC is still previous local calendar day.
        let t1 = Utc.with_ymd_and_hms(2026, 11, 20, 4, 59, 59).unwrap();
        assert_eq!(period_key(BudgetDuration::Daily, t1, tz), "2026-11-19");
        let t2 = Utc.with_ymd_and_hms(2026, 11, 20, 5, 0, 0).unwrap();
        assert_eq!(period_key(BudgetDuration::Daily, t2, tz), "2026-11-20");
    }

    #[test]
    fn period_key_weekly_utc() {
        let wed = Utc.with_ymd_and_hms(2026, 3, 18, 0, 0, 0).unwrap();
        assert_eq!(
            period_key(BudgetDuration::Weekly, wed, chrono_tz::UTC),
            "2026-03-16"
        );
        let sun = Utc.with_ymd_and_hms(2026, 3, 22, 23, 59, 59).unwrap();
        assert_eq!(
            period_key(BudgetDuration::Weekly, sun, chrono_tz::UTC),
            "2026-03-16"
        );
        let mon = Utc.with_ymd_and_hms(2026, 3, 23, 0, 0, 0).unwrap();
        assert_eq!(
            period_key(BudgetDuration::Weekly, mon, chrono_tz::UTC),
            "2026-03-23"
        );
        let mon_same = Utc.with_ymd_and_hms(2026, 3, 16, 0, 0, 0).unwrap();
        assert_eq!(
            period_key(BudgetDuration::Weekly, mon_same, chrono_tz::UTC),
            "2026-03-16"
        );
    }

    #[test]
    fn period_key_weekly_year_end() {
        let mon = Utc.with_ymd_and_hms(2026, 12, 28, 0, 0, 0).unwrap();
        assert_eq!(
            period_key(BudgetDuration::Weekly, mon, chrono_tz::UTC),
            "2026-12-28"
        );
        let sun = Utc.with_ymd_and_hms(2027, 1, 3, 23, 59, 59).unwrap();
        assert_eq!(
            period_key(BudgetDuration::Weekly, sun, chrono_tz::UTC),
            "2026-12-28"
        );
        let mon2 = Utc.with_ymd_and_hms(2027, 1, 4, 0, 0, 0).unwrap();
        assert_eq!(
            period_key(BudgetDuration::Weekly, mon2, chrono_tz::UTC),
            "2027-01-04"
        );
    }

    #[test]
    fn period_key_monthly_utc() {
        let end_march = Utc.with_ymd_and_hms(2026, 3, 31, 23, 59, 59).unwrap();
        assert_eq!(
            period_key(BudgetDuration::Monthly, end_march, chrono_tz::UTC),
            "2026-03"
        );
        let apr = Utc.with_ymd_and_hms(2026, 4, 1, 0, 0, 0).unwrap();
        assert_eq!(
            period_key(BudgetDuration::Monthly, apr, chrono_tz::UTC),
            "2026-04"
        );
    }

    #[test]
    fn period_key_monthly_us_eastern_still_march() {
        let tz = chrono_tz::America::New_York;
        let t = Utc.with_ymd_and_hms(2026, 4, 1, 3, 59, 59).unwrap();
        assert_eq!(period_key(BudgetDuration::Monthly, t, tz), "2026-03");
    }

    #[test]
    fn identity_spend_key_with_period() {
        assert_eq!(
            identity_spend_key("acme", "key-1", ""),
            "oxigate:org:acme:spend:key-1"
        );
        assert_eq!(
            identity_spend_key("acme", "key-1", "2026-03"),
            "oxigate:org:acme:spend:key-1:2026-03"
        );
        assert_eq!(
            identity_spend_key("acme", "key-1", "2026-03-16"),
            "oxigate:org:acme:spend:key-1:2026-03-16"
        );
    }

    #[test]
    fn spend_key_ttl_secs_values() {
        assert_eq!(
            spend_key_ttl_secs(BudgetDuration::None),
            LEGACY_SPEND_KEY_TTL_SECS
        );
        assert_eq!(spend_key_ttl_secs(BudgetDuration::Daily), 2 * 24 * 3600);
        assert_eq!(spend_key_ttl_secs(BudgetDuration::Weekly), 14 * 24 * 3600);
        assert_eq!(spend_key_ttl_secs(BudgetDuration::Monthly), 62 * 24 * 3600);
    }

    #[test]
    fn get_next_daily() {
        let now = Utc.with_ymd_and_hms(2026, 3, 19, 12, 0, 0).unwrap();
        let next = get_next_standardized_reset_time(BudgetDuration::Daily, now, chrono_tz::UTC);
        assert_eq!(next, Utc.with_ymd_and_hms(2026, 3, 20, 0, 0, 0).unwrap());
    }

    #[test]
    fn get_next_weekly() {
        let now = Utc.with_ymd_and_hms(2026, 3, 18, 12, 0, 0).unwrap();
        let next = get_next_standardized_reset_time(BudgetDuration::Weekly, now, chrono_tz::UTC);
        assert_eq!(next, Utc.with_ymd_and_hms(2026, 3, 23, 0, 0, 0).unwrap());
    }

    #[test]
    fn get_next_monthly() {
        let now = Utc.with_ymd_and_hms(2026, 3, 15, 0, 0, 0).unwrap();
        let next = get_next_standardized_reset_time(BudgetDuration::Monthly, now, chrono_tz::UTC);
        assert_eq!(next, Utc.with_ymd_and_hms(2026, 4, 1, 0, 0, 0).unwrap());
    }

    /// DST spring/fall windows: must not panic on `single()`-style resolution.
    #[test]
    fn get_next_reset_no_panic_across_dst_new_york() {
        let tz = chrono_tz::America::New_York;
        for day in 1..=31_u32 {
            let t = Utc.with_ymd_and_hms(2024, 3, day, 12, 0, 0).unwrap();
            for d in [
                BudgetDuration::Daily,
                BudgetDuration::Weekly,
                BudgetDuration::Monthly,
            ] {
                let next = get_next_standardized_reset_time(d, t, tz);
                assert!(next > t, "next reset after {t:?} for {d:?}");
            }
        }
        for day in 1..=30_u32 {
            let t = Utc.with_ymd_and_hms(2024, 11, day, 12, 0, 0).unwrap();
            for d in [
                BudgetDuration::Daily,
                BudgetDuration::Weekly,
                BudgetDuration::Monthly,
            ] {
                let next = get_next_standardized_reset_time(d, t, tz);
                assert!(next > t, "next reset after {t:?} for {d:?}");
            }
        }
    }

    /// DST spring-forward at midnight: verifies fallback loop resolves non-existent midnight.
    /// Covers zones where clocks jump 00:00 -> 01:00 (e.g. Havana).
    #[test]
    fn get_next_daily_dst_gap_midnight_representative() {
        let tz = chrono_tz::America::Havana;
        // 2026-03-08: Cuba springs forward 00:00 -> 01:00 (midnight does not exist).
        let now = Utc.with_ymd_and_hms(2026, 3, 7, 12, 0, 0).unwrap();
        let next = get_next_standardized_reset_time(BudgetDuration::Daily, now, tz);
        // Expected: 01:00 local on DST day, which is 05:00 UTC (UTC-4 after spring-forward).
        assert_eq!(next, Utc.with_ymd_and_hms(2026, 3, 8, 5, 0, 0).unwrap());
    }

    #[test]
    fn team_spend_key_no_period() {
        assert_eq!(
            team_spend_key("acme", "engineering", ""),
            "oxigate:org:acme:team:engineering:spend"
        );
    }

    #[test]
    fn team_spend_key_with_period() {
        assert_eq!(
            team_spend_key("acme", "engineering", "2026-03"),
            "oxigate:org:acme:team:engineering:spend:2026-03"
        );
    }

    #[test]
    fn tag_spend_key_no_period() {
        assert_eq!(
            tag_spend_key("acme", "project:chat-bot", ""),
            "oxigate:org:acme:tag:project:chat-bot:spend"
        );
    }

    #[test]
    fn tag_spend_key_with_period() {
        assert_eq!(
            tag_spend_key("acme", "project:chat-bot", "2026-03"),
            "oxigate:org:acme:tag:project:chat-bot:spend:2026-03"
        );
    }

    #[test]
    fn budget_scope_identity_warn_key() {
        let key = BudgetScope::Identity("abc-uuid".to_owned()).warn_dedup_key("acme", 80);
        assert_eq!(key, "oxigate:budget:warned:acme:identity:abc-uuid:80");
    }

    #[test]
    fn budget_scope_team_warn_key() {
        let key = BudgetScope::Team("engineering".to_owned()).warn_dedup_key("acme", 90);
        assert_eq!(key, "oxigate:budget:warned:acme:team:engineering:90");
    }

    #[test]
    fn budget_scope_tag_warn_key() {
        // Tag kv contains a colon — canonical format preserves it verbatim.
        let key = BudgetScope::Tag("project:chat-bot".to_owned()).warn_dedup_key("acme", 100);
        assert_eq!(key, "oxigate:budget:warned:acme:tag:project:chat-bot:100");
    }

    #[test]
    fn budget_scope_sort_key() {
        // Discriminant ordering: Identity < Team < Tag
        assert!(
            BudgetScope::Identity("z".to_owned()).sort_key()
                < BudgetScope::Team("a".to_owned()).sort_key()
        );
        assert!(
            BudgetScope::Team("z".to_owned()).sort_key()
                < BudgetScope::Tag("a".to_owned()).sort_key()
        );
        // Within Tag: alphabetical by kv string
        assert!(
            BudgetScope::Tag("a:x".to_owned()).sort_key()
                < BudgetScope::Tag("b:x".to_owned()).sort_key()
        );
    }
}
