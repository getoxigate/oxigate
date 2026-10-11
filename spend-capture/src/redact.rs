// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Credential redaction for recorded exchanges.
//!
//! Redaction runs in two passes over one exchange. First, [`KnownCredentials`] collects every
//! value held by a credential carrier — a credential header, a credential query parameter, the
//! userinfo of a URI — alongside any values the operator declared. Then [`redact_header`] and
//! [`redact_uri`] replace each carrier's value with [`REDACTED`] and scrub any collected value
//! that appears anywhere else in a header or URI.
//!
//! Bodies are never rewritten. A body that contains a collected value is reported by
//! [`KnownCredentials::found_in`], and the caller refuses the recording.

use http::HeaderMap;

use crate::recording::{FieldValue, HeaderField, end_to_end_headers};

/// Replacement written in place of a credential value.
pub const REDACTED: &str = "[REDACTED]";

/// Header names that always carry a credential.
const CREDENTIAL_HEADERS: &[&str] = &[
    "authorization",
    "proxy-authorization",
    "x-api-key",
    "api-key",
    "x-goog-api-key",
    "cookie",
    "set-cookie",
    "x-amz-security-token",
];

/// A header or query parameter name with any of these as a whole segment (split on every
/// non-alphanumeric character) is treated as a credential carrier. Whole-segment matching keeps
/// `x-session-token` in and the providers' `...-tokens` rate-limit headers out.
const CREDENTIAL_NAME_SEGMENTS: &[&str] = &[
    "apikey",
    "auth",
    "authorization",
    "cookie",
    "credential",
    "credentials",
    "key",
    "passwd",
    "password",
    "secret",
    "session",
    "sig",
    "signature",
    "token",
];

/// Response headers whose value is a URI, which can echo a credential in its query or userinfo.
const URI_VALUED_HEADERS: &[&str] = &["location", "content-location"];

/// Whether a header or query parameter with this name carries a credential. Case-insensitive.
pub fn is_credential_name(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    CREDENTIAL_HEADERS.contains(&lower.as_str())
        || lower
            .split(|c: char| !c.is_ascii_alphanumeric())
            .any(|segment| CREDENTIAL_NAME_SEGMENTS.contains(&segment))
}

/// The credential values seen in one exchange, plus any the operator declared.
#[derive(Clone, Default)]
pub struct KnownCredentials {
    values: Vec<Vec<u8>>,
}

/// Prints how many values are held, never the values.
impl std::fmt::Debug for KnownCredentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KnownCredentials")
            .field("values", &self.values.len())
            .finish()
    }
}

impl KnownCredentials {
    /// Starts from operator-declared credential values.
    pub fn with_declared<I, V>(declared: I) -> Self
    where
        I: IntoIterator<Item = V>,
        V: AsRef<[u8]>,
    {
        let mut known = Self::default();
        for value in declared {
            known.insert(value.as_ref());
        }
        known
    }

    /// Collects the credential values carried by one header, request or response.
    ///
    /// A credential header contributes its whole value, and the value without its framing: the
    /// token of an `<scheme> <token>` authorization value, and the unquoted value of each cookie
    /// pair (`Set-Cookie`: only the leading pair; its attributes are not secrets). Framing is
    /// never collected on its own, because every collected value is searched for in bodies with
    /// no length floor, and a fragment such as `=` would refuse nearly every recording.
    pub fn collect_header(&mut self, name: &str, value: &[u8]) {
        let lower = name.to_ascii_lowercase();
        if is_credential_name(&lower) {
            self.insert(value);
            let mut cookie_pairs = value.split(|b| *b == b';');
            match lower.as_str() {
                "cookie" => cookie_pairs.for_each(|pair| self.insert_cookie_value(pair)),
                "set-cookie" => cookie_pairs
                    .next()
                    .into_iter()
                    .for_each(|pair| self.insert_cookie_value(pair)),
                _ => {
                    if let Some(token) = auth_token(value) {
                        self.insert(token);
                    }
                }
            }
        } else if URI_VALUED_HEADERS.contains(&lower.as_str())
            && let Ok(uri) = std::str::from_utf8(value)
        {
            self.collect_uri(uri);
        }
    }

    /// Collects the credential values carried in a URI's query and userinfo, in both their
    /// written and percent-decoded forms.
    pub fn collect_uri(&mut self, uri: &str) {
        let parts = UriParts::split(uri);
        if let Some(userinfo) = parts.userinfo {
            self.insert_encoded(userinfo);
            if let Some((_, password)) = userinfo.split_once(':') {
                self.insert_encoded(password);
            }
        }
        for (name, value) in parts.query_pairs() {
            if is_credential_name(&String::from_utf8_lossy(&percent_decode(name))) {
                self.insert_encoded(value);
            }
        }
    }

    /// Whether `body` contains any collected value. There is no length floor: a short credential
    /// refuses a recording whose body happens to contain it, rather than reaching disk.
    pub fn found_in(&self, body: &[u8]) -> bool {
        self.values.iter().any(|value| find(body, value).is_some())
    }

    fn insert_encoded(&mut self, value: &str) {
        self.insert(value.as_bytes());
        self.insert(&percent_decode(value));
    }

    /// Inserts the value of one `name=value` cookie pair, without surrounding quotes.
    fn insert_cookie_value(&mut self, pair: &[u8]) {
        let Some(eq) = pair.iter().position(|b| *b == b'=') else {
            return;
        };
        let value = pair[eq + 1..].trim_ascii();
        let unquoted = value
            .strip_prefix(b"\"")
            .and_then(|inner| inner.strip_suffix(b"\""))
            .unwrap_or(value);
        self.insert(unquoted);
    }

    fn insert(&mut self, value: &[u8]) {
        let value = value.trim_ascii();
        if !value.is_empty() && !self.values.iter().any(|known| known == value) {
            self.values.push(value.to_vec());
        }
    }

    /// Marks every byte of `input` that lies inside an occurrence of any collected value,
    /// overlapping occurrences included. Matching runs against `input` only, never against
    /// redacted output, so a replacement marker can never be matched.
    fn coverage(&self, input: &[u8]) -> Vec<bool> {
        let mut covered = vec![false; input.len()];
        for value in &self.values {
            let mut from = 0;
            while let Some(at) = find(&input[from..], value) {
                let start = from + at;
                covered[start..start + value.len()].fill(true);
                from = start + 1;
            }
        }
        covered
    }

    /// `input` with every collected value replaced by [`REDACTED`]. Used for any text that is
    /// not a header or URI, such as a log line.
    pub fn scrub(&self, input: &[u8]) -> Vec<u8> {
        replace_covered(input, &self.coverage(input), &[])
    }
}

/// Copies `input`, writing one [`REDACTED`] in place of each run of covered bytes, and one at
/// each position in `empty_carriers` — a carrier with an empty value — unless that position
/// already touches a covered run. An empty carrier is marked where it is, never searched for.
fn replace_covered(input: &[u8], covered: &[bool], empty_carriers: &[usize]) -> Vec<u8> {
    let mut out = Vec::with_capacity(input.len());
    let mut i = 0;
    loop {
        let touches_run = (i > 0 && covered[i - 1]) || covered.get(i).copied().unwrap_or(false);
        if !touches_run && empty_carriers.contains(&i) {
            out.extend_from_slice(REDACTED.as_bytes());
        }
        if i == input.len() {
            return out;
        }
        if covered[i] {
            out.extend_from_slice(REDACTED.as_bytes());
            while i < input.len() && covered[i] {
                i += 1;
            }
        } else {
            out.push(input[i]);
            i += 1;
        }
    }
}

/// The recorded form of one header: its value replaced by [`REDACTED`] when the header is a
/// credential carrier, credential query and userinfo redacted when it is URI-valued, and every
/// collected credential value scrubbed from it otherwise.
pub fn redact_header(name: &str, value: &[u8], known: &KnownCredentials) -> FieldValue {
    let lower = name.to_ascii_lowercase();
    if is_credential_name(&lower) {
        return FieldValue::Text(REDACTED.to_owned());
    }
    if URI_VALUED_HEADERS.contains(&lower.as_str())
        && let Ok(uri) = std::str::from_utf8(value)
    {
        return FieldValue::Text(redact_uri(uri, known));
    }
    FieldValue::from_bytes(known.scrub(value))
}

/// The recorded form of a header map: its end-to-end headers ([`end_to_end_headers`]), each
/// redacted with [`redact_header`], in order.
pub fn recorded_headers(headers: &HeaderMap, known: &KnownCredentials) -> Vec<HeaderField> {
    end_to_end_headers(headers)
        .iter()
        .map(|(name, value)| HeaderField {
            name: name.as_str().to_owned(),
            value: redact_header(name.as_str(), value.as_bytes(), known),
        })
        .collect()
}

/// The recorded form of a URI (a path-and-query or an absolute URI): credential query values
/// and userinfo replaced by [`REDACTED`], and every collected credential value scrubbed.
pub fn redact_uri(uri: &str, known: &KnownCredentials) -> String {
    let parts = UriParts::split(uri);
    let mut covered = known.coverage(uri.as_bytes());
    let mut empty_carriers = Vec::new();
    let mut cover = |part: &str| {
        // Every part is a subslice of `uri`, so its offset is the pointer difference.
        let start = part.as_ptr() as usize - uri.as_ptr() as usize;
        if part.is_empty() {
            empty_carriers.push(start);
        } else {
            covered[start..start + part.len()].fill(true);
        }
    };
    if let Some(userinfo) = parts.userinfo {
        cover(userinfo);
    }
    for (name, value) in parts.query_pairs() {
        if is_credential_name(&String::from_utf8_lossy(&percent_decode(name))) {
            cover(value);
        }
    }
    let redacted = replace_covered(uri.as_bytes(), &covered, &empty_carriers);
    String::from_utf8_lossy(&redacted).into_owned()
}

/// Bytes that delimit a URI's path segments and query parameters. A redacted value in a URI is
/// matched only within one segment or parameter value, so a request cannot add or drop one under
/// a redaction marker.
pub const URI_DELIMITERS: &[u8] = b"/?&#";

/// Whether `observed` has the recorded form `recorded`: equal bytes outside each [`REDACTED`]
/// marker of `recorded`, and in place of each marker at least one byte, none of them in
/// `delimiters`. A marker stands for any credential value, including one the observer has no way
/// to recognise as a credential.
pub fn matches_redacted(recorded: &[u8], observed: &[u8], delimiters: &[u8]) -> bool {
    let marker = REDACTED.as_bytes();
    let mut literals = Vec::new();
    let mut rest = recorded;
    while let Some(at) = find(rest, marker) {
        literals.push(&rest[..at]);
        rest = &rest[at + marker.len()..];
    }
    let Some((first, middle)) = literals.split_first() else {
        return recorded == observed;
    };
    let last = rest;
    let Some(mut between) = observed
        .strip_prefix(*first)
        .and_then(|after| after.strip_suffix(last))
    else {
        return false;
    };
    let value = |gap: &[u8]| !gap.is_empty() && !gap.iter().any(|b| delimiters.contains(b));
    // Each literal after the first follows a marker, so at least one byte precedes it. Taking
    // the leftmost occurrence leaves the most room for the literals after it, and the shortest
    // value before it: if that value holds a delimiter, every longer one does too.
    for literal in middle {
        let Some(at) = between.get(1..).and_then(|after| {
            if literal.is_empty() {
                Some(0)
            } else {
                find(after, literal)
            }
        }) else {
            return false;
        };
        if !value(&between[..1 + at]) {
            return false;
        }
        between = &between[1 + at + literal.len()..];
    }
    value(between)
}

/// The credential-carrying parts of a URI, as subslices of it.
struct UriParts<'a> {
    /// Userinfo of an absolute or scheme-relative URI, without its trailing `@`.
    userinfo: Option<&'a str>,
    /// Query, without its leading `?` or any fragment.
    query: Option<&'a str>,
}

impl<'a> UriParts<'a> {
    fn split(uri: &'a str) -> Self {
        let before_fragment = uri.split_once('#').map_or(uri, |(head, _)| head);
        let (before_query, query) = match before_fragment.split_once('?') {
            Some((head, query)) => (head, Some(query)),
            None => (before_fragment, None),
        };
        // `scheme://authority…`, or a scheme-relative `//authority…`. A `://` later in a path is
        // not a scheme: the scheme must be the URI's leading run of scheme characters.
        let scheme_len = before_query
            .find("://")
            .filter(|&at| {
                at > 0
                    && before_query[..at]
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'+' | b'-' | b'.'))
            })
            .map(|at| at + 3)
            .or_else(|| before_query.starts_with("//").then_some(2));
        let userinfo = scheme_len.and_then(|len| {
            let after_scheme = &before_query[len..];
            let authority = &after_scheme[..after_scheme.find('/').unwrap_or(after_scheme.len())];
            authority.rfind('@').map(|at| &authority[..at])
        });
        Self { userinfo, query }
    }

    fn query_pairs(&self) -> impl Iterator<Item = (&'a str, &'a str)> {
        self.query
            .into_iter()
            .flat_map(|query| query.split('&'))
            .filter_map(|pair| pair.split_once('='))
    }
}

/// The token of an `<scheme> <token>` authorization value, e.g. `Bearer <token>`; `None` for
/// any other shape, whose whole value is collected instead.
fn auth_token(value: &[u8]) -> Option<&[u8]> {
    let value = value.trim_ascii();
    let space = value.iter().position(u8::is_ascii_whitespace)?;
    let (scheme, token) = (&value[..space], value[space..].trim_ascii());
    let single_word = !token.is_empty() && !token.iter().any(u8::is_ascii_whitespace);
    (!scheme.is_empty() && scheme.iter().all(u8::is_ascii_alphanumeric) && single_word)
        .then_some(token)
}

/// Decodes `%XX` escapes; anything else, including a malformed escape, is kept as is.
fn percent_decode(input: &str) -> Vec<u8> {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let decoded = (bytes[i] == b'%')
            .then(|| bytes.get(i + 1..i + 3))
            .flatten()
            .and_then(|hex| std::str::from_utf8(hex).ok())
            .and_then(|hex| u8::from_str_radix(hex, 16).ok());
        match decoded {
            Some(byte) => {
                out.push(byte);
                i += 3;
            }
            None => {
                out.push(bytes[i]);
                i += 1;
            }
        }
    }
    out
}

/// Position of the first occurrence of `needle` in `haystack`.
fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return None;
    }
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SECRET: &str = "sk-test-0123456789abcdef";

    fn text(value: FieldValue) -> String {
        match value {
            FieldValue::Text(text) => text,
            FieldValue::Bytes(bytes) => panic!("expected text, got {bytes:?}"),
        }
    }

    /// Collects from one header and returns its recorded value.
    fn record_header(name: &str, value: &str) -> (KnownCredentials, String) {
        let mut known = KnownCredentials::default();
        known.collect_header(name, value.as_bytes());
        let recorded = text(redact_header(name, value.as_bytes(), &known));
        (known, recorded)
    }

    #[test]
    fn debug_output_never_prints_a_credential() {
        let (known, _) = record_header("authorization", &format!("Bearer {SECRET}"));
        let printed = format!("{known:?}");
        // A derived `Debug` would print each value as a byte list, not as text.
        let as_bytes = format!("{:?}", SECRET.as_bytes());
        let inner = &as_bytes[1..as_bytes.len() - 1];
        assert!(!printed.contains(SECRET), "{printed}");
        assert!(!printed.contains(inner), "{printed}");
    }

    #[test]
    fn named_credential_headers_are_redacted_in_any_case() {
        for name in CREDENTIAL_HEADERS {
            for variant in [
                name.to_string(),
                name.to_ascii_uppercase(),
                title_case(name),
            ] {
                let (_, recorded) = record_header(&variant, SECRET);
                assert_eq!(recorded, REDACTED, "{variant}");
            }
        }
    }

    fn title_case(name: &str) -> String {
        name.split('-')
            .map(|part| {
                let mut chars = part.chars();
                chars.next().map_or_else(String::new, |first| {
                    first.to_ascii_uppercase().to_string() + chars.as_str()
                })
            })
            .collect::<Vec<_>>()
            .join("-")
    }

    #[test]
    fn unlisted_credential_headers_are_caught_by_the_name_pattern() {
        for name in [
            "X-Session-Token",
            "x-client-secret",
            "X-Upstream-Auth",
            "x_api_key",
        ] {
            let (_, recorded) = record_header(name, SECRET);
            assert_eq!(recorded, REDACTED, "{name}");
        }
    }

    #[test]
    fn accounting_and_tracing_headers_are_kept_verbatim() {
        for name in [
            "x-ratelimit-remaining-tokens",
            "x-ratelimit-limit-tokens",
            "anthropic-ratelimit-input-tokens-remaining",
            "anthropic-ratelimit-output-tokens-reset",
            "x-request-id",
            "request-id",
            "openai-organization",
            "openai-processing-ms",
            "anthropic-version",
            "content-type",
            "keep-alive",
        ] {
            assert!(!is_credential_name(name), "{name}");
            let (_, recorded) = record_header(name, "149984");
            assert_eq!(recorded, "149984", "{name}");
        }
    }

    #[test]
    fn a_bearer_token_is_found_in_a_body_without_its_scheme() {
        let (known, _) = record_header("Authorization", &format!("Bearer {SECRET}"));
        assert!(known.found_in(format!("{{\"echo\":\"{SECRET}\"}}").as_bytes()));
        assert!(!known.found_in(b"{\"echo\":\"Bearer\"}"));
    }

    #[test]
    fn every_cookie_value_is_collected() {
        let mut known = KnownCredentials::default();
        known.collect_header("cookie", b"a=first-cookie-value; b=second-cookie-value");
        assert!(known.found_in(b"..first-cookie-value.."));
        assert!(known.found_in(b"..second-cookie-value.."));
    }

    #[test]
    fn set_cookie_attributes_are_not_treated_as_credentials() {
        let mut known = KnownCredentials::default();
        known.collect_header(
            "set-cookie",
            b"sess=session-value-123; Domain=example.com; Path=/; HttpOnly",
        );
        assert!(known.found_in(b"session-value-123"));
        assert!(!known.found_in(b"see example.com"));
    }

    #[test]
    fn short_credentials_are_still_scanned() {
        let (known, recorded) = record_header("x-api-key", "s3cr3t");
        assert_eq!(recorded, REDACTED);
        assert!(known.found_in(b"{\"echo\":\"s3cr3t\"}"));
    }

    #[test]
    fn framing_around_a_credential_is_not_collected_on_its_own() {
        let mut known = KnownCredentials::default();
        known.collect_header("Authorization", b"Basic dXNlcjpwYXNzd29yZA==");
        assert!(known.found_in(b"dXNlcjpwYXNzd29yZA=="));
        assert!(!known.found_in(b"{\"a\":\"x==\",\"b\":\"Basic\"}"));
    }

    #[test]
    fn quoted_cookie_values_are_collected_without_their_quotes() {
        for (name, value) in [
            ("Cookie", "theme=dark; session=\"secret-12345\""),
            ("Set-Cookie", "session=\"secret-12345\"; Path=/; HttpOnly"),
        ] {
            let mut known = KnownCredentials::default();
            known.collect_header(name, value.as_bytes());
            assert!(known.found_in(b"echo: secret-12345"), "{name}");
            assert_eq!(
                text(redact_header("x-debug-echo", b"got secret-12345", &known)),
                format!("got {REDACTED}"),
                "{name}"
            );
        }
    }

    #[test]
    fn a_scheme_relative_location_is_redacted() {
        let location = "//user:secret-12345@example.com/next";
        let mut known = KnownCredentials::default();
        known.collect_header("Location", location.as_bytes());

        let recorded = text(redact_header("Location", location.as_bytes(), &known));
        assert_eq!(recorded, format!("//{REDACTED}@example.com/next"));
        assert!(known.found_in(b"echo: secret-12345"));
    }

    #[test]
    fn percent_encoded_userinfo_is_collected_decoded() {
        let location = "https://user:secret%2F12345@example.com/next";
        let mut known = KnownCredentials::default();
        known.collect_header("Location", location.as_bytes());

        assert!(known.found_in(b"echo: secret/12345"));
        assert!(known.found_in(b"echo: secret%2F12345"));
        let recorded = text(redact_header("Location", location.as_bytes(), &known));
        assert_eq!(recorded, format!("https://{REDACTED}@example.com/next"));
    }

    #[test]
    fn a_scheme_inside_a_path_is_not_an_authority() {
        let known = KnownCredentials::default();
        assert_eq!(
            redact_uri("/v1/https://a@b/c?x=1", &known),
            "/v1/https://a@b/c?x=1"
        );
    }

    #[test]
    fn declared_values_are_scanned() {
        let known = KnownCredentials::with_declared([SECRET]);
        assert!(known.found_in(format!("prefix{SECRET}suffix").as_bytes()));
        assert!(!known.found_in(&SECRET.as_bytes()[..SECRET.len() - 1]));
    }

    #[test]
    fn a_credential_query_parameter_is_redacted_and_collected() {
        let uri = format!("/v1beta/models/m:streamGenerateContent?alt=sse&key={SECRET}");
        let mut known = KnownCredentials::default();
        known.collect_uri(&uri);

        assert_eq!(
            redact_uri(&uri, &known),
            format!("/v1beta/models/m:streamGenerateContent?alt=sse&key={REDACTED}")
        );
        assert!(known.found_in(SECRET.as_bytes()));
    }

    #[test]
    fn a_percent_encoded_credential_is_collected_in_both_forms() {
        let mut known = KnownCredentials::default();
        known.collect_uri("/p?access_token=abc%2Fdef%2Bghi");
        assert!(known.found_in(b"abc%2Fdef%2Bghi"));
        assert!(known.found_in(b"abc/def+ghi"));
    }

    #[test]
    fn a_location_echoing_a_credential_is_redacted() {
        let location = format!("https://user:{SECRET}@api.example.com/v1/x?token={SECRET}&page=2");
        let mut known = KnownCredentials::default();
        known.collect_header("Location", location.as_bytes());

        let recorded = text(redact_header("Location", location.as_bytes(), &known));
        assert!(!recorded.contains(SECRET), "{recorded}");
        assert!(
            recorded.contains("api.example.com/v1/x?token="),
            "{recorded}"
        );
        assert!(recorded.ends_with("&page=2"), "{recorded}");
    }

    #[test]
    fn a_collected_value_echoed_in_an_unrelated_header_is_scrubbed() {
        let mut known = KnownCredentials::default();
        known.collect_header("x-api-key", SECRET.as_bytes());
        let echoed = format!("received {SECRET} ok");

        assert_eq!(
            text(redact_header("x-debug-echo", echoed.as_bytes(), &known)),
            format!("received {REDACTED} ok")
        );
        assert_eq!(
            redact_uri(&format!("/v1/{SECRET}/x"), &known),
            format!("/v1/{REDACTED}/x")
        );
    }

    #[test]
    fn credentials_differing_only_in_value_redact_identically() {
        let redacted = |uri: &str| {
            let mut known = KnownCredentials::default();
            known.collect_uri(uri);
            redact_uri(uri, &known)
        };
        assert_eq!(redacted("/p?key=A"), format!("/p?key={REDACTED}"));
        assert_eq!(redacted("/p?key=B"), format!("/p?key={REDACTED}"));
    }

    #[test]
    fn an_empty_credential_carrier_still_gets_a_marker() {
        let redacted = |uri: &str| {
            let mut known = KnownCredentials::default();
            known.collect_uri(uri);
            assert!(!known.found_in(b"any body at all"), "{uri}");
            redact_uri(uri, &known)
        };
        assert_eq!(redacted("/p?key="), redacted("/p?key=A"));
        assert_eq!(
            redacted("/p?key=&alt=sse"),
            format!("/p?key={REDACTED}&alt=sse")
        );
        assert_eq!(
            redacted("https://@host/x"),
            redacted("https://user:secret-12345@host/x")
        );
    }

    #[test]
    fn a_marker_is_never_rewritten_by_a_later_match() {
        let mut known = KnownCredentials::default();
        known.collect_header("x-api-key", b"long-credential-1");
        known.collect_header("cookie", b"a=D");
        assert_eq!(
            text(redact_header("x-echo", b"long-credential-1 D", &known)),
            format!("{REDACTED} {REDACTED}")
        );
    }

    #[test]
    fn overlapping_credentials_leave_no_fragment() {
        let known = KnownCredentials::with_declared(["abcd", "cdef"]);
        assert_eq!(
            text(redact_header("x-echo", b"xabcdefx", &known)),
            format!("x{REDACTED}x")
        );
    }

    #[test]
    fn a_marker_matches_any_value_and_nothing_else_differs() {
        let matches = |recorded: &str, observed: &str| {
            matches_redacted(
                recorded.replace('*', REDACTED).as_bytes(),
                observed.as_bytes(),
                b"",
            )
        };
        assert!(matches("plain", "plain"));
        assert!(!matches("plain", "plain "));
        assert!(matches("*", "anything"));
        assert!(matches("*", REDACTED));
        assert!(!matches("*", ""));
        assert!(matches("id=*;v=1", "id=S;v=1"));
        assert!(!matches("id=*;v=1", "id=S;v=2"));
        assert!(!matches("id=*;v=1", "id=;v=1"));
        assert!(matches("/v1/*/x?key=*", "/v1/a/x/y/x?key=b"));
        assert!(matches("a*b*c", "a-b-b-c"));
        assert!(!matches("a*b*c", "abc"));
        assert!(!matches("a*b*c", "a-bc"));
        assert!(matches("**", "xy"));
        assert!(!matches("**", "x"));
    }

    #[test]
    fn a_marker_in_a_uri_never_spans_a_segment_or_parameter() {
        let matches = |recorded: &str, observed: &str| {
            matches_redacted(
                recorded.replace('*', REDACTED).as_bytes(),
                observed.as_bytes(),
                URI_DELIMITERS,
            )
        };
        assert!(matches("/v1/foo?key=*", "/v1/foo?key=B"));
        assert!(!matches("/v1/foo?key=*", "/v1/foo?key=B&alt=sse"));
        assert!(matches("/v1/foo?key=*&alt=sse", "/v1/foo?key=B&alt=sse"));
        assert!(!matches(
            "/v1/foo?key=*&alt=sse",
            "/v1/foo?key=B&x=1&alt=sse"
        ));
        assert!(matches("/v1/*/x", "/v1/a/x"));
        assert!(!matches("/v1/*/x", "/v1/a/b/x"));
        assert!(!matches("/v1/foo?key=*", "/v1/foo?key=B#frag"));
    }

    #[test]
    fn a_non_utf8_value_survives_redaction_byte_for_byte() {
        let value = [b'o', b'k', 0xff];
        let known = KnownCredentials::default();
        assert_eq!(
            redact_header("x-odd", &value, &known),
            FieldValue::Bytes(value.to_vec())
        );
    }
}
