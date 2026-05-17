// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
// Metrics wrapper and tracing setup.
// Tracing subscriber init lives here; metrics definitions in `metrics.rs`.
//: Prometheus exporter is installed by `init_metrics()` below.

pub mod metrics;

use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use miette::Result;
use tracing_subscriber::{EnvFilter, Registry, fmt, prelude::*, reload};

/// Concrete type for runtime log-level reload.
pub type LogLevelHandle = reload::Handle<EnvFilter, Registry>;

fn yaml_or_default_filter(yaml_log_level: &str) -> EnvFilter {
    EnvFilter::try_new(yaml_log_level).unwrap_or_else(|_| EnvFilter::new("info"))
}

/// Install the Prometheus metrics exporter and return a scrape handle.
///
/// Must be called once at gateway startup before the axum server starts.
/// Wires all `metrics::counter!` / `metrics::histogram!` / `metrics::gauge!` calls
/// (including fallback/retry metrics) to the Prometheus recorder.
///
/// Sets explicit histogram buckets for `oxigate_request_duration_seconds`.
/// All other histograms use the exporter's defaults.
///
/// # Errors
/// Returns an error if the global recorder is already installed (double-init)
/// or if the bucket configuration is invalid. Fatal at startup — do not ignore.
pub fn init_metrics() -> Result<PrometheusHandle> {
    let handle = PrometheusBuilder::new()
        .set_buckets_for_metric(
            metrics_exporter_prometheus::Matcher::Full(
                crate::observability::metrics::REQUEST_DURATION_SECONDS.to_owned(),
            ),
            &[
                0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
            ],
        )
        .map_err(|e| miette::miette!("metrics bucket config failed: {e}"))?
        .install_recorder()
        .map_err(|e| miette::miette!("metrics recorder install failed: {e}"))?;
    Ok(handle)
}

/// Initialize the tracing subscriber (JSON format, env > YAML > info).
///
/// Returns a handle used to hot-reload the log level on SIGHUP.
pub fn init_tracing(yaml_log_level: &str) -> Result<LogLevelHandle> {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| yaml_or_default_filter(yaml_log_level));
    let (filter_layer, handle) = reload::Layer::new(filter);
    let subscriber = tracing_subscriber::registry()
        .with(filter_layer)
        .with(fmt::layer().json().flatten_event(true));
    subscriber.try_init().map_err(|e| miette::miette!("{e}"))?;
    Ok(handle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;
    use std::sync::{Arc, Mutex};

    use serde_json::Value;
    use tracing::{debug, info, info_span};
    use tracing_subscriber::fmt::writer::MakeWriter;

    #[derive(Clone, Default)]
    struct SharedBuffer(Arc<Mutex<Vec<u8>>>);

    struct BufferWriter(Arc<Mutex<Vec<u8>>>);

    impl io::Write for BufferWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            let mut guard = self.0.lock().expect("buffer lock poisoned");
            guard.extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl<'a> MakeWriter<'a> for SharedBuffer {
        type Writer = BufferWriter;

        fn make_writer(&'a self) -> Self::Writer {
            BufferWriter(Arc::clone(&self.0))
        }
    }

    impl SharedBuffer {
        fn json_lines(&self) -> Vec<Value> {
            let bytes = self.0.lock().expect("buffer lock poisoned").clone();
            String::from_utf8(bytes)
                .expect("valid utf-8 logs")
                .lines()
                .filter(|line| !line.trim().is_empty())
                .map(|line| serde_json::from_str(line).expect("valid json line"))
                .collect()
        }
    }

    fn with_test_subscriber(
        filter: EnvFilter,
        sink: SharedBuffer,
        test_fn: impl FnOnce(LogLevelHandle),
    ) {
        let (filter_layer, handle) = reload::Layer::new(filter);
        let subscriber = tracing_subscriber::registry().with(filter_layer).with(
            fmt::layer()
                .json()
                .flatten_event(true)
                .with_writer(sink.clone()),
        );
        tracing::subscriber::with_default(subscriber, || test_fn(handle));
    }

    #[test]
    fn test_json_log_contains_required_fields() {
        let sink = SharedBuffer::default();
        with_test_subscriber(EnvFilter::new("info"), sink.clone(), |_| {
            let span = info_span!("request");
            let _entered = span.enter();
            info!("hello");
        });

        let lines = sink.json_lines();
        assert_eq!(lines.len(), 1, "expected exactly one log event");
        let event = &lines[0];
        assert!(event.get("timestamp").is_some());
        assert!(event.get("level").is_some());
        assert!(event.get("target").is_some());
        assert!(event.get("message").is_some());
        assert!(event.get("span").is_some());
    }

    #[test]
    fn test_json_log_omits_span_without_active_span() {
        let sink = SharedBuffer::default();
        with_test_subscriber(EnvFilter::new("info"), sink.clone(), |_| {
            info!("hello-no-span");
        });

        let lines = sink.json_lines();
        assert_eq!(lines.len(), 1, "expected exactly one log event");
        let event = &lines[0];
        assert_eq!(
            event.get("message").and_then(Value::as_str),
            Some("hello-no-span")
        );
        assert!(event.get("span").is_none());
    }

    #[test]
    fn test_log_level_debug_filtered_at_info() {
        let sink = SharedBuffer::default();
        with_test_subscriber(EnvFilter::new("info"), sink.clone(), |_| {
            debug!("debug-hidden");
            info!("info-visible");
        });

        let lines = sink.json_lines();
        assert_eq!(lines.len(), 1);
        assert_eq!(
            lines[0].get("message").and_then(Value::as_str),
            Some("info-visible")
        );
    }

    #[test]
    fn test_warn_filter_suppresses_info() {
        let sink = SharedBuffer::default();
        with_test_subscriber(EnvFilter::new("warn"), sink.clone(), |_| {
            info!("suppressed");
            tracing::warn!("visible");
        });

        let lines = sink.json_lines();
        assert_eq!(lines.len(), 1);
        assert_eq!(
            lines[0].get("message").and_then(Value::as_str),
            Some("visible")
        );
    }

    #[test]
    fn test_yaml_or_default_filter_invalid_falls_back_to_info() {
        let sink = SharedBuffer::default();
        with_test_subscriber(yaml_or_default_filter("=invalid"), sink.clone(), |_| {
            debug!("suppressed-at-info");
            info!("visible-at-info");
        });

        let lines = sink.json_lines();
        assert_eq!(lines.len(), 1);
        assert_eq!(
            lines[0].get("message").and_then(Value::as_str),
            Some("visible-at-info")
        );
    }

    #[test]
    fn test_reload_handle_changes_level() {
        let sink = SharedBuffer::default();
        with_test_subscriber(EnvFilter::new("warn"), sink.clone(), |handle| {
            info!("suppressed-before-reload");
            handle
                .reload(EnvFilter::new("info"))
                .expect("reload should succeed");
            info!("visible-after-reload");
        });

        let lines = sink.json_lines();
        assert_eq!(lines.len(), 1);
        assert_eq!(
            lines[0].get("message").and_then(Value::as_str),
            Some("visible-after-reload")
        );
    }
}
