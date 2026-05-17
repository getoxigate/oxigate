// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! AWS EventStream binary frame parser for Bedrock Converse streaming .
//!
//! Frame parsing and CRC32 validation is delegated to [aws-smithy-eventstream]
//! (`MessageFrameDecoder`). This module owns only the Bedrock-specific event
//! dispatch logic that maps decoded frames to `ConverseEvent` variants.

use aws_smithy_eventstream::frame::{DecodedFrame, MessageFrameDecoder};
use aws_smithy_types::event_stream::HeaderValue;
use bytes::BytesMut;

use crate::domain::ports::ProviderError;

use super::translate::bedrock_stop;

// AWS EventStream frame header names (Binary Format spec, §3.1).
mod header_name {
    pub const EVENT_TYPE: &str = ":event-type";
    pub const EXCEPTION_TYPE: &str = ":exception-type";
}

// AWS Converse streaming event type values (Converse streaming API spec).
mod event_type {
    pub const CONTENT_BLOCK_DELTA: &str = "contentBlockDelta";
    pub const MESSAGE_STOP: &str = "messageStop";
    pub const METADATA: &str = "metadata";
    pub const CONTENT_BLOCK_START: &str = "contentBlockStart";
    pub const CONTENT_BLOCK_STOP: &str = "contentBlockStop";
    pub const MESSAGE_START: &str = "messageStart";
    pub const INTERNAL_SERVER_EXCEPTION: &str = "internalServerException";
    pub const MODEL_STREAM_ERROR: &str = "modelStreamErrorException";
    pub const THROTTLING_EXCEPTION: &str = "throttlingException";
    pub const VALIDATION_EXCEPTION: &str = "validationException";
}

// JSON payload field names in Converse streaming event payloads.
mod converse_field {
    pub const STOP_REASON: &str = "stopReason";
}

/// A parsed Converse streaming event.
#[derive(Debug)]
pub enum ConverseEvent {
    ContentBlockDelta {
        text: String,
    },
    MessageStop {
        stop_reason: String,
    },
    Metadata {
        input_tokens: u64,
        output_tokens: u64,
    },
    Ignored,
    StreamError(ProviderError),
}

/// Stateful parser that accumulates bytes and extracts complete EventStream frames.
pub struct EventStreamParser {
    decoder: MessageFrameDecoder,
    buf: BytesMut,
}

impl Default for EventStreamParser {
    fn default() -> Self {
        Self::new()
    }
}

impl EventStreamParser {
    pub fn new() -> Self {
        Self {
            decoder: MessageFrameDecoder::new(),
            buf: BytesMut::new(),
        }
    }

    /// Appends `chunk` to the internal buffer and returns all fully-parsed events.
    ///
    /// Returns `Err` only on CRC corruption or structural protocol violation — caller should abort the stream.
    pub fn feed(&mut self, chunk: &[u8]) -> Result<Vec<ConverseEvent>, ProviderError> {
        self.buf.extend_from_slice(chunk);
        let mut events = Vec::new();

        loop {
            match self.decoder.decode_frame(&mut self.buf) {
                Ok(DecodedFrame::Incomplete) => break,
                Ok(DecodedFrame::Complete(msg)) => {
                    let event_kind = msg
                        .headers()
                        .iter()
                        .find(|h| {
                            let n = h.name().as_str();
                            n == header_name::EVENT_TYPE || n == header_name::EXCEPTION_TYPE
                        })
                        .and_then(|h| {
                            if let HeaderValue::String(s) = h.value() {
                                Some(s.as_str())
                            } else {
                                None
                            }
                        });
                    events.push(dispatch_event(event_kind, msg.payload().as_ref()));
                }
                Err(e) => {
                    return Err(ProviderError::ProviderUnavailable(format!(
                        "bedrock EventStream: frame decode error — {e}"
                    )));
                }
            }
        }

        Ok(events)
    }
}

/// Dispatches a parsed event based on its type string and payload JSON.
fn dispatch_event(event_kind: Option<&str>, payload: &[u8]) -> ConverseEvent {
    let json: serde_json::Value = match serde_json::from_slice(payload) {
        Ok(v) => v,
        Err(_) => return ConverseEvent::Ignored,
    };

    match event_kind {
        Some(event_type::CONTENT_BLOCK_DELTA) => {
            let text = json
                .get("delta")
                .and_then(|d| d.get("text"))
                .and_then(|t| t.as_str())
                .unwrap_or("")
                .to_string();
            ConverseEvent::ContentBlockDelta { text }
        }
        Some(event_type::MESSAGE_STOP) => {
            let stop_reason = json
                .get(converse_field::STOP_REASON)
                .and_then(|v| v.as_str())
                .unwrap_or(bedrock_stop::END_TURN)
                .to_string();
            ConverseEvent::MessageStop { stop_reason }
        }
        Some(event_type::METADATA) => {
            let input = json
                .pointer("/usage/inputTokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let output = json
                .pointer("/usage/outputTokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            ConverseEvent::Metadata {
                input_tokens: input,
                output_tokens: output,
            }
        }
        Some(event_type::INTERNAL_SERVER_EXCEPTION) | Some(event_type::MODEL_STREAM_ERROR) => {
            let msg = json
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown stream error")
                .to_string();
            ConverseEvent::StreamError(ProviderError::ProviderUnavailable(format!(
                "bedrock stream error: {}",
                msg
            )))
        }
        Some(event_type::THROTTLING_EXCEPTION) => {
            ConverseEvent::StreamError(ProviderError::RateLimited { retry_after: None })
        }
        Some(event_type::VALIDATION_EXCEPTION) => {
            let msg = json
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("validation error")
                .to_string();
            ConverseEvent::StreamError(ProviderError::InvalidRequest(msg))
        }
        Some(event_type::CONTENT_BLOCK_START)
        | Some(event_type::CONTENT_BLOCK_STOP)
        | Some(event_type::MESSAGE_START) => ConverseEvent::Ignored,
        _ => ConverseEvent::Ignored,
    }
}

/// Builds a complete EventStream frame. Available only under `test` or the `testing` feature.
#[cfg(any(test, feature = "testing"))]
pub fn build_frame(event_kind: &str, payload: &[u8]) -> Vec<u8> {
    use aws_smithy_eventstream::frame::write_message_to;
    use aws_smithy_types::event_stream::{Header, Message};

    let msg = Message::new_from_parts(
        vec![Header::new(
            header_name::EVENT_TYPE,
            HeaderValue::String(event_kind.to_string().into()),
        )],
        payload.to_vec(),
    );
    let mut buf = Vec::new();
    write_message_to(&msg, &mut buf).expect("EventStream frame encoding failed");
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    fn content_block_delta_payload(text: &str) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "contentBlockIndex": 0,
            "delta": {"text": text}
        }))
        .unwrap()
    }

    fn message_stop_payload(reason: &str) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({converse_field::STOP_REASON: reason})).unwrap()
    }

    fn metadata_payload(input: u64, output: u64) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "usage": {"inputTokens": input, "outputTokens": output, "totalTokens": input + output}
        }))
        .unwrap()
    }

    #[test]
    fn test_eventstream_parses_content_block_delta() {
        let frame = build_frame(
            event_type::CONTENT_BLOCK_DELTA,
            &content_block_delta_payload("Hello"),
        );
        let mut parser = EventStreamParser::new();
        let events = parser.feed(&frame).unwrap();
        assert_eq!(events.len(), 1);
        if let ConverseEvent::ContentBlockDelta { text } = &events[0] {
            assert_eq!(text, "Hello");
        } else {
            panic!("expected ContentBlockDelta");
        }
    }

    #[test]
    fn test_eventstream_parses_message_stop() {
        let frame = build_frame(
            event_type::MESSAGE_STOP,
            &message_stop_payload(bedrock_stop::END_TURN),
        );
        let mut parser = EventStreamParser::new();
        let events = parser.feed(&frame).unwrap();
        assert_eq!(events.len(), 1);
        if let ConverseEvent::MessageStop { stop_reason } = &events[0] {
            assert_eq!(stop_reason, bedrock_stop::END_TURN);
        } else {
            panic!("expected MessageStop");
        }
    }

    #[test]
    fn test_eventstream_parses_metadata_usage() {
        let frame = build_frame(event_type::METADATA, &metadata_payload(10, 5));
        let mut parser = EventStreamParser::new();
        let events = parser.feed(&frame).unwrap();
        assert_eq!(events.len(), 1);
        if let ConverseEvent::Metadata {
            input_tokens,
            output_tokens,
        } = events[0]
        {
            assert_eq!(input_tokens, 10);
            assert_eq!(output_tokens, 5);
        } else {
            panic!("expected Metadata");
        }
    }

    #[test]
    fn test_eventstream_rejects_bad_prelude_crc() {
        let mut frame = build_frame(
            event_type::CONTENT_BLOCK_DELTA,
            &content_block_delta_payload("x"),
        );
        // Corrupt the prelude CRC (bytes 8..12)
        frame[8] ^= 0xFF;
        let mut parser = EventStreamParser::new();
        let result = parser.feed(&frame);
        assert!(result.is_err(), "bad prelude CRC must return error");
    }

    #[test]
    fn test_eventstream_rejects_bad_message_crc() {
        let mut frame = build_frame(
            event_type::CONTENT_BLOCK_DELTA,
            &content_block_delta_payload("x"),
        );
        // Corrupt the message CRC (last 4 bytes)
        let len = frame.len();
        frame[len - 1] ^= 0xFF;
        let mut parser = EventStreamParser::new();
        let result = parser.feed(&frame);
        assert!(result.is_err(), "bad message CRC must return error");
    }

    #[test]
    fn test_eventstream_model_stream_error() {
        let payload = serde_json::to_vec(&serde_json::json!({
            "message": "upstream failure",
            "originalStatusCode": 503
        }))
        .unwrap();
        let frame = build_frame(event_type::MODEL_STREAM_ERROR, &payload);
        let mut parser = EventStreamParser::new();
        let events = parser.feed(&frame).unwrap();
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], ConverseEvent::StreamError(_)));
    }

    #[test]
    fn test_eventstream_throttling_exception_is_rate_limited_error() {
        let payload = serde_json::to_vec(&serde_json::json!({
            "message": "Too many requests"
        }))
        .unwrap();
        let frame = build_frame(event_type::THROTTLING_EXCEPTION, &payload);
        let mut parser = EventStreamParser::new();
        let events = parser.feed(&frame).unwrap();
        assert_eq!(events.len(), 1);
        assert!(
            matches!(
                &events[0],
                ConverseEvent::StreamError(ProviderError::RateLimited { .. })
            ),
            "throttlingException must map to RateLimited, not Ignored"
        );
    }

    #[test]
    fn test_eventstream_validation_exception_is_invalid_request_error() {
        let payload = serde_json::to_vec(&serde_json::json!({
            "message": "Input too long"
        }))
        .unwrap();
        let frame = build_frame(event_type::VALIDATION_EXCEPTION, &payload);
        let mut parser = EventStreamParser::new();
        let events = parser.feed(&frame).unwrap();
        assert_eq!(events.len(), 1);
        assert!(
            matches!(
                &events[0],
                ConverseEvent::StreamError(ProviderError::InvalidRequest(_))
            ),
            "validationException must map to InvalidRequest, not Ignored"
        );
    }

    #[test]
    fn test_eventstream_partial_frame_buffering() {
        let full_frame = build_frame(
            event_type::MESSAGE_STOP,
            &message_stop_payload(bedrock_stop::END_TURN),
        );
        let mid = full_frame.len() / 2;
        let (part1, part2) = full_frame.split_at(mid);

        let mut parser = EventStreamParser::new();
        // Feed first half — no complete frame yet.
        let events1 = parser.feed(part1).unwrap();
        assert!(events1.is_empty(), "partial frame must not yield events");
        // Feed second half — frame now complete.
        let events2 = parser.feed(part2).unwrap();
        assert_eq!(events2.len(), 1);
        assert!(matches!(events2[0], ConverseEvent::MessageStop { .. }));
    }
}
