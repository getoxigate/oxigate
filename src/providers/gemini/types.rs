// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Gemini/Vertex AI API wire types.
//!
//! Serde structs for request and response shapes.

use serde::{Deserialize, Serialize};

/// GenerateContent request body (Gemini API and Vertex AI).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiChatRequest {
    pub contents: Vec<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<GeminiTool>>,
    /// Tool calling mode and optional name filter. Omit when tool_choice is "none".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_config: Option<ToolConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GenerationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<Content>,
}

/// Gemini tool-use mode configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolConfig {
    pub function_calling_config: FunctionCallingConfig,
}

/// Function-calling mode and optional name filter.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FunctionCallingConfig {
    /// Mode string: "AUTO", "ANY", or "NONE".
    pub mode: String,
    /// When present, restricts the model to only these named functions.
    /// The full `function_declarations[]` is still sent; this is an additional filter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_function_names: Option<Vec<String>>,
}

/// Content block with role and parts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Content {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    pub parts: Vec<Part>,
}

/// Gemini 3.x thinking effort level.
/// Serialises as SCREAMING_SNAKE_CASE to match the Google wire format exactly.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ThinkingLevel {
    /// Minimal reasoning effort (Flash-Lite default).
    Minimal,
    /// Low reasoning effort.
    Low,
    /// Balanced reasoning effort (default when level not specified).
    Medium,
    /// Maximum reasoning effort; activates Deep Think on supported models.
    High,
}

/// Gemini ThinkingConfig — mutually exclusive fields by model generation.
/// Gemini 2.5: set `thinking_budget` only.
/// Gemini 3.x: set `thinking_level` only.
/// Both fields MUST NOT be set on the same request (Google returns 400).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThinkingConfig {
    /// Gemini 2.5 only. Token budget: -1 = dynamic, 0 = disable, N = explicit.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_budget: Option<i32>,
    /// Gemini 3.x only. Typed enum: Minimal, Low, Medium, High.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_level: Option<ThinkingLevel>,
}

/// Part of content: text, thought, function call, function response, or multimodal.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Part {
    /// Gemini 2.5 internal reasoning/thought chunk. MUST be listed BEFORE Text.
    Thought {
        thought: bool,
        text: String,
    },
    Text {
        text: String,
    },
    FunctionCall {
        #[serde(rename = "functionCall")]
        function_call: GeminiFunctionCall,
    },
    FunctionResponse {
        #[serde(rename = "functionResponse")]
        function_response: GeminiFunctionResponse,
    },
    /// Inline binary content (images, PDFs, audio). Billed as tokens.
    InlineData {
        #[serde(rename = "inlineData")]
        inline_data: InlineData,
    },
    /// Reference to a file uploaded via the File API or stored in GCS.
    FileData {
        #[serde(rename = "fileData")]
        file_data: FileData,
    },
    /// Model-generated executable code (from code execution feature).
    ExecutableCode {
        #[serde(rename = "executableCode")]
        executable_code: ExecutableCode,
    },
    /// Result of code execution (stdout/stderr).
    CodeExecutionResult {
        #[serde(rename = "codeExecutionResult")]
        code_execution_result: CodeExecutionResult,
    },
    /// Video clip metadata for trimming and frame-rate control.
    VideoMetadata {
        #[serde(rename = "videoMetadata")]
        video_metadata: VideoMetadata,
    },
}

/// Inline binary data (base64-encoded).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InlineData {
    pub mime_type: String,
    /// Base64-encoded bytes.
    pub data: String,
}

/// Reference to a file uploaded via the Gemini File API or GCS.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FileData {
    pub mime_type: String,
    pub file_uri: String,
}

/// Model-generated executable code block.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExecutableCode {
    pub language: String,
    pub code: String,
}

/// Result of executing code from an ExecutableCode part.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CodeExecutionResult {
    pub outcome: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
}

/// Video clipping and frame-rate metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VideoMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_offset: Option<VideoOffset>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_offset: Option<VideoOffset>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fps: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoOffset {
    pub seconds: i64,
    pub nanos: i32,
}

/// Function call from model.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiFunctionCall {
    pub name: String,
    pub args: Option<serde_json::Value>,
}

/// Function response from user.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiFunctionResponse {
    pub name: String,
    pub response: serde_json::Value,
}

/// Tool with function declarations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiTool {
    #[serde(rename = "functionDeclarations")]
    pub function_declarations: Vec<FunctionDeclaration>,
}

/// Function declaration for tools.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FunctionDeclaration {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

/// Generation config.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidate_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_config: Option<ThinkingConfig>,
}

/// GenerateContent response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiChatResponse {
    #[serde(default)]
    pub candidates: Vec<Candidate>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage_metadata: Option<UsageMetadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_feedback: Option<PromptFeedback>,
}

/// Prompt feedback (safety block).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptFeedback {
    #[serde(rename = "blockReason")]
    pub block_reason: Option<String>,
}

/// Candidate in response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Candidate {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Content>,
    #[serde(rename = "finishReason")]
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<u32>,
}

/// Token usage metadata.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct UsageMetadata {
    #[serde(rename = "promptTokenCount")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_token_count: Option<u32>,
    #[serde(rename = "candidatesTokenCount")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidates_token_count: Option<u32>,
    #[serde(rename = "totalTokenCount")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_token_count: Option<u32>,
    #[serde(rename = "cachedContentTokenCount")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_content_token_count: Option<u32>,
    #[serde(rename = "thoughtsTokenCount")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thoughts_token_count: Option<u32>,
}

/// Gemini API embed content request.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiEmbeddingRequest {
    pub content: EmbedContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_type: Option<&'static str>,
}

/// Content for embedding (Gemini API).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedContent {
    pub parts: Vec<EmbedPart>,
}

/// Part for embedding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedPart {
    pub text: String,
}

/// Vertex AI embedding predict request.
#[derive(Debug, Clone, Serialize)]
pub struct VertexEmbeddingRequest {
    pub instances: Vec<VertexEmbeddingInstance>,
}

/// Instance for Vertex embedding.
#[derive(Debug, Clone, Serialize)]
pub struct VertexEmbeddingInstance {
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_type: Option<&'static str>,
}

/// Token statistics returned per embedding element by Gemini API.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EmbedStatistics {
    pub token_count: u64,
}

/// A single embedding result from Gemini (single or batch).
#[derive(Debug, Clone, Deserialize)]
pub struct EmbedContentItem {
    pub values: Vec<f32>,
    pub statistics: Option<EmbedStatistics>,
}

/// One item inside a batchEmbedContents request.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiBatchEmbedItem {
    /// Full model path: "models/{model}".
    pub model: String,
    pub content: EmbedContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_type: Option<&'static str>,
}

/// Request body for batchEmbedContents.
#[derive(Debug, Clone, Serialize)]
pub struct GeminiBatchEmbedRequest {
    pub requests: Vec<GeminiBatchEmbedItem>,
}

/// Response body from batchEmbedContents.
#[derive(Debug, Clone, Deserialize)]
pub struct GeminiBatchEmbedResponse {
    pub embeddings: Vec<EmbedContentItem>,
}

/// Response body from embedContent (single input).
#[derive(Debug, Clone, Deserialize)]
pub struct GeminiSingleEmbedResponse {
    pub embedding: EmbedContentItem,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CRITICAL: Part::Thought must appear before Part::Text in the untagged enum.
    /// Otherwise {"thought": true, "text": "..."} silently deserialises as Part::Text,
    /// leaking internal reasoning into user-facing content (root cause of empty-stream bug).
    #[test]
    fn test_thought_part_deserialises_before_text_part() {
        let json = r#"{"thought": true, "text": "internal reasoning"}"#;
        let part: Part = serde_json::from_str(json).expect("must deserialise");
        match &part {
            Part::Thought { thought, text } => {
                assert!(*thought);
                assert_eq!(text, "internal reasoning");
            }
            _ => panic!("expected Part::Thought, got {:?}", part),
        }
    }

    #[test]
    fn test_text_part_deserialises_without_thought_field() {
        let json = r#"{"text": "hello"}"#;
        let part: Part = serde_json::from_str(json).expect("must deserialise");
        match &part {
            Part::Text { text } => assert_eq!(text, "hello"),
            _ => panic!("expected Part::Text, got {:?}", part),
        }
    }

    #[test]
    fn test_thinking_level_serialises_screaming_snake_case() {
        assert_eq!(
            serde_json::to_string(&ThinkingLevel::Medium).unwrap(),
            "\"MEDIUM\""
        );
        assert_eq!(
            serde_json::to_string(&ThinkingLevel::High).unwrap(),
            "\"HIGH\""
        );
        assert_eq!(
            serde_json::to_string(&ThinkingLevel::Low).unwrap(),
            "\"LOW\""
        );
        assert_eq!(
            serde_json::to_string(&ThinkingLevel::Minimal).unwrap(),
            "\"MINIMAL\""
        );
    }

    #[test]
    fn test_thinking_level_deserialises_from_screaming_snake_case() {
        let level: ThinkingLevel = serde_json::from_str("\"HIGH\"").expect("must deserialise");
        assert_eq!(level, ThinkingLevel::High);
    }

    #[test]
    fn test_thinking_config_wire_field_is_thinking_level_typed_enum() {
        let tc = ThinkingConfig {
            thinking_budget: None,
            thinking_level: Some(ThinkingLevel::High),
        };
        let json = serde_json::to_string(&tc).unwrap();
        assert!(json.contains("\"thinkingLevel\":\"HIGH\""));
        assert!(!json.contains("thinkingBudget"));
    }

    #[test]
    fn test_thinking_config_wire_field_is_thinking_budget_for_25() {
        let tc = ThinkingConfig {
            thinking_budget: Some(512),
            thinking_level: None,
        };
        let json = serde_json::to_string(&tc).unwrap();
        assert!(json.contains("\"thinkingBudget\":512"));
        assert!(!json.contains("thinkingLevel"));
    }

    #[test]
    fn test_unknown_thinking_level_string_returns_error_on_deserialise() {
        let result: Result<ThinkingLevel, _> = serde_json::from_str("\"SUPER_HIGH\"");
        assert!(result.is_err(), "invalid thinking_level must fail");
    }

    #[test]
    fn test_inline_data_part_serialises_correctly() {
        let part = Part::InlineData {
            inline_data: InlineData {
                mime_type: "image/png".into(),
                data: "abc".into(),
            },
        };
        let json = serde_json::to_value(&part).unwrap();
        assert_eq!(
            json.get("inlineData")
                .and_then(|d| d.get("mimeType"))
                .unwrap(),
            "image/png"
        );
        assert_eq!(
            json.get("inlineData").and_then(|d| d.get("data")).unwrap(),
            "abc"
        );
    }

    #[test]
    fn test_file_data_part_serialises_correctly() {
        let part = Part::FileData {
            file_data: FileData {
                mime_type: "application/pdf".into(),
                file_uri: "gs://bucket/file.pdf".into(),
            },
        };
        let json = serde_json::to_value(&part).unwrap();
        assert_eq!(
            json.get("fileData").and_then(|d| d.get("fileUri")).unwrap(),
            "gs://bucket/file.pdf"
        );
    }

    #[test]
    fn test_executable_code_part_round_trips() {
        let part = Part::ExecutableCode {
            executable_code: ExecutableCode {
                language: "PYTHON".into(),
                code: "print(1+1)".into(),
            },
        };
        let json = serde_json::to_string(&part).unwrap();
        let round: Part = serde_json::from_str(&json).unwrap();
        match &round {
            Part::ExecutableCode { executable_code } => {
                assert_eq!(executable_code.language, "PYTHON");
                assert_eq!(executable_code.code, "print(1+1)");
            }
            _ => panic!("expected ExecutableCode"),
        }
    }

    #[test]
    fn test_code_execution_result_round_trips() {
        let part = Part::CodeExecutionResult {
            code_execution_result: CodeExecutionResult {
                outcome: "OUTCOME_OK".into(),
                output: Some("2".into()),
            },
        };
        let json = serde_json::to_string(&part).unwrap();
        let round: Part = serde_json::from_str(&json).unwrap();
        match &round {
            Part::CodeExecutionResult {
                code_execution_result,
            } => {
                assert_eq!(code_execution_result.outcome, "OUTCOME_OK");
                assert_eq!(code_execution_result.output.as_deref(), Some("2"));
            }
            _ => panic!("expected CodeExecutionResult"),
        }
    }

    #[test]
    fn test_video_metadata_round_trips() {
        let part = Part::VideoMetadata {
            video_metadata: VideoMetadata {
                start_offset: Some(VideoOffset {
                    seconds: 5,
                    nanos: 0,
                }),
                end_offset: Some(VideoOffset {
                    seconds: 10,
                    nanos: 500_000_000,
                }),
                fps: Some(30),
            },
        };
        let json = serde_json::to_string(&part).unwrap();
        let round: Part = serde_json::from_str(&json).unwrap();
        match &round {
            Part::VideoMetadata { video_metadata } => {
                assert_eq!(video_metadata.start_offset.as_ref().unwrap().seconds, 5);
                assert_eq!(video_metadata.end_offset.as_ref().unwrap().seconds, 10);
                assert_eq!(video_metadata.fps, Some(30));
            }
            _ => panic!("expected VideoMetadata"),
        }
    }
}
