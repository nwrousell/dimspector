use miette::{Diagnostic, SourceSpan};
use thiserror::Error;
use tower_lsp::lsp_types::{Diagnostic as LspDiagnostic, DiagnosticSeverity, Position, Range};

use crate::analysis::{DimVar, Shape};

#[derive(Diagnostic, Error, Debug, Clone)]
pub enum ShapeError {
    #[error("Mismatched dims: {dim1} != {dim2}")]
    #[diagnostic(code(shape::mismatched_dims))]
    MismatchedDims {
        dim1: DimVar,
        dim2: DimVar,
        #[label("mismatch occurs here")]
        span: SourceSpan,
        // span: SourceSpan,
    },

    #[error("Can't infer return shape")]
    UninferrableCall {},

    #[error("Dimension {dim_ref} out of range for Tensor of rank {rank}")]
    #[diagnostic(code(shape::mismatched_dims))]
    DimOutRange { dim_ref: i64, rank: usize },

    #[error("Can't reshape {src} to {tgt}")]
    #[diagnostic(code(shape::mismatched_dims))]
    BadReshape {
        src: Shape,
        tgt: Shape,
        // #[label("mismatch occurs here")]
        // span: SourceSpan,
    },

    #[error(
        "Rank of tensor one ({tensor_1}), {rank_1}, does not equal rank of tensor two ({tensor_2}), {rank_2}"
    )]
    UnequalRank {
        tensor_1: Shape,
        tensor_2: Shape,
        rank_1: usize,
        rank_2: usize,
        #[label("mismatch occurs here")]
        span: SourceSpan,
    },
}

impl ShapeError {
    pub fn mismatched(dim1: &DimVar, dim2: &DimVar, span: SourceSpan) -> ShapeError {
        Self::MismatchedDims {
            dim1: dim1.clone(),
            dim2: dim2.clone(),
            span,
        }
    }

    pub fn unequal_rank(
        tensor_1: &Shape,
        tensor_2: &Shape,
        rank_1: usize,
        rank_2: usize,
        span: SourceSpan,
    ) -> ShapeError {
        Self::UnequalRank {
            tensor_1: tensor_1.clone(),
            tensor_2: tensor_2.clone(),
            rank_1,
            rank_2,
            span,
        }
    }

    /// Convert SourceSpan byte offsets to LSP Range (line/character positions)
    fn span_to_range(span: &SourceSpan, file_content: &str) -> Range {
        let start_offset = span.offset().into();
        let end_offset = start_offset + span.len();

        let mut line = 0;
        let mut character = 0;

        // Find start position
        for (idx, ch) in file_content.char_indices() {
            if idx >= start_offset {
                break;
            }
            if ch == '\n' {
                line += 1;
                character = 0;
            } else {
                character += 1;
            }
        }

        let start = Position {
            line: line as u32,
            character: character as u32,
        };

        // Find end position
        let mut end_line = line;
        let mut end_character = character;
        for (idx, ch) in file_content.char_indices() {
            if idx >= end_offset {
                break;
            }
            if idx >= start_offset {
                if ch == '\n' {
                    end_line += 1;
                    end_character = 0;
                } else {
                    end_character += 1;
                }
            }
        }

        let end = Position {
            line: end_line as u32,
            character: end_character as u32,
        };

        Range { start, end }
    }

    /// Convert this ShapeError to an LSP Diagnostic
    /// Requires file_content to convert byte offsets to line/character positions
    pub fn to_diagnostic(&self, file_content: &str) -> Option<LspDiagnostic> {
        let message = self.to_string();
        let severity = DiagnosticSeverity::ERROR;

        // Get range from span if available
        let range = match self {
            ShapeError::MismatchedDims { span, .. } => Self::span_to_range(span, file_content),
            ShapeError::UnequalRank { span, .. } => Self::span_to_range(span, file_content),
            // For errors without spans, use a default range at the start of the file
            ShapeError::UninferrableCall {}
            | ShapeError::DimOutRange { .. }
            | ShapeError::BadReshape { .. } => Range {
                start: Position {
                    line: 0,
                    character: 0,
                },
                end: Position {
                    line: 0,
                    character: 0,
                },
            },
        };

        Some(LspDiagnostic {
            range,
            severity: Some(severity),
            code: None,
            code_description: None,
            source: Some("dimspector".to_string()),
            message,
            related_information: None,
            tags: None,
            data: None,
        })
    }
}
