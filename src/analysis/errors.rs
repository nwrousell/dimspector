use miette::{Diagnostic, SourceSpan};
use thiserror::Error;
use tower_lsp::lsp_types::{Diagnostic as LspDiagnostic, DiagnosticSeverity, Position, Range};

use crate::analysis::{DimVar, Shape};

#[derive(Diagnostic, Error, Debug, Clone)]
pub enum ShapeError {
    /// Generic dimension mismatch (kept for backwards compatibility, prefer specific errors)
    #[error("Mismatched dims: {dim1} != {dim2}")]
    #[diagnostic(code(shape::mismatched_dims))]
    MismatchedDims {
        dim1: DimVar,
        dim2: DimVar,
        #[label("mismatch occurs here")]
        span: SourceSpan,
    },

    /// Argument shape doesn't match parameter's expected shape
    #[error("expected parameter `{param_name}` with shape {expected}, got {actual}")]
    #[diagnostic(code(shape::signature_param_mismatch))]
    SignatureParamMismatch {
        func_name: String,
        param_name: String,
        expected: Shape,
        actual: Shape,
        #[label("argument has shape {actual}")]
        span: SourceSpan,
    },

    /// Same dimension variable resolved to different values across arguments
    #[error("in `{func_name}`: dimension `{dimvar_name}` used inconsistently")]
    #[diagnostic(code(shape::inconsistent_dimvars))]
    InconsistentDimVars {
        func_name: String,
        dimvar_name: String,
        first_param_name: String,
        second_param_name: String,
        first_resolved: DimVar,
        second_resolved: DimVar,
        // None for single DimVar params, Some for Tensor params
        first_shape: Option<Shape>,
        second_shape: Option<Shape>,
        #[label("`{first_param_name}` has {dimvar_name}={first_resolved}{}", first_shape.as_ref().map(|s| format!(" in {s}")).unwrap_or_default())]
        first_span: SourceSpan,
        #[label("`{second_param_name}` has {dimvar_name}={second_resolved}{}", second_shape.as_ref().map(|s| format!(" in {s}")).unwrap_or_default())]
        second_span: SourceSpan,
    },

    /// Matmul inner dimensions don't match
    #[error("matmul inner dimensions don't match: {left_shape} @ {right_shape}")]
    #[diagnostic(code(shape::matmul_mismatch))]
    MatmulMismatch {
        left_shape: Shape,
        right_shape: Shape,
        left_dim: DimVar,
        right_dim: DimVar,
        #[label("left has inner dim {left_dim}, right has inner dim {right_dim}")]
        span: SourceSpan,
    },

    /// Broadcasting failed because dimensions don't match and neither is 1
    #[error(
        "cannot broadcast {left_shape} with {right_shape}: at dimension -{dim_position}, {left_dim} != {right_dim} and neither is 1"
    )]
    #[diagnostic(code(shape::broadcast_mismatch))]
    BroadcastMismatch {
        left_shape: Shape,
        right_shape: Shape,
        dim_position: usize, // 1-indexed from the right for human readability
        left_dim: DimVar,
        right_dim: DimVar,
        #[label("broadcast mismatch")]
        span: SourceSpan,
    },

    #[error("Can't infer return shape")]
    #[diagnostic(code(shape::uninferrable_call))]
    UninferrableCall {},

    #[error("Dimension {dim_ref} out of range for Tensor of rank {rank}")]
    #[diagnostic(code(shape::dim_out_of_range))]
    DimOutRange { dim_ref: i64, rank: usize },

    /// Reshape total elements don't match
    #[error("cannot reshape {src} to {tgt}: total elements differ")]
    #[diagnostic(code(shape::bad_reshape))]
    BadReshape {
        src: Shape,
        tgt: Shape,
        #[label("reshape mismatch")]
        span: SourceSpan,
    },

    #[error(
        "Rank of tensor one ({tensor_1}), {rank_1}, does not equal rank of tensor two ({tensor_2}), {rank_2}"
    )]
    #[diagnostic(code(shape::unequal_rank))]
    UnequalRank {
        tensor_1: Shape,
        tensor_2: Shape,
        rank_1: usize,
        rank_2: usize,
        #[label("mismatch occurs here")]
        span: SourceSpan,
    },

    /// Missing required argument for function call
    #[error("missing required argument `{param_name}`")]
    #[diagnostic(code(shape::missing_argument))]
    MissingArgument {
        func_name: String,
        param_name: String,
        #[label("argument `{param_name}` is required but not provided")]
        span: SourceSpan,
    },

    /// Matmul with scalar tensors (rank 0)
    #[error("matmul cannot be used with scalar tensors (rank 0)")]
    #[diagnostic(code(shape::matmul_with_scalar))]
    MatmulWithScalar {
        left_shape: Shape,
        right_shape: Shape,
        #[label("matmul with shapes {left_shape} @ {right_shape} is not allowed")]
        span: SourceSpan,
    },

    /// Dimension variable used in signature but not defined by parameters
    #[error("dimension `{dimvar_name}` is not defined by any parameter")]
    #[diagnostic(code(shape::undefined_dimvar))]
    UndefinedDimVar {
        dimvar_name: String,
        func_name: String,
        substitutions: String, // Human-readable representation of available substitutions
        #[label("dimension `{dimvar_name}` must appear in a function parameter to be used here")]
        span: SourceSpan,
    },

    /// Return type uses dimension variable not defined by parameters
    #[error("return type uses dimension `{dimvar_name}` which is not defined by any parameter")]
    #[diagnostic(code(shape::undefined_return_dimvar))]
    UndefinedReturnDimVar {
        dimvar_name: String,
        func_name: String,
        is_method: bool,
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
    pub fn to_diagnostic(&self, file_content: &str, file_uri: &str) -> Option<LspDiagnostic> {
        use tower_lsp::lsp_types::{DiagnosticRelatedInformation, Location, Url};

        // Customize messages for specific error types
        let message = match self {
            ShapeError::UndefinedReturnDimVar { dimvar_name, is_method, .. } => {
                let base = format!("return type uses dimension `{}` which is not defined by any parameter", dimvar_name);
                let hint = if *is_method {
                    format!("\nHelp: dimension `{}` must appear in a function parameter or in __init__ parameters", dimvar_name)
                } else {
                    format!("\nHelp: dimension `{}` must appear in a function parameter", dimvar_name)
                };
                format!("{}{}", base, hint)
            }
            ShapeError::UndefinedDimVar { dimvar_name, substitutions, .. } => {
                format!(
                    "dimension `{}` is not defined by any parameter\nAvailable substitutions: {}",
                    dimvar_name, substitutions
                )
            }
            _ => self.to_string(),
        };
        let severity = DiagnosticSeverity::ERROR;

        // Get range and related_information from span if available
        let (range, related_information) = match self {
            ShapeError::InconsistentDimVars {
                first_span,
                second_span,
                dimvar_name,
                first_param_name,
                second_param_name,
                first_resolved,
                second_resolved,
                first_shape,
                second_shape,
                ..
            } => {
                // Primary diagnostic at second occurrence
                let range = Self::span_to_range(second_span, file_content);
                let uri =
                    Url::parse(file_uri).unwrap_or_else(|_| Url::parse("file:///unknown").unwrap());
                // Related information showing both occurrences
                let first_shape_info = first_shape
                    .as_ref()
                    .map(|s| format!(" in {}", s))
                    .unwrap_or_default();
                let second_shape_info = second_shape
                    .as_ref()
                    .map(|s| format!(" in {}", s))
                    .unwrap_or_default();
                let related = vec![
                    DiagnosticRelatedInformation {
                        location: Location {
                            uri: uri.clone(),
                            range: Self::span_to_range(first_span, file_content),
                        },
                        message: format!(
                            "`{}` has {}={}{}",
                            first_param_name, dimvar_name, first_resolved, first_shape_info
                        ),
                    },
                    DiagnosticRelatedInformation {
                        location: Location {
                            uri,
                            range: Self::span_to_range(second_span, file_content),
                        },
                        message: format!(
                            "`{}` has {}={}{}",
                            second_param_name, dimvar_name, second_resolved, second_shape_info
                        ),
                    },
                ];
                (range, Some(related))
            }
            // Most errors just need span converted to range with no related information
            ShapeError::MismatchedDims { span, .. }
            | ShapeError::SignatureParamMismatch { span, .. }
            | ShapeError::MatmulMismatch { span, .. }
            | ShapeError::BroadcastMismatch { span, .. }
            | ShapeError::BadReshape { span, .. }
            | ShapeError::UnequalRank { span, .. }
            | ShapeError::MissingArgument { span, .. }
            | ShapeError::MatmulWithScalar { span, .. }
            | ShapeError::UndefinedDimVar { span, .. }
            | ShapeError::UndefinedReturnDimVar { span, .. } => {
                (Self::span_to_range(span, file_content), None)
            }
            // For errors without spans, use a default range at the start of the file
            ShapeError::UninferrableCall {} | ShapeError::DimOutRange { .. } => {
                let default_range = Range {
                    start: Position {
                        line: 0,
                        character: 0,
                    },
                    end: Position {
                        line: 0,
                        character: 0,
                    },
                };
                (default_range, None)
            }
        };

        Some(LspDiagnostic {
            range,
            severity: Some(severity),
            code: None,
            code_description: None,
            source: Some("dimspector".to_string()),
            message,
            related_information,
            tags: None,
            data: None,
        })
    }
}

/// Context for where a generic analysis error occurred
#[derive(Debug, Clone)]
pub enum ErrorContext {
    Class { name: String },
    Function { name: String },
    Method { class_name: String, method_name: String },
}

impl std::fmt::Display for ErrorContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ErrorContext::Class { name } => write!(f, "class {}", name),
            ErrorContext::Function { name } => write!(f, "function {}", name),
            ErrorContext::Method { class_name, method_name } => {
                write!(f, "method {}.{}", class_name, method_name)
            }
        }
    }
}

/// Generic analysis error (non-shape-related) with context
#[derive(Debug, Clone)]
pub struct GenericAnalysisError {
    pub message: String,
    pub context: ErrorContext,
    pub source_span: Option<SourceSpan>,
}

impl GenericAnalysisError {
    pub fn new(message: String, context: ErrorContext) -> Self {
        Self {
            message,
            context,
            source_span: None,
        }
    }

    pub fn with_span(message: String, context: ErrorContext, span: SourceSpan) -> Self {
        Self {
            message,
            context,
            source_span: Some(span),
        }
    }
}

impl std::fmt::Display for GenericAnalysisError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.context, self.message)
    }
}

/// Combined error type for analysis, encompassing both shape errors and generic errors
#[derive(Debug, Clone)]
pub enum AnalysisError {
    Shape(ShapeError),
    Generic(GenericAnalysisError),
}

impl From<ShapeError> for AnalysisError {
    fn from(error: ShapeError) -> Self {
        AnalysisError::Shape(error)
    }
}

impl From<GenericAnalysisError> for AnalysisError {
    fn from(error: GenericAnalysisError) -> Self {
        AnalysisError::Generic(error)
    }
}

impl std::fmt::Display for AnalysisError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnalysisError::Shape(e) => write!(f, "{}", e),
            AnalysisError::Generic(e) => write!(f, "{}", e),
        }
    }
}
