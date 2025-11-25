use miette::Diagnostic;

use crate::analysis::types::Axis;

#[derive(Diagnostic, Error, Debug)]
pub enum ShapeError {
    #[error("Mismatched dims: {dim1} != {dim2}")]
    #[diagnostic(code(shape::mismatched_dims))]
    MismatchedDims {
        dim1: Axis,
        dim2: Axis,
        #[label("mismatch occurs here")]
        span: SourceSpan,
    },
}
