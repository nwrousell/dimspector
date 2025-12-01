use miette::{Diagnostic, SourceSpan};
use thiserror::Error;

use crate::shape_analysis::types::DimVar;

#[derive(Diagnostic, Error, Debug)]
pub enum ShapeError {
    #[error("Mismatched dims: {dim1} != {dim2}")]
    #[diagnostic(code(shape::mismatched_dims))]
    MismatchedDims {
        dim1: DimVar,
        dim2: DimVar,
        // #[label("mismatch occurs here")]
        // span: SourceSpan,
    },
}

impl ShapeError {
    pub fn mismatched(dim1: &DimVar, dim2: &DimVar) -> ShapeError {
        Self::MismatchedDims {
            dim1: dim1.clone(),
            dim2: dim2.clone(),
        } // TODO: span
    }
}
