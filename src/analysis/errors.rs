use miette::Diagnostic;
use thiserror::Error;

use crate::analysis::{DimVar, Shape};

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

    #[error("Can't infer return shape")]
    UninferrableCall {},

    #[error("Dimension {dim_ref} out of range for Tensor of rank {rank}")]
    #[diagnostic(code(shape::mismatched_dims))]
    DimOutRange { dim_ref: i64, rank: usize },
}

impl ShapeError {
    pub fn mismatched(dim1: &DimVar, dim2: &DimVar) -> ShapeError {
        Self::MismatchedDims {
            dim1: dim1.clone(),
            dim2: dim2.clone(),
        } // TODO: span
    }
}
