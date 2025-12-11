use miette::{Diagnostic, SourceSpan};
use thiserror::Error;

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

    #[error("Rank of tensor one, {rank_1} does not equal rank of tensor two, {rank_2}")]
    UnequalRank { rank_1: usize, rank_2: usize },
}

impl ShapeError {
    pub fn mismatched(dim1: &DimVar, dim2: &DimVar, span: SourceSpan) -> ShapeError {
        Self::MismatchedDims {
            dim1: dim1.clone(),
            dim2: dim2.clone(),
            span,
        }
    }
}
