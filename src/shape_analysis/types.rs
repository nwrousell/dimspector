// NOTE: this representation disallows A[..., d] shapes

use std::{collections::HashSet, fmt::Display};

use crate::ir::types::Identifier;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum DimKind {
    Named(String),
    Concrete(u32),
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct DimVar {
    kind: DimKind,
    // TODO: more info potent around this, related to origin of this dimvar which would be
    // useful for debugging
}

impl DimVar {
    pub fn new(kind: DimKind) -> Self {
        Self { kind }
    }

    pub fn kind(&self) -> DimKind {
        self.kind.clone()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Variable {
    NonTensor,
    DimVar(DimVar),
    Tensor(HashSet<Shape>),
    Top,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum Shape {
    Unknown,
    Known(Vec<DimVar>),
}

impl Shape {
    pub fn from_str(s: &str) -> Self {
        let mut dims = Vec::new();
        for dim in s.split(' ') {
            if let Ok(n) = dim.parse::<u32>() {
                dims.push(DimVar {
                    kind: DimKind::Concrete(n),
                });
            } else if !dim.trim().is_empty() {
                dims.push(DimVar {
                    kind: DimKind::Named(dim.to_string()),
                });
            }
        }

        Self::Known(dims)
    }
}
