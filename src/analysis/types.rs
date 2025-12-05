// NOTE: this representation disallows A[..., d] shapes

use std::collections::HashSet;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum DimKind {
    Named(String),
    Concrete(i64),
    // TODO: DimExpr: expr between dimvars
    // Add {
    //     left: Box<DimExpr>,
    //     right: Box<DimExpr>,
    // },
    // Mul {
    //     left: Box<DimExpr>,
    //     right: Box<DimExpr>,
    // },
}

// impl Eq for DimExpr {}

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

impl From<i64> for DimVar {
    fn from(value: i64) -> Self {
        Self {
            kind: DimKind::Concrete(value),
        }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum Variable {
    Top,
    DimVar(DimVar),
    Tensor(Shape),
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
            if let Ok(n) = dim.parse::<i64>() {
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
