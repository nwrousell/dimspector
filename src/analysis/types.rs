// NOTE: this representation disallows A[..., d] shapes

use std::ops::Add;

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
    pub kind: DimKind,
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
    Tuple(Vec<Variable>),
    None,
}

impl Variable {
    pub fn as_dimvar(&self) -> Option<&DimVar> {
        match self {
            Variable::Top => None,
            Variable::DimVar(dim_var) => Some(dim_var),
            Variable::Tensor(_) => None,
            Variable::Tuple(_) => None,
            Variable::None => None,
        }
    }

    pub fn as_shape(&self) -> Option<Shape> {
        match self {
            Variable::Top => None,
            Variable::DimVar(_) => None,
            Variable::Tensor(shape) => Some(shape.clone()),
            Variable::Tuple(vars) => {
                if vars.iter().all(|var| matches!(var, Variable::DimVar(_))) {
                    let shape = Shape(
                        vars.iter()
                            .map(|var| {
                                let Variable::DimVar(dvar) = var else {
                                    unreachable!("all var in vars should be dimvar")
                                };
                                dvar.clone()
                            })
                            .collect(),
                    );
                    Some(shape)
                } else {
                    None
                }
            }
            Variable::None => None,
        }
    }

    pub fn as_shape_dims(&self) -> Option<Vec<DimVar>> {
        match self {
            Variable::Top => None,
            Variable::DimVar(_) => None,
            Variable::Tensor(shape) => Some(shape.0.clone()),
            Variable::Tuple(vars) => {
                if vars.iter().all(|var| matches!(var, Variable::DimVar(_))) {
                    Some(
                        vars.iter()
                            .map(|var| {
                                let Variable::DimVar(dvar) = var else {
                                    unreachable!("all var in vars should be dimvar")
                                };
                                dvar.clone()
                            })
                            .collect(),
                    )
                } else {
                    None
                }
            }
            Variable::None => None,
        }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Shape(pub Vec<DimVar>);

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

        Self(dims)
    }
}
