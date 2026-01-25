use itertools::Either;

use crate::analysis::dimvars::{DimKind, DimVar, parse_dimvar};
use crate::ir::types::Identifier;
use crate::ir::{Class, Expr};
use anyhow::Result;
use std::collections::{BTreeMap, HashMap};

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum Variable {
    Top,
    DimVar(DimVar),
    Tensor(Shape),
    ClassInstance(ClassInstance),
    Collection(Collection),
    None,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum Collection {
    Tuple(Vec<Variable>),
    List(Vec<Variable>),
    Dict(BTreeMap<DictKey, Variable>),
}

impl Collection {
    fn eval_vectored_index(
        &self,
        tuple_elts: &[Variable],
        index_or_slice: &Either<Variable, DimSlice>,
    ) -> Option<Variable> {
        match index_or_slice {
            Either::Left(var) => {
                if let Some(c) = var.as_concrete_dimvar() {
                    tuple_elts.get(c as usize).cloned()
                } else {
                    Some(Variable::Top)
                }
            }
            Either::Right(DimSlice { lower, upper }) => {
                let lower = if let Some(l) = lower {
                    l.as_concrete_dimvar().map(|c| c as usize)
                } else {
                    Some(0)
                };

                let upper = if let Some(u) = upper {
                    u.as_concrete_dimvar().map(|c| c as usize)
                } else {
                    Some(tuple_elts.len())
                };

                match (lower, upper) {
                    (Some(l), Some(u)) => {
                        let tuple =
                            Variable::Collection(Collection::Tuple(tuple_elts[l..u].to_vec()));
                        Some(tuple)
                    }
                    _ => Some(Variable::Top),
                }
            }
        }
    }

    pub fn read_at_index(
        &self,
        index_or_slices: &[&Either<Variable, DimSlice>],
    ) -> Option<Variable> {
        assert!(index_or_slices.len() == 1);
        let index_or_slice = index_or_slices.first().unwrap();
        match self {
            Collection::Tuple(elts) | Collection::List(elts) => {
                self.eval_vectored_index(elts, index_or_slice)
            }
            Collection::Dict(items) => {
                // TODO: string keys. right now this would require adding string to variable or
                // routing exprs here directly instead, which could be ok with string constant
                // folding at ir
                let &Either::Left(index) = index_or_slice else {
                    return None;
                };
                let Some(key) = index.as_concrete_dimvar().map(|k| DictKey::Int(k)) else {
                    return None;
                };

                Some(items[&key].clone())
            }
        }
    }

    pub fn tup_from_elts(elts: Vec<Variable>) -> Variable {
        Variable::Collection(Collection::Tuple(elts))
    }

    pub fn list_from_elts(elts: Vec<Variable>) -> Variable {
        Variable::Collection(Collection::List(elts))
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum DictKey {
    Int(i64),
    Str(String),
}

/// Represents a class instance with concrete dimension variable substitutions.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ClassInstance {
    /// The canonical identifier of the class (e.g., "module.ClassName")
    pub class_id: Identifier,
    /// Mapping from parameter dimvars to concrete dimvars
    pub substitutions: BTreeMap<DimVar, DimVar>,
}

impl Variable {
    pub fn as_dimvar(&self) -> Option<&DimVar> {
        match self {
            Variable::Top => None,
            Variable::DimVar(dim_var) => Some(dim_var),
            Variable::Tensor(_) => None,
            Variable::Collection(_) => None,
            Variable::None => None,
            Variable::ClassInstance(_) => None,
        }
    }

    pub fn as_class_instance(&self) -> Option<&ClassInstance> {
        match self {
            Variable::ClassInstance(instance) => Some(instance),
            _ => None,
        }
    }

    pub fn as_vec(&self) -> Option<&Vec<Variable>> {
        match self {
            Variable::Collection(Collection::Tuple(vars) | Collection::List(vars)) => Some(vars),
            _ => None,
        }
    }

    pub fn as_shape(&self) -> Option<Shape> {
        match self {
            Variable::Top => None,
            Variable::DimVar(_) => None,
            Variable::Tensor(shape) => Some(shape.clone()),
            Variable::Collection(col) => match col {
                Collection::Tuple(vars) | Collection::List(vars) => {
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
                _ => None,
            },
            Variable::None => None,
            Variable::ClassInstance(_) => None,
        }
    }

    pub fn as_shape_dims(&self) -> Option<Vec<DimVar>> {
        match self {
            Variable::Top => None,
            Variable::DimVar(_) => None,
            Variable::Tensor(shape) => Some(shape.0.clone()),
            Variable::Collection(col) => match col {
                Collection::Tuple(vars) | Collection::List(vars) => {
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
                _ => None,
            },
            Variable::None => None,
            Variable::ClassInstance(_) => None,
        }
    }

    pub fn as_concrete_dimvar(&self) -> Option<i64> {
        if let Some(dimvar) = self.as_dimvar() {
            match dimvar.kind() {
                DimKind::Concrete(c) => Some(c),
                _ => None,
            }
        } else {
            None
        }
    }

    /// Apply DimVar substitutions to a Variable, recursively substituting DimVars in shapes and collections.
    pub fn substitute(
        &self,
        substitutions: &HashMap<String, DimVar>,
        func_name: &str,
        span: miette::SourceSpan,
    ) -> Result<Variable> {
        match self {
            Variable::DimVar(dv) => {
                // Substitute the DimVar
                let substituted = dv.substitute(substitutions, func_name, span)?;
                Ok(Variable::DimVar(substituted))
            }
            Variable::Tensor(shape) => {
                // Recursively substitute each DimVar in the shape
                let substituted_dims: Result<Vec<DimVar>> = shape
                    .0
                    .iter()
                    .map(|dv| dv.substitute(substitutions, func_name, span))
                    .collect();
                Ok(Variable::Tensor(Shape(substituted_dims?)))
            }
            Variable::Collection(col) => {
                let sub_col = match col {
                    Collection::Tuple(vars) => {
                        let substituted_vars: Result<Vec<Variable>> = vars
                            .iter()
                            .map(|v| v.substitute(substitutions, func_name, span))
                            .collect();
                        Collection::Tuple(substituted_vars?)
                    }
                    Collection::List(vars) => {
                        let substituted_vars: Result<Vec<Variable>> = vars
                            .iter()
                            .map(|v| v.substitute(substitutions, func_name, span))
                            .collect();
                        Collection::List(substituted_vars?)
                    }
                    Collection::Dict(map) => {
                        let substituted_map: Result<BTreeMap<_, _>> = map
                            .iter()
                            .map(|(k, v)| {
                                v.substitute(substitutions, func_name, span)
                                    .map(|sub_v| (k.clone(), sub_v))
                            })
                            .collect();
                        Collection::Dict(substituted_map?)
                    }
                };
                Ok(Variable::Collection(sub_col))
            }
            Variable::ClassInstance(_) => {
                // TODO: Handle nested class instances
                Ok(self.clone())
            }
            Variable::Top | Variable::None => Ok(self.clone()),
        }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Shape(pub Vec<DimVar>);

impl Shape {
    pub fn from_str(s: &str) -> Self {
        let dims = s
            .split_whitespace()
            .filter(|x| !x.is_empty())
            .map(parse_dimvar)
            .collect::<Result<Vec<_>, _>>()
            .expect("invalid shape expression");
        Self(dims)
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct DimSlice {
    pub lower: Option<Variable>,
    pub upper: Option<Variable>,
}
