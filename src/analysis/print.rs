use std::fmt;

use crate::utils::write_comma_separated;

use super::types::{DimKind, DimVar, Shape, Variable};

impl fmt::Display for DimVar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind() {
            DimKind::Concrete(c) => f.write_fmt(format_args!("{c}"))?,
            DimKind::Named(n) => f.write_fmt(format_args!("{n}"))?,
        }
        Ok(())
    }
}

impl fmt::Display for Variable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Variable::NonTensor => write!(f, "NonTensor"),
            Variable::DimVar(dim_var) => write!(f, "{}", dim_var),
            Variable::Tensor(hash_set) => {
                write!(f, "{{")?;
                write_comma_separated(f, hash_set)?;
                write!(f, "}}")
            }
            Variable::Top => write!(f, "⟙"),
        }
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Shape::Unknown => write!(f, "Unknown"),
            Shape::Known(dim_vars) => {
                write!(f, "[")?;
                write_comma_separated(f, dim_vars)?;
                write!(f, "]")
            }
        }
    }
}
