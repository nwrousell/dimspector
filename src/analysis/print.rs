use std::{collections::HashMap, fmt};

use itertools::Itertools;

use crate::{
    analysis::{AnalysisDomain, FunctionAnalysis, GlobalAnalysis},
    utils::write_comma_separated,
};

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
            Variable::Top => write!(f, "NonTensor"),
            Variable::DimVar(dim_var) => write!(f, "{}", dim_var),
            Variable::Tensor(shape) => write!(f, "{}", shape),
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
impl fmt::Display for FunctionAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Function {}\n", self.id)?;
        for (loc, domain) in self
            .state
            .iter()
            .sorted_by(|(l_a, d_a), (l_b, d_b)| Ord::cmp(*l_a, *l_b))
        {
            write!(f, "  {}\n", loc)?;
            for (path, vars) in domain.iter() {
                write!(f, "    {} => {{", path)?;
                write_comma_separated(f, vars)?;
                write!(f, "}}\n")?
            }
        }
        Ok(())
    }
}

impl fmt::Display for GlobalAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (_, func) in self.functions.iter() {
            write!(f, "{}\n\n", func)?;
        }
        Ok(())
    }
}
