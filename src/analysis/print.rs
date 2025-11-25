use std::fmt::Display;

use crate::analysis::{FunctionAnalysis, types::Axis};

impl Display for FunctionAnalysis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{}:\n", self.func.name))?;
        for (k, v) in self.domain.iter() {
            f.write_fmt(format_args!("{k} -> {v:?}\n"))?;
        }
        Ok(())
    }
}

impl Display for Axis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Axis::Concrete(c) => f.write_fmt(format_args!("{c}"))?,
            Axis::Named(n) => f.write_fmt(format_args!("{n}"))?,
        }
        Ok(())
    }
}
