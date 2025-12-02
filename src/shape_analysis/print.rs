use std::fmt::Display;

use crate::shape_analysis::types::{DimKind, DimVar};

impl Display for DimVar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.kind() {
            DimKind::Concrete(c) => f.write_fmt(format_args!("{c}"))?,
            DimKind::Named(n) => f.write_fmt(format_args!("{n}"))?,
        }
        Ok(())
    }
}
