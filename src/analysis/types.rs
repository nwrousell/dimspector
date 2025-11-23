// NOTE: this representation disallows A[..., d] shapes

use std::fmt::Display;

#[derive(Debug, Clone)]
pub enum Variable {
    NotTensor,
    Tensor(Shape),
}

#[derive(Debug, Clone)]
pub enum Shape {
    Unknown,
    Known(Vec<Axis>),
}

impl Shape {
    pub fn from_str(s: &str) -> Self {
        let mut axes = Vec::new();
        for axis in s.split(' ') {
            if let Ok(n) = axis.parse::<u32>() {
                axes.push(Axis::Concrete(n));
            } else if !axis.trim().is_empty() {
                axes.push(Axis::Named(axis.to_string()));
            }
        }

        Self::Known(axes)
    }
}

#[derive(Debug, Clone)]
pub enum Axis {
    Named(String),
    Concrete(u32),
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
