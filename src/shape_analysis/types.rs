// NOTE: this representation disallows A[..., d] shapes

use std::fmt::Display;

#[derive(Debug, Clone)]
pub enum Variable {
    NotTensor,
    // AxixLength { axis: u32, root: Identifier, loc: Location }
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
