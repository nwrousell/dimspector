mod errors;
mod print;
mod types;

use std::cmp::{max, min};
use std::collections::{HashMap, HashSet};

use itertools::EitherOrBoth::{Both, Left, Right};
use itertools::{Itertools, rev};
use petgraph::graph::NodeIndex;
use types::Variable;

use crate::ir::types::{Expr, Identifier, Statement};
use crate::ir::{Function, Program};
use crate::shape_analysis::errors::ShapeError;
use crate::shape_analysis::types::{DimKind, DimVar, Shape};
use miette::{Result, miette};

pub struct FunctionAnalysis {
    // func: StmtFunctionDef,
    // func: Function,
    // TODO: currently just using Hash{Set,Map}s, but would beneifit perhaps
    // from inenvitably using bitsets, if the speedup is worth it
    domain: HashMap<Identifier, HashSet<Variable>>,
}

impl FunctionAnalysis {
    fn new() -> Self {
        Self {
            domain: HashMap::new(),
        }
    }

    fn broadcast_resolve(&self, l_shape: &Shape, r_shape: &Shape) -> Result<Shape> {
        match (l_shape, r_shape) {
            (Shape::Known(l_shape), Shape::Known(r_shape)) => {
                let mut out_shape = Vec::new();
                for pair in l_shape.iter().rev().zip_longest(r_shape.iter().rev()) {
                    out_shape.push(match pair {
                        Both(l_dim, r_dim) => match (l_dim.kind(), r_dim.kind()) {
                            (DimKind::Named(l_sym), DimKind::Named(r_sym)) => {
                                // TODO: should we assert they are the same symbol, or potent add constraint?
                                if l_sym != r_sym {
                                    let err = ShapeError::mismatched(l_dim, r_dim);
                                    return Err(err.into());
                                }
                                l_dim.clone()
                            }
                            (DimKind::Named(sym), DimKind::Concrete(n))
                            | (DimKind::Concrete(n), DimKind::Named(sym)) => {
                                if n != 1 {
                                    let err = ShapeError::mismatched(l_dim, r_dim);
                                    return Err(err.into());
                                }
                                DimVar::new(DimKind::Named(sym))
                            }
                            (DimKind::Concrete(l_n), DimKind::Concrete(r_n)) => {
                                if l_n != r_n && (l_n != 1 && r_n != 1) {
                                    let err = ShapeError::mismatched(l_dim, r_dim);
                                    return Err(err.into());
                                }
                                DimVar::new(DimKind::Concrete(max(l_n, r_n)))
                            }
                        },
                        Left(v) | Right(v) => v.clone(),
                    });
                }
                out_shape.reverse();
                Ok(Shape::Known(out_shape))
            }
            (_, _) => Ok(Shape::Unknown), // TODO:
        }
    }

    fn eval_expr(&mut self, expr: &Expr) -> Result<HashSet<Shape>> {
        // TODO:
        // - flows of dimvars out of .shape or .size()
        // - reshapes, rearranges
        // - pytorch stub representation (in IR)
        match expr {
            Expr::Binop {
                left,
                right,
                is_matmul,
            } => {
                let l_shapes = self.eval_expr(left)?;
                let r_shapes = self.eval_expr(right)?;

                let mut out_shapes = HashSet::new();
                for l_shape in l_shapes.iter() {
                    for r_shape in r_shapes.iter() {
                        if *is_matmul {
                            // TODO: maybe this should just resolve to the tensor dot stub
                        } else {
                            let out_shape = self.broadcast_resolve(&l_shape, &r_shape)?;
                            out_shapes.insert(out_shape);
                        }
                    }
                }
            }
            Expr::Call {
                receiver,
                function,
                args,
            } => {}
            Expr::Constant => {}
            Expr::Identifier(id) => {}
        }
        Ok(HashSet::new())
    }

    fn analyze_stmt(&mut self, stmt: &Statement) -> Result<()> {
        Ok(())
    }

    fn analyze_func(func: &Function) -> Result<FunctionAnalysis> {
        let mut analysis = Self::new();
        let blocks: Vec<_> = func.blocks().collect();
        for block in blocks {
            let block = func.data(block);
            for stmt in block.statements() {
                analysis.analyze_stmt(stmt);
            }
        }
        Ok(analysis)
    }
}

pub fn analyze(prog: Program) -> Result<()> {
    for func in prog.functions {
        // TODO: maybe do some nice caching later for modularity with user's own funcs
        let _ = FunctionAnalysis::analyze_func(&func);
    }
    Ok(())
}
