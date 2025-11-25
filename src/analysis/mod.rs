mod errors;
mod print;
mod types;
use miette::{Result, SourceSpan};
use std::collections::{HashMap, HashSet};

use rustpython_parser::ast::Ranged;
use rustpython_parser::ast::{Expr, Identifier, StmtFunctionDef};

use crate::analysis::errors::ShapeError;
use crate::{
    analysis::types::{Axis, Shape},
    ir::{Function, Program},
};
use types::Variable;

pub struct FunctionAnalysis {
    // func: StmtFunctionDef,
    func: Function,
    domain: HashMap<Identifier, HashSet<Variable>>,
}

impl FunctionAnalysis {
    fn new(func: StmtFunctionDef) -> Self {
        Self {
            func,
            domain: HashMap::new(),
        }
    }

    fn parse_args(&mut self) {
        for arg in self.func.args.args.clone() {
            let identifier = arg.def.arg;
            if let Some(annotation) = arg.def.annotation {
                if let Expr::Subscript(subscript) = *annotation {
                    if let Expr::Name(name) = *subscript.value {
                        if name.id.as_str() == "T" {
                            if let Expr::Constant(shape_str) = *subscript.slice {
                                let shape_str = shape_str.value.expect_str();
                                self.domain.insert(
                                    identifier,
                                    Variable::Tensor(Shape::from_str(&shape_str)),
                                );
                            } else {
                                self.domain
                                    .insert(identifier, Variable::Tensor(Shape::Unknown));
                            }
                        } else {
                            self.domain.insert(identifier, Variable::NotTensor);
                        }
                    } else {
                        self.domain.insert(identifier, Variable::NotTensor);
                    }
                } else {
                    self.domain.insert(identifier, Variable::NotTensor);
                }
            } else {
                self.domain.insert(identifier, Variable::NotTensor);
            }
        }
    }

    fn resolve_var(&self, id: &Identifier) -> Variable {
        self.domain.get(id).unwrap_or(&Variable::NotTensor).clone()
    }

    /// Convert a rustpython_parser TextRange to a miette SourceSpan
    fn text_range_to_source_span<T: Ranged>(node: &T) -> SourceSpan {
        let range = node.range();
        let start: usize = range.start().into();
        let end: usize = range.end().into();
        let length = end - start;
        SourceSpan::new(start.into(), length.into())
    }

    fn axes_matching(&self, a1: &Axis, a2: &Axis, span: SourceSpan) -> Result<()> {
        let is_error = match (a1, a2) {
            (Axis::Named(n1), Axis::Named(n2)) => n1 != n2,
            (Axis::Concrete(c1), Axis::Concrete(c2)) => c1 != c2,
            (Axis::Named(_), Axis::Concrete(_)) => false,
            (Axis::Concrete(_), Axis::Named(_)) => false,
        };

        if is_error {
            let err = ShapeError::MismatchedDims {
                dim1: a1.clone(),
                dim2: a2.clone(),
                span,
            };
            Err(err.into())
        } else {
            Ok(())
        }
    }

    fn handle_expr(&self, expr: Expr) -> Result<Variable> {
        match expr {
            Expr::BinOp(expr) => {
                // Get the range before moving parts of expr
                let span = Self::text_range_to_source_span(&expr);
                let left = self.handle_expr(*expr.left)?;
                let right = self.handle_expr(*expr.right)?;

                match expr.op {
                    rustpython_parser::ast::Operator::MatMult => match (left, right) {
                        (
                            Variable::Tensor(Shape::Known(axes_left)),
                            Variable::Tensor(Shape::Known(axes_right)),
                        ) => {
                            // check last axis of s1 with first axis of s2
                            self.axes_matching(
                                axes_left
                                    .last()
                                    .expect("tensor must have at least one axis"),
                                axes_right
                                    .first()
                                    .expect("tensor must have at least one axis"),
                                span,
                            )?;

                            if axes_left.len() != 2 || axes_right.len() != 2 {
                                todo!("handle batched matmul!");
                            }

                            Ok(Variable::Tensor(Shape::Known(vec![
                                axes_left.first().unwrap().clone(),
                                axes_right.last().unwrap().clone(),
                            ])))
                        }
                        _ => Ok(Variable::Tensor(Shape::Unknown)),
                    },
                    _ => match (left, right) {
                        (Variable::NotTensor, Variable::NotTensor) => Ok(Variable::NotTensor),
                        (Variable::Tensor(_), Variable::Tensor(_)) => {
                            todo!("handle broadcasting")
                        }
                        (Variable::Tensor(s), _) | (_, Variable::Tensor(s)) => {
                            Ok(Variable::Tensor(s))
                        }
                    },
                }
            }
            Expr::Constant(_) => Ok(Variable::NotTensor),
            Expr::Name(expr_name) => Ok(self.resolve_var(&expr_name.id)),

            _ => todo!("not handled"),
        }
    }

    fn analyze_func(&mut self) -> Result<()> {
        self.parse_args();

        for stmt in self.func.body.clone() {
            match stmt {
                rustpython_parser::ast::Stmt::Assign(assign) => {
                    let id = assign
                        .targets
                        .first()
                        .expect("assign doesn't have a target")
                        .as_name_expr()
                        .expect("assuming can only assign to names right now")
                        .id
                        .clone();

                    let val = self.handle_expr(*assign.value)?;
                    self.domain.insert(id, val);
                }

                _ => todo!("not handled"),
            }
        }

        Ok(())
    }
}

pub fn analyze(prog: Program) -> Result<()> {
    for func in prog.functions.iter() {
        let mut func_analysis = FunctionAnalysis::new(func.clone());
        func_analysis.analyze_func()?;
        log::debug!("{}", func_analysis)
    }

    Ok(())
}
