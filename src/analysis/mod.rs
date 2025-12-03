mod errors;
mod print;
mod types;

use std::cmp::max;
use std::collections::{HashMap, HashSet};

use itertools::EitherOrBoth::{Both, Left, Right};
use itertools::{Either, Itertools};
pub use types::{DimVar, Shape, Variable};

use crate::analysis::errors::ShapeError;
use crate::analysis::types::DimKind;
use crate::ir::types::{Binop, Constant, ExprKind, Location};
use crate::ir::{Expr, Parameter, Path, Statement, Terminator};
use crate::ir::{Function, Program};
use anyhow::Result;
type AnalysisDomain = HashMap<Path, HashSet<Variable>>;

pub trait JoinSemiLattice: Eq {
    fn join(&mut self, other: &Self);
}

impl JoinSemiLattice for AnalysisDomain {
    fn join(&mut self, other: &Self) {
        for (path, vars) in other.iter() {
            if let Some(e) = self.get_mut(&path) {
                e.extend(vars.iter().cloned());
            } else {
                self.insert(path.clone(), vars.clone());
            }
        }
    }
}

pub struct GlobalAnalysis {
    functions: HashMap<Path, FunctionAnalysis>,
}

impl GlobalAnalysis {
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
        }
    }

    pub fn analyze_func(&mut self, func: &Function) -> Result<()> {
        let name = func.identifier.clone();
        let mut func_analysis = FunctionAnalysis::new(func);
        func_analysis.analyze_func(func)?;
        self.functions.insert(name, func_analysis);
        Ok(())
    }
}

pub struct FunctionAnalysis {
    // func: StmtFunctionDef,
    // func: Function,
    // TODO: currently just using Hash{Set,Map}s, but would beneifit perhaps
    // from inenvitably using bitsets, if the speedup is worth it
    pub id: Path,
    pub state: HashMap<Location, AnalysisDomain>,
}

impl FunctionAnalysis {
    fn new(func: &Function) -> Self {
        // populate state with initial params
        let mut state = HashMap::new();

        let mut start_domain = AnalysisDomain::new();

        for Parameter(path, var) in &func.params {
            if let Some(var) = var {
                start_domain.insert(path.clone(), HashSet::from([var.clone()]));
            }
        }
        state.insert(Location::START, start_domain);

        Self {
            id: func.identifier.clone(),
            state,
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

    fn eval_expr(&mut self, domain: &AnalysisDomain, expr: &Expr) -> Result<HashSet<Variable>> {
        // TODO:
        // - flows of dimvars out of .shape or .size()
        // - reshapes, rearranges
        // - pytorch stub representation (in IR)
        match &expr.kind {
            ExprKind::Binop { left, right, op } => {
                let l_vars = self.eval_expr(domain, left)?;
                let r_vars = self.eval_expr(domain, right)?;
                let mut out_vars = HashSet::new();
                let is_matmul = matches!(op, Binop::MatMult);
                for l_var in l_vars.iter() {
                    for r_var in r_vars.iter() {
                        match (l_var, r_var) {
                            (Variable::Top, _) | (_, Variable::Top) => {
                                out_vars.insert(Variable::Top);
                            }
                            (Variable::Tensor(l_shape), Variable::Tensor(r_shape)) => {
                                if is_matmul {
                                    // TODO: maybe this should just resolve to the tensor dot stub
                                    todo!()
                                } else {
                                    let out_shape = self.broadcast_resolve(&l_shape, &r_shape)?;
                                    out_vars.insert(Variable::Tensor(out_shape));
                                }
                            }
                            (Variable::Tensor(shape), _) | (_, Variable::Tensor(shape)) => {
                                // other should be some number, will retain tensor operand shape
                                out_vars.insert(Variable::Tensor(shape.clone()));
                            }
                            (Variable::DimVar(l_dvar), Variable::DimVar(r_dvar)) => {
                                // TODO: in the future, we want to get some symbolic expr out of this
                                todo!()
                            }
                        }
                    }
                }
                Ok(out_vars)
            }
            ExprKind::Call {
                receiver,
                function,
                pos_args,
                keyword_args,
            } => {
                todo!()
            }
            ExprKind::Constant(c) => match c {
                Constant::Int(i) => {
                    Ok(HashSet::from_iter(vec![Variable::DimVar(DimVar::from(*i))]))
                }
                _ => Ok(HashSet::from_iter(vec![Variable::Top])),
            },
            ExprKind::Path(p) => Ok(domain
                .get(p)
                .unwrap_or(&HashSet::from_iter(vec![Variable::Top]))
                .clone()),
            ExprKind::Slice { receiver, slice } => todo!(),
        }
    }

    fn handle_stmt(&mut self, domain: &mut AnalysisDomain, stmt: &Statement) -> Result<()> {
        let res_var = self.eval_expr(domain, &stmt.value)?;
        if let Some(path) = &stmt.target {
            domain.insert(path.clone(), res_var); // TODO: intern
        }
        Ok(())
    }

    fn handle_term(&mut self, domain: &mut AnalysisDomain, term: &Terminator) -> Result<()> {
        Ok(())
    }

    fn analyze_func(&mut self, func: &Function) -> Result<()> {
        for loc in func.locations.iter() {
            let mut domain = AnalysisDomain::new();
            let preds = func.predecessors(&loc);
            if preds.len() == 0 {
                domain = self.state.get(&Location::START).unwrap().clone();
            } else {
                for pred_loc in preds {
                    domain.join(self.state.entry(pred_loc).or_insert(AnalysisDomain::new()));
                }
            }
            match func.instr(&loc) {
                Either::Left(stmt) => self.handle_stmt(&mut domain, stmt),
                Either::Right(term) => self.handle_term(&mut domain, term),
            }?;
            self.state.insert(*loc, domain);
        }
        Ok(())
    }
}

pub fn analyze(prog: Program) -> Result<GlobalAnalysis> {
    let mut global_analysis = GlobalAnalysis::new();
    for func in prog.functions {
        // TODO: maybe do some nice caching later for modularity with user's own funcs
        global_analysis.analyze_func(&func)?;
    }
    Ok(global_analysis)
}
