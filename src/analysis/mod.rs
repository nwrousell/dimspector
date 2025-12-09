mod dimvars;
mod errors;
mod models;
mod print;
mod types;

use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use itertools::{Either, Itertools};
pub use types::{Shape, Variable};

pub use crate::analysis::dimvars::{DimKind, DimVar};
use crate::analysis::models::{Model, ModelContext};
use crate::analysis::types::DimSlice;
use crate::ir::types::{Binop, Constant, ExprKind, Location, Slice};
use crate::ir::{Expr, Parameter, Path, Statement, Terminator};
use crate::ir::{Function, Program};
use anyhow::Result;
type AnalysisDomain = HashMap<Path, HashSet<Variable>>;

pub use print::{ir_with_inferred_shapes_to_string, print_ir_with_inferred_shapes};

pub trait JoinSemiLattice: Eq {
    fn join(&mut self, other: &Self);
}

impl JoinSemiLattice for AnalysisDomain {
    fn join(&mut self, other: &Self) {
        for (path, vars) in other.iter() {
            if let Some(e) = self.get_mut(path) {
                e.extend(vars.iter().cloned());
            } else {
                self.insert(path.clone(), vars.clone());
            }
        }
    }
}

pub struct GlobalAnalysis {
    pub functions: HashMap<Path, FunctionAnalysis>,
    pub models: Rc<ModelContext>,
}

impl GlobalAnalysis {
    pub fn new(funcs: &Vec<Function>) -> Self {
        Self {
            functions: HashMap::new(),
            models: Rc::new(ModelContext::new(funcs)),
        }
    }

    pub fn analyze_func(&mut self, func: &Function) -> Result<()> {
        let name = func.identifier.clone();
        let mut func_analysis = FunctionAnalysis::new(func, Rc::clone(&self.models));
        func_analysis.analyze_func(func)?;
        self.functions.insert(name, func_analysis);
        Ok(())
    }
}

pub struct FunctionAnalysis {
    // TODO: currently just using Hash{Set,Map}s, but would benefit perhaps
    // from using bitsets, if the speedup is worth it
    pub id: Path,
    pub state: HashMap<Location, AnalysisDomain>,
    pub models: Rc<ModelContext>,
}

impl FunctionAnalysis {
    fn new(func: &Function, models: Rc<ModelContext>) -> Self {
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
            models,
        }
    }

    fn fold_dimvars(&self, left_dimvar: DimVar, right_dimvar: DimVar, op: Binop) -> Variable {
        match op {
            Binop::Add => Variable::DimVar(left_dimvar + right_dimvar),
            Binop::Sub => Variable::DimVar(left_dimvar - right_dimvar),
            Binop::Mult => Variable::DimVar(left_dimvar * right_dimvar),
            _ => Variable::Top,
        }
    }

    fn eval_binop(
        &mut self,
        domain: &AnalysisDomain,
        left: &Expr,
        right: &Expr,
        op: Binop,
    ) -> Result<HashSet<Variable>> {
        let l_vars = self.eval_expr(domain, left)?;
        let r_vars = self.eval_expr(domain, right)?;

        let mut out_vars = HashSet::new();

        let is_matmul = matches!(op, Binop::MatMult);

        for (l_var, r_var) in l_vars.iter().cartesian_product(r_vars.iter()) {
            let out_var = match (l_var, r_var) {
                (Variable::Top, _) | (_, Variable::Top) => Variable::Top,
                (Variable::Tensor(_), Variable::Tensor(_)) => {
                    if is_matmul {
                        let out_shape = self
                            .models
                            .torch
                            .matmul
                            .infer(vec![l_var, r_var], HashMap::new())?;
                        Variable::Tensor(out_shape)
                    } else {
                        let out_shape = self
                            .models
                            .torch
                            .broadcast
                            .infer(vec![l_var, r_var], HashMap::new())?;
                        Variable::Tensor(out_shape)
                    }
                }
                (Variable::Tensor(shape), _) | (_, Variable::Tensor(shape)) => {
                    // other should be some number, will retain tensor operand shape
                    Variable::Tensor(shape.clone())
                }
                (Variable::Tuple(l_vars), Variable::Tuple(r_vars)) => match op {
                    Binop::Add => {
                        let mut out = l_vars.clone();
                        out.extend(r_vars.iter().cloned());
                        Variable::Tuple(out)
                    }
                    _ => Variable::Top,
                },
                (Variable::DimVar(l_dvar), Variable::DimVar(r_dvar)) => {
                    self.fold_dimvars(l_dvar.clone(), r_dvar.clone(), op)
                }
                _ => {
                    panic!("runtime error")
                }
            };

            out_vars.insert(out_var);
        }
        Ok(out_vars)
    }

    fn eval_expr(&mut self, domain: &AnalysisDomain, expr: &Expr) -> Result<HashSet<Variable>> {
        match &expr.kind {
            ExprKind::Binop { left, right, op } => self.eval_binop(domain, left, right, *op),
            ExprKind::Call {
                receiver,
                function,
                pos_args,
                keyword_args,
            } => {
                let args = pos_args
                    .iter()
                    .map(|arg_expr| self.eval_expr(domain, arg_expr))
                    .collect::<Result<Vec<HashSet<Variable>>>>()?;
                let kwargs = keyword_args
                    .iter()
                    .map(|(n, arg_expr)| Ok((n.clone(), self.eval_expr(domain, arg_expr)?)))
                    .collect::<Result<Vec<(String, HashSet<Variable>)>>>()?;

                let args_products = args.iter().multi_cartesian_product();
                let kw = kwargs.iter().map(|(n, _)| n.clone());
                let kwargs_products: Vec<HashMap<_, _>> = kwargs
                    .iter()
                    .map(|(_, vars)| vars)
                    .multi_cartesian_product()
                    .map(|vars| kw.clone().zip(vars).collect())
                    .collect();

                let mut out_vars = HashSet::new();

                match receiver {
                    None => {
                        let func = function.to_dot_string();

                        match self.models.resolve(&func) {
                            Some(model) => {
                                for (args, kwargs) in
                                    args_products.cartesian_product(kwargs_products)
                                {
                                    let any_top = args.iter().any(|v| matches!(v, Variable::Top))
                                        || kwargs.iter().any(|(_, v)| matches!(v, Variable::Top));
                                    if any_top {
                                        out_vars.insert(Variable::Top);
                                    } else {
                                        out_vars
                                            .insert(Variable::Tensor(model.infer(args, kwargs)?));
                                    }
                                }
                            }
                            None => {
                                println!("couldn't resolve function {} to model", func);
                                out_vars.insert(Variable::Top);
                            }
                        }
                    }
                    Some(receiver) => todo!(),
                }

                Ok(out_vars)
            }
            ExprKind::Constant(c) => match c {
                Constant::Int(i) => {
                    Ok(HashSet::from_iter(vec![Variable::DimVar(DimVar::from(*i))]))
                }
                _ => Ok(HashSet::from_iter(vec![Variable::Top])),
            },
            ExprKind::Path(p) => {
                if p.parts().last().unwrap() == "shape" {
                    let prefix: Path = Path::new(&p.parts()[0..(p.parts().len() - 1)]);
                    return match domain.get(&prefix) {
                        Some(vars) => {
                            let shape_dims: Vec<_> =
                                vars.iter().filter_map(|v| v.as_shape_dims()).collect();
                            let tuples = shape_dims.into_iter().map(|ds| {
                                Variable::Tuple(ds.into_iter().map(Variable::DimVar).collect())
                            });
                            let set = HashSet::from_iter(tuples);
                            Ok(set)
                        }
                        None => Ok(HashSet::from_iter([Variable::Top])),
                    };
                }

                Ok(domain
                    .get(p)
                    .unwrap_or(&HashSet::from_iter([Variable::Top]))
                    .clone())
            }
            ExprKind::Index { receiver, index } => {
                // TODO:
                // - parse None dims i.e. x[None,:]
                let vars = self.eval_expr(domain, receiver)?;
                let indices: Vec<HashSet<Either<Variable, DimSlice>>> = index
                    .iter()
                    .map(|v| -> Result<HashSet<Either<Variable, DimSlice>>> {
                        Ok(match v {
                            Either::Left(l) => self
                                .eval_expr(domain, l)?
                                .into_iter()
                                .map(Either::Left)
                                .collect::<HashSet<_>>(),
                            Either::Right(Slice { lower, upper }) => {
                                let lowers = match lower {
                                    Some(expr_lower) => Some(self.eval_expr(domain, expr_lower)?),
                                    None => None,
                                };
                                let uppers = match upper {
                                    Some(expr_upper) => Some(self.eval_expr(domain, expr_upper)?),
                                    None => None,
                                };

                                match (lowers, uppers) {
                                    (Some(lowers), Some(uppers)) => lowers
                                        .iter()
                                        .cartesian_product(uppers.iter())
                                        .map(|(l, u)| {
                                            Either::Right(DimSlice {
                                                lower: Some(l.clone()),
                                                upper: Some(u.clone()),
                                            })
                                        })
                                        .collect::<HashSet<_>>(),
                                    (Some(lowers), None) => lowers
                                        .into_iter()
                                        .map(|l| {
                                            Either::Right(DimSlice {
                                                lower: Some(l),
                                                upper: None,
                                            })
                                        })
                                        .collect::<HashSet<_>>(),
                                    (None, Some(uppers)) => uppers
                                        .into_iter()
                                        .map(|u| {
                                            Either::Right(DimSlice {
                                                lower: None,
                                                upper: Some(u),
                                            })
                                        })
                                        .collect::<HashSet<_>>(),
                                    (None, None) => vec![Either::Right(DimSlice {
                                        lower: None,
                                        upper: None,
                                    })]
                                    .into_iter()
                                    .collect::<HashSet<_>>(),
                                }
                            }
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;

                let mut set = HashSet::new();
                for (var, index) in vars
                    .iter()
                    .cartesian_product(indices.iter().multi_cartesian_product())
                {
                    match var {
                        Variable::Tensor(Shape(dims)) => {
                            let mut out_dims = Vec::new();

                            for (i, dim) in index.iter().enumerate() {
                                match dim {
                                    // TODO: actually parse this, assuming it's a dimvar right now
                                    Either::Left(_v) => continue,
                                    Either::Right(DimSlice { lower, upper }) => {
                                        let l_bound = match lower {
                                            Some(Variable::DimVar(dvar)) => dvar.clone(),
                                            None => DimVar {
                                                kind: DimKind::Concrete(0),
                                            },
                                            _ => unreachable!("bad lower bound"),
                                        };
                                        let u_bound = match upper {
                                            Some(Variable::DimVar(dvar)) => match dvar.kind() {
                                                DimKind::Concrete(n) => {
                                                    if n < 0 {
                                                        dims[i].clone() + n.into()
                                                    } else {
                                                        dvar.clone()
                                                    }
                                                }
                                                _ => dvar.clone(),
                                            },
                                            None => dims[i].clone(),
                                            _ => unreachable!("bad upper bound"),
                                        };

                                        out_dims.push(u_bound - l_bound)
                                    }
                                }
                            }

                            set.insert(Variable::Tensor(Shape(out_dims)));
                        }
                        Variable::Tuple(elts) => {
                            assert!(index.len() == 1);
                            match index.first().unwrap() {
                                Either::Left(var) => {
                                    if let Some(c) = var.as_concrete_dimvar() {
                                        set.insert(elts.get(c as usize).unwrap().clone());
                                    } else {
                                        set.insert(Variable::Top);
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
                                        Some(elts.len())
                                    };

                                    match (lower, upper) {
                                        (Some(l), Some(u)) => {
                                            let tuple = Variable::Tuple(elts[l..u].to_vec());
                                            set.insert(tuple);
                                        }
                                        _ => {
                                            set.insert(Variable::Top);
                                        }
                                    }
                                }
                            }
                        }
                        Variable::Top => {
                            set.insert(Variable::Top);
                        }
                        Variable::DimVar(_) => (),
                        Variable::None => (),
                    }
                }

                Ok(set)
            }

            ExprKind::Tuple(exprs) => {
                let results = exprs
                    .iter()
                    .map(|e| self.eval_expr(domain, e))
                    .collect::<Result<Vec<HashSet<Variable>>>>()?;

                let products = results
                    .iter()
                    .map(|set| set.iter().cloned())
                    .multi_cartesian_product();

                Ok(HashSet::from_iter(products.map(Variable::Tuple)))
            }
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
            let preds = func.predecessors(loc);
            if preds.is_empty() {
                domain = self.state.get(&Location::START).unwrap().clone();
            } else {
                for pred_loc in preds {
                    domain.join(self.state.entry(pred_loc).or_default());
                }
            }
            match func.instr(loc) {
                Either::Left(stmt) => self.handle_stmt(&mut domain, stmt),
                Either::Right(term) => self.handle_term(&mut domain, term),
            }?;
            self.state.insert(*loc, domain);
        }
        Ok(())
    }
}

pub fn analyze(prog: Program) -> Result<GlobalAnalysis> {
    let mut global_analysis = GlobalAnalysis::new(&prog.functions);
    for func in prog.functions {
        // TODO: maybe do some nice caching later for modularity with user's own funcs
        global_analysis.analyze_func(&func)?;
    }
    Ok(global_analysis)
}
