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
use crate::ir::types::{Binop, Constant, ExprKind, Location};
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
            if let Some(e) = self.get_mut(&path) {
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
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
            models: Rc::new(ModelContext::new()),
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
    // func: StmtFunctionDef,
    // func: Function,
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
                            self.fold_dimvars(l_dvar.clone(), r_dvar.clone(), *op)
                        }
                        _ => {
                            panic!("runtime error")
                        }
                    };

                    out_vars.insert(out_var);
                }
                Ok(out_vars)
            }
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
                    let prefix = Path::new(&p.parts()[0..(p.parts().len() - 1)]);
                    return match domain.get(&prefix) {
                        Some(vars) => {
                            let shape_dims: Vec<_> =
                                vars.iter().filter_map(|v| v.as_shape_dims()).collect();
                            let tuples = shape_dims.into_iter().map(|ds| {
                                Variable::Tuple(
                                    ds.into_iter().map(|d| Variable::DimVar(d)).collect(),
                                )
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
            ExprKind::Slice { receiver, slice } => todo!(),
            ExprKind::Index { expr, index } => {
                let vars = self.eval_expr(domain, expr)?;
                let indices = self.eval_expr(domain, index)?;

                let mut set = HashSet::new();
                for (var, index) in vars.iter().cartesian_product(indices.iter()) {
                    println!("{:?}, {:?}", var, index);
                    let var = match (var, index) {
                        // in the case where we know the tensor but not the index, we could technically do a union over the DimVars
                        (Variable::Top, _) | (_, Variable::Top) => Variable::Top,

                        (Variable::Tuple(elts), Variable::DimVar(d)) => {
                            if let DimKind::Concrete(c) = d.kind() {
                                match elts.get(c as usize) {
                                    Some(v) => v.clone(),
                                    None => Variable::Top,
                                }
                            } else {
                                Variable::Top
                            }
                        }
                        _ => Variable::Top,
                    };

                    set.insert(var);
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

                Ok(HashSet::from_iter(
                    products.map(|vars| Variable::Tuple(vars)),
                ))
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
