mod dimvars;
mod errors;
mod models;
mod print;
mod types;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use itertools::{Either, Itertools};
use miette::SourceSpan;
use tower_lsp::lsp_types::InlayHint;
pub use types::{Shape, Variable};

pub use crate::analysis::dimvars::{DimKind, DimVar};
use crate::analysis::models::{Model, ModelContext};
use crate::analysis::types::DimSlice;
use crate::ir::types::{Binop, Constant, ExprKind, Location, Slice};
use crate::ir::{Expr, Parameter, Path, Statement, Terminator};
use crate::ir::{Function, Program};
use anyhow::Result;
pub use errors::ShapeError;
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

#[derive(Debug)]
pub struct GlobalAnalysis {
    pub functions: HashMap<Path, FunctionAnalysis>,
    pub models: Arc<ModelContext>,
}

fn vars_to_inlay(vars: &HashSet<Variable>) -> Option<String> {
    if vars.len() == 0 {
        None
    } else if vars.len() == 1 {
        let var = vars.iter().next().unwrap();
        Some(format!(": {}", var))
    } else {
        let mut var_strings: Vec<String> = vars.iter().map(|v| format!("{}", v)).collect();
        var_strings.sort();
        Some(": {".to_owned() + &var_strings.join(", ") + "}")
    }
}

impl GlobalAnalysis {
    pub fn new(funcs: &Vec<Function>) -> Self {
        Self {
            functions: HashMap::new(),
            models: Arc::new(ModelContext::new(funcs)),
        }
    }

    pub fn analyze_func(&mut self, func: &Function) -> Result<()> {
        let name = func.identifier.clone();
        let mut func_analysis = FunctionAnalysis::new(func, Arc::clone(&self.models));
        func_analysis.analyze_func(func)?;
        self.functions.insert(name, func_analysis);
        Ok(())
    }

    pub fn inlay_hints(&self) -> Vec<InlayHint> {
        let mut hints = Vec::new();

        for (_, func_analysis) in &self.functions {
            let func = &func_analysis.function;
            // for each location
            for loc in &func.locations {
                if let Either::Left(stmt) = func.instr(&loc) {
                    if let Some(target) = &stmt.target
                        && let Some(position) = stmt.assign_end
                    {
                        let vars = func_analysis.state.get(loc).unwrap().get(target).unwrap();
                        if let Some(label) = vars_to_inlay(vars) {
                            hints.push(InlayHint {
                                position,
                                label: tower_lsp::lsp_types::InlayHintLabel::String(label),
                                kind: None,
                                text_edits: None,
                                tooltip: None,
                                padding_left: None,
                                padding_right: None,
                                data: None,
                            });
                        }
                    }
                }
            }
        }

        hints
    }
}

#[derive(Debug)]
pub struct FunctionAnalysis {
    // TODO: currently just using Hash{Set,Map}s, but would benefit perhaps
    // from using bitsets, if the speedup is worth it
    pub id: Path,
    pub state: HashMap<Location, AnalysisDomain>,
    pub models: Arc<ModelContext>,
    pub function: Function,
}

impl FunctionAnalysis {
    fn new(func: &Function, models: Arc<ModelContext>) -> Self {
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
            function: func.clone(),
        }
    }

    fn fold_dimvars(&self, left_dimvar: DimVar, right_dimvar: DimVar, op: Binop) -> Variable {
        left_dimvar.binop(&right_dimvar, op)
    }

    /// Dispatch expr to eval_* methods, returning set of possible lattice elements.
    fn eval_expr(&mut self, domain: &AnalysisDomain, expr: &Expr) -> Result<HashSet<Variable>> {
        match &expr.kind {
            ExprKind::Binop { left, right, op } => {
                self.eval_binop(domain, left, right, *op, expr.span)
            }
            ExprKind::Call {
                function,
                pos_args,
                keyword_args,
            } => self.eval_call(domain, function, pos_args, keyword_args, expr.span),
            ExprKind::Constant(c) => self.eval_constant(c),
            ExprKind::Ident(name) => self.eval_ident(domain, expr),
            ExprKind::Attribute { value, attr } => self.eval_attribute(domain, value, attr, expr),
            ExprKind::Index { receiver, index } => {
                self.eval_index(domain, receiver, index, expr.span)
            }
            ExprKind::Tuple(exprs) => self.eval_tuple(domain, exprs),
        }
    }

    /// For each combo of possible left/right lattice elements computes new lattice element.
    /// Uses broadcast model if op isn't matmul.
    fn eval_binop(
        &mut self,
        domain: &AnalysisDomain,
        left: &Expr,
        right: &Expr,
        op: Binop,
        span: SourceSpan,
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
                        let out_shape = self.models.torch.matmul.infer(
                            vec![l_var, r_var],
                            HashMap::new(),
                            span,
                        )?;
                        Variable::Tensor(out_shape)
                    } else {
                        let out_shape = self.models.torch.broadcast.infer(
                            vec![l_var, r_var],
                            HashMap::new(),
                            span,
                        )?;
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

    /// Creat constant DimVar if possible
    fn eval_constant(&self, c: &Constant) -> Result<HashSet<Variable>> {
        Ok(match c {
            Constant::Int(i) => HashSet::from([Variable::DimVar(DimVar::from(*i))]),
            _ => HashSet::from([Variable::Top]),
        })
    }

    /// Lookup identifier
    fn eval_ident(&self, domain: &AnalysisDomain, expr: &Expr) -> Result<HashSet<Variable>> {
        Ok(domain
            .get(expr)
            .unwrap_or(&HashSet::from([Variable::Top]))
            .clone())
    }

    /// Attribute lookup
    fn eval_attribute(
        &mut self,
        domain: &AnalysisDomain,
        value: &Expr,
        attr: &str,
        expr: &Expr,
    ) -> Result<HashSet<Variable>> {
        // Special case for .shape attribute
        if attr == "shape" {
            return self.eval_shape_attribute(domain, value);
        }

        // Fallback: treat as path lookup
        Ok(domain
            .get(expr)
            .unwrap_or(&HashSet::from([Variable::Top]))
            .clone())
    }

    /// Special case .shape attribute
    fn eval_shape_attribute(
        &mut self,
        domain: &AnalysisDomain,
        value: &Expr,
    ) -> Result<HashSet<Variable>> {
        match domain.get(value) {
            Some(vars) => {
                let shape_dims: Vec<_> = vars.iter().filter_map(|v| v.as_shape_dims()).collect();
                let tuples = shape_dims
                    .into_iter()
                    .map(|ds| Variable::Tuple(ds.into_iter().map(Variable::DimVar).collect()));
                let set = HashSet::from_iter(tuples);
                Ok(set)
            }
            None => Ok(HashSet::from([Variable::Top])),
        }
    }

    /// Calls eval_expr on each expr in tuple and returns cartesian product
    fn eval_tuple(&mut self, domain: &AnalysisDomain, exprs: &[Expr]) -> Result<HashSet<Variable>> {
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

    /// Enumerates possible arg/kwargs and dispatches call to correct model.
    fn eval_call(
        &mut self,
        domain: &AnalysisDomain,
        function: &Expr,
        pos_args: &[Expr],
        keyword_args: &[(String, Expr)],
        span: SourceSpan,
    ) -> Result<HashSet<Variable>> {
        let args_sets = self.eval_call_args(domain, pos_args)?;
        let kwargs_sets = self.eval_call_kwargs(domain, keyword_args)?;

        let args_products = args_sets.iter().multi_cartesian_product();
        let kw_names = kwargs_sets.iter().map(|(n, _)| n.clone());
        let kwargs_products: Vec<HashMap<_, _>> = kwargs_sets
            .iter()
            .map(|(_, vars)| vars)
            .multi_cartesian_product()
            .map(|vars| kw_names.clone().zip(vars).collect())
            .collect();

        let mut out_vars = HashSet::new();

        // Resolve function name - for now assume it's an ident or attribute chain
        let func_name = self.expr_to_dot_string(function);

        match self.models.resolve(&func_name) {
            Some(model) => {
                for (args, kwargs) in args_products.cartesian_product(kwargs_products) {
                    let any_top = args.iter().any(|v| matches!(v, Variable::Top))
                        || kwargs.iter().any(|(_, v)| matches!(v, Variable::Top));
                    if any_top {
                        out_vars.insert(Variable::Top);
                    } else {
                        out_vars.insert(Variable::Tensor(model.infer(args, kwargs, span)?));
                    }
                }
            }
            None => {
                println!("couldn't resolve function {} to model", func_name);
                out_vars.insert(Variable::Top);
            }
        }

        Ok(out_vars)
    }

    /// Calls eval_expr on each arg
    fn eval_call_args(
        &mut self,
        domain: &AnalysisDomain,
        args: &[Expr],
    ) -> Result<Vec<HashSet<Variable>>> {
        args.iter()
            .map(|arg_expr| self.eval_expr(domain, arg_expr))
            .collect()
    }

    /// Calls eval_expr on each kwarg
    fn eval_call_kwargs(
        &mut self,
        domain: &AnalysisDomain,
        kwargs: &[(String, Expr)],
    ) -> Result<Vec<(String, HashSet<Variable>)>> {
        kwargs
            .iter()
            .map(|(n, arg_expr)| Ok((n.clone(), self.eval_expr(domain, arg_expr)?)))
            .collect()
    }

    /// Convert an Expr to a dot-separated string (e.g., "torch.nn.functional.relu")
    fn expr_to_dot_string(&self, expr: &Expr) -> String {
        match &expr.kind {
            ExprKind::Ident(name) => name.clone(),
            ExprKind::Attribute { value, attr } => {
                format!("{}.{}", self.expr_to_dot_string(value), attr)
            }
            _ => String::new(),
        }
    }

    /// Applies an index (sequence of slices/indices) to an Expr, returning the set of possible outcomes
    fn eval_index(
        &mut self,
        domain: &AnalysisDomain,
        receiver: &Expr,
        index: &[Either<Expr, Slice>],
        span: SourceSpan,
    ) -> Result<HashSet<Variable>> {
        let vars = self.eval_expr(domain, receiver)?;
        let indices = self.eval_indices(domain, index)?;

        let mut set = HashSet::new();
        'possible_slices_loop: for (var, index) in vars
            .iter()
            .cartesian_product(indices.iter().multi_cartesian_product())
        {
            match var {
                Variable::Tensor(Shape(dims)) => {
                    if let Some(result) = self.eval_tensor_index(dims, &index)? {
                        set.insert(result);
                    } else {
                        set.insert(Variable::Top);
                        continue 'possible_slices_loop;
                    }
                }
                Variable::Tuple(elts) => {
                    if let Some(result) = self.eval_tuple_index(elts, &index) {
                        set.insert(result);
                    } else {
                        set.insert(Variable::Top);
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

    /// Turns an index (sequence of indices/slices from a Subscript) into elements in the lattice (Variable::Dimvar or DimSlice).
    /// Helper for eval_index.
    fn eval_indices(
        &mut self,
        domain: &AnalysisDomain,
        index: &[Either<Expr, Slice>],
    ) -> Result<Vec<HashSet<Either<Variable, DimSlice>>>> {
        index
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
            .collect()
    }

    /// Given a sequence of DimVars, apply a sequence of indices/slices.
    /// Helper for eval_index.
    fn eval_tensor_index(
        &self,
        dims: &[DimVar],
        index_or_slice: &[&Either<Variable, DimSlice>],
    ) -> Result<Option<Variable>> {
        let mut out_dims = Vec::new();

        for (i, dim) in index_or_slice.iter().enumerate() {
            match dim {
                // TODO: actually parse this, assuming it's a dimvar right now
                Either::Left(_v) => continue,
                Either::Right(DimSlice { lower, upper }) => {
                    let l_bound = match lower {
                        Some(Variable::DimVar(dvar)) => dvar.clone(),
                        None => DimVar {
                            kind: DimKind::Concrete(0),
                        },
                        _ => {
                            // bound doesn't make sense
                            return Ok(None);
                        }
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
                        _ => {
                            // bound doesn't make sense
                            return Ok(None);
                        }
                    };

                    out_dims.push(u_bound - l_bound)
                }
            }
        }

        Ok(Some(Variable::Tensor(Shape(out_dims))))
    }

    /// Grabs an element or slice out of tuple_elts based on index (if it's an index or slice).
    /// Currently only handling case where there is a single index/slice.
    /// Helper for eval_index.
    fn eval_tuple_index(
        &self,
        tuple_elts: &[Variable],
        index_or_slices: &[&Either<Variable, DimSlice>],
    ) -> Option<Variable> {
        assert!(index_or_slices.len() == 1);
        match index_or_slices.first().unwrap() {
            Either::Left(var) => {
                if let Some(c) = var.as_concrete_dimvar() {
                    tuple_elts.get(c as usize).cloned()
                } else {
                    Some(Variable::Top)
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
                    Some(tuple_elts.len())
                };

                match (lower, upper) {
                    (Some(l), Some(u)) => {
                        let tuple = Variable::Tuple(tuple_elts[l..u].to_vec());
                        Some(tuple)
                    }
                    _ => Some(Variable::Top),
                }
            }
        }
    }

    /// Eval expr, updating target if this statement is an assignment.
    fn handle_stmt(&mut self, domain: &mut AnalysisDomain, stmt: &Statement) -> Result<()> {
        let res_var = self.eval_expr(domain, &stmt.value)?;
        if let Some(path) = &stmt.target {
            domain.insert(path.clone(), res_var);
        }
        Ok(())
    }

    /// Eval potential exprs to check for shape consistency.
    /// We don't model side effects so the domain won't change.
    fn handle_term(&mut self, domain: &mut AnalysisDomain, term: &Terminator) -> Result<()> {
        match term {
            Terminator::CondJump { cond, .. } => {
                if let Some(expr) = cond {
                    self.eval_expr(domain, expr)?;
                }
            }
            Terminator::Return(expr) => {
                if let Some(expr) = expr {
                    self.eval_expr(domain, expr)?;
                }
            }
            Terminator::Jump(_) => (),
        }

        Ok(())
    }

    /// Run single-pass dataflow analysis on function
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
            let result = match func.instr(loc) {
                Either::Left(stmt) => self.handle_stmt(&mut domain, stmt),
                Either::Right(term) => self.handle_term(&mut domain, term),
            };
            if let Err(e) = result {
                eprintln!(
                    "{}",
                    print::ir_with_inferred_shapes_to_string(func, self, Some(*loc))
                );
                return Err(e);
            }
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
