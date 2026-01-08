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
use crate::ir::{Class, Function, Program};
use crate::ir::{Expr, Identifier, Parameter, Statement, Terminator, intern, resolve};
use anyhow::Result;
pub use errors::ShapeError;
type AnalysisDomain = HashMap<Identifier, HashSet<Variable>>;

pub use print::{
    class_with_inferred_shapes_to_string, function_with_inferred_shapes_to_string,
    print_class_with_inferred_shapes, print_function_with_inferred_shapes,
};

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
    pub functions: HashMap<Identifier, FunctionAnalysis>,
    pub classes: HashMap<Identifier, ClassAnalysis>,
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
            classes: HashMap::new(),
            models: Arc::new(ModelContext::new(funcs)),
        }
    }

    pub fn analyze_func(&mut self, func: &Function) -> Result<()> {
        let name = func.identifier.clone();
        let mut func_analysis = FunctionAnalysis::new(func, Arc::clone(&self.models), None);
        func_analysis.analyze_func(func)?;
        self.functions.insert(name, func_analysis);
        Ok(())
    }

    pub fn analyze_class(&mut self, class: &Class) -> Result<()> {
        let name = class.identifier.clone();
        let class_analysis = analyze_class(class, Arc::clone(&self.models))?;
        self.classes.insert(name, class_analysis);
        Ok(())
    }

    /// Format the entire analysis result as a string, including both classes and functions.
    pub fn format_all(&self, prog: &Program) -> String {
        let mut output = String::new();

        // Print classes first
        for (name, facts) in self
            .classes
            .iter()
            .sorted_by(|(a, _), (b, _)| resolve(**a).cmp(&resolve(**b)))
        {
            if let Some(class) = prog.classes.iter().find(|c| c.identifier == *name) {
                output.push_str(&class_with_inferred_shapes_to_string(class, facts, None));
                output.push_str("\n\n");
            }
        }

        // Then print functions
        for (name, facts) in self
            .functions
            .iter()
            .sorted_by(|(a, _), (b, _)| resolve(**a).cmp(&resolve(**b)))
        {
            if let Some(func) = prog.functions.iter().find(|f| f.identifier == *name) {
                output.push_str(&function_with_inferred_shapes_to_string(func, facts, None));
                output.push_str("\n\n");
            }
        }

        output
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
                        // Only show hints for Ident targets
                        if let ExprKind::Ident(name) = &target.kind {
                            if let Some(vars) =
                                func_analysis.state.get(loc).and_then(|d| d.get(name))
                            {
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
            }
        }

        hints
    }
}

#[derive(Debug)]
pub struct FunctionAnalysis {
    // TODO: currently just using Hash{Set,Map}s, but would benefit perhaps
    // from using bitsets, if the speedup is worth it
    pub id: Identifier,
    pub state: HashMap<Location, AnalysisDomain>,
    pub models: Arc<ModelContext>,
    pub function: Function,
}

impl FunctionAnalysis {
    /// Create a new FunctionAnalysis.
    ///
    /// If `class_attributes` is `Some`, this is a method analysis and the state will include
    /// both the class attributes (self.X) and function parameters.
    /// If `class_attributes` is `None`, this is a top-level function and only params will be used.
    fn new(
        func: &Function,
        models: Arc<ModelContext>,
        class_attributes: Option<AnalysisDomain>,
    ) -> Self {
        let mut state = HashMap::new();

        let mut start_domain = AnalysisDomain::new();

        // populate with params
        for Parameter(ident, var) in &func.params {
            if let Some(var) = var {
                start_domain.insert(ident.clone(), HashSet::from([var.clone()]));
            }
        }

        // If this is a method, also include class attributes (self.X)
        if let Some(class_attrs) = class_attributes {
            start_domain.extend(class_attrs);
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
            ExprKind::Ident(_name) => self.eval_ident(domain, expr),
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
        if let ExprKind::Ident(name) = &expr.kind {
            Ok(domain
                .get(name)
                .unwrap_or(&HashSet::from([Variable::Top]))
                .clone())
        } else {
            Ok(HashSet::from([Variable::Top]))
        }
    }

    /// Attribute lookup
    fn eval_attribute(
        &mut self,
        domain: &AnalysisDomain,
        value: &Expr,
        attr: &Identifier,
        _expr: &Expr,
    ) -> Result<HashSet<Variable>> {
        // Special case for .shape attribute
        if resolve(*attr) == "shape" {
            return self.eval_shape_attribute(domain, value);
        }

        // Handle self.foo case for methods
        // Check if value is "self" identifier
        if let ExprKind::Ident(self_ident) = &value.kind {
            if resolve(*self_ident) == "self" {
                // This is a method (determined by presence of self.X in domain)
                // Look up "self.foo" in the domain
                let self_attr_name = intern(&format!("self.{}", resolve(*attr)));
                if let Some(vars) = domain.get(&self_attr_name) {
                    return Ok(vars.clone());
                }
            }
        }

        // For now, we're not doing true heap modelling so we can't resolve attributes
        Ok(HashSet::from([Variable::Top]))
    }

    /// Special case .shape attribute
    fn eval_shape_attribute(
        &mut self,
        domain: &AnalysisDomain,
        value: &Expr,
    ) -> Result<HashSet<Variable>> {
        // Try to get the identifier from the value expression
        let vars = if let ExprKind::Ident(name) = &value.kind {
            domain.get(name)
        } else {
            None
        };

        match vars {
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
        keyword_args: &[(Identifier, Expr)],
        span: SourceSpan,
    ) -> Result<HashSet<Variable>> {
        let args_sets = self.eval_call_args(domain, pos_args)?;
        let kwargs_sets = self.eval_call_kwargs(domain, keyword_args)?;

        let args_products = args_sets.iter().multi_cartesian_product();
        let kw_names = kwargs_sets.iter().map(|(n, _)| *n);
        let kwargs_products: Vec<HashMap<Identifier, _>> = kwargs_sets
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
        kwargs: &[(Identifier, Expr)],
    ) -> Result<Vec<(Identifier, HashSet<Variable>)>> {
        kwargs
            .iter()
            .map(|(n, arg_expr)| Ok((*n, self.eval_expr(domain, arg_expr)?)))
            .collect()
    }

    /// Convert an Expr to a dot-separated string (e.g., "torch.nn.functional.relu")
    fn expr_to_dot_string(&self, expr: &Expr) -> String {
        match &expr.kind {
            ExprKind::Ident(name) => resolve(*name),
            ExprKind::Attribute { value, attr } => {
                format!("{}.{}", self.expr_to_dot_string(value), resolve(*attr))
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
        _span: SourceSpan,
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

        if let Some(target) = &stmt.target {
            match &target.kind {
                ExprKind::Ident(name) => {
                    domain.insert(name.clone(), res_var);
                }
                ExprKind::Attribute { value, attr } => {
                    // Handle self.X assignments for methods
                    if let ExprKind::Ident(self_ident) = &value.kind {
                        if resolve(*self_ident) == "self" {
                            // Store as "self.X" in domain
                            let self_attr_name = intern(&format!("self.{}", resolve(*attr)));
                            domain.insert(self_attr_name, res_var);
                        }
                    }
                    // Ignore other attribute assignments for now
                }
                _ => {
                    // Ignore other cases for now
                }
            }
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
                    print::function_with_inferred_shapes_to_string(func, self, Some(*loc))
                );
                return Err(e);
            }
            self.state.insert(*loc, domain);
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct ClassAnalysis {
    /// The identifier of the class being analyzed
    pub id: Identifier,
    /// Mapping from attribute names (like "fc1", "fc2") to their inferred Variables.
    /// The Variables' dimvars are in terms of the annotated dimvars of the __init__ method.
    pub attributes: HashMap<Identifier, HashSet<Variable>>,
    /// Analysis results for each method (excluding __init__).
    /// These are used for consistency checking.
    pub methods: HashMap<Identifier, FunctionAnalysis>,
}

/// Analyze a class to infer tensor shapes for its attributes and methods.
///
/// This function:
/// 1. Analyzes the `__init__` method as a special case to extract attribute shapes
/// 2. Analyzes other methods for consistency checking
pub fn analyze_class(class: &Class, models: Arc<ModelContext>) -> Result<ClassAnalysis> {
    let init_method_name = intern("__init__");

    // Find the __init__ method
    let init_method = class.methods.get(&init_method_name).ok_or_else(|| {
        anyhow::anyhow!(
            "Class {} missing __init__ method",
            resolve(class.identifier)
        )
    })?;
    // TODO: Handle superclass inheritance - if __init__ is missing, check parent classes

    // Analyze __init__ method to extract attribute shapes
    let mut init_analysis = FunctionAnalysis::new(init_method, Arc::clone(&models), None);
    init_analysis.analyze_func(init_method)?;

    // Extract self.X assignments from final locations only
    // We look at final locations (Return terminators) and collect any "self.X" entries
    let mut attributes: HashMap<Identifier, HashSet<Variable>> = HashMap::new();
    let final_locs = init_method.final_locations();
    for final_loc in &final_locs {
        if let Some(domain) = init_analysis.state.get(final_loc) {
            for (ident, vars) in domain {
                let ident_str = resolve(*ident);
                if ident_str.starts_with("self.") {
                    // Extract attribute name (everything after "self.")
                    let attr_name = &ident_str[5..]; // "self.".len() == 5
                    let attr_ident = intern(attr_name);
                    // Use the first occurrence or join with existing
                    if let Some(existing_vars) = attributes.get_mut(&attr_ident) {
                        existing_vars.extend(vars.iter().cloned());
                    } else {
                        attributes.insert(attr_ident, vars.clone());
                    }
                }
            }
        } else {
            unreachable!("FunctionAnalysis didn't add domain at final location");
        }
    }

    // Build initial state for other methods with self.X attributes
    let mut method_class_attributes = AnalysisDomain::new();
    for (attr_ident, vars) in &attributes {
        let self_attr_name = intern(&format!("self.{}", resolve(*attr_ident)));
        method_class_attributes.insert(self_attr_name, vars.clone());
    }

    // Check other methods
    let mut methods = HashMap::new();
    for (method_name, method_func) in &class.methods {
        if *method_name == init_method_name {
            continue;
        }

        let mut method_analysis = FunctionAnalysis::new(
            method_func,
            Arc::clone(&models),
            Some(method_class_attributes.clone()),
        );
        method_analysis.analyze_func(method_func)?;
        methods.insert(*method_name, method_analysis);
    }

    Ok(ClassAnalysis {
        id: class.identifier,
        attributes,
        methods,
    })
}

pub fn analyze(prog: Program) -> Result<GlobalAnalysis> {
    let mut global_analysis = GlobalAnalysis::new(&prog.functions);
    for class in prog.classes {
        global_analysis.analyze_class(&class)?;
    }
    for func in prog.functions {
        global_analysis.analyze_func(&func)?;
    }
    Ok(global_analysis)
}
