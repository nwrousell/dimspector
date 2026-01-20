use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use itertools::{Either, Itertools};
use miette::SourceSpan;

use crate::analysis::dimvars::{DimKind, DimVar};
use crate::analysis::models::{ArgSpans, Model};
use crate::analysis::types::{Collection, DimSlice, Shape, Variable};
use crate::ir::types::{Binop, Constant, ExprKind, Location, Slice, Type};
use crate::ir::{Expr, Function, Identifier, Parameter, Statement, Terminator, intern, resolve};
use anyhow::Result;

use super::print;
use super::{AnalysisDomain, GlobalAnalysis, JoinSemiLattice};

#[derive(Debug)]
pub struct FunctionAnalysis {
    pub id: Identifier,
    pub file_path: PathBuf,
    pub state: HashMap<Location, AnalysisDomain>,
    pub function: Function,

    /// Stores the assigned variable(s) for each assignment statement location
    pub assignments: HashMap<Location, HashSet<Variable>>,
}

impl FunctionAnalysis {
    /// Create a new FunctionAnalysis.
    ///
    /// If `class_attributes` is `Some`, this is a method analysis and the state will include
    /// both the class attributes (self.X) and function parameters.
    /// If `class_attributes` is `None`, this is a top-level function and only params will be used.
    pub(crate) fn new(func: &Function, class_attributes: Option<AnalysisDomain>) -> Self {
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
            file_path: func.file_path.clone(),
            state,
            assignments: HashMap::new(),
            function: func.clone(),
        }
    }

    fn fold_dimvars(&self, left_dimvar: DimVar, right_dimvar: DimVar, op: Binop) -> Variable {
        left_dimvar.binop(&right_dimvar, op)
    }

    /// Dispatch expr to eval_* methods, returning set of possible lattice elements.
    fn eval_expr(
        &mut self,
        domain: &AnalysisDomain,
        expr: &Expr,
        global: &GlobalAnalysis,
    ) -> Result<HashSet<Variable>> {
        match &expr.kind {
            ExprKind::Binop { left, right, op } => {
                self.eval_binop(domain, left, right, *op, expr.span, global)
            }
            ExprKind::Call {
                function,
                pos_args,
                keyword_args,
            } => self.eval_call(
                domain,
                function,
                pos_args,
                keyword_args,
                expr.span,
                expr.ty.clone(),
                global,
            ),
            ExprKind::Constant(c) => self.eval_constant(c),
            ExprKind::Ident(_name) => self.eval_ident(domain, expr),
            ExprKind::Attribute { value, attr } => {
                self.eval_attribute(domain, value, attr, expr, global)
            }
            ExprKind::Index { receiver, index } => {
                self.eval_index(domain, receiver, index, expr.span, global)
            }
            ExprKind::Tuple(exprs) => self.eval_tuple(domain, exprs, global),
            ExprKind::List(exprs) => self.eval_list(domain, exprs, global),
            ExprKind::Dict(items) => self.eval_dict(domain, items, global),
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
        global: &GlobalAnalysis,
    ) -> Result<HashSet<Variable>> {
        let l_vars = self.eval_expr(domain, left, global)?;
        let r_vars = self.eval_expr(domain, right, global)?;

        let mut out_vars = HashSet::new();

        let is_matmul = matches!(op, Binop::MatMult);

        for (l_var, r_var) in l_vars.iter().cartesian_product(r_vars.iter()) {
            let out_var = match (l_var, r_var) {
                (Variable::Top, _) | (_, Variable::Top) => Variable::Top,
                (Variable::Tensor(_), Variable::Tensor(_)) => {
                    let arg_spans = ArgSpans::new(vec![left.span, right.span], HashMap::new());
                    if is_matmul {
                        let out_shape = global.models.torch.matmul.infer(
                            vec![l_var, r_var],
                            HashMap::new(),
                            span,
                            &arg_spans,
                        )?;
                        Variable::Tensor(out_shape)
                    } else {
                        let out_shape = global.models.torch.broadcast.infer(
                            vec![l_var, r_var],
                            HashMap::new(),
                            span,
                            &arg_spans,
                        )?;
                        Variable::Tensor(out_shape)
                    }
                }
                (Variable::Tensor(shape), _) | (_, Variable::Tensor(shape)) => {
                    // other should be some number, will retain tensor operand shape
                    Variable::Tensor(shape.clone())
                }
                (
                    Variable::Collection(Collection::Tuple(l_vars)),
                    Variable::Collection(Collection::Tuple(r_vars)),
                ) => match op {
                    Binop::Add => {
                        let mut out = l_vars.clone();
                        out.extend(r_vars.iter().cloned());
                        Variable::Collection(Collection::Tuple(out))
                    }
                    _ => Variable::Top,
                },
                (
                    Variable::Collection(Collection::List(l_vars)),
                    Variable::Collection(Collection::List(r_vars)),
                ) => match op {
                    Binop::Add => {
                        let mut out = l_vars.clone();
                        out.extend(r_vars.iter().cloned());
                        Variable::Collection(Collection::List(out))
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
    /// doesn't include method calls, as eval_call doesn't recur there.
    fn eval_attribute(
        &mut self,
        domain: &AnalysisDomain,
        value: &Expr,
        attr: &Identifier,
        _expr: &Expr,
        _global: &GlobalAnalysis, // will be useful to access the symbol table when we handle cross-module variable lookup
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
                let tuples = shape_dims.into_iter().map(|ds| {
                    Collection::tup_from_elts(ds.into_iter().map(Variable::DimVar).collect())
                });
                let set = HashSet::from_iter(tuples);
                Ok(set)
            }
            None => Ok(HashSet::from([Variable::Top])),
        }
    }

    /// Calls eval_expr on each expr in tuple and returns cartesian product
    fn eval_tuple(
        &mut self,
        domain: &AnalysisDomain,
        exprs: &[Expr],
        global: &GlobalAnalysis,
    ) -> Result<HashSet<Variable>> {
        let results = exprs
            .iter()
            .map(|e| self.eval_expr(domain, e, global))
            .collect::<Result<Vec<HashSet<Variable>>>>()?;

        let products = results
            .iter()
            .map(|set| set.iter().cloned())
            .multi_cartesian_product();

        Ok(HashSet::from_iter(
            products.map(|e| Variable::Collection(Collection::Tuple(e))),
        ))
    }

    fn eval_list(
        &mut self,
        domain: &AnalysisDomain,
        exprs: &[Expr],
        global: &GlobalAnalysis,
    ) -> Result<HashSet<Variable>> {
        let results = exprs
            .iter()
            .map(|e| self.eval_expr(domain, e, global))
            .collect::<Result<Vec<HashSet<Variable>>>>()?;

        let products = results
            .iter()
            .map(|set| set.iter().cloned())
            .multi_cartesian_product();

        Ok(HashSet::from_iter(
            products.map(|e| Variable::Collection(Collection::List(e))),
        ))
    }

    fn eval_dict(
        &mut self,
        domain: &AnalysisDomain,
        exprs: &[(Expr, Expr)],
        global: &GlobalAnalysis,
    ) -> Result<HashSet<Variable>> {
        use crate::analysis::types::DictKey;
        use std::collections::BTreeMap;

        let mut kv_results: Vec<(DictKey, HashSet<Variable>)> = Vec::new();

        for (key_expr, value_expr) in exprs {
            let dict_key = match &key_expr.kind {
                ExprKind::Constant(Constant::Int(i)) => Some(DictKey::Int(*i as i64)),
                ExprKind::Constant(Constant::Str(s)) => Some(DictKey::Str(s.clone())),
                _ => None,
            };

            if let Some(key) = dict_key {
                let values = self.eval_expr(domain, value_expr, global)?;
                kv_results.push((key, values));
            }
            // Skip non-constant keys (can't track them in abstract interpretation)
        }

        // If no valid keys, return empty set or Top
        if kv_results.is_empty() {
            return Ok(HashSet::from([Variable::Top]));
        }

        // Get cartesian product of all possible value combinations
        let value_sets: Vec<&HashSet<Variable>> =
            kv_results.iter().map(|(_, values)| values).collect();

        let value_products = value_sets
            .iter()
            .map(|set| set.iter())
            .multi_cartesian_product();

        // Build a BTreeMap for each combination of values
        let mut result = HashSet::new();
        for value_combination in value_products {
            let mut map = BTreeMap::new();
            for (i, value_var) in value_combination.iter().enumerate() {
                let key = kv_results[i].0.clone();
                map.insert(key, (*value_var).clone());
            }
            result.insert(Variable::Collection(Collection::Dict(map)));
        }

        Ok(result)
    }

    /// Enumerates possible arg/kwargs and dispatches call to correct model.
    fn eval_call(
        &mut self,
        domain: &AnalysisDomain,
        function: &Expr,
        pos_args: &[Expr],
        keyword_args: &[(Identifier, Expr)],
        span: SourceSpan,
        call_ty: Type,
        global: &GlobalAnalysis,
    ) -> Result<HashSet<Variable>> {
        let args_sets = self.eval_call_args(domain, pos_args, global)?;
        let kwargs_sets = self.eval_call_kwargs(domain, keyword_args, global)?;

        // Collect argument spans for better error messages
        let arg_spans = ArgSpans::new(
            pos_args.iter().map(|e| e.span).collect(),
            keyword_args.iter().map(|(id, e)| (*id, e.span)).collect(),
        );

        match &call_ty {
            Type::Constructor(class_id) => {
                self.eval_constructor(*class_id, &args_sets, &kwargs_sets, global)
            }
            Type::Method => self.eval_method(
                function,
                domain,
                &args_sets,
                &kwargs_sets,
                span,
                &arg_spans,
                global,
            ),
            _ => {
                // check if this is an instance call (e.g., `model(x)` where model is a ClassInstance)
                if let ExprKind::Ident(ident) = &function.kind
                    && let Some(vars) = domain.get(ident)
                {
                    for var in vars {
                        if let Some(instance) = var.as_class_instance()
                            && let Some(class_analysis) = global.classes.get(&instance.class_id)
                        {
                            let method_name = if class_analysis.is_nn_module {
                                intern("forward")
                            } else {
                                intern("__call__")
                            };

                            // Create a synthetic attribute expression for the method call
                            let method_expr = Expr::attribute(
                                function.clone(),
                                method_name,
                                ruff_text_size::TextRange::default(),
                                Type::Method,
                            );

                            return self.eval_method(
                                &method_expr,
                                domain,
                                &args_sets,
                                &kwargs_sets,
                                span,
                                &arg_spans,
                                global,
                            );
                        }
                    }
                }

                // fallback to regular function call
                self.eval_function(function, &args_sets, &kwargs_sets, span, &arg_spans, global)
            }
        }
    }

    fn eval_constructor(
        &mut self,
        class_id: Identifier,
        args_sets: &[HashSet<Variable>],
        kwargs_sets: &[(Identifier, HashSet<Variable>)],
        global: &GlobalAnalysis,
    ) -> Result<HashSet<Variable>> {
        let mut out_vars = HashSet::new();

        // Resolve class_id to canonical path
        let local_name = resolve(class_id);
        let canonical = global
            .symbol_table
            .resolve(&self.function.file_path, &local_name)
            .cloned()
            .unwrap_or(local_name);
        let canonical_id = intern(&canonical);

        if let Some(class_analysis) = global.classes.get(&canonical_id) {
            let args_products = args_sets.iter().multi_cartesian_product();
            let kw_names = kwargs_sets.iter().map(|(n, _)| *n);
            let kwargs_products: Vec<HashMap<Identifier, _>> = kwargs_sets
                .iter()
                .map(|(_, vars)| vars)
                .multi_cartesian_product()
                .map(|vars| kw_names.clone().zip(vars).collect())
                .collect();

            // For each combination of args/kwargs, create an instance
            for (args_refs, kwargs_ref) in args_products.cartesian_product(kwargs_products.iter()) {
                // Convert references to owned values
                let args: Vec<Variable> = args_refs.iter().map(|v| (*v).clone()).collect();
                let kwargs: HashMap<Identifier, Variable> =
                    kwargs_ref.iter().map(|(k, v)| (*k, (*v).clone())).collect();

                let any_top = args.iter().any(|v| matches!(v, Variable::Top))
                    || kwargs.iter().any(|(_, v)| matches!(v, Variable::Top));
                if any_top {
                    out_vars.insert(Variable::Top);
                } else {
                    // Create class instance with concrete substitutions
                    let instance = class_analysis.create_instance(&args, &kwargs)?;
                    out_vars.insert(instance);
                }
            }
        } else {
            // Class not found in analysis
            out_vars.insert(Variable::Top);
        }

        Ok(out_vars)
    }

    fn eval_method(
        &mut self,
        function: &Expr,
        domain: &AnalysisDomain,
        args_sets: &[HashSet<Variable>],
        kwargs_sets: &[(Identifier, HashSet<Variable>)],
        span: SourceSpan,
        arg_spans: &ArgSpans,
        global: &GlobalAnalysis,
    ) -> Result<HashSet<Variable>> {
        let mut out_vars = HashSet::new();

        // Extract receiver and method name from the function expression
        let (receiver_expr, method_name) = match &function.kind {
            ExprKind::Attribute { value, attr } => (value, *attr),
            _ => {
                // Shouldn't happen if type inference is correct
                out_vars.insert(Variable::Top);
                return Ok(out_vars);
            }
        };

        // Evaluate the receiver to get ClassInstance variables
        let receiver_vars = self.eval_expr(domain, receiver_expr, global)?;

        // Compute products for method calls
        let method_args_products: Vec<Vec<&Variable>> =
            args_sets.iter().multi_cartesian_product().collect();
        let method_kw_names = kwargs_sets.iter().map(|(n, _)| *n);
        let method_kwargs_products: Vec<HashMap<Identifier, _>> = kwargs_sets
            .iter()
            .map(|(_, vars)| vars)
            .multi_cartesian_product()
            .map(|vars| method_kw_names.clone().zip(vars).collect())
            .collect();

        // For each receiver ClassInstance and each combination of args/kwargs
        for receiver_var in &receiver_vars {
            if let Variable::ClassInstance(instance) = receiver_var {
                // Get the class analysis from the instance's class_id
                let Some(class_analysis) = global.classes.get(&instance.class_id) else {
                    log::debug!(
                        "couldn't find class analysis for {}, not in {:?}",
                        resolve(instance.class_id),
                        global
                            .classes
                            .keys()
                            .map(|k| resolve(*k))
                            .collect::<Vec<_>>()
                    );
                    out_vars.insert(Variable::Top);
                    continue;
                };

                // Get the substituted signature for this method
                let signature_model = class_analysis.get_method_signature(method_name, instance)?;

                // For each combination of args/kwargs, use the signature model
                for (args_refs, kwargs_ref) in method_args_products
                    .iter()
                    .cartesian_product(method_kwargs_products.iter())
                {
                    // Convert references to &Variable for the model
                    let args: Vec<&Variable> = args_refs.iter().map(|v| *v).collect();
                    let kwargs: HashMap<Identifier, &Variable> =
                        kwargs_ref.iter().map(|(k, v)| (*k, *v)).collect();

                    let any_top = args.iter().any(|v| matches!(**v, Variable::Top))
                        || kwargs.iter().any(|(_, v)| matches!(**v, Variable::Top));
                    if any_top {
                        out_vars.insert(Variable::Top);
                    } else {
                        // Use the signature model to infer the result
                        let result_shape = signature_model.infer(args, kwargs, span, arg_spans)?;
                        out_vars.insert(Variable::Tensor(result_shape));
                    }
                }
            } else {
                // Receiver is not a ClassInstance
                out_vars.insert(Variable::Top);
            }
        }

        Ok(out_vars)
    }

    fn eval_function(
        &mut self,
        function: &Expr,
        args_sets: &[HashSet<Variable>],
        kwargs_sets: &[(Identifier, HashSet<Variable>)],
        span: SourceSpan,
        arg_spans: &ArgSpans,
        global: &GlobalAnalysis,
    ) -> Result<HashSet<Variable>> {
        let mut out_vars = HashSet::new();

        // Resolve function name - for now assume it's an ident or attribute chain
        let func_name = self.expr_to_dot_string(function, global);

        let args_products = args_sets.iter().multi_cartesian_product();
        let kw_names = kwargs_sets.iter().map(|(n, _)| *n);
        let kwargs_products: Vec<HashMap<Identifier, _>> = kwargs_sets
            .iter()
            .map(|(_, vars)| vars)
            .multi_cartesian_product()
            .map(|vars| kw_names.clone().zip(vars).collect())
            .collect();

        match global.models.resolve(&func_name) {
            Some(model) => {
                for (args, kwargs) in args_products.cartesian_product(kwargs_products) {
                    let any_top = args.iter().any(|v| matches!(v, Variable::Top))
                        || kwargs.iter().any(|(_, v)| matches!(v, Variable::Top));
                    if any_top {
                        out_vars.insert(Variable::Top);
                    } else {
                        out_vars.insert(Variable::Tensor(
                            model.infer(args, kwargs, span, arg_spans)?,
                        ));
                    }
                }
            }
            None => {
                log::debug!("couldn't resolve function {} to model", func_name);
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
        global: &GlobalAnalysis,
    ) -> Result<Vec<HashSet<Variable>>> {
        args.iter()
            .map(|arg_expr| self.eval_expr(domain, arg_expr, global))
            .collect()
    }

    /// Calls eval_expr on each kwarg
    fn eval_call_kwargs(
        &mut self,
        domain: &AnalysisDomain,
        kwargs: &[(Identifier, Expr)],
        global: &GlobalAnalysis,
    ) -> Result<Vec<(Identifier, HashSet<Variable>)>> {
        kwargs
            .iter()
            .map(|(n, arg_expr)| Ok((*n, self.eval_expr(domain, arg_expr, global)?)))
            .collect()
    }

    /// Convert an Expr to a dot-separated string (e.g., "torch.nn.functional.relu")
    /// Uses SymbolTable to resolve identifiers to canonical paths when available
    fn expr_to_dot_string(&self, expr: &Expr, global: &GlobalAnalysis) -> String {
        match &expr.kind {
            ExprKind::Ident(name) => {
                let local_name = resolve(*name);
                if let Some(canonical) = global
                    .symbol_table
                    .resolve(&self.function.file_path, &local_name)
                {
                    return canonical.clone();
                }
                local_name
            }
            ExprKind::Attribute { value, attr } => {
                format!(
                    "{}.{}",
                    self.expr_to_dot_string(value, global),
                    resolve(*attr)
                )
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
        global: &GlobalAnalysis,
    ) -> Result<HashSet<Variable>> {
        let vars = self.eval_expr(domain, receiver, global)?;
        let indices = self.eval_indices(domain, index, global)?;

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
                Variable::Collection(col) => {
                    if let Some(result) = col.read_at_index(&index) {
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
                Variable::ClassInstance(_) => {}
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
        global: &GlobalAnalysis,
    ) -> Result<Vec<HashSet<Either<Variable, DimSlice>>>> {
        index
            .iter()
            .map(|v| -> Result<HashSet<Either<Variable, DimSlice>>> {
                Ok(match v {
                    Either::Left(l) => self
                        .eval_expr(domain, l, global)?
                        .into_iter()
                        .map(Either::Left)
                        .collect::<HashSet<_>>(),
                    Either::Right(Slice { lower, upper }) => {
                        let lowers = match lower {
                            Some(expr_lower) => Some(self.eval_expr(domain, expr_lower, global)?),
                            None => None,
                        };
                        let uppers = match upper {
                            Some(expr_upper) => Some(self.eval_expr(domain, expr_upper, global)?),
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
                Either::Left(v) => {
                    match v {
                        Variable::DimVar(_) => {
                            continue; // any index removes this dimension from the resulting shape
                        }
                        _ => {
                            // index doesn't make sense
                            return Ok(None);
                        }
                    }
                }
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

        // Add all remaining dimensions that weren't indexed
        for remaining_dim in &dims[index_or_slice.len()..] {
            out_dims.push(remaining_dim.clone());
        }

        Ok(Some(Variable::Tensor(Shape(out_dims))))
    }

    /// Eval expr, updating target if this statement is an assignment.
    fn handle_stmt(
        &mut self,
        loc: Location,
        domain: &mut AnalysisDomain,
        stmt: &Statement,
        global: &GlobalAnalysis,
    ) -> Result<()> {
        let res_var = self.eval_expr(domain, &stmt.value, global)?;

        if let Some(target) = &stmt.target {
            // Store the assignment result for inlay hints
            self.assignments.insert(loc, res_var.clone());

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
    fn handle_term(
        &mut self,
        domain: &mut AnalysisDomain,
        term: &Terminator,
        global: &GlobalAnalysis,
    ) -> Result<()> {
        match term {
            Terminator::CondJump { cond, .. } => {
                if let Some(expr) = cond {
                    self.eval_expr(domain, expr, global)?;
                }
            }
            Terminator::Return(expr) => {
                if let Some(expr) = expr {
                    self.eval_expr(domain, expr, global)?;
                }
            }
            Terminator::Jump(_) => (),
        }

        Ok(())
    }

    /// Run single-pass dataflow analysis on function
    pub(crate) fn analyze_func(&mut self, func: &Function, global: &GlobalAnalysis) -> Result<()> {
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
                Either::Left(stmt) => self.handle_stmt(*loc, &mut domain, stmt, global),
                Either::Right(term) => self.handle_term(&mut domain, term, global),
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

    /// Generate inlay hints for this function's analyzed state
    pub fn inlay_hints(&self) -> Vec<tower_lsp::lsp_types::InlayHint> {
        use tower_lsp::lsp_types::InlayHint;

        let mut hints = Vec::new();

        // Generate inlay hint for each assignment
        for loc in &self.function.locations {
            if let Either::Left(stmt) = self.function.instr(loc) {
                if stmt.target.is_some()
                    && let Some(position) = stmt.assign_end
                {
                    if let Some(vars) = self.assignments.get(loc) {
                        if let Some(label) = super::vars_to_inlay(vars) {
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
