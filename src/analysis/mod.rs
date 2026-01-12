mod dimvars;
mod errors;
mod models;
mod print;
mod types;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use itertools::{Either, Itertools};
use miette::SourceSpan;
use tower_lsp::lsp_types::InlayHint;
pub use types::{Shape, Variable};

pub use crate::analysis::dimvars::{DimKind, DimVar};
use crate::analysis::models::{Model, ModelContext, SignatureModel};
use crate::analysis::types::{ClassInstance, Collection, CollectionKey, DimSlice};
use crate::ir::types::{Binop, Constant, ExprKind, Location, Slice, Type};
use crate::ir::{Class, File, Function, Project};
use crate::ir::{Expr, Identifier, Parameter, Statement, Terminator, intern, resolve};
use crate::parse::SymbolTable;
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
    pub symbol_table: Arc<SymbolTable>,
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
    pub fn new(symbol_table: &SymbolTable, functions: &Vec<Function>) -> Self {
        let symbol_table = Arc::new(symbol_table.clone());
        Self {
            functions: HashMap::new(),
            classes: HashMap::new(),
            models: Arc::new(ModelContext::new(functions, &symbol_table)),
            symbol_table,
        }
    }

    pub fn analyze_func(&mut self, func: &Function) -> Result<()> {
        let local_name = resolve(func.identifier);
        let canonical = self
            .symbol_table
            .resolve(&func.file_path, &local_name)
            .cloned()
            .unwrap_or(local_name);
        let canonical_id = intern(&canonical);

        let mut func_analysis = FunctionAnalysis::new(func, None);
        func_analysis.analyze_func(func, self)?;
        self.functions.insert(canonical_id, func_analysis);
        Ok(())
    }

    pub fn analyze_class(&mut self, class: &Class) -> Result<()> {
        let local_name = resolve(class.identifier);
        let canonical = self
            .symbol_table
            .resolve(&class.file_path, &local_name)
            .cloned()
            .unwrap_or(local_name);
        let canonical_id = intern(&canonical);

        let class_analysis = analyze_class(class, canonical_id, self)?;
        self.classes.insert(canonical_id, class_analysis);
        Ok(())
    }

    /// Format the entire analysis result as a string, including both classes and functions.
    pub fn format_all(&self, project: &Project) -> String {
        let mut output = String::new();

        println!("format_all");

        // Collect all classes and functions from all files
        let all_classes: Vec<&Class> = project.files.iter().flat_map(|f| &f.classes).collect();
        let all_functions: Vec<&Function> =
            project.files.iter().flat_map(|f| &f.functions).collect();

        // Print classes first
        for (canonical_id, facts) in self
            .classes
            .iter()
            .sorted_by(|(a, _), (b, _)| resolve(**a).cmp(&resolve(**b)))
        {
            let canonical_path = resolve(*canonical_id);
            if let Some(class) = all_classes.iter().find(|c| {
                let local = resolve(c.identifier);
                canonical_path.ends_with(&local) || canonical_path == local
            }) {
                output.push_str(&class_with_inferred_shapes_to_string(class, facts, None));
                output.push_str("\n\n");
            }
        }

        // Then print functions
        for (canonical_id, facts) in self
            .functions
            .iter()
            .sorted_by(|(a, _), (b, _)| resolve(**a).cmp(&resolve(**b)))
        {
            let canonical_path = resolve(*canonical_id);
            if let Some(func) = all_functions.iter().find(|f| {
                let local = resolve(f.identifier);
                // Resolve the function's canonical path using the symbol table
                let func_canonical = self
                    .symbol_table
                    .resolve(&f.file_path, &local)
                    .cloned()
                    .unwrap_or(local);
                func_canonical == canonical_path
            }) {
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
    pub function: Function,
}

impl FunctionAnalysis {
    /// Create a new FunctionAnalysis.
    ///
    /// If `class_attributes` is `Some`, this is a method analysis and the state will include
    /// both the class attributes (self.X) and function parameters.
    /// If `class_attributes` is `None`, this is a top-level function and only params will be used.
    fn new(func: &Function, class_attributes: Option<AnalysisDomain>) -> Self {
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
                    if is_matmul {
                        let out_shape = global.models.torch.matmul.infer(
                            vec![l_var, r_var],
                            HashMap::new(),
                            span,
                        )?;
                        Variable::Tensor(out_shape)
                    } else {
                        let out_shape = global.models.torch.broadcast.infer(
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

        Ok(HashSet::from_iter(products.map(Variable::Tuple)))
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

        Ok(HashSet::from_iter(products.map(|vars| {
            let elements: BTreeMap<CollectionKey, Variable> = vars
                .into_iter()
                .enumerate()
                .map(|(i, v)| (CollectionKey::Int(i as i64), v))
                .collect();
            Variable::Collection(Collection { elements })
        })))
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

        match &call_ty {
            Type::Constructor(class_id) => {
                self.eval_constructor(*class_id, &args_sets, &kwargs_sets, global)
            }
            Type::Method(class_id) => self.eval_method(
                *class_id,
                function,
                domain,
                &args_sets,
                &kwargs_sets,
                span,
                global,
            ),
            _ => self.eval_function(function, &args_sets, &kwargs_sets, span, global, domain),
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
        class_id: Identifier,
        function: &Expr,
        domain: &AnalysisDomain,
        args_sets: &[HashSet<Variable>],
        kwargs_sets: &[(Identifier, HashSet<Variable>)],
        span: SourceSpan,
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

        // Get the class analysis
        if let Some(class_analysis) = global.classes.get(&class_id) {
            // For each receiver ClassInstance and each combination of args/kwargs
            for receiver_var in &receiver_vars {
                if let Variable::ClassInstance(instance) = receiver_var {
                    // Get the substituted signature for this method
                    let signature_model =
                        class_analysis.get_method_signature(method_name, instance)?;

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
                            let result_shape = signature_model.infer(args, kwargs, span)?;
                            out_vars.insert(Variable::Tensor(result_shape));
                        }
                    }
                } else {
                    // Receiver is not a ClassInstance
                    out_vars.insert(Variable::Top);
                }
            }
        } else {
            // Class not found in analysis
            out_vars.insert(Variable::Top);
        }

        Ok(out_vars)
    }

    fn eval_function(
        &mut self,
        function: &Expr,
        args_sets: &[HashSet<Variable>],
        kwargs_sets: &[(Identifier, HashSet<Variable>)],
        span: SourceSpan,
        global: &GlobalAnalysis,
        domain: &AnalysisDomain,
    ) -> Result<HashSet<Variable>> {
        let mut out_vars = HashSet::new();

        if let ExprKind::Attribute { value, attr } = &function.kind {
            let receiver_vars = self.eval_expr(domain, value, global)?;

            let class_instances: Vec<_> = receiver_vars
                .iter()
                .filter_map(|v| {
                    if let Variable::ClassInstance(inst) = v {
                        Some(inst)
                    } else {
                        None
                    }
                })
                .collect();

            if !class_instances.is_empty() {
                let method_name = *attr;
                let method_args_products: Vec<Vec<&Variable>> =
                    args_sets.iter().multi_cartesian_product().collect();
                let method_kw_names = kwargs_sets.iter().map(|(n, _)| *n);
                let method_kwargs_products: Vec<HashMap<Identifier, _>> = kwargs_sets
                    .iter()
                    .map(|(_, vars)| vars)
                    .multi_cartesian_product()
                    .map(|vars| method_kw_names.clone().zip(vars).collect())
                    .collect();

                for instance in class_instances {
                    if let Some(class_analysis) = global.classes.get(&instance.class_id) {
                        if let Ok(signature_model) =
                            class_analysis.get_method_signature(method_name, instance)
                        {
                            for (args_refs, kwargs_ref) in method_args_products
                                .iter()
                                .cartesian_product(method_kwargs_products.iter())
                            {
                                let args: Vec<&Variable> = args_refs.iter().map(|v| *v).collect();
                                let kwargs: HashMap<Identifier, &Variable> =
                                    kwargs_ref.iter().map(|(k, v)| (*k, *v)).collect();

                                let any_top = args.iter().any(|v| matches!(**v, Variable::Top))
                                    || kwargs.iter().any(|(_, v)| matches!(**v, Variable::Top));
                                if any_top {
                                    out_vars.insert(Variable::Top);
                                } else {
                                    let result_shape = signature_model.infer(args, kwargs, span)?;
                                    out_vars.insert(Variable::Tensor(result_shape));
                                }
                            }
                        } else {
                            out_vars.insert(Variable::Top);
                        }
                    } else {
                        out_vars.insert(Variable::Top);
                    }
                }

                if receiver_vars
                    .iter()
                    .any(|v| !matches!(v, Variable::ClassInstance(_)))
                {
                    out_vars.insert(Variable::Top);
                }

                return Ok(out_vars);
            }
        }

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
                        out_vars.insert(Variable::Tensor(model.infer(args, kwargs, span)?));
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
                Variable::ClassInstance(_) => {}
                Variable::Collection(col) => {
                    if let Some(result) = self.eval_collection_index(col, &index) {
                        set.insert(result);
                    } else {
                        set.insert(Variable::Top);
                    }
                }
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

    fn eval_collection_index(
        &self,
        col: &Collection,
        index_or_slices: &[&Either<Variable, DimSlice>],
    ) -> Option<Variable> {
        assert!(index_or_slices.len() == 1);
        match index_or_slices.first().unwrap() {
            Either::Left(var) => var
                .as_concrete_dimvar()
                .and_then(|i| col.elements.get(&i.into()).cloned())
                .or(Some(Variable::Top)),
            Either::Right(_) => {
                // TODO: deal with potential slicing
                Some(Variable::Top)
            }
        }
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
    fn handle_stmt(
        &mut self,
        domain: &mut AnalysisDomain,
        stmt: &Statement,
        global: &GlobalAnalysis,
    ) -> Result<()> {
        let res_var = self.eval_expr(domain, &stmt.value, global)?;

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
    fn analyze_func(&mut self, func: &Function, global: &GlobalAnalysis) -> Result<()> {
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
                Either::Left(stmt) => self.handle_stmt(&mut domain, stmt, global),
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
}

#[derive(Debug)]
pub struct ClassAnalysis {
    /// The identifier of the class being analyzed
    pub id: Identifier,
    /// Mapping from attribute names to their inferred Variables.
    /// The Variables' dimvars are in terms of the annotated dimvars of the __init__ method.
    pub attributes: HashMap<Identifier, HashSet<Variable>>,
    /// Analysis results for each method (including __init__).
    /// These are used for consistency checking.
    pub methods: HashMap<Identifier, FunctionAnalysis>,
}

/// Analyze a class to infer tensor shapes for its attributes and methods.
///
/// This function:
/// 1. Analyzes the `__init__` method as a special case to extract attribute shapes
/// 2. Analyzes other methods for consistency checking
pub fn analyze_class(
    class: &Class,
    canonical_id: Identifier,
    global: &GlobalAnalysis,
) -> Result<ClassAnalysis> {
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
    let mut init_analysis = FunctionAnalysis::new(init_method, None);
    init_analysis.analyze_func(init_method, global)?;

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
    methods.insert(init_method_name, init_analysis);

    for (method_name, method_func) in &class.methods {
        if *method_name == init_method_name {
            continue;
        }

        let mut method_analysis =
            FunctionAnalysis::new(method_func, Some(method_class_attributes.clone()));
        method_analysis.analyze_func(method_func, global)?;
        methods.insert(*method_name, method_analysis);
    }

    Ok(ClassAnalysis {
        id: canonical_id,
        attributes,
        methods,
    })
}

impl ClassAnalysis {
    /// Create a class instance with concrete dimension variable substitutions.
    /// Takes resolved arguments and keyword arguments and computes the DimVar substitutions from
    /// parameter annotations to concrete values.
    pub fn create_instance(
        &self,
        args: &[Variable],
        kwargs: &HashMap<Identifier, Variable>,
    ) -> Result<Variable> {
        let init_method_name = intern("__init__");
        let init_analysis = self.methods.get(&init_method_name).ok_or_else(|| {
            anyhow::anyhow!(
                "Class {} missing __init__ method in methods",
                resolve(self.id)
            )
        })?;

        let mut substitutions = BTreeMap::new();

        // Match up all (param, arg) pairs
        let mut param_arg_pairs = Vec::new();
        let mut matched_param_indices = std::collections::HashSet::new();

        // Match positional arguments to parameters (skip instance parameter)
        for (i, Parameter(param_name, param_annotation)) in
            init_analysis.function.params.iter().skip(1).enumerate()
        {
            let arg_value = if i < args.len() {
                Some(&args[i])
            } else if let Some(kwarg_value) = kwargs.get(param_name) {
                Some(kwarg_value)
            } else {
                None // Parameter not provided
            };

            if let Some(arg) = arg_value {
                if let Some(param_var) = param_annotation {
                    param_arg_pairs.push((param_var.clone(), arg.clone()));
                    matched_param_indices.insert(i + 1); // we skipped instance param
                }
            }
        }

        // Also check kwargs that weren't matched positionally
        for (kwarg_name, kwarg_value) in kwargs {
            // Check if this kwarg matches a parameter name (excluding the first parameter)
            if let Some((param_idx, param)) = init_analysis
                .function
                .params
                .iter()
                .enumerate()
                .find(|(idx, p)| idx > &0 && &p.0 == kwarg_name)
            {
                // Skip if this parameter was already matched
                if matched_param_indices.contains(&param_idx) {
                    continue;
                }

                let Parameter(_, param_annotation) = param;
                if let Some(param_var) = param_annotation {
                    param_arg_pairs.push((param_var.clone(), kwarg_value.clone()));
                    matched_param_indices.insert(param_idx);
                }
            }
        }

        // Process each (param, arg) pair to extract substitutions
        for (param_var, arg_var) in param_arg_pairs {
            if let Some(new_substitutions) =
                self.extract_substitutions_from_pair(&param_var, &arg_var)?
            {
                substitutions.extend(new_substitutions);
            }
        }

        Ok(Variable::ClassInstance(ClassInstance {
            class_id: self.id,
            substitutions,
        }))
    }

    /// Extract dimension variable substitutions from a (param, arg) pair.
    /// Returns a map of substitutions if any can be extracted, None otherwise.
    fn extract_substitutions_from_pair(
        &self,
        param_var: &Variable,
        arg_var: &Variable,
    ) -> Result<Option<BTreeMap<DimVar, DimVar>>> {
        // If types don't match, we can't extract substitutions
        if !std::mem::discriminant(param_var).eq(&std::mem::discriminant(arg_var)) {
            return Ok(None);
        }

        let mut substitutions = BTreeMap::new();

        match (param_var, arg_var) {
            (Variable::DimVar(param_dimvar), Variable::DimVar(arg_dimvar)) => {
                // TODO: handle dim exprs
                substitutions.insert(param_dimvar.clone(), arg_dimvar.clone());
            }

            // Tensor -> recur for each dimvar
            (Variable::Tensor(param_shape), Variable::Tensor(arg_shape)) => {
                if param_shape.0.len() == arg_shape.0.len() {
                    for (param_dim, arg_dim) in param_shape.0.iter().zip(arg_shape.0.iter()) {
                        if let Some(recursive_subs) = self.extract_substitutions_from_pair(
                            &Variable::DimVar(param_dim.clone()),
                            &Variable::DimVar(arg_dim.clone()),
                        )? {
                            substitutions.extend(recursive_subs);
                        }
                    }
                }
            }

            // Tuple -> recur for each element
            (Variable::Tuple(param_vars), Variable::Tuple(arg_vars)) => {
                if param_vars.len() == arg_vars.len() {
                    for (param_elem, arg_elem) in param_vars.iter().zip(arg_vars.iter()) {
                        if let Some(recursive_subs) =
                            self.extract_substitutions_from_pair(param_elem, arg_elem)?
                        {
                            substitutions.extend(recursive_subs);
                        }
                    }
                }
            }
            _ => {
                // Other matching types (Top, None, ClassInstance) don't have dimvars to extract
            }
        }

        if substitutions.is_empty() {
            Ok(None)
        } else {
            Ok(Some(substitutions))
        }
    }

    /// Get a SignatureModel for a method with substitutions applied from a ClassInstance.
    /// This substitutes the method's parameter annotations and return type using the instance's substitutions.
    pub fn get_method_signature(
        &self,
        method_name: Identifier,
        instance: &ClassInstance,
    ) -> Result<SignatureModel> {
        let method_analysis = self.methods.get(&method_name).ok_or_else(|| {
            anyhow::anyhow!(
                "Method {} not found in class {}",
                resolve(method_name),
                resolve(self.id)
            )
        })?;
        let method = &method_analysis.function;

        // Apply substitutions to parameters and return type
        let substituted_params: Vec<Parameter> = method
            .params
            .iter()
            .skip(1) // skip instance param
            .map(|Parameter(name, var_opt)| {
                let substituted_var = var_opt
                    .as_ref()
                    .map(|v| Self::substitute_variable(v, &instance.substitutions));
                Parameter(*name, substituted_var)
            })
            .collect();

        let substituted_returns = method_analysis.function.returns.as_ref().map(|returns| {
            returns
                .iter()
                .map(|v| Self::substitute_variable(v, &instance.substitutions))
                .collect()
        });

        Ok(SignatureModel {
            params: substituted_params,
            returns: substituted_returns,
        })
    }

    /// Apply DimVar substitutions to a Variable, recursively substituting DimVars in shapes and tuples.
    fn substitute_variable(var: &Variable, substitutions: &BTreeMap<DimVar, DimVar>) -> Variable {
        match var {
            Variable::DimVar(dv) => {
                let mut string_map = HashMap::new();
                for (param_dv, concrete_dv) in substitutions {
                    if let DimKind::Named(name) = param_dv.kind() {
                        string_map.insert(name, concrete_dv.clone());
                    }
                }
                if let Ok(substituted) = dv.substitute(&string_map) {
                    Variable::DimVar(substituted)
                } else {
                    var.clone()
                }
            }
            Variable::Tensor(shape) => {
                let substituted_dims: Vec<DimVar> = shape
                    .0
                    .iter()
                    .map(|dv| {
                        let mut string_map = HashMap::new();
                        for (param_dv, concrete_dv) in substitutions {
                            if let DimKind::Named(name) = param_dv.kind() {
                                string_map.insert(name, concrete_dv.clone());
                            }
                        }
                        dv.substitute(&string_map).unwrap_or_else(|_| dv.clone())
                    })
                    .collect();
                Variable::Tensor(Shape(substituted_dims))
            }
            Variable::Tuple(vars) => {
                // Recursively substitute in tuple elements
                let substituted_vars: Vec<Variable> = vars
                    .iter()
                    .map(|v| Self::substitute_variable(v, substitutions))
                    .collect();
                Variable::Tuple(substituted_vars)
            }
            Variable::ClassInstance(inst) => {
                let mut string_map = HashMap::new();
                for (param_dv, concrete_dv) in substitutions {
                    if let DimKind::Named(name) = param_dv.kind() {
                        string_map.insert(name, concrete_dv.clone());
                    }
                }

                let new_subs: BTreeMap<DimVar, DimVar> = inst
                    .substitutions
                    .iter()
                    .map(|(param_dv, concrete_dv)| {
                        let substituted = concrete_dv
                            .substitute(&string_map)
                            .unwrap_or_else(|_| concrete_dv.clone());
                        (param_dv.clone(), substituted)
                    })
                    .collect();

                Variable::ClassInstance(ClassInstance {
                    class_id: inst.class_id,
                    substitutions: new_subs,
                })
            }
            Variable::Collection(col) => {
                let substituted_elements: BTreeMap<CollectionKey, Variable> = col
                    .elements
                    .iter()
                    .map(|(k, v)| (k.clone(), Self::substitute_variable(v, substitutions)))
                    .collect();
                Variable::Collection(Collection {
                    elements: substituted_elements,
                })
            }
            Variable::Top | Variable::None => var.clone(),
        }
    }
}

pub fn analyze(project: Project, symbol_table: &SymbolTable) -> Result<GlobalAnalysis> {
    // Collect all functions/classes from all files (clone to get owned values)
    let all_functions: Vec<Function> = project
        .files
        .iter()
        .flat_map(|f| f.functions.clone())
        .collect();
    let all_classes: Vec<&Class> = project.files.iter().flat_map(|f| &f.classes).collect();

    // Create GlobalAnalysis with all functions so ModelContext can build signature models
    let mut global_analysis = GlobalAnalysis::new(symbol_table, &all_functions);

    // Analyze classes
    for class in all_classes {
        global_analysis.analyze_class(class)?;
    }

    // Analyze functions
    for func in &all_functions {
        global_analysis.analyze_func(func)?;
    }

    Ok(global_analysis)
}
