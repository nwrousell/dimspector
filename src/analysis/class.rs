use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;

use crate::analysis::dimvars::{DimKind, DimVar};
use crate::analysis::models::SignatureModel;
use crate::analysis::types::{ClassInstance, Collection, Shape, Variable};
use crate::ir::{Class, Identifier, Parameter, intern, resolve};
use anyhow::Result;

use super::{AnalysisDomain, FunctionAnalysis, GlobalAnalysis};

#[derive(Debug)]
pub struct ClassAnalysis {
    /// Canonical id (includes module path). This should be used for lookups, etc.
    pub canonical_id: Identifier,
    /// The identifier of the class being analyzed
    pub id: Identifier,
    /// The file path where this class is defined
    pub file_path: PathBuf,
    /// Mapping from attribute names to their inferred Variables.
    /// The Variables' dimvars are in terms of the annotated dimvars of the __init__ method.
    pub attributes: HashMap<Identifier, HashSet<Variable>>,
    /// Analysis results for each method (including __init__).
    /// These are used for consistency checking.
    pub methods: HashMap<Identifier, FunctionAnalysis>,
    /// Whether this class inherits from torch.nn.Module
    pub is_nn_module: bool,
}

/// Analyze a class to infer tensor shapes for its attributes and methods.
///
/// This function:
/// 1. Analyzes the `__init__` method as a special case to extract attribute shapes
/// 2. Analyzes other methods for consistency checking
///
/// Returns the ClassAnalysis and a vector of errors from method analysis
pub fn analyze_class(
    class: &Class,
    global: &GlobalAnalysis,
) -> Result<(ClassAnalysis, Vec<anyhow::Error>)> {
    let init_method_name = intern("__init__");

    // Compute canonical id from symbol table (e.g., "module.ClassName")
    let local_name = resolve(class.identifier);
    let canonical_name = global
        .symbol_table
        .resolve(&class.file_path, &local_name)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Class {} not found in symbol table for file {}",
                local_name,
                class.file_path.display()
            )
        })?;
    let canonical_id = intern(canonical_name);

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
    let mut failed_methods = Vec::new();

    for (method_name, method_func) in &class.methods {
        if *method_name == init_method_name {
            continue;
        }

        let mut method_analysis =
            FunctionAnalysis::new(method_func, Some(method_class_attributes.clone()));

        // Continue even if method analysis fails - collect the error
        if let Err(e) = method_analysis.analyze_func(method_func, global) {
            failed_methods.push(e);
        }
        methods.insert(*method_name, method_analysis);
    }

    let class_analysis = ClassAnalysis {
        canonical_id,
        id: class.identifier,
        file_path: class.file_path.clone(),
        attributes,
        methods,
        is_nn_module: class.is_nn_module,
    };

    Ok((class_analysis, failed_methods))
}

impl ClassAnalysis {
    /// Generate inlay hints for all methods in this class
    pub fn inlay_hints(&self) -> Vec<tower_lsp::lsp_types::InlayHint> {
        self.methods
            .values()
            .flat_map(|method_analysis| method_analysis.inlay_hints())
            .collect()
    }

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
        for (i, Parameter(param_name, param_annotation)) in init_analysis
            .function
            .param_types
            .iter()
            .skip(1)
            .enumerate()
        {
            let arg_value = if i < args.len() {
                Some(&args[i])
            } else if let Some(kwarg_value) = kwargs.get(&param_name) {
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
                .param_types
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
            class_id: self.canonical_id,
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

            // Collection -> recur for each element/value
            (Variable::Collection(param_col), Variable::Collection(arg_col)) => {
                match (param_col, arg_col) {
                    (Collection::Tuple(param_vars), Collection::Tuple(arg_vars))
                    | (Collection::List(param_vars), Collection::List(arg_vars)) => {
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
                    (Collection::Dict(param_map), Collection::Dict(arg_map)) => {
                        // Match keys and recursively extract from values
                        for (key, param_val) in param_map {
                            if let Some(arg_val) = arg_map.get(key) {
                                if let Some(recursive_subs) =
                                    self.extract_substitutions_from_pair(param_val, arg_val)?
                                {
                                    substitutions.extend(recursive_subs);
                                }
                            }
                        }
                    }
                    _ => {
                        // Mismatched collection types
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
        span: miette::SourceSpan,
    ) -> Result<SignatureModel> {
        let method_analysis = self.methods.get(&method_name).ok_or_else(|| {
            anyhow::anyhow!(
                "Method {} not found in class {}",
                resolve(method_name),
                resolve(self.id)
            )
        })?;
        let method = &method_analysis.function;

        let method_name = resolve(method_name);

        // Apply substitutions to parameters and return type
        let substituted_params: Vec<Parameter> = method
            .param_types
            .iter()
            .skip(1) // skip instance param
            .map(|Parameter(name, var_opt)| {
                let substituted_var = var_opt.as_ref().map(|v| {
                    Self::substitute_variable(v, &instance.substitutions, &method_name, span)
                });
                Parameter(*name, substituted_var)
            })
            .collect();

        let substituted_return = method_analysis
            .function
            .return_type
            .as_ref()
            .map(|(ret_var, _span)| {
                Self::substitute_variable(ret_var, &instance.substitutions, &method_name, span)
            });

        Ok(SignatureModel {
            name: method_name,
            param_types: substituted_params,
            return_type: substituted_return,
        })
    }

    /// Apply DimVar substitutions to a Variable, recursively substituting DimVars in shapes and tuples.
    pub fn substitute_variable(
        var: &Variable,
        substitutions: &BTreeMap<DimVar, DimVar>,
        func_name: &str,
        span: miette::SourceSpan,
    ) -> Variable {
        match var {
            Variable::DimVar(dv) => {
                // Substitute the DimVar using the map
                // Convert BTreeMap<DimVar, DimVar> to HashMap<String, DimVar> for DimVar::substitute
                let mut string_map = HashMap::new();
                for (param_dv, concrete_dv) in substitutions {
                    if let DimKind::Named(name) = param_dv.kind() {
                        string_map.insert(name, concrete_dv.clone());
                    }
                }
                if let Ok(substituted) = dv.substitute(&string_map, func_name, span) {
                    Variable::DimVar(substituted)
                } else {
                    var.clone()
                }
            }
            Variable::Tensor(shape) => {
                // Substitute DimVars in the shape
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
                        dv.substitute(&string_map, func_name, span)
                            .unwrap_or_else(|_| dv.clone())
                    })
                    .collect();
                Variable::Tensor(Shape(substituted_dims))
            }
            Variable::Collection(col) => {
                let sub_col = match col {
                    Collection::Tuple(vars) => {
                        let substituted_vars: Vec<Variable> = vars
                            .iter()
                            .map(|v| Self::substitute_variable(v, substitutions, func_name, span))
                            .collect();
                        Collection::Tuple(substituted_vars)
                    }
                    Collection::List(vars) => {
                        let substituted_vars: Vec<Variable> = vars
                            .iter()
                            .map(|v| Self::substitute_variable(v, substitutions, func_name, span))
                            .collect();
                        Collection::List(substituted_vars)
                    }
                    Collection::Dict(map) => {
                        let substituted_map: BTreeMap<_, _> = map
                            .iter()
                            .map(|(k, v)| {
                                (
                                    k.clone(),
                                    Self::substitute_variable(v, substitutions, func_name, span),
                                )
                            })
                            .collect();
                        Collection::Dict(substituted_map)
                    }
                };
                Variable::Collection(sub_col)
            }
            Variable::ClassInstance(_) => {
                // TODO: Handle nested class instances
                var.clone()
            }
            Variable::Top | Variable::None => var.clone(),
        }
    }
}
