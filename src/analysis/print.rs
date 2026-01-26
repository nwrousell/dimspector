use std::{fmt, fmt::Write};

use itertools::Itertools;

use crate::{
    analysis::{
        AnalysisDomain, ClassAnalysis, FunctionAnalysis, GlobalAnalysis,
        dimvars::{CanonicalDimVar, DimVar, NamedPow, Term},
        types::{Collection, DictKey},
    },
    ir::{
        Class, Function, resolve,
        types::{ExprKind, Location},
    },
    utils::{indent, write_comma_separated},
};

use super::types::{Shape, Variable};

impl fmt::Display for DimVar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.canonical())
    }
}

impl fmt::Display for Variable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Variable::Top => write!(f, "⟙"),
            Variable::DimVar(dim_var) => write!(f, "{}", dim_var),
            Variable::Tensor(shape) => write!(f, "{}", shape),
            Variable::Collection(collection) => write!(f, "{}", collection),
            Variable::None => write!(f, "None"),
            Variable::ClassInstance(instance) => {
                use crate::ir::types::resolve;
                write!(f, "{}", resolve(instance.class_id))?;
                if !instance.substitutions.is_empty() {
                    write!(f, "{{")?;
                    let mut first = true;
                    for (param_dv, concrete_dv) in &instance.substitutions {
                        if !first {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}->{}", param_dv, concrete_dv)?;
                        first = false;
                    }
                    write!(f, "}}")?;
                }
                Ok(())
            }
        }
    }
}

impl fmt::Display for Collection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Collection::Tuple(vars) => {
                write!(f, "(")?;
                write_comma_separated(f, vars)?;
                write!(f, ")")
            }
            Collection::List(vars) => {
                write!(f, "[")?;
                write_comma_separated(f, vars)?;
                write!(f, "]")
            }
            Collection::Dict(map) => {
                write!(f, "{{")?;
                let mut first = true;
                for (key, value) in map {
                    if !first {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {}", key, value)?;
                    first = false;
                }
                write!(f, "}}")
            }
        }
    }
}

impl fmt::Display for DictKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DictKey::Int(i) => write!(f, "{}", i),
            DictKey::Str(s) => write!(f, "{}", s),
        }
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        write_comma_separated(f, &self.0)?;
        write!(f, "]")
    }
}

impl fmt::Display for FunctionAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Function {}\n", resolve(self.id))?;
        for (loc, domain) in self
            .state
            .iter()
            .sorted_by(|(l_a, _), (l_b, _)| Ord::cmp(*l_a, *l_b))
        {
            write!(f, "  {}\n", loc)?;
            for (path, vars) in domain.iter() {
                write!(f, "    {} => {{", resolve(*path))?;
                write_comma_separated(f, vars)?;
                write!(f, "}}\n")?
            }
        }
        Ok(())
    }
}

impl fmt::Display for GlobalAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (_, func) in self.functions.iter() {
            write!(f, "{}\n\n", func)?;
        }
        Ok(())
    }
}

impl fmt::Display for CanonicalDimVar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Sort: variable terms first, then constant terms
        let (var_terms, const_terms): (Vec<_>, Vec<_>) =
            self.0.iter().partition(|t| !t.variables.is_empty());

        let mut first = true;
        for term in var_terms.iter().chain(const_terms.iter()) {
            if first {
                write!(f, "{}", term)?;
                first = false;
            } else if term.constant < 0 {
                // Show as subtraction
                let negated = Term::new(-term.constant, term.variables.clone());
                write!(f, "-{}", negated)?;
            } else {
                write!(f, "+{}", term)?;
            }
        }
        Ok(())
    }
}

impl fmt::Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.variables.is_empty() {
            write!(f, "{}", self.constant)
        } else if self.constant == 1 {
            for np in &self.variables {
                write!(f, "{}", np)?;
            }
            Ok(())
        } else if self.constant == -1 {
            write!(f, "-")?;
            for np in &self.variables {
                write!(f, "{}", np)?;
            }
            Ok(())
        } else {
            write!(f, "{}", self.constant)?;
            for np in &self.variables {
                write!(f, "{}", np)?;
            }
            Ok(())
        }
    }
}

impl fmt::Display for NamedPow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.pow == 1 {
            write!(f, "{}", self.variable)
        } else {
            write!(f, "{}^{}", self.variable, self.pow)
        }
    }
}

#[allow(dead_code)]
fn format_annotation(domain: &AnalysisDomain, target: &crate::ir::Identifier) -> String {
    if let Some(vars) = domain.get(target) {
        let vars = vars.iter().map(|v| format!("{}", v)).join(", ");
        "{".to_owned() + &vars + "}"
    } else {
        "{}".to_owned()
    }
}

#[allow(dead_code)]
pub fn function_with_inferred_shapes_to_string(
    ir: &Function,
    func_facts: &FunctionAnalysis,
    stop_at: Option<Location>,
) -> String {
    let mut output = String::new();

    write!(output, "def {}(", resolve(ir.identifier)).unwrap();
    for (i, param) in ir.param_types.iter().enumerate() {
        if i > 0 {
            output.push_str(", ");
        }
        write!(output, "{}", param).unwrap();
    }
    output.push_str(")");

    if let Some((ret_var, _span)) = &ir.return_type {
        output.push_str(" -> ");
        // Handle tuple returns by printing elements comma-separated
        match ret_var {
            Variable::Collection(Collection::Tuple(vars)) => {
                output.push_str(&vars.iter().join(", "));
            }
            _ => output.push_str(&format!("{}", ret_var)),
        }
    }
    output.push_str(":\n");

    'outer: for block_idx in ir.blocks() {
        let block = ir.data(block_idx);
        let mut block_content = String::new();

        for (instr_idx, stmt) in block.statements.iter().enumerate() {
            let loc = Location {
                block: block_idx,
                instr: instr_idx,
            };

            if let Some(stop) = stop_at {
                if loc == stop {
                    write!(output, "  {}:\n", block_idx).unwrap();
                    output.push_str(&indent(&indent(&block_content)));
                    break 'outer;
                }
            }

            let Some(domain) = func_facts.state.get(&loc) else {
                log::debug!("No analysis domain found for location {}", loc);
                continue;
            };

            let annotated_stmt = if let Some(target) = &stmt.target {
                // Show annotation for Ident targets and self.X Attribute targets
                let annotation = match &target.kind {
                    ExprKind::Ident(name) => format_annotation(domain, name),
                    ExprKind::Attribute { value, attr } => {
                        // Handle self.X assignments
                        if let ExprKind::Ident(self_ident) = &value.kind {
                            if crate::ir::resolve(*self_ident) == "self" {
                                // Look up "self.X" in domain
                                let self_attr_name = crate::ir::intern(&format!(
                                    "self.{}",
                                    crate::ir::resolve(*attr)
                                ));
                                format_annotation(domain, &self_attr_name)
                            } else {
                                "{}".to_owned()
                            }
                        } else {
                            "{}".to_owned()
                        }
                    }
                    _ => "{}".to_owned(),
                };
                format!("{}: {} = {}", target, annotation, stmt.value)
            } else {
                format!("{}", stmt.value)
            };

            writeln!(block_content, "{}", annotated_stmt).unwrap();
        }

        // Check if terminator is the stop location
        let term_loc = Location {
            block: block_idx,
            instr: block.statements.len(),
        };
        if let Some(stop) = stop_at {
            if term_loc == stop {
                write!(output, "  {}:\n", block_idx).unwrap();
                output.push_str(&indent(&indent(&block_content)));
                break 'outer;
            }
        }

        writeln!(block_content, "{}", block.terminator).unwrap();

        write!(output, "  {}:\n", block_idx).unwrap();
        output.push_str(&indent(&indent(&block_content)));
    }

    output
}

#[allow(dead_code)]
pub fn print_function_with_inferred_shapes(
    ir: &Function,
    func_facts: &FunctionAnalysis,
    stop_at: Option<Location>,
) {
    print!(
        "{}",
        function_with_inferred_shapes_to_string(ir, func_facts, stop_at)
    );
}

#[allow(dead_code)]
pub fn class_with_inferred_shapes_to_string(
    class: &Class,
    class_facts: &ClassAnalysis,
    stop_at: Option<Location>,
) -> String {
    let mut output = String::new();

    write!(output, "class {}:\n", resolve(class.identifier)).unwrap();

    // Print inferred attributes as class variable annotations
    if !class_facts.attributes.is_empty() {
        for (attr_ident, vars) in class_facts
            .attributes
            .iter()
            .sorted_by(|(a, _), (b, _)| resolve(**a).cmp(&resolve(**b)))
        {
            let vars_str = vars.iter().map(|v| format!("{}", v)).join(", ");
            write!(output, "    {}: {{{}}}\n", resolve(*attr_ident), vars_str).unwrap();
        }
        output.push_str("\n");
    }

    // Print methods
    if !class.methods.is_empty() {
        let mut first = true;
        for (method_name, method_func) in class
            .methods
            .iter()
            .sorted_by(|(a, _), (b, _)| resolve(**a).cmp(&resolve(**b)))
        {
            if !first {
                output.push_str("\n");
            }
            first = false;

            // Get method analysis if available
            if let Some(method_analysis) = class_facts.methods.get(method_name) {
                let method_output =
                    function_with_inferred_shapes_to_string(method_func, method_analysis, stop_at);
                // Indent the method content
                output.push_str(&indent(&method_output));
            } else {
                // Fallback to basic method printing if no analysis available
                let method_content = format!("{}", method_func);
                output.push_str(&indent(&method_content));
            }
        }
    }

    output
}

#[allow(dead_code)]
pub fn print_class_with_inferred_shapes(
    class: &Class,
    class_facts: &ClassAnalysis,
    stop_at: Option<Location>,
) {
    print!(
        "{}",
        class_with_inferred_shapes_to_string(class, class_facts, stop_at)
    );
}
