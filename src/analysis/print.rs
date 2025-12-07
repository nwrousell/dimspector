use std::{fmt, fmt::Write};

use itertools::Itertools;

use crate::{
    analysis::{
        AnalysisDomain, FunctionAnalysis, GlobalAnalysis,
        dimvars::{DimKind, DimVar},
    },
    ir::{
        Function,
        types::{Location, Path},
    },
    utils::{indent, write_comma_separated},
};

use super::types::{Shape, Variable};

impl fmt::Display for DimVar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind() {
            DimKind::Concrete(c) => write!(f, "{}", c),
            DimKind::Named(n) => write!(f, "{}", n),
            DimKind::Add { left, right } => {
                write!(f, "{} + {}", *left, *right)
            }
            DimKind::Mul { left, right } => write!(f, "{} * {}", *left, *right),
        }
    }
}

impl fmt::Display for Variable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Variable::Top => write!(f, "⟙"),
            Variable::DimVar(dim_var) => write!(f, "{}", dim_var),
            Variable::Tensor(shape) => write!(f, "{}", shape),
            Variable::Tuple(vars) => write_comma_separated(f, vars),
            Variable::None => write!(f, "None"),
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
        write!(f, "Function {}\n", self.id)?;
        for (loc, domain) in self
            .state
            .iter()
            .sorted_by(|(l_a, _), (l_b, _)| Ord::cmp(*l_a, *l_b))
        {
            write!(f, "  {}\n", loc)?;
            for (path, vars) in domain.iter() {
                write!(f, "    {} => {{", path)?;
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

#[allow(dead_code)]
fn format_annotation(domain: &AnalysisDomain, target: &Path) -> String {
    if let Some(vars) = domain.get(target) {
        let vars = vars.iter().map(|v| format!("{}", v)).join(", ");
        "{".to_owned() + &vars + "}"
    } else {
        "{}".to_owned()
    }
}

#[allow(dead_code)]
pub fn ir_with_inferred_shapes_to_string(ir: &Function, func_facts: &FunctionAnalysis) -> String {
    let mut output = String::new();

    write!(output, "def {}(", ir.identifier).unwrap();
    for (i, param) in ir.params.iter().enumerate() {
        if i > 0 {
            output.push_str(", ");
        }
        write!(output, "{}", param).unwrap();
    }
    output.push_str("):\n");

    for block_idx in ir.blocks() {
        let block = ir.data(block_idx);
        let mut block_content = String::new();

        for (instr_idx, stmt) in block.statements.iter().enumerate() {
            let loc = Location {
                block: block_idx,
                instr: instr_idx,
            };
            let domain = func_facts.state.get(&loc).unwrap();

            let annotated_stmt = if let Some(target) = &stmt.target {
                let annotation = format_annotation(domain, target);
                format!("{}: {} = {}", target, annotation, stmt.value)
            } else {
                format!("{}", stmt.value)
            };

            writeln!(block_content, "{}", annotated_stmt).unwrap();
        }

        writeln!(block_content, "{}", block.terminator).unwrap();

        write!(output, "  {}:\n", block_idx).unwrap();
        output.push_str(&indent(indent(&block_content)));
    }

    output
}

#[allow(dead_code)]
pub fn print_ir_with_inferred_shapes(ir: &Function, func_facts: &FunctionAnalysis) {
    print!("{}", ir_with_inferred_shapes_to_string(ir, func_facts));
}
