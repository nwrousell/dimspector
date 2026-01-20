mod class;
mod dimvars;
mod errors;
mod function;
mod global;
mod models;
mod print;
mod types;

use std::collections::{HashMap, HashSet};

pub use types::{Shape, Variable};

pub use crate::analysis::dimvars::{DimKind, DimVar};
use crate::ir::{Class, Function, Identifier, Project};
use crate::parse::SymbolTable;
use anyhow::Result;
pub use errors::ShapeError;

pub use class::ClassAnalysis;
pub use function::FunctionAnalysis;
pub use global::GlobalAnalysis;

pub use print::{
    class_with_inferred_shapes_to_string, function_with_inferred_shapes_to_string,
    print_class_with_inferred_shapes, print_function_with_inferred_shapes,
};

pub(crate) type AnalysisDomain = HashMap<Identifier, HashSet<Variable>>;

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

pub(crate) fn vars_to_inlay(vars: &HashSet<Variable>) -> Option<String> {
    if vars.is_empty() {
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
