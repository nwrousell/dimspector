mod lower;
mod print;
pub mod types;

use anyhow::Result;
use ty_python_semantic::SemanticModel;

use crate::parse::ParsedFile;
use lower::{lower_class, lower_func};
pub use types::{
    Annotation, BasicBlock, BasicBlockIdx, Cfg, Class, Expr, Function, Identifier, Parameter,
    Program, Statement, Terminator, intern, resolve,
};

pub fn lower(parsed: &ParsedFile) -> Result<Program> {
    let semantic_model = SemanticModel::new(&parsed.db, parsed.file);

    // Gather all class names first
    use crate::ir::types::intern;
    let class_names: std::collections::HashSet<_> = parsed
        .classes
        .iter()
        .map(|class_def| intern(class_def.name.as_str()))
        .collect();

    let mut functions = Vec::new();
    for func in &parsed.functions {
        let lowered_func = lower_func(func, &parsed.db, &semantic_model, &class_names)?;
        functions.push(lowered_func);
    }

    let mut classes = Vec::new();
    for class_def in &parsed.classes {
        let lowered_class = lower_class(class_def, &parsed.db, &semantic_model, &class_names)?;
        classes.push(lowered_class);
    }

    Ok(Program { functions, classes })
}
