mod lower;
mod print;
pub mod types;

use crate::parse::{ParsedFile, ParsedProject};
use anyhow::Result;
use lower::{lower_class, lower_func};
pub use types::{
    Annotation, BasicBlock, BasicBlockIdx, Cfg, Class, Expr, File, Function, Identifier, Parameter,
    Project, Statement, Terminator, intern, resolve,
};

pub fn lower_file(parsed: &ParsedFile) -> Result<File> {
    // Gather all class names first
    use crate::ir::types::intern;
    let class_names: std::collections::HashSet<_> = parsed
        .classes
        .iter()
        .map(|class_def| intern(class_def.name.as_str()))
        .collect();

    let mut functions = Vec::new();
    for func in &parsed.functions {
        let lowered_func = lower_func(func, &class_names, &parsed.path)?;
        functions.push(lowered_func);
    }

    let mut classes = Vec::new();
    for class_def in &parsed.classes {
        let lowered_class = lower_class(class_def, &class_names, &parsed.path)?;
        classes.push(lowered_class);
    }

    Ok(File {
        path: parsed.path.clone(),
        functions,
        classes,
    })
}

pub fn lower_project(parsed_project: &ParsedProject) -> Result<Project> {
    let mut files = Vec::new();

    for parsed_file in &parsed_project.files {
        let file = lower_file(parsed_file)?;
        files.push(file);
    }

    Ok(Project { files })
}
