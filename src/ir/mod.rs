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

impl File {
    /// Lower a parsed file to IR
    pub fn from_parsed(parsed: &ParsedFile) -> Result<Self> {
        // Gather all class names first
        use crate::ir::types::intern;
        let class_names: std::collections::HashSet<_> = parsed
            .classes
            .iter()
            .map(|class_def| intern(class_def.name.as_str()))
            .collect();

        let mut functions = Vec::new();
        for func in &parsed.functions {
            let lowered_func = lower_func(func, &class_names, &parsed.path, &parsed.source)?;
            functions.push(lowered_func);
        }

        let mut classes = Vec::new();
        for class_def in &parsed.classes {
            let lowered_class = lower_class(class_def, &class_names, &parsed.path, &parsed.source)?;
            classes.push(lowered_class);
        }

        Ok(Self {
            path: parsed.path.clone(),
            functions,
            classes,
        })
    }
}

impl Project {
    /// Lower a parsed project to IR
    pub fn from_parsed_project(parsed_project: &ParsedProject) -> Result<Self> {
        let mut files = Vec::new();

        for parsed_file in &parsed_project.files {
            let file = File::from_parsed(parsed_file)?;
            files.push(file);
        }

        Ok(Self { files })
    }

    /// Update a single file in the project by re-lowering it from a parsed file
    pub fn update_file(&mut self, parsed_file: &ParsedFile) -> Result<()> {
        let file_path = &parsed_file.path;
        let updated_file = File::from_parsed(parsed_file)?;

        // Replace the file if it exists, or add it
        if let Some(idx) = self.files.iter().position(|f| f.path == *file_path) {
            self.files[idx] = updated_file;
        } else {
            self.files.push(updated_file);
        }

        Ok(())
    }
}
