mod lower;
mod print;
pub mod types;

use std::collections::HashSet;

pub use crate::ir::types::intern;
use crate::parse::{ParsedFile, ParsedProject, SymbolTable};
use anyhow::Result;
use lower::{lower_class, lower_func};
use string_interner::symbol::SymbolU32;
pub use types::{
    Annotation, BasicBlock, BasicBlockIdx, Cfg, Class, Expr, File, Function, Identifier, Parameter,
    Project, Statement, Terminator, resolve,
};

impl File {
    /// Lower a parsed file to IR
    pub fn from_parsed(
        parsed: &ParsedFile,
        all_class_names: &std::collections::HashSet<Identifier>,
        symbol_table: &SymbolTable,
    ) -> Result<Self> {
        let mut functions = Vec::new();
        for func in &parsed.functions {
            let lowered_func = lower_func(func, all_class_names, &parsed.path, &parsed.source)?;
            functions.push(lowered_func);
        }

        let mut classes = Vec::new();
        for class_def in &parsed.classes {
            let lowered_class = lower_class(
                class_def,
                all_class_names,
                &parsed.path,
                &parsed.source,
                symbol_table,
            )?;
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
    pub fn from_parsed_project(
        parsed_project: &ParsedProject,
        symbol_table: &SymbolTable,
    ) -> Result<Self> {
        let mut files = Vec::new();

        // collect class names
        let class_names = parsed_project
            .files
            .iter()
            .flat_map(|f| f.classes.iter().map(|c| intern(&c.name)))
            .collect();

        for parsed_file in &parsed_project.files {
            let file = File::from_parsed(parsed_file, &class_names, symbol_table)?;
            files.push(file);
        }

        Ok(Self { files })
    }

    /// Update a single file in the project by re-lowering it from a parsed file
    pub fn update_file(
        &mut self,
        parsed_file: &ParsedFile,
        symbol_table: &SymbolTable,
    ) -> Result<()> {
        let mut class_names: HashSet<SymbolU32> = self
            .files
            .iter()
            .flat_map(|f| f.classes.iter().map(|c| c.identifier))
            .collect();

        for class in &parsed_file.classes {
            class_names.insert(intern(&class.name));
        }

        let file_path = &parsed_file.path;
        let updated_file = File::from_parsed(parsed_file, &class_names, symbol_table)?;

        // issue here is we don't have all the clas names. we could look through each of the

        // Replace the file if it exists, or add it
        if let Some(idx) = self.files.iter().position(|f| f.path == *file_path) {
            self.files[idx] = updated_file;
        } else {
            self.files.push(updated_file);
        }

        Ok(())
    }
}
