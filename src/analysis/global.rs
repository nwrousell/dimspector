use std::collections::HashMap;
use std::sync::Arc;

use itertools::Itertools;

use crate::analysis::models::ModelContext;
use crate::ir::types::intern;
use crate::ir::{Class, File, Function, Identifier, Project, resolve};
use crate::parse::SymbolTable;
use anyhow::Result;

use super::class::{ClassAnalysis, analyze_class};
use super::errors::ShapeError;
use super::function::FunctionAnalysis;
use super::print::{class_with_inferred_shapes_to_string, function_with_inferred_shapes_to_string};

#[derive(Debug)]
pub struct GlobalAnalysis {
    pub functions: HashMap<Identifier, FunctionAnalysis>,
    pub classes: HashMap<Identifier, ClassAnalysis>,
    pub models: Arc<ModelContext>,
    pub symbol_table: Arc<SymbolTable>,
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
        let result = func_analysis.analyze_func(func, self);
        // Insert the analysis even if it failed - partial state still has inlay hints before the error
        self.functions.insert(canonical_id, func_analysis);
        result
    }

    pub fn analyze_class(&mut self, class: &Class) -> Result<()> {
        let local_name = resolve(class.identifier);
        let canonical = self
            .symbol_table
            .resolve(&class.file_path, &local_name)
            .cloned()
            .unwrap_or(local_name);
        let canonical_id = intern(&canonical);

        let class_analysis = analyze_class(class, self)?;
        self.classes.insert(canonical_id, class_analysis);
        Ok(())
    }

    /// Analyze a file, collecting up to 1 error per function
    pub fn analyze_file(
        &mut self,
        file: &File,
        errors: &mut Vec<(std::path::PathBuf, ShapeError)>,
    ) {
        for class in &file.classes {
            if let Err(e) = self.analyze_class(class) {
                if let Some(shape_error) = e.downcast_ref::<ShapeError>() {
                    errors.push((file.path.clone(), shape_error.clone()));
                }
            }
        }

        for func in &file.functions {
            if let Err(e) = self.analyze_func(func) {
                if let Some(shape_error) = e.downcast_ref::<ShapeError>() {
                    errors.push((file.path.clone(), shape_error.clone()));
                }
            }
        }
    }

    /// Analyze a project, collecting errors but continuing on failure
    pub fn analyze_project(
        &mut self,
        project: &Project,
        errors: &mut Vec<(std::path::PathBuf, ShapeError)>,
    ) -> Result<()> {
        // Analyze all classes
        let all_classes: Vec<&Class> = project.files.iter().flat_map(|f| &f.classes).collect();
        for class in all_classes {
            if let Err(e) = self.analyze_class(class) {
                if let Some(shape_error) = e.downcast_ref::<ShapeError>() {
                    errors.push((class.file_path.clone(), shape_error.clone()));
                } else {
                    return Err(e);
                }
            }
        }

        // Analyze all functions
        let all_functions: Vec<&Function> =
            project.files.iter().flat_map(|f| &f.functions).collect();
        for func in all_functions {
            if let Err(e) = self.analyze_func(func) {
                if let Some(shape_error) = e.downcast_ref::<ShapeError>() {
                    errors.push((func.file_path.clone(), shape_error.clone()));
                } else {
                    return Err(e);
                }
            }
        }
        
        Ok(())
    }

    /// Format the entire analysis result as a string, including both classes and functions.
    pub fn format_all(&self, project: &Project) -> String {
        let mut output = String::new();

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
}
