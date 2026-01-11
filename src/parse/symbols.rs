use crate::parse::helpers::{module_name_to_path, path_to_module_name};
use crate::parse::{Import, ParsedProject};
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct SymbolTable {
    // Map (file_path, identifier_string) -> canonical_path
    name_map: HashMap<(PathBuf, String), String>,
    project_root: PathBuf,
}

impl SymbolTable {
    /// Build symbol table from parsed project
    pub fn build(project: &ParsedProject) -> Self {
        let mut table = Self {
            name_map: HashMap::new(),
            project_root: project.project_root.clone(),
        };

        // First, register all functions/classes with their canonical paths
        for file in &project.files {
            let module_name = path_to_module_name(&file.path, &project.project_root);

            // Register functions
            for func in &file.functions {
                let func_name = func.name.as_str().to_string();
                let canonical = format!("{}.{}", module_name, func_name);
                table
                    .name_map
                    .insert((file.path.clone(), func_name.clone()), canonical.clone());
            }

            // Register classes
            for class in &file.classes {
                let class_name = class.name.as_str().to_string();
                let canonical = format!("{}.{}", module_name, class_name);
                table
                    .name_map
                    .insert((file.path.clone(), class_name.clone()), canonical.clone());
            }
        }

        // Then, process imports to map imported names to canonical paths
        for file in &project.files {
            table.process_imports(file);
        }

        table
    }

    fn process_imports(&mut self, file: &crate::parse::ParsedFile) {
        for import in &file.imports {
            match import {
                Import::Import { names } => {
                    // import X
                    // Map X -> X's canonical path (if X is a module we know about)
                    for name in names {
                        if let Some(target_path) = module_name_to_path(name, &self.project_root) {
                            if let Some(target_module) = self.get_module_name(&target_path) {
                                self.name_map.insert(
                                    (file.path.clone(), name.clone()),
                                    target_module.clone(),
                                );
                            }
                        }
                    }
                }
                Import::ImportFrom { module, names } => {
                    // from X import Y
                    if let Some(target_path) = module_name_to_path(module, &self.project_root) {
                        if let Some(target_module) = self.get_module_name(&target_path) {
                            for name in names {
                                self.name_map.insert(
                                    (file.path.clone(), name.clone()),
                                    format!("{}.{}", target_module, name),
                                );
                            }
                        }
                    }
                }
                Import::ImportFromRelative {
                    level: _,
                    module: _,
                    names: _,
                } => {
                    // from .X import Y (simplified - just skip for now)
                    // TODO: Handle relative imports
                }
            }
        }
    }

    fn get_module_name(&self, path: &PathBuf) -> Option<String> {
        Some(path_to_module_name(path, &self.project_root))
    }

    /// Resolve identifier to canonical path
    pub fn resolve(&self, file: &PathBuf, identifier: &str) -> Option<&String> {
        self.name_map.get(&(file.clone(), identifier.to_string()))
    }
}
