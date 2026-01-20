use std::collections::HashSet;
use std::path::{Path, PathBuf};

use crate::analysis::{self, ShapeError};
use crate::parse;
use crate::{ParsedProject, ir};

/// High-level API that supports incremental analysis
pub struct Dimspector {
    pub parsed_project: parse::ParsedProject,
    symbol_table: parse::SymbolTable,
    project_ir: ir::Project,
    global_analysis: analysis::GlobalAnalysis,
}

impl Dimspector {
    /// Build a Dimspector from a project root, parsing and analyzing everything
    pub fn from_project_root(
        project_root: &Path,
    ) -> anyhow::Result<(Self, Vec<(PathBuf, ShapeError)>)> {
        log::info!(
            "Building Dimspector from project root: {}",
            project_root.display()
        );

        // Parse entire project
        let parsed_project = parse::ParsedProject::from_project_root(project_root)?;

        // Build symbol table
        let symbol_table = parse::SymbolTable::build(&parsed_project);

        // Lower to IR
        let project_ir = ir::Project::from_parsed_project(&parsed_project)?;

        // Collect all functions for signature models
        let all_functions: Vec<ir::Function> = project_ir
            .files
            .iter()
            .flat_map(|f| f.functions.clone())
            .collect();

        // Create GlobalAnalysis with all functions (for signature models)
        let mut global_analysis = analysis::GlobalAnalysis::new(&symbol_table, &all_functions);

        // Analyze everything and collect errors
        let mut errors = Vec::new();
        global_analysis.analyze_project(&project_ir, &mut errors);

        Ok((
            Self {
                parsed_project,
                symbol_table,
                project_ir,
                global_analysis,
            },
            errors,
        ))
    }

    pub fn from_single_file(file: &Path) -> anyhow::Result<(Self, Vec<(PathBuf, ShapeError)>)> {
        log::info!("Building Dimspector from single file: {}", file.display());

        // Parse
        let parsed_file = parse::ParsedFile::from_path(&file.to_path_buf())?;

        let parsed_project = ParsedProject {
            files: vec![parsed_file],
            project_root: file.parent().unwrap().to_path_buf(),
        };

        // Build symbol table
        let symbol_table = parse::SymbolTable::build(&parsed_project);

        // Lower to IR
        let project_ir = ir::Project::from_parsed_project(&parsed_project)?;

        // Create GlobalAnalysis
        let all_functions = &project_ir.files.first().unwrap().functions;
        let mut global_analysis = analysis::GlobalAnalysis::new(&symbol_table, all_functions);

        // Analyze everything and collect errors
        let mut errors = Vec::new();
        global_analysis.analyze_project(&project_ir, &mut errors);

        Ok((
            Self {
                parsed_project,
                symbol_table,
                project_ir,
                global_analysis,
            },
            errors,
        ))
    }

    /// Re-parse, re-lower, and re-analyze a single file, returning errors
    pub fn analyze_file(
        &mut self,
        file_path: &Path,
        file_content: &str,
    ) -> anyhow::Result<Vec<(PathBuf, ShapeError)>> {
        // Update parsed_project - re-parse the file
        self.parsed_project.update_file(file_path, file_content)?;

        // Re-build symbol table
        self.symbol_table = parse::SymbolTable::build(&self.parsed_project);

        // Re-lower the file to IR
        let parsed_file_ref = self
            .parsed_project
            .files
            .iter()
            .find(|f| f.path == file_path)
            .unwrap();
        self.project_ir.update_file(parsed_file_ref)?;

        // Re-create GlobalAnalysis with updated functions (for signature models)
        let all_functions: Vec<ir::Function> = self
            .project_ir
            .files
            .iter()
            .flat_map(|f| f.functions.clone())
            .collect();
        self.global_analysis = analysis::GlobalAnalysis::new(&self.symbol_table, &all_functions);

        // First, analyze all classes from all files so they're available for type resolution
        for file in &self.project_ir.files {
            for class in &file.classes {
                // Ignore errors from other files' classes - we just need them registered
                let _ = self.global_analysis.analyze_class(class);
            }
        }

        // Analyze the specific file and collect errors
        let mut errors = Vec::new();
        if let Some(file) = self.project_ir.files.iter().find(|f| f.path == file_path) {
            self.global_analysis.analyze_file(file, &mut errors);
        }

        Ok(errors)
    }

    /// Analyze a set of files, calling analyze_file for each
    pub fn analyze_files(
        &mut self,
        files: &HashSet<PathBuf>,
        file_contents: &std::collections::HashMap<PathBuf, String>,
    ) -> anyhow::Result<Vec<(PathBuf, ShapeError)>> {
        let mut all_errors = Vec::new();

        for file_path in files {
            if let Some(content) = file_contents.get(file_path) {
                let errors = self.analyze_file(file_path, content)?;
                all_errors.extend(errors);
            }
        }

        Ok(all_errors)
    }

    /// Get inlay hints for given files
    pub fn inlay_hints(&self, files: &HashSet<PathBuf>) -> Vec<tower_lsp::lsp_types::InlayHint> {
        use crate::ir::types::ExprKind;
        use itertools::Either;
        use tower_lsp::lsp_types::InlayHint;

        let mut hints = Vec::new();

        // Iterate through all analyzed functions and filter by requested files
        for (_, func_analysis) in &self.global_analysis.functions {
            let func = &func_analysis.function;

            // Only process functions from requested files
            if !files.contains(&func.file_path) {
                continue;
            }

            // Generate hints for this function (similar to GlobalAnalysis::inlay_hints)
            for loc in &func.locations {
                if let Either::Left(stmt) = func.instr(&loc) {
                    if let Some(target) = &stmt.target
                        && let Some(position) = stmt.assign_end
                    {
                        // Only show hints for Ident targets
                        if let ExprKind::Ident(name) = &target.kind {
                            if let Some(vars) =
                                func_analysis.state.get(loc).and_then(|d| d.get(name))
                            {
                                if let Some(label) = crate::analysis::vars_to_inlay(vars) {
                                    hints.push(InlayHint {
                                        position,
                                        label: tower_lsp::lsp_types::InlayHintLabel::String(label),
                                        kind: None,
                                        text_edits: None,
                                        tooltip: None,
                                        padding_left: None,
                                        padding_right: None,
                                        data: None,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }

        hints
    }

    /// Format the entire analysis result as a string
    pub fn format_all(&self) -> String {
        self.global_analysis.format_all(&self.project_ir)
    }
}
