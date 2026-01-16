pub mod helpers;
pub mod prints;
pub mod symbols;

use anyhow::Result;
use ruff_python_ast::{StmtClassDef, StmtFunctionDef};
use ruff_python_parser::parse_module;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

#[derive(Debug, Clone)]
pub enum Import {
    Import {
        names: Vec<String>,
    },
    ImportFrom {
        module: String,
        names: Vec<String>,
    },
    ImportFromRelative {
        level: usize,
        module: Option<String>,
        names: Vec<String>,
    },
}

pub struct ParsedFile {
    pub path: PathBuf,
    pub source: String,
    pub functions: Vec<StmtFunctionDef>,
    pub classes: Vec<StmtClassDef>,
    pub imports: Vec<Import>,
}

impl ParsedFile {
    /// Parse a file from disk
    pub fn from_path(path: &PathBuf) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Self::from_content(path, &content)
    }

    /// Parse a file from content string
    pub fn from_content(path: &PathBuf, content: &str) -> Result<Self> {
        let parsed = parse_module(content)?;

        let mut functions = Vec::new();
        let mut classes = Vec::new();
        let mut imports = Vec::new();

        for stmt in parsed.syntax().body.iter() {
            match stmt {
                ruff_python_ast::Stmt::FunctionDef(f) => functions.push(f.clone()),
                ruff_python_ast::Stmt::ClassDef(c) => classes.push(c.clone()),
                ruff_python_ast::Stmt::Import(i) => {
                    let names: Vec<String> = i
                        .names
                        .iter()
                        .map(|alias| alias.name.as_str().to_string())
                        .collect();
                    imports.push(Import::Import { names });
                }
                ruff_python_ast::Stmt::ImportFrom(i) => {
                    if let Some(module) = &i.module {
                        let module_str = module.as_str().to_string();
                        let names: Vec<String> = i
                            .names
                            .iter()
                            .map(|alias| alias.name.as_str().to_string())
                            .collect();

                        if i.level > 0 {
                            imports.push(Import::ImportFromRelative {
                                level: i.level as usize,
                                module: Some(module_str),
                                names,
                            });
                        } else {
                            imports.push(Import::ImportFrom {
                                module: module_str,
                                names,
                            });
                        }
                    }
                }
                _ => {}
            }
        }

        Ok(Self {
            path: path.clone(),
            source: content.to_string(),
            functions,
            classes,
            imports,
        })
    }
}

pub struct ParsedProject {
    pub project_root: PathBuf,
    pub files: Vec<ParsedFile>,
}

impl ParsedProject {
    /// Parse an entire project from a project root directory
    pub fn from_project_root(project_root: &Path) -> Result<Self> {
        let mut files = Vec::new();

        // Discover all Python files
        for entry in WalkDir::new(project_root)
            .follow_links(true)
            .into_iter()
            .filter_entry(|e| {
                // Skip hidden directories (starting with .)
                let path = e.path();
                if path.is_dir() {
                    // Check if directory name starts with . (like .venv, .git, etc.)
                    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                        if name.starts_with('.') {
                            return false;
                        }
                    }
                }
                true
            })
        {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("py") {
                log::info!("Discovered Python file: {}", path.display());
                let parsed = ParsedFile::from_path(&path.to_path_buf())?;
                files.push(parsed);
            }
        }

        log::info!(
            "Collected {} Python files from project root: {}",
            files.len(),
            project_root.display()
        );
        for file in &files {
            log::debug!("  - {}", file.path.display());
        }

        Ok(Self {
            project_root: project_root.to_path_buf(),
            files,
        })
    }

    /// Re-parse a single file and update it in the project
    pub fn update_file(&mut self, file_path: &Path, file_content: &str) -> Result<()> {
        let file_path_buf = file_path.to_path_buf();
        let parsed_file = ParsedFile::from_content(&file_path_buf, file_content)?;

        // Replace the file if it exists, or add it
        if let Some(idx) = self.files.iter().position(|f| f.path == file_path_buf) {
            self.files[idx] = parsed_file;
        } else {
            self.files.push(parsed_file);
        }

        Ok(())
    }
}

pub use symbols::SymbolTable;
