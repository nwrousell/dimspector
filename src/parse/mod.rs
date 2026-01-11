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
    pub functions: Vec<StmtFunctionDef>,
    pub classes: Vec<StmtClassDef>,
    pub imports: Vec<Import>,
}

pub fn parse_file(path: &PathBuf) -> Result<ParsedFile> {
    let content = std::fs::read_to_string(path)?;
    let parsed = parse_module(&content)?;

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

    Ok(ParsedFile {
        path: path.clone(),
        functions,
        classes,
        imports,
    })
}

pub struct ParsedProject {
    pub project_root: PathBuf,
    pub files: Vec<ParsedFile>,
}

pub fn parse_project(project_root: &Path) -> Result<ParsedProject> {
    let mut files = Vec::new();

    // Discover all Python files
    for entry in WalkDir::new(project_root)
        .follow_links(true)
        .into_iter()
        .filter_entry(|e| !e.path().starts_with("."))
    {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("py") {
            let parsed = parse_file(&path.to_path_buf())?;
            files.push(parsed);
        }
    }

    Ok(ParsedProject {
        project_root: project_root.to_path_buf(),
        files,
    })
}

pub use symbols::SymbolTable;
