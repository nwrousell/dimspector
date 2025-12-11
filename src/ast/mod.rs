use miette::{NamedSource, SourceCode};
use rustpython_parser::{self, Mode};
use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::Result;

mod print;
mod types;

pub use types::Program;

/// A source file provided by the user.
#[derive(Clone)]
pub struct Input {
    contents: String,
    path: PathBuf,
}

impl Input {
    // pub fn path(&self) -> &Path {
    //     &self.path
    // }

    /// View the source file as a Miette [`SourceCode`] for getting eg line info.
    pub fn as_source(&self) -> impl SourceCode {
        self.contents.as_str()
    }

    /// Transform into a Miette [`NamedSource`] for diagnostic reporting.
    pub fn into_named_source(self) -> NamedSource<String> {
        let file_name = self.path.file_name().unwrap().to_string_lossy().to_string();
        NamedSource::new(file_name, self.contents)
    }
}

/// Read a source file from disk.
pub fn read(path: &Path) -> Result<Input> {
    let contents = fs::read_to_string(path)?;
    Ok(Input {
        contents,
        path: path.to_path_buf(),
    })
}

pub fn parse(input: &Input) -> Result<Program> {
    let ast = rustpython_parser::parse(&input.contents, Mode::Module, "simple.py")?;

    let mut functions = Vec::new();
    if let rustpython_parser::ast::Mod::Module(module) = ast {
        for stmt in module.body {
            match stmt {
                rustpython_parser::ast::Stmt::FunctionDef(func_def) => {
                    functions.push(func_def);
                }
                _ => (),
            }
        }
    } else {
        unreachable!("parse on file gave non module")
    }

    let prog = Program { functions };

    Ok(prog)
}
