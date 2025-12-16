use miette::{NamedSource, SourceCode};
use rustpython_parser::{self, Mode};
use std::{
    fs,
    path::{Path, PathBuf},
};
use tower_lsp::lsp_types::Position;

use anyhow::Result;

mod print;
mod types;

pub use types::Program;

/// Index for fast byte offset -> line/column conversion.
#[derive(Clone, Debug)]
pub struct LineIndex {
    /// Byte offset of the start of each line (line 0 starts at offset 0)
    line_starts: Vec<usize>,
}

impl LineIndex {
    pub fn new(text: &str) -> Self {
        let mut line_starts = vec![0];
        for (i, ch) in text.char_indices() {
            if ch == '\n' {
                line_starts.push(i + 1);
            }
        }
        LineIndex { line_starts }
    }

    /// Convert a byte offset to an LSP Position (0-indexed line and character)
    pub fn offset_to_position(&self, offset: usize) -> Position {
        // Binary search for the line containing this offset
        let line = self
            .line_starts
            .partition_point(|&start| start <= offset)
            .saturating_sub(1);
        let col = offset.saturating_sub(self.line_starts[line]);
        Position::new(line as u32, col as u32)
    }
}

/// A source file provided by the user.
#[derive(Clone)]
pub struct Input {
    pub contents: String,
    pub path: PathBuf,
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
