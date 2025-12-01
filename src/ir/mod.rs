use anyhow::Error;

pub use crate::ir::types::{Function, Program};

pub mod types;

pub fn lower(program: rustpython_parser::ast::ModModule) -> Result<Program, Error> {
    Ok(Program {
        functions: Vec::new(),
    })
}
