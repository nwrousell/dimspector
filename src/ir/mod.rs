mod lower;
mod print;
pub mod types;

use anyhow::Result;

use lower::lower_func;
pub use types::{
    Annotation, BasicBlock, BasicBlockIdx, Cfg, Expr, Function, Parameter, Path, Program,
    Statement, Terminator,
};

use crate::ast::{self, LineIndex};

pub fn lower(program: ast::Program, line_index: &LineIndex) -> Result<Program> {
    let functions = program
        .functions
        .iter()
        .map(|func| lower_func(func.clone(), line_index))
        .collect::<Result<Vec<Function>, anyhow::Error>>()?;

    Ok(Program { functions })
}
