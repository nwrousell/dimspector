mod lower;
mod print;
mod types;

use anyhow::Result;

use lower::lower_func;
pub use types::{
    Annotation, BasicBlock, BasicBlockIdx, Cfg, Expr, Function, Parameter, Path, Program,
    Statement, Terminator,
};

use crate::ast;

pub fn lower(program: ast::Program) -> Result<Program> {
    let functions = program
        .functions
        .iter()
        .map(|func| lower_func(func.clone()))
        .collect::<Result<Vec<Function>, anyhow::Error>>()?;

    Ok(Program { functions })
}
