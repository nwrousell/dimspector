mod lower;
mod print;
pub mod types;

use anyhow::Result;

use crate::parse::ParsedFile;
use lower::lower_func;
pub use types::{
    Annotation, BasicBlock, BasicBlockIdx, Cfg, Expr, Function, Parameter, Path, Program,
    Statement, Terminator,
};

pub fn lower(parsed: &ParsedFile) -> Result<Program> {
    let mut functions = Vec::new();
    for func in &parsed.functions {
        let lowered_func = lower_func(func, &parsed.db, &parsed.semantic_model)?;
        functions.push(lowered_func);
    }
    Ok(Program { functions })
}
