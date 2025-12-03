use clap::Parser;
// use miette::{MietteHandlerOpts, Result};
use std::path::PathBuf;

use anyhow::Result;

use crate::ast::Input;

mod analysis;
mod ast;
mod ir;

#[derive(Parser, Debug)]
struct Args {
    /// Path of the file to check
    file: PathBuf,
}

fn main() -> Result<()> {
    env_logger::init();

    // Configure miette to show more surrounding source code
    // miette::set_hook(Box::new(|_| {
    //     Box::new(
    //         MietteHandlerOpts::new()
    //             .context_lines(5) // Show 6 lines above and below (default is 3)
    //             .build(),
    //     )
    // }))?;

    let args = Args::parse();

    let input = ast::read(&args.file)?;

    run(&args, &input)
}

fn run(_args: &Args, input: &Input) -> Result<()> {
    let program = ast::parse(input)?;
    log::debug!("AST:\n{}", program);

    let ir = ir::lower(program)?;
    log::debug!("IR:\n{}", ir);

    // analyze(ir)?;

    Ok(())
}
