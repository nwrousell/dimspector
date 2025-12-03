use clap::Parser;
use torch_infer2::analysis::analyze;
// use miette::{MietteHandlerOpts, Result};
use std::path::PathBuf;

use anyhow::Result;

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

    run(args.file)
}

fn run(file: PathBuf) -> Result<()> {
    let input = torch_infer2::ast::read(&file)?;

    let program = torch_infer2::ast::parse(&input)?;
    log::debug!("AST:\n{}", program);

    let ir = torch_infer2::ir::lower(program)?;
    log::debug!("IR:\n{}", ir);

    let res = analyze(ir)?;
    log::debug!("Analysis:\n{}", res);

    Ok(())
}
