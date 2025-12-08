use clap::Parser;
use dimspector::{
    analysis::{analyze, print_ir_with_inferred_shapes},
    ast, ir,
};
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
    let input = ast::read(&file)?;

    let program = ast::parse(&input)?;
    // log::debug!("AST:\n{}", program);

    let ir = ir::lower(program)?;
    log::debug!("IR:\n{}", ir);

    // println!("IR full:\n{:#?}", ir);

    let res = analyze(ir.clone())?;
    // log::debug!("Analysis:\n{}", res);

    for (name, facts) in &res.functions {
        let func = ir.functions.iter().find(|f| f.identifier == *name).unwrap();
        print_ir_with_inferred_shapes(func, facts);
        println!("\n")
    }

    Ok(())
}
