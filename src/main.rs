use clap::Parser;
use dimspector::{
    analysis::{ShapeError, analyze, print_ir_with_inferred_shapes},
    ast, ir,
};
use miette::{MietteHandlerOpts, Report, Result};
use std::path::PathBuf;

#[derive(Parser, Debug)]
struct Args {
    /// Path of the file to check
    file: PathBuf,
}

fn main() -> Result<()> {
    env_logger::init();

    // Configure miette to show more surrounding source code
    miette::set_hook(Box::new(|_| {
        Box::new(MietteHandlerOpts::new().context_lines(5).build())
    }))?;

    let args = Args::parse();

    if let Err(err) = run(args.file) {
        eprintln!("{:?}", err);
        std::process::exit(1);
    }
    Ok(())
}

fn run(file: PathBuf) -> Result<()> {
    let input = match ast::read(&file) {
        Err(e) => return Err(Report::msg(e)),
        Ok(input) => input,
    };

    let program = match ast::parse(&input) {
        Err(e) => return Err(Report::msg(e)),
        Ok(program) => program,
    };
    // log::debug!("AST:\n{}", program);

    let ir = match ir::lower(program) {
        Err(e) => return Err(Report::msg(e)),
        Ok(ir) => ir,
    };
    log::debug!("IR:\n{}", ir);

    let res = analyze(ir.clone());
    let res = match res {
        Ok(res) => res,
        Err(err) => {
            let named_source = input.into_named_source();
            if let Some(shape_error) = err.downcast_ref::<ShapeError>() {
                let report = Report::new(shape_error.clone()).with_source_code(named_source);
                return Err(report.into());
            } else {
                return Err(Report::msg(err));
            }
        }
    };

    for (name, facts) in &res.functions {
        let func = ir.functions.iter().find(|f| f.identifier == *name).unwrap();
        print_ir_with_inferred_shapes(func, facts);
        println!("\n")
    }

    Ok(())
}
