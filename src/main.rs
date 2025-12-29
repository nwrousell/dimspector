use clap::{Parser, Subcommand};
use dimspector::{
    analysis::{ShapeError, analyze, print_ir_with_inferred_shapes},
    ir, lsp,
    parse::parse_file,
};
use miette::{MietteHandlerOpts, NamedSource, Result};
use std::path::PathBuf;

#[derive(Parser, Debug)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Analyze a file and report shape errors
    Check {
        /// Path of the file to check
        file: PathBuf,
    },
    /// Start the language server (communicates over stdio)
    Server,
}

fn main() -> Result<()> {
    env_logger::init();

    // Configure miette to show more surrounding source code
    miette::set_hook(Box::new(|_| {
        Box::new(MietteHandlerOpts::new().context_lines(5).build())
    }))?;

    let args = Args::parse();

    match args.command {
        Command::Check { file } => {
            if let Err(err) = check(file) {
                eprintln!("{:?}", err);
                std::process::exit(1);
            }
        }
        Command::Server => {
            lsp::start_server();
        }
    }

    Ok(())
}

fn check(file: PathBuf) -> anyhow::Result<()> {
    if !file.exists() {
        anyhow::bail!("file not found: {}", file.display());
    }

    let abs_file = std::fs::canonicalize(&file)?;
    let parsed = parse_file(&abs_file)?;

    let ir = ir::lower(&parsed)?;
    log::debug!("IR:\n{}", ir);

    let file_contents = std::fs::read_to_string(&abs_file)?;
    let named_source = NamedSource::new(file.display().to_string(), file_contents);

    let res = analyze(ir.clone());
    let res = match res {
        Ok(res) => res,
        Err(err) => {
            if let Some(shape_error) = err.downcast_ref::<ShapeError>() {
                use miette::Report;
                let report = Report::new(shape_error.clone()).with_source_code(named_source);
                eprintln!("{}", report);
                anyhow::bail!("shape analysis failed");
            } else {
                return Err(err);
            }
        }
    };

    for (name, facts) in &res.functions {
        let func = ir.functions.iter().find(|f| f.identifier == *name).unwrap();
        print_ir_with_inferred_shapes(func, facts, None);
        println!("\n")
    }

    Ok(())
}
