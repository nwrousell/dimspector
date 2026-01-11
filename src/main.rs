use clap::{Parser, Subcommand};
use dimspector::{
    analysis::{ShapeError, analyze},
    ir, lsp,
    parse::{ParsedProject, SymbolTable, parse_file, parse_project},
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
    /// Analyze a file or directory and report shape errors
    Check {
        /// Path of the file (.py) or directory to check
        path: PathBuf,
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
        Command::Check { path } => {
            if let Err(err) = check(path) {
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

fn check(path: PathBuf) -> anyhow::Result<()> {
    if !path.exists() {
        anyhow::bail!("path not found: {}", path.display());
    }

    let abs_path = std::fs::canonicalize(&path)?;

    // Determine if it's a single file or directory
    let parsed_project =
        if abs_path.is_file() && abs_path.extension().and_then(|s| s.to_str()) == Some("py") {
            // Single file mode: parse just this file
            let parsed_file = parse_file(&abs_path)?;
            let project_root = abs_path.parent().unwrap_or(&abs_path);
            ParsedProject {
                project_root: project_root.to_path_buf(),
                files: vec![parsed_file],
            }
        } else if abs_path.is_dir() {
            // Directory mode: parse entire project
            parse_project(&abs_path)?
        } else {
            anyhow::bail!("path must be a Python file (.py) or a directory");
        };

    log::debug!("Parsed project:\n{}", parsed_project);

    // Build symbol table
    let symbol_table = SymbolTable::build(&parsed_project);

    // Lower to IR
    let project_ir = ir::lower_project(&parsed_project)?;
    log::debug!("IR:\n{}", project_ir);

    // For error reporting, use the first file if single file mode, or the original path
    let error_file = if abs_path.is_file() {
        &abs_path
    } else {
        // For directory mode, we'll use the first file for error context if needed
        &parsed_project.files[0].path
    };
    let file_contents = std::fs::read_to_string(error_file)?;
    let named_source = NamedSource::new(error_file.display().to_string(), file_contents);

    let res = analyze(project_ir.clone(), &symbol_table);
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

    print!("{}", res.format_all(&project_ir));

    Ok(())
}
