use clap::{Parser, Subcommand};
use dimspector::{api::Dimspector, lsp};
use miette::{MietteHandlerOpts, Result};
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

    // Determine project root
    let (dimspector, errors) = if abs_path.is_file() {
        Dimspector::from_single_file(&abs_path)?
    } else if abs_path.is_dir() {
        Dimspector::from_project_root(&abs_path)?
    } else {
        anyhow::bail!("path must be a Python file (.py) or a directory");
    };

    // Print errors if any
    if !errors.is_empty() {
        for (file_path, error) in &errors {
            eprintln!("Error in {}: {}", file_path.display(), error);
        }
        anyhow::bail!("shape analysis found {} errors", errors.len());
    }

    // Print analysis results
    print!("{}", dimspector.format_all());

    Ok(())
}
