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

    // Determine project root
    let project_root = if abs_path.is_file() {
        abs_path.parent().unwrap_or(&abs_path)
    } else if abs_path.is_dir() {
        &abs_path
    } else {
        anyhow::bail!("path must be a Python file (.py) or a directory");
    };

    // Use Dimspector API to analyze the project
    let (dimspector, errors) = Dimspector::from_project_root(project_root)?;

    // Print errors if any
    if !errors.is_empty() {
        for (file_path, error) in &errors {
            eprintln!("Error in {}: {:?}", file_path.display(), error);
        }
        anyhow::bail!("shape analysis found {} errors", errors.len());
    }

    // Print analysis results
    print!("{}", dimspector.format_all());

    Ok(())
}
