pub mod analysis;
pub mod ir;
pub mod lsp;
pub mod parse;
pub mod utils;

pub use parse::{ParsedFile, ParsedProject, SymbolTable, parse_file, parse_project};
