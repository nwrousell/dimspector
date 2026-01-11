use crate::parse::{ParsedFile, ParsedProject};
use std::fmt;

impl fmt::Display for ParsedProject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "ParsedProject (root: {})", self.project_root.display())?;
        writeln!(f, "Found {} files:", self.files.len())?;
        for parsed_file in &self.files {
            writeln!(f, "  - {}", parsed_file.path.display())?;
        }
        Ok(())
    }
}

impl fmt::Display for ParsedFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "ParsedFile: {}", self.path.display())?;
        writeln!(f, "  Functions: {}", self.functions.len())?;
        writeln!(f, "  Classes: {}", self.classes.len())?;
        writeln!(f, "  Imports: {}", self.imports.len())?;
        Ok(())
    }
}
