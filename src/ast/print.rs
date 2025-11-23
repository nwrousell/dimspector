use std::fmt::Display;

use crate::ast::types::Program;

impl Display for Program {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for func in self.functions.iter() {
            f.write_fmt(format_args!("{:#?}", func))?;
        }
        Ok(())
    }
}
