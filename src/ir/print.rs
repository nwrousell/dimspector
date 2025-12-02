use std::fmt::Display;

use crate::ir::{
    Function, Program,
    types::{Annotation, BasicBlock, Expr, Identifier, Statement},
};

impl Display for Program {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for func in &self.functions {
            writeln!(f, "{}", func)?;
        }
        Ok(())
    }
}

impl Display for Function {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Function name placeholder
        write!(f, "f(")?;

        // Print parameters
        for (i, (id, ann)) in self.params.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}: {}", id, ann)?;
        }

        // Return type placeholder
        write!(f, ") -> float:\n\n")?;

        // Print all basic blocks
        for (idx, node_idx) in self.cfg.node_indices().enumerate() {
            let bb = &self.cfg[node_idx];
            write!(f, "  bb{}:\n\n", idx)?;
            for stmt in &bb.stmts {
                write!(f, "    {}\n", stmt)?;
            }
            writeln!(f)?;
        }

        Ok(())
    }
}

impl Display for BasicBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for stmt in &self.stmts {
            write!(f, "    {}\n", stmt)?;
        }
        Ok(())
    }
}

impl Display for Statement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(target) = &self.target {
            write!(f, "{} = {}", target, self.value)
        } else {
            write!(f, "{}", self.value)
        }
    }
}

impl Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expr::Binop {
                left,
                right,
                is_matmul,
            } => {
                let op = if *is_matmul { "@" } else { "+" };
                write!(f, "{} {} {}", left, op, right)
            }
            Expr::Call {
                receiver,
                function,
                args,
            } => {
                if let Some(recv) = receiver {
                    write!(f, "{}.{}(", recv, function)?;
                } else {
                    write!(f, "{}(", function)?;
                }
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            }
            Expr::Constant => write!(f, "const"),
            Expr::Identifier => write!(f, "id"),
        }
    }
}

impl Display for Identifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "id")
    }
}

impl Display for Annotation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "float")
    }
}
