use core::fmt;

use torch_infer2::utils::{indent, write_comma_separated};

use crate::ir::{
    Function, Program,
    types::{BasicBlock, BasicBlockIdx, Expr, ExprKind, Parameter, Path, Statement, Terminator},
};

impl fmt::Display for Program {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for func in &self.functions {
            write!(f, "{func}\n\n")?;
        }
        Ok(())
    }
}

impl fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "def {}(", self.identifier)?;
        write_comma_separated(f, &self.params)?;
        write!(f, "):\n")?;

        // Print basic blocks in reverse post-order
        for idx in self.blocks() {
            let block_idx = BasicBlockIdx::from(idx);
            let block = self.data(idx);
            write!(f, "{}:\n", block_idx)?;
            let block_content = format!("{}", block);
            write!(f, "{}", indent(&block_content))?;
        }

        Ok(())
    }
}

impl fmt::Display for Parameter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)?;
        if let Some(annotation) = &self.1 {
            write!(f, ": {}", annotation)?;
        }
        Ok(())
    }
}

impl fmt::Display for BasicBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Print statements
        for stmt in &self.statements {
            write!(f, "{}\n", stmt)?;
        }

        // Print terminator
        write!(f, "{}", self.terminator)?;

        Ok(())
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            ExprKind::Path(path) => write!(f, "{}", path),
            ExprKind::Constant => write!(f, "?"), // Placeholder for constants
            ExprKind::Binop {
                left,
                right,
                is_matmul,
            } => {
                let op = if *is_matmul { "@" } else { "+" }; // Default to + for non-matmul
                write!(f, "{} {} {}", left, op, right)
            }
            ExprKind::Call {
                receiver,
                function,
                pos_args,
                keyword_args,
            } => {
                if let Some(recv) = receiver {
                    write!(f, "{}.", recv)?;
                }
                write!(f, "{}", function)?;
                write!(f, "(")?;

                let mut first = true;
                for arg in pos_args {
                    if !first {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                    first = false;
                }

                for (key, value) in keyword_args {
                    if !first {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}={}", key, value)?;
                    first = false;
                }

                write!(f, ")")
            }
        }
    }
}

impl fmt::Display for Statement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(target) = &self.target {
            write!(f, "{} = {}", target, self.value)?;
        } else {
            write!(f, "{}", self.value)?;
        }
        Ok(())
    }
}

impl fmt::Display for BasicBlockIdx {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bb{}", self.index())
    }
}

impl fmt::Display for Terminator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Terminator::Return(expr) => {
                write!(f, "return {}", expr)
            }
            Terminator::Jump(dst) => {
                write!(f, "jump bb{}", dst.index())
            }
            Terminator::CondJump {
                cond,
                true_dst,
                false_dst,
            } => {
                write!(
                    f,
                    "if {}: jump bb{} else jump bb{}",
                    cond,
                    true_dst.index(),
                    false_dst.index()
                )
            }
        }
    }
}

impl fmt::Display for Path {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, part) in self.parts().iter().enumerate() {
            if i > 0 {
                write!(f, ".")?;
            }
            write!(f, "{}", part)?;
        }
        Ok(())
    }
}
