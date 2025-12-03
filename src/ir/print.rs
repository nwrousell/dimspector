use core::fmt;

use crate::ir::types::{Constant, DimRange, Location};
use crate::utils::{indent, write_comma_separated};

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
            write!(f, "  {}:\n", block_idx)?;
            let block_content = format!("{}", block);
            write!(f, "{}", indent(indent(&block_content)))?;
        }

        Ok(())
    }
}

impl fmt::Display for Location {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}[{}]", self.block, self.instr)
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
        write!(f, "{}\n", self.terminator)?;

        Ok(())
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            ExprKind::Path(path) => write!(f, "{}", path),
            ExprKind::Constant(constant) => write!(f, "{}", constant), // Placeholder for constants
            ExprKind::Slice { receiver, slice } => {
                write!(f, "{}[", receiver)?;
                write_comma_separated(f, slice)?;
                write!(f, "]")
            }
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

impl fmt::Display for DimRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DimRange::DimVar(dim_var) => write!(f, "{}", dim_var),
            DimRange::Range { left, right } => {
                if let Some(left) = left {
                    write!(f, "{}", left)?;
                }
                write!(f, ":")?;
                if let Some(right) = right {
                    write!(f, "{}", right)?;
                }
                Ok(())
            }
        }
    }
}

impl fmt::Display for Constant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Constant::Bool(b) => write!(f, "{}", b),
            Constant::Str(s) => write!(f, "'{}'", s),
            Constant::Int(i) => write!(f, "{}", i),
            Constant::Tuple(constants) => {
                write!(f, "(")?;
                write_comma_separated(f, constants)?;
                write!(f, ")")
            }
            Constant::Float(float) => write!(f, "{}", float),
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
            Terminator::Return(expr) => match expr {
                Some(expr) => write!(f, "return {}", expr),
                None => write!(f, "return"),
            },
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
                    "jmp bb{} if {} else bb{}",
                    true_dst.index(),
                    cond,
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
