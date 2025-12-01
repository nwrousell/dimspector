mod types;

use std::collections::{HashMap, HashSet};

use petgraph::graph::NodeIndex;
use types::Variable;

use crate::ir::types::{Expr, Identifier, Statement};
use crate::ir::{Function, Program};
use crate::shape_analysis::types::Shape;
use miette::Result;

pub struct FunctionAnalysis {
    // func: StmtFunctionDef,
    // func: Function,
    // TODO: currently just using Hash{Set,Map}s, but would beneifit perhaps
    // from inenvitably using bitsets, if the speedup is worth it
    domain: HashMap<Identifier, HashSet<Variable>>,
}

impl FunctionAnalysis {
    fn new() -> Self {
        Self {
            domain: HashMap::new(),
        }
    }

    fn analyze_stmt(&mut self, stmt: &Statement) -> Result<()> {
        let shape = Shape::from_str("B C D");
        match &stmt.value {
            Expr::Binop {
                left,
                right,
                is_matmul,
            } => {}
            Expr::Call {
                receiver,
                function,
                args,
            } => {}
            Expr::Constant => {}
            Expr::Identifier(id) => {}
        }
        Ok(())
    }

    fn analyze_func(func: &Function) -> Result<FunctionAnalysis> {
        let mut analysis = Self::new();
        let blocks: Vec<_> = func.blocks().collect();
        for block in blocks {
            let block = func.data(block);
            for stmt in block.statements() {
                analysis.analyze_stmt(stmt);
            }
        }
        Ok(analysis)
    }
}

pub fn analyze(prog: Program) -> Result<()> {
    for func in prog.functions {
        // TODO: maybe do some nice caching later for modularity with user's own funcs
        let _ = FunctionAnalysis::analyze_func(&func);
    }
    Ok(())
}
