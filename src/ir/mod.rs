use anyhow::{Error, Result};
use petgraph::graph::DiGraph;
use rustpython_parser::{
    ast::{
        Expr as ASTExpr, ExprTuple, Stmt as ASTStmt, StmtAnnAssign, StmtAssign, StmtAugAssign,
        StmtFunctionDef as ASTFunction,
    },
    text_size::TextRange,
};

pub use crate::ir::types::{Function, Program};
use crate::{
    analysis::{Shape, Variable},
    ir::types::{BasicBlock, BasicBlockIdx, CFG, Expr, Identifier, Statement, Terminator},
};

mod print;
mod types;

pub fn lower(program: rustpython_parser::ast::ModModule) -> Result<Program, Error> {
    Ok(Program {
        functions: Vec::new(),
    })
}

fn lower_func(func: ASTFunction) -> Function {
    let lower_body = LowerBody::new(func);

    Function {
        cfg: DiGraph::new(),
    }
}

type PartialCFG = DiGraph<Option<BasicBlock>, ()>;

struct LowerBody {
    pub params: Vec<(Identifier, Variable)>,
    pub graph: PartialCFG, // might need to turn this into DiGraph<Option<BasicBlock>, ()>
    pub cur_block: Vec<Statement>,
    pub cur_loc: BasicBlockIdx,
    pub start_block: BasicBlockIdx,
}

impl LowerBody {
    fn new(func: ASTFunction) -> Self {
        let mut graph = PartialCFG::new();
        let start_block = BasicBlockIdx::from(graph.add_node(None));
        let mut params = Vec::new();

        // populate params
        for arg in func.args.args.clone() {
            let identifier = arg.def.arg;
            let mut ty = Variable::NotTensor;
            if let Some(annotation) = arg.def.annotation {
                if let ASTExpr::Subscript(subscript) = *annotation {
                    if let ASTExpr::Name(name) = *subscript.value {
                        if name.id.as_str() == "T" {
                            if let ASTExpr::Constant(shape_str) = *subscript.slice {
                                let shape_str = shape_str.value.expect_str();
                                ty = Variable::Tensor(Shape::from_str(&shape_str));
                            }
                        }
                    }
                }
            }

            params.push((identifier, ty));
        }

        LowerBody {
            params,
            graph,
            cur_block: Vec::new(),
            cur_loc: start_block,
            start_block,
        }
    }

    fn lower_body(&mut self, body: Vec<ASTStmt>) -> Result<()> {
        for stmt in body {
            self.lower_statement(stmt)?;
        }
        Ok(())
    }

    fn finish_block(&mut self, new_block: BasicBlockIdx, terminator: Terminator) {
        let statements = self.cur_block.drain(..).collect::<Vec<_>>();
        let cur_block = BasicBlock {
            statements,
            terminator,
        };
        match cur_block.terminator {
            Terminator::Jump(dst) => {
                self.graph.add_edge(self.cur_loc.into(), dst.into(), ());
            }
            Terminator::CondJump {
                true_dst,
                false_dst,
                ..
            } => {
                self.graph
                    .add_edge(self.cur_loc.into(), true_dst.into(), ());
                self.graph
                    .add_edge(self.cur_loc.into(), false_dst.into(), ());
            }
            Terminator::Return(_) => {}
        }
        *self.graph.node_weight_mut(self.cur_loc.into()).unwrap() = Some(cur_block);
        self.cur_loc = new_block;
    }

    fn add_statement(&mut self, value: Expr, target: Option<Identifier>, range: TextRange) {
        let stmt = Statement {
            target,
            value,
            range,
        };
        self.cur_block.push(stmt);
    }

    fn lower_statement(&mut self, stmt: ASTStmt) -> Result<()> {
        // if it's an assign, set target and then call handle_expr on value
        // method call/func call without target -> handle_expr, no target

        // if statement
        // - add handle_expr(condition) to finish current block
        // - lower_body on then and else, set up edges to both and from both to a new block

        // while/for loop
        // - add handle_expr(condition) to finish current block
        // - lower_body on body
        // - add edge to body and edge from body to new block

        match stmt {
            ASTStmt::Assign(StmtAssign {
                range,
                targets,
                value,
                ..
            }) => {
                // split targets/value into pairs
                let target_value_pairs = if targets.len() > 1 {
                    if let Some(ExprTuple { elts, .. }) = value.as_tuple_expr() {
                        assert!(elts.len() == targets.len());
                        targets
                            .into_iter()
                            .zip(elts.into_iter())
                            .map(|(t, v)| (t, v.clone()))
                            .collect()
                    } else {
                        return Err(anyhow::anyhow!(
                            "ICE: Assign with multiple targets didn't have tuple value"
                        ));
                    }
                } else {
                    vec![(targets.first().unwrap().clone(), *value)]
                };

                for (target, value) in target_value_pairs {
                    let target = self.lower_expr_to_identifier(target)?;
                    let value = self.lower_expr_to_expr(value)?;
                    self.add_statement(value, Some(target), range);
                }
            }
            ASTStmt::AugAssign(StmtAugAssign {
                op,
                range,
                target,
                value,
            }) => todo!(),
            ASTStmt::AnnAssign(stmt_ann_assign) => todo!(),

            ASTStmt::Delete(stmt_delete) => todo!(),
            ASTStmt::Assert(stmt_assert) => todo!(),
            ASTStmt::Expr(stmt_expr) => todo!(),
            ASTStmt::Raise(stmt_raise) => todo!(),
            ASTStmt::Return(stmt_return) => todo!(),

            ASTStmt::For(stmt_for) => todo!(),
            ASTStmt::AsyncFor(stmt_async_for) => todo!(),
            ASTStmt::While(stmt_while) => todo!(),
            ASTStmt::If(stmt_if) => todo!(),

            ASTStmt::Import(stmt_import) => todo!(),
            ASTStmt::ImportFrom(stmt_import_from) => todo!(),

            ASTStmt::FunctionDef(stmt_function_def) => todo!(),
            ASTStmt::AsyncFunctionDef(stmt_async_function_def) => todo!(),
            ASTStmt::ClassDef(stmt_class_def) => todo!(),
            ASTStmt::TypeAlias(stmt_type_alias) => todo!(),
            ASTStmt::With(stmt_with) => todo!(),
            ASTStmt::AsyncWith(stmt_async_with) => todo!(),
            ASTStmt::Match(stmt_match) => todo!(),
            ASTStmt::Try(stmt_try) => todo!(),
            ASTStmt::TryStar(stmt_try_star) => todo!(),
            ASTStmt::Global(stmt_global) => todo!(),
            ASTStmt::Nonlocal(stmt_nonlocal) => todo!(),
            ASTStmt::Pass(stmt_pass) => todo!(),
            ASTStmt::Break(stmt_break) => todo!(),
            ASTStmt::Continue(stmt_continue) => todo!(),
        }

        Ok(())
    }

    fn lower_expr_to_identifier(&mut self, expr: ASTExpr) -> Result<Identifier> {
        todo!()
    }

    fn lower_expr_to_expr(&mut self, expr: ASTExpr) -> Result<Expr> {
        todo!()
    }
}
