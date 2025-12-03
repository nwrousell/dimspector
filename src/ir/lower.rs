use std::collections::{BTreeSet, HashMap, HashSet};

use anyhow::Result;
use itertools::Itertools;
use petgraph::{
    graph::NodeIndex,
    visit::{Dfs, EdgeRef, Walker},
};
use rustpython_parser::{
    ast::{
        Expr as ASTExpr, ExprAttribute, ExprBinOp, ExprCall, ExprCompare, ExprConstant, ExprName,
        ExprSlice, ExprSubscript, ExprTuple, ExprUnaryOp, Keyword, Stmt as ASTStmt, StmtAssign,
        StmtAugAssign, StmtExpr, StmtFor, StmtFunctionDef as ASTFunction, StmtIf, StmtReturn,
        StmtWhile,
    },
    text_size::TextRange,
};

use crate::ir::types::{Binop, Constant, Function};
use crate::{
    analysis::{Shape, Variable},
    ir::types::{BasicBlock, BasicBlockIdx, Cfg, Expr, PartialCfg, Path, Statement, Terminator},
};

pub fn lower_func(func: ASTFunction) -> Result<Function> {
    let mut lowerer = LowerBody::new(func.clone());

    lowerer.lower_func_body(func.body)?;

    // Find all basic blocks reachable from the start.
    let reachable = Dfs::new(&lowerer.graph, lowerer.start_block.into())
        .iter(&lowerer.graph)
        .map(BasicBlockIdx::from)
        .collect::<BTreeSet<_>>();

    // Copy all the reachable blocks into the final CFG.
    let mut cfg = Cfg::new();
    let mut node_map = HashMap::new();
    for block in reachable {
        let block_data = lowerer
            .graph
            .node_weight_mut(NodeIndex::from(block))
            .as_mut()
            .unwrap()
            .take()
            .unwrap();
        let new_node = cfg.add_node(block_data);
        node_map.insert(block, BasicBlockIdx::from(new_node));
    }

    for block_data in cfg.node_weights_mut() {
        block_data.terminator.remap(&node_map);
    }

    for edge in lowerer.graph.edge_references() {
        let src = BasicBlockIdx::from(edge.source());
        let dst = BasicBlockIdx::from(edge.target());
        if let (Some(new_src), Some(new_dst)) = (node_map.get(&src), node_map.get(&dst)) {
            cfg.add_edge((*new_src).into(), (*new_dst).into(), ());
        }
    }

    Ok(Function::new(
        Path::new(&[func.name.to_string()]),
        cfg,
        lowerer.params,
    ))
}

struct LowerBody {
    pub params: Vec<(Path, Option<Variable>)>,
    pub graph: PartialCfg, // might need to turn this into DiGraph<Option<BasicBlock>, ()>
    pub cur_block: Vec<Statement>,
    pub cur_loc: Option<BasicBlockIdx>,
    pub start_block: BasicBlockIdx,

    /// used to distinguish between method calls + function calls
    pub known_paths: HashSet<String>,
}

impl LowerBody {
    fn new(func: ASTFunction) -> Self {
        let mut graph = PartialCfg::new();
        let start_block = BasicBlockIdx::from(graph.add_node(None));
        let mut params = Vec::new();

        let mut known_paths = HashSet::new();

        // populate params
        for arg in func.args.args.clone() {
            let identifier = arg.def.arg;
            let mut ty = Variable::NonTensor;
            if let Some(annotation) = arg.def.annotation {
                if let ASTExpr::Subscript(subscript) = *annotation {
                    if let ASTExpr::Name(name) = *subscript.value {
                        if name.id.as_str() == "T" {
                            if let ASTExpr::Constant(shape_str) = *subscript.slice {
                                let shape_str = shape_str.value.expect_str();
                                let mut shapes = HashSet::new();
                                shapes.insert(Shape::from_str(&shape_str));
                                ty = Variable::Tensor(shapes);
                            }
                        }
                    }
                }
            }

            params.push((Path::new(&[identifier.to_string()]), Some(ty)));
            known_paths.insert(identifier.to_string());
        }

        LowerBody {
            params,
            graph,
            cur_block: Vec::new(),
            cur_loc: Some(start_block),
            start_block,
            known_paths,
        }
    }

    fn lower_func_body(&mut self, body: Vec<ASTStmt>) -> Result<()> {
        self.lower_body(body)?;

        // functions implicitly return None
        if let Some(_) = self.cur_loc {
            self.finish_block(None, Terminator::Return(None));
        }

        Ok(())
    }

    fn lower_body(&mut self, body: Vec<ASTStmt>) -> Result<()> {
        for stmt in body {
            self.lower_statement(stmt)?;
        }
        Ok(())
    }

    fn new_block(&mut self) -> BasicBlockIdx {
        self.graph.add_node(None).into()
    }

    fn finish_block(&mut self, new_block: Option<BasicBlockIdx>, terminator: Terminator) {
        if let Some(cur_loc_inner) = self.cur_loc {
            let statements = self.cur_block.drain(..).collect::<Vec<_>>();
            let cur_block = BasicBlock {
                statements,
                terminator,
            };
            match cur_block.terminator {
                Terminator::Jump(dst) => {
                    self.graph.add_edge(cur_loc_inner.into(), dst.into(), ());
                }
                Terminator::CondJump {
                    true_dst,
                    false_dst,
                    ..
                } => {
                    self.graph
                        .add_edge(cur_loc_inner.into(), true_dst.into(), ());
                    self.graph
                        .add_edge(cur_loc_inner.into(), false_dst.into(), ());
                }
                Terminator::Return(_) => {}
            }
            *self.graph.node_weight_mut(cur_loc_inner.into()).unwrap() = Some(cur_block);
            self.cur_loc = new_block;
        } else {
            unreachable!("finish_block was called with a None cur_loc")
        }
    }

    fn add_statement(&mut self, value: Expr, target: Option<Path>, range: TextRange) {
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
                    let target = self.lower_expr_to_path(target)?;
                    let value = self.lower_expr_to_expr(value)?;
                    self.add_statement(value, Some(target), range);
                }
            }
            ASTStmt::Expr(StmtExpr { value, range }) => {
                let value = self.lower_expr_to_expr(*value)?;
                self.add_statement(value, None, range);
            }

            ASTStmt::While(StmtWhile {
                body, orelse, test, ..
            }) => {
                if orelse.len() > 1 {
                    todo!("handle while else")
                }

                let cond_block = self.new_block();
                let body_block = self.new_block();
                let after_block = self.new_block();

                // jmp to cond block
                let jmp = Terminator::Jump(cond_block);
                self.finish_block(Some(cond_block), jmp);

                // which jumps to body or new block
                let cond = self.lower_expr_to_expr(*test)?;
                let jmp = Terminator::CondJump {
                    cond: Some(cond),
                    true_dst: body_block,
                    false_dst: after_block,
                };
                self.finish_block(Some(body_block), jmp);

                // lower body, jump to cond block
                self.lower_body(body)?;
                let jmp = Terminator::Jump(cond_block);
                match self.cur_loc {
                    Some(_) => {
                        self.finish_block(Some(after_block), jmp);
                    }
                    None => {
                        self.cur_loc = Some(after_block);
                    }
                };
            }
            ASTStmt::If(StmtIf {
                body, orelse, test, ..
            }) => {
                // make then and else blocks
                let then_block = self.new_block();
                let else_block = self.new_block();
                let join_block = self.new_block();

                // cond jump from current to then/else
                let cond = self.lower_expr_to_expr(*test)?;
                let jmp = Terminator::CondJump {
                    cond: Some(cond),
                    true_dst: then_block,
                    false_dst: else_block,
                };
                self.finish_block(Some(then_block), jmp);

                // lower then body
                let jmp = Terminator::Jump(join_block);
                self.lower_body(body)?;
                match self.cur_loc {
                    Some(_) => {
                        self.finish_block(Some(else_block), jmp.clone());
                    }
                    None => self.cur_loc = Some(else_block),
                };

                // lower else body
                self.lower_body(orelse)?;
                match self.cur_loc {
                    Some(_) => {
                        self.finish_block(Some(join_block), jmp);
                    }
                    None => self.cur_loc = Some(join_block),
                };
            }
            ASTStmt::For(StmtFor { body, orelse, .. }) => {
                if orelse.len() > 1 {
                    todo!("handle for else")
                }

                let cond_block = self.new_block();
                let body_block = self.new_block();
                let after_block = self.new_block();

                // jmp to cond block
                let jmp = Terminator::Jump(cond_block);
                self.finish_block(Some(cond_block), jmp);

                // which jumps to body block or after block
                let jmp = Terminator::CondJump {
                    cond: None, // we're not modeling this, as next(iter) is complex
                    true_dst: body_block,
                    false_dst: after_block,
                };
                self.finish_block(Some(body_block), jmp);

                // lower body
                self.lower_body(body)?;
                let jmp = Terminator::Jump(cond_block);
                match self.cur_loc {
                    Some(_) => {
                        self.finish_block(Some(after_block), jmp);
                    }
                    None => {
                        self.cur_loc = Some(after_block);
                    }
                };
            }

            ASTStmt::AugAssign(StmtAugAssign {
                op,
                range,
                target,
                value,
            }) => todo!(),
            ASTStmt::AnnAssign(stmt_ann_assign) => todo!(),
            ASTStmt::Return(StmtReturn { value, .. }) => {
                let value = match value {
                    None => None,
                    Some(expr) => Some(self.lower_expr_to_expr(*expr)?),
                };
                let ret = Terminator::Return(value);
                self.finish_block(None, ret);
            }

            ASTStmt::Delete(stmt_delete) => todo!(),
            ASTStmt::Assert(stmt_assert) => todo!(),
            ASTStmt::Raise(stmt_raise) => todo!(),

            ASTStmt::AsyncFor(stmt_async_for) => todo!(),

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

    fn lower_expr_to_expr(&mut self, expr: ASTExpr) -> Result<Expr> {
        match expr {
            ASTExpr::Name(ExprName { id, range, .. }) => {
                Ok(Expr::path(Path::new(&[id.to_string()]), range))
            }

            ASTExpr::BinOp(ExprBinOp {
                left,
                op,
                right,
                range,
            }) => {
                let left = self.lower_expr_to_expr(*left)?;
                let right = self.lower_expr_to_expr(*right)?;
                Ok(Expr::binop(left, right, Binop::from(op), range))
            }

            ASTExpr::Constant(ExprConstant { range, value, .. }) => {
                let constant = Constant::from(value);
                Ok(Expr::constant(range, constant))
            }

            ASTExpr::Call(ExprCall {
                args,
                func,
                keywords,
                range,
            }) => {
                let pos_args = args
                    .iter()
                    .map(|arg| self.lower_expr_to_expr(arg.clone()))
                    .collect::<Result<Vec<_>, anyhow::Error>>()?;

                let keyword_args = keywords
                    .iter()
                    .map(|Keyword { arg, value, .. }| {
                        let e = self.lower_expr_to_expr(value.clone())?;
                        Ok((
                            arg.clone()
                                .expect("got a keyword argument without a keyword?")
                                .to_string(),
                            e,
                        ))
                    })
                    .collect::<Result<Vec<(String, Expr)>, anyhow::Error>>()?;

                let mut path = self.lower_expr_to_path_inner(*func)?;

                // currently we distinguish between method vs. function call by checking if the prefix is in known_paths.
                // This doesn't catch nested cases, like 'self.foo.bar()' if 'self.foo' hasn't been written to in this function.
                // which is something important to improve upon for object-oriented programming patterns (ex. pl.Lightning or nn.Module patterns),
                // both for the self.foo.bar() pattern, as well as the class_instance.foo.bar() pattern. Seems tricky with Python's dynamism :/
                let prefix_path = Path::new(&path[0..(path.len() - 1)]);
                let classified_as_method_call =
                    self.known_paths.contains(&prefix_path.to_dot_string());

                if classified_as_method_call {
                    Ok(Expr::call(
                        Some(prefix_path),
                        Path::new(&[path.pop().unwrap()]),
                        pos_args,
                        keyword_args,
                        range,
                    ))
                } else {
                    Ok(Expr::call(
                        None,
                        Path::new(&path),
                        pos_args,
                        keyword_args,
                        range,
                    ))
                }
            }
            ASTExpr::Attribute(ExprAttribute {
                attr, value, range, ..
            }) => {
                let mut path = self.lower_expr_to_path_inner(*value)?;
                path.push(attr.to_string());
                Ok(Expr::path(Path::new(&path), range))
            }
            ASTExpr::UnaryOp(ExprUnaryOp { operand, .. }) => self.lower_expr_to_expr(*operand),
            ASTExpr::Compare(ExprCompare {
                comparators,
                left,
                ops,
                range,
                ..
            }) => {
                let left = self.lower_expr_to_expr(*left)?;
                let right = self.lower_expr_to_expr(comparators.first().unwrap().clone())?;
                let mut bool_expr = Expr::binop(
                    left,
                    right,
                    Binop::from(ops.first().unwrap().clone()),
                    range,
                );

                for (cmp, op) in comparators.into_iter().zip_eq(ops).skip(1) {
                    let left = bool_expr;
                    let right = self.lower_expr_to_expr(cmp)?;
                    bool_expr = Expr::binop(left, right, Binop::from(op), range);
                }

                Ok(bool_expr)
            }

            ASTExpr::List(expr_list) => todo!(),
            ASTExpr::Tuple(expr_tuple) => todo!(),
            ASTExpr::Slice(expr_slice) => todo!(),
            ASTExpr::Subscript(expr_subscript) => todo!(),

            ASTExpr::BoolOp(expr_bool_op) => todo!(),
            ASTExpr::NamedExpr(expr_named_expr) => todo!(),
            ASTExpr::Lambda(expr_lambda) => todo!(),
            ASTExpr::IfExp(expr_if_exp) => todo!(),
            ASTExpr::Dict(expr_dict) => todo!(),
            ASTExpr::Set(expr_set) => todo!(),
            ASTExpr::ListComp(expr_list_comp) => todo!(),
            ASTExpr::SetComp(expr_set_comp) => todo!(),
            ASTExpr::DictComp(expr_dict_comp) => todo!(),
            ASTExpr::GeneratorExp(expr_generator_exp) => todo!(),
            ASTExpr::Await(expr_await) => todo!(),
            ASTExpr::Yield(expr_yield) => todo!(),
            ASTExpr::YieldFrom(expr_yield_from) => todo!(),
            ASTExpr::FormattedValue(expr_formatted_value) => todo!(),
            ASTExpr::JoinedStr(expr_joined_str) => todo!(),
            ASTExpr::Starred(expr_starred) => todo!(),
        }
    }

    fn lower_expr_to_path(&mut self, expr: ASTExpr) -> Result<Path> {
        let path = self.lower_expr_to_path_inner(expr)?;
        let path = Path::new(&path);
        self.known_paths.insert(path.to_dot_string());
        Ok(path)
    }

    // forces expr to be interpreted as a path
    fn lower_expr_to_path_inner(&mut self, expr: ASTExpr) -> Result<Vec<String>> {
        match expr {
            ASTExpr::Attribute(ExprAttribute { attr, value, .. }) => {
                let mut path = self.lower_expr_to_path_inner(*value)?;
                path.push(attr.to_string());
                Ok(path)
            }
            ASTExpr::Subscript(ExprSubscript { slice, value, .. }) => {
                let mut path = self.lower_expr_to_path_inner(*value)?;
                let slice_path = self.lower_expr_to_path_inner(*slice)?;
                path.extend(slice_path);
                Ok(path)
            }
            ASTExpr::Name(ExprName { id, .. }) => Ok(vec![id.to_string()]),
            ASTExpr::Call(ExprCall { .. }) => {
                todo!("call as part of path")
            }

            ASTExpr::Slice(ExprSlice { .. }) => todo!("slice as part of path"),

            _ => unreachable!("got something weird as part of a path: {:?}", expr),
        }
    }
}

// we want field sensitivity, at least for tuples
// how to do this with our identifier/path -> Variable mapping?

// all dot paths (self.foo, a.flatten()) is just represented as Attribute chains.
// I probably should untangle the first one into the path 'self.foo' and the second
// into the target 'a', method of 'flatten'
