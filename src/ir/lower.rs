use std::collections::{BTreeSet, HashMap, HashSet};

use anyhow::Result;
use itertools::{Either, Itertools};
use petgraph::{
    graph::NodeIndex,
    visit::{Dfs, EdgeRef, Walker},
};

use ruff_python_ast::{
    Expr as ASTExpr, ExprAttribute, ExprBinOp, ExprCall, ExprCompare, ExprFString, ExprList,
    ExprName, ExprNumberLiteral, ExprSlice, ExprStringLiteral, ExprSubscript, ExprTuple,
    ExprUnaryOp, Keyword, Number, Stmt as ASTStmt, StmtAssign, StmtAugAssign, StmtExpr, StmtFor,
    StmtFunctionDef, StmtIf, StmtReturn, StmtWhile, StmtWith, UnaryOp,
};
use ruff_text_size::TextRange;
use ty_project::ProjectDatabase;
use ty_python_semantic::SemanticModel;

use crate::{
    analysis::{DimKind, DimVar},
    ir::types::{Binop, Constant, ExprKind, Function, Slice},
};
use crate::{
    analysis::{Shape, Variable},
    ir::types::{BasicBlock, BasicBlockIdx, Cfg, Expr, PartialCfg, Path, Statement, Terminator},
};

pub fn lower_func<'db>(
    func: &StmtFunctionDef,
    db: &'db ProjectDatabase,
    model: &'db SemanticModel<'db>,
) -> Result<Function> {
    let mut lowerer = LowerBody::new(func, db, model);

    lowerer.lower_func_body(&func.body)?;

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
        lowerer.returns,
    ))
}

struct LowerBody<'db> {
    pub params: Vec<(Path, Option<Variable>)>,
    pub returns: Option<Vec<Variable>>,
    pub graph: PartialCfg, // might need to turn this into DiGraph<Option<BasicBlock>, ()>
    pub cur_block: Vec<Statement>,
    pub cur_loc: Option<BasicBlockIdx>,
    pub start_block: BasicBlockIdx,

    /// used to distinguish between method calls + function calls
    pub known_paths: HashSet<String>,

    pub db: &'db ProjectDatabase,
    /// semantic model for type inference
    pub model: &'db SemanticModel<'db>,
}

impl<'db> LowerBody<'db> {
    fn new(
        func: &StmtFunctionDef,
        db: &'db ProjectDatabase,
        model: &'db SemanticModel<'db>,
    ) -> Self {
        let mut graph = PartialCfg::new();
        let start_block = BasicBlockIdx::from(graph.add_node(None));
        let mut params = Vec::new();

        let mut known_paths = HashSet::new();

        // populate params
        for (i, param) in func.parameters.args.iter().enumerate() {
            let identifier = &param.parameter.name;

            // TODO: have new Variable type so that not every non-tensor-annotated param is a DimVar
            let mut ty = Variable::DimVar(DimVar::new(crate::analysis::DimKind::Named(format!(
                "s{}",
                i
            ))));

            if let Some(annotation) = &param.parameter.annotation {
                if let ASTExpr::Subscript(subscript) = annotation.as_ref() {
                    if let ASTExpr::Name(name) = subscript.value.as_ref() {
                        if name.id.as_str() == "T" {
                            if let ASTExpr::StringLiteral(shape_str) = subscript.slice.as_ref() {
                                let shape_str = shape_str.value.to_str();
                                ty = Variable::Tensor(Shape::from_str(shape_str));
                            }
                        } else if name.id.as_str() == "int" {
                            if let ASTExpr::StringLiteral(dvar_str) = subscript.slice.as_ref() {
                                let dvar = dvar_str.value.to_str();
                                ty =
                                    Variable::DimVar(DimVar::new(DimKind::Named(dvar.to_string())));
                            }
                        }
                    }
                }
            }

            params.push((Path::new(&[identifier.to_string()]), Some(ty)));
            known_paths.insert(identifier.to_string());
        }

        // TODO: handle tuple returns + factor out the logic identical from above
        let mut returns = None;
        if let Some(ret_ty) = &func.returns {
            if let ASTExpr::Subscript(subscript) = ret_ty.as_ref() {
                if let ASTExpr::Name(name) = subscript.value.as_ref() {
                    if name.id.as_str() == "T" {
                        if let ASTExpr::StringLiteral(shape_str) = subscript.slice.as_ref() {
                            let shape_str = shape_str.value.to_str();
                            returns = Some(vec![Variable::Tensor(Shape::from_str(shape_str))]);
                        }
                    }
                }
            }
        }

        LowerBody {
            params,
            returns,
            graph,
            cur_block: Vec::new(),
            cur_loc: Some(start_block),
            start_block,
            known_paths,
            db,
            model,
        }
    }

    fn lower_func_body(&mut self, body: &[ASTStmt]) -> Result<()> {
        self.lower_body(body)?;

        // functions implicitly return None
        if let Some(_) = self.cur_loc {
            self.finish_block(None, Terminator::Return(None));
        }

        Ok(())
    }

    fn lower_body(&mut self, body: &[ASTStmt]) -> Result<()> {
        for stmt in body {
            self.lower_statement(stmt.clone())?;
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

    fn add_statement(
        &mut self,
        value: Expr,
        target: Option<Path>,
        range: TextRange,
        assign_end: Option<tower_lsp::lsp_types::Position>,
    ) {
        let stmt = Statement {
            target,
            value,
            range,
            assign_end,
        };
        self.cur_block.push(stmt);
    }

    fn lower_statement(&mut self, stmt: ASTStmt) -> Result<()> {
        match stmt {
            ASTStmt::Assign(StmtAssign {
                range,
                targets,
                value,
                ..
            }) => {
                // ! this doesn't work, as tuple assigns are represented as a tuple and not multiple targets
                // split targets/value into pairs
                let target_value_pairs = if targets.len() > 1 {
                    if let ASTExpr::Tuple(ExprTuple { elts, .. }) = value.as_ref() {
                        assert!(elts.len() == targets.len());
                        targets
                            .iter()
                            .zip(elts.iter())
                            .map(|(t, v)| (t.clone(), v.clone()))
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
                    let target_range = match &target {
                        ASTExpr::Name(ExprName { range, .. }) => *range,
                        ASTExpr::Attribute(ExprAttribute { range, .. }) => *range,
                        ASTExpr::Subscript(ExprSubscript { range, .. }) => *range,
                        _ => range,
                    };
                    let target_path = self.lower_expr_to_path(target)?;
                    let value = self.lower_expr_to_expr(value)?;
                    let assign_end_byte = target_range.end().to_usize();
                    let assign_end = Some(tower_lsp::lsp_types::Position::new(
                        0,
                        assign_end_byte as u32,
                    ));
                    self.add_statement(value, Some(target_path), range, assign_end);
                }
            }
            ASTStmt::AugAssign(StmtAugAssign {
                op,
                range,
                target,
                value,
                ..
            }) => {
                let target_range = match target.as_ref() {
                    ASTExpr::Name(ExprName { range, .. }) => *range,
                    ASTExpr::Attribute(ExprAttribute { range, .. }) => *range,
                    ASTExpr::Subscript(ExprSubscript { range, .. }) => *range,
                    _ => range,
                };
                let target_path = self.lower_expr_to_path(*target)?;
                let value = self.lower_expr_to_expr(*value)?;
                let range_converted = range;
                let expr = Expr::binop(
                    Expr::path(target_path.clone(), range_converted),
                    value.clone(),
                    op.into(),
                    range_converted,
                );
                let assign_end_byte = target_range.end().to_usize();
                let assign_end = Some(tower_lsp::lsp_types::Position::new(
                    0,
                    assign_end_byte as u32,
                ));
                self.add_statement(expr, Some(target_path), range_converted, assign_end);
            }

            ASTStmt::Expr(StmtExpr { value, range, .. }) => {
                let value = self.lower_expr_to_expr(*value)?;
                self.add_statement(value, None, range, None);
            }

            ASTStmt::While(StmtWhile {
                body, orelse, test, ..
            }) => {
                if !orelse.is_empty() {
                    todo!("handle while loop else")
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
                self.lower_body(&body)?;
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
                body,
                test,
                elif_else_clauses,
                ..
            }) => {
                todo!(
                    "re-implement lowering for if statements (ruff's AST has an arbitrary number of elif clauses)"
                )
                // let mut current_else_block = self.new_block();
                // let join_block = self.new_block();

                // let then_block = self.new_block();
                // let cond = self.lower_expr_to_expr(*test)?;
                // let jmp = Terminator::CondJump {
                //     cond: Some(cond),
                //     true_dst: then_block,
                //     false_dst: current_else_block,
                // };
                // self.finish_block(Some(then_block), jmp);

                // self.lower_body(&body)?;
                // let jmp = Terminator::Jump(join_block);
                // match self.cur_loc {
                //     Some(_) => {
                //         self.finish_block(Some(current_else_block), jmp.clone());
                //     }
                //     None => self.cur_loc = Some(current_else_block),
                // };

                // for clause in elif_else_clauses.iter() {
                //     if let Some(test) = &clause.test {
                //         let elif_block = self.new_block();
                //         let next_else_block = self.new_block();
                //         let cond = self.lower_expr_to_expr(test.clone())?;
                //         let jmp = Terminator::CondJump {
                //             cond: Some(cond),
                //             true_dst: elif_block,
                //             false_dst: next_else_block,
                //         };
                //         self.finish_block(Some(elif_block), jmp);

                //         self.lower_body(&clause.body)?;
                //         let jmp_to_join = Terminator::Jump(join_block);
                //         match self.cur_loc {
                //             Some(_) => {
                //                 self.finish_block(Some(next_else_block), jmp_to_join.clone());
                //             }
                //             None => self.cur_loc = Some(next_else_block),
                //         };
                //         current_else_block = next_else_block;
                //     } else {
                //         self.lower_body(&clause.body)?;
                //         let jmp_to_join = Terminator::Jump(join_block);
                //         match self.cur_loc {
                //             Some(_) => {
                //                 self.finish_block(Some(join_block), jmp_to_join);
                //             }
                //             None => self.cur_loc = Some(join_block),
                //         };
                //     }
                // }

                // if !elif_else_clauses.is_empty()
                //     && elif_else_clauses
                //         .last()
                //         .map(|c| c.test.is_some())
                //         .unwrap_or(false)
                // {
                //     let jmp_to_join = Terminator::Jump(join_block);
                //     match self.cur_loc {
                //         Some(_) => {
                //             self.finish_block(Some(join_block), jmp_to_join);
                //         }
                //         None => self.cur_loc = Some(join_block),
                //     };
                // }
            }
            ASTStmt::For(StmtFor { body, orelse, .. }) => {
                if !orelse.is_empty() {
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
                self.lower_body(&body)?;
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

            ASTStmt::Return(StmtReturn { value, .. }) => {
                let value = match value {
                    None => None,
                    Some(expr) => Some(self.lower_expr_to_expr(*expr)?),
                };
                let ret = Terminator::Return(value);
                self.finish_block(None, ret);
            }

            ASTStmt::With(StmtWith {
                body, items, range, ..
            }) => {
                let range_converted = range;
                for item in items.iter() {
                    let expr = self.lower_expr_to_expr(item.context_expr.clone())?;
                    self.add_statement(expr, None, range, None);
                }
                self.lower_body(&body)?
            }

            _ => todo!("unhandled statement: {stmt:?}"),
        }

        Ok(())
    }

    fn lower_expr_to_slice(&mut self, expr_slice: ExprSlice) -> Result<Slice> {
        let lower = match &expr_slice.lower {
            Some(expr_lower) => Some(self.lower_expr_to_expr((**expr_lower).clone())?),
            None => None,
        };
        let upper = match &expr_slice.upper {
            Some(expr_upper) => Some(self.lower_expr_to_expr((**expr_upper).clone())?),
            None => None,
        };

        Ok(Slice { lower, upper })
    }

    fn lower_expr_to_index(&mut self, expr: ASTExpr) -> Result<Vec<Either<Expr, Slice>>> {
        Ok(match expr {
            ASTExpr::Slice(expr_slice) => {
                vec![Either::Right(self.lower_expr_to_slice(expr_slice)?)]
            }
            ASTExpr::Tuple(expr_tuple) => {
                let mut res = Vec::new();
                for elt in expr_tuple.elts {
                    res.push(match elt {
                        ASTExpr::Slice(expr_slice) => {
                            Either::Right(self.lower_expr_to_slice(expr_slice)?)
                        }
                        _ => Either::Left(self.lower_expr_to_expr(elt)?),
                    })
                }

                res
            }
            _ => vec![Either::Left(self.lower_expr_to_expr(expr)?)],
        })
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
                ..
            }) => {
                let left = self.lower_expr_to_expr(*left)?;
                let right = self.lower_expr_to_expr(*right)?;
                Ok(Expr::binop(left, right, Binop::from(op), range))
            }

            ASTExpr::NumberLiteral(ExprNumberLiteral { range, value, .. }) => {
                let constant = match value {
                    Number::Int(i) => Constant::Int(i.as_i64().unwrap_or(0)),
                    Number::Float(f) => Constant::Float(f),
                    Number::Complex { .. } => Constant::None,
                };
                Ok(Expr::constant(range, constant))
            }
            ASTExpr::StringLiteral(ExprStringLiteral { range, .. }) => {
                Ok(Expr::constant(range, Constant::Str("".to_string())))
            }
            ASTExpr::BooleanLiteral(expr) => {
                Ok(Expr::constant(expr.range, Constant::Bool(expr.value)))
            }
            ASTExpr::NoneLiteral(expr) => Ok(Expr::constant(expr.range, Constant::None)),

            ASTExpr::Call(ExprCall {
                arguments,
                func,
                range,
                ..
            }) => {
                let pos_args = arguments
                    .args
                    .iter()
                    .map(|arg| self.lower_expr_to_expr(arg.clone()))
                    .collect::<Result<Vec<_>, anyhow::Error>>()?;

                let keyword_args = arguments
                    .keywords
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
            ASTExpr::UnaryOp(ExprUnaryOp { operand, op, .. }) => {
                let operand = self.lower_expr_to_expr(*operand)?;
                match (op, &operand.kind) {
                    (UnaryOp::USub, ExprKind::Constant(c)) => Ok(Expr {
                        kind: ExprKind::Constant(c.negate_if_num().unwrap()),
                        span: operand.span,
                    }),
                    _ => Ok(operand),
                }
            }
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

            ASTExpr::Tuple(ExprTuple { range, elts, .. }) => {
                let elts = elts
                    .into_iter()
                    .map(|e| self.lower_expr_to_expr(e))
                    .collect::<Result<Vec<Expr>>>()?;
                Ok(Expr::tuple(elts, range))
            }
            ASTExpr::Subscript(ExprSubscript {
                value,
                slice,
                range,
                ..
            }) => {
                let expr = self.lower_expr_to_expr(*value)?;
                let index = self.lower_expr_to_index(*slice)?;

                Ok(Expr::index(range, expr, index))
            }

            ASTExpr::FString(ExprFString { range, .. }) => {
                Ok(Expr::constant(range, Constant::None))
            }

            ASTExpr::List(ExprList { elts, range, .. }) => {
                let elts = elts
                    .into_iter()
                    .map(|e| self.lower_expr_to_expr(e))
                    .collect::<Result<Vec<Expr>>>()?;

                Ok(Expr::tuple(elts, range))
            }

            _ => todo!("unhandled expr: {expr:#?}"),
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
            ASTExpr::Call(ExprCall { func, .. }) => {
                let path = self.lower_expr_to_path_inner(*func)?;
                Ok(path)
            }

            ASTExpr::Slice(ExprSlice { .. }) => todo!("slice as part of path"),

            _ => unreachable!("got something weird as part of a path: {:#?}", expr),
        }
    }
}
