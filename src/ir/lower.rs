use std::collections::{BTreeSet, HashMap};

use anyhow::Result;
use itertools::{Either, Itertools};
use petgraph::{
    graph::NodeIndex,
    visit::{Dfs, EdgeRef, Walker},
};

use ruff_python_ast::{
    Expr as ASTExpr, ExprAttribute, ExprBinOp, ExprCall, ExprCompare, ExprEllipsisLiteral,
    ExprFString, ExprList, ExprName, ExprNumberLiteral, ExprSlice, ExprStringLiteral,
    ExprSubscript, ExprTuple, ExprUnaryOp, Keyword, Number, Stmt as ASTStmt, StmtAssign,
    StmtAugAssign, StmtClassDef, StmtExpr, StmtFor, StmtFunctionDef, StmtGlobal, StmtIf,
    StmtReturn, StmtWhile, StmtWith, UnaryOp,
};
use ruff_text_size::TextRange;
use std::path::PathBuf;

use crate::{
    analysis::{DimKind, DimVar},
    ir::types::{Binop, Class, Constant, ExprKind, Function, Identifier, Slice, Type, intern},
};
use crate::{
    analysis::{Shape, Variable},
    ir::types::{BasicBlock, BasicBlockIdx, Cfg, Expr, PartialCfg, Statement, Terminator},
};

pub fn lower_class(
    class_def: &StmtClassDef,
    class_names: &std::collections::HashSet<Identifier>,
    file_path: &PathBuf,
    source: &str,
) -> Result<Class> {
    let mut methods = HashMap::new();

    // Process methods: iterate through body to find StmtFunctionDef
    for stmt in &class_def.body {
        if let ASTStmt::FunctionDef(method) = stmt {
            let lowered_method = lower_func(method, class_names, file_path, source)?;
            methods.insert(lowered_method.identifier, lowered_method);
        }
    }

    Ok(Class {
        identifier: intern(class_def.name.as_str()),
        file_path: file_path.clone(),
        methods,
    })
}

pub fn lower_func(
    func: &StmtFunctionDef,
    class_names: &std::collections::HashSet<Identifier>,
    file_path: &PathBuf,
    source: &str,
) -> Result<Function> {
    let mut lowerer = LowerBody::new(func, class_names, source);

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
        intern(func.name.as_str()),
        file_path.clone(),
        cfg,
        lowerer.params,
        lowerer.returns,
    ))
}

struct LowerBody {
    pub params: Vec<(crate::ir::Identifier, Option<Variable>)>,
    pub returns: Option<Vec<Variable>>,
    pub graph: PartialCfg, // might need to turn this into DiGraph<Option<BasicBlock>, ()>
    pub cur_block: Vec<Statement>,
    pub cur_loc: Option<BasicBlockIdx>,
    pub start_block: BasicBlockIdx,

    /// Set of class identifiers for callable type inference heuristic
    pub class_names: std::collections::HashSet<Identifier>,
    /// Mapping from instance identifiers to their class identifiers
    pub instance_identifiers: std::collections::HashMap<Identifier, Identifier>,
    /// Source content for position calculation
    pub source: String,
}

impl LowerBody {
    fn new(
        func: &StmtFunctionDef,
        class_names: &std::collections::HashSet<Identifier>,
        source: &str,
    ) -> Self {
        let mut graph = PartialCfg::new();
        let start_block = BasicBlockIdx::from(graph.add_node(None));
        let mut params = Vec::new();

        // populate params
        for (i, param) in func.parameters.args.iter().enumerate() {
            let identifier = &param.parameter.name;

            // TODO: have new Variable type so that not every non-tensor-annotated param is a DimVar
            let mut ty = Variable::DimVar(DimVar::new(crate::analysis::DimKind::Named(format!(
                "s{}",
                i
            ))));

            if let Some(annotation) = &param.parameter.annotation {
                // Try to extract shape string from annotation (supports both T["shape"] and Float[Tensor, "shape"])
                if let Some(shape_str) = LowerBody::extract_shape_string(annotation) {
                    ty = Variable::Tensor(Shape::from_str(&shape_str));
                } else if let ASTExpr::Subscript(subscript) = annotation.as_ref() {
                    // Handle int["dvar"] format
                    if let ASTExpr::Name(name) = subscript.value.as_ref() {
                        if name.id.as_str() == "int" {
                            if let ASTExpr::StringLiteral(dvar_str) = subscript.slice.as_ref() {
                                let dvar = dvar_str.value.to_str();
                                ty =
                                    Variable::DimVar(DimVar::new(DimKind::Named(dvar.to_string())));
                            }
                        }
                    }
                }
            }

            params.push((intern(identifier.as_str()), Some(ty)));
        }

        // TODO: handle tuple returns + factor out the logic identical from above
        let mut returns = None;
        if let Some(ret_ty) = &func.returns {
            if let Some(shape_str) = LowerBody::extract_shape_string(ret_ty) {
                returns = Some(vec![Variable::Tensor(Shape::from_str(&shape_str))]);
            }
        }

        LowerBody {
            params,
            returns,
            graph,
            cur_block: Vec::new(),
            cur_loc: Some(start_block),
            start_block,
            class_names: class_names.clone(),
            instance_identifiers: std::collections::HashMap::new(),
            source: source.to_string(),
        }
    }

    /// Convert a byte offset to an LSP Position (line, character)
    fn byte_offset_to_position(&self, offset: usize) -> tower_lsp::lsp_types::Position {
        let mut line = 0u32;
        let mut character = 0u32;

        for (idx, ch) in self.source.char_indices() {
            if idx >= offset {
                break;
            }
            if ch == '\n' {
                line += 1;
                character = 0;
            } else {
                character += 1;
            }
        }

        tower_lsp::lsp_types::Position::new(line, character)
    }

    /// Extract shape string from type annotation.
    /// Supports both T["shape"] and Float[Tensor, "shape"] formats.
    fn extract_shape_string(annotation: &ASTExpr) -> Option<String> {
        if let ASTExpr::Subscript(subscript) = annotation {
            // Handle T["shape"] format
            if let ASTExpr::Name(name) = subscript.value.as_ref() {
                if name.id.as_str() == "T" {
                    if let ASTExpr::StringLiteral(shape_str) = subscript.slice.as_ref() {
                        return Some(shape_str.value.to_str().to_string());
                    }
                }
            }
            // Handle Float[Tensor, "shape"] or other jaxtyping formats
            if let ASTExpr::Name(name) = subscript.value.as_ref() {
                // Check for jaxtyping types: Float, Int, Bool, etc.
                let jaxtyping_types = ["Float", "Int", "Bool", "UInt", "Complex", "BFloat16"];
                if jaxtyping_types.contains(&name.id.as_str()) {
                    if let ASTExpr::Tuple(tuple) = subscript.slice.as_ref() {
                        // Tuple should have [Tensor, "shape"]
                        if tuple.elts.len() >= 2 {
                            // First element should be Tensor (or other tensor types)
                            if let ASTExpr::Name(tensor_type) = &tuple.elts[0] {
                                let tensor_types = ["Tensor", "Array", "Shaped"];
                                if tensor_types.contains(&tensor_type.id.as_str()) {
                                    // Second element should be the shape string
                                    if let ASTExpr::StringLiteral(shape_str) = &tuple.elts[1] {
                                        return Some(shape_str.value.to_str().to_string());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Infer callable type by heuristic: if it's in the list of class names, it's a constructor.
    /// If it's an attribute off of an instance identifier, it's a method, else default to function.
    fn infer_callable_type(&self, expr: &ASTExpr) -> Type {
        match expr {
            ASTExpr::Name(ExprName { id, .. }) => {
                let name = intern(id.as_str());
                if self.class_names.contains(&name) {
                    Type::Constructor(name)
                } else {
                    Type::Function
                }
            }
            ASTExpr::Attribute(ExprAttribute { value, attr, .. }) => {
                // Check if the value is an instance identifier (assigned from constructor)
                if let ASTExpr::Name(ExprName { id, .. }) = value.as_ref() {
                    let receiver_id = intern(id.as_str());
                    if let Some(class_id) = self.instance_identifiers.get(&receiver_id) {
                        return Type::Method(*class_id);
                    }
                }
                // Otherwise, it's a function call on some other object
                Type::Function
            }
            _ => Type::Other,
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
        target: Option<Expr>,
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
                    let target_expr = self.lower_expr(target.clone())?;

                    // Check if the value is a constructor call - if so, track the target identifier
                    if let ASTExpr::Call(ExprCall { func, .. }) = &value {
                        if let ASTExpr::Name(ExprName { id, .. }) = func.as_ref() {
                            let func_name = intern(id.as_str());
                            if self.class_names.contains(&func_name) {
                                // This is a constructor call - track the target identifier
                                if let ASTExpr::Name(ExprName { id: target_id, .. }) = &target {
                                    let target_ident = intern(target_id.as_str());
                                    self.instance_identifiers.insert(target_ident, func_name);
                                }
                            }
                        }
                    }

                    let value = self.lower_expr(value)?;
                    let assign_end_byte = target_range.end().to_usize();
                    let assign_end = Some(self.byte_offset_to_position(assign_end_byte));
                    self.add_statement(value, Some(target_expr), range, assign_end);
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
                let target_expr = self.lower_expr(*target.clone())?;
                let value = self.lower_expr(*value)?;
                let range_converted = range;
                let expr = Expr::binop(
                    target_expr.clone(),
                    value.clone(),
                    op.into(),
                    range_converted,
                    Type::Other,
                );
                let assign_end_byte = target_range.end().to_usize();
                let assign_end = Some(self.byte_offset_to_position(assign_end_byte));
                self.add_statement(expr, Some(target_expr), range_converted, assign_end);
            }

            ASTStmt::Expr(StmtExpr { value, range, .. }) => {
                let value = self.lower_expr(*value)?;
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
                let cond = self.lower_expr(*test)?;
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
                let mut current_else_block = self.new_block();
                let join_block = self.new_block();

                let then_block = self.new_block();
                let cond = self.lower_expr(*test)?;
                let jmp = Terminator::CondJump {
                    cond: Some(cond),
                    true_dst: then_block,
                    false_dst: current_else_block,
                };
                self.finish_block(Some(then_block), jmp);

                self.lower_body(&body)?;
                let jmp = Terminator::Jump(join_block);
                match self.cur_loc {
                    Some(_) => {
                        self.finish_block(Some(current_else_block), jmp.clone());
                    }
                    None => self.cur_loc = Some(current_else_block),
                };

                for clause in elif_else_clauses.iter() {
                    if let Some(test) = &clause.test {
                        // elif
                        let elif_block = self.new_block();
                        let next_else_block = self.new_block();
                        let cond = self.lower_expr(test.clone())?;
                        let jmp = Terminator::CondJump {
                            cond: Some(cond),
                            true_dst: elif_block,
                            false_dst: next_else_block,
                        };
                        self.finish_block(Some(elif_block), jmp);

                        self.lower_body(&clause.body)?;
                        let jmp_to_join = Terminator::Jump(join_block);
                        match self.cur_loc {
                            Some(_) => {
                                self.finish_block(Some(next_else_block), jmp_to_join.clone());
                            }
                            None => self.cur_loc = Some(next_else_block),
                        };
                        current_else_block = next_else_block;
                    } else {
                        // plain else
                        self.lower_body(&clause.body)?;
                        let jmp_to_join = Terminator::Jump(join_block);
                        match self.cur_loc {
                            Some(_) => {
                                self.finish_block(Some(join_block), jmp_to_join);
                            }
                            None => self.cur_loc = Some(join_block),
                        };
                    }
                }

                if !elif_else_clauses.is_empty()
                    && elif_else_clauses
                        .last()
                        .map(|c| c.test.is_some())
                        .unwrap_or(false)
                {
                    let jmp_to_join = Terminator::Jump(join_block);
                    match self.cur_loc {
                        Some(_) => {
                            self.finish_block(Some(join_block), jmp_to_join);
                        }
                        None => self.cur_loc = Some(join_block),
                    };
                }
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
                    Some(expr) => Some(self.lower_expr(*expr)?),
                };
                let ret = Terminator::Return(value);
                self.finish_block(None, ret);
            }

            ASTStmt::With(StmtWith {
                body, items, range, ..
            }) => {
                for item in items.iter() {
                    let expr = self.lower_expr(item.context_expr.clone())?;
                    self.add_statement(expr, None, range, None);
                }
                self.lower_body(&body)?
            }

            ASTStmt::Global(StmtGlobal { names, .. }) => {
                // globals not supported right now
            }

            _ => todo!("unhandled statement: {stmt:?}"),
        }

        Ok(())
    }

    fn lower_expr_to_slice(&mut self, expr_slice: ExprSlice) -> Result<Slice> {
        let lower = match &expr_slice.lower {
            Some(expr_lower) => Some(self.lower_expr((**expr_lower).clone())?),
            None => None,
        };
        let upper = match &expr_slice.upper {
            Some(expr_upper) => Some(self.lower_expr((**expr_upper).clone())?),
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
                        _ => Either::Left(self.lower_expr(elt)?),
                    })
                }

                res
            }
            ASTExpr::EllipsisLiteral(_) => {
                todo!("handle ellipsis")
            }
            _ => vec![Either::Left(self.lower_expr(expr)?)],
        })
    }

    fn lower_expr(&mut self, expr: ASTExpr) -> Result<Expr> {
        // For non-call expressions, use Other as the callable type
        let ty = Type::Other;
        match expr {
            ASTExpr::Name(ExprName { id, range, .. }) => {
                Ok(Expr::ident(intern(id.as_str()), range, ty))
            }

            ASTExpr::BinOp(ExprBinOp {
                left,
                op,
                right,
                range,
                ..
            }) => {
                let left = self.lower_expr(*left)?;
                let right = self.lower_expr(*right)?;
                Ok(Expr::binop(left, right, Binop::from(op), range, ty))
            }

            ASTExpr::NumberLiteral(ExprNumberLiteral { range, value, .. }) => {
                let constant = match value {
                    Number::Int(i) => Constant::Int(i.as_i64().unwrap_or(0)),
                    Number::Float(f) => Constant::Float(f),
                    Number::Complex { .. } => Constant::None,
                };
                Ok(Expr::constant(range, constant, ty))
            }
            ASTExpr::StringLiteral(ExprStringLiteral { range, .. }) => {
                Ok(Expr::constant(range, Constant::Str("".to_string()), ty))
            }
            ASTExpr::BooleanLiteral(expr) => {
                Ok(Expr::constant(expr.range, Constant::Bool(expr.value), ty))
            }
            ASTExpr::NoneLiteral(expr) => Ok(Expr::constant(expr.range, Constant::None, ty)),

            ASTExpr::Call(ExprCall {
                arguments,
                func,
                range,
                ..
            }) => {
                let pos_args = arguments
                    .args
                    .iter()
                    .map(|arg| self.lower_expr(arg.clone()))
                    .collect::<Result<Vec<_>, anyhow::Error>>()?;

                let keyword_args = arguments
                    .keywords
                    .iter()
                    .map(|Keyword { arg, value, .. }| {
                        let e = self.lower_expr(value.clone())?;
                        Ok((
                            intern(
                                arg.clone()
                                    .expect("got a keyword argument without a keyword?")
                                    .as_str(),
                            ),
                            e,
                        ))
                    })
                    .collect::<Result<Vec<_>, anyhow::Error>>()?;

                // Infer callable type from the function expression (can be Ident or Attribute chain)
                let callable_ty = self.infer_callable_type(func.as_ref());

                // Lower the function expression (can be Ident or Attribute chain)
                let function_expr = self.lower_expr(*func)?;

                Ok(Expr::call(
                    function_expr,
                    pos_args,
                    keyword_args,
                    range,
                    callable_ty,
                ))
            }
            ASTExpr::Attribute(ExprAttribute {
                attr, value, range, ..
            }) => {
                let value_expr = self.lower_expr(*value)?;
                Ok(Expr::attribute(
                    value_expr,
                    intern(attr.as_str()),
                    range,
                    ty,
                ))
            }
            ASTExpr::UnaryOp(ExprUnaryOp { operand, op, .. }) => {
                let operand = self.lower_expr(*operand)?;
                match (op, &operand.kind) {
                    (UnaryOp::USub, ExprKind::Constant(c)) => Ok(Expr {
                        kind: ExprKind::Constant(c.negate_if_num().unwrap()),
                        ty,
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
                let left = self.lower_expr(*left)?;
                let right = self.lower_expr(comparators.first().unwrap().clone())?;
                let mut bool_expr = Expr::binop(
                    left,
                    right,
                    Binop::from(ops.first().unwrap().clone()),
                    range,
                    ty.clone(),
                );

                for (cmp, op) in comparators.into_iter().zip_eq(ops).skip(1) {
                    let left = bool_expr;
                    let right = self.lower_expr(cmp)?;
                    bool_expr = Expr::binop(left, right, Binop::from(op), range, ty.clone());
                }

                Ok(bool_expr)
            }

            ASTExpr::Tuple(ExprTuple { range, elts, .. }) => {
                let elts = elts
                    .into_iter()
                    .map(|e| self.lower_expr(e))
                    .collect::<Result<Vec<Expr>>>()?;
                Ok(Expr::tuple(elts, range, ty))
            }
            ASTExpr::Subscript(ExprSubscript {
                value,
                slice,
                range,
                ..
            }) => {
                let expr = self.lower_expr(*value)?;
                let index = self.lower_expr_to_index(*slice)?;

                Ok(Expr::index(range, expr, index, ty))
            }

            ASTExpr::FString(ExprFString { range, .. }) => {
                Ok(Expr::constant(range, Constant::None, ty))
            }

            ASTExpr::List(ExprList { elts, range, .. }) => {
                let elts = elts
                    .into_iter()
                    .map(|e| self.lower_expr(e))
                    .collect::<Result<Vec<Expr>>>()?;

                Ok(Expr::tuple(elts, range, ty))
            }

            _ => todo!("unhandled expr: {expr:#?}"),
        }
    }
}
