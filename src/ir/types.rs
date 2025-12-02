use petgraph::graph::{DiGraph, NodeIndex};
use rustpython_parser::text_size::TextRange;

use crate::analysis::Variable;

pub struct Program {
    pub functions: Vec<Function>,
}

pub type CFG = DiGraph<BasicBlock, ()>;

pub struct Function {
    pub cfg: CFG,
    pub params: Vec<(Identifier, Variable)>,
}

pub struct Annotation;

pub struct BasicBlock {
    pub statements: Vec<Statement>,
    pub terminator: Terminator,
}

pub enum Terminator {
    Jump(BasicBlockIdx),
    CondJump {
        cond: Expr,
        true_dst: BasicBlockIdx,
        false_dst: BasicBlockIdx,
    },
    Return(Expr),
}

#[derive(Clone, Copy, Debug)]
pub struct BasicBlockIdx(usize);

impl BasicBlockIdx {
    fn new(idx: usize) -> Self {
        BasicBlockIdx(idx)
    }
}

impl From<NodeIndex> for BasicBlockIdx {
    fn from(value: NodeIndex) -> Self {
        BasicBlockIdx::new(value.index())
    }
}

impl From<BasicBlockIdx> for NodeIndex {
    fn from(value: BasicBlockIdx) -> Self {
        NodeIndex::new(value.0)
    }
}

pub struct Location {
    pub block: BasicBlockIdx,
    pub instr: usize,
}

pub type Identifier = rustpython_parser::ast::Identifier;

pub struct Statement {
    pub value: Expr,
    pub target: Option<Identifier>,
    pub range: TextRange,
}

pub enum Expr {
    Binop {
        left: Box<Expr>,
        right: Box<Expr>,
        is_matmul: bool,
    },
    Call {
        receiver: Option<Identifier>,
        function: Identifier,
        args: Vec<Expr>,
    },
    Constant,
    Identifier,
}
