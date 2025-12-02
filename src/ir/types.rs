use petgraph::{
    Direction,
    graph::{DiGraph, NodeIndex},
};
use smallvec::{SmallVec, smallvec};
use torch_infer2::utils;

use rustpython_parser::text_size::TextRange;

use crate::analysis::Variable;


pub struct Program {
    pub functions: Vec<Function>,
}

type Cfg = DiGraph<BasicBlock, ()>;

#[derive(Eq, Hash, PartialEq)]
pub struct Location {
    pub block: NodeIndex,
    pub instr: usize,
}

pub struct Function {
    // param info
    // body features
    pub cfg: Cfg,
    pub params: Vec<(Identifier, Variable)>,
    pub rpo: Vec<NodeIndex>,
}

impl Function {
    pub fn new(cfg: Cfg) -> Function {
        let rpo: Vec<NodeIndex> = utils::reverse_post_order(&cfg, 0.into())
            .into_iter()
            .collect();

        Self { cfg, rpo }
    }

    pub fn predecessors(&self, loc: Location) -> SmallVec<[Location; 2]> {
        if loc.instr == 0 {
            self.cfg
                .neighbors_directed(loc.block.into(), Direction::Incoming)
                .map(|block| {
                    let instr = self.data(block).terminator_index();
                    Location { block, instr }
                })
                .collect()
        } else {
            smallvec![Location {
                block: loc.block,
                instr: loc.instr - 1
            }]
        }
    }

    pub fn blocks(&self) -> impl DoubleEndedIterator<Item = NodeIndex> {
        self.rpo.iter().copied()
    }

    pub fn data(&self, idx: NodeIndex) -> &BasicBlock {
        self.cfg.node_weight(idx).unwrap()
    }
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
    Identifier(Identifier),
}
