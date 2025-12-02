use petgraph::{
    Direction,
    graph::{DiGraph, NodeIndex},
};
use smallvec::{SmallVec, smallvec};
use torch_infer2::utils;

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
    cfg: Cfg,
    rpo: Vec<NodeIndex>,
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

pub struct BasicBlock {
    stmts: Vec<Statement>,
}

impl BasicBlock {
    pub fn statements(&self) -> &Vec<Statement> {
        &self.stmts
    }

    fn terminator_index(&self) -> usize {
        self.stmts.len() - 1
    }
}

#[derive(Eq, Hash, PartialEq)]
pub struct Identifier {
    name: String,
}

pub struct Statement {
    pub value: Expr,
    pub target: Option<Identifier>,
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
