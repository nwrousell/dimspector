use petgraph::graph::{DiGraph, NodeIndex};
use torch_infer2::utils;

pub struct Program {
    pub functions: Vec<Function>,
}

type Cfg = DiGraph<BasicBlock, ()>;

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
}

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
