use std::collections::HashMap;

use crate::{analysis::DimVar, utils};
use itertools::Either;
use petgraph::{
    Direction,
    graph::{DiGraph, NodeIndex},
};
use smallvec::{SmallVec, smallvec};

use rustpython_parser::text_size::TextRange;

use crate::analysis::Variable;

pub struct Program {
    pub functions: Vec<Function>,
}

pub type Cfg = DiGraph<BasicBlock, ()>;
pub type PartialCfg = DiGraph<Option<BasicBlock>, ()>;

#[derive(Eq, Hash, PartialEq, Clone, Copy)]
pub struct Location {
    pub block: BasicBlockIdx,
    pub instr: usize,
}

pub struct Function {
    pub identifier: Path,
    pub cfg: Cfg,
    pub params: Vec<Parameter>,
    pub locations: Vec<Location>,
    pub rpo: Vec<BasicBlockIdx>,
}

pub struct Parameter(pub Path, pub Option<Variable>);

impl Parameter {
    pub fn new(param: Path, annotation: Option<Variable>) -> Self {
        Self(param, annotation)
    }
}

impl Function {
    pub fn new(identifier: Path, cfg: Cfg, params: Vec<(Path, Option<Variable>)>) -> Function {
        let rpo: Vec<BasicBlockIdx> = utils::reverse_post_order(&cfg, 0.into())
            .into_iter()
            .map(BasicBlockIdx::from)
            .collect();

        let params = params
            .into_iter()
            .map(|(p, a)| Parameter::new(p, a))
            .collect();

        let locations: Vec<_> = rpo
            .iter()
            .copied()
            .flat_map(|block| {
                let num_instrs = cfg.node_weight(block.into()).unwrap().statements.len() + 1;
                (0..num_instrs).map(move |instr| Location { block, instr })
            })
            .collect();

        Self {
            identifier,
            cfg,
            locations,
            rpo,
            params,
        }
    }

    pub fn predecessors(&self, loc: &Location) -> SmallVec<[Location; 2]> {
        if loc.instr == 0 {
            self.cfg
                .neighbors_directed(loc.block.into(), Direction::Incoming)
                .map(|block| {
                    let instr = self.data(block.into()).statements.len();
                    Location {
                        block: block.into(),
                        instr,
                    }
                })
                .collect()
        } else {
            smallvec![Location {
                block: loc.block,
                instr: loc.instr - 1
            }]
        }
    }

    pub fn blocks(&self) -> impl DoubleEndedIterator<Item = BasicBlockIdx> {
        self.rpo.iter().copied()
    }

    pub fn data(&self, idx: BasicBlockIdx) -> &BasicBlock {
        self.cfg.node_weight(idx.into()).unwrap()
    }

    pub fn instr(&self, loc: &Location) -> Either<&Statement, &Terminator> {
        self.data(loc.block).get(loc.instr)
    }
}

pub struct Annotation;

pub struct BasicBlock {
    pub statements: Vec<Statement>,
    pub terminator: Terminator,
}

#[derive(Clone)]
pub enum Terminator {
    Jump(BasicBlockIdx),
    CondJump {
        cond: Expr,
        true_dst: BasicBlockIdx,
        false_dst: BasicBlockIdx,
    },
    Return(Option<Expr>),
}

impl Terminator {
    /// Remap the basic blocks inside the terminator, used during CFG construction.
    pub fn remap(&mut self, map: &HashMap<BasicBlockIdx, BasicBlockIdx>) {
        match self {
            Terminator::Jump(block) => *block = map[block],
            Terminator::CondJump {
                true_dst,
                false_dst,
                ..
            } => {
                *true_dst = map[true_dst];
                *false_dst = map[false_dst];
            }
            Terminator::Return(..) => {}
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BasicBlockIdx(usize);

impl BasicBlock {
    pub fn get(&self, i: usize) -> Either<&Statement, &Terminator> {
        assert!(i <= self.statements.len());
        if i == self.statements.len() {
            Either::Right(&self.terminator)
        } else {
            Either::Left(&self.statements[i])
        }
    }
}

impl BasicBlockIdx {
    fn new(idx: usize) -> Self {
        BasicBlockIdx(idx)
    }

    pub fn index(&self) -> usize {
        self.0
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

#[derive(Clone, Hash, Eq, PartialEq)]
pub struct Path(Vec<String>);

impl Path {
    pub fn new(path: &[String]) -> Self {
        Self(path.into())
    }

    pub fn to_dot_string(&self) -> String {
        let mut dot_string = self.0.first().unwrap().clone();
        for s in self.0.iter().skip(1) {
            dot_string += s;
        }
        dot_string
    }

    pub fn parts(&self) -> &[String] {
        &self.0
    }
}

pub struct Statement {
    pub value: Expr,
    pub target: Option<Path>,
    pub range: TextRange,
}

#[derive(Clone)]
pub struct Expr {
    pub kind: ExprKind,
    pub range: TextRange,
}

impl Expr {
    pub fn binop(left: Expr, right: Expr, is_matmul: bool, range: TextRange) -> Expr {
        Expr {
            kind: ExprKind::Binop {
                left: Box::new(left),
                right: Box::new(right),
                is_matmul,
            },
            range,
        }
    }

    pub fn call(
        receiver: Option<Path>,
        function: Path,
        pos_args: Vec<Expr>,
        keyword_args: Vec<(String, Expr)>,
        range: TextRange,
    ) -> Expr {
        Expr {
            kind: ExprKind::Call {
                receiver,
                function,
                pos_args,
                keyword_args,
            },
            range,
        }
    }

    pub fn constant(range: TextRange, constant: Constant) -> Expr {
        Expr {
            kind: ExprKind::Constant(constant),
            range,
        }
    }

    pub fn path(path: Path, range: TextRange) -> Expr {
        Expr {
            kind: ExprKind::Path(path),
            range,
        }
    }
}

#[derive(Clone)]
pub enum Constant {
    Bool(bool),
    Str(String),
    Int(i64),
    Tuple(Vec<Constant>),
    Float(f64),
}

#[derive(Clone)]
pub enum ExprKind {
    Binop {
        left: Box<Expr>,
        right: Box<Expr>,
        is_matmul: bool,
    },
    Call {
        receiver: Option<Path>,
        function: Path,
        pos_args: Vec<Expr>,
        keyword_args: Vec<(String, Expr)>,
    },
    Constant(Constant),
    Path(Path),
    Slice {
        receiver: Path,
        slice: Vec<DimRange>,
    },
}

#[derive(Clone)]
pub enum DimRange {
    DimVar(DimVar),
    Range {
        left: Option<DimVar>,
        right: Option<DimVar>,
    },
}
