use std::{cmp::Ordering, collections::HashMap};

use crate::{analysis::DimVar, utils};
use itertools::Either;
use miette::{SourceOffset, SourceSpan};
use petgraph::{
    Direction,
    graph::{DiGraph, NodeIndex},
};
use smallvec::{SmallVec, smallvec};

use ruff_python_ast::{CmpOp, Operator};
use ruff_text_size::TextRange;

use tower_lsp::lsp_types::Position;

use crate::analysis::Variable;

#[derive(Clone, Debug)]
pub struct Program {
    pub functions: Vec<Function>,
}

pub type Cfg = DiGraph<BasicBlock, ()>;
pub type PartialCfg = DiGraph<Option<BasicBlock>, ()>;

#[derive(Eq, Hash, PartialEq, Clone, Copy, Debug)]
pub struct Location {
    pub block: BasicBlockIdx,
    pub instr: usize,
}

impl PartialOrd for Location {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Location {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.block.cmp(&other.block) {
            Ordering::Equal => self.instr.cmp(&other.instr),
            ord => ord,
        }
    }
}

impl Location {
    pub const START: Location = Location {
        block: BasicBlockIdx(0),
        instr: 0,
    };
}

#[derive(Clone, Debug)]
pub struct Function {
    pub identifier: Path,
    pub cfg: Cfg,
    pub params: Vec<Parameter>,
    pub returns: Option<Vec<Variable>>,
    pub locations: Vec<Location>,
    pub rpo: Vec<BasicBlockIdx>,
}

#[derive(Clone, Debug)]
pub struct Parameter(pub Path, pub Option<Variable>);

impl Parameter {
    pub fn new(param: Path, annotation: Option<Variable>) -> Self {
        Self(param, annotation)
    }
}

impl Function {
    pub fn new(
        identifier: Path,
        cfg: Cfg,
        params: Vec<(Path, Option<Variable>)>,
        returns: Option<Vec<Variable>>,
    ) -> Function {
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
            returns,
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

#[derive(Clone, Debug)]
pub struct BasicBlock {
    pub statements: Vec<Statement>,
    pub terminator: Terminator,
}

#[derive(Clone, Debug)]
pub enum Terminator {
    Jump(BasicBlockIdx),
    CondJump {
        /// None when we choose not to model the condition (ex. next(iter)) but still
        /// want to retain that this is a conditional jump
        cond: Option<Expr>,

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

#[derive(Clone, Hash, Eq, PartialEq, Debug)]
pub struct Path(Vec<String>);

impl Path {
    pub fn new(path: &[String]) -> Self {
        Self(path.into())
    }

    pub fn to_dot_string(&self) -> String {
        self.0.join(".")
    }

    pub fn parts(&self) -> &[String] {
        &self.0
    }
}

#[derive(Clone, Debug)]
pub struct Statement {
    pub value: Expr,
    pub target: Option<Path>,
    pub range: TextRange,
    pub assign_end: Option<Position>,
}

#[derive(Clone, Debug)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: SourceSpan,
}

pub fn range_to_span(range: TextRange) -> SourceSpan {
    let start = SourceOffset::from(range.start().to_usize());
    let length = range.len().to_usize();
    SourceSpan::new(start, length)
}

impl Expr {
    pub fn binop(left: Expr, right: Expr, op: Binop, range: TextRange) -> Expr {
        Expr {
            kind: ExprKind::Binop {
                left: Box::new(left),
                right: Box::new(right),
                op,
            },
            span: range_to_span(range),
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
            span: range_to_span(range),
        }
    }

    pub fn constant(range: TextRange, constant: Constant) -> Expr {
        Expr {
            kind: ExprKind::Constant(constant),
            span: range_to_span(range),
        }
    }

    pub fn path(path: Path, range: TextRange) -> Expr {
        Expr {
            kind: ExprKind::Path(path),
            span: range_to_span(range),
        }
    }

    pub fn index(range: TextRange, expr: Expr, index: Vec<Either<Expr, Slice>>) -> Expr {
        Expr {
            kind: ExprKind::Index {
                receiver: Box::new(expr),
                index,
            },
            span: range_to_span(range),
        }
    }

    pub fn tuple(elts: Vec<Expr>, range: TextRange) -> Expr {
        Expr {
            kind: ExprKind::Tuple(elts),
            span: range_to_span(range),
        }
    }
}

#[derive(Clone, Debug)]
pub enum Constant {
    None,
    Bool(bool),
    Str(String),
    Int(i64),
    Float(f64),
}

impl Constant {
    pub fn negate_if_num(&self) -> Option<Self> {
        match self {
            Constant::Int(n) => Some(Constant::Int(-n)),
            Constant::Float(n) => Some(Constant::Float(-n)),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Binop {
    Add,
    Sub,
    Div,
    Mult,
    MatMult,
    Mod,
    Pow,
    FloorDiv,

    Eq,
    NotEq,
    Is,
    IsNot,
    In,
    NotIn,
    Lt,
    Lte,
    Gt,
    Gte,
    And,

    LShift,
    RShift,
    BitOr,
    BitXor,
    BitAnd,
}

impl From<CmpOp> for Binop {
    fn from(value: CmpOp) -> Self {
        match value {
            CmpOp::Eq => Binop::Eq,
            CmpOp::NotEq => Binop::NotEq,
            CmpOp::Lt => Binop::Lt,
            CmpOp::LtE => Binop::Lte,
            CmpOp::Gt => Binop::Gt,
            CmpOp::GtE => Binop::Gte,
            CmpOp::Is => Binop::Is,
            CmpOp::IsNot => Binop::IsNot,
            CmpOp::In => Binop::In,
            CmpOp::NotIn => Binop::NotIn,
        }
    }
}

impl From<Operator> for Binop {
    fn from(value: Operator) -> Self {
        match value {
            Operator::Add => Binop::Add,
            Operator::Sub => Binop::Sub,
            Operator::Mult => Binop::Mult,
            Operator::MatMult => Binop::MatMult,
            Operator::Div => Binop::Div,
            Operator::Mod => Binop::Mod,
            Operator::Pow => Binop::Pow,
            Operator::LShift => Binop::LShift,
            Operator::RShift => Binop::RShift,
            Operator::BitOr => Binop::BitOr,
            Operator::BitXor => Binop::BitXor,
            Operator::BitAnd => Binop::BitAnd,
            Operator::FloorDiv => Binop::FloorDiv,
        }
    }
}

#[derive(Clone, Debug)]
pub enum ExprKind {
    Binop {
        left: Box<Expr>,
        right: Box<Expr>,
        op: Binop,
    },
    Call {
        receiver: Option<Path>,
        function: Path,
        pos_args: Vec<Expr>,
        keyword_args: Vec<(String, Expr)>,
    },
    Constant(Constant),
    Path(Path),
    Tuple(Vec<Expr>),
    Index {
        receiver: Box<Expr>,
        index: Vec<Either<Expr, Slice>>,
    },
}

#[derive(Clone, Debug)]
pub struct Slice {
    pub lower: Option<Expr>,
    pub upper: Option<Expr>,
}

#[derive(Clone, Debug)]
pub enum DimRange {
    DimVar(DimVar),
    Range {
        left: Option<DimVar>,
        right: Option<DimVar>,
    },
}
