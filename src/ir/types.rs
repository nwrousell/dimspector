use std::{cmp::Ordering, collections::HashMap};

use crate::{analysis::DimVar, utils};
use itertools::Either;
use miette::{SourceOffset, SourceSpan};
use petgraph::{
    Direction,
    graph::{DiGraph, NodeIndex},
};
use smallvec::{SmallVec, smallvec};
use std::sync::{LazyLock, Mutex};
use string_interner::{DefaultSymbol, StringInterner, backend::StringBackend};

use ruff_python_ast::{CmpOp, Operator};
use ruff_text_size::TextRange;

use tower_lsp::lsp_types::Position;

use crate::analysis::Variable;

/// Interned string type for identifiers
pub type Identifier = DefaultSymbol;

/// Global string interner for identifiers
static INTERNER: LazyLock<Mutex<StringInterner<StringBackend<DefaultSymbol>>>> =
    LazyLock::new(|| Mutex::new(StringInterner::default()));

/// Intern a string and return its symbol
pub fn intern(s: &str) -> Identifier {
    INTERNER.lock().unwrap().get_or_intern(s)
}

/// Resolve a symbol back to its string
pub fn resolve(sym: Identifier) -> String {
    INTERNER.lock().unwrap().resolve(sym).unwrap().to_string()
}

#[derive(Clone, Debug)]
pub struct Project {
    pub files: Vec<File>,
}

#[derive(Clone, Debug)]
pub struct File {
    pub path: std::path::PathBuf,
    pub functions: Vec<Function>,
    pub classes: Vec<Class>,
}

#[derive(Clone, Debug)]
pub struct Class {
    pub identifier: Identifier,
    pub file_path: std::path::PathBuf,
    pub methods: HashMap<Identifier, Function>,
    /// Whether this class inherits from torch.nn.Module
    pub is_nn_module: bool,
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
    pub identifier: Identifier,
    pub file_path: std::path::PathBuf,
    pub cfg: Cfg,
    pub params: Vec<Parameter>,
    pub returns: Option<Vec<Variable>>,
    pub locations: Vec<Location>,
    pub rpo: Vec<BasicBlockIdx>,
}

#[derive(Clone, Debug)]
pub struct Parameter(pub Identifier, pub Option<Variable>);

impl Parameter {
    pub fn new(param: Identifier, annotation: Option<Variable>) -> Self {
        Self(param, annotation)
    }
}

impl Function {
    pub fn new(
        identifier: Identifier,
        file_path: std::path::PathBuf,
        cfg: Cfg,
        params: Vec<(Identifier, Option<Variable>)>,
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
            file_path,
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

    /// Returns all final locations in the function (locations at Return terminators).
    /// These are the locations where the function terminates, including implicit returns.
    pub fn final_locations(&self) -> Vec<Location> {
        let mut final_locs = Vec::new();
        for block_idx in self.blocks() {
            let block = self.data(block_idx);
            if matches!(block.terminator, Terminator::Return(_)) {
                // The final location is at the terminator position (after all statements)
                final_locs.push(Location {
                    block: block_idx,
                    instr: block.statements.len(),
                });
            }
        }
        final_locs
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

#[derive(Clone, Debug)]
pub struct Statement {
    pub value: Expr,
    pub target: Option<Expr>,
    pub range: TextRange,
    pub assign_end: Option<Position>,
}

#[derive(Clone, Debug)]
pub struct Expr {
    pub kind: ExprKind,
    pub ty: Type,
    pub span: SourceSpan,
}

// Manual implementations of Hash, Eq, PartialEq that exclude span
// (span is only for error reporting, not semantic equality)
impl std::hash::Hash for Expr {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.kind.hash(state);
        self.ty.hash(state);
    }
}

impl PartialEq for Expr {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind && self.ty == other.ty
    }
}

impl Eq for Expr {}

pub fn range_to_span(range: TextRange) -> SourceSpan {
    let start = SourceOffset::from(range.start().to_usize());
    let length = range.len().to_usize();
    SourceSpan::new(start, length)
}

impl Expr {
    pub fn binop(left: Expr, right: Expr, op: Binop, range: TextRange, ty: Type) -> Expr {
        Expr {
            kind: ExprKind::Binop {
                left: Box::new(left),
                right: Box::new(right),
                op,
            },
            ty,
            span: range_to_span(range),
        }
    }

    pub fn ident(name: Identifier, range: TextRange, ty: Type) -> Expr {
        Expr {
            kind: ExprKind::Ident(name),
            ty,
            span: range_to_span(range),
        }
    }

    pub fn attribute(value: Expr, attr: Identifier, range: TextRange, ty: Type) -> Expr {
        Expr {
            kind: ExprKind::Attribute {
                value: Box::new(value),
                attr,
            },
            ty,
            span: range_to_span(range),
        }
    }

    pub fn call(
        function: Expr,
        pos_args: Vec<Expr>,
        keyword_args: Vec<(Identifier, Expr)>,
        range: TextRange,
        ty: Type,
    ) -> Expr {
        Expr {
            kind: ExprKind::Call {
                function: Box::new(function),
                pos_args,
                keyword_args,
            },
            ty,
            span: range_to_span(range),
        }
    }

    pub fn constant(range: TextRange, constant: Constant, ty: Type) -> Expr {
        Expr {
            kind: ExprKind::Constant(constant),
            ty,
            span: range_to_span(range),
        }
    }

    pub fn index(range: TextRange, expr: Expr, index: Vec<Either<Expr, Slice>>, ty: Type) -> Expr {
        Expr {
            kind: ExprKind::Index {
                receiver: Box::new(expr),
                index,
            },
            ty,
            span: range_to_span(range),
        }
    }

    pub fn tuple(elts: Vec<Expr>, range: TextRange, ty: Type) -> Expr {
        Expr {
            kind: ExprKind::Tuple(elts),
            ty,
            span: range_to_span(range),
        }
    }

    pub fn list(elts: Vec<Expr>, range: TextRange, ty: Type) -> Expr {
        Expr {
            kind: ExprKind::List(elts),
            ty,
            span: range_to_span(range),
        }
    }

    pub fn dict(items: Vec<(Expr, Expr)>, range: TextRange, ty: Type) -> Expr {
        Expr {
            kind: ExprKind::Dict(items),
            ty,
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

// Manual implementations for Hash/Eq since f64 doesn't implement these
impl std::hash::Hash for Constant {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Constant::None => 0.hash(state),
            Constant::Bool(b) => {
                1.hash(state);
                b.hash(state);
            }
            Constant::Str(s) => {
                2.hash(state);
                s.hash(state);
            }
            Constant::Int(i) => {
                3.hash(state);
                i.hash(state);
            }
            Constant::Float(f) => {
                4.hash(state);
                f.to_bits().hash(state); // Use bit representation for hashing
            }
        }
    }
}

impl PartialEq for Constant {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Constant::None, Constant::None) => true,
            (Constant::Bool(a), Constant::Bool(b)) => a == b,
            (Constant::Str(a), Constant::Str(b)) => a == b,
            (Constant::Int(a), Constant::Int(b)) => a == b,
            (Constant::Float(a), Constant::Float(b)) => a.to_bits() == b.to_bits(),
            _ => false,
        }
    }
}

impl Eq for Constant {}

impl Constant {
    pub fn negate_if_num(&self) -> Option<Self> {
        match self {
            Constant::Int(n) => Some(Constant::Int(-n)),
            Constant::Float(n) => Some(Constant::Float(-n)),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
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

/// Represents the callable type of an expression.
/// Relies on hacky heuristics to avoid having to implement full type inference right now.
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub enum Type {
    /// Class constructor - the identifier is the class name
    Constructor(Identifier),
    /// Bound method - the class is derived from the receiver at analysis time
    Method,
    /// Regular function (not a method or constructor)
    Function,
    /// Other types or unknown
    Other,
}

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub enum ExprKind {
    Ident(Identifier),
    Attribute {
        value: Box<Expr>,
        attr: Identifier,
    },
    Binop {
        left: Box<Expr>,
        right: Box<Expr>,
        op: Binop,
    },
    Call {
        function: Box<Expr>,
        pos_args: Vec<Expr>,
        keyword_args: Vec<(Identifier, Expr)>,
    },
    Constant(Constant),
    Tuple(Vec<Expr>),
    List(Vec<Expr>),
    Dict(Vec<(Expr, Expr)>),
    Index {
        receiver: Box<Expr>,
        index: Vec<Either<Expr, Slice>>,
    },
}

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
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
