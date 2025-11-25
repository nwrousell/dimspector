use petgraph::graph::DiGraph;

pub struct Program {
    pub functions: Vec<Function>,
}

pub struct Function {
    cfg: DiGraph<BasicBlock, ()>,
}

pub struct BasicBlock {
    stmts: Vec<Statement>,
}

pub struct Identifier {}

pub struct Statement {
    value: Expr,
    target: Option<Identifier>,
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
