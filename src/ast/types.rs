use rustpython_parser::ast::StmtFunctionDef;

pub struct Program {
    pub functions: Vec<StmtFunctionDef>,
}
