use anyhow::Result;
use ruff_db::system::{OsSystem, SystemPath, SystemPathBuf};
use ruff_db::{files::system_path_to_file, parsed::parsed_module};
use ruff_python_ast::{StmtClassDef, StmtFunctionDef};
use std::path::Path;
use ty_project::{ProjectDatabase, ProjectMetadata};
use ty_python_semantic::SemanticModel;

pub struct ParsedFile {
    pub db: ProjectDatabase,
    pub semantic_model: Box<SemanticModel<'static>>,
    pub functions: Vec<StmtFunctionDef>,
    pub classes: Vec<StmtClassDef>,
}

pub fn parse_file(path: &Path) -> Result<ParsedFile> {
    let abs_file = std::fs::canonicalize(path)?;
    let file_path = SystemPathBuf::from(abs_file.to_str().unwrap());
    let parent_dir = file_path.parent().unwrap_or(SystemPath::new("/"));

    let system = OsSystem::new(parent_dir);
    let project_metadata = ProjectMetadata::new("single_file".into(), parent_dir.to_path_buf());
    let db = ProjectDatabase::new(project_metadata, system)?;

    let file = system_path_to_file(&db, &file_path)?;
    let semantic_model = SemanticModel::new(&db, file);

    let parsed = parsed_module(&db, file).load(&db);
    let mut functions = Vec::new();
    let mut classes = Vec::new();

    // for now, just grab all the top-level functions and classes
    for stmt in parsed.syntax().body.iter() {
        match stmt {
            ruff_python_ast::Stmt::FunctionDef(f) => functions.push(f.clone()),
            ruff_python_ast::Stmt::ClassDef(c) => classes.push(c.clone()),
            _ => {}
        }
    }

    // Box the semantic_model to avoid lifetime issues
    // This is safe because ProjectDatabase and SemanticModel are designed to work together
    let semantic_model =
        unsafe { std::mem::transmute::<SemanticModel<'_>, SemanticModel<'static>>(semantic_model) };

    Ok(ParsedFile {
        db,
        semantic_model: Box::new(semantic_model),
        functions,
        classes,
    })
}
