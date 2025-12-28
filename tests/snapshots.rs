use std::path::Path;

use anyhow::Result;
use dimspector::{
    analysis::{self, ir_with_inferred_shapes_to_string},
    ast, ir,
};
use walkdir::WalkDir;

fn run_snapshot_test<F>(suffix: &str, process: F) -> Result<()>
where
    F: Fn(&Path) -> Result<String>,
{
    let root = Path::new("tests/programs").canonicalize()?;

    for entry in WalkDir::new(&root) {
        let entry = entry?;

        if !entry.file_type().is_file() {
            continue;
        }

        let path = entry.path();

        let ext = match path.extension() {
            Some(ext) => ext.to_str().unwrap(),
            None => continue,
        };

        if ext != "py" {
            continue;
        }

        let output = process(path)?;
        let base_name = path.file_name().unwrap().to_str().unwrap();
        let snapshot_name = format!("{}.{}", base_name, suffix);
        let snapshot_path = path.parent().unwrap();

        insta::with_settings!({
            snapshot_path => snapshot_path,
            prepend_module_to_snapshot => false,
        }, {
            insta::assert_snapshot!(snapshot_name, output);
        });
    }

    Ok(())
}

fn lower_to_ir_string(path: &Path) -> Result<String> {
    let input = ast::read(path)?;
    let line_index = ast::LineIndex::new(&input.contents);
    let program = ast::parse(&input)?;
    let ir = ir::lower(program, &line_index)?;
    Ok(format!("{}", ir))
}

fn analyze(path: &Path) -> Result<String> {
    let run = || -> Result<String> {
        let input = ast::read(path)?;
        let line_index = ast::LineIndex::new(&input.contents);
        let program = ast::parse(&input)?;
        let ir = ir::lower(program, &line_index)?;

        match analysis::analyze(ir.clone()) {
            Ok(res) => {
                let mut output = String::new();
                for (name, facts) in &res.functions {
                    if let Some(func) = ir.functions.iter().find(|f| f.identifier == *name) {
                        output.push_str(&ir_with_inferred_shapes_to_string(func, facts, None));
                        output.push('\n');
                    }
                }
                Ok(output)
            }
            Err(err) => Ok(format!("analysis error: {err:?}")),
        }
    };

    match run() {
        Ok(out) => Ok(out),
        Err(err) => Ok(format!("error: {err:?}")),
    }
}

#[test]
fn ir_snapshots() -> Result<()> {
    run_snapshot_test("ir", lower_to_ir_string)
}

#[test]
fn analyze_snapshots() -> Result<()> {
    run_snapshot_test("analysis", analyze)
}
