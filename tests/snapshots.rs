use std::path::Path;

use anyhow::Result;
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
    let input = torch_infer2::ast::read(path)?;
    let program = torch_infer2::ast::parse(&input)?;
    let ir = torch_infer2::ir::lower(program)?;
    Ok(format!("{}", ir))
}

#[test]
fn ir_snapshots() -> Result<()> {
    run_snapshot_test("ir", lower_to_ir_string)
}
