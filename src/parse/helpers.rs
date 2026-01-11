use std::path::{Path, PathBuf};

/// Convert file path to module name
/// e.g., /path/to/project/mypackage/utils.py -> "mypackage.utils"
pub fn path_to_module_name(file_path: &Path, project_root: &Path) -> String {
    let rel_path = file_path
        .strip_prefix(project_root)
        .expect("file not in project root");

    let mut parts: Vec<&str> = rel_path
        .parent()
        .iter()
        .flat_map(|p| p.components())
        .filter_map(|c| c.as_os_str().to_str())
        .collect();

    let stem = file_path
        .file_stem()
        .and_then(|s| s.to_str())
        .expect("invalid filename");

    // Don't include "__init__" in module name
    if stem != "__init__" {
        parts.push(stem);
    }

    parts.join(".")
}

/// Convert module name to file path
/// e.g., "mypackage.utils" -> /path/to/project/mypackage/utils.py
/// Tries .py first, then __init__.py
pub fn module_name_to_path(module_name: &str, project_root: &Path) -> Option<PathBuf> {
    let parts: Vec<&str> = module_name.split('.').collect();
    let mut path = project_root.to_path_buf();

    for part in parts {
        path.push(part);
    }

    // Try .py first
    let py_path = path.with_extension("py");
    if py_path.exists() {
        return Some(py_path);
    }

    // Try __init__.py
    let init_path = path.join("__init__.py");
    if init_path.exists() {
        return Some(init_path);
    }

    None
}
