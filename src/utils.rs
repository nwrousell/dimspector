use petgraph::{Graph, visit::DfsPostOrder};

pub fn indent(s: &str) -> String {
    s.lines()
        .map(|line| format!("  {}", line))
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn write_comma_separated<T: std::fmt::Display>(
    f: &mut std::fmt::Formatter<'_>,
    items: impl IntoIterator<Item = T>,
) -> std::fmt::Result {
    let mut iter = items.into_iter();
    if let Some(first) = iter.next() {
        write!(f, "{}", first)?;
        for item in iter {
            write!(f, ", {}", item)?;
        }
    }
    Ok(())
}

pub fn reverse_post_order<N, E>(
    graph: &Graph<N, E>,
    start: petgraph::graph::NodeIndex,
) -> Vec<petgraph::graph::NodeIndex> {
    let mut dfs = DfsPostOrder::new(graph, start);
    let mut result = Vec::new();
    while let Some(node) = dfs.next(graph) {
        result.push(node);
    }
    result.reverse();
    result
}
