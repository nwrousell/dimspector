use std::fmt;

use petgraph::visit::{DfsPostOrder, GraphRef, IntoNeighbors, Visitable};

pub fn reverse_post_order<N, G>(g: G, entry: G::NodeId) -> Vec<N>
where
    N: Copy,
    G: GraphRef + Visitable<NodeId = N> + IntoNeighbors<NodeId = N>,
    G::NodeId: PartialEq,
{
    let mut postorder = Vec::new();
    let mut dfs = DfsPostOrder::new(g, entry);
    while let Some(node) = dfs.next(g) {
        postorder.push(node);
    }
    postorder.reverse();
    postorder
}

pub fn write_comma_separated<T: fmt::Display>(
    f: &mut fmt::Formatter,
    items: impl IntoIterator<Item = T>,
) -> fmt::Result {
    for (i, item) in items.into_iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        write!(f, "{}", item)?;
    }
    Ok(())
}

pub fn write_newline_separated<T: fmt::Display>(
    f: &mut fmt::Formatter,
    items: impl IntoIterator<Item = T>,
) -> fmt::Result {
    for (i, item) in items.into_iter().enumerate() {
        if i > 0 {
            write!(f, "\n")?;
        }
        write!(f, "{}", item)?;
    }
    Ok(())
}

pub fn indent(s: impl AsRef<str>) -> String {
    textwrap::indent(s.as_ref(), "  ")
}
