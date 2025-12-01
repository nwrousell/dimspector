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
