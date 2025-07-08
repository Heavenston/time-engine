use crate::{ default, AutoSmallVec };

use std::{ fmt::{ Debug, Display }, ops::{ Deref, DerefMut, Index }, sync::Arc };

use parking_lot::RwLock;
use itertools::Itertools;

#[cfg(debug_assertions)]
mod graph_id {
    use std::sync::atomic::{AtomicU64, Ordering};

    static GRAPH_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

    #[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
    pub struct GraphId(u64);

    impl GraphId {
        pub fn new() -> Self {
            Self(GRAPH_ID_COUNTER.fetch_add(1, Ordering::Relaxed))
        }
    }
}
#[cfg(not(debug_assertions))]
mod graph_id {
    #[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
    pub struct GraphId(());

    impl GraphId {
        pub fn new() -> Self {
            Self(())
        }
    }
}
use graph_id::*;

pub trait DeltaGraphDataType: Sized + 'static {
    type RootData: Clone + Debug + 'static;
    type PartialData: Clone + Debug + 'static;
    type IntegratedData: 'static + ?Sized;
    type Ctx<'a>;

    fn integrate_root(ctx: &mut Self::Ctx<'_>, graph: &DeltaGraph<Self>, handle: RootNodeHandle, root: &Self::RootData) -> Arc<Self::IntegratedData>;
    fn integrate_partial(ctx: &mut Self::Ctx<'_>, graph: &DeltaGraph<Self>, handle: InnerNodeHandle, running: &Self::IntegratedData, partial: &Self::PartialData) -> Arc<Self::IntegratedData>;
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct NodeHandle {
    graph_id: GraphId,
    idx: usize,
}

impl PartialOrd for NodeHandle {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.graph_id != other.graph_id {
            return None;
        }

        self.idx.partial_cmp(&other.idx)
    }
}

impl Display for NodeHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.idx)
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct RootNodeHandle {
    graph_id: GraphId,
    idx: usize,
}

impl PartialOrd for RootNodeHandle {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.graph_id != other.graph_id {
            return None;
        }

        self.idx.partial_cmp(&other.idx)
    }
}

impl Display for RootNodeHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.idx)
    }
}

impl From<RootNodeHandle> for NodeHandle {
    fn from(val: RootNodeHandle) -> Self {
        NodeHandle {
            graph_id: val.graph_id,
            idx: val.idx,
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct InnerNodeHandle {
    graph_id: GraphId,
    idx: usize,
}

impl PartialOrd for InnerNodeHandle {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.graph_id != other.graph_id {
            return None;
        }

        self.idx.partial_cmp(&other.idx)
    }
}

impl Display for InnerNodeHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.idx)
    }
}

impl From<InnerNodeHandle> for NodeHandle {
    fn from(val: InnerNodeHandle) -> Self {
        NodeHandle {
            graph_id: val.graph_id,
            idx: val.idx,
        }
    }
}

pub trait GenericNode {
    fn previous(&self) -> Option<NodeHandle>;
    fn children(&self) -> &[InnerNodeHandle];
    fn children_mut(&mut self) -> &mut Vec<InnerNodeHandle>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct RootNode<D: DeltaGraphDataType> {
    pub data: D::RootData,
    pub children: Vec<InnerNodeHandle>,
}

impl<D: DeltaGraphDataType> GenericNode for RootNode<D> {
    fn previous(&self) -> Option<NodeHandle> {
        None
    }

    fn children(&self) -> &[InnerNodeHandle] {
        &self.children
    }

    fn children_mut(&mut self) -> &mut Vec<InnerNodeHandle> {
        &mut self.children
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct InnerNode<D: DeltaGraphDataType> {
    pub data: D::PartialData,
    pub previous: NodeHandle,
    pub children: Vec<InnerNodeHandle>,
}

impl<D: DeltaGraphDataType> GenericNode for InnerNode<D> {
    fn previous(&self) -> Option<NodeHandle> {
        Some(self.previous)
    }

    fn children(&self) -> &[InnerNodeHandle] {
        &self.children
    }

    fn children_mut(&mut self) -> &mut Vec<InnerNodeHandle> {
        &mut self.children
    }
}

#[derive(Debug, Clone)]
pub enum Node<D: DeltaGraphDataType> {
    Root(RootNode<D>),
    Inner(InnerNode<D>),
}

impl<D: DeltaGraphDataType> Deref for Node<D> {
    type Target = dyn GenericNode;

    fn deref(&self) -> &Self::Target {
        match self {
            Node::Root(node) => node,
            Node::Inner(node) => node,
        }
    }
}

impl<D: DeltaGraphDataType> DerefMut for Node<D> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Node::Root(node) => node,
            Node::Inner(node) => node,
        }
    }
}

impl<D: DeltaGraphDataType> From<RootNode<D>> for Node<D> {
    fn from(root: RootNode<D>) -> Self {
        Self::Root(root)
    }
}

impl<D: DeltaGraphDataType> From<InnerNode<D>> for Node<D> {
    fn from(inner: InnerNode<D>) -> Self {
        Self::Inner(inner)
    }
}

#[derive(Debug, Clone)]
pub enum NodeRef<'a, D: DeltaGraphDataType> {
    Root {
        root_node: &'a RootNode<D>,
        root_handle: RootNodeHandle,
    },
    Inner {
        inner_node: &'a InnerNode<D>,
        inner_handle: InnerNodeHandle,
    },
}

#[derive(Debug)]
pub struct DeltaGraph<D: DeltaGraphDataType> {
    id: GraphId,
    nodes: Vec<Node<D>>,
    /// Used to efficiently provide the list of all leafs
    /// assumed to be pretty small
    leafs: Vec<NodeHandle>,
    /// Used to cache the integrated version of all nodes
    /// should be pretty dense (nodes quickly get integrated)
    integration_cache: RwLock<Vec<Option<Arc<D::IntegratedData>>>>,
}

impl<D: DeltaGraphDataType> DeltaGraph<D> {
    pub fn new() -> Self {
        Self {
            id: GraphId::new(),
            nodes: default(),
            leafs: vec![],
            integration_cache: RwLock::new(vec![]),
        }
    }

    fn handle(&self, idx: usize) -> NodeHandle {
        NodeHandle {
            idx,
            graph_id: self.id,
        }
    }

    fn root_handle(&self, idx: usize) -> RootNodeHandle {
        debug_assert!(matches!(self.nodes[idx], Node::Root(..)));
        RootNodeHandle {
            idx,
            graph_id: self.id,
        }
    }

    fn inner_handle(&self, idx: usize) -> InnerNodeHandle {
        debug_assert!(matches!(self.nodes[idx], Node::Inner(..)));
        InnerNodeHandle {
            idx,
            graph_id: self.id,
        }
    }

    #[cfg(debug_assertions)]
    fn assert_handle(&self, handle: NodeHandle) {
        debug_assert_eq!(handle.graph_id, self.id, "Using handles from another graph");
        debug_assert!(handle.idx < self.nodes.len(), "Should be impossible (expept by cloning the graph...)");
    }

    #[cfg(not(debug_assertions))]
    fn assert_handle(&self, _handle: NodeHandle) { }

    pub fn get(&self, handle: NodeHandle) -> NodeRef<'_, D> {
        self.assert_handle(handle);
        match &self.nodes[handle.idx] {
            Node::Root(root_node) => NodeRef::Root {
                root_node,
                root_handle: self.root_handle(handle.idx),
            },
            Node::Inner(inner_node) => NodeRef::Inner {
                inner_node,
                inner_handle: self.inner_handle(handle.idx),
            },
        }
    }

    pub fn iter_ancestry(&'_ self, handle: NodeHandle) -> AncestryIterator<'_, D> {
        AncestryIterator::new(self, Some(handle))
    }

    pub fn nodes(&self) -> impl Iterator<Item = (NodeHandle, &'_ Node<D>)> {
        self.nodes.iter().enumerate()
            .map(|(idx, node)| (self.handle(idx), node))
    }

    pub fn leafs(&self) -> &[NodeHandle] {
        &self.leafs
    }

    pub fn insert_root(&mut self, data: D::RootData) -> RootNodeHandle {
        self.nodes.push(RootNode {
            data,
            children: vec![],
        }.into());
        let handle = self.root_handle(self.nodes.len() - 1);
        self.leafs.push(handle.into());
        handle
    }

    pub fn insert(&mut self, data: D::PartialData, previous: impl Into<NodeHandle>) -> InnerNodeHandle {
        let previous = previous.into();
        if let Some((leaf_idx, _)) = self.leafs.iter().find_position(|&&handle_| previous == handle_) {
            self.leafs.remove(leaf_idx);
        }

        self.nodes.push(InnerNode {
            data,
            previous,
            children: vec![],
        }.into());
        let handle = self.inner_handle(self.nodes.len()-1);
        self.leafs.push(handle.into());

        self.nodes[previous.idx].children_mut().push(handle);

        handle
    }

    fn set_integration_cache(&self, idx: usize, val: Arc<D::IntegratedData>) {
        let mut integration_cache = self.integration_cache.write();
        if integration_cache.len() <= idx {
            integration_cache.resize_with(idx + 1, || None);
        }
        integration_cache[idx] = Some(val);
    }

    pub fn integrate<'a, 'b>(&'a self, ctx: &'b mut D::Ctx<'b>, mut handle: NodeHandle) -> Arc<D::IntegratedData> {
        self.assert_handle(handle);

        // iterative implementation of the naive recursive implementation

        let mut stack = AutoSmallVec::<InnerNodeHandle>::new();

        // backward loop (find root/cached integrated parent)
        let mut integrated = loop {
            if let Some(integrated) = self.integration_cache.read().get(handle.idx).map(Option::as_ref).flatten() {
                break Arc::clone(integrated);
            }

            handle = match self.get(handle) {
                NodeRef::Root { root_node, root_handle } => {
                    let integrated = D::integrate_root(
                        ctx,
                        self,
                        root_handle,
                        &root_node.data,
                    );
                    self.set_integration_cache(handle.idx, Arc::clone(&integrated));
                    break integrated;
                },
                NodeRef::Inner { inner_node, inner_handle } => {
                    stack.push(inner_handle);
                    inner_node.previous
                },
            };
        };

        // forward loop (integrate all nodes on the path)
        for current_handle in stack.into_iter().rev() {
            let inner_node = &self[current_handle];
            integrated = D::integrate_partial(
                ctx,
                self,
                current_handle,
                &*integrated,
                &inner_node.data
            );
            self.set_integration_cache(current_handle.idx, Arc::clone(&integrated));
        }

        integrated
    }
}

impl<'a, D: DeltaGraphDataType, I> Index<&'a I> for DeltaGraph<D>
    where I: Clone,
          DeltaGraph<D>: Index<I>
{
    type Output = <DeltaGraph<D> as Index<I>>::Output;

    fn index(&'_ self, index: &'a I) -> &'_ <DeltaGraph<D> as Index<I>>::Output {
        &self[index.clone()]
    }
}

impl<D: DeltaGraphDataType> Index<NodeHandle> for DeltaGraph<D> {
    type Output = Node<D>;

    fn index(&self, handle: NodeHandle) -> &Self::Output {
        self.assert_handle(handle);
        &self.nodes[handle.idx]
    }
}

impl<D: DeltaGraphDataType> Index<RootNodeHandle> for DeltaGraph<D> {
    type Output = RootNode<D>;

    fn index(&self, handle: RootNodeHandle) -> &Self::Output {
        match &self[NodeHandle::from(handle)] {
            Node::Root(root_node) => root_node,
            Node::Inner(_) => unreachable!(),
        }
    }
}

impl<D: DeltaGraphDataType> Index<InnerNodeHandle> for DeltaGraph<D> {
    type Output = InnerNode<D>;

    fn index(&self, handle: InnerNodeHandle) -> &Self::Output {
        match &self[NodeHandle::from(handle)] {
            Node::Inner(inner_node) => inner_node,
            Node::Root(_) => unreachable!(),
        }
    }
}

impl<D: DeltaGraphDataType> Default for DeltaGraph<D> {
    fn default() -> Self {
        Self::new()
    }
}

pub struct AncestryIterator<'a, D: DeltaGraphDataType> {
    graph: &'a DeltaGraph<D>,
    latest: Option<NodeHandle>,
}

impl<'a, D: DeltaGraphDataType> AncestryIterator<'a, D> {
    pub fn new(graph: &'a DeltaGraph<D>, latest: Option<NodeHandle>) -> Self {
        Self {
            graph,
            latest,
        }
    }
}

impl<'a, D: DeltaGraphDataType> Iterator for AncestryIterator<'a, D> {
    type Item = (NodeHandle, &'a Node<D>);

    fn next(&mut self) -> Option<Self::Item> {
        let handle = self.latest?;
        let node = &self.graph[handle];
        self.latest = node.previous();
        Some((handle, node))
    }
}
