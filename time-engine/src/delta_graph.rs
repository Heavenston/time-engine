use crate::AutoSmallVec;

use std::{ fmt::Debug, ops::{ Deref, Index }, sync::Arc };

use parking_lot::RwLock;
use itertools::{ Itertools, chain };

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

mod node_handle {
    use super::GraphId;

    use std::fmt::{ Debug, Display };

    /// Bitmask to extract the actual index from the handle value
    const IDX_MASK: usize = !0 >> 1;
    /// If this bit is set to 1 then the node is an inner node
    const INNER_BIT: usize = !IDX_MASK;

    #[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
    pub struct NodeHandle {
        graph_id: GraphId,
        value: usize,
    }

    impl NodeHandle {
        pub fn idx(self) -> usize {
            self.value & IDX_MASK
        }

        pub(super) fn graph_id(self) -> GraphId {
            self.graph_id
        }

        pub fn is_inner(self) -> bool {
            (self.value & INNER_BIT) != 0
        }

        pub fn is_root(self) -> bool {
            !self.is_inner()
        }

        pub fn either(self) -> EitherHandle {
            self.into()
        }
    }

    impl PartialOrd for NodeHandle {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            if self.graph_id != other.graph_id {
                return None;
            }

            // This should place inner nodes after root nodes
            self.value.partial_cmp(&other.value)
        }
    }

    impl Display for NodeHandle {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.idx())?;
            if self.is_inner() {
                write!(f, "i")?;
            }
            else /* self.is_root() */ {
                write!(f, "r")?;
            }

            Ok(())
        }
    }

    #[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
    pub struct RootNodeHandle {
        handle: NodeHandle,
    }

    impl RootNodeHandle {
        pub(super) fn new(graph_id: GraphId, idx: usize) -> Self {
            assert_eq!(idx, idx & IDX_MASK, "Index overflow!");
            Self {
                handle: NodeHandle {
                    graph_id,
                    value: idx,
                },
            }
        }

        pub fn idx(self) -> usize {
            self.handle.idx()
        }
    }

    impl PartialOrd for RootNodeHandle {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            self.handle.partial_cmp(&other.handle)
        }
    }

    impl Display for RootNodeHandle {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.handle)
        }
    }

    impl From<RootNodeHandle> for NodeHandle {
        fn from(val: RootNodeHandle) -> Self {
            val.handle
        }
    }

    impl TryFrom<NodeHandle> for RootNodeHandle {
        type Error = ();

        fn try_from(handle: NodeHandle) -> Result<Self, Self::Error> {
            if handle.is_root() {
                Ok(Self { handle })
            }
            else {
                Err(())
            }
        }
    }

    #[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
    pub struct InnerNodeHandle {
        handle: NodeHandle,
    }

    impl InnerNodeHandle {
        pub(super) fn new(graph_id: GraphId, idx: usize) -> Self {
            assert_eq!(idx, idx & IDX_MASK, "Index overflow!");
            Self {
                handle: NodeHandle {
                    graph_id,
                    value: idx | INNER_BIT,
                },
            }
        }

        pub fn idx(self) -> usize {
            self.handle.idx()
        }
    }

    impl PartialOrd for InnerNodeHandle {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            self.handle.partial_cmp(&other.handle)
        }
    }

    impl Display for InnerNodeHandle {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.handle)
        }
    }

    impl From<InnerNodeHandle> for NodeHandle {
        fn from(val: InnerNodeHandle) -> Self {
            val.handle
        }
    }

    impl TryFrom<NodeHandle> for InnerNodeHandle {
        type Error = ();

        fn try_from(handle: NodeHandle) -> Result<Self, Self::Error> {
            if handle.is_inner() {
                Ok(Self { handle })
            }
            else {
                Err(())
            }
        }
    }

    #[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
    pub enum EitherHandle {
        Root(RootNodeHandle),
        Inner(InnerNodeHandle),
    }
    
    impl EitherHandle {
        pub fn as_root(self) -> Option<RootNodeHandle> {
            match self {
                Self::Root(handle) => Some(handle),
                _ => None,
            }
        }

        pub fn as_inner(self) -> Option<InnerNodeHandle> {
            match self {
                Self::Inner(handle) => Some(handle),
                _ => None,
            }
        }
    }

    impl From<NodeHandle> for EitherHandle {
        fn from(handle: NodeHandle) -> Self {
            if handle.is_root() {
                EitherHandle::Root(RootNodeHandle { handle })
            }
            else /* self.is_inner() */ {
                EitherHandle::Inner(InnerNodeHandle { handle })
            }
        }
    }

    impl From<RootNodeHandle> for EitherHandle {
        fn from(handle: RootNodeHandle) -> Self {
            EitherHandle::Root(handle)
        }
    }

    impl From<InnerNodeHandle> for EitherHandle {
        fn from(handle: InnerNodeHandle) -> Self {
            EitherHandle::Inner(handle)
        }
    }
}
pub use node_handle::*;

pub trait DeltaGraphDataType: Sized + 'static {
    type RootData: Clone + Debug + 'static;
    type PartialData: Clone + Debug + 'static;
    type IntegratedData: 'static + ?Sized;
    type Ctx<'a>;

    fn integrate_root(ctx: &mut Self::Ctx<'_>, graph: &DeltaGraph<Self>, handle: RootNodeHandle, root: &Self::RootData) -> Arc<Self::IntegratedData>;
    fn integrate_partial(ctx: &mut Self::Ctx<'_>, graph: &DeltaGraph<Self>, handle: InnerNodeHandle, running: &Self::IntegratedData, partial: &Self::PartialData) -> Arc<Self::IntegratedData>;
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

#[derive(Debug, Clone, Copy)]
pub struct RootNodeRef<'a, D: DeltaGraphDataType> {
    root_node: &'a RootNode<D>,
    root_handle: RootNodeHandle,
}

impl<'a, D: DeltaGraphDataType> Deref for RootNodeRef<'a, D> {
    type Target = RootNode<D>;

    fn deref(&self) -> &Self::Target {
        self.root_node
    }
}

#[derive(Debug, Clone, Copy)]
pub struct InnerNodeRef<'a, D: DeltaGraphDataType> {
    inner_node: &'a InnerNode<D>,
    inner_handle: InnerNodeHandle,
}

impl<'a, D: DeltaGraphDataType> Deref for InnerNodeRef<'a, D> {
    type Target = InnerNode<D>;

    fn deref(&self) -> &Self::Target {
        self.inner_node
    }
}

#[derive(Debug, Clone, Copy)]
pub enum NodeRef<'a, D: DeltaGraphDataType> {
    Root(RootNodeRef<'a, D>),
    Inner(InnerNodeRef<'a, D>),
}

impl<'a, D: DeltaGraphDataType> NodeRef<'a, D> {
    pub fn handle(&self) -> NodeHandle {
        match self {
            NodeRef::Root(ref_) => ref_.root_handle.into(),
            NodeRef::Inner(ref_) => ref_.inner_handle.into(),
        }
    }
}

impl<'a, D: DeltaGraphDataType> Deref for NodeRef<'a, D> {
    type Target = dyn GenericNode;

    fn deref(&self) -> &Self::Target {
        match self {
            NodeRef::Root(ref_) => ref_.root_node,
            NodeRef::Inner(ref_) => ref_.inner_node,
        }
    }
}

mod node_cache {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct DenseNodeCache<T> {
        root_values: Vec<Option<T>>,
        inner_values: Vec<Option<T>>,
    }

    impl<T> DenseNodeCache<T> {
        pub fn new() -> Self {
            Self {
                root_values: vec![],
                inner_values: vec![],
            }
        }

        fn array_of(&self, handle: NodeHandle) -> &Vec<Option<T>> {
            if handle.is_root() {
                &self.root_values
            }
            else /* handle.is_inner() */ {
                &self.inner_values
            }
        }

        fn array_of_mut(&mut self, handle: NodeHandle) -> &mut Vec<Option<T>> {
            if handle.is_root() {
                &mut self.root_values
            }
            else /* handle.is_inner() */ {
                &mut self.inner_values
            }
        }

        pub fn get(&self, handle: NodeHandle) -> Option<&T> {
            let array = self.array_of(handle);
            array.get(handle.idx()).map(Option::as_ref).flatten()
        }

        pub fn set(&mut self, handle: NodeHandle, value: T) -> &T {
            let array = self.array_of_mut(handle);
            if array.len() <= handle.idx() {
                array.resize_with(handle.idx() + 1, || None);
            }

            array[handle.idx()] = Some(value);
            array[handle.idx()].as_ref().expect("Present")
        }
    }

    impl<T> Default for DenseNodeCache<T> {
        fn default() -> Self {
            Self::new()
        }
    }
}
pub use node_cache::*;

#[derive(Debug)]
pub struct DeltaGraph<D: DeltaGraphDataType> {
    id: GraphId,
    root_nodes: Vec<RootNode<D>>,
    inner_nodes: Vec<InnerNode<D>>,
    /// Used to efficiently provide the list of all leafs
    /// assumed to be pretty small
    leafs: Vec<NodeHandle>,
    /// Used to cache the integrated version of all nodes
    /// should be pretty dense (nodes quickly get integrated)
    integration_cache: RwLock<DenseNodeCache<Arc<D::IntegratedData>>>,
}

impl<D: DeltaGraphDataType> DeltaGraph<D> {
    pub fn new() -> Self {
        Self {
            id: GraphId::new(),
            root_nodes: vec![],
            inner_nodes: vec![],
            leafs: vec![],
            integration_cache: RwLock::new(DenseNodeCache::new()),
        }
    }

    fn root_handle(&self, idx: usize) -> RootNodeHandle {
        RootNodeHandle::new(self.id, idx)
    }

    fn inner_handle(&self, idx: usize) -> InnerNodeHandle {
        InnerNodeHandle::new(self.id, idx)
    }

    #[cfg(debug_assertions)]
    fn assert_handle(&self, handle: NodeHandle) {
        debug_assert_eq!(handle.graph_id(), self.id, "Using handles from another graph");
        if handle.is_root() {
            debug_assert!(handle.idx() < self.root_nodes.len(), "Should be impossible (expept by cloning the graph...)");
        }
        else {
            debug_assert!(handle.idx() < self.inner_nodes.len(), "Should be impossible (expept by cloning the graph...)");
        }
    }

    #[cfg(not(debug_assertions))]
    fn assert_handle(&self, _handle: NodeHandle) { }

    pub fn get(&self, handle: NodeHandle) -> NodeRef<'_, D> {
        self.assert_handle(handle);
        match handle.either() {
            EitherHandle::Root(root_handle) => {
                NodeRef::Root(RootNodeRef {
                    root_node: &self.root_nodes[handle.idx()],
                    root_handle,
                })
            },
            EitherHandle::Inner(inner_handle) => {
                NodeRef::Inner(InnerNodeRef {
                    inner_node: &self.inner_nodes[handle.idx()],
                    inner_handle,
                })
            },
        }
    }

    /// PRIVATE because manually mutating nodes is not part of the public api
    fn get_mut(&mut self, handle: NodeHandle) -> &mut dyn GenericNode {
        if handle.is_root() {
            &mut self.root_nodes[handle.idx()]
        }
        else /* handle.is_inner() */ {
            &mut self.inner_nodes[handle.idx()]
        }
    }

    pub fn iter_ancestry(&'_ self, handle: NodeHandle) -> AncestryIterator<'_, D> {
        AncestryIterator::new(self, Some(handle))
    }

    pub fn root_nodes(&self) -> impl Iterator<Item = RootNodeRef<'_, D>> {
        self.root_nodes.iter().enumerate()
            .map(|(idx, root_node)| RootNodeRef {
                root_node,
                root_handle: self.root_handle(idx),
            })
    }

    pub fn inner_nodes(&self) -> impl Iterator<Item = InnerNodeRef<'_, D>> {
        self.inner_nodes.iter().enumerate()
            .map(|(idx, inner_node)| InnerNodeRef {
                inner_node,
                inner_handle: self.inner_handle(idx),
            })
    }

    /// Iterate over all nodes, sorted by their handles
    pub fn nodes(&self) -> impl Iterator<Item = NodeRef<'_, D>> {
        chain(
            self.root_nodes().map(NodeRef::Root),
            self.inner_nodes().map(NodeRef::Inner),
        )
    }

    pub fn leafs(&self) -> &[NodeHandle] {
        &self.leafs
    }

    pub fn insert_root(&mut self, data: D::RootData) -> RootNodeHandle {
        self.root_nodes.push(RootNode {
            data,
            children: vec![],
        }.into());
        let handle = self.root_handle(self.root_nodes.len() - 1);
        self.leafs.push(handle.into());
        handle
    }

    pub fn insert(&mut self, data: D::PartialData, previous: impl Into<NodeHandle>) -> InnerNodeHandle {
        let previous = previous.into();
        if let Some((leaf_idx, _)) = self.leafs.iter().find_position(|&&handle_| previous == handle_) {
            self.leafs.remove(leaf_idx);
        }

        self.inner_nodes.push(InnerNode {
            data,
            previous,
            children: vec![],
        }.into());
        let handle = self.inner_handle(self.inner_nodes.len()-1);
        self.leafs.push(handle.into());

        self.get_mut(previous).children_mut().push(handle);

        handle
    }

    pub fn integrate<'a, 'b>(&'a self, ctx: &'b mut D::Ctx<'b>, mut handle: NodeHandle) -> Arc<D::IntegratedData> {
        self.assert_handle(handle);

        // iterative implementation of the naive recursive implementation

        let mut stack = AutoSmallVec::<InnerNodeHandle>::new();

        // backward loop (find root/cached integrated parent)
        let mut integrated = loop {
            if let Some(integrated) = self.integration_cache.read().get(handle) {
                break Arc::clone(integrated);
            }

            handle = match self.get(handle) {
                NodeRef::Root(RootNodeRef { root_node, root_handle }) => {
                    let integrated = D::integrate_root(
                        ctx,
                        self,
                        root_handle,
                        &root_node.data,
                    );
                    self.integration_cache.write().set(handle, Arc::clone(&integrated));
                    break integrated;
                },
                NodeRef::Inner(InnerNodeRef { inner_node, inner_handle }) => {
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
            self.integration_cache.write()
                .set(current_handle.into(), Arc::clone(&integrated));
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

/// Returns a trait object because we cannot return a &NodeRef with the Index trait
impl<D: DeltaGraphDataType> Index<NodeHandle> for DeltaGraph<D> {
    type Output = dyn GenericNode;

    fn index(&self, handle: NodeHandle) -> &Self::Output {
        self.assert_handle(handle);
        if handle.is_root() {
            &self.root_nodes[handle.idx()]
        }
        else /* handle.is_inner() */ {
            &self.inner_nodes[handle.idx()]
        }
    }
}

impl<D: DeltaGraphDataType> Index<RootNodeHandle> for DeltaGraph<D> {
    type Output = RootNode<D>;

    fn index(&self, handle: RootNodeHandle) -> &Self::Output {
        self.assert_handle(handle.into());
        &self.root_nodes[handle.idx()]
    }
}

impl<D: DeltaGraphDataType> Index<InnerNodeHandle> for DeltaGraph<D> {
    type Output = InnerNode<D>;

    fn index(&self, handle: InnerNodeHandle) -> &Self::Output {
        self.assert_handle(handle.into());
        &self.inner_nodes[handle.idx()]
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
    type Item = NodeRef<'a, D>;

    fn next(&mut self) -> Option<Self::Item> {
        let handle = self.latest?;
        let ref_ = self.graph.get(handle);
        self.latest = ref_.previous();
        Some(ref_)
    }
}
