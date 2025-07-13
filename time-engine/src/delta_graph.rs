use crate::SmallVec;

use std::{ ops::{ Deref, Index }, sync::Arc };

use derive_where::derive_where;
use parking_lot::RwLock;
use itertools::{ Itertools, chain };

pub trait DeltaGraphDataType: Sized + 'static {
    type RootData;
    type PartialData;
    type IntegratedData: ?Sized;
    type Ctx<'a>;
    type InsertCtx<'a> = ();

    fn integrate_root(
        ctx: &mut Self::Ctx<'_>,
        graph: &DeltaGraph<Self>,
        handle: RootNodeHandle<Self>,
        root: &Self::RootData
    ) -> Arc<Self::IntegratedData>;
    fn integrate_partial(
        ctx: &mut Self::Ctx<'_>,
        graph: &DeltaGraph<Self>,
        handle: InnerNodeHandle<Self>,
        running: &Self::IntegratedData,
        partial: &Self::PartialData
    ) -> Arc<Self::IntegratedData>;

    /// Can be used to check the validity of an insert
    /// called after the insert has been fully performed
    #[expect(unused_variables)]
    fn insert_hook(
        ctx: &mut Self::InsertCtx<'_>,
        graph: &DeltaGraph<Self>,
        handle: NodeHandle<Self>,
    ) { }
}

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
    use super::{ GraphId, DeltaGraphDataType };

    use std::{ fmt::Display, marker::PhantomData };

    use derive_where::derive_where;

    /// Bitmask to extract the actual index from the handle value
    const IDX_MASK: usize = !0 >> 1;
    /// If this bit is set to 1 then the node is an inner node
    const INNER_BIT: usize = !IDX_MASK;

    #[derive_where(Debug, Clone, Copy, Hash, PartialEq, Eq)]
    pub struct NodeHandle<D: DeltaGraphDataType> {
        graph_id: GraphId,
        value: usize,
        _data_type: PhantomData<*const D>,
    }

    impl<D: DeltaGraphDataType> NodeHandle<D> {
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

        pub fn either(self) -> EitherHandle<D> {
            self.into()
        }
    }

    impl<D: DeltaGraphDataType> PartialOrd for NodeHandle<D> {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            if self.graph_id != other.graph_id {
                return None;
            }

            // This should place inner nodes after root nodes
            self.value.partial_cmp(&other.value)
        }
    }

    impl<D: DeltaGraphDataType> Display for NodeHandle<D> {
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

    #[derive_where(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd)]
    pub struct RootNodeHandle<D: DeltaGraphDataType> {
        handle: NodeHandle<D>,
    }

    impl<D: DeltaGraphDataType> RootNodeHandle<D> {
        pub(super) fn new(graph_id: GraphId, idx: usize) -> Self {
            assert_eq!(idx, idx & IDX_MASK, "Index overflow!");
            Self {
                handle: NodeHandle::<D> {
                    graph_id,
                    value: idx,
                    _data_type: PhantomData::default(),
                },
            }
        }

        pub fn idx(self) -> usize {
            self.handle.idx()
        }
    }

    impl<D: DeltaGraphDataType> Display for RootNodeHandle<D> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.handle)
        }
    }

    impl<D: DeltaGraphDataType> From<RootNodeHandle<D>> for NodeHandle<D> {
        fn from(val: RootNodeHandle<D>) -> Self {
            val.handle
        }
    }

    impl<D: DeltaGraphDataType> TryFrom<NodeHandle<D>> for RootNodeHandle<D> {
        type Error = ();

        fn try_from(handle: NodeHandle<D>) -> Result<Self, Self::Error> {
            if handle.is_root() {
                Ok(Self { handle })
            }
            else {
                Err(())
            }
        }
    }

    #[derive_where(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd)]
    pub struct InnerNodeHandle<D: DeltaGraphDataType> {
        handle: NodeHandle<D>,
    }

    impl<D: DeltaGraphDataType> InnerNodeHandle<D> {
        pub(super) fn new(graph_id: GraphId, idx: usize) -> Self {
            assert_eq!(idx, idx & IDX_MASK, "Index overflow!");
            Self {
                handle: NodeHandle::<D> {
                    graph_id,
                    value: idx | INNER_BIT,
                    _data_type: PhantomData::default(),
                },
            }
        }

        pub fn idx(self) -> usize {
            self.handle.idx()
        }
    }

    impl<D: DeltaGraphDataType> Display for InnerNodeHandle<D> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.handle)
        }
    }

    impl<D: DeltaGraphDataType> From<InnerNodeHandle<D>> for NodeHandle<D> {
        fn from(val: InnerNodeHandle<D>) -> Self {
            val.handle
        }
    }

    impl<D: DeltaGraphDataType> TryFrom<NodeHandle<D>> for InnerNodeHandle<D> {
        type Error = ();

        fn try_from(handle: NodeHandle<D>) -> Result<Self, Self::Error> {
            if handle.is_inner() {
                Ok(Self { handle })
            }
            else {
                Err(())
            }
        }
    }

    #[derive_where(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd)]
    pub enum EitherHandle<D: DeltaGraphDataType> {
        Root(RootNodeHandle<D>),
        Inner(InnerNodeHandle<D>),
    }
    
    impl<D: DeltaGraphDataType> EitherHandle<D> {
        pub fn as_root(self) -> Option<RootNodeHandle<D>> {
            match self {
                Self::Root(handle) => Some(handle),
                _ => None,
            }
        }

        pub fn as_inner(self) -> Option<InnerNodeHandle<D>> {
            match self {
                Self::Inner(handle) => Some(handle),
                _ => None,
            }
        }
    }

    impl<D: DeltaGraphDataType> From<NodeHandle<D>> for EitherHandle<D> {
        fn from(handle: NodeHandle<D>) -> Self {
            if handle.is_root() {
                EitherHandle::<D>::Root(RootNodeHandle { handle })
            }
            else /* self.is_inner() */ {
                EitherHandle::<D>::Inner(InnerNodeHandle { handle })
            }
        }
    }

    impl<D: DeltaGraphDataType> From<RootNodeHandle<D>> for EitherHandle<D> {
        fn from(handle: RootNodeHandle<D>) -> Self {
            EitherHandle::<D>::Root(handle)
        }
    }

    impl<D: DeltaGraphDataType> From<InnerNodeHandle<D>> for EitherHandle<D> {
        fn from(handle: InnerNodeHandle<D>) -> Self {
            EitherHandle::<D>::Inner(handle)
        }
    }
}
pub use node_handle::*;

pub trait GenericNode<D: DeltaGraphDataType> {
    fn previous(&self) -> Option<NodeHandle<D>>;
    fn children(&self) -> &[InnerNodeHandle<D>];
    fn children_mut(&mut self) -> &mut Vec<InnerNodeHandle<D>>;
}

#[derive_where(Debug, Clone, PartialEq; D::RootData)]
pub struct RootNode<D: DeltaGraphDataType> {
    pub data: D::RootData,
    pub children: Vec<InnerNodeHandle<D>>,
}

impl<D: DeltaGraphDataType> GenericNode<D> for RootNode<D> {
    fn previous(&self) -> Option<NodeHandle<D>> {
        None
    }

    fn children(&self) -> &[InnerNodeHandle<D>] {
        &self.children
    }

    fn children_mut(&mut self) -> &mut Vec<InnerNodeHandle<D>> {
        &mut self.children
    }
}

#[derive_where(Debug, Clone, PartialEq; D::PartialData)]
pub struct InnerNode<D: DeltaGraphDataType> {
    pub data: D::PartialData,
    pub previous: NodeHandle<D>,
    pub children: Vec<InnerNodeHandle<D>>,
}

impl<D: DeltaGraphDataType> GenericNode<D> for InnerNode<D> {
    fn previous(&self) -> Option<NodeHandle<D>> {
        Some(self.previous)
    }

    fn children(&self) -> &[InnerNodeHandle<D>] {
        &self.children
    }

    fn children_mut(&mut self) -> &mut Vec<InnerNodeHandle<D>> {
        &mut self.children
    }
}

#[derive_where(Clone, Copy)]
#[derive_where(PartialEq, Debug; D::RootData)]
pub struct RootNodeRef<'a, D: DeltaGraphDataType> {
    pub root_node: &'a RootNode<D>,
    pub root_handle: RootNodeHandle<D>,
}

impl<'a, D: DeltaGraphDataType> Deref for RootNodeRef<'a, D> {
    type Target = RootNode<D>;

    fn deref(&self) -> &Self::Target {
        self.root_node
    }
}

#[derive_where(Clone, Copy)]
#[derive_where(PartialEq, Debug; D::PartialData)]
pub struct InnerNodeRef<'a, D: DeltaGraphDataType> {
    pub inner_node: &'a InnerNode<D>,
    pub inner_handle: InnerNodeHandle<D>,
}

impl<'a, D: DeltaGraphDataType> Deref for InnerNodeRef<'a, D> {
    type Target = InnerNode<D>;

    fn deref(&self) -> &Self::Target {
        self.inner_node
    }
}

#[derive_where(Clone, Copy)]
#[derive_where(PartialEq, Debug; D::PartialData, D::RootData)]
pub enum NodeRef<'a, D: DeltaGraphDataType> {
    Root(RootNodeRef<'a, D>),
    Inner(InnerNodeRef<'a, D>),
}

impl<'a, D: DeltaGraphDataType> NodeRef<'a, D> {
    pub fn handle(&self) -> NodeHandle<D> {
        match self {
            NodeRef::Root(ref_) => ref_.root_handle.into(),
            NodeRef::Inner(ref_) => ref_.inner_handle.into(),
        }
    }
}

impl<'a, D: DeltaGraphDataType> Deref for NodeRef<'a, D> {
    type Target = dyn GenericNode<D>;

    fn deref(&self) -> &Self::Target {
        match self {
            NodeRef::Root(ref_) => ref_.root_node,
            NodeRef::Inner(ref_) => ref_.inner_node,
        }
    }
}

mod node_cache {
    use super::*;

    use std::marker::PhantomData;

    use derive_where::derive_where;

    #[derive_where(Debug, Clone; T)]
    pub struct DenseNodeCache<D: DeltaGraphDataType, T> {
        root_values: Vec<Option<T>>,
        inner_values: Vec<Option<T>>,
        _data_type: PhantomData<*const D>,
    }

    impl<D: DeltaGraphDataType, T> DenseNodeCache<D, T> {
        pub fn new() -> Self {
            Self {
                root_values: vec![],
                inner_values: vec![],
                _data_type: PhantomData::default(),
            }
        }

        fn array_of(&self, handle: NodeHandle<D>) -> &Vec<Option<T>> {
            if handle.is_root() {
                &self.root_values
            }
            else /* handle.is_inner() */ {
                &self.inner_values
            }
        }

        fn array_of_mut(&mut self, handle: NodeHandle<D>) -> &mut Vec<Option<T>> {
            if handle.is_root() {
                &mut self.root_values
            }
            else /* handle.is_inner() */ {
                &mut self.inner_values
            }
        }

        pub fn get(&self, handle: NodeHandle<D>) -> Option<&T> {
            let array = self.array_of(handle);
            array.get(handle.idx()).map(Option::as_ref).flatten()
        }

        pub fn set(&mut self, handle: NodeHandle<D>, value: T) -> &T {
            let array = self.array_of_mut(handle);
            if array.len() <= handle.idx() {
                array.resize_with(handle.idx() + 1, || None);
            }

            array[handle.idx()] = Some(value);
            array[handle.idx()].as_ref().expect("Present")
        }

        pub fn get_or_insert(&mut self, handle: NodeHandle<D>, fun: impl FnOnce() -> T) -> &T {
            let array = self.array_of_mut(handle);

            if array.len() <= handle.idx() {
                array.resize_with(handle.idx() + 1, || None);
            }

            if !array[handle.idx()].is_some() {
                array[handle.idx()] = Some(fun());
            }
            array[handle.idx()].as_ref().expect("Present")
        }
    }
}
pub use node_cache::*;

#[derive_where(Debug; D::PartialData, D::RootData, D::IntegratedData)]
pub struct DeltaGraph<D: DeltaGraphDataType> {
    id: GraphId,
    root_nodes: Vec<RootNode<D>>,
    inner_nodes: Vec<InnerNode<D>>,
    /// Used to efficiently provide the list of all leafs
    /// assumed to be pretty small
    leafs: Vec<NodeHandle<D>>,
    /// Used to cache the integrated version of all nodes
    /// should be pretty dense (nodes quickly get integrated)
    integration_cache: RwLock<DenseNodeCache<D, Arc<D::IntegratedData>>>,
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

    fn root_handle(&self, idx: usize) -> RootNodeHandle<D> {
        RootNodeHandle::new(self.id, idx)
    }

    fn inner_handle(&self, idx: usize) -> InnerNodeHandle<D> {
        InnerNodeHandle::new(self.id, idx)
    }

    #[cfg(debug_assertions)]
    fn assert_handle(&self, handle: NodeHandle<D>) {
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

    pub fn get_root(&self, root_handle: RootNodeHandle<D>) -> RootNodeRef<'_, D> {
        self.assert_handle(root_handle.into());
        RootNodeRef {
            root_node: &self.root_nodes[root_handle.idx()],
            root_handle,
        }
    }

    pub fn get_inner(&self, inner_handle: InnerNodeHandle<D>) -> InnerNodeRef<'_, D> {
        self.assert_handle(inner_handle.into());
        InnerNodeRef {
            inner_node: &self.inner_nodes[inner_handle.idx()],
            inner_handle,
        }
    }

    pub fn get(&self, handle: NodeHandle<D>) -> NodeRef<'_, D> {
        self.assert_handle(handle);
        match handle.either() {
            EitherHandle::Root(root_handle) => {
                NodeRef::Root(self.get_root(root_handle))
            },
            EitherHandle::Inner(inner_handle) => {
                NodeRef::Inner(self.get_inner(inner_handle))
            },
        }
    }

    /// PRIVATE because manually mutating nodes is not part of the public api
    fn get_mut(&mut self, handle: NodeHandle<D>) -> &mut dyn GenericNode<D> {
        self.assert_handle(handle);
        if handle.is_root() {
            &mut self.root_nodes[handle.idx()]
        }
        else /* handle.is_inner() */ {
            &mut self.inner_nodes[handle.idx()]
        }
    }

    pub fn iter_ancestry(&'_ self, handle: NodeHandle<D>) -> AncestryIterator<'_, D> {
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

    pub fn leafs(&self) -> &[NodeHandle<D>] {
        &self.leafs
    }

    pub fn directed_distance(&self, parent: NodeHandle<D>, mut child: NodeHandle<D>) -> Option<usize> {
        // We know the handle of a parent must have been created before its children
        // which means if this condition is true the relationship cannot be true
        if child < parent {
            return None;
        }

        let mut distance = 0;
        while child != parent {
            let Some(new_parent) = self[child].previous()
            else { return None };
            distance += 1;
            child = new_parent;
        }
        Some(distance)
    }

    pub fn distance(&self, handle1: NodeHandle<D>, handle2: NodeHandle<D>) -> Option<isize> {
        if handle1 < handle2 {
            self.directed_distance(handle1, handle2)
                .map(|dist| isize::try_from(dist).expect("No overflow"))
        }
        else {
            self.directed_distance(handle2, handle1)
                .map(|dist| -isize::try_from(dist).expect("No overflow"))
        }
    }

    pub fn is_parent(&self, parent: NodeHandle<D>, child: NodeHandle<D>) -> bool {
        self.directed_distance(parent, child).is_some()
    }

    pub fn is_directly_related(&self, handle1: NodeHandle<D>, handle2: NodeHandle<D>) -> bool {
        self.distance(handle1, handle2).is_some()
    }

    /// Returns the list of nodes going up from the given child to the given
    /// parent (with both the child and parent included in the vec)
    /// Returns None if and only if `is_parent(to_parent, from_child)` would return false
    pub fn find_path(&self, from_child: NodeHandle<D>, to_parent: NodeHandle<D>) -> Option<Vec<NodeHandle<D>>> {
        // Case already handled by the code after
        // if from_child < to_parent {
        //     return None;
        // }

        let result = self.iter_ancestry(from_child)
            .map(|node| node.handle())
            .take_while_inclusive(|&last| last > to_parent)
            .collect_vec();

        if result.last() == Some(&to_parent) {
            Some(result)
        }
        else {
            None
        }
    }

    pub fn children(&self, handle: NodeHandle<D>) -> impl Iterator<Item = InnerNodeRef<'_, D>> + '_ {
        let node = self.get(handle);
        (0..node.children().len()).into_iter()
            .map(move |i| self.get_inner(node.children()[i]))
    }

    pub fn insert_root(&mut self, ctx: &mut D::InsertCtx<'_>, data: D::RootData) -> RootNodeHandle<D> {
        self.root_nodes.push(RootNode {
            data,
            children: vec![],
        }.into());
        let handle = self.root_handle(self.root_nodes.len() - 1);
        self.leafs.push(handle.into());
        D::insert_hook(ctx, &*self, handle.into());
        handle
    }

    pub fn insert(&mut self, ctx: &mut D::InsertCtx<'_>, data: D::PartialData, previous: impl Into<NodeHandle<D>>) -> InnerNodeHandle<D> {
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

        D::insert_hook(ctx, &*self, handle.into());
        handle
    }

    pub fn integrate<'a, 'b>(&'a self, ctx: &'b mut D::Ctx<'b>, mut handle: NodeHandle<D>) -> Arc<D::IntegratedData> {
        self.assert_handle(handle);

        // iterative implementation of the naive recursive implementation

        let mut stack = SmallVec::<InnerNodeHandle<D>, 2>::new();

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
impl<D: DeltaGraphDataType> Index<NodeHandle<D>> for DeltaGraph<D> {
    type Output = dyn GenericNode<D>;

    fn index(&self, handle: NodeHandle<D>) -> &Self::Output {
        self.assert_handle(handle);
        if handle.is_root() {
            &self.root_nodes[handle.idx()]
        }
        else /* handle.is_inner() */ {
            &self.inner_nodes[handle.idx()]
        }
    }
}

impl<D: DeltaGraphDataType> Index<RootNodeHandle<D>> for DeltaGraph<D> {
    type Output = RootNode<D>;

    fn index(&self, handle: RootNodeHandle<D>) -> &Self::Output {
        self.assert_handle(handle.into());
        &self.root_nodes[handle.idx()]
    }
}

impl<D: DeltaGraphDataType> Index<InnerNodeHandle<D>> for DeltaGraph<D> {
    type Output = InnerNode<D>;

    fn index(&self, handle: InnerNodeHandle<D>) -> &Self::Output {
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
    latest: Option<NodeHandle<D>>,
}

impl<'a, D: DeltaGraphDataType> AncestryIterator<'a, D> {
    pub fn new(graph: &'a DeltaGraph<D>, latest: Option<NodeHandle<D>>) -> Self {
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
