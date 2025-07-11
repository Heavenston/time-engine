use super::*;

use std::{ fmt::Display, ops::{ Deref, DerefMut, Index }, range::Range, sync::atomic::AtomicU64 };

use glam::{ Affine2, Vec2 };
use itertools::Itertools;

#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct PortalTraversal {
    // TODO: Reduce size to maybe u8 or u16
    pub half_portal_idx: usize,
    pub direction: PortalDirection,
    pub range: Range<f32>,
    pub is_swaped: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Snapshot {
    pub object_id: usize,
    /// Handle of the snapshot node that 'created' this snapshot
    pub handle: NodeHandle,
    /// Each snapshot graph node can create multiple snapshots, each indexed
    /// by their sub_id
    pub sub_id: usize,
    /// Time this snapshot was extrapolated by from its original time
    pub extrapolated_by: Positive,

    pub timeline_id: TimelineId,
    pub age: Positive,
    pub time: f32,

    pub linvel: Vec2,
    pub angvel: f32,

    pub pos: Vec2,
    pub rot: f32,

    pub portal_traversals: AutoSmallVec<PortalTraversal>,
    pub force_transform: Affine2,

    #[deprecated]
    pub validity_time_range: TimeRange,
}

impl Snapshot {
    /// Uniquely identity this snapshot (even across extroplation)
    pub fn id(&self) -> (NodeHandle, usize) {
        (self.handle, self.sub_id)
    }

    pub fn get_transform(&self) -> Affine2 {
        Affine2::from_angle_translation(self.rot, self.pos)
    }

    pub fn copy_without_portal_traversals(&self) -> Self {
        Self {
            portal_traversals: default(),
            ..*self
        }
    }

    pub fn apply_force_transform(&mut self, transform: Affine2) {
        self.linvel = transform.transform_vector2(self.linvel);
        self.pos = transform.transform_point2(self.pos);
        // FIXME: Correct?
        self.rot += transform.to_scale_angle_translation().1;
        self.force_transform *= transform;
    }

    pub fn time_offseted(mut self, dt: f32) -> Self {
        self.time += dt;
        self
    }

    pub fn integrate_by(&mut self, delta: Positive) {
        self.pos += self.linvel * delta.get();
        self.rot += self.angvel * delta.get();
        self.age += delta;
        self.time += delta.get();
    }

    #[deprecated]
    pub fn extrapolate_to(&self, to: f32) -> Option<Self> {
        let to = if (to - self.time).abs() <= DEFAULT_EPSILON { self.time } else { to };
        let Ok(dt) = Positive::new(to - self.time)
        else {
            unreachable!("New time must be higher than current time ({to} is lower than {})", self.time);
        };

        if self.validity_time_range.end().is_some_and(|end| end <= to) {
            return None;
        }

        Some(Self {
            age: self.age + dt,
            time: to,
            extrapolated_by: self.extrapolated_by + dt,

            pos: self.pos + self.linvel * dt.get(),
            rot: self.rot + self.angvel * dt.get(),

            portal_traversals: self.portal_traversals.clone(),

            ..*self
        })
    }
}

impl Display for Snapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "obj{}({}.{}+{}s)", self.object_id, self.handle.idx, self.sub_id, self.extrapolated_by)
    }
}

#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct RootSnapshot {
    pub object_id: usize,

    pub pos: Vec2,
    pub rot: f32,

    pub linvel: Vec2,
    pub angvel: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PartialPortalTraversal {
    pub half_portal_idx: usize,
    pub in_direction: PortalDirection,
    pub sub_id_in: usize,
    pub sub_id_out: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PartialSnapshot {
    /// How much time *after* the previous snapshot does this one
    /// happens in local time
    pub delta_age: Positive,
    /// Is in *local* space
    pub linear_impulse: Vec2,
    /// Is in *local* space... but because there is never any mirroring
    /// this is the same as global space
    pub angular_impulse: f32,
    /// After the impulse is applied this is the list of current portal traversals
    pub portal_traversal: AutoSmallVec<PartialPortalTraversal>,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct NodeHandle {
    #[cfg(debug_assertions)]
    graph_id: u64,
    idx: usize,
}

impl PartialOrd for NodeHandle {
    #[cfg(debug_assertions)]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        (self.graph_id != other.graph_id).then_some(())
            .and_then(|()| self.idx.partial_cmp(&other.idx))
    }

    #[cfg(not(debug_assertions))]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.idx.partial_cmp(&other.idx)
    }
}

impl Display for NodeHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.idx)
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct RootNodeHandle {
    #[cfg(debug_assertions)]
    graph_id: u64,
    idx: usize,
}

impl Display for RootNodeHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.idx)
    }
}

impl From<RootNodeHandle> for NodeHandle {
    fn from(val: RootNodeHandle) -> Self {
        NodeHandle {
            #[cfg(debug_assertions)]
            graph_id: val.graph_id,
            idx: val.idx,
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct InnerNodeHandle {
    #[cfg(debug_assertions)]
    graph_id: u64,
    idx: usize,
}

impl Display for InnerNodeHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.idx)
    }
}

impl From<InnerNodeHandle> for NodeHandle {
    fn from(val: InnerNodeHandle) -> Self {
        NodeHandle {
            #[cfg(debug_assertions)]
            graph_id: val.graph_id,
            idx: val.idx,
        }
    }
}

pub trait GenericNode {
    fn previous(&self) -> Option<NodeHandle>;
    fn children(&self) -> &[InnerNodeHandle];
    fn children_mut(&mut self) -> &mut Vec<InnerNodeHandle>;

    #[deprecated]
    fn timeline_id(&self) -> TimelineId {
        TimelineId::root()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RootNode {
    pub snapshot: RootSnapshot,
    pub children: Vec<InnerNodeHandle>,
}

impl GenericNode for RootNode {
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
pub struct InnerNode {
    pub partial: PartialSnapshot,
    pub previous: NodeHandle,
    pub children: Vec<InnerNodeHandle>,
}

impl GenericNode for InnerNode {
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

#[derive(Debug, Clone, PartialEq)]
pub enum Node {
    Root(RootNode),
    Inner(InnerNode),
}

impl Deref for Node {
    type Target = dyn GenericNode;

    fn deref(&self) -> &Self::Target {
        match self {
            Node::Root(node) => node,
            Node::Inner(node) => node,
        }
    }
}

impl DerefMut for Node {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Node::Root(node) => node,
            Node::Inner(node) => node,
        }
    }
}

impl From<RootNode> for Node {
    fn from(root: RootNode) -> Self {
        Self::Root(root)
    }
}

impl From<InnerNode> for Node {
    fn from(inner: InnerNode) -> Self {
        Self::Inner(inner)
    }
}

#[cfg(debug_assertions)]
static GRAPH_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone)]
pub struct SnapshotGraph {
    #[cfg(debug_assertions)]
    id: u64,
    nodes: Vec<Node>,
    leafs: Vec<NodeHandle>,
}

impl SnapshotGraph {
    pub fn new() -> Self {
        Self {
            #[cfg(debug_assertions)]
            id: GRAPH_ID_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            nodes: default(),
            leafs: vec![]
        }
    }

    fn handle(&self, idx: usize) -> NodeHandle {
        NodeHandle {
            idx,
            #[cfg(debug_assertions)]
            graph_id: self.id,
        }
    }

    fn root_handle(&self, idx: usize) -> RootNodeHandle {
        debug_assert!(matches!(self.nodes[idx], Node::Root(..)));
        RootNodeHandle {
            idx,
            #[cfg(debug_assertions)]
            graph_id: self.id,
        }
    }

    fn inner_handle(&self, idx: usize) -> InnerNodeHandle {
        debug_assert!(matches!(self.nodes[idx], Node::Inner(..)));
        InnerNodeHandle {
            idx,
            #[cfg(debug_assertions)]
            graph_id: self.id,
        }
    }

    pub fn get(&self, handle: NodeHandle) -> Option<&Node> {
        #[cfg(debug_assertions)]
        debug_assert_eq!(handle.graph_id, self.id, "Using handles from another graph");
        self.nodes.get(handle.idx)
    }

    pub fn iter_ancestry(&'_ self, handle: NodeHandle) -> AncestryIterator<'_> {
        AncestryIterator::new(self, Some(handle))
    }

    pub fn nodes(&self) -> impl Iterator<Item = (NodeHandle, &'_ Node)> {
        self.nodes.iter().enumerate()
            .map(|(idx, node)| (self.handle(idx), node))
    }

    pub fn leafs(&self) -> &[NodeHandle] {
        &self.leafs
    }

    pub fn insert_root(&mut self, snapshot: RootSnapshot) -> RootNodeHandle {
        self.nodes.push(RootNode {
            snapshot,
            children: vec![],
        }.into());
        let handle = self.root_handle(self.nodes.len() - 1);
        self.leafs.push(handle.into());
        handle
    }

    pub fn insert(&mut self, partial: PartialSnapshot, previous: impl Into<NodeHandle>) -> InnerNodeHandle {
        let previous = previous.into();
        if let Some((leaf_idx, _)) = self.leafs.iter().find_position(|&&handle_| previous == handle_) {
            self.leafs.remove(leaf_idx);
        }

        self.nodes.push(InnerNode {
            partial,
            previous,
            children: vec![],
        }.into());
        let handle = self.inner_handle(self.nodes.len()-1);

        self.nodes[previous.idx].children_mut().push(handle);

        // // NOTE: Imperfect test
        // debug_assert!(
        //     self[previous].timeline_id() <= partial.timeline_id,
        //     "A timeline parent must have a parent timeline",
        // );
        // debug_assert!(
        //     self[previous].children().iter()
        //         .map(|child| self[child].timeline_id())
        //         .all_unique(),
        //     "A node cannot have multiple nodes on the same timeline",
        // );

        handle
    }
}

impl<'a, I> Index<&'a I> for SnapshotGraph
    where I: Clone,
          SnapshotGraph: Index<I>
{
    type Output = <SnapshotGraph as Index<I>>::Output;

    fn index(&'_ self, index: &'a I) -> &'_ <SnapshotGraph as Index<I>>::Output {
        &self[index.clone()]
    }
}

impl Index<NodeHandle> for SnapshotGraph {
    type Output = Node;

    fn index(&self, index: NodeHandle) -> &Self::Output {
        self.get(index).expect("Invalid handle")
    }
}

impl Index<RootNodeHandle> for SnapshotGraph {
    type Output = RootNode;

    fn index(&self, handle: RootNodeHandle) -> &Self::Output {
        match &self[NodeHandle::from(handle)] {
            Node::Root(root_node) => root_node,
            Node::Inner(_) => unreachable!(),
        }
    }
}

impl Index<InnerNodeHandle> for SnapshotGraph {
    type Output = InnerNode;

    fn index(&self, handle: InnerNodeHandle) -> &Self::Output {
        match &self[NodeHandle::from(handle)] {
            Node::Inner(inner_node) => inner_node,
            Node::Root(_) => unreachable!(),
        }
    }
}

impl Default for SnapshotGraph {
    fn default() -> Self {
        Self::new()
    }
}

pub struct AncestryIterator<'a> {
    graph: &'a SnapshotGraph,
    latest: Option<NodeHandle>,
}

impl<'a> AncestryIterator<'a> {
    pub fn new(graph: &'a SnapshotGraph, latest: Option<NodeHandle>) -> Self {
        Self {
            graph,
            latest,
        }
    }
}

impl<'a> Iterator for AncestryIterator<'a> {
    type Item = (NodeHandle, &'a Node);

    fn next(&mut self) -> Option<Self::Item> {
        let handle = self.latest?;
        let node = &self.graph[handle];
        self.latest = node.previous();
        Some((handle, node))
    }
}
