use std::{
    cmp::{ max, min }, collections::HashMap, iter::{ empty, once }, ops::{ BitAnd, ControlFlow, Range, RangeFrom }, sync::Arc
};

use parking_lot::RwLock;
use rayon::{ prelude::*, iter as pariter };
use glam::{ Affine2, Vec2 };
use itertools::Itertools;
use ordered_float::OrderedFloat as OF;

use super::*;
use sg::GenericNode as _;

/// All portals are eternal (always existed and will always exist)
/// and are present in all timelines
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct HalfPortal {
    pub transform: Affine2,
    pub height: f32,
    pub linked_to: usize,
    /// How much time changes when going from this portal to the linked one
    pub time_offset: f32,
}

#[derive(Debug, Clone, Default)]
struct CollisionNewState {
    vel: Vec2,
    // portal_traversals: AutoSmallVec<sg::PortalTraversal>,
    // time_offset: f32,
}

#[derive(Debug, Clone)]
struct Collision<const N: usize> {
    debug_reason: &'static str,
    impact_time: f32,
    impact_delta_time: Positive,
    states: [CollisionNewState; N],
}

#[derive(Debug, Clone)]
struct SnapshotWithHandle {
    handle: sg::NodeHandle,
    snap: sg::Snapshot,
}

#[derive(Debug, Clone)]
struct SimCollisionInfo<const N: usize> {
    col: Collision<N>,
    snaps: [SnapshotWithHandle; N],
}

impl<const N: usize> SimCollisionInfo<N> {
    fn get_snap(&self, i: usize) -> SnapshotWithHandle {
        todo!()
    }

    fn child_timeline(&self, multiverse: &TimelineMultiverse) -> TimelineId {
        assert!(N > 0);
        debug_assert!(
            self.snaps.iter().map(|snap| snap.snap.timeline_id)
                .tuple_windows::<(_, _)>()
                .all(|(a, b)| multiverse.is_related(a, b))
        );
        self.snaps.iter()
            .map(|snap| snap.snap.timeline_id)
            .max()
            .expect("No empty")
    }

    fn parent_timeline(&self, multiverse: &TimelineMultiverse) -> TimelineId {
        assert!(N > 0);
        debug_assert!(
            self.snaps.iter().map(|snap| snap.snap.timeline_id)
                .tuple_windows::<(_, _)>()
                .all(|(a, b)| multiverse.is_related(a, b))
        );
        self.snaps.iter()
            .map(|snap| snap.snap.timeline_id)
            .min()
            .expect("No empty")
    }
}

#[derive(Debug, Clone)]
enum SimGenericCollisionInfo {
    One(SimCollisionInfo<1>),
    Two(SimCollisionInfo<2>),
}

impl SimGenericCollisionInfo {
    fn impact_time(&self) -> f32 {
        match self {
            Self::One(i) => i.col.impact_time,
            Self::Two(i) => i.col.impact_time,
        }
    }

    fn impact_delta_time(&self) -> Positive {
        match self {
            Self::One(i) => i.col.impact_delta_time,
            Self::Two(i) => i.col.impact_delta_time,
        }
    }

    fn simple_debug(&self) -> String {
        match self {
            SimGenericCollisionInfo::One(info) => {
                format!("one({}, {}s, idx {})", info.col.debug_reason, info.col.impact_time, info.snaps[0].snap.object_id)
            },
            SimGenericCollisionInfo::Two(info) => {
                format!("two({}, {}s, idx {} on idx {})", info.col.debug_reason, info.col.impact_time, info.snaps[0].snap.object_id, info.snaps[1].snap.object_id)
            },
        }
    }

    fn handles(&self) -> AutoSmallVec<sg::NodeHandle> {
        match self {
            SimGenericCollisionInfo::One(info) => info.snaps.iter().map(|snap| snap.handle).collect(),
            SimGenericCollisionInfo::Two(info) => info.snaps.iter().map(|snap| snap.handle).collect(),
        }
    }

    fn child_timeline(&self, multiverse: &TimelineMultiverse) -> TimelineId {
        match self {
            SimGenericCollisionInfo::One(sim) => sim.child_timeline(multiverse),
            SimGenericCollisionInfo::Two(sim) => sim.child_timeline(multiverse),
        }
    }

    fn parent_timeline(&self, multiverse: &TimelineMultiverse) -> TimelineId {
        match self {
            SimGenericCollisionInfo::One(sim) => sim.parent_timeline(multiverse),
            SimGenericCollisionInfo::Two(sim) => sim.parent_timeline(multiverse),
        }
    }
}

impl From<SimCollisionInfo<1>> for SimGenericCollisionInfo {
    fn from(value: SimCollisionInfo<1>) -> Self {
        Self::One(value)
    }
}

impl From<SimCollisionInfo<2>> for SimGenericCollisionInfo {
    fn from(value: SimCollisionInfo<2>) -> Self {
        Self::Two(value)
    }
}

#[derive(Debug, Clone)]
pub enum SimTimeRange {
    Range(Range<f32>),
    RangeFrom(RangeFrom<f32>),
}

impl SimTimeRange {
    pub fn new(start: f32, end: Option<f32>) -> Self {
        match end {
            Some(end) => Self::Range(start..end),
            None => Self::RangeFrom(start..),
        }
    }

    pub fn start(&self) -> f32 {
        match self {
            SimTimeRange::Range(range) => range.start,
            SimTimeRange::RangeFrom(range_from) => range_from.start,
        }
    }

    pub fn end(&self) -> Option<f32> {
        match self {
            SimTimeRange::Range(range) => Some(range.end),
            SimTimeRange::RangeFrom(_) => None,
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            SimTimeRange::Range(range) => range.start >= range.end,
            SimTimeRange::RangeFrom(_) => false,
        }
    }

    pub fn contains(&self, val: f32) -> bool {
        val >= self.start() && self.end().is_none_or(|end| val < end)
    }

    pub fn offset(&self, offset: f32) -> Self {
        Self::new(self.start() + offset, self.end().map(|end| end + offset))
    }

    pub fn up_to(&self, to: f32) -> Option<Self> {
        if self.start() >= to {
            None
        }
        else {
            Some(Self::new(self.start(), self.end().map(|end| f32::min(end, to)).or(Some(to))))
        }
    }
}

impl From<Range<f32>> for SimTimeRange {
    fn from(range: Range<f32>) -> Self {
        Self::Range(range)
    }
}

impl From<RangeFrom<f32>> for SimTimeRange {
    fn from(range_from: RangeFrom<f32>) -> Self {
        Self::RangeFrom(range_from)
    }
}

impl BitAnd for SimTimeRange {
    type Output = SimTimeRange;

    /// Returns a range for values that are in both ranges
    fn bitand(self, rhs: Self) -> Self::Output {
        let start = self.start().max(rhs.start());
        let end = self.end()
            .zip(rhs.end())
            .map(|(lhs, rhs)| f32::min(lhs, rhs))
            .or(rhs.end());
        // Makes sure start <= end
        let start = end.unwrap_or(start).max(start);

        Self::new(start, end)
    }
}

#[derive(Debug, Clone)]
pub struct SimTimeRanges {
    pub main_range: SimTimeRange,
    pub ghost_ranges: AutoSmallVec<SimTimeRange>,
}

impl SimTimeRanges {
    pub fn contains(&self, val: f32) -> bool {
        once(&self.main_range).chain(self.ghost_ranges.iter())
            .any(|range| range.contains(val))
    }
}

pub struct TimelineQueryResult {
    pub before: Vec<sg::NodeHandle>,
    pub after: Option<sg::NodeHandle>,
}

#[derive(Debug)]
pub struct Simulator {
    world_state: Arc<WorldState>,
    multiverse: TimelineMultiverse,
    snapshots: sg::SnapshotGraph,
    // TODO: Use a 'sorted map' since handles are linear...
    integration_cache: RwLock<HashMap<sg::NodeHandle, Arc<sg::Snapshot>>>,

    /// The first snapshot of all spheres
    starts: Ro<Box<[sg::RootNodeHandle]>>,
    half_portals: Vec<HalfPortal>,

    /// Cannot be changed without recomputing everything, i think
    max_time: Ro<f32>,
}

impl Simulator {
    pub fn new(world_state: Arc<WorldState>, max_time: f32) -> Self {
        let multiverse = TimelineMultiverse::new();

        let mut snapshots = sg::SnapshotGraph::new();
        let starts = world_state.spheres.iter()
            .enumerate()
            .map(|(object_id, sphere)| sg::RootSnapshot {
                object_id,

                pos: sphere.initial_pos,
                rot: 0.,

                linvel: sphere.initial_velocity,
                angvel: 0.,
            })
            .map(|snapshot| snapshots.insert_root(snapshot))
            .collect_vec();

        let mut half_portals = Vec::<HalfPortal>::new();
        for &portal in &world_state.portals {
            half_portals.push(HalfPortal {
                transform: portal.in_transform,
                height: portal.height,
                linked_to: half_portals.len() + 1,
                time_offset: portal.time_offset,
            });
            half_portals.push(HalfPortal {
                transform: portal.out_transform,
                height: portal.height,
                linked_to: half_portals.len() - 1,
                time_offset: -portal.time_offset,
            });
        }

        Self {
            world_state,
            multiverse,
            snapshots,
            integration_cache: default(),

            starts: starts.clone().into_boxed_slice().into(),
            half_portals,

            max_time: Ro::new(max_time),
        }
    }

    pub fn snapshots(&self) -> &sg::SnapshotGraph {
        &self.snapshots
    }

    pub fn multiverse(&self) -> &TimelineMultiverse {
        &self.multiverse
    }

    // TODO: Find another name
    pub fn max_time(&self) -> f32 {
        *self.max_time
    }

    fn node_has_children(&self, handle: sg::NodeHandle, timeline_id: TimelineId) -> bool {
        let Some(node) = self.snapshots.get(handle)
        else { return false; };

        node.children().iter()
            .any(|&child_handle| self.multiverse.is_related(self.snapshots[child_handle].timeline_id(), timeline_id))
    }

    pub fn integrate(&self, node: sg::NodeHandle) -> Arc<sg::Snapshot> {
        if let Some(snap) = self.integration_cache.read().get(&node).map(Arc::clone) {
            return snap;
        }

        let (snapshot, partial) = match &self.snapshots[node] {
            sg::Node::Root(root_node) => {
                return Arc::clone(self.integration_cache.write().entry(node)
                    .or_insert_with(|| Arc::new(root_node.snapshot.into())));
            },
            sg::Node::Inner(inner_node) => (self.integrate(inner_node.previous), inner_node.partial),
        };
        debug_assert!(self.multiverse.is_parent(snapshot.timeline_id, partial.timeline_id));

        // TODO: Portal traversals

        Arc::clone(self.integration_cache.write().entry(node).or_insert_with(|| {
            Arc::new(sg::Snapshot {
                object_id: snapshot.object_id,
                timeline_id: partial.timeline_id,

                age: snapshot.age + partial.delta_age,
                time: snapshot.time + partial.delta_age.get(),
                extrapolated_by: Positive::new(0.).expect("positive"),

                linvel: partial.linvel,
                angvel: partial.angvel,

                pos: snapshot.pos + snapshot.linvel * partial.delta_age.get(),
                rot: snapshot.rot + snapshot.angvel * partial.delta_age.get(),

                portal_traversals: default(),
            })
        }))
    }

    #[inline(always)]
    fn get_node_time_ranges<'b>(&'b self, handle: sg::NodeHandle, timeline_id: Option<TimelineId>) -> SimTimeRanges {
        let node = &self.snapshots[handle];
        let snap = self.integrate(handle);
        let max_delta_age = node.children().iter().copied()
            .filter(|&child_link| timeline_id.is_none_or(|timeline_id| self.multiverse.is_parent(self.snapshots[child_link].timeline_id(), timeline_id)))
            .map(|child_link| self.snapshots[child_link].partial.delta_age)
            .max()
        ;
        let end_time = max_delta_age.map(|dt| snap.time + dt.get());

        let range = SimTimeRange::new(snap.time, end_time);
        // TODO
        // let range_ = range.clone();
        // let iterator = snap.portal_traversals.into_iter()
        //     .filter_map(move |traversal| {
        //         let offset = self.half_portals[traversal.half_portal_idx].time_offset;
        //         range_.offset(offset)
        //             .up_to(traversal.end_age - snap.age + snap.time + traversal.time_offset())
        //     })
        // ;
        let iterator = empty();

        SimTimeRanges {
            main_range: range,
            ghost_ranges: iterator.collect(),
        }
    }

    pub fn time_query(&self, time: f32) -> Vec<(SimTimeRanges, sg::NodeHandle)> {
        debug_assert!(self.starts.iter().copied().all_unique());
        let mut result = Vec::new();

        for (link, _) in self.snapshots.nodes() {
            let range = self.get_node_time_ranges(link, None);
            if range.contains(time) {
                result.push((range, link));
            }
        }

        result
    }

    /// Return an iterator of all nodes that are in the given timeline
    pub fn timeline_query(&self, timeline_id: TimelineId) -> impl Iterator<Item = sg::NodeHandle> + Clone + '_ {
        #[derive(Debug, Clone, Copy)]
        struct StackEntry {
            handle: sg::NodeHandle,
            is_in_timeline: bool,
        }

        let mut stack = self.starts.iter().copied()
            .inspect(|&link| debug_assert_eq!(self.snapshots[link].timeline_id(), self.multiverse.root()))
            .map(|handle| StackEntry {
                handle: handle.into(),
                // Necessarily true as link is a root
                is_in_timeline: true,
            })
            .collect_vec();

        std::iter::from_fn(move || {
            while let Some(StackEntry { handle: link, is_in_timeline }) = stack.pop() {
                debug_assert!(stack.iter().copied()
                    .map(|entry| entry.handle)
                    .all_unique());

                let node = &self.snapshots[link];

                if is_in_timeline {
                    let children = node.children().iter().copied()
                        // take the distance of each child's timeline with the target timeline
                        .filter_map(|child_handle| self.multiverse.distance(self.snapshots[child_handle].timeline_id(), timeline_id)
                            // checks that it is a parent
                            .and_then(|distance| (distance >= 0).then_some(distance))
                            .map(|distance| (child_handle, distance)))
                        .collect_smallvec()
                    ;
                    let min_distance = children.iter()
                        .map(|&(_, distance)| distance)
                        .min();

                    children.iter()
                        .map(|&(handle, distance)| StackEntry {
                            handle: handle.into(),
                            is_in_timeline: distance == min_distance.expect("Non empty")
                        })
                        .collect_into(&mut stack);

                    return Some(link);
                }
                else {
                    node.children().iter().copied()
                        .map(|child_handle| (child_handle, self.snapshots[child_handle].timeline_id()))
                        .filter(|&(_, child_timeline_id)| self.multiverse.is_parent(child_timeline_id, timeline_id))
                        .map(|(child_handle, child_timeline_id)| StackEntry {
                            handle: child_handle.into(),
                            is_in_timeline: child_timeline_id == timeline_id
                        })
                        .collect_into(&mut stack);
                }
            }

            None
        })
    }

    fn apply_collision_1(&mut self, info: SimCollisionInfo<1>) {
        let new_partial = sg::PartialSnapshot {
            timeline_id: info.snaps[0].snap.timeline_id,
            delta_age: info.snaps[0].snap.extrapolated_by + info.col.impact_delta_time,
            linvel: info.col.states[0].vel,
            angvel: 0.,
        };

        self.snapshots.insert(new_partial, info.snaps[0].handle);
    }

    fn apply_collision_2(&mut self, info: SimCollisionInfo<2>) {
        for i in 0..2 {
            let st = self.integrate(info.snaps[i].handle).time;
            let dt = info.col.impact_time - st;
            let new_partial = sg::PartialSnapshot {
                timeline_id: info.snaps[i].snap.timeline_id,
                delta_age: dt.try_into().expect("Positive"),
                linvel: info.col.states[i].vel,
                angvel: 0.,
            };

            self.snapshots.insert(new_partial, info.snaps[i].handle);
        }
    }

    fn apply_collision(&mut self, info: SimGenericCollisionInfo) {
        match info {
            SimGenericCollisionInfo::One(info) => {
                self.apply_collision_1(info);
            },
            SimGenericCollisionInfo::Two(info) => {
                self.apply_collision_2(info);
            },
        }
    }

    fn snapshot_collision_shape(&self, snap: &sg::Snapshot) -> impl parry2d::shape::Shape {
        let radius = self.world_state.spheres()[snap.object_id].radius;
        parry2d::shape::Ball::new(radius)
    }

    fn snapshot_affine(&self, snap: &sg::Snapshot) -> Affine2 {
        Affine2::from_angle_translation(snap.rot, snap.pos)
    }

    fn snapshot_motion(&self, snap: &sg::Snapshot) -> parry2d::query::NonlinearRigidMotion {
        parry2d::query::NonlinearRigidMotion {
            start: affine_to_isometry(self.snapshot_affine(snap)),
            local_center: default(),
            linvel: snap.linvel.to_na(),
            angvel: snap.angvel,
        }
    }

    fn cast_sphere_wall_collision(
        &self,
        snap: &sg::Snapshot,
        dt_range: Range<f32>,
    ) -> Option<Collision<1>> {
        let walls_shape = self.world_state.get_static_body_collision();
        // TODO
        // let walls_shape = snap.portal_traversals.iter()
        //     .fold(walls_shape, |walls_shape, traversal|
        //         clip_shapes_on_portal(walls_shape, traversal.portal_in.transform, traversal.direction)
        //     );
        let walls_shape = i_shape_to_parry_shape(walls_shape);
        let sphere_shape = self.snapshot_collision_shape(snap);

        let collision = parry2d::query::cast_shapes_nonlinear(
            &parry2d::query::NonlinearRigidMotion::identity(), &walls_shape,
            &self.snapshot_motion(snap), &sphere_shape,
            dt_range.start, dt_range.end, false,
        ).expect("Supported")?;

        let impact_normal = Vec2::new(collision.normal1.x, collision.normal1.y);
        let impact_dt = collision.time_of_impact;
        let impact_time = snap.time + impact_dt;
        let new_vel = snap.linvel - 2.0 * snap.linvel.dot(impact_normal) * impact_normal;

        Some(Collision {
            debug_reason: "sphere-wall",
            impact_time,
            impact_delta_time: Positive::new(impact_dt).expect("Positive"),
            states: [CollisionNewState { vel: new_vel, ..default() }]
        })
    }

    fn cast_sphere_sphere_collision(
        &self,
        s1: &sg::Snapshot,
        s2: &sg::Snapshot,
        dt_range: Range<f32>,
    ) -> Option<Collision<2>> {
        debug_assert!((s1.time - s2.time).abs() <= DEFAULT_EPSILON);
        debug_assert!(self.multiverse.is_related(s1.timeline_id, s2.timeline_id));

        let rad1 = self.world_state.spheres()[s1.object_id].radius;
        let rad2 = self.world_state.spheres()[s2.object_id].radius;

        let collision = parry2d::query::cast_shapes_nonlinear(
            &self.snapshot_motion(s1), &self.snapshot_collision_shape(s1),
            &self.snapshot_motion(s2), &self.snapshot_collision_shape(s2),
            dt_range.start, dt_range.end, false,
        ).expect("Supported")?;

        let impact_dt = collision.time_of_impact;
        let collision_normal = collision.normal1.to_gl();
        let relative_velocity = s1.linvel - s2.linvel;
        let velocity_along_normal = relative_velocity.dot(collision_normal);
        
        // Compute masses from radii (equal density) and elastic impulse
        let m1 = rad1 * rad1;
        let m2 = rad2 * rad2;
        let impulse_scalar = 2.0 * m1 * m2 / (m1 + m2) * velocity_along_normal;
        let impulse = impulse_scalar * collision_normal;
        let vel1 = s1.linvel - impulse / m1;
        let vel2 = s2.linvel + impulse / m2;
        
        Some(Collision {
            debug_reason: "sphere-sphere",
            impact_time: s1.time + impact_dt,
            impact_delta_time: Positive::new(impact_dt).expect("Positive"),
            states: [
                CollisionNewState { vel: vel1, ..default() },
                CollisionNewState { vel: vel2, ..default() },
            ],
        })
    }

    pub fn step(&mut self) -> ControlFlow<(), ()> {
        let this = &*self;

        // println!();

        let leafs = self.snapshots.nodes()
            .filter(|(_, node)| node.children().is_empty())
            .map(|(handle, _)| (handle, (&*self.integrate(handle)).clone()))
            .collect_vec()
        ;

        // println!("{}", leafs.iter().map(|(_, snap)| format!("obj {} -> {}s", snap.object_id, snap.time)).join("\n"));
        
        let timelines_maxs = leafs.iter()
            .map(|(_, snap)| (snap.timeline_id, snap.time))
            .into_grouping_map()
            .max_by_key(|_, &t| OF(t))
        ;

        // println!("{}", timelines_maxs.iter().map(|(tid, max)| format!("tid {tid} -> {max}s")).join("\n"));

        let mut leafs = leafs;
        for (_, snap) in &mut leafs {
            *snap = snap.extrapolate_to(timelines_maxs[&snap.timeline_id]);
        }
        let leafs = leafs;
     
        let groups = leafs.par_iter().cloned()
            .flat_map(|(handle, snap)| {
                let tid = snap.timeline_id;
                let t = snap.time;
                debug_assert_eq!(t, timelines_maxs[&tid]);
                let world = self.timeline_query(tid).par_bridge()
                    .filter(move |&handle_| handle < handle_)
                    .filter(move |&handle_| {
                        this.get_node_time_ranges(handle_, Some(tid))
                            .contains(t)
                    })
                    .map(move |handle_| (handle_, this.integrate(handle_).extrapolate_to(t)))
                    .collect::<Vec<_>>();
                pariter::repeat((handle, snap)).zip(world)
            })
            // Already the case by only outputing h1 < h2 in the previous step
            // .unique_by(|&((h1, _), (h2, _))| (min(h1, h2), max(h1, h2)))
            .collect::<Vec<_>>()
        ;

        debug_assert!(
            groups.iter()
                .map(|&((h1, _), (h2, _))| (min(h1, h2), max(h1, h2)))
                .all_unique()
        );

        // println!("{}", groups.len());
        // println!("{}", groups.iter().map(|((_, s1), (_, s2))| format!("{}-{}s and {}-{}s", s1.object_id, s1.time, s2.object_id, s2.time)).join("\n"));

        let collision = pariter::empty()
            // sphere-wall collisions
            .chain(
                leafs.par_iter().map(|(handle, snap)| (*handle, snap))
                .filter_map(|(handle, snap)| {
                    let end = *self.max_time - snap.time;
                    self.cast_sphere_wall_collision(snap, 0. .. end)
                    .map(|col| SimCollisionInfo {
                        col,
                        snaps: [SnapshotWithHandle { handle, snap: snap.clone() }],
                    })
                    .map(SimGenericCollisionInfo::from)
                })
            )
            // sphere-sphere collisions
            .chain(
                groups.par_iter()
                    .map(|((h1, s1), (h2, s2))| ((*h1, s1), (*h2, s2)))
                    .filter_map(|((h1, s1), (h2, s2))| {
                        debug_assert_eq!(s1.time, s2.time);
                        let end = *self.max_time - s1.time;
                        self.cast_sphere_sphere_collision(s1, s2, 0. .. end)
                        .map(|col| SimCollisionInfo {
                            col,
                            snaps: [
                                SnapshotWithHandle { handle: h1, snap: s1.clone() },
                                SnapshotWithHandle { handle: h2, snap: s2.clone() },
                            ],
                        })
                        .map(SimGenericCollisionInfo::from)
                    })
            )
            .filter(|col| col.impact_time() < *self.max_time)
            .min_by_key(|col| OF(col.impact_time()))
        ;

        let Some(collision) = collision
        else { return ControlFlow::Break(()) };

        // println!("{}", collisions.iter().map(|col| col.simple_debug()).join("\n"));

        self.apply_collision(collision);
        
        ControlFlow::Continue(())
    }

    pub fn run(&mut self) {
        while self.step().is_continue() { }
    }
}
