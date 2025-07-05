use std::{
    collections::HashMap, iter::{ empty, repeat, zip }, ops::ControlFlow, range::Range, sync::Arc
};

use parking_lot::RwLock;
use glam::{ Affine2, Vec2 };
use itertools::Itertools;
use parry2d::shape::Shape;
use smallvec::smallvec;
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
    pub linear_impulse: Vec2,
    pub angular_impulse: f32,
}

#[derive(Debug, Clone)]
struct Collision<const N: usize> {
    debug_reason: &'static str,
    impact_time: f32,
    impact_delta_time: Positive,
    states: [CollisionNewState; N],
}

#[derive(Debug, Clone)]
struct CollisionSimulationEvent<'a, const N: usize> {
    col: Collision<N>,
    snaps: [&'a sg::Snapshot; N],
}

impl<'a, const N: usize> CollisionSimulationEvent<'a, N> {
    fn child_timeline(&self, multiverse: &TimelineMultiverse) -> TimelineId {
        assert!(N > 0);
        debug_assert!(
            self.snaps.iter().map(|snap| snap.timeline_id)
                .tuple_windows::<(_, _)>()
                .all(|(a, b)| multiverse.is_related(a, b))
        );
        self.snaps.iter()
            .map(|snap| snap.timeline_id)
            .max()
            .expect("No empty")
    }

    fn parent_timeline(&self, multiverse: &TimelineMultiverse) -> TimelineId {
        assert!(N > 0);
        debug_assert!(
            self.snaps.iter().map(|snap| snap.timeline_id)
                .tuple_windows::<(_, _)>()
                .all(|(a, b)| multiverse.is_related(a, b))
        );
        self.snaps.iter()
            .map(|snap| snap.timeline_id)
            .min()
            .expect("No empty")
    }
}

#[derive(Debug, Clone)]
struct NewPortalTraversalData {
    half_portal_idx: usize,
    direction: PortalDirection,
    traversal_direction: sg::PortalTraversalDirection,
    delta_range: Range<Positive>,
    range: Range<f32>,
}

#[derive(Debug, Clone)]
struct PortalTraversalSimulationEvent<'a> {
    data: NewPortalTraversalData,
    snap: &'a sg::Snapshot,
}

#[derive(Debug, Clone)]
enum GenericSimulationEventInfo<'a> {
    OneCollision(CollisionSimulationEvent<'a, 1>),
    TwoCollision(CollisionSimulationEvent<'a, 2>),
    PortalTraversal(PortalTraversalSimulationEvent<'a>),
}

impl<'a> GenericSimulationEventInfo<'a> {
    fn impact_time(&self) -> f32 {
        match self {
            Self::OneCollision(i) => i.col.impact_time,
            Self::TwoCollision(i) => i.col.impact_time,
            Self::PortalTraversal(i) => i.data.range.start,
        }
    }

    fn impact_delta_time(&self) -> Positive {
        match self {
            Self::OneCollision(i) => i.col.impact_delta_time,
            Self::TwoCollision(i) => i.col.impact_delta_time,
            Self::PortalTraversal(i) => i.data.delta_range.start,
        }
    }

    fn simple_debug(&self) -> String {
        match self {
            GenericSimulationEventInfo::OneCollision(info) => {
                format!("one({}, dt {}s, {})", info.col.debug_reason, info.col.impact_delta_time, info.snaps[0])
            },
            GenericSimulationEventInfo::TwoCollision(info) => {
                format!("two({}, dt {}s, {} and {})", info.col.debug_reason, info.col.impact_delta_time, info.snaps[0], info.snaps[1])
            },
            Self::PortalTraversal(info) => {
                format!("portal_traversal(range {:?}, {} on portal {})", info.data.delta_range, info.snap, info.data.half_portal_idx)
            },
        }
    }

    fn child_timeline(&self, multiverse: &TimelineMultiverse) -> TimelineId {
        match self {
            Self::OneCollision(sim) => sim.child_timeline(multiverse),
            Self::TwoCollision(sim) => sim.child_timeline(multiverse),
            Self::PortalTraversal(i) => i.snap.timeline_id,
        }
    }

    fn parent_timeline(&self, multiverse: &TimelineMultiverse) -> TimelineId {
        match self {
            Self::OneCollision(sim) => sim.parent_timeline(multiverse),
            Self::TwoCollision(sim) => sim.parent_timeline(multiverse),
            Self::PortalTraversal(i) => i.snap.timeline_id,
        }
    }
}

impl<'a> From<CollisionSimulationEvent<'a, 1>> for GenericSimulationEventInfo<'a> {
    fn from(value: CollisionSimulationEvent<'a, 1>) -> Self {
        Self::OneCollision(value)
    }
}

impl<'a> From<CollisionSimulationEvent<'a, 2>> for GenericSimulationEventInfo<'a> {
    fn from(value: CollisionSimulationEvent<'a, 2>) -> Self {
        Self::TwoCollision(value)
    }
}

impl<'a> From<PortalTraversalSimulationEvent<'a>> for GenericSimulationEventInfo<'a> {
    fn from(value: PortalTraversalSimulationEvent<'a>) -> Self {
        Self::PortalTraversal(value)
    }
}

pub struct TimelineQueryResult {
    pub before: Vec<sg::NodeHandle>,
    pub after: Option<sg::NodeHandle>,
}

#[derive(Debug, Clone, Copy)]
struct PortalTraversalCheckResult {
    pub direction: PortalDirection,
    pub duration: Range<f32>,
}

#[derive(Debug)]
pub struct Simulator {
    world_state: Arc<WorldState>,
    multiverse: TimelineMultiverse,
    snapshots: sg::SnapshotGraph,
    // TODO: Use a 'sorted map' since handles are linear...
    integration_cache: RwLock<HashMap<sg::NodeHandle, Arc<[sg::Snapshot]>>>,

    /// The first snapshot of all balls
    starts: Ro<Box<[sg::RootNodeHandle]>>,
    half_portals: Vec<HalfPortal>,

    timeline_timestamps: HashMap<TimelineId, TimestampList>,

    /// Cannot be changed without recomputing everything, i think
    max_time: Ro<f32>,
}

impl Simulator {
    pub fn new(world_state: Arc<WorldState>, max_time: f32) -> Self {
        let multiverse = TimelineMultiverse::new();

        let mut snapshots = sg::SnapshotGraph::new();
        let starts = world_state.balls.iter()
            .enumerate()
            .map(|(object_id, ball)| sg::RootSnapshot {
                object_id,

                pos: ball.initial_pos,
                rot: 0.,

                linvel: ball.initial_velocity,
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

        let timeline_timestamps = [
            (multiverse.root(), TimestampList::new())
        ].into();

        Self {
            world_state,
            multiverse,
            snapshots,
            integration_cache: default(),

            starts: starts.clone().into_boxed_slice().into(),
            half_portals,

            timeline_timestamps,

            max_time: Ro::new(max_time),
        }
    }

    pub fn empty() -> Self {
        Self::new(Arc::new(WorldState::new(0., 0.)), 0.)
    }

    pub fn snapshots(&self) -> &sg::SnapshotGraph {
        &self.snapshots
    }

    pub fn multiverse(&self) -> &TimelineMultiverse {
        &self.multiverse
    }

    pub fn half_portals(&self) -> &[HalfPortal] {
        &self.half_portals
    }

    // TODO: Find another name
    pub fn max_time(&self) -> f32 {
        *self.max_time
    }

    // pub fn is_snapshot_ghost(&self, snap: &sg::Snapshot) -> bool {
    //     snap.portal_traversals.iter()
    //         // only take finished traversal (we cant be fully ghost if we are still traversing the portal)
    //         .filter(|traversal| traversal.range.is_finished(snap.time))
    //         .any(|traversal| {
    //             let inv_trans = self.half_portals[traversal.half_portal_idx].transform.inverse();
    //             let rel_pos = inv_trans.transform_point2(snap.pos);
    //             (rel_pos.x < 0.) != traversal.direction.is_front()
    //         })
    // }

    // pub fn compute_snapshot_ghostification_time(&self, snap: &sg::Snapshot) -> Option<f32> {
    //     snap.portal_traversals.iter()
    //         .filter_map(|traversal| {
    //             // clamp end time to at least snap.time
    //             // as the traversal may have finished in the past
    //             // also
    //             // traversals that never finish can never make a ghost
    //             let et = traversal.range.end.max(snap.time);
    //             let dt = et - snap.time;
    //             let pos = snap.pos + snap.linvel * dt;

    //             let inv_trans = self.half_portals[traversal.half_portal_idx].transform.inverse();
    //             let rel_pos = inv_trans.transform_point2(pos);

    //             ((rel_pos.x < 0.) != traversal.direction.is_front())
    //                 .then_some(et)
    //         })
    //         .reduce(f32::min)
    // }

    fn integrate_root_node(
        &self,
        handle: sg::NodeHandle,
        root_snap: &sg::RootSnapshot,
    ) -> sg::Snapshot {
        sg::Snapshot {
            object_id: root_snap.object_id,
            handle,
            sub_id: 0,
            extrapolated_by: Positive::new(0.).expect("Positive"),

            timeline_id: self.multiverse.root(),
            timestamp: self.timeline_timestamps[&self.multiverse.root()].first_timestamp(),
            // All objects starts with age 0 at time 0
            age: Positive::new(0.).expect("Positive"),
            time: 0.,

            linvel: root_snap.linvel,
            angvel: root_snap.angvel,

            pos: root_snap.pos,
            rot: root_snap.rot,

            // computed later
            portal_traversals: smallvec![],
            force_transform: Affine2::IDENTITY,

            validity_time_range: TimeRange::from(..),
        }
    }

    fn integrate_inner_node(
        &self,
        handle: sg::NodeHandle,
        snapshot: &sg::Snapshot,
        partial: &sg::PartialSnapshot
    ) -> SmallVec<sg::Snapshot, 2> {
        let mut new_snapshot = snapshot.clone();
        new_snapshot.integrate_by(partial.delta_age);
        new_snapshot.handle = handle;

        // If a 'going out' is finished we are fully behind
        // a portal so the snapshot is now useless
        let is_ghost = new_snapshot.portal_traversals.iter()
            .filter(|traversal| traversal.traversal_direction.is_going_out())
            .any(|traversal| traversal.time_range.is_finished(new_snapshot.time));
        if is_ghost {
            return smallvec![];
        }

        new_snapshot.portal_traversals.retain(|traversal| {
            !traversal.time_range.is_finished(new_snapshot.time)
        });

        new_snapshot.timestamp = partial.new_timestamp;
        
        // Portal traversals only affect a single sub_id
        // whereas impulses affect all sub_id s
        let ghost_snapshot: Option<sg::Snapshot> = match &partial.delta {
            snapgraph::PartialSnapshotDelta::Impulse { linear, angular } => {
                new_snapshot.linvel += snapshot.force_transform.transform_vector2(*linear);
                new_snapshot.angvel += angular;
                let not_moving = new_snapshot.linvel.element_sum().abs() <= DEFAULT_EPSILON;

                // Re-compute the end of each traversals as change in velocity
                // means changes to when traversal ends
                for i in 0..new_snapshot.portal_traversals.len() {
                    let traversal = &new_snapshot.portal_traversals[i];
                    let Some((new_delta_start, new_delta_end, traversal_direction)) = self.cast_portal_traversal_start_end(&new_snapshot, traversal.half_portal_idx)
                    else { unreachable!("Should have been known/'detected' by the time_range before ?") };

                    let traversal = &mut new_snapshot.portal_traversals[i];
                    traversal.time_range = new_snapshot.time + new_delta_start..new_snapshot.time + new_delta_end;

                    if not_moving {
                        traversal.traversal_direction = sg::PortalTraversalDirection::NotMoving;
                    }
                    else if traversal_direction == traversal.direction {
                        traversal.traversal_direction = sg::PortalTraversalDirection::GoingIn;
                    }
                    else {
                        traversal.traversal_direction = sg::PortalTraversalDirection::GoingOut;
                    }
                }

                None
            },
            snapgraph::PartialSnapshotDelta::PortalTraversal { traversal }
                if traversal.sub_id_in == snapshot.sub_id ||
                    traversal.sub_id_out == snapshot.sub_id
            => {
                let mut ghost_snapshot = new_snapshot.clone();
                let half_portal = &self.half_portals[traversal.half_portal_idx];
                let out_half_portal = &self.half_portals[half_portal.linked_to];

                debug_assert_ne!(traversal.sub_id_in, traversal.sub_id_out);
                let (traversal_direction, other_id) = if traversal.sub_id_in == snapshot.sub_id {
                    (sg::PortalTraversalDirection::GoingIn, traversal.sub_id_out)
                } else {
                    (sg::PortalTraversalDirection::GoingOut, traversal.sub_id_in)
                };

                let traversal_direction = if traversal.delta_end.is_infinite() {
                    sg::PortalTraversalDirection::NotMoving
                } else {
                    traversal_direction
                };

                let transform = out_half_portal.transform * half_portal.transform.inverse();
                        
                new_snapshot.portal_traversals.push(sg::PortalTraversal {
                    half_portal_idx: traversal.half_portal_idx,
                    direction: traversal.direction,
                    traversal_direction,
                    time_range: new_snapshot.time..new_snapshot.time + traversal.delta_end.get(),
                });

                ghost_snapshot.sub_id = other_id;
                ghost_snapshot.time += half_portal.time_offset;
                ghost_snapshot.apply_force_transform(transform);
                ghost_snapshot.portal_traversals.push(sg::PortalTraversal {
                    half_portal_idx: half_portal.linked_to,
                    direction: traversal.direction.swap(),
                    traversal_direction: traversal_direction.swap(),
                    time_range: ghost_snapshot.time..ghost_snapshot.time + traversal.delta_end.get(),
                });

                Some(ghost_snapshot)
            },
            _ => None,
        };

        if let Some(additional) = ghost_snapshot {
            smallvec![new_snapshot, additional]
        }
        else {
            smallvec![new_snapshot]
        }
    }

    pub fn integrate(&self, handle: sg::NodeHandle) -> Arc<[sg::Snapshot]> {
        if let Some(snap) = self.integration_cache.read().get(&handle).map(Arc::clone) {
            return snap;
        }

        let out_snapshots = match &self.snapshots[handle] {
            sg::Node::Root(root_node) => {
                let root_snap = &root_node.snapshot;
                vec![self.integrate_root_node(handle, root_snap)]
            },
            sg::Node::Inner(inner_node) => {
                let snapshots = self.integrate(inner_node.previous);
                let partial = &inner_node.partial;

                // applies the 'partial' to all snapshots
                snapshots.iter()
                    .inspect(|snapshot| debug_assert_eq!(snapshot.handle, inner_node.previous))
                    .inspect(|snapshot| debug_assert_eq!(snapshot.extrapolated_by.get(), 0.))
                    .flat_map(|snapshot| self.integrate_inner_node(handle, snapshot, partial))
                    .inspect(|snapshot| debug_assert_eq!(snapshot.handle, handle))
                    .collect_vec()
            },
        };

        debug_assert!(out_snapshots.iter().map(|snap| snap.sub_id).all_unique(), "{out_snapshots:#?}");

        if !out_snapshots.is_empty() {
            println!("handle = {handle}, snaps: {}", out_snapshots.iter().join(" "));
        }

        return Arc::clone(self.integration_cache.write().entry(handle)
            .or_insert(out_snapshots.into_boxed_slice().into()));
    }

    pub fn get_node_max_dt(&self, node: &sg::Node, timeline_id: Option<TimelineId>) -> Option<Positive> {
        node.children().iter().copied()
            .filter(|&child_handle| timeline_id.is_none_or(|timeline_id| self.multiverse.is_parent(self.snapshots[child_handle].timeline_id(), timeline_id)))
            .map(|child_handle| self.snapshots[child_handle].partial.delta_age)
            .max()
    }

    pub fn get_node_next_timestamp(&self, node: &sg::Node, timeline_id: TimelineId) -> Option<Timestamp> {
        node.children().iter().copied()
            .filter(|&child_handle| self.multiverse.is_parent(self.snapshots[child_handle].timeline_id(), timeline_id))
            .map(|child_handle| self.snapshots[child_handle].partial.new_timestamp)
            .at_most_one().expect("Cannot be two children for the same timeline id")
    }

    /// Uses this node's children to find the duration for which it is valid
    #[deprecated]
    pub fn time_filtered_integrate(&self, handle: sg::NodeHandle, query_time: f32) -> impl Iterator<Item = sg::Snapshot> {
        let max_dt = self.get_node_max_dt(&self.snapshots[handle], None);
        let snaps = self.integrate(handle);
        (0..snaps.len()).into_iter()
            .filter_map(move |i| {
                let snap = &snaps[i];
                TimeRange::new(Some(snap.time), max_dt.map(|dt| snap.time + dt.get()))
                    .contains(&query_time)
                    .then(|| snap.clone())
            })
    }

    pub fn timestamp_filtered_integrate(&self, handle: sg::NodeHandle, timeline_id: TimelineId, timestamp: Timestamp) -> impl Iterator<Item = sg::Snapshot> {
        let next_ts = self.get_node_next_timestamp(&self.snapshots[handle], timeline_id);
        let snaps = self.integrate(handle);
        (0..snaps.len()).into_iter()
            .filter_map(move |i| {
                let snap = &snaps[i];
                (snap.timestamp <= timestamp && next_ts.is_none_or(|next_ts| timestamp < next_ts))
                    .then(|| snap.clone())
            })
    }

    pub fn time_query(&self, query_time: f32) -> impl Iterator<Item = sg::Snapshot> {
        debug_assert!(self.starts.iter().copied().all_unique());

        self.snapshots.nodes().flat_map(move |(handle, _)| self.time_filtered_integrate(handle, query_time))
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
                // Necessarily true as handle is a root
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

    fn snapshot_collision_shape(&self, snap: &sg::Snapshot) -> impl parry2d::shape::Shape {
        let radius = self.world_state.balls()[snap.object_id].radius;
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

    /// Computes when the given snapshot will start colliding with the portal
    /// and when it will stop, do not take into account the size of the 
    fn cast_portal_traversal_start_end(
        &self,
        snap: &sg::Snapshot,
        half_portal_idx: usize,
    ) -> Option<(f32, f32, PortalVelocityDirection)> {
        let collision_shape = self.snapshot_collision_shape(snap);
        debug_assert!(collision_shape.is_convex(), "Only convex shapes are supported");
        // bounding sphere is supposed to be rotation-independant
        let bounding_sphere = collision_shape.compute_local_bounding_sphere();
        let center = bounding_sphere.center.to_gl() + snap.pos;
        let radius = bounding_sphere.radius;

        let half_portal = &self.half_portals[half_portal_idx];
        let inv_portal_trans = half_portal.transform.inverse();

        let rel_center = inv_portal_trans.transform_point2(center);
        let rel_vel = inv_portal_trans.transform_vector2(snap.linvel);

        // |p + v * t| <= r
        // -(p + v * t) <= r <= (p + v * t)
        // -(p + v * t) <= r && -(p + v * t) >= r

        let p = rel_center.x;
        let v = rel_vel.x;
        let r = radius;

        // v ~= 0
        if v.abs() <= DEFAULT_EPSILON {
            // Touching and always touching
            if v <= r {
                return Some((f32::NEG_INFINITY, f32::INFINITY, PortalVelocityDirection::NotMoving));
            }
            else {
                return None;
            }
        }

        let t0 = (-r - p) / v;
        let t1 = ( r - p) / v;

        if v < 0. {
            Some((t1, t0, PortalVelocityDirection::IntoFront))
        }
        else /* v > 0. */ {
            Some((t0, t1, PortalVelocityDirection::IntoBack))
        }
    }

    fn cast_portal_traversal(
        &self,
        snap: &sg::Snapshot,
        half_portal_idx: usize,
        dt_range: Range<f32>,
    ) -> Option<NewPortalTraversalData> {
        // ignore if already traversing
        if snap.portal_traversals.iter().any(|traversal| traversal.half_portal_idx == half_portal_idx) {
            return None;
        }

        let collision_shape = self.snapshot_collision_shape(snap);
        debug_assert!(collision_shape.is_convex(), "Only convex shapes are supported");

        let half_portal = &self.half_portals[half_portal_idx];

        let (t0, t1, moving_direction) = self.cast_portal_traversal_start_end(snap, half_portal_idx)?;

        println!("t0, t1: {t0}..{t1}");
        let traversal_range = t0..t1;
        let TimeRange::Range(range_overlap) = TimeRange::from(dt_range) & TimeRange::from(traversal_range)
        else { unreachable!() };
        println!("range overlap: {range_overlap:?}");
        if range_overlap.is_empty() {
            return None;
        }

        let h2 = half_portal.height/2.;
        let portal_shape = parry2d::shape::Polyline::new(vec![
            half_portal.transform.transform_point2(Vec2::new(0., -h2)).to_na(),
            half_portal.transform.transform_point2(Vec2::new(0., h2)).to_na(),
        ], None);

        let collision = parry2d::query::cast_shapes_nonlinear(
            &self.snapshot_motion(snap), &collision_shape,
            &parry2d::query::NonlinearRigidMotion::identity(), &portal_shape,
            range_overlap.start, range_overlap.end,
            true
        ).expect("Compatible")?;

        let direction = if collision.normal1.x > 0. {
            PortalDirection::Front
        } else {
            PortalDirection::Back
        };

        // The collision is a better 'estimate' of when the traversal starsts
        let delta_range = Positive::new(collision.time_of_impact).expect("positive")..Positive::new(t1).expect("positive");
        let range = delta_range.start.get() + snap.time..delta_range.end.get() + snap.time;

        debug_assert!(delta_range.start < delta_range.end);

        Some(NewPortalTraversalData {
            half_portal_idx,
            direction,
            traversal_direction: if moving_direction.is_not_moving() {
                sg::PortalTraversalDirection::NotMoving
            } else if moving_direction == direction {
                sg::PortalTraversalDirection::GoingIn
            } else {
                sg::PortalTraversalDirection::GoingOut
            },
            delta_range,
            range,
        })
    }

    fn cast_ball_wall_collision(
        &self,
        snap: &sg::Snapshot,
        dt_range: Range<f32>,
    ) -> Option<Collision<1>> {
        let walls_shape = self.world_state.get_static_body_collision();
        let walls_shape = snap.portal_traversals.iter()
            .fold(walls_shape, |walls_shape, traversal|
                clip_shapes_on_portal(walls_shape, self.half_portals[traversal.half_portal_idx].transform, traversal.direction)
            );
        let walls_shape = i_shape_to_parry_shape(walls_shape)?;
        let ball_shape = self.snapshot_collision_shape(snap);

        let Some(collision) = parry2d::query::cast_shapes_nonlinear(
            &parry2d::query::NonlinearRigidMotion::identity(), &walls_shape,
            &self.snapshot_motion(snap), &ball_shape,
            dt_range.start, dt_range.end, false,
        ).expect("Supported") else {
            // No collision detected
            return None;
        };

        let impact_normal = Vec2::new(collision.normal1.x, collision.normal1.y);
        let impact_dt = collision.time_of_impact;
        let linacc = - 2.0 * snap.linvel.dot(impact_normal) * impact_normal;

        Some(Collision {
            debug_reason: "ball-wall",
            impact_time: snap.time + impact_dt,
            impact_delta_time: Positive::new(impact_dt).expect("Positive"),
            states: [CollisionNewState { linear_impulse: linacc, ..default() }]
        })
    }

    fn cast_ball_ball_collision(
        &self,
        s1: &sg::Snapshot,
        s2: &sg::Snapshot,
        dt_range: Range<f32>,
    ) -> Option<Collision<2>> {
        debug_assert!((s1.time - s2.time).abs() <= DEFAULT_EPSILON, "{} != {}", s1.time, s2.time);
        debug_assert!(self.multiverse.is_related(s1.timeline_id, s2.timeline_id));

        let rad1 = self.world_state.balls()[s1.object_id].radius;
        let rad2 = self.world_state.balls()[s2.object_id].radius;

        let Some(collision) = parry2d::query::cast_shapes_nonlinear(
            &self.snapshot_motion(s1), &self.snapshot_collision_shape(s1),
            &self.snapshot_motion(s2), &self.snapshot_collision_shape(s2),
            dt_range.start, dt_range.end, false,
        ).expect("Supported") else {
            return None;
        };

        let impact_dt = collision.time_of_impact;
        let collision_normal = collision.normal1.to_gl();
        let relative_velocity = s1.linvel - s2.linvel;
        let velocity_along_normal = relative_velocity.dot(collision_normal);
        
        // Compute masses from radii (equal density) and elastic impulse
        let m1 = rad1 * rad1;
        let m2 = rad2 * rad2;
        let impulse_scalar = 2.0 * m1 * m2 / (m1 + m2) * velocity_along_normal;
        let impulse = impulse_scalar * collision_normal;
        
        Some(Collision {
            debug_reason: "ball-ball",
            impact_time: s1.time + impact_dt,
            impact_delta_time: Positive::new(impact_dt).expect("Positive"),
            states: [
                CollisionNewState { linear_impulse: -impulse / m1, ..default() },
                CollisionNewState { linear_impulse:  impulse / m2, ..default() },
            ],
        })
    }

    fn apply_collision<const N: usize>(&mut self, event: CollisionSimulationEvent<N>) {
        assert!(event.snaps.iter().map(|snap| snap.handle).all_unique(), "Self-collision of a single handle is not yet implemented");

        // TEMP: Maybe possible with time travel? (i think not)
        debug_assert!(event.snaps.iter().map(|snap| (snap.object_id, snap.sub_id)).all_unique());

        let child_timeline = event.child_timeline(&self.multiverse);
        let parent_timeline = event.parent_timeline(&self.multiverse);

        let new_timestamp = if child_timeline != parent_timeline {
            todo!()
        }
        else {
            self.timeline_timestamps.get_mut(&parent_timeline)
                .expect("Exists")
                .push(TimestampDelta {
                    delta_t: event.col.impact_delta_time,
                })
        };

        for i in 0..N {
            // TEMP: This can only happen whith timeline branches
            debug_assert!(self.snapshots[event.snaps[i].handle].children().is_empty());

            let invforcetrans = event.snaps[i].force_transform.inverse();
            let new_partial = sg::PartialSnapshot {
                delta_age: event.snaps[i].extrapolated_by + event.col.impact_delta_time,
                new_timestamp,
                delta: sg::PartialSnapshotDelta::Impulse {
                    linear: invforcetrans.transform_vector2(event.col.states[i].linear_impulse) ,
                    angular: event.col.states[i].angular_impulse,
                },
            };

            println!("[{}/{N}] {new_partial:?}", i + 1);
            let n = self.snapshots.insert(new_partial, event.snaps[i].handle);
            println!("Created handle {n}");
        }
    }

    fn apply_portal_traversal(&mut self, event: PortalTraversalSimulationEvent) {
        debug_assert!(!event.snap.portal_traversals.iter()
            .any(|traversal| traversal.half_portal_idx == event.data.half_portal_idx));

        let snap_sub_id = event.snap.sub_id;
        // TODO: Maybe just a random sub_id should be good
        // Generate a new sub_id by just incrementing from the max one
        let gened_sub_id = self.integrate(event.snap.handle).iter()
            .map(|snap| snap.sub_id).max().unwrap_or(0) + 1;

        let (sub_id_in, sub_id_out) = if event.data.traversal_direction.is_going_in() {
            (snap_sub_id, gened_sub_id)
        } else {
            (gened_sub_id, snap_sub_id)
        };

        let new_timestamp = self.timeline_timestamps.get_mut(&event.snap.timeline_id)
            .expect("Exists")
            .push(TimestampDelta {
                delta_t: event.data.delta_range.start,
            });

        let new_partial = sg::PartialSnapshot {
            delta_age: event.snap.extrapolated_by + event.data.delta_range.start,
            new_timestamp,
            delta: sg::PartialSnapshotDelta::PortalTraversal {
                traversal: sg::PartialPortalTraversal {
                    half_portal_idx: event.data.half_portal_idx,
                    direction: event.data.direction,
                    // FIXME: Is this correct?
                    delta_end: Positive::new(event.data.delta_range.end - event.data.delta_range.start).expect("positive"),
                    sub_id_in,
                    sub_id_out,
                },
            }
        };
        println!("[1/1] {new_partial:?}");
        let n = self.snapshots.insert(new_partial, event.snap.handle);
        println!("Created handle {n}");
    }

    fn apply_simulation_event(&mut self, info: GenericSimulationEventInfo) {
        match info {
            GenericSimulationEventInfo::OneCollision(event) => {
                self.apply_collision(event);
            },
            GenericSimulationEventInfo::TwoCollision(event) => {
                self.apply_collision(event);
            },
            GenericSimulationEventInfo::PortalTraversal(event) => {
                self.apply_portal_traversal(event);
            },
        }
    }

    pub fn step(&mut self) -> ControlFlow<(), ()> {
        const MAX_DT: f32 = 1.;

        // make a non-mutable reference, so a `Copy` reference, for move closures
        let this = &*self;

        println!();

        if self.timeline_timestamps.iter()
            .flat_map(|(_, list)| list.iter().map(|(_, data)| data.running_sum.get()))
            .reduce(f32::min)
            .unwrap_or(*self.max_time) >= *self.max_time {
            println!("FINISHED max time is {}", *self.max_time);
            return ControlFlow::Break(());
        }

        let leafs = self.snapshots.leafs().iter().copied()
            .map(|handle| this.integrate(handle))
            .flat_map(|snaps| {
                (0..snaps.len()).into_iter()
                .filter_map(move |i| {
                    let tss = &this.timeline_timestamps[&snaps[i].timeline_id];
                    let last_ts = tss.last_timestamp();
                    let last_t = tss[last_ts].running_sum.get();
                    if last_t >= *this.max_time {
                        return None;
                    }
                    snaps[i].extrapolate_to_timestamp(last_t, last_ts)
                })
            })
            .collect_vec()
        ;

        if leafs.is_empty() {
            println!("FINISHED: NO LEAFS");
            return ControlFlow::Break(());
        }

        println!("leafs: {}", leafs.iter().join(", "));

        let worlds: HashMap<TimelineId, Vec<sg::Snapshot>> = leafs.iter()
            .map(|snap| snap.timeline_id)
            .unique()
            .map(|tid| {
                let tss = &self.timeline_timestamps[&tid];
                let last_ts = tss.last_timestamp();

                let world = this.timeline_query(tid)
                    .flat_map(move |handle_| 
                        this.timestamp_filtered_integrate(handle_, tid, last_ts)
                        .filter_map(move |snap| snap.extrapolate_to_timestamp(tss[last_ts].running_sum.get(), last_ts))
                    )
                    .collect::<Vec<_>>();
                (tid, world)
            })
            .collect();

        println!("{}", worlds.iter().map(|(k, v)| (k, v.iter().join(", "))).map(|(k, v)| format!("tid {k} ({}s) -> {v}", self.timeline_timestamps[k].last().running_sum)).join("\n"));

        let groups = leafs.iter()
            .flat_map(|snap| {
                let world = &worlds[&snap.timeline_id];
                zip(
                    repeat(snap),
                    world.iter().filter(|snap_| snap.id() != snap_.id())
                )
            })
            .collect::<Vec<_>>()
        ;

        println!("ball-ball group count: {}", groups.len());
        println!("ball-ball groups: {}", groups.iter().map(|(s1, s2)| format!("{s1} and {s2}")).join(", "));

        let collision = empty()
            // ball-portal events
            .chain(
                leafs.iter()
                .cartesian_product(0..self.half_portals.len())
                .filter_map(|(snap, half_portal_idx)| {
                    let end = f32::min(*self.max_time - snap.time, MAX_DT);
                    debug_assert!(end >= 0.);
                    self.cast_portal_traversal(snap, half_portal_idx, 0. .. end)
                    .map(|data| PortalTraversalSimulationEvent {
                        data,
                        snap,
                    })
                    .map(GenericSimulationEventInfo::from)
                })
            )
            // ball-wall collisions
            .chain(
                leafs.iter()
                .filter_map(|snap| {
                    let end = f32::min(*self.max_time - snap.time, MAX_DT);
                    debug_assert!(end >= 0.);
                    self.cast_ball_wall_collision(snap, 0. .. end)
                    .map(|col| CollisionSimulationEvent {
                        col,
                        snaps: [snap],
                    })
                    .map(GenericSimulationEventInfo::from)
                })
            )
            // ball-ball collisions
            .chain(
                groups.iter().copied()
                .filter_map(|(s1, s2)| {
                    debug_assert!((s1.time - s2.time).abs() <= DEFAULT_EPSILON, "{} != {}", s1.time, s2.time);
                    let end = f32::min(*self.max_time - s1.time, MAX_DT);
                    debug_assert!(end >= 0.);
                    self.cast_ball_ball_collision(s1, s2, 0. .. end)
                    .map(|col| CollisionSimulationEvent {
                        col,
                        snaps: [s1, s2],
                    })
                    .map(GenericSimulationEventInfo::from)
                })
            )
            .inspect(|col| debug_assert!(col.impact_delta_time() <= MAX_DT))
            // .filter(|col| col.impact_delta_time() < *self.max_time)
            .min_by_key(|col| (col.parent_timeline(&self.multiverse), OF(col.impact_time())))
        ;

        let Some(collision) = collision
        else {
            println!("NO COLLISION FOUND");
            for tid in leafs.iter().map(|snap| snap.timeline_id).unique() {
                let tss = self.timeline_timestamps.get_mut(&tid).expect("exists");
                tss.push(TimestampDelta { delta_t: Positive::new(MAX_DT).expect("positive") });
            }
            return ControlFlow::Continue(())
        };
        println!("Selected collision: {}", collision.simple_debug());

        assert_eq!(collision.parent_timeline(&self.multiverse), collision.child_timeline(&self.multiverse), "Unsupported yet");
        // assert!(self.timeline_presents[&tid] <= collision.impact_time(), "assert({} <= {}) ({collision:#?})", self.timeline_presents[&tid], collision.impact_time());

        self.apply_simulation_event(collision);
        
        ControlFlow::Continue(())
    }

    pub fn run(&mut self) {
        while self.step().is_continue() { }
    }
}
