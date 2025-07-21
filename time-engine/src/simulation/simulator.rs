use std::{
    cmp::{max, min}, collections::{HashMap, HashSet}, convert::identity, iter::{ empty, repeat, zip }, ops::ControlFlow, range::Range, sync::Arc
};

use glam::{ Affine2, Vec2 };
use itertools::{chain, izip, multizip, Itertools};
use parry2d::shape::Shape;
use ordered_float::OrderedFloat as OF;

use crate::{sg::DeprecatedTimelineIdDummy as _, timestamp_graph::TimestampGraphExt};

use super::*;

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
    #[deprecated]
    #[expect(deprecated)]
    fn child_timeline(&self, _multiverse: &TimelineMultiverse) -> TimelineId {
        unimplemented!()
        // assert!(N > 0);
        // debug_assert!(
        //     self.snaps.iter().map(|snap| snap.timeline_id)
        //         .tuple_windows::<(_, _)>()
        //         .all(|(a, b)| multiverse.is_related(a, b))
        // );
        // self.snaps.iter()
        //     .map(|snap| snap.timeline_id)
        //     .max()
        //     .expect("No empty")
    }

    #[deprecated]
    #[expect(deprecated)]
    fn parent_timeline(&self, _multiverse: &TimelineMultiverse) -> TimelineId {
        unimplemented!()
        // assert!(N > 0);
        // debug_assert!(
        //     self.snaps.iter().map(|snap| snap.timeline_id)
        //         .tuple_windows::<(_, _)>()
        //         .all(|(a, b)| multiverse.is_related(a, b))
        // );
        // self.snaps.iter()
        //     .map(|snap| snap.timeline_id)
        //     .min()
        //     .expect("No empty")
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
struct GhostificationData {
    delta_time: Positive,
    time: f32,
}

#[derive(Debug, Clone)]
struct GhostificationSimulationEvent<'a> {
    data: GhostificationData,
    snap: &'a sg::Snapshot,
}

#[derive(Debug, Clone)]
struct AlreadingExistingNodeData {
    /// Must be a children of the event's snapshot
    next_node: sg::NodeHandle,
    delta_time: Positive,
    time: f32,
}

#[derive(Debug, Clone)]
struct AlreadyExistingNodeEvent<'a> {
    data: AlreadingExistingNodeData,
    snap: &'a sg::Snapshot,
}

#[derive(Debug, Clone)]
enum GenericSimulationEventInfo<'a> {
    OneCollision(CollisionSimulationEvent<'a, 1>),
    TwoCollision(CollisionSimulationEvent<'a, 2>),
    PortalTraversal(PortalTraversalSimulationEvent<'a>),
    Ghostification(GhostificationSimulationEvent<'a>),
    AlreadyExistingNode(AlreadyExistingNodeEvent<'a>),
}

impl<'a> GenericSimulationEventInfo<'a> {
    fn impact_delta_time(&self) -> Positive {
        match self {
            Self::OneCollision(e) => e.col.impact_delta_time,
            Self::TwoCollision(e) => e.col.impact_delta_time,
            Self::PortalTraversal(e) => e.data.delta_range.start,
            Self::Ghostification(e) => e.data.delta_time,
            Self::AlreadyExistingNode(e) => e.data.delta_time,
        }
    }

    // TODO: Deprecate and remove the absolute time from the event datas
    fn impact_time(&self) -> f32 {
        match self {
            Self::OneCollision(e) => e.col.impact_time,
            Self::TwoCollision(e) => e.col.impact_time,
            Self::PortalTraversal(e) => e.data.range.start,
            Self::Ghostification(e) => e.data.time,
            Self::AlreadyExistingNode(e) => e.data.time,
        }
    }

    fn simple_debug(&self) -> String {
        match self {
            Self::OneCollision(e) => {
                format!("one({}, dt {}s, {})", e.col.debug_reason, e.col.impact_delta_time, e.snaps[0])
            },
            Self::TwoCollision(e) => {
                format!("two({}, dt {}s, {} and {})", e.col.debug_reason, e.col.impact_delta_time, e.snaps[0], e.snaps[1])
            },
            Self::PortalTraversal(e) => {
                format!("portal_traversal(range {:?}, {} on portal {})", e.data.delta_range, e.snap, e.data.half_portal_idx)
            },
            Self::Ghostification(e) => {
                format!("ghostification({} after {}s)", e.snap, e.data.delta_time)
            },
            Self::AlreadyExistingNode(e) => {
                format!("existing_node({} -> {})", e.snap, e.data.next_node)
            },
        }
    }

    #[deprecated]
    #[expect(deprecated)]
    fn child_timeline(&self, _multiverse: &TimelineMultiverse) -> TimelineId {
        unimplemented!()
        // match self {
        //     Self::OneCollision(event) => event.child_timeline(multiverse),
        //     Self::TwoCollision(event) => event.child_timeline(multiverse),
        //     Self::PortalTraversal(event) => event.snap.timeline_id,
        //     Self::Ghostification(event) => event.snap.timeline_id,
        // }
    }

    #[deprecated]
    #[expect(deprecated)]
    fn parent_timeline(&self, _multiverse: &TimelineMultiverse) -> TimelineId {
        unimplemented!()
        // match self {
        //     Self::OneCollision(event) => event.parent_timeline(multiverse),
        //     Self::TwoCollision(event) => event.parent_timeline(multiverse),
        //     Self::PortalTraversal(event) => event.snap.timeline_id,
        //     Self::Ghostification(event) => event.snap.timeline_id,
        // }
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
    pub(super) world_state: Arc<WorldState>,
    pub(super) snapshots: sg::SnapshotGraph,
    pub(super) timestamps: tsg::TimestampGraph,

    pub(super) initial_timestamp: tsg::Timestamp,

    /// The first snapshot of all balls
    #[deprecated]
    pub(super) starts: Ro<Box<[sg::RootNodeHandle]>>,
    pub(super) half_portals: Vec<HalfPortal>,

    /// Cannot be changed without recomputing everything, i think
    pub(super) max_time: Ro<f32>,
}

impl Simulator {
    pub fn new(world_state: Arc<WorldState>, max_time: f32) -> Self {
        let mut snapshots = sg::SnapshotGraph::new();
        world_state.balls.iter()
            .enumerate()
            .map(|(object_id, ball)| sg::RootSnapshot {
                object_id,

                pos: ball.initial_pos,
                rot: 0.,

                linvel: ball.initial_velocity,
                angvel: 0.,
            })
            .for_each(|snapshot| {
                snapshots.insert_root(&mut (), snapshot);
            });

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

        let mut timestamps = tsg::TimestampGraph::new();

        let initial_timestamp = timestamps.insert_root(
            &mut (),
            tsg::TimestampRoot {
                time: 0.,
                // all nodes are leafs at this point
                handles: snapshots.leafs().collect(),
            }
        ).into();

        Self {
            world_state,
            snapshots,
            timestamps: tsg::TimestampGraph::new(),

            initial_timestamp,

            #[expect(deprecated)]
            starts: Ro::new(default()),
            half_portals,

            max_time: Ro::new(max_time),
        }
    }

    pub fn empty() -> Self {
        Self::new(Arc::new(WorldState::new(0., 0.)), 0.)
    }

    pub fn snapshots(&self) -> &sg::SnapshotGraph {
        &self.snapshots
    }

    #[deprecated]
    #[expect(deprecated)]
    pub fn multiverse(&self) -> &TimelineMultiverse {
        unimplemented!()
    }

    pub fn half_portals(&self) -> &[HalfPortal] {
        &self.half_portals
    }

    // TODO: Find another name
    pub fn max_time(&self) -> f32 {
        *self.max_time
    }

    pub fn integrate_snapshot(&self, handle: sg::NodeHandle) -> Arc<[sg::Snapshot]> {
        self.snapshots.integrate(&mut &*self, handle)
    }

    pub fn integrate_timestamp(&self, timestamp: tsg::Timestamp) -> Arc<tsg::TimestampData> {
        self.timestamps.integrate(&mut (), timestamp)
    }

    #[deprecated]
    #[expect(deprecated)]
    pub fn get_node_max_dt(&self, node: sg::NodeRef, timeline_id: Option<TimelineId>) -> Option<Positive> {
        node.children().iter().copied()
            .filter(|&child_handle| timeline_id.is_none_or(|timeline_id| self.multiverse().is_parent(self.snapshots[child_handle].timeline_id(), timeline_id)))
            .map(|child_handle| self.snapshots[child_handle].data.delta_age)
            .max()
    }

    #[deprecated]
    #[expect(deprecated)]
    pub fn get_node_next_timestamp(&self, _node: sg::NodeRef, _timeline_id: TimelineId) -> Option<Timestamp> {
        unimplemented!()
    }

    /// Uses this node's children to find the duration for which it is valid
    #[deprecated]
    #[expect(deprecated)]
    pub fn time_filtered_integrate(&self, _handle: sg::NodeHandle, _query_time: f32) -> impl Iterator<Item = sg::Snapshot> {
        unimplemented!();
        empty()
        // let max_dt = self.get_node_max_dt(self.snapshots.get(handle), None);
        // let snaps = self.integrate_snapshot(handle);
        // (0..snaps.len()).into_iter()
        //     .filter_map(move |i| {
        //         let snap = &snaps[i];
        //         TimeRange::new(Some(snap.time), max_dt.map(|dt| snap.time + dt.get()))
        //             .contains(&query_time)
        //             .then(|| snap.clone())
        //     })
    }

    #[deprecated]
    #[expect(deprecated)]
    pub fn timestamp_filtered_integrate(&self, _handle: sg::NodeHandle, _timeline_id: TimelineId, _timestamp: Timestamp) -> impl Iterator<Item = sg::Snapshot> {
        unimplemented!();
        empty()
        // let next_ts = self.get_node_next_timestamp(self.snapshots.get(handle), timeline_id);
        // let snaps = self.integrate_snapshot(handle);
        // (0..snaps.len()).into_iter()
        //     .filter_map(move |i| {
        //         let snap = &snaps[i];
        //         (snap.timestamp_old <= timestamp && next_ts.is_none_or(|next_ts| timestamp < next_ts))
        //             .then(|| snap.clone())
        //     })
    }

    pub fn time_query(&self, query_time: f32) -> impl Iterator<Item = sg::Snapshot> {
        debug_assert!(self.starts.iter().copied().all_unique());

        self.snapshots.nodes()
            .flat_map(move |node|
                self.time_filtered_integrate(node.handle(), query_time)
            )
    }

    /// Return an iterator of all nodes that are in the given timeline
    #[deprecated]
    #[expect(deprecated)]
    pub fn timeline_query(&self, timeline_id: TimelineId) -> impl Iterator<Item = sg::NodeHandle> + Clone + '_ {
        #[derive(Debug, Clone, Copy)]
        struct StackEntry {
            handle: sg::NodeHandle,
            is_in_timeline: bool,
        }

        let mut stack = self.starts.iter().copied()
            .inspect(|&link| debug_assert_eq!(self.snapshots[link].timeline_id(), self.multiverse().root()))
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
                        .filter_map(|child_handle| self.multiverse().distance(self.snapshots[child_handle].timeline_id(), timeline_id)
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
                        .filter(|&(_, child_timeline_id)| self.multiverse().is_parent(child_timeline_id, timeline_id))
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

    /// DUMMY function just for documenting the algorithm
    #[deprecated = "Do not use, useless"]
    #[expect(dead_code)]
    #[allow(warnings)]
    fn integrate_snapshot_timestamps(&self, handle: sg::NodeHandle) -> Vec<tsg::Timestamp> {
        let ancestry = {
            let mut ancestry = self.snapshots.iter_ancestry(handle).collect_vec();
            ancestry.reverse();
            ancestry
        };

        debug_assert!(ancestry[0].handle().is_root());

        ancestry.iter()
        // skip the root node that do not do anything
        .skip(1)
        .fold(vec![self.initial_timestamp], |mut timestamps, node| {
            let sg::NodeRef::Inner(node) = node
            else { unreachable!() };

            let delta_t = node.data.delta_age;

            // timestamps = timestamps.iter().copied()
            //     .flat_map(|ts| self.timestamps.advance(ts, delta_t))
            //     .collect()
            // ;
            
            timestamps
        })
    }

    pub fn timestamp_query(&self, timestamp: tsg::Timestamp) -> impl Iterator<Item = sg::Snapshot> + Clone + '_ {
        #[derive(Debug, Clone, Copy)]
        struct StackEntry {
            handle: sg::NodeHandle,
            timestamp: tsg::Timestamp,
        }

        let mut stack = self.snapshots.root_nodes()
            .map(|node| StackEntry {
                handle: node.root_handle.into(),
                timestamp: self.initial_timestamp,
            })
            .collect_vec();

        let iter = std::iter::from_fn(move || {
            debug_assert!(stack.iter().copied()
                .map(|entry| (entry.handle, entry.timestamp))
                .all_unique());

            while let Some(StackEntry {
                handle,
                timestamp: current_timestamp,
            }) = stack.pop() {
                let node = self.snapshots.get(handle);

            }

            None::<std::iter::Empty<_>>
        });


        iter.flatten()
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
    pub(super) fn cast_portal_traversal_start_end(
        &self,
        snap: &sg::Snapshot,
        half_portal_idx: usize,
    ) -> Option<(f32, f32, PortalDirection, PortalVelocityDirection)> {
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

        let direction = if rel_center.x <= 0. {
            PortalDirection::Front
        } else {
            PortalDirection::Back
        };

        // v ~= 0
        if v.abs() <= DEFAULT_EPSILON {
            // Touching and always touching
            if v <= r {
                return Some((f32::NEG_INFINITY, f32::INFINITY, direction, PortalVelocityDirection::NotMoving));
            }
            else {
                return None;
            }
        }

        let t0 = (-r - p) / v;
        let t1 = ( r - p) / v;

        if v < 0. {
            Some((t1, t0, direction, PortalVelocityDirection::IntoBack))
        }
        else /* v > 0. */ {
            Some((t0, t1, direction, PortalVelocityDirection::IntoFront))
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

        let (t0, t1, direction, velocity_direction) = self.cast_portal_traversal_start_end(snap, half_portal_idx)?;

        let traversal_range = t0..t1;
        let TimeRange::Range(range_overlap) = TimeRange::from(dt_range) & TimeRange::from(traversal_range)
        else { unreachable!() };
        if range_overlap.is_empty() {
            return None;
        }

        // When just touching a portal but with a velocity that makes it quickly
        // untouch the portal we ignore this very small traversal
        if range_overlap.end <= DEFAULT_EPSILON {
            return None;
        }

        let h2 = half_portal.height/2.;
        let portal_shape = parry2d::shape::Polyline::new(
            vec![Vec2::new(0., -h2).to_na(), Vec2::new(0., h2).to_na()],
            None
        );

        let collision = parry2d::query::cast_shapes_nonlinear(
            &parry2d::query::NonlinearRigidMotion::constant_position(
                affine_to_isometry(half_portal.transform)
            ), &portal_shape,
            &self.snapshot_motion(snap), &collision_shape,
            range_overlap.start, range_overlap.end,
            true
        ).expect("Compatible")?;

        // The collision is a better 'estimate' of when the traversal starsts
        let delta_range = Positive::new(collision.time_of_impact).expect("positive")..Positive::new(t1).expect("positive");
        let range = delta_range.start.get() + snap.time..delta_range.end.get() + snap.time;

        debug_assert!(delta_range.start < delta_range.end);

        Some(NewPortalTraversalData {
            half_portal_idx,
            direction,
            traversal_direction: sg::PortalTraversalDirection::from_velocity_direction(velocity_direction, direction),
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
        // TODO: Same but with timestamps
        // debug_assert!(self.multiverse.is_related(s1.timeline_id, s2.timeline_id));

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

    fn cast_ball_ghostification(
        &self,
        snap: &sg::Snapshot,
        dt_range: Range<f32>,
    ) -> Option<GhostificationData> {
        snap.portal_traversals.iter()
            .filter(|traversal| traversal.traversal_direction.is_going_in())
            .map(|traversal| traversal.time_range.end)
            .filter(|&time| dt_range.contains(&(time - snap.time)))
            .reduce(f32::min)
            .map(|end| GhostificationData {
                delta_time: Positive::new(end - snap.time).expect("Positive"),
                time: end,
            })
    }

    fn apply_collision<const N: usize>(&mut self, event: CollisionSimulationEvent<N>) {
        assert!(event.snaps.iter().map(|snap| snap.handle).all_unique(), "Self-collision of a single handle is not yet implemented");

        // TEMP: Maybe possible with time travel? (i think not)
        debug_assert!(event.snaps.iter().map(|snap| (snap.object_id, snap.sub_id)).all_unique());

        for i in 0..N {
            let handle = event.snaps[i].handle;

            // TEMP: This can only happen whith timeline branches
            debug_assert!(self.snapshots[handle].children().is_empty());

            let invforcetrans = event.snaps[i].force_transform.inverse();
            let new_partial = sg::PartialSnapshot {
                delta_age: event.snaps[i].extrapolated_by + event.col.impact_delta_time,
                delta: sg::PartialSnapshotDelta::Impulse {
                    linear: invforcetrans.transform_vector2(event.col.states[i].linear_impulse) ,
                    angular: event.col.states[i].angular_impulse,
                },
            };

            println!("[{}/{N}] {new_partial:?}", i + 1);
            let n = self.snapshots.insert(&mut (), new_partial, handle);
            println!("Created handle {n}");
        }
    }

    fn apply_portal_traversal(&mut self, event: PortalTraversalSimulationEvent) {
        debug_assert!(!event.snap.portal_traversals.iter()
            .any(|traversal| traversal.half_portal_idx == event.data.half_portal_idx));

        let output_timestamp: tsg::Timestamp = todo!();

        let new_partial = sg::PartialSnapshot {
            delta_age: event.snap.extrapolated_by + event.data.delta_range.start,
            delta: sg::PartialSnapshotDelta::PortalTraversal {
                traversal: sg::PartialPortalTraversal {
                    half_portal_idx: event.data.half_portal_idx,
                    direction: event.data.direction,
                    // FIXME: Is this correct?
                    duration: Positive::new(event.data.delta_range.end - event.data.delta_range.start).expect("positive"),
                    sub_id: event.snap.sub_id,
                    traversal_direction: event.data.traversal_direction,
                    output_branch_timestamp: output_timestamp,
                },
            }
        };
        println!("[1/1] {new_partial:?}");
        let n = self.snapshots.insert(&mut (), new_partial, event.snap.handle);
        println!("Created handle {n}");
    }

    fn apply_ghostification(&mut self, event: GhostificationSimulationEvent) {
        let new_partial = sg::PartialSnapshot {
            delta_age: event.snap.extrapolated_by + event.data.delta_time,
            delta: sg::PartialSnapshotDelta::Ghostification { sub_id: event.snap.sub_id },
        };
        println!("[1/1] {new_partial:?}");
        let n = self.snapshots.insert(&mut (), new_partial, event.snap.handle);
        println!("Created handle {n}");
    }

    fn apply_existing_node(&mut self, event: AlreadyExistingNodeEvent) {
        todo!("{event:?}");
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
            GenericSimulationEventInfo::Ghostification(event) => {
                self.apply_ghostification(event);
            },
            GenericSimulationEventInfo::AlreadyExistingNode(event) => {
                self.apply_existing_node(event);
            },
        }
    }

    pub fn step(&mut self) -> ControlFlow<(), ()> {
        const MAX_DT: f32 = 1.;

        #[derive(Debug, Clone)]
        struct TimestampSnapshot {
            snapshot: sg::Snapshot,
            /// How 'late' the timestamp is from its timestamp
            timestamp_delta_t: Positive,
            is_leaf: bool,
        }

        let timestamps: HashMap<tsg::Timestamp, Vec<TimestampSnapshot>> = self.timestamps.leafs()
            .flat_map(|timestamp| zip(
                repeat(timestamp),
                cloned_arc_slice_iter(self.integrate_timestamp(timestamp), |ts| &ts.snapshot_links, tsg::TimestampSnapshotLink::clone),
            ))
            .map(|(timestamp, link)| (
                timestamp,
                cloned_arc_slice_iter(self.integrate_snapshot(link.snapshot_handle), |x| x, sg::Snapshot::clone)
                    .map(|snapshot| TimestampSnapshot {
                        is_leaf: self.snapshots.is_leaf(snapshot.handle),
                        timestamp_delta_t: link.delta_since_add,
                        snapshot,
                    })
                    .collect_vec()
            ))
            .collect::<HashMap<_, _>>()
        ;

        #[cfg(debug_assertions)]
        for (timestamp, snapshot) in timestamps.iter()
            .map(|(timestamp, snapshots)| (self.integrate_timestamp(*timestamp), snapshots))
            .flat_map(|(timestamp, snapshots)| zip(repeat(timestamp), snapshots))
        {
            let t1 = snapshot.snapshot.time;
            let t2 = timestamp.time;
            debug_assert!(t1 <= t2, "{t1} <= {t2} should be true for {snapshot:#?} and {timestamp:#?}");
        }

        #[derive(Debug, Clone)]
        struct SnapshotLeaf<'a> {
            snap: &'a sg::Snapshot,
            min_delta_t: Positive,
            timestamps: HashSet<tsg::Timestamp>,
        }

        let snap_leafs = timestamps.iter()
            .flat_map(|(timestamp, snapshots)| zip(
                repeat(*timestamp),
                snapshots.iter().filter(|tsnap| tsnap.is_leaf)
            ))
            .into_grouping_map_by(|&(_, tsnap)| tsnap.snapshot.handle)
            .fold_with(
                |_, (timestamp, tsnap)| SnapshotLeaf {
                    snap: &tsnap.snapshot,
                    min_delta_t: tsnap.timestamp_delta_t,
                    timestamps: [*timestamp].into(),
                },
                |mut acc, _, (timestamp, tsnap)| {
                    acc.min_delta_t = min(acc.min_delta_t, tsnap.timestamp_delta_t);
                    acc.timestamps.insert(timestamp);
                    acc
                }
            ).into_values().collect_vec()
        ;

        #[derive(Debug, Clone)]
        struct SnapshotGroup<'a> {
            s1: &'a sg::Snapshot,
            s1_delta_t: Positive,
            s2: &'a sg::Snapshot,
            s2_delta_t: Positive,
            shared_timestamps: HashSet<tsg::Timestamp>,
        }

        #[derive(Debug, Clone)]
        struct RegulatedSnapshotGroup<'a> {
            s1: &'a sg::Snapshot,
            s2: sg::Snapshot,
            min_t: Positive,
        }

        impl<'a> SnapshotGroup<'a> {
            fn regulate(&self) -> RegulatedSnapshotGroup<'a> {
                let diff = self.s2_delta_t - self.s1_delta_t;
                let (late, early, delta) =
                    if let Ok(delta) = Positive::new(diff) /* self.s1_delta_t <= self.s2_delta_t */ {
                        (self.s2, self.s1, delta)
                    }
                    else if let Ok(delta) = Positive::new(-diff) /* self.s1_delta_t >= self.s2_delta_t */ {
                        (self.s1, self.s2, delta)
                    }
                    else {
                        unreachable!()
                    }
                ;

                RegulatedSnapshotGroup {
                    s1: late,
                    s2: early.extrapolate_by(delta),
                    min_t: Positive::new(
                        max(self.s1_delta_t, self.s2_delta_t).get() - diff.abs()
                    ).expect("positive"),
                }
            }
        }

        let groups = timestamps.iter()
            .flat_map(|(timestamp, snapshots)| {
                // partition but without efficiency
                let (leafs, not_leafs) = (
                    snapshots.iter().filter(|&tsnap| tsnap.is_leaf),
                    snapshots.iter().filter(|&tsnap| !tsnap.is_leaf),
                );

                chain(leafs.clone().cartesian_product(not_leafs), leafs.tuple_combinations())
                    .map(|(a, b)| (*timestamp, a, b))
            })
            .map(|(timestamp, ts1, ts2)| {
                let (h1, h2) = (ts1.snapshot.handle, ts2.snapshot.handle);
                if h1 < h2 { (timestamp, ts1, ts2) } else { (timestamp, ts2, ts1) }
            })
            .into_grouping_map_by(|&(_, ts1, ts2)| (ts1.snapshot.handle, ts2.snapshot.handle))
            .fold_with(
                |_, (timestamp, ts1, ts2)| SnapshotGroup {
                    s1: &ts1.snapshot,
                    s1_delta_t: ts1.timestamp_delta_t,
                    s2: &ts2.snapshot,
                    s2_delta_t: ts2.timestamp_delta_t,
                    shared_timestamps: [*timestamp].into(),
                },
                |mut acc, _, (timestamp, ts1, ts2)| {
                    acc.s1_delta_t = min(acc.s1_delta_t, ts1.timestamp_delta_t);
                    acc.s2_delta_t = min(acc.s2_delta_t, ts2.timestamp_delta_t);
                    acc.shared_timestamps.insert(timestamp);
                    acc
                }
            ).into_values().collect_vec()
        ;

        let collision = empty()
            // ball-portal events
            .chain(
                snap_leafs.iter()
                .cartesian_product(0..self.half_portals.len())
                .filter_map(|(SnapshotLeaf { snap, .. }, half_portal_idx)| {
                    let end = f32::min(*self.max_time - snap.time, MAX_DT);
                    debug_assert!(end >= 0.);
                    self.cast_portal_traversal(snap, half_portal_idx, 0. .. end)
                    .map(|data| PortalTraversalSimulationEvent { data, snap })
                    .map(GenericSimulationEventInfo::PortalTraversal)
                })
            )
            // ball-wall collisions
            .chain(
                snap_leafs.iter()
                .filter_map(|SnapshotLeaf { snap, .. }| {
                    let end = f32::min(*self.max_time - snap.time, MAX_DT);
                    debug_assert!(end >= 0.);
                    self.cast_ball_wall_collision(snap, 0. .. end)
                    .map(|col| CollisionSimulationEvent { col, snaps: [snap] })
                    .map(GenericSimulationEventInfo::OneCollision)
                })
            )
            // ball-ball collisions
            .chain(
                groups.iter()
                .filter_map(|group| {
                    debug_assert!((s1.time - s2.time).abs() <= DEFAULT_EPSILON, "{} != {}", s1.time, s2.time);
                    let end = f32::min(*self.max_time - s1.time, MAX_DT);
                    debug_assert!(end >= 0.);
                    self.cast_ball_ball_collision(s1, s2, 0. .. end)
                    .map(|col| CollisionSimulationEvent { col, snaps: [s1, s2] })
                    .map(GenericSimulationEventInfo::TwoCollision)
                })
            )
            // ghostifications
            .chain(
                timestamps.iter()
                .filter_map(|snap| {
                    let end = f32::min(*self.max_time - snap.time, MAX_DT);
                    self.cast_ball_ghostification(snap, 0. .. end)
                    .map(|data| GhostificationSimulationEvent { data, snap })
                    .map(GenericSimulationEventInfo::Ghostification)
                })
            )
            // existing nodes
            .chain(
                // TODO
                empty()
            )
            .inspect(|col| debug_assert!(col.impact_delta_time() <= MAX_DT))
            // .filter(|col| col.impact_delta_time() < *self.max_time)
            .min_by_key(|col| (col.parent_timeline(&self.multiverse()), OF(col.impact_time())))
        ;

        // let Some(collision) = collision
        // else {
        //     println!("NO COLLISION FOUND");
        //     for tid in timestamps.iter().map(|snap| snap.timeline_id).unique() {
        //         let tss = self.timeline_timestamps.get_mut(&tid).expect("exists");
        //         tss.push(TimestampDelta { delta_t: Positive::new(MAX_DT).expect("positive") });
        //     }
        //     return ControlFlow::Continue(())
        // };
        // println!("Selected collision: {}", collision.simple_debug());

        // assert_eq!(collision.parent_timeline(&self.multiverse()), collision.child_timeline(&self.multiverse()), "Unsupported yet");
        // // assert!(self.timeline_presents[&tid] <= collision.impact_time(), "assert({} <= {}) ({collision:#?})", self.timeline_presents[&tid], collision.impact_time());

        // self.apply_simulation_event(collision);
        
        // ControlFlow::Continue(())
        
        todo!()
    }

    // pub fn step(&mut self) -> ControlFlow<(), ()> {
    //     const MAX_DT: f32 = 1.;

    //     // make a non-mutable reference, so a `Copy` reference, for move closures
    //     let this = &*self;

    //     println!();

    //     if self.timeline_timestamps.iter()
    //         .flat_map(|(_, list)| list.iter().map(|(_, data)| data.running_sum.get()))
    //         .reduce(f32::min)
    //         .unwrap_or(*self.max_time) >= *self.max_time {
    //         println!("FINISHED max time is {}", *self.max_time);
    //         return ControlFlow::Break(());
    //     }

    //     let leafs = self.snapshots.leafs().iter().copied()
    //         .map(|handle| this.integrate_snapshot(handle))
    //         .flat_map(|snaps| {
    //             (0..snaps.len()).into_iter()
    //             .filter_map(move |i| {
    //                 let tss = &this.timeline_timestamps[&snaps[i].timeline_id];
    //                 let last_ts = tss.last_timestamp();
    //                 let last_t = tss[last_ts].running_sum.get();
    //                 if last_t >= *this.max_time {
    //                     return None;
    //                 }
    //                 snaps[i].extrapolate_to_timestamp(last_t, last_ts)
    //             })
    //         })
    //         .collect_vec()
    //     ;

    //     if leafs.is_empty() {
    //         println!("FINISHED: NO LEAFS");
    //         return ControlFlow::Break(());
    //     }

    //     println!("leafs: {}", leafs.iter().join(", "));

    //     let worlds: HashMap<TimelineId, Vec<sg::Snapshot>> = leafs.iter()
    //         .map(|snap| snap.timeline_id)
    //         .unique()
    //         .map(|tid| {
    //             let tss = &self.timeline_timestamps[&tid];
    //             let last_ts = tss.last_timestamp();

    //             let world = this.timeline_query(tid)
    //                 .flat_map(move |handle_| 
    //                     this.timestamp_filtered_integrate(handle_, tid, last_ts)
    //                     .filter_map(move |snap| snap.extrapolate_to_timestamp(tss[last_ts].running_sum.get(), last_ts))
    //                 )
    //                 .collect::<Vec<_>>();
    //             (tid, world)
    //         })
    //         .collect();

    //     println!("{}", worlds.iter().map(|(k, v)| (k, v.iter().join(", "))).map(|(k, v)| format!("tid {k} ({}s) -> {v}", self.timeline_timestamps[k].last().running_sum)).join("\n"));

    //     let groups = leafs.iter()
    //         .flat_map(|snap| {
    //             let world = &worlds[&snap.timeline_id];
    //             zip(
    //                 repeat(snap),
    //                 world.iter().filter(|snap_| snap.id() != snap_.id())
    //             )
    //         })
    //         .collect::<Vec<_>>()
    //     ;

    //     println!("ball-ball group count: {}", groups.len());
    //     println!("ball-ball groups: {}", groups.iter().map(|(s1, s2)| format!("{s1} and {s2}")).join(", "));

    //     let collision = empty()
    //         // ball-portal events
    //         .chain(
    //             leafs.iter()
    //             .cartesian_product(0..self.half_portals.len())
    //             .filter_map(|(snap, half_portal_idx)| {
    //                 let end = f32::min(*self.max_time - snap.time, MAX_DT);
    //                 debug_assert!(end >= 0.);
    //                 self.cast_portal_traversal(snap, half_portal_idx, 0. .. end)
    //                 .map(|data| PortalTraversalSimulationEvent { data, snap })
    //                 .map(GenericSimulationEventInfo::PortalTraversal)
    //             })
    //         )
    //         // ball-wall collisions
    //         .chain(
    //             leafs.iter()
    //             .filter_map(|snap| {
    //                 let end = f32::min(*self.max_time - snap.time, MAX_DT);
    //                 debug_assert!(end >= 0.);
    //                 self.cast_ball_wall_collision(snap, 0. .. end)
    //                 .map(|col| CollisionSimulationEvent { col, snaps: [snap] })
    //                 .map(GenericSimulationEventInfo::OneCollision)
    //             })
    //         )
    //         // ball-ball collisions
    //         .chain(
    //             groups.iter().copied()
    //             .filter_map(|(s1, s2)| {
    //                 debug_assert!((s1.time - s2.time).abs() <= DEFAULT_EPSILON, "{} != {}", s1.time, s2.time);
    //                 let end = f32::min(*self.max_time - s1.time, MAX_DT);
    //                 debug_assert!(end >= 0.);
    //                 self.cast_ball_ball_collision(s1, s2, 0. .. end)
    //                 .map(|col| CollisionSimulationEvent { col, snaps: [s1, s2] })
    //                 .map(GenericSimulationEventInfo::TwoCollision)
    //             })
    //         )
    //         .chain(
    //             leafs.iter()
    //             .filter_map(|snap| {
    //                 let end = f32::min(*self.max_time - snap.time, MAX_DT);
    //                 self.cast_ball_ghostification(snap, 0. .. end)
    //                 .map(|data| GhostificationSimulationEvent { data, snap })
    //                 .map(GenericSimulationEventInfo::Ghostification)
    //             })
    //         )
    //         .inspect(|col| debug_assert!(col.impact_delta_time() <= MAX_DT))
    //         // .filter(|col| col.impact_delta_time() < *self.max_time)
    //         .min_by_key(|col| (col.parent_timeline(&self.multiverse()), OF(col.impact_time())))
    //     ;

    //     let Some(collision) = collision
    //     else {
    //         println!("NO COLLISION FOUND");
    //         for tid in leafs.iter().map(|snap| snap.timeline_id).unique() {
    //             let tss = self.timeline_timestamps.get_mut(&tid).expect("exists");
    //             tss.push(TimestampDelta { delta_t: Positive::new(MAX_DT).expect("positive") });
    //         }
    //         return ControlFlow::Continue(())
    //     };
    //     println!("Selected collision: {}", collision.simple_debug());

    //     assert_eq!(collision.parent_timeline(&self.multiverse()), collision.child_timeline(&self.multiverse()), "Unsupported yet");
    //     // assert!(self.timeline_presents[&tid] <= collision.impact_time(), "assert({} <= {}) ({collision:#?})", self.timeline_presents[&tid], collision.impact_time());

    //     self.apply_simulation_event(collision);
        
    //     ControlFlow::Continue(())
    // }

    pub fn run(&mut self) {
        while self.step().is_continue() { }
    }
}
