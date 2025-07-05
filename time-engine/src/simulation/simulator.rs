use std::{
    collections::HashMap, iter::{ empty, repeat, zip }, ops::ControlFlow, range::Range, sync::Arc
};

use nalgebra as na;
use parking_lot::RwLock;
use glam::{ Affine2, Vec2 };
use itertools::Itertools;
use parry2d::shape::Shape;
use smallvec::smallvec;
use ordered_float::OrderedFloat as OF;

use crate::sg::PartialPortalTraversal;

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

    timeline_presents: HashMap<TimelineId, f32>,

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

        let timeline_presents = [
            (multiverse.root(), 0.)
        ].into();

        Self {
            world_state,
            multiverse,
            snapshots,
            integration_cache: default(),

            starts: starts.clone().into_boxed_slice().into(),
            half_portals,

            timeline_presents,

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

    pub fn integrate(&self, handle: sg::NodeHandle) -> Arc<[sg::Snapshot]> {
        if let Some(snap) = self.integration_cache.read().get(&handle).map(Arc::clone) {
            return snap;
        }

        let new_snapshots = match &self.snapshots[handle] {
            sg::Node::Root(root_node) => {
                let root_snap = root_node.snapshot;
                vec![sg::Snapshot {
                    object_id: root_snap.object_id,
                    handle,
                    sub_id: 0,
                    extrapolated_by: Positive::new(0.).expect("Positive"),

                    timeline_id: self.multiverse.root(),
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
                }]
            },
            sg::Node::Inner(inner_node) => {
                let snapshots = self.integrate(inner_node.previous);
                let partial = &inner_node.partial;

                // applies the 'partial' to all snapshots
                // TODO?: also remove 'ghosts' (fully behind portals)
                snapshots.iter().filter_map(|snapshot| {
                    debug_assert_eq!(snapshot.handle, inner_node.previous);

                    // TODO: Portal traversals

                    let mut new_snapshot = snapshot.clone();
                    new_snapshot.integrate_by(partial.delta_age);
                    Some(sg::Snapshot {
                        handle,

                        linvel: snapshot.linvel + snapshot.force_transform.transform_vector2(partial.linear_impulse),
                        angvel: snapshot.angvel + partial.angular_impulse,

                        ..new_snapshot
                    })
                }).collect_vec()
            },
        };
        let mut out_snapshots = new_snapshots;

        out_snapshots.iter_mut().enumerate()
            .for_each(|(i, snap)| snap.sub_id = i);

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

        let (t0, t1) = 'compute: {
            let p = rel_center.x;
            let v = rel_vel.x;
            let r = radius;

            // v ~= 0
            if v.abs() <= DEFAULT_EPSILON {
                // Touching and always touching
                if v <= r {
                    break 'compute (f32::NEG_INFINITY, f32::INFINITY);
                }
                else {
                    return None;
                }
            }

            let t0 = (-r - p) / v;
            let t1 = ( r - p) / v;

            if v < 0. {
                (t1, t0)
            }
            else /* v > 0. */ {
                (t0, t1)
            }
        };

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

        let collision = dbg!(parry2d::query::cast_shapes_nonlinear(
            &parry2d::query::NonlinearRigidMotion::identity(), &portal_shape,
            &self.snapshot_motion(snap), &collision_shape,
            range_overlap.start, range_overlap.end,
            true
        )).expect("Compatible")?;

        let direction = if collision.normal1.x < 0. {
            PortalDirection::Front
        } else {
            PortalDirection::Back
        };

        let delta_range = Positive::new(collision.time_of_impact).expect("positive")..Positive::new(t1).expect("positive");
        let range = delta_range.start.get() + snap.time..delta_range.end.get() + snap.time;

        debug_assert!(delta_range.start < delta_range.end);

        Some(NewPortalTraversalData {
            half_portal_idx,
            direction,
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

    fn get_snapshot_portal_traversal_end_dt(&self, snap: &sg::Snapshot, half_portal_idx: usize) -> Option<Positive> {
        let half_portal = self.half_portals[half_portal_idx];
        let radius = self.world_state.balls()[snap.object_id].radius;

        let inv_portal_trans = half_portal.transform.inverse();
        let rel_vel = inv_portal_trans.transform_vector2(snap.linvel);
        let rel_pos = inv_portal_trans.transform_point2(snap.pos);

        if rel_vel.x.abs() < DEFAULT_EPSILON {
            if rel_pos.x.abs() <= radius {
                Some(Positive::new(f32::INFINITY).expect("Positive"))
            }
            else {
                None
            }
        }
        else {
            let t0 = ( radius - rel_pos.x) / rel_vel.x;
            let t1 = (-radius - rel_pos.x) / rel_vel.x;

            let max = f32::max(t0, t1);
            if max < 0. {
                None
            }
            else {
                Some(Positive::new(max).expect("Positive"))
            }
        }
    }

    fn get_snapshot_portal_traversal(&self, snap: &sg::Snapshot, half_portal_idx: usize) -> Option<PortalTraversalCheckResult> {
        let snap_shape = self.snapshot_collision_shape(snap);
        let portal = self.half_portals[half_portal_idx];
        let portal_shape = parry2d::shape::Polyline::new(vec![
            na::point![0., -portal.height / 2.],
            na::point![0., portal.height / 2.],
        ], None);
        let portal_isometry = affine_to_isometry(portal.transform);

        let contact = parry2d::query::contact(
            &affine_to_isometry(self.snapshot_affine(snap)), &snap_shape,
            &portal_isometry, &portal_shape,
            0.
        ).expect("supported");

        let result = contact
            .map(|contact| {
                parry2d::query::ShapeCastHit {
                    time_of_impact: 0.,
                    witness1: contact.point1,
                    witness2: contact.point2,
                    normal1: contact.normal1,
                    normal2: contact.normal2,
                    // unused anyway
                    status: parry2d::query::ShapeCastStatus::Converged,
                }
            })
            .or_else(|| {
                parry2d::query::cast_shapes_nonlinear(
                    &parry2d::query::NonlinearRigidMotion::constant_position(portal_isometry),
                    &portal_shape,
                    &self.snapshot_motion(snap), &snap_shape,

                    // TODO: Use real number \o/
                    0., 15., true
                ).expect("supported")
            })?;

        let impact_dt = dbg!(result.time_of_impact);
        let impact_time = impact_dt + snap.time;
        let end_dt = self.get_snapshot_portal_traversal_end_dt(snap, half_portal_idx)
            .expect("Collision exists");
        let end_t = snap.time + end_dt.get();
        let direction = if result.normal1.x < 0. { PortalDirection::Front } else { PortalDirection::Back };

        Some(PortalTraversalCheckResult {
            direction,
            duration: impact_time..end_t,
        })
    }

    fn apply_collision_1(&mut self, event: CollisionSimulationEvent<1>) {
        debug_assert!(self.snapshots[event.snaps[0].handle].children().is_empty());

        let invforcetrans = event.snaps[0].force_transform.inverse();
        let new_partial = sg::PartialSnapshot {
            delta_age: event.snaps[0].extrapolated_by + event.col.impact_delta_time,
            linear_impulse: invforcetrans.transform_vector2(event.col.states[0].linear_impulse) ,
            angular_impulse: event.col.states[0].angular_impulse,
            portal_traversal: smallvec![],
        };

        println!("[1/1] {new_partial:#?}");
        let n = self.snapshots.insert(new_partial, event.snaps[0].handle);
        println!("Created handle {n}");
    }

    fn apply_collision_2(&mut self, event: CollisionSimulationEvent<2>) {
        for i in 0..2 {
            debug_assert!(self.snapshots[event.snaps[i].handle].children().is_empty());

            let invforcetrans = event.snaps[i].force_transform.inverse();
            let new_partial = sg::PartialSnapshot {
                delta_age: event.snaps[i].extrapolated_by + event.col.impact_delta_time,
                linear_impulse: invforcetrans.transform_vector2(event.col.states[i].linear_impulse) ,
                angular_impulse: event.col.states[i].angular_impulse,
                portal_traversal: smallvec![],
            };

            println!("[{}/2] {new_partial:#?}", i + 1);
            let n = self.snapshots.insert(new_partial, event.snaps[i].handle);
            println!("Created handle {n}");
        }
    }

    fn apply_portal_traversal(&mut self, event: PortalTraversalSimulationEvent) {
        // debug_assert!(!event.snap.portal_traversals.iter()
        //     .any(|traversal| traversal.half_portal_idx == event.data.half_portal_idx));
        debug_assert!(event.snap.portal_traversals.is_empty());
        let new_partial = sg::PartialSnapshot {
            delta_age: event.snap.extrapolated_by + event.data.delta_range.start,
            linear_impulse: Vec2::ZERO,
            angular_impulse: 0.,
            portal_traversal: smallvec![
                PartialPortalTraversal {
                    half_portal_idx: event.data.half_portal_idx,
                    in_direction: event.data.direction,
                    // TODO
                    // sub_id_in: 0,
                    // sub_id_out: 0,
                },
            ],
        };
        println!("[1/1] {new_partial:#?}");
        let n = self.snapshots.insert(new_partial, event.snap.handle);
        println!("Created handle {n}");
    }

    fn apply_simulation_event(&mut self, info: GenericSimulationEventInfo) {
        match info {
            GenericSimulationEventInfo::OneCollision(event) => {
                self.apply_collision_1(event);
            },
            GenericSimulationEventInfo::TwoCollision(event) => {
                self.apply_collision_2(event);
            },
            GenericSimulationEventInfo::PortalTraversal(event) => {
                self.apply_portal_traversal(event);
            },
        }
    }

    pub fn step(&mut self) -> ControlFlow<(), ()> {
        const MAX_DT: f32 = 1.;

        // make a non-mutable reference so a `Copy` reference for move closures
        let this = &*self;

        println!();
        dbg!(&self.timeline_presents);

        if self.timeline_presents.values().copied()
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
                    snaps[i].extrapolate_to(this.timeline_presents[&snaps[i].timeline_id])
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
                let t = this.timeline_presents[&tid];
                let world = this.timeline_query(tid)
                    .flat_map(move |handle_| 
                        this.time_filtered_integrate(handle_, t)
                        .filter_map(move |snap| snap.extrapolate_to(t))
                    )
                    .collect::<Vec<_>>();
                (tid, world)
            })
            .collect();

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
                let present = self.timeline_presents.get_mut(&tid).expect("present");
                *present += MAX_DT;
            }
            return ControlFlow::Continue(())
        };
        let tid = collision.parent_timeline(&self.multiverse);

        println!("Selected collision: {}", collision.simple_debug());

        assert_eq!(collision.parent_timeline(&self.multiverse), collision.child_timeline(&self.multiverse), "Unsupported yet");
        assert!(self.timeline_presents[&tid] <= collision.impact_time(), "assert({} <= {}) ({collision:#?})", self.timeline_presents[&tid], collision.impact_time());

        self.timeline_presents.insert(tid, collision.impact_time());
        self.apply_simulation_event(collision);
        
        ControlFlow::Continue(())
    }

    pub fn run(&mut self) {
        while self.step().is_continue() { }
    }
}
