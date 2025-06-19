use std::{
    cmp::max,
    iter::{ empty, once, repeat },
    ops::{ BitAnd, ControlFlow, Range, RangeFrom }
};

use glam::{ Affine2, Vec2 };
use itertools::{izip, Itertools};
use ordered_float::OrderedFloat as OF;
use nalgebra as na;

use super::*;

/// All portals are eternal (always existed and will always exist)
/// and are present in all timelines, except the output which only
/// exists in output timelines
///
/// All timelines are input timelines (as the portal is created at the start of
/// the simulation and so in all timelines) though not all timelines are output ones
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct SimPortal {
    pub transform: Affine2,
    pub height: f32,
    pub linked_to: usize,
    pub time_offset: f32,
}

#[derive(Debug, Clone, Copy, Default)]
struct SimSphereNewState {
    vel: Vec2,
    pos: Vec2,
    portal_traversals: tinyvec::ArrayVec<[SimPortalTraversal; 4]>,
    time_offset: f32,
}

#[derive(Debug, Clone, Copy)]
struct SimSphereCollision<const N: usize> {
    #[allow(dead_code)]
    debug_reason: &'static str,
    impact_time: f32,
    impact_delta_time: f32,
    states: [SimSphereNewState; N],
}

#[derive(Debug, Clone, Copy)]
struct SimLinkedSnapshot {
    link: SimSnapshotLink,
    snap: SimSnapshot,
}

#[derive(Debug, Clone, Copy)]
struct SimCollisionInfo<const N: usize> {
    col: SimSphereCollision<N>,
    snaps: [SimLinkedSnapshot; N],
}

impl<const N: usize> SimCollisionInfo<N> {
    fn get_snap(&self, i: usize) -> (SimSnapshotLink, SimSnapshot) {
        assert!(i < N);

        let link = self.snaps[i].link;
        let snap = self.snaps[i].snap.advanced(self.col.impact_time, self.col.states[i].pos, self.col.states[i].vel)
            .with_portal_traversals(self.col.states[i].portal_traversals)
            .offset_time(self.col.states[i].time_offset);

        (link, snap)
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

#[derive(Debug, Clone, Copy)]
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

    fn impact_delta_time(&self) -> f32 {
        match self {
            Self::One(i) => i.col.impact_delta_time,
            Self::Two(i) => i.col.impact_delta_time,
        }
    }

    fn simple_debug(&self) -> String {
        match self {
            SimGenericCollisionInfo::One(info) => {
                format!("one({}, {}s, idx {})", info.col.debug_reason, info.col.impact_time, info.snaps[0].snap.original_idx)
            },
            SimGenericCollisionInfo::Two(info) => {
                format!("two({}, {}s, idx {} on idx {})", info.col.debug_reason, info.col.impact_time, info.snaps[0].snap.original_idx, info.snaps[1].snap.original_idx)
            },
        }
    }

    fn links(&self) -> arrayvec::ArrayVec<SimSnapshotLink, 2> {
        match self {
            SimGenericCollisionInfo::One(info) => info.snaps.iter().map(|snap| snap.link).collect(),
            SimGenericCollisionInfo::Two(info) => info.snaps.iter().map(|snap| snap.link).collect(),
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SimStepBreakReason {
    /// No collision will in the given time
    ExceededMaxTime,
    /// There will be no more collisions, ever
    Finished,
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

pub struct TimelineQueryResult {
    pub previous_snapshots: Vec<SimSnapshotLink>,
    pub next_snapshot: Option<SimSnapshotLink>,
}

#[derive(Debug)]
pub struct Simulator<'a> {
    world_state: &'a WorldState,
    multiverse: TimelineMultiverse,
    snapshots: SimSnapshotContainer,

    /// The first snapshot of all spheres
    starts: Ro<Box<[SimSnapshotLink]>>,
    portals: Vec<SimPortal>,

    max_time: f32,
}

impl<'a> Simulator<'a> {
    pub fn new(world_state: &'a WorldState, max_time: f32) -> Self {
        let multiverse = TimelineMultiverse::new();

        let mut snapshots = SimSnapshotContainer::new();
        let starts = world_state.spheres.iter()
            .enumerate()
            .map(|(original_idx, sphere)| SimSnapshot {
                original_idx,
                radius: sphere.radius,
                time: 0.,
                timeline_id: multiverse.root(),
                age: 0.,
                pos: sphere.initial_pos,
                vel: sphere.initial_velocity,
                portal_traversals: default(),
            })
            .map(|snapshot| snapshots.insert(&multiverse, snapshot, None))
            .collect_vec();

        let mut portals = Vec::<SimPortal>::new();
        for &portal in &world_state.portals {
            portals.push(SimPortal {
                transform: portal.in_transform,
                height: portal.height,
                linked_to: portals.len() + 1,
                time_offset: 0.,
            });
            portals.push(SimPortal {
                transform: portal.out_transform,
                height: portal.height,
                linked_to: portals.len() - 1,
                time_offset: portal.time_offset,
            });
        }

        Self {
            world_state,
            multiverse,
            snapshots,

            starts: starts.clone().into_boxed_slice().into(),
            portals,

            max_time,
        }
    }

    pub fn snapshots(&self) -> &SimSnapshotContainer {
        &self.snapshots
    }

    pub fn multiverse(&self) -> &TimelineMultiverse {
        &self.multiverse
    }

    // TODO: Find another name
    pub fn max_time(&self) -> f32 {
        self.max_time
    }

    fn node_has_children(&self, link: SimSnapshotLink, timeline_id: TimelineId) -> bool {
        let Some(node) = self.snapshots.get_node(link)
        else { return false; };

        node.age_children.iter()
            .any(|&child_link| self.multiverse.is_related(self.snapshots[child_link].timeline_id, timeline_id))
    }

    #[inline(always)]
    fn get_node_time_ranges<'b>(&'b self, link: SimSnapshotLink, timeline_id: Option<TimelineId>) -> (SimTimeRange, impl Iterator<Item = SimTimeRange> + 'b) {
        let node = self.snapshots.get_node(link).expect("Valid link");
        let snap = node.snapshot;
        let end_time = node.age_children.iter().copied()
            .filter(|&child_link| timeline_id.is_none_or(|timeline_id| self.multiverse.is_parent(self.snapshots[child_link].timeline_id, timeline_id)))
            .map(|child_link| self.snapshots[child_link].age)
            .reduce(f32::max)
            .map(|next_age| next_age - snap.age)
            .map(|age_diff| snap.time + age_diff)
        ;

        let range = SimTimeRange::new(snap.time, end_time);
        let range_ = range.clone();
        let iterator = snap.portal_traversals.into_iter()
            .filter_map(move |traversal| {
                range_.offset(traversal.time_offset())
                    .up_to(traversal.end_age - snap.age + snap.time + traversal.time_offset())
            })
        ;

        (range, iterator)
    }

    pub fn time_query(&self, time: f32) -> Vec<(SimTimeRange, SimSnapshotLink)> {
        debug_assert!(self.starts.iter().copied().all_unique());
        let mut result = Vec::new();

        for (link, _) in self.snapshots.nodes() {
            let (range, time_ranges) = self.get_node_time_ranges(link, None);
            if once(range.clone()).chain(time_ranges).any(|range| range.contains(time)) {
                result.push((range, link));
            }
        }

        result
    }

    /// Return an iterator of all nodes that are in the given timeline
    pub fn timeline_query(&self, timeline_id: TimelineId) -> impl Iterator<Item = SimSnapshotLink> + Clone + '_ {
        #[derive(Debug, Clone, Copy)]
        struct StackEntry {
            link: SimSnapshotLink,
            is_in_timeline: bool,
        }

        let mut stack = self.starts.iter().copied()
            .inspect(|&link| debug_assert_eq!(self.snapshots[link].timeline_id, self.multiverse.root()))
            .map(|link| StackEntry {
                link,
                // Necessarily true as link is a root
                is_in_timeline: true,
            })
            .collect_vec();

        std::iter::from_fn(move || {
            while let Some(StackEntry { link, is_in_timeline }) = stack.pop() {
                debug_assert!(stack.iter().copied()
                    .map(|entry| entry.link)
                    .all_unique());

                let node = self.snapshots.get_node(link).expect("Valid");

                if is_in_timeline {
                    let children = node.age_children.iter().copied()
                        // take the distance of each child's timeline with the target timeline
                        .filter_map(|child_link| self.multiverse.distance(self.snapshots[child_link].timeline_id, timeline_id)
                            // checks that it is a parent
                            .and_then(|distance| (distance >= 0).then_some(distance))
                            .map(|distance| (child_link, distance)))
                        .collect_vec()
                    ;
                    let min_distance = children.iter()
                        .map(|&(_, distance)| distance)
                        .min();

                    children.iter()
                        .map(|&(link, distance)| StackEntry {
                            link,
                            is_in_timeline: distance == min_distance.expect("Non empty")
                        })
                        .collect_into(&mut stack);

                    return Some(link);
                }
                else {
                    node.age_children.iter().copied()
                        .map(|child_link| (child_link, self.snapshots[child_link].timeline_id))
                        .filter(|&(_, child_timeline_id)| self.multiverse.is_parent(child_timeline_id, timeline_id))
                        .map(|(child_link, child_timeline_id)| StackEntry {
                            link: child_link,
                            is_in_timeline: child_timeline_id == timeline_id
                        })
                        .collect_into(&mut stack);
                }
            }

            None
        })
    }

    fn apply_collision_1(&mut self, info: SimCollisionInfo<1>) {
        let (link, snap) = info.get_snap(0);
        self.snapshots.insert(&self.multiverse, snap.unghostify(), Some(link));
    }

    fn apply_collision_2(&mut self, info: SimCollisionInfo<2>) {
        let (link0, new_snapshot0) = info.get_snap(0);
        let (link1, new_snapshot1) = info.get_snap(1);
        let child_timeline = info.child_timeline(&self.multiverse);

        let new_timeline = if
            self.node_has_children(link0, child_timeline) ||
            self.node_has_children(link1, child_timeline)
        {
            self.multiverse.create_children(child_timeline)
        }
        else {
            child_timeline
        };

        self.snapshots.insert(
            &self.multiverse,
            new_snapshot0.replace_timeline(new_timeline).unghostify(),
            Some(link0)
        );
        self.snapshots.insert(
            &self.multiverse,
            new_snapshot1.replace_timeline(new_timeline).unghostify(),
            Some(link1)
        );
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

    fn cast_sphere_wall_collision(&self, snap: &SimSnapshot) -> Option<SimSphereCollision<1>> {
        let walls_shape = self.world_state.get_static_body_collision();
        let walls_shape = snap.portal_traversals.iter()
            .fold(walls_shape, |walls_shape, traversal|
                clip_shapes_on_portal(walls_shape, traversal.portal_in.transform, traversal.direction)
            );
        let walls_shape = i_shape_to_parry_shape(walls_shape);
        let sphere_shape = parry2d::shape::Ball::new(snap.radius);

        let collision = parry2d::query::cast_shapes(
            &default(), &default(), &walls_shape,
            &na::Isometry2::translation(snap.pos.x, snap.pos.y), &na::vector![snap.vel.x, snap.vel.y], &sphere_shape,
            parry2d::query::ShapeCastOptions {
                stop_at_penetration: false,
                ..default()
            }
        ).expect("Supported")?;

        let impact_normal = Vec2::new(collision.normal1.x, collision.normal1.y);
        let impact_dt = collision.time_of_impact;
        let impact_time = snap.time + impact_dt;
        let new_vel = snap.vel - 2.0 * snap.vel.dot(impact_normal) * impact_normal;
        let new_pos = snap.pos + snap.vel * impact_dt;

        Some(SimSphereCollision {
            debug_reason: "sphere-wall",
            impact_time,
            impact_delta_time: impact_dt,
            states: [SimSphereNewState { vel: new_vel, pos: new_pos, ..default() }]
        })
    }

    fn cast_sphere_sphere_collision(&self, s1: &SimSnapshot, s2: &SimSnapshot) -> Option<SimSphereCollision<2>> {
        debug_assert!((s1.time - s2.time).abs() <= DEFAULT_EPSILON);
        debug_assert!(self.multiverse.is_related(s1.timeline_id, s2.timeline_id));

        // NOTE: Generated by claude-4-sonnet
        // TODO: Try to use parry2d::query::cast_shapes too, maybe the perf is
        //       better anyway
        
        let dp = s2.pos - s1.pos;
        let dv = s2.vel - s1.vel;
        let sum_radii = s1.radius + s2.radius;
        
        // Solve quadratic equation: |p1(t) - p2(t)|² = (r1 + r2)²
        // where p1(t) = s1.pos + s1.vel * t, p2(t) = s2.pos + s2.vel * t
        // This becomes: |dp + dv * t|² = sum_radii²
        // Expanding: (dp + dv * t) · (dp + dv * t) = sum_radii²
        // dp·dp + 2*dp·dv*t + dv·dv*t² = sum_radii²
        // dv·dv*t² + 2*dp·dv*t + (dp·dp - sum_radii²) = 0
        
        let a = dv.dot(dv);
        let b = 2.0 * dp.dot(dv);
        let c = dp.dot(dp) - sum_radii * sum_radii;
        
        // If a is near zero, spheres have same velocity (parallel motion)
        if a.abs() < 1e-6 {
            return None;
        }
        
        let discriminant = b * b - 4.0 * a * c;
        
        // No real solutions means no collision
        if discriminant < 0.0 {
            return None;
        }
        
        let sqrt_discriminant = discriminant.sqrt();
        let t1 = (-b - sqrt_discriminant) / (2.0 * a);
        let t2 = (-b + sqrt_discriminant) / (2.0 * a);

        // We want the earliest positive collision time
        let impact_dt = if t1 > DEFAULT_EPSILON {
            t1
        } else if t2 > DEFAULT_EPSILON {
            t2
        } else { 
            return None 
        };
        
        let pos1 = s1.pos + s1.vel * impact_dt;
        let pos2 = s2.pos + s2.vel * impact_dt;
        
        // Calculate collision response
        let collision_normal = (pos2 - pos1).normalize();
        let relative_velocity = s1.vel - s2.vel;
        let velocity_along_normal = relative_velocity.dot(collision_normal);

        // Don't resolve if velocities are separating
        if velocity_along_normal < DEFAULT_EPSILON {
            return None;
        }
        
        // Assume equal mass elastic collision
        let impulse = velocity_along_normal * collision_normal;
        let vel1 = s1.vel - impulse;
        let vel2 = s2.vel + impulse;
        
        Some(SimSphereCollision {
            debug_reason: "sphere-sphere",
            impact_time: s1.time + impact_dt,
            impact_delta_time: impact_dt,
            states: [
                SimSphereNewState { vel: vel1, pos: pos1, ..default() },
                SimSphereNewState { vel: vel2, pos: pos2, ..default() },
            ],
        })
    }

    fn cast_sphere_portal_traversal_start(&self, snap: &SimSnapshot, portal_idx: usize) -> Option<SimSphereCollision<1>> {
        if snap.is_ghost() {
            return None;
        }

        let sphere_shape = parry2d::shape::Ball::new(snap.radius);
        let portal = self.portals[portal_idx];
        let portal_shape = parry2d::shape::Polyline::new(vec![
            na::point![0., -portal.height / 2.],
            na::point![0., portal.height / 2.],
        ], None);

        if snap.portal_traversals.iter().any(|tr| tr.portal_in_idx == portal_idx) {
            return None;
        }

        // Anoying and imperfect conversion from glam's Isometry to nalgebra's 
        let plane_iso = {
            let (scale, angle, trans) = portal.transform.to_scale_angle_translation();
            assert_eq!(scale, Vec2::splat(1.));
            na::Isometry::from_parts(
                na::Translation2::new(trans.x, trans.y),
                na::UnitComplex::from_angle(angle)
            )
        };

        let contact = parry2d::query::contact(
            &na::Isometry2::translation(snap.pos.x, snap.pos.y),  &sphere_shape,
            &plane_iso, &portal_shape,
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
                parry2d::query::cast_shapes(
                    &plane_iso, &default(), &portal_shape,

                    &na::Isometry2::translation(snap.pos.x, snap.pos.y), &na::vector![snap.vel.x, snap.vel.y], &sphere_shape,

                    parry2d::query::ShapeCastOptions {
                        stop_at_penetration: true,
                        ..default()
                    }
                ).expect("supported")
            })?;

        let impact_dt = result.time_of_impact;
        let impact_time = impact_dt + snap.time;
        let direction = if result.normal1.x < 0. { PortalDirection::Front } else { PortalDirection::Back };

        let vel = snap.vel;
        let pos = snap.pos + vel * result.time_of_impact;
        let traversal = SimPortalTraversal {
            portal_in_idx: portal_idx,
            portal_in: portal,
            portal_out_idx: portal.linked_to,
            portal_out: self.portals[portal.linked_to],
            end_age: snap.age + result.time_of_impact + SimPortalTraversal::compute_end_dt(snap.radius, pos, vel, portal.transform),
            direction,
        };

        Some(SimSphereCollision {
            debug_reason: "sphere-portal-start",
            impact_time,
            impact_delta_time: impact_dt,
            states: [SimSphereNewState {
                vel, pos,
                portal_traversals: tinyvec::array_vec!(_ => traversal),
                time_offset: 0.,
            }],
        })
    }

    fn cast_sphere_portal_traversal_end(&self, snap: &SimSnapshot, portal_idx: usize) -> Option<SimSphereCollision<1>> {
        if snap.is_ghost() {
            return None;
        }

        let traversal = snap.portal_traversals.iter().copied()
            .find(|traversal| traversal.portal_in_idx == portal_idx)?;

        let portal_in = traversal.portal_in;
        let portal_out = traversal.portal_out;

        let inv_portal_trans = portal_in.transform.inverse();
        let rel_vel = inv_portal_trans.transform_vector2(snap.vel);
        let rel_pos = inv_portal_trans.transform_point2(snap.pos);

        // compute when the center of the sphere touches the portal
        // nothing in parry2d can help so we hardcode the solution
        let impact_dt = -rel_pos.x / rel_vel.x;
        // Impact in the past or alread on the impact -> nothing to be done
        if impact_dt <= 0. {
            return None;
        }

        let impact_time = snap.time + impact_dt;

        let traveled_pos = portal_out.transform.transform_point2(rel_pos + rel_vel * impact_dt);
        let traveled_vel = portal_out.transform.transform_vector2(rel_vel);

        Some(SimSphereCollision {
            debug_reason: "sphere-portal-end",
            impact_time,
            impact_delta_time: impact_dt,
            states: [SimSphereNewState {
                vel: traveled_vel, pos: traveled_pos,
                portal_traversals: tinyvec::array_vec!(_ => traversal.swap()),
                time_offset: traversal.time_offset(),
            }],
        })
    }

    fn next_collision_for_snap(
        &self,
        link: SimSnapshotLink,
        snap: SimSnapshot,
    ) -> Option<SimGenericCollisionInfo> {
        let timeline_id = snap.timeline_id;

        let world: AutoSmallVec<(SimSnapshotLink, SimSnapshot)> =
            self.timeline_query(snap.timeline_id)
            .filter(|&link_| link != link_)
            .flat_map(|link_| {
                let snap = self.snapshots[link_];
                let (range, ranges) = self.get_node_time_ranges(link_, Some(timeline_id));
                let mut ranges = ranges.map(Some).collect_smallvec();

                repeat(link_).zip(
                    once((range, snap))
                    .chain(snap.get_ghosts()
                        .map(move |ghost| (ranges[ghost.index].take().expect("Taken once"), ghost.snapshot)))
                ).map(|(a, (b, c))| (a, b, c))
            })
            .filter(|(_, range, _)| {
                range.contains(snap.time)
            })
            .map(|(link, _, snap_)| (link, snap_.extrapolate(snap.time)))
            .collect()
        ;
        // Find a snapshot that already exist and is in the future
        // It means we cannot expect the current snapshot to be valid after this
        // time
        let next_snapshot_time: Option<f32> =
            self.timeline_query(snap.timeline_id)
            .filter(|&link_| link != link_)
            .flat_map(|link_| {
                let (range_, ranges_) = self.get_node_time_ranges(link_, Some(timeline_id));
                once(range_).chain(ranges_)
                    .map(|r| r.start())
            })
            .filter(|&r| r > snap.time)
            .min_by_key(|&r| OF(r))
        ;

        if let Some(next_snapshot_time) = next_snapshot_time {
            dbg!(next_snapshot_time);
        }

        let collision_infos = empty::<SimGenericCollisionInfo>()
            .chain(
                once((link, snap))
                .filter_map(|(link, snap)| self.cast_sphere_wall_collision(&snap)
                    .map(|col| SimCollisionInfo {
                        col,
                        snaps: [SimLinkedSnapshot { link, snap }]
                    })
                    .map(SimGenericCollisionInfo::from)
                )
            )
            .chain(
                once((link, snap))
                .cartesian_product(world.iter().copied())
                .filter_map(|((l1, s1), (l2, s2))| self.cast_sphere_sphere_collision(&s1, &s2)
                    .map(|col| SimCollisionInfo {
                        col,
                        snaps: [
                            SimLinkedSnapshot { link: l1, snap: s1 },
                            SimLinkedSnapshot { link: l2, snap: s2 },
                        ]
                    })
                    .map(SimGenericCollisionInfo::from)
                )
            )
            .chain(
                once((link, snap))
                .cartesian_product(0..self.portals.len())
                .filter_map(|((l1, s1), portal_idx)| self.cast_sphere_portal_traversal_start(&s1, portal_idx)
                    .map(|col| SimCollisionInfo {
                        col,
                        snaps: [SimLinkedSnapshot { link: l1, snap: s1 }]
                    })
                    .map(SimGenericCollisionInfo::from)
                )
            )
            .chain(
                once((link, snap))
                .cartesian_product(0..self.portals.len())
                .filter_map(|((l1, s1), portal_idx)| self.cast_sphere_portal_traversal_end(&s1, portal_idx)
                    .map(|col| SimCollisionInfo {
                        col,
                        snaps: [SimLinkedSnapshot { link: l1, snap: s1 }]
                    })
                    .map(SimGenericCollisionInfo::from)
                )
            )
            .filter(|result|
                result.impact_time() <= self.max_time
                // && next_snapshot_time.is_none_or(|next_time| result.impact_time() <= next_time)
            )
            .min_set_by_key(|result| OF(result.impact_delta_time()))
        ;

        assert!(collision_infos.len() <= 1);
        
        println!(
            "{} collisions for idx {} from {}s at {}s ({}, {})x({}, {})",
            if collision_infos.is_empty() { "No" } else { "One" },
            snap.original_idx, snap.time,
            collision_infos.first().map(|t| t.impact_time()).unwrap_or(f32::INFINITY),
            snap.pos.x, snap.pos.y,
            snap.vel.x, snap.vel.y,
        );
        if collision_infos.is_empty() {
            None
        }
        else {
            Some(collision_infos[0])
        }
    }

    pub fn step(&mut self) -> ControlFlow<SimStepBreakReason, ()> {
        let next_collisions = self.snapshots.nodes()
            .filter(|(_, node)| node.age_children.is_empty())
            .flat_map(|(link, node)| {
                repeat(link).zip(
                    once(node.snapshot)
                    .chain(node.snapshot.get_ghosts()
                        .map(move |ghost| ghost.snapshot))
                )
            })
            .filter_map(|(link, snap)| self.next_collision_for_snap(link, snap))
            // .unique_by(|col| {
            //     let mut links = col.links();
            //     links.sort();
            //     (OF(col.impact_delta_time()), links)
            // })
            .inspect(|h| { println!("option: {}", h.simple_debug()); })
            // Non trivial at all and may not be correct
            // what we actually want is the collisions that hat no collisions 'before' it
            // but with time travel this is non trivial, using the most parent timeline
            // first seems to be the way but who knows
            .min_set_by_key(|col| OF(col.impact_time()))
        ;
        let Some(next_collision) = next_collisions.first().cloned()
        else { return ControlFlow::Break(SimStepBreakReason::Finished) };
        println!("\nchosen: {}\n", next_collision.simple_debug());

        self.apply_collision(next_collision);
        
        ControlFlow::Continue(())
    }

    pub fn extrapolate_to(&mut self, to: f32) {
        // gathering the list of snapshots that need extrapolating
        // have to collect_vec for the borrow checker
        let to_extrapolate = self.snapshots
            .nodes()
            .filter(|(_, node)| node.snapshot.time < to)
            .filter(|(_, node)| node.age_children.is_empty())
            .map(|(link, _)| link)
            .collect_vec();
        for link in to_extrapolate {
            let snap = self.snapshots[link];
            let new_snap = snap.extrapolate(to);
            self.snapshots.insert(&self.multiverse, new_snap, Some(link));
        }
    }

    pub fn run(&mut self) {
        while self.step().is_continue() { }
    }
}
