use std::{ collections::HashMap, iter::{empty, once}, ops::{ ControlFlow, Range, RangeFrom } };

use glam::{ Affine2, Vec2 };
use itertools::Itertools;
use ordered_float::OrderedFloat as OF;
use std::cmp::max;
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
    debug_reason: &'static str,
    impact_time: f32,
    states: [SimSphereNewState; N],
}

#[derive(Debug, Clone, Copy)]
struct SimLinkedSnapshot<'a> {
    link: SimSnapshotLink,
    snap: &'a SimSnapshot,
}

#[derive(Debug, Clone, Copy)]
struct SimCollisionInfo<'a, const N: usize> {
    col: SimSphereCollision<N>,
    snaps: [SimLinkedSnapshot<'a>; N],
}

#[derive(Debug, Clone, Copy)]
enum SimGenericCollisionInfo<'a> {
    One(SimCollisionInfo<'a, 1>),
    Two(SimCollisionInfo<'a, 2>),
}

impl<'a> SimGenericCollisionInfo<'a> {
    fn impact_time(&self) -> f32 {
        match self {
            Self::One(i) => i.col.impact_time,
            Self::Two(i) => i.col.impact_time,
        }
    }
}

impl<'a> From<SimCollisionInfo<'a, 1>> for SimGenericCollisionInfo<'a> {
    fn from(value: SimCollisionInfo<'a, 1>) -> Self {
        Self::One(value)
    }
}

impl<'a> From<SimCollisionInfo<'a, 2>> for SimGenericCollisionInfo<'a> {
    fn from(value: SimCollisionInfo<'a, 2>) -> Self {
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

    pub timelines_present: HashMap<TimelineId, f32>,
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

        let timelines_present: HashMap<TimelineId, f32> = starts.iter()
            .map(|&link| {
                let snap = &snapshots[link];
                (snap.timeline_id, snap.time)
            })
            .fold(HashMap::<TimelineId, f32>::new(), |mut map, (tid, time)| {
                map.entry(tid)
                    .and_modify(|e| { *e = time.min(*e); })
                    .or_insert(time);
                map
            });

        Self {
            world_state,
            multiverse,
            snapshots,

            starts: starts.clone().into_boxed_slice().into(),
            portals,

            max_time,

            timelines_present,
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

    pub fn minmax_time(&self) -> (f32, f32) {
        let min = self.snapshots.nodes().map(|(_, node)| node.snapshot.time)
            .min_by_key(|&t| OF(t))
            .unwrap_or(0.);

        // TODO: Which one
        // let max = self.snapshots.nodes().map(|(_, node)| node)
        //     .filter(|node| node.age_children.is_empty())
        //     .map(|node| (node.snapshot.timeline_id, node.snapshot.time))
        //     .fold(HashMap::<TimelineId, f32>::new(), |mut map, (timeline_id, time)| {
        //         map.entry(timeline_id)
        //             .and_modify(|t| *t = f32::min(*t, time))
        //             .or_insert(time);
        //         map
        //     })
        //     .into_values()
        //     .max_by_key(|&t| OF(t));
        let max = self.timelines_present.values().copied()
            .reduce(f32::min)
            .unwrap_or(self.max_time);

        (min, max)
    }

    fn node_has_children(&self, link: SimSnapshotLink, timeline_id: TimelineId) -> bool {
        let Some(node) = self.snapshots.get_node(link)
        else { return false; };

        node.age_children.iter()
            .any(|&child_link| self.multiverse.is_related(self.snapshots[child_link].timeline_id, timeline_id))
    }

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
        let range_ = SimTimeRange::new(snap.time, end_time);
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

    /// Giving None as the time returns all nodes with no children on the given timeline
    /// Giving Some returns all nodes that covers the given time
    pub fn timeline_query(&self, time: Option<f32>, timeline_id: TimelineId) -> TimelineQueryResult {
        debug_assert!(self.starts.iter().copied().all_unique());

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

        let mut previous_snapshots = Vec::new();
        let mut next_snapshot = None::<SimSnapshotLink>;

        while let Some(StackEntry { link, is_in_timeline }) = stack.pop() {
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

                // This checks that if the snapshot represent the correct time
                // interval for this timeline we add it to the results
                if let Some(time) = time {
                    // TODO: Handle ghosts ranges too
                    let (range, _) = self.get_node_time_ranges(link, Some(timeline_id));
                    if range.start() > time && next_snapshot.is_none_or(|next| self.snapshots[next].time < time) {
                        next_snapshot = Some(link);
                    }
                    if range.contains(time) {
                        previous_snapshots.push(link);
                    }
                }
                else {
                    if node.age_children.is_empty() { // FIXME: Is this sufficient? This may miss some nodes
                        previous_snapshots.push(link);
                    }
                }

                children.iter()
                    .map(|&(link, distance)| StackEntry {
                        link,
                        is_in_timeline: distance == min_distance.expect("Non empty")
                    })
                    .collect_into(&mut stack);
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

            debug_assert!(stack.iter().copied()
                .map(|entry| entry.link)
                .all_unique());
        }

        TimelineQueryResult {
            previous_snapshots,
            next_snapshot,
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
            states: [SimSphereNewState { vel: new_vel, pos: new_pos, ..default() }]
        })
    }

    fn cast_sphere_sphere_collision(&self, s1: &SimSnapshot, s2: &SimSnapshot) -> Option<SimSphereCollision<2>> {
        debug_assert_eq!(s1.time, s2.time);
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
            states: [
                SimSphereNewState { vel: vel1, pos: pos1, ..default() },
                SimSphereNewState { vel: vel2, pos: pos2, ..default() },
            ],
        })
    }

    fn cast_sphere_portal_traversal_start(&self, snap: &SimSnapshot, portal_idx: usize) -> Option<SimSphereCollision<1>> {
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

        let result = parry2d::query::cast_shapes(
            &plane_iso, &default(), &portal_shape,

            &na::Isometry2::translation(snap.pos.x, snap.pos.y), &na::vector![snap.vel.x, snap.vel.y], &sphere_shape,

            parry2d::query::ShapeCastOptions {
                stop_at_penetration: true,
                ..default()
            }
        ).expect("Ball on ball should be supported")?;

        let impact_time = result.time_of_impact + snap.time;
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
            states: [SimSphereNewState {
                vel, pos,
                portal_traversals: tinyvec::array_vec!(_ => traversal),
                time_offset: 0.,
            }],
        })
    }

    fn cast_sphere_portal_traversal_end(&self, snap: &SimSnapshot, portal_idx: usize) -> Option<SimSphereCollision<1>> {
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
            states: [SimSphereNewState {
                vel: traveled_vel, pos: traveled_pos,
                portal_traversals: tinyvec::array_vec!(_ => traversal.swap()),
                time_offset: traversal.time_offset(),
            }],
        })
    }

    pub fn finished(&self) -> bool {
        self.timelines_present.is_empty()
    }

    pub fn step(&mut self) -> ControlFlow<SimStepBreakReason, ()> {
        let Some((&timeline_id, &time)) = self.timelines_present.iter()
            .min_by_key(|&(_, &t)| OF(t))
        else { return ControlFlow::Break(SimStepBreakReason::Finished) };

        let time = time + DEFAULT_EPSILON;

        if time >= self.max_time {
            return ControlFlow::Break(SimStepBreakReason::Finished);
        }

        let query = self.timeline_query(Some(time), timeline_id);
        let next_snapshot_time = query.next_snapshot
            .map(|link| self.snapshots[link].time);

        let snapshots = query.previous_snapshots.into_iter()
            .map(|link| (link, self.snapshots[link]))
            .filter(|(_, snap)| snap.time <= time)
            .map(|(link, snap)| (link, snap.extrapolate(time)))
            .collect_vec()
        ;

        let new_snapshots = snapshots.iter().copied()
            .filter(|&(link, _)| !self.node_has_children(link, timeline_id))
            .collect_vec()
        ;

        // No snapshots means we are too 'early' in the timeline so we need to find
        // the latest time there is something to do
        if new_snapshots.is_empty() {
            let new_min = self.timeline_query(None, timeline_id).previous_snapshots.into_iter()
                .map(|link| self.snapshots[link].time)
                .reduce(f32::min);
            if let Some(new_min) = new_min {
                self.timelines_present.insert(timeline_id, new_min);
            }
            else {
                self.timelines_present.remove(&timeline_id);
            }
            return ControlFlow::Continue(());
        }

        let collision_infos = empty::<SimGenericCollisionInfo>()
            .chain(
                new_snapshots.iter().map(|(l, s)| (*l, s))
                .filter_map(|(link, snap)| self.cast_sphere_wall_collision(snap)
                    .map(|col| SimCollisionInfo {
                        col,
                        snaps: [SimLinkedSnapshot { link, snap }]
                    })
                    .map(SimGenericCollisionInfo::from)
                )
            )
            .chain(
                snapshots.iter().map(|(l, s)| (*l, s))
                .tuple_combinations::<(_, _)>()
                .filter(|&((l1, _), (l2, _))| !self.node_has_children(l1, timeline_id) || !self.node_has_children(l2, timeline_id))
                .filter_map(|((l1, s1), (l2, s2))| self.cast_sphere_sphere_collision(s1, s2)
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
                new_snapshots.iter().map(|(l, s)| (*l, s))
                .cartesian_product(0..self.portals.len())
                .filter_map(|((l1, s1), portal_idx)| self.cast_sphere_portal_traversal_start(s1, portal_idx)
                    .map(|col| SimCollisionInfo {
                        col,
                        snaps: [SimLinkedSnapshot { link: l1, snap: s1 }]
                    })
                    .map(SimGenericCollisionInfo::from)
                )
            )
            .chain(
                new_snapshots.iter().map(|(l, s)| (*l, s))
                .cartesian_product(0..self.portals.len())
                .filter_map(|((l1, s1), portal_idx)| self.cast_sphere_portal_traversal_end(s1, portal_idx)
                    .map(|col| SimCollisionInfo {
                        col,
                        snaps: [SimLinkedSnapshot { link: l1, snap: s1 }]
                    })
                    .map(SimGenericCollisionInfo::from)
                )
            )
            .filter(|result|
                result.impact_time() <= self.max_time &&
                next_snapshot_time.is_none_or(|next_time| result.impact_time() <= next_time)
            )
            .min_set_by_key(|result| OF(result.impact_time()))
        ;

        if let Some(next_time) = next_snapshot_time {
            self.timelines_present.insert(timeline_id, next_time);
        }
        else {
            self.timelines_present.remove(&timeline_id);
        }

        let (ones_collisions, twos_collisions): (Vec::<_>, Vec::<_>) = collision_infos.into_iter()
            .partition_map(|h| match h {
                SimGenericCollisionInfo::One(i1) => itertools::Either::Left(i1),
                SimGenericCollisionInfo::Two(i2) => itertools::Either::Right(i2),
            });

        for info in ones_collisions {
            let new_snapshot = info.snaps[0].snap.advanced(info.col.impact_time, info.col.states[0].pos, info.col.states[0].vel)
                .with_portal_traversals(info.col.states[0].portal_traversals)
                .offset_time(info.col.states[0].time_offset);
            self.snapshots.insert(&self.multiverse, new_snapshot, Some(info.snaps[0].link));

            let new_time = [info.col.impact_time, info.col.impact_time + info.col.states[0].time_offset]
                .into_iter().reduce(f32::min).expect("not empty");

            self.timelines_present.entry(new_snapshot.timeline_id)
                .and_modify(|t| *t = f32::min(*t, new_time))
                .or_insert(new_time);
        }

        for info in twos_collisions {
            let link0 = info.snaps[0].link;
            let link1 = info.snaps[1].link;
            let timeline_id_0 = info.snaps[0].snap.timeline_id;
            let timeline_id_1 = info.snaps[1].snap.timeline_id;

            debug_assert!(self.multiverse.is_related(timeline_id_0, timeline_id_1));
            let child_timeline = max(timeline_id_0, timeline_id_1);

            let new_timeline = if
                self.node_has_children(link0, child_timeline) ||
                self.node_has_children(link1, child_timeline)
            {
                self.multiverse.create_children(child_timeline)
            }
            else {
                child_timeline
            };

            let new_snapshot0 = info.snaps[0].snap.advanced(info.col.impact_time, info.col.states[0].pos, info.col.states[0].vel)
                .with_portal_traversals(info.col.states[0].portal_traversals)
                .offset_time(info.col.states[0].time_offset)
                .replace_timeline(new_timeline);
            let new_snapshot1 = info.snaps[1].snap.advanced(info.col.impact_time, info.col.states[1].pos, info.col.states[1].vel)
                .with_portal_traversals(info.col.states[1].portal_traversals)
                .offset_time(info.col.states[1].time_offset)
                .replace_timeline(new_timeline);

            self.snapshots.insert(&self.multiverse, new_snapshot0, Some(link0));
            self.snapshots.insert(&self.multiverse, new_snapshot1, Some(link1));

            let new_time = [
                info.col.impact_time,
                info.col.impact_time + info.col.states[0].time_offset,
                info.col.impact_time + info.col.states[1].time_offset,
            ]
                .into_iter().reduce(f32::min).expect("not empty");

            self.timelines_present.entry(new_timeline)
                .and_modify(|t| *t = f32::min(*t, new_time))
                .or_insert(new_time);
        }

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
