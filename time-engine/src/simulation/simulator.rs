use std::{collections::{self, HashMap}, ops::ControlFlow};

use glam::{ Affine2, Vec2 };
use i_overlay::i_shape::base::data::Shapes;
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
#[derive(Debug, Clone)]
struct SimPortal {
    pub in_transform: Affine2,
    pub out_transform: Affine2,
    pub time_offset: f32,

    /// Maps input timelines to output timelines
    pub timelines_in_out: HashMap<TimelineId, TimelineId>,
}

#[derive(Debug, Clone, Copy)]
struct SimSphereNewState {
    vel: Vec2,
    pos: Vec2,
}

#[derive(Debug, Clone, Copy)]
struct SimSphereCollision<const N: usize> {
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

    fn states(&self) -> impl Iterator<Item = (SimLinkedSnapshot<'a>, SimSphereNewState)> {
        match self {
            Self::One(i) => i.snaps.iter().copied().zip(i.col.states.iter().copied()),
            Self::Two(i) => i.snaps.iter().copied().zip(i.col.states.iter().copied()),
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

pub struct Simulator<'a> {
    world_state: &'a WorldState,
    multiverse: TimelineMultiverse,
    snapshots: SimSnapshotContainer,

    /// The first snapshot of all spheres
    starts: Ro<Box<[SimSnapshotLink]>>,
    age_roots: Vec<SimSnapshotLink>,
    portals: Vec<SimPortal>,

    max_time: f32,

    timelines_present: HashMap<TimelineId, f32>,
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
                timeline: multiverse.root(),
                age: 0.,
                pos: sphere.initial_pos,
                vel: sphere.initial_velocity,
            })
            .map(|snapshot| snapshots.insert(&multiverse, snapshot, None))
            .collect_vec();

        let portals = world_state.portals.iter()
                .map(|portal| SimPortal {
                    in_transform: portal.in_transform,
                    out_transform: portal.out_transform,
                    time_offset: portal.time_offset,

                    timelines_in_out: default(),
                })
                .collect_vec();

        let timelines_present: HashMap<TimelineId, f32> = starts.iter()
            .map(|&link| {
                let snap = &snapshots[link];
                (snap.timeline, snap.time)
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
            age_roots: starts.clone(),
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
        self.snapshots.nodes()
            .map(|(_, node)| node.snapshot.time)
            .minmax_by_key(|&t| OF(t))
            .into_option()
            .unwrap_or((0., 0.))
    }

    fn interpolate_snapshot_to(&self, time: f32, snap: &SimSnapshot) -> SimSnapshot {
        assert!(time >= snap.time);
        let dt = time - snap.time;
        SimSnapshot {
            original_idx: snap.original_idx,
            radius: snap.radius,
            time,
            timeline: snap.timeline,
            age: snap.age + dt,
            pos: snap.pos + snap.vel * dt,
            vel: snap.vel,
        }
    }

    fn node_has_children(&self, link: SimSnapshotLink, timeline_id: TimelineId) -> bool {
        let Some(node) = self.snapshots.get_node(link)
        else { return false; };

        node.age_children.iter()
            .any(|&child_link| self.multiverse.is_parent(self.snapshots[child_link].timeline, timeline_id))
    }

    pub fn time_query(&self, time: Option<f32>) -> Vec<SimSnapshotLink> {
        debug_assert!(self.starts.iter().copied().all_unique());
        let mut result = Vec::new();

        for (link, node) in self.snapshots.nodes() {
            let should_take = time.is_none_or(|time| {
                if node.snapshot.time > time {
                    return false;
                }
                let min_time = node.age_children.iter()
                    .map(|&link| self.snapshots[link].time)
                    .reduce(f32::min)
                    // if the minimum children type 
                    .map(|min_t| min_t.max(node.snapshot.time));
                min_time.is_none_or(|min_time| min_time > time)
            });

            if should_take {
                result.push(link);
            }
        }

        result
    }

    /// Giving None as the time returns all nodes with no children on the given timeline
    /// Giving Some returns all nodes that covers the given time
    pub fn timeline_query(&self, time: Option<f32>, timeline_id: TimelineId) -> Vec<SimSnapshotLink> {
        debug_assert!(self.starts.iter().copied().all_unique());

        #[derive(Debug, Clone, Copy)]
        struct StackEntry {
            link: SimSnapshotLink,
            is_in_timeline: bool,
        }

        let mut stack = self.starts.iter().copied()
            .inspect(|&link| debug_assert_eq!(self.snapshots[link].timeline, self.multiverse.root()))
            .map(|link| StackEntry {
                link,
                // Necessarily true as link is a root
                is_in_timeline: true,
            })
            .collect_vec();

        let mut result = Vec::new();

        while let Some(StackEntry { link, is_in_timeline }) = stack.pop() {
            let node = self.snapshots.get_node(link).expect("Valid");

            if is_in_timeline {
                let children = node.age_children.iter().copied()
                    // take the distance of each child's timeline with the target timeline
                    .filter_map(|child_link| self.multiverse.distance(self.snapshots[child_link].timeline, timeline_id)
                        // checks that it is a parent
                        .and_then(|distance| (distance >= 0).then_some(distance))
                        .map(|distance| (child_link, distance)))
                    .collect_vec()
                ;
                let min_distance = children.iter()
                    .map(|&(_, distance)| distance)
                    .min();
                let min_time = children.iter()
                    .map(|&(link, _)| self.snapshots[link].time)
                    .reduce(f32::min)
                    // if the minimum children type 
                    .map(|min_t| min_t.max(node.snapshot.time));

                // This checks that if the snapshot represent the correct time
                // interval for this timeline we add it to the results
                if time.is_some_and(|time|
                    node.snapshot.time >= time && min_time.is_none_or(|min_time| min_time > time)
                ) || node.age_children.is_empty() /* FIXME: Is this sufficient? */ {
                    result.push(link);
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
                    .map(|child_link| (child_link, self.snapshots[child_link].timeline))
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

        result
    }

    fn cast_sphere_wall_collision(&self, snap: &SimSnapshot) -> Option<SimSphereCollision<1>> {
        let walls_shape = i_shape_to_parry_shape(self.world_state.get_static_body_collision());
        let sphere_shape = parry2d::shape::Ball::new(snap.radius);

        let collision = parry2d::query::cast_shapes(
            &default(), &default(), &walls_shape,
            &na::Isometry2::translation(snap.pos.x, snap.pos.y), &na::vector![snap.vel.x, snap.vel.y], &sphere_shape,
            parry2d::query::ShapeCastOptions {
                max_time_of_impact: f32::MAX,
                target_distance: 0.0,
                stop_at_penetration: false,
                compute_impact_geometry_on_penetration: true,
            }
        ).expect("Supported")?;

        let impact_normal = Vec2::new(collision.normal1.x, collision.normal1.y);
        let impact_dt = collision.time_of_impact;
        let impact_time = snap.time + impact_dt;
        let new_vel = snap.vel - 2.0 * snap.vel.dot(impact_normal) * impact_normal;
        let new_pos = snap.pos + snap.vel * impact_dt;

        Some(SimSphereCollision {
            impact_time,
            states: [SimSphereNewState {
                vel: new_vel,
                pos: new_pos,
            }]
        })
    }

    fn cast_sphere_sphere_collision(&self, s1: &SimSnapshot, s2: &SimSnapshot) -> Option<SimSphereCollision<2>> {
        debug_assert_eq!(s1.time, s2.time);
        debug_assert!(self.multiverse.is_related(s1.timeline, s2.timeline));

        // NOTE: Generated by claude-4-sonnet
        
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
            impact_time: s1.time + impact_dt,
            states: [
                SimSphereNewState { vel: vel1, pos: pos1 },
                SimSphereNewState { vel: vel2, pos: pos2 },
            ],
        })
    }

    pub fn finished(&self) -> bool {
        self.timelines_present.values().copied()
            .all(|t| t >= self.max_time)
    }

    pub fn step(&mut self) -> ControlFlow<SimStepBreakReason, ()> {
        let Some((&timeline_id, &time)) = self.timelines_present.iter()
            .min_by_key(|&(_, &t)| OF(t))
        else { return ControlFlow::Break(SimStepBreakReason::Finished) };

        if time >= self.max_time {
            return ControlFlow::Break(SimStepBreakReason::Finished);
        }

        let snapshots = self.timeline_query(Some(time), timeline_id).into_iter()
            .map(|link| (link, self.snapshots[link]))
            .map(|(link, snap)| (link, self.interpolate_snapshot_to(time, &snap)))
            .collect_vec()
        ;

        // No snapshots means we are too 'early' in the timeline so we need to find
        // the latest time there is something to do
        if snapshots.is_empty() {
            let new_min = self.timeline_query(None, timeline_id).into_iter()
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

        let collision_infos = std::iter::empty::<SimGenericCollisionInfo>()
            .chain(
                snapshots.iter().map(|(l, s)| (*l, s))
                .filter_map(|(link, snap)| self.cast_sphere_wall_collision(snap)
                    .map(|col| SimCollisionInfo::<1> {
                        col,
                        snaps: [SimLinkedSnapshot { link, snap }]
                    })
                    .map(SimGenericCollisionInfo::from)
                )
            )
            .chain(
                snapshots.iter().map(|(l, s)| (*l, s))
                .array_combinations::<2>()
                .filter_map(|[(l1, s1), (l2, s2)]| self.cast_sphere_sphere_collision(s1, s2)
                    .map(|col| SimCollisionInfo::<2> {
                        col,
                        snaps: [
                            SimLinkedSnapshot { link: l1, snap: s1 },
                            SimLinkedSnapshot { link: l2, snap: s2 },
                        ]
                    })
                    .map(SimGenericCollisionInfo::from)
                )
            )
            .min_set_by_key(|result| OF(result.impact_time()))
        ;

        self.timelines_present.remove(&timeline_id);

        let (ones_collisions, twos_collisions): (Vec::<_>, Vec::<_>) = collision_infos.into_iter()
            .partition_map(|h| match h {
                SimGenericCollisionInfo::One(i1) => itertools::Either::Left(i1),
                SimGenericCollisionInfo::Two(i2) => itertools::Either::Right(i2),
            });

        for info in ones_collisions {
            let new_snapshot = info.snaps[0].snap.advanced(info.col.impact_time, info.col.states[0].pos, info.col.states[0].vel);
            self.snapshots.insert(&self.multiverse, new_snapshot, Some(info.snaps[0].link));

            self.timelines_present.entry(new_snapshot.timeline)
                .and_modify(|t| *t = f32::min(*t, info.col.impact_time))
                .or_insert(info.col.impact_time);
        }

        for info in twos_collisions {
            let link0 = info.snaps[0].link;
            let new_snapshot1 = info.snaps[0].snap.advanced(info.col.impact_time, info.col.states[0].pos, info.col.states[0].vel);
            let link1 = info.snaps[1].link;
            let new_snapshot2 = info.snaps[1].snap.advanced(info.col.impact_time, info.col.states[1].pos, info.col.states[1].vel);

            debug_assert!(self.multiverse.is_related(new_snapshot1.timeline, new_snapshot2.timeline));
            let child_timeline = max(new_snapshot1.timeline, new_snapshot2.timeline);

            if self.node_has_children(link0, child_timeline) || self.node_has_children(link1, child_timeline) {
                todo!("Branching not yet implemented");
            }

            self.snapshots.insert(&self.multiverse, new_snapshot1, Some(link0));
            self.snapshots.insert(&self.multiverse, new_snapshot2, Some(link1));

            self.timelines_present.entry(child_timeline)
                .and_modify(|t| *t = f32::min(*t, info.col.impact_time))
                .or_insert(info.col.impact_time);
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
