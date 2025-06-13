use std::{collections::HashMap, ops::ControlFlow};

use glam::{ Affine2, Vec2 };
use itertools::Itertools;
use ordered_float::OrderedFloat as OF;
use std::cmp::max;

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
struct SimSphereCollision {
    impact_time: f32,
    vel1: Vec2,
    pos1: Vec2,
    vel2: Vec2,
    pos2: Vec2,
}

#[derive(Debug, Clone, Copy)]
struct SimCollisionInfo<'a> {
    col: SimSphereCollision,
    l1: SimSnapshotLink,
    s1: &'a SimSnapshot,
    l2: SimSnapshotLink,
    s2: &'a SimSnapshot,
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

    /// Giving None as the time returns all nodes with no children on the given timeline
    /// Giving Some returns all nodes that covers the given time
    fn timeline_query(&self, time: Option<f32>, timeline_id: TimelineId) -> Vec<SimSnapshotLink> {
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

    fn cast_sphere_collision(&self, s1: &SimSnapshot, s2: &SimSnapshot) -> Option<SimSphereCollision> {
        debug_assert_eq!(s1.time, s2.time);
        todo!()
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

        let collision_infos = snapshots.iter().map(|(l, s)| (*l, s))
            .array_combinations::<2>()
            .filter_map(|[(l1, s1), (l2, s2)]| self.cast_sphere_collision(s1, s2)
                .map(|collision| SimCollisionInfo {
                    col: collision,
                    l1, s1,
                    l2, s2,
                })
            )
            .min_set_by_key(|result| OF(result.col.impact_time))
        ;

        println!("{collision_infos:#?}");

        if collision_infos.is_empty() {
            self.timelines_present.remove(&timeline_id);
        }

        for info in collision_infos {
            let new_snapshot1 = info.s1.advanced(info.col.impact_time, info.col.pos1, info.col.vel1);
            let new_snapshot2 = info.s2.advanced(info.col.impact_time, info.col.pos2, info.col.vel2);

            debug_assert!(self.multiverse.is_related(new_snapshot1.timeline, new_snapshot2.timeline));
            let child_timeline = max(new_snapshot1.timeline, new_snapshot2.timeline);

            if self.node_has_children(info.l1, child_timeline) || self.node_has_children(info.l2, child_timeline) {
                todo!("Branching not yet implemented");
            }

            self.snapshots.insert(&self.multiverse, new_snapshot1, Some(info.l1));
            self.snapshots.insert(&self.multiverse, new_snapshot2, Some(info.l2));

            self.timelines_present.entry(child_timeline)
                .and_modify(|t| *t = f32::min(*t, info.col.impact_time))
                .or_insert(info.col.impact_time);
        }

        ControlFlow::Continue(())
    }

    pub fn run(mut self) -> SimulationResult {
        while self.step().is_continue() { }
    }
}
