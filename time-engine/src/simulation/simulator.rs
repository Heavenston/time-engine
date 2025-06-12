use std::{collections::{HashMap, HashSet}, sync::Arc};

use glam::{Affine2, Vec2};
use itertools::Itertools;

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

pub struct Simulator<'a> {
    world_state: &'a WorldState,
    multiverse: TimelineMultiverse,
    snapshots: SimSnapshotContainer,

    /// The first snapshot of all spheres
    starts: Ro<Box<[SimSnapshotLink]>>,
    age_roots: Vec<SimSnapshotLink>,
    portals: Vec<SimPortal>,
}

impl<'a> Simulator<'a> {
    pub fn new(world_state: &'a WorldState) -> Self {
        let multiverse = TimelineMultiverse::new();

        let mut snapshots = SimSnapshotContainer::new();
        let roots = world_state.spheres.iter()
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

        Self {
            world_state,
            multiverse,
            snapshots,

            starts: roots.clone().into_boxed_slice().into(),
            age_roots: roots.clone(),
            portals,
        }
    }

    /// Finds if the given timeline is already started before in the ancestry
    /// of the given snapshot link
    fn node_is_in_timeline(&self, link: SimSnapshotLink, timeline_id: TimelineId) -> bool {
        {
            let Some(node) = self.snapshots.get_node(link)
            else { return false; };

            if node.snapshot.timeline == timeline_id {
                return true;
            }
        
            if !self.multiverse.is_parent(node.snapshot.timeline, timeline_id) {
                return false;
            }
        }

        let mut done = HashSet::<SimSnapshotLink>::new();

        for (ancester_link, ancester_node) in self.snapshots.iter_ancestry(link) {
            done.insert(ancester_link);
            debug_assert!(
                self.multiverse.distance(ancester_node.snapshot.timeline, timeline_id)
                    .is_some_and(|dist| dist > 0)
            );

            // If we are in a parent timeline and the timeline starts later we
            // know it cannot have started in another branch
            if ancester_node.snapshot.time < timeline_id.start() {
                return true;
            }

            // no need to check the rest for the start node
            if ancester_link == link {
                continue;
            }

            // look for any children that 'gets closer' to the given timeline than
            // the current ancestry
            let has_better = ancester_node.age_children.iter().copied()
                .filter(|child_link| !done.contains(child_link))
                .map(|child_link| self.snapshots[child_link])
                .any(|child_snap|
                    // this is equivalent to the child timeline being
                    // 'closer' to timeline_id
                    child_snap.timeline != ancester_node.snapshot.timeline &&
                    self.multiverse.is_parent(child_snap.timeline, timeline_id)
                );

            if has_better {
                return false;
            }
        }
        
        true
    }

    fn timeline_query(&self, time: f32, timeline_id: TimelineId) -> Vec<SimSnapshotLink> {
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
                let new_children = node.age_children.iter().copied()
                    // take the distance of each child's timeline with the target timeline
                    .filter_map(|child_link| self.multiverse.distance(self.snapshots[child_link].timeline, timeline_id)
                        // checks that it is a parent
                        .and_then(|distance| (distance >= 0).then_some(distance))
                        .map(|distance| (child_link, distance)))
                    .collect_vec()
                ;
                let min_distance = new_children.iter()
                    .map(|&(_, distance)| distance)
                    .min();
                let min_time = new_children.iter()
                    .map(|&(link, _)| self.snapshots[link].time)
                    .reduce(f32::min)
                    // if the minimum children type 
                    .map(|min_t| min_t.max(node.snapshot.time));

                // This checks that if the snapshot represent the correct time
                // interval for this timeline we add it to the results
                if node.snapshot.time >= time && min_time.is_none_or(|min_time| min_time > time) {
                    result.push(link);
                }

                new_children.iter()
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

    pub fn run(mut self) -> SimulationResult {
        // For each timelines sets what is considered the 'present', i.e. what
        // is currently being simulated
        let mut timelines_present: HashMap<TimelineId, f32> =
            [(self.multiverse.root(), 0.)].into();

        todo!()
    }
}
