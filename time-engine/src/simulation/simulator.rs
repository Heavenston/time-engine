use std::collections::HashMap;

use glam::{Affine2, Vec2};
use itertools::Itertools as _;

use super::*;

#[derive(Debug, Clone)]
struct SimulationSphereSnapshot {
    pub t: f32,
    pub pos: Vec2,
    pub vel: Vec2,
}

#[derive(Debug, Clone)]
struct SimulationSphere {
    pub original_idx: usize,
    pub radius: f32,
    pub snapshots: HashMap<TimelineId, Vec<SimulationSphereSnapshot>>,
}

impl SimulationSphere {
}

/// All portals are eternal (always existed and will always exist)
/// and are present in all timelines, except the output which only
/// exists in output timelines
///
/// All timelines are input timelines (as the portal is created at the start of
/// the simulation and so in all timelines) though not all timelines are output ones
#[derive(Debug, Clone)]
struct SimulationPortal {
    pub in_transform: Affine2,
    pub out_transform: Affine2,
    pub time_offset: f32,

    /// Maps input timelines to output timelines
    pub timelines_in_out: HashMap<TimelineId, TimelineId>,
}

impl SimulationPortal {
}

pub struct Simulator<'a> {
    world_state: &'a WorldState,
    multiverse: TimelineMultiverse,
    spheres: Vec<SimulationSphere>,
    portals: Vec<SimulationPortal>,
}

impl<'a> Simulator<'a> {
    pub fn new(world_state: &'a WorldState) -> Self {
        let multiverse = TimelineMultiverse::new();

        let spheres = world_state.spheres.iter()
            .enumerate()
            .map(|(original_idx, sphere)| SimulationSphere {
                original_idx,
                radius: sphere.radius,
                snapshots: [(multiverse.root(), vec![SimulationSphereSnapshot {
                    t: 0.,
                    pos: sphere.initial_pos,
                    vel: sphere.initial_velocity,
                }])].into(),
            })
            .collect_vec();

        let portals = world_state.portals.iter()
                .map(|portal| SimulationPortal {
                    in_transform: portal.in_transform,
                    out_transform: portal.out_transform,
                    time_offset: portal.time_offset,

                    timelines_in_out: default(),
                })
                .collect_vec();

        Self {
            world_state,
            multiverse,
            spheres,
            portals,
        }
    }

    fn sphere_latest_snapshot(&self, idx: usize, mut tid: TimelineId, t: f32) -> Option<&SimulationSphereSnapshot> {
        let sphere = self.spheres.get(idx)?;
        loop {
            let snapshots = sphere.snapshots.get(&tid)?;
            if let Some(snap) = snapshots.iter().rev()
                .find(|snap| snap.t <= t) {
                return Some(snap);
            }

            tid = self.multiverse.try_parent_of(tid)?;
        }
    }

    pub fn run(mut self) -> SimulationResult {
        // For each timelines sets what is considered the 'present', i.e. what
        // is currently being simulated
        let mut timelines_present: HashMap<TimelineId, f32> =
            [(self.multiverse.root(), 0.)].into();

        todo!()
    }
}
