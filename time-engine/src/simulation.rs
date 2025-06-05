use crate::{ default, WorldState };

use glam::f32::Vec2;
use parry2d::{ math as pmath, query, shape::{self, SharedShape} };
use nalgebra as na;
use itertools::Itertools;

#[derive(Debug, Clone, Copy)]
pub struct SpherePositionSnapshot {
    pub t: f32,
    pub pos: Vec2,
}

#[derive(Default, Clone, Debug)]
pub struct SimulationResult {
    pub sphere_positions: Vec<Vec<SpherePositionSnapshot>>,
}

impl SimulationResult {
    pub fn new() -> Self {
        default()
    }

    pub fn sphere_count(&self) -> usize {
        self.sphere_positions.len()
    }

    /// Does any necessary interpolation to get the position of the given
    /// sphere at the given time.
    /// Returns None if the sphere is not spawned at this time
    pub fn get_sphere_pos(&self, idx: usize, t: f32) -> Option<Vec2> {
        let positions = self.sphere_positions.get(idx)?;

        // Find the first snapshot that is after the given time
        let (after_idx, after) = positions.iter().enumerate()
            .find_or_last(|(_, snap)| snap.t >= t)?;

        // The first snapshot is after the given time so the sphere is not
        // spawned yet
        if after_idx == 0 && after.t > t {
            return None;
        }

        // No interpolation needed in this case
        if after.t == t {
            return Some(after.pos);
        }

        let before = &positions[after_idx - 1];

        let lerp_fact = (t - before.t) / (after.t - before.t);
        debug_assert!((0. ..=1.).contains(&lerp_fact));

        Some(before.pos.lerp(after.pos, lerp_fact))
    }
}

#[derive(Debug, Clone, Copy)]
struct SphereSimulationSnapshot {
    t: f32,
    pos: Vec2,
    vel: Vec2,
}

impl SphereSimulationSnapshot {
    pub fn extrapolate_after(&self, dt: f32) -> Self {
        assert!(dt >= 0.);
        Self {
            t: self.t + dt,
            pos: self.pos + self.vel * dt,
            vel: self.vel,
        }
    }

    pub fn extrapolate_to(&self, t: f32) -> Self {
        assert!(t >= self.t);
        self.extrapolate_after(t - self.t)
    }
}

impl From<SphereSimulationSnapshot> for SpherePositionSnapshot {
    fn from(val: SphereSimulationSnapshot) -> Self {
        SpherePositionSnapshot { t: val.t, pos: val.pos }
    }
}

pub(crate) struct Simulation<'a> {
    world_state: &'a WorldState,
    end_time: f32,
    snapshots: Vec<Vec<SphereSimulationSnapshot>>,
    /// Shape of the simulation's walls
    box_shape: shape::Compound,
}

impl<'a> Simulation<'a> {
    pub fn new(state: &'a WorldState, end_time: f32) -> Self {
        let snapshots = state.spheres.iter().map(|sphere| vec![SphereSimulationSnapshot {
            t: sphere.initial_time,
            pos: sphere.initial_pos,
            vel: sphere.initial_velocity,
        }]).collect_vec();

        let axis_x = na::Unit::new_normalize(na::vector![1., 0.]);
        let axis_y = na::Unit::new_normalize(na::vector![0., 1.]);
        let box_shape = shape::Compound::new(vec![
            (pmath::Isometry::translation(0., 0.), SharedShape::new(shape::HalfSpace::new(axis_x))),
            (pmath::Isometry::translation(0., 0.), SharedShape::new(shape::HalfSpace::new(axis_y))),
            (pmath::Isometry::translation(state.width, 0.), SharedShape::new(shape::HalfSpace::new(-axis_x))),
            (pmath::Isometry::translation(0., state.height), SharedShape::new(shape::HalfSpace::new(-axis_y))),
        ]);

        Self {
            world_state: state,
            end_time,
            snapshots,
            box_shape,
        }
    }

    /// Get the latest snapshot of a sphere checking that the sphere exists at this
    /// point in time.
    /// If Some is returned, asserts that this is the last snapshot, so pushing
    /// into the snapshot vec adds it just after the returned one.
    fn get_last_sphere_snapshot(&self, idx: usize, t: f32) -> Option<&SphereSimulationSnapshot> {
        let snaps = &self.snapshots.get(idx)?;

        debug_assert!(snaps.iter().map(|snap| snap.t).is_sorted());
        let last_snap = snaps.last().expect("All spheres must have at least one snapshot");

        if last_snap.t <= t { Some(last_snap) }
        else {
            // If the last snapshot is in the future we must 
            debug_assert!(snaps.len() == 1);
            None
        }
    }

    /// Computes when the given sphere will collision with the walls of the
    /// simulation, giving a snapshot of its position if any collision is found
    #[must_use]
    fn get_next_sphere_wall_collision(&self, idx: usize, t: f32) -> Option<SphereSimulationSnapshot> {
        let last_snap = self.get_last_sphere_snapshot(idx, t)?;
        // The snapshot at time `t`
        let SphereSimulationSnapshot { t: _, pos, vel } = last_snap.extrapolate_to(t);
        
        let sphere = &self.world_state.spheres[idx];
        // Compute collision from time `t`
        let result = query::cast_shapes(
            &default(), &default(), &self.box_shape,

            &pmath::Isometry::translation(pos.x, pos.y),
            &na::vector![last_snap.vel.x, last_snap.vel.y], &shape::Ball::new(sphere.radius),

            // We use default config which tries to find a collision at any point in time
            // in the future
            query::ShapeCastOptions {
                stop_at_penetration: false,
                max_time_of_impact: self.end_time - t,
                ..default()
            }
        ).expect("Compound on ball should be supported")?;

        let impact_t = result.time_of_impact + t;
        let impact_normal = Vec2::new(result.normal1.x, result.normal1.y);
        let impact_signs = -(impact_normal.abs() * 2. - 1.);

        Some(SphereSimulationSnapshot {
            t: impact_t,
            pos: pos + vel * result.time_of_impact,
            // We only care about what is the axis of collision to reverse it
            vel: vel * impact_signs,
        })
    }

    pub fn run(mut self) -> SimulationResult {
        let state = self.world_state;

        // we start the simulation when the first ball starts
        let start_time = state.spheres.iter().map(|s| s.initial_time)
            .reduce(f32::min)
            .unwrap_or(0.);

        let mut current_time = start_time;

        while current_time < self.end_time {
            // Find the next collision hapenning
            let Some(next_collision_data) = (0..state.spheres.len())
                .filter_map(|sphere_idx|
                    self.get_next_sphere_wall_collision(sphere_idx, current_time)
                        .map(|collision| (sphere_idx, collision))
                )
                .reduce(|best, current| if current.1.t < best.1.t { current } else { best })
            else {
                // NO collisions detected -> finished
                break;
            };

            let (col_sphere_idx, col_snapshot) = next_collision_data;

            println!("current_time = {current_time}");
            current_time = col_snapshot.t;
            self.snapshots[col_sphere_idx].push(col_snapshot);
        }
        println!("current_time = {current_time}");
        println!("FINISHED");

        SimulationResult {
            sphere_positions: self.snapshots.iter().map(|snaps| {
                let mut out = snaps.iter().copied()
                    .map_into::<SpherePositionSnapshot>()
                    .collect_vec();
                // Add a last snapshot at the end of the simulation
                if let Some(last) = snaps.last() { if last.t < self.end_time {
                    out.push(last.extrapolate_to(self.end_time).into());
                } }
                out
            }).collect_vec(),
        }
    }
}

