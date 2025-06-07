use std::{collections::HashMap, time::Instant};

use crate::{ clip_shapes_on_portal, default, i_shape_to_parry_shape, Portal, TimelineId, TimelineMultiverse, WorldState };

use ordered_float::OrderedFloat as OF;
use glam::f32::Vec2;
use i_overlay::i_shape::base::data::Shapes;
use parry2d::{ math as pmath, query, shape };
use nalgebra as na;
use itertools::Itertools;
use tinyvec::ArrayVec;

const MAX_ITERATIONS: usize = 1_000;
const MAX_STAGNATION: usize = 100;

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortalDirection {
    #[default]
    Front,
    Back,
}

impl PortalDirection {
    pub fn is_front(self) -> bool {
        self == Self::Front
    }

    pub fn is_back(self) -> bool {
        self == Self::Back
    }

    pub fn swap(self) -> Self {
        match self {
            PortalDirection::Front => PortalDirection::Back,
            PortalDirection::Back => PortalDirection::Front,
        }
    }
}

#[derive(Default, Debug, Clone, Copy)]
pub struct SpherePortalTraversal {
    pub portal_in_idx: usize,
    pub portal_out_idx: usize,
    pub direction: PortalDirection,
    pub end_t: f32,
}

impl SpherePortalTraversal {
    pub fn swap(self) -> Self {
        Self {
            portal_in_idx: self.portal_out_idx,
            portal_out_idx: self.portal_in_idx,
            direction: self.direction.swap(),
            end_t: self.end_t,
        }
    }

    /// Finding when a sphere wont intersect with the portal anymore
    /// nothing in parry2d can help so we hardcode a sphere-line solution
    fn compute_end_t(sphere_radius: f32, portal: &Portal, snap: &SphereSnapshot) -> f32 {
        let inv_portal_trans = portal.initial_transform.inverse();
        let rel_vel = inv_portal_trans.transform_vector2(snap.vel);
        let rel_pos = inv_portal_trans.transform_point2(snap.pos);

        let t0 = (sphere_radius - rel_pos.x) / rel_vel.x;
        let t1 = (-sphere_radius - rel_pos.x) / rel_vel.x;

        // Not sure on the math of alaways taking the max but anyway it seems to work
        t0.max(t1) + snap.t
    }

    // pub fn end_t(&self, snap: &SphereSnapshot) -> f32 {
    //     let inv_portal_trans = self.portal_in.initial_transform.inverse();
    //     let rel_vel = inv_portal_trans.transform_vector2(snap.vel);
    //     let rel_pos = inv_portal_trans.transform_point2(snap.pos);

    //     // Finding when the sphere wont intersect with the portal anymore
    //     // nothing in parry2d can help so we hardcode a sphere-line solution
    //     let exit_dt = {
    //         let t0 = (sphere.radius - rel_pos.x) / rel_vel.x;
    //         let t1 = (-sphere.radius - rel_pos.x) / rel_vel.x;

    //         // Not sure on the math of alaways taking the max but anyway
    //         t0.max(t1)
    //     };
    // }
}

#[derive(Debug, Clone, Copy)]
pub struct SphereSnapshot {
    pub t: f32,
    pub tid: TimelineId,
    pub pos: Vec2,
    pub age: f32,
    pub vel: Vec2,
    // NOTE: Usage of ArrayVec is mainly for having Copy and not really for
    // performance
    pub portal_traversals: ArrayVec<[SpherePortalTraversal; 4]>,
}

impl SphereSnapshot {
    pub fn extrapolate_after(&self, dt: f32) -> Self {
        assert!(dt >= 0.);
        let t = self.t + dt;
        Self {
            t,
            tid: self.tid,
            pos: self.pos + self.vel * dt,
            vel: self.vel,
            age: self.age + dt,
            portal_traversals: self.portal_traversals.iter().copied()
                .filter(|traversal| traversal.end_t > t)
                .collect(),
        }
    }

    pub fn extrapolate_to(&self, t: f32) -> Self {
        assert!(t >= self.t);
        self.extrapolate_after(t - self.t)
    }

    // FIXME: Function assumption: Adding portal traversal has no side effect on the other
    // traversals the sphere already has
    pub fn with_portal_traversal(mut self, traversal: SpherePortalTraversal) -> Self {
        // asserts we arn't already traversing this portal which would be a contradiction
        // FIXME: This can technically happen but how to handle it ?
        debug_assert!(!self.portal_traversals.iter()
            .flat_map(|t| [t.portal_in_idx, t.portal_out_idx])
            .any(|t2| t2 == traversal.portal_in_idx || t2 == traversal.portal_out_idx));

        self.portal_traversals.push(traversal);
        self
    }

    /// Changing the velocity requires re-computing the end_t of all portal traversals
    pub fn with_vel(mut self, new_vel: Vec2, sphere_radius: f32, world_state: &WorldState) -> Self {
        self.vel = new_vel;

        // Temporarily take the traversals to not have to borrow self mutably
        let mut traversals = self.portal_traversals;
        for traversal in &mut traversals {
            let portal = &world_state.portals[traversal.portal_in_idx];
            traversal.end_t = SpherePortalTraversal::compute_end_t(sphere_radius, portal, &self);
        }
        self.portal_traversals = traversals;

        self
    }
}

#[derive(Clone, Debug)]
pub struct SimulatedSphere {
    pub radius: f32,
    pub snapshots: Vec<SphereSnapshot>,
}

impl SimulatedSphere {
    pub fn interpolate_snapshot(&self, t: f32) -> Option<SphereSnapshot> {
        // Find the first snapshot that is after the given time
        let (after_idx, after) = self.snapshots.iter().enumerate()
            .find(|(_, snap)| snap.t >= t)?;

        // The first snapshot is after the given time so the sphere is not
        // spawned yet
        if after_idx == 0 && after.t > t {
            return None;
        }

        // No interpolation needed in this case
        if after.t == t {
            return Some(*after);
        }

        let before = &self.snapshots[after_idx - 1];

        let lerp_fact = (t - before.t) / (after.t - before.t);
        debug_assert!((0. ..=1.).contains(&lerp_fact));

        Some(SphereSnapshot {
            t,
            tid: before.tid,
            pos: before.pos.lerp(after.pos, lerp_fact),
            age: before.age + ((after.age - before.age) * lerp_fact),
            vel: before.vel,
            portal_traversals: before.portal_traversals.iter().copied()
                .filter(|traversal| traversal.end_t >= t)
                .collect(),
        })
    }
}

#[derive(Default, Clone, Debug)]
pub struct SimulationResult {
    pub spheres: Vec<SimulatedSphere>,
}

impl SimulationResult {
    pub fn new() -> Self {
        default()
    }

    pub fn max_t(&self) -> f32 {
        self.spheres.iter()
            .flat_map(|sphere| sphere.snapshots.iter().map(|snap| snap.t))
            .reduce(f32::max)
            .unwrap_or(f32::INFINITY)
    }
}

pub(crate) struct Simulation<'a> {
    world_state: &'a WorldState,
    multiverse: TimelineMultiverse,
    end_time: f32,
    spheres: Vec<SimulatedSphere>,
    /// Shape of the simulation's walls
    walls_shapes: Shapes<Vec2>,
}

impl<'a> Simulation<'a> {
    pub fn new(state: &'a WorldState, end_time: f32) -> Self {
        let multiverse = TimelineMultiverse::new();
        Self {
            spheres: state.start_spheres.iter().map(|sphere| SimulatedSphere {
                radius: sphere.radius,
                snapshots: vec![SphereSnapshot {
                    t: sphere.initial_time,
                    tid: multiverse.root(),
                    pos: sphere.initial_pos,
                    vel: sphere.initial_velocity,
                    age: 0.,
                    portal_traversals: default()
                }]
            }).collect_vec(),

            world_state: state,
            multiverse,
            end_time,
            walls_shapes: state.get_static_body_collision(),
        }
    }

    /// Get the earliest snapshot for the given sphere at the given time and timeline.
    fn get_last_sphere_snapshot(&self, idx: usize, t: f32, tid: TimelineId) -> Option<&SphereSnapshot> {
        let snaps = &self.spheres.get(idx)?.snapshots;

        debug_assert!(
            snaps.iter()
                .filter(|snap| self.multiverse.is_parent(snap.tid, tid))
                .map(|snap| snap.t)
                .is_sorted()
        );

        snaps.iter().rev()
            .filter(|snap| self.multiverse.is_parent(snap.tid, tid))
            .find(|snap| snap.t <= t)
    }

    /// Get an extrapolated snapshot of the given sphere up to t
    /// Returns None if the sphere has not yet appeared
    fn get_sphere_snapshot(&self, idx: usize, t: f32, tid: TimelineId) -> Option<SphereSnapshot> {
        // If not snapshot is available before t, this means the sphere does not exist
        // at this time
        let last_snap = self.get_last_sphere_snapshot(idx, t, tid)?;
        Some(last_snap.extrapolate_to(t))
    }

    fn get_sphere_collision_shape(&self, idx: usize, _t: f32, _tid: TimelineId) -> Option<impl shape::Shape> {
        let sphere = self.spheres.get(idx)?;
        Some(shape::Ball::new(sphere.radius))
    }

    /// Clips the reverse side of any portal the given sphere is currently traversing from
    /// the given shapes
    fn clip_shape_for_sphere_collisions(&self, sphere_idx: usize, t: f32, tid: TimelineId, mut shapes: Shapes<Vec2>) -> Shapes<Vec2> {
        let snap = self.get_sphere_snapshot(sphere_idx, t, tid).unwrap();

        for traversal in snap.portal_traversals {
            let portal = &self.world_state.portals[traversal.portal_in_idx];
            shapes = clip_shapes_on_portal(shapes, portal, traversal.direction);
        }

        shapes
    }

    fn get_portal_shape(&self, idx: usize, _t: f32) -> impl shape::Shape {
        let portal = &self.world_state.portals[idx];
        let h2 = portal.height / 2.;
        shape::Polyline::new(vec![
            na::point![0., -h2],
            na::point![0., h2],
        ], None)
    }

    /// Computes when the given sphere will collision with the walls of the
    /// simulation, giving a snapshot of its position if any collision is found
    #[must_use]
    fn get_next_sphere_wall_collision(&self, idx: usize, t: f32, tid: TimelineId) -> Option<SphereSnapshot> {
        let snap = self.get_sphere_snapshot(idx, t, tid)?;
        let sphere = self.spheres.get(idx)?;
        let sphere_shape = self.get_sphere_collision_shape(idx, t, tid)?;

        let wall_shapes = self.clip_shape_for_sphere_collisions(idx, t, tid, self.walls_shapes.clone());
        let wall_collision = i_shape_to_parry_shape(wall_shapes);
        
        // Compute collision from time `t`
        let result = query::cast_shapes(
            &default(), &default(), &wall_collision,

            &pmath::Isometry::translation(snap.pos.x, snap.pos.y),
            &na::vector![snap.vel.x, snap.vel.y], &sphere_shape,

            query::ShapeCastOptions {
                stop_at_penetration: false,
                max_time_of_impact: self.end_time - t,
                ..default()
            }
        ).expect("Compound on ball should be supported")?;

        let impact_t = result.time_of_impact + t;
        let impact_normal = Vec2::new(result.normal1.x, result.normal1.y).normalize();
        let new_vel = snap.vel - 2. * impact_normal * snap.vel.dot(impact_normal);

        Some(snap.extrapolate_to(impact_t).with_vel(new_vel, sphere.radius, self.world_state))
    }

    #[must_use]
    fn get_next_sphere_sphere_collision(&self, idx1: usize, idx2: usize, t: f32, tid: TimelineId) -> Option<(SphereSnapshot, SphereSnapshot)> {
        let snap1 = self.get_sphere_snapshot(idx1, t, tid)?;
        let snap2 = self.get_sphere_snapshot(idx2, t, tid)?;
        
        let sphere1 = self.spheres.get(idx1)?;
        let sphere2 = self.spheres.get(idx2)?;

        let sphere_shape1 = self.get_sphere_collision_shape(idx1, t, tid)?;
        let sphere_shape2 = self.get_sphere_collision_shape(idx2, t, tid)?;

        // Compute collision from time `t`
        let result = query::cast_shapes(
            &pmath::Isometry::translation(snap1.pos.x, snap1.pos.y),
            &na::vector![snap1.vel.x, snap1.vel.y], &sphere_shape1,

            &pmath::Isometry::translation(snap2.pos.x, snap2.pos.y),
            &na::vector![snap2.vel.x, snap2.vel.y], &sphere_shape2,

            query::ShapeCastOptions {
                stop_at_penetration: false,
                max_time_of_impact: self.end_time - t,
                ..default()
            }
        ).expect("Ball on ball should be supported")?;

        let impact_t = result.time_of_impact + t;

        // Completely elastic collision with two sphere of the same weight
        // simplified as just a swap of velocities
        Some((
            snap1.extrapolate_to(impact_t).with_vel(snap2.vel, sphere1.radius, self.world_state),
            snap2.extrapolate_to(impact_t).with_vel(snap1.vel, sphere2.radius, self.world_state),
        ))
    }

    fn get_next_sphere_portal_traversals_start(&self, sphere_idx: usize, portal_idx: usize, t: f32, tid: TimelineId) -> Option<SphereSnapshot> {
        let snap = self.get_sphere_snapshot(sphere_idx, t, tid)?;
        if snap.portal_traversals.iter().any(|tr| tr.portal_in_idx == portal_idx) {
            return None;
        }
        let sphere = self.spheres.get(sphere_idx)?;
        let sphere_shape = self.get_sphere_collision_shape(sphere_idx, t, tid)?;
        let portal = &self.world_state.portals[portal_idx];
        let portal_shape = self.get_portal_shape(portal_idx, t);

        // Anoying and imperfect conversion from glam's Isometry to nalgebra's 
        let plane_iso = {
            let (scale, angle, trans) = portal.initial_transform.to_scale_angle_translation();
            assert_eq!(scale, Vec2::splat(1.));
            pmath::Isometry::from_parts(
                na::Translation2::new(trans.x, trans.y),
                na::UnitComplex::from_angle(angle)
            )
        };

        let result = query::cast_shapes(
            &plane_iso,
            // portal has no velocity
            &na::vector![0., 0.],
            &portal_shape,

            &pmath::Isometry::translation(snap.pos.x, snap.pos.y),
            &na::vector![snap.vel.x, snap.vel.y],
            &sphere_shape,

            query::ShapeCastOptions {
                stop_at_penetration: true,
                max_time_of_impact: self.end_time - t,
                ..default()
            }
        ).expect("Ball on ball should be supported")?;

        let impact_t = result.time_of_impact + t;
        let impact_snap = snap.extrapolate_to(impact_t);
        let direction = if result.normal1.x < 0. { PortalDirection::Front } else { PortalDirection::Back };

        Some(impact_snap.with_portal_traversal(SpherePortalTraversal {
            portal_in_idx: portal_idx,
            portal_out_idx: portal.link_to,
            direction,
            end_t: SpherePortalTraversal::compute_end_t(sphere.radius, portal, &impact_snap),
        }))
    }

    fn get_next_sphere_portal_traversals_end(&self, sphere_idx: usize, portal_idx: usize, t: f32, tid: TimelineId) -> Option<[SphereSnapshot; 2]> {
        let snap = self.get_sphere_snapshot(sphere_idx, t, tid)?;
        if !snap.portal_traversals.iter().any(|tr| tr.portal_in_idx == portal_idx) {
            return None;
        }
        let portal = &self.world_state.portals[portal_idx];
        let out_portal_idx = portal.link_to;
        let out_portal = &self.world_state.portals[out_portal_idx];

        let inv_portal_trans = portal.initial_transform.inverse();
        let rel_vel = inv_portal_trans.transform_vector2(snap.vel);
        let rel_pos = inv_portal_trans.transform_point2(snap.pos);

        // compute when the center of the sphere touches the portal
        // nothing in parry2d can help so we hardcode the solution
        let impact_dt = -rel_pos.x / rel_vel.x;
        // Impact in the past or alread on the impact -> nothing to be done
        if impact_dt <= 0. {
            return None;
        }

        let impact_t = t + impact_dt;

        let impact_snap = snap.extrapolate_to(impact_t + 0.00001);
        let after_impact_snap = SphereSnapshot {
            portal_traversals: impact_snap.portal_traversals.into_iter().map(|traversal| {
                if traversal.portal_in_idx == portal_idx {
                    traversal.swap()
                }
                else {
                    traversal
                }
            }).collect(),
            pos: out_portal.initial_transform.transform_point2(inv_portal_trans.transform_point2(impact_snap.pos)),
            vel: out_portal.initial_transform.transform_vector2(inv_portal_trans.transform_vector2(impact_snap.vel)),
            ..impact_snap
        };

        Some([impact_snap, after_impact_snap])
    }

    pub fn run(mut self) -> SimulationResult {
        let state = self.world_state;

        // we start the simulation when the first ball starts
        let initial_start_time = self.world_state.start_spheres.iter().map(|s| s.initial_time)
            .reduce(f32::min)
            .unwrap_or(0.);

        let mut current_times = HashMap::<TimelineId, f32>::new();
        current_times.insert(self.multiverse.root(), initial_start_time);

        let mut iterations = 0;
        // Used to detect when the same collision is detected multiple times
        let mut stagnation = 0;
        let start_instant = Instant::now();
        while current_times.values().all(|&t| t < self.end_time) && iterations < MAX_ITERATIONS {
            // println!("\n{iterations}");
            iterations += 1;

            let Some((&tid, t)) = current_times.iter_mut().min_by_key(|(_, t)| OF(**t))
            else {
                continue;
            };

            // Find the next collision hapenning
            let next_collision_datas = (0..self.spheres.len())
                .filter_map(|sphere_idx|
                    self.get_next_sphere_wall_collision(sphere_idx, *t, tid)
                        .map(|collision| (sphere_idx, collision))
                )
                .chain(
                    (0..self.spheres.len()).array_combinations::<2>()
                    .flat_map(|[idx1, idx2]| {
                        self.get_next_sphere_sphere_collision(idx1, idx2, *t, tid)
                            .into_iter()
                            .flat_map(move |(a, b)| [(idx1, a), (idx2, b)])
                    })
                )
                .chain(
                    (0..self.spheres.len())
                    .cartesian_product(0..state.portals.len())
                    .flat_map(|(sphere_idx, portal_idx)| {
                        self.get_next_sphere_portal_traversals_start(sphere_idx, portal_idx, *t, tid)
                            .into_iter()
                            .map(move |snapshot| (sphere_idx, snapshot))
                    })
                )
                .chain(
                    (0..self.spheres.len())
                    .cartesian_product(0..state.portals.len())
                    .flat_map(|(sphere_idx, portal_idx)| {
                        self.get_next_sphere_portal_traversals_end(sphere_idx, portal_idx, *t, tid)
                            .into_iter()
                            .flatten()
                            .map(move |snapshot| (sphere_idx, snapshot))
                    })
                )
                .min_set_by(|best, current| best.1.t.total_cmp(&current.1.t));
            if next_collision_datas.is_empty() {
                break;
            }

            if let Some((_, col_snapshot)) = next_collision_datas.first() {
                debug_assert!(*t <= col_snapshot.t);
                if *t == col_snapshot.t {
                    stagnation += 1;
                }
                else {
                    stagnation = 0;
                }
                if stagnation > MAX_STAGNATION {
                    eprintln!("Stagnation detected on {t}s ({next_collision_datas:#?})");
                    break;
                }
                *t = col_snapshot.t;
            }

            for (col_sphere_idx, col_snapshot) in next_collision_datas {
                debug_assert_eq!(*t, col_snapshot.t);
                debug_assert!(
                    self.spheres[col_sphere_idx].snapshots.iter().rev()
                        .find(|snap| self.multiverse.is_parent(snap.tid, tid))
                        .expect("Always at least one snapshot")
                        .t <= col_snapshot.t
                );
                self.spheres[col_sphere_idx].snapshots.push(col_snapshot);
            }
        }
        assert!(iterations < MAX_ITERATIONS, "Max iterations reached ({MAX_ITERATIONS}) {current_times:#?}");
        println!("FINISHED in {iterations} iterations (took {:?})", start_instant.elapsed());

        // If there was an abrupt simulation stop we must no simulate up to end_time
        let stop_time = if stagnation >= MAX_STAGNATION {
            current_times.values().copied().reduce(f32::max).unwrap()
        } else {
            self.end_time
        };

        for sphere in &mut self.spheres {
            let mut out = sphere.snapshots.iter().copied()
                .map_into::<SphereSnapshot>()
                .collect_vec();
            // Add a last snapshot at the end of the simulation
            if let Some(last) = sphere.snapshots.last() { if last.t < stop_time {
                out.push(last.extrapolate_to(stop_time));
            } }
        }

        SimulationResult {
            spheres: self.spheres,
        }
    }
}

