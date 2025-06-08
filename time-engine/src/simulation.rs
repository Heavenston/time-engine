use std::{collections::HashMap, env::set_current_dir, iter::once, time::Instant};

use crate::{ clip_shapes_on_portal, default, i_shape_to_parry_shape, Portal, StreamId, TimelineId, TimelineMultiverse, WorldState };

use ordered_float::OrderedFloat as OF;
use glam::f32::Vec2;
use i_overlay::i_shape::base::data::Shapes;
use parry2d::{ math as pmath, query, shape };
use nalgebra as na;
use itertools::Itertools;
use tinyvec::ArrayVec;

const MAX_ITERATIONS: usize = 1_000;
const MAX_STAGNATION: usize = 50;

fn resolve_disk_collision(
    m1: f32,
    v1: Vec2,
    m2: f32,
    v2: Vec2,
    normal: Vec2,
) -> (Vec2, Vec2) {
    // Project velocities onto the collision normal
    let v1n = v1.dot(normal);
    let v2n = v2.dot(normal);

    // Tangential components remain unchanged
    let v1t = v1 - normal * v1n;
    let v2t = v2 - normal * v2n;

    // 1D elastic collision equations for normal components
    let v1n_after = (v1n * (m1 - m2) + 2.0 * m2 * v2n) / (m1 + m2);
    let v2n_after = (v2n * (m2 - m1) + 2.0 * m1 * v1n) / (m1 + m2);

    // Combine new normal and unchanged tangential components
    let new_v1 = v1t + normal * v1n_after;
    let new_v2 = v2t + normal * v2n_after;

    (new_v1, new_v2)
}

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
    pub end_age: f32,
}

impl SpherePortalTraversal {
    pub fn swap(self) -> Self {
        Self {
            portal_in_idx: self.portal_out_idx,
            portal_out_idx: self.portal_in_idx,
            direction: self.direction.swap(),
            end_age: self.end_age,
        }
    }

    /// Finding when a sphere wont intersect with the portal anymore
    /// nothing in parry2d can help so we hardcode a sphere-line solution
    fn compute_end_age(sphere_radius: f32, portal: &Portal, snap: &SphereSnapshot) -> f32 {
        let inv_portal_trans = portal.initial_transform.inverse();
        let rel_vel = inv_portal_trans.transform_vector2(snap.vel);
        let rel_pos = inv_portal_trans.transform_point2(snap.pos);

        let t0 = (sphere_radius - rel_pos.x) / rel_vel.x;
        let t1 = (-sphere_radius - rel_pos.x) / rel_vel.x;

        // Not sure on the math of alaways taking the max but anyway it seems to work
        t0.max(t1) + snap.age
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
    pub sid: StreamId,
    pub pos: Vec2,
    pub age: f32,
    pub vel: Vec2,
    pub dead: bool,
    // NOTE: Usage of ArrayVec is mainly for having Copy and not really for
    // performance
    pub portal_traversals: ArrayVec<[SpherePortalTraversal; 4]>,
}

impl SphereSnapshot {
    pub fn extrapolate_after(&self, dt: f32) -> Self {
        assert!(dt >= 0.);
        assert!(!self.dead);
        let t = self.t + dt;
        let age = self.age + dt;
        Self {
            t,
            tid: self.tid,
            sid: self.sid,
            pos: self.pos + self.vel * dt,
            vel: self.vel,
            age,
            dead: false,
            portal_traversals: self.portal_traversals.iter().copied()
                .filter(|traversal| traversal.end_age > age)
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

    pub fn with_swaped_traversal(mut self, portal_in_idx: usize) -> Self {
        let Some((pos, _)) = self.portal_traversals.iter()
            .find_position(|traversal| traversal.portal_in_idx == portal_in_idx)
        else {
            return self;
        };
        self.portal_traversals[pos] = self.portal_traversals[pos].swap();
        self
    }

    /// Changing the velocity requires re-computing the end_t of all portal traversals
    pub fn with_vel(mut self, new_vel: Vec2, sphere_radius: f32, world_state: &WorldState) -> Self {
        self.vel = new_vel;

        // Temporarily take the traversals to not have to borrow self mutably
        let mut traversals = self.portal_traversals;
        for traversal in &mut traversals {
            let portal = &world_state.portals[traversal.portal_in_idx];
            traversal.end_age = SpherePortalTraversal::compute_end_age(sphere_radius, portal, &self);
        }
        self.portal_traversals = traversals;

        self
    }

    pub fn dead(mut self) -> Self {
        self.dead = true;
        self
    }

    /// Change vel and pos without any other computation, should be used
    /// only for portal traversals as this does not change the [PortalTraversal::end_t]
    pub fn teleport(mut self, new_pos: Vec2, new_vel: Vec2) -> Self {
        self.vel = new_vel;
        self.pos = new_pos;
        self
    }

    pub fn offset_time(mut self, multiverse: &TimelineMultiverse, dt: f32) -> Self {
        self.t += dt;
        if dt < 0. {
            print!("{:?} -> ", self.tid);
            self.tid = multiverse.create_children(self.tid, self.t);
            println!("{:?}", self.tid);
            println!("{multiverse:#?}");
            self.sid = StreamId::new();
        }
        self
    }

    pub fn with_tid(mut self, tid: TimelineId) -> Self {
        self.tid = tid;
        self
    }
}

#[derive(Clone, Debug)]
pub struct SimulatedSphere {
    pub radius: f32,
    pub snapshots: Vec<SphereSnapshot>,
}

impl SimulatedSphere {
    pub fn sids(&self) -> impl Iterator<Item = StreamId> + Clone {
        self.snapshots.iter()
            .map(|snap| snap.sid)
            .unique()
    }

    pub fn tids(&self, sid: StreamId) -> impl Iterator<Item = TimelineId> + Clone {
        self.snapshots.iter()
            .filter(move |snap| snap.sid == sid)
            .map(|snap| snap.tid)
            .unique()
    }

    /// Get the earliest snapshot for the given sphere at the given time and timeline.
    pub fn get_last_snapshot(&self, multiverse: &TimelineMultiverse, t: f32, sid: StreamId, tid: TimelineId) -> Option<&SphereSnapshot> {
        let snaps = &self.snapshots;

        debug_assert!(
            snaps.iter()
                .filter(|snap| snap.sid == sid)
                .filter(|snap| snap.tid == tid)
                .map(|snap| snap.age)
                .is_sorted(),
        );

        // FIXME: Sort is probably not required if done some other way
        // NOTE: The assert above is only for a single timeline, we have
        //       multiple timelines here
        let sorted = snaps.iter().rev()
            .filter(|snap| snap.sid == sid)
            .filter(|snap| multiverse.is_parent(snap.tid, tid))
            .sorted_by_key(|snap| OF(-snap.age))
            .collect_vec();

        sorted.iter().copied()
            .filter(|snap| snap.tid == tid)
            .find(|snap| snap.t <= t)
            .or(
                sorted.iter().copied()
                    .find(|snap| snap.t <= t)
            )
    }

    /// Get an extrapolated snapshot of the given sphere up to t
    /// Returns None if the sphere has not yet appeared or is dead
    pub fn get_snapshot(&self, multiverse: &TimelineMultiverse, t: f32, sid: StreamId, tid: TimelineId) -> Option<SphereSnapshot> {
        if t < tid.start() {
            return None;
        }

        // If not snapshot is available before t, this means the sphere does not exist
        // at this time
        let last_snap = self.get_last_snapshot(multiverse, t, sid, tid)?;
        if last_snap.dead {
            return None;
        }
        Some(last_snap.extrapolate_to(t))
    }
}

#[derive(Default, Debug)]
pub struct SimulationResult {
    pub spheres: Vec<SimulatedSphere>,
    pub multiverse: TimelineMultiverse,
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

#[derive(Debug, Clone)]
struct SimulationCollisionResult {
    /// Used for debugging
    #[allow(dead_code)]
    kind: &'static str,
    at_t: f32,
    new_snapshots: Vec<(usize, SphereSnapshot)>,
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
                    sid: default(),
                    pos: sphere.initial_pos,
                    vel: sphere.initial_velocity,
                    age: 0.,
                    dead: false,
                    portal_traversals: default()
                }]
            }).collect_vec(),

            world_state: state,
            multiverse,
            end_time,
            walls_shapes: state.get_static_body_collision(),
        }
    }

    fn get_sphere_collision_shape(&self, idx: usize) -> Option<impl shape::Shape> {
        let sphere = self.spheres.get(idx)?;
        Some(shape::Ball::new(sphere.radius))
    }

    /// Clips the reverse side of any portal the given sphere is currently traversing from
    /// the given shapes
    fn clip_shape_for_sphere_collisions(&self, sphere_idx: usize, t: f32, sid: StreamId, tid: TimelineId, mut shapes: Shapes<Vec2>) -> Shapes<Vec2> {
        let snap = self.spheres[sphere_idx].get_snapshot(&self.multiverse, t, sid, tid)
            .expect("present");

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
    fn get_next_sphere_wall_collision(&self, idx: usize, t: f32, sid: StreamId, tid: TimelineId) -> Option<SimulationCollisionResult> {
        let sphere = self.spheres.get(idx)?;

        let snap = sphere.get_snapshot(&self.multiverse, t, sid, tid)?;
        if snap.tid != tid {
            return None;
        }

        let sphere_shape = self.get_sphere_collision_shape(idx)?;

        let wall_shapes = self.clip_shape_for_sphere_collisions(idx, t, sid, tid, self.walls_shapes.clone());
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

        Some(SimulationCollisionResult {
            kind: "sphere-wall",
            at_t: impact_t,
            new_snapshots: vec![
                (idx, snap.extrapolate_to(impact_t).with_vel(new_vel, sphere.radius, self.world_state))
            ],
        })
    }

    #[must_use]
    fn get_next_sphere_sphere_collision(&self, idx1: usize, sid1: StreamId, idx2: usize, sid2: StreamId, t: f32, tid: TimelineId) -> Option<SimulationCollisionResult> {
        // println!("{idx1}.{sid1} <-> {idx2}.{sid2} in tid {tid} at {t}");
        let c1 = (idx1, sid1);
        let c2 = (idx2, sid2);
        debug_assert_ne!(c1, c2);
        
        let sphere1 = self.spheres.get(idx1)?;
        let sphere2 = self.spheres.get(idx2)?;

        let snap1 = sphere1.get_snapshot(&self.multiverse, t, sid1, tid)?;
        let snap2 = sphere2.get_snapshot(&self.multiverse, t, sid2, tid)?;
        if snap1.tid != tid && snap2.tid != tid {
            return None;
        }

        let sphere_shape1 = self.get_sphere_collision_shape(idx1)?;
        let sphere_shape2 = self.get_sphere_collision_shape(idx2)?;

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

        // println!("Found impact at {impact_t}");

        let normal = Vec2::new(result.normal1.x, result.normal1.y);
        let (vel1, vel2) = resolve_disk_collision(1., snap1.vel, 1., snap2.vel, normal);

        Some(SimulationCollisionResult {
            kind: "sphere-sphere",
            at_t: impact_t,
            new_snapshots: vec![
                (
                    idx1,
                    snap1.extrapolate_to(impact_t)
                        .with_tid(tid)
                        .with_vel(vel1, sphere1.radius, self.world_state)
                ),
                (
                    idx2,
                    snap2.extrapolate_to(impact_t)
                        .with_tid(tid)
                        .with_vel(vel2, sphere2.radius, self.world_state)
                ),
            ],
        })
    }

    fn get_next_sphere_portal_traversals_start(&self, sphere_idx: usize, portal_idx: usize, t: f32, sid: StreamId, tid: TimelineId) -> Option<SimulationCollisionResult> {
        let sphere = self.spheres.get(sphere_idx)?;
        let sphere_shape = self.get_sphere_collision_shape(sphere_idx)?;
        let portal = &self.world_state.portals[portal_idx];
        let portal_shape = self.get_portal_shape(portal_idx, t);

        let snap = sphere.get_snapshot(&self.multiverse, t, sid, tid)?;
        if snap.tid != tid {
            return None;
        }
        if snap.portal_traversals.iter().any(|tr| tr.portal_in_idx == portal_idx) {
            return None;
        }

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
        let direction = if result.normal1.x < 0. { PortalDirection::Front } else { PortalDirection::Back };

        let impact_snap = snap.extrapolate_to(impact_t);
        let impact_snap = impact_snap.with_portal_traversal(SpherePortalTraversal {
            portal_in_idx: portal_idx,
            portal_out_idx: portal.link_to,
            direction,
            end_age: SpherePortalTraversal::compute_end_age(sphere.radius, portal, &impact_snap),
        });

        println!("start");
        Some(SimulationCollisionResult {
            kind: "sphere-portal-start",
            at_t: impact_t,
            new_snapshots: vec![
                (sphere_idx, impact_snap)
            ]
        })
    }

    fn get_next_sphere_portal_traversals_end(&self, sphere_idx: usize, portal_idx: usize, t: f32, sid: StreamId, tid: TimelineId) -> Option<SimulationCollisionResult> {
        let sphere = self.spheres.get(sphere_idx)?;
        let snap = sphere.get_snapshot(&self.multiverse, t, sid, tid)?;
        if snap.tid != tid {
            return None;
        }
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

        let impact_t = t + impact_dt + 0.00001;
        let time_travel_dt = out_portal.time_offset - portal.time_offset;

        let impact_snap = snap
            .extrapolate_to(impact_t);
        let after_impact_snap = impact_snap
            .teleport(
                out_portal.initial_transform.transform_point2(inv_portal_trans.transform_point2(impact_snap.pos)),
                out_portal.initial_transform.transform_vector2(inv_portal_trans.transform_vector2(impact_snap.vel)),
            )
            .with_swaped_traversal(portal_idx)
            .offset_time(&self.multiverse, time_travel_dt);

        println!("end");
        Some(SimulationCollisionResult {
            kind: "sphere-portal-end",
            at_t: impact_t,
            new_snapshots: vec![
                (sphere_idx, impact_snap.dead()),
                (sphere_idx, after_impact_snap),
            ],
        })
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
        'simulation: while current_times.values().all(|&t| t < self.end_time) && iterations < MAX_ITERATIONS {
            print!("\niteration {iterations}: ");
            iterations += 1;

            let Some((&tid, &t)) = current_times.iter().min_by_key(|(_, t)| OF(**t))
            else {
                break;
            };
            println!("{tid:?} at {t}s");
            println!("{:?}", self.multiverse);

            let spheres_iter = (0..self.spheres.len())
                .flat_map(|sphere_idx| self.spheres[sphere_idx].sids().map(move |sid| (sphere_idx, sid)))
                // keep only spheres that exist in this timeline
                .filter(|&(idx, sid)| self.spheres[idx].tids(sid)
                    .any(|tid_| self.multiverse.is_parent(tid_, tid)))
            ;

            // Find the next collision hapenning
            let next_collision_datas = std::iter::empty()
                .chain(
                    spheres_iter.clone()
                    .filter_map(|(sphere_idx, sid)|
                        self.get_next_sphere_wall_collision(sphere_idx, t, sid, tid)
                    )
                )
                .chain(
                    spheres_iter.clone().array_combinations::<2>()
                    .flat_map(|[(idx1, sid1), (idx2, sid2)]| {
                        self.get_next_sphere_sphere_collision(idx1, sid1, idx2, sid2, t, tid)
                            .into_iter()
                    })
                )
                .chain(
                    spheres_iter.clone()
                    .cartesian_product(0..state.portals.len())
                    .flat_map(|((sphere_idx, sid), portal_idx)| {
                        self.get_next_sphere_portal_traversals_start(sphere_idx, portal_idx, t, sid, tid)
                            .into_iter()
                    })
                )
                .chain(
                    spheres_iter.clone()
                    .cartesian_product(0..state.portals.len())
                    .flat_map(|((sphere_idx, sid), portal_idx)| {
                        self.get_next_sphere_portal_traversals_end(sphere_idx, portal_idx, t, sid, tid)
                            .into_iter()
                    })
                )
                .min_set_by_key(|val| OF(val.at_t));

            // why do i have to manually drop it i though rust was smart >:(
            drop(spheres_iter);

            // println!("{next_collision_datas:#?}");
            if next_collision_datas.is_empty() {
                // If not collision was detected it may be because we are too early
                // we need to wait for a ball to spawn
                // this is done by finding a snaphshot in the future which
                // can only happen when a ball goes from dead to not dead
                let next_ball_spawn = self.spheres.iter()
                    .flat_map(|s| &s.snapshots)
                    .filter(|snap| self.multiverse.is_parent(snap.tid, tid))
                    .filter(|snap| snap.t > t)
                    .min_by_key(|snap| OF(snap.t));
                if let Some(next_ball_spawn) = next_ball_spawn {
                    debug_assert!(!next_ball_spawn.dead);
                    println!("Found ball spawn: {next_ball_spawn:?}");
                    current_times.insert(tid, next_ball_spawn.t);
                }
                else {
                    // timeline is assumed dead
                    println!("Dead timeline {tid:?}");
                    current_times.remove(&tid);
                    continue;
                }
            }

            for (tid, mut snapshots) in next_collision_datas.iter()
                .flat_map(|data| &data.new_snapshots)
                .map(|(_, snap)| snap)
                .sorted_by_key(|snap| snap.tid)
                .chunk_by(|snap| snap.tid)
                .into_iter()
            {
                let min_t = snapshots.next().expect("Chunks are not empty").t;

                // Snapshots of a single timeline must be sorted
                debug_assert!(once(min_t).chain(snapshots.map(|snap| snap.t)).is_sorted());
                let timeline_t = current_times.get(&tid).copied();
                if let Some(timeline_t) = timeline_t {
                    // cannot backtrack a single timeline
                    debug_assert!(timeline_t <= min_t, "Tried to backtrack a timeline: {min_t}s is before {timeline_t}s");
                    if timeline_t == min_t {
                        stagnation += 1;
                    }
                    else {
                        stagnation = 0;
                    }
                    if stagnation > MAX_STAGNATION {
                        eprintln!("Stagnation detected on {timeline_t}s ({next_collision_datas:#?})");
                        break 'simulation;
                    }
                }
                println!("tid {tid} now at {min_t}");
                current_times.insert(tid, min_t);
            }
            println!("current_times: {current_times:?}");

            for SimulationCollisionResult { kind: _, at_t, new_snapshots } in next_collision_datas {
                debug_assert_eq!(
                    Some(at_t),
                    current_times.get(&tid).copied(),
                );
                for (idx, snap) in new_snapshots {
                    let snaps = &mut self.spheres[idx].snapshots;
                    // Asserts that the new snapshot does not break sorting 
                    debug_assert!(snaps.iter().filter(|s| s.tid == snap.tid).all(|s| s.t <= snap.t));
                    snaps.push(snap);
                }
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
            for sid in sphere.sids() {
                for tid in sphere.tids(sid) {
                    let last = sphere.snapshots.iter().rev()
                        .find(|snap| snap.sid == sid && snap.tid == tid)
                        .expect("(tid, sid) couple exist");
                    // Add a last snapshot at the end of the simulation
                    if last.t < stop_time && !last.dead {
                        out.push(last.extrapolate_to(stop_time));
                    }
                }
            }
            sphere.snapshots = out;
        }

        SimulationResult {
            spheres: self.spheres,
            multiverse: self.multiverse,
        }
    }
}

