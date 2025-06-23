use std::{any::Any, sync::Arc};

use rand::{Rng, SeedableRng};
use time_engine as te;
use macroquad::prelude::{Affine2, Vec2};

pub trait Scene: Send + Sync + Any + 'static {
    fn name(&self) -> &'static str;
    fn create_world_state(&self) -> te::WorldState;

    fn default_max_time(&self) -> f32 {
        15.
    }
}

pub struct PolchinskiParadox {
    enable_time_travel: bool,
}

impl Scene for PolchinskiParadox {
    fn name(&self) -> &'static str {
        if self.enable_time_travel {
            "Polchinski Paradox"
        }
        else {
            "Polchinski Paradox but without time travel"
        }
    }

    fn create_world_state(&self) -> time_engine::WorldState {
        let mut sim = te::WorldState::new(100., 100.);
        sim.push_portal(te::Portal {
            height: 20.,
            in_transform: Affine2::from_angle_translation(
                std::f32::consts::PI,
                Vec2::new(95., 50.),
            ),
            out_transform: Affine2::from_angle_translation(
                std::f32::consts::FRAC_PI_2,
                Vec2::new(50., 95.),
            ),
            time_offset: if self.enable_time_travel { 2.8 } else { 0. },
        });
        sim.push_ball(te::Ball {
            initial_pos: Vec2::new(50., 10.), 
            initial_velocity: Vec2::new(0., 30.),
            radius: 3.,
            ..Default::default()
        });
        sim
    }
}

pub struct CollisionBehindPortal;

impl Scene for CollisionBehindPortal {
    fn name(&self) -> &'static str {
        "Collision behind portal"
    }

    fn create_world_state(&self) -> time_engine::WorldState {
        let mut sim = te::WorldState::new(50., 50.);
        sim.push_portal(te::Portal {
            height: 10.,
            in_transform: Affine2::from_angle_translation(
                0.,
                Vec2::new(48., 25.),
            ),
            out_transform: Affine2::from_angle_translation(
                std::f32::consts::FRAC_PI_2,
                Vec2::new(25., 48.),
            ),
            time_offset: 0.,
        });
        sim.push_ball(te::Ball {
            initial_pos: Vec2::new(25., 3.), 
            initial_velocity: Vec2::new(0., 15.),
            radius: 3.,
            ..Default::default()
        });
        sim
    }

    fn default_max_time(&self) -> f32 {
        5.
    }
}

pub struct SinglePortalScene {
    pub inverted: bool,
}

impl Scene for SinglePortalScene {
    fn name(&self) -> &'static str {
        if self.inverted {
            "Inversed Ghost collision"
        }
        else {
            "Ghost collision"
        }
    }

    fn create_world_state(&self) -> te::WorldState {
        let mut sim = te::WorldState::new(100., 100.);
        sim.push_portal(te::Portal {
            height: 15.,
            in_transform: Affine2::from_angle_translation(
                if !self.inverted { std::f32::consts::PI } else { 0. },
                Vec2::new(25., 70.),
            ),
            out_transform: Affine2::from_angle_translation(
                0.,
                Vec2::new(25., 30.),
            ),
            time_offset: 0.,
        });
        sim.push_ball(te::Ball {
            initial_pos: Vec2::new(if self.inverted { 24.5 } else { 25.5 }, 70.), 
            initial_velocity: Vec2::new(0., 0.),
            radius: 3.,
            ..Default::default()
        });
        sim.push_ball(te::Ball {
            initial_pos: Vec2::new(29., 3.),
            initial_velocity: Vec2::new(0., 30.),
            radius: 3.,
            ..Default::default()
        });
        sim
    }
}

pub struct InfiniteMovement;

impl Scene for InfiniteMovement {
    fn name(&self) -> &'static str {
        "Infinite movement"
    }

    fn create_world_state(&self) -> time_engine::WorldState {
        let mut sim = te::WorldState::new(50., 50.);
        sim.push_portal(te::Portal {
            height: 15.,
            in_transform: Affine2::from_angle_translation(
                0.,
                Vec2::new(40., 25.),
            ),
            out_transform: Affine2::from_angle_translation(
                0.,
                Vec2::new(10., 25.),
            ),
            time_offset: 0.,
        });

        sim.push_ball(te::Ball {
            initial_pos: Vec2::new(25., 25.),
            initial_velocity: Vec2::new(30., 0.),
            radius: 3.,
            ..Default::default()
        });
        sim
    }
}

pub struct BasicTwoBallScene;

impl Scene for BasicTwoBallScene {
    fn name(&self) -> &'static str {
        "Basic two ball"
    }

    fn create_world_state(&self) -> te::WorldState {
        let mut sim = te::WorldState::new(50., 50.);
        sim.push_ball(te::Ball {
            initial_pos: Vec2::new(40., 25.),
            initial_velocity: Vec2::new(-15., -0.),
            radius: 1.5,
            ..Default::default()
        });
        sim.push_ball(te::Ball {
            initial_pos: Vec2::new(10., 25.),
            initial_velocity: Vec2::new(15., 0.1),
            radius: 3.,
            ..Default::default()
        });
        sim
    }
}

pub struct BasicBouncingScene {
    pub seed: u64,
    pub name: &'static str,
    pub ball_count: usize,
    pub width: f32,
    pub height: f32,
}

impl Default for BasicBouncingScene {
    fn default() -> Self {
        Self {
            seed: 0,
            name: "Basic Bouncing Scene",
            ball_count: 5,
            width: 100.,
            height: 100.,
        }
    }
}

impl Scene for BasicBouncingScene {
    fn name(&self) -> &'static str {
        self.name
    }

    fn create_world_state(&self) -> te::WorldState {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(self.seed);

        let mut sim = te::WorldState::new(self.width, self.height);

        let mut max_tries = self.ball_count * 10;

        'ball_spawn: while sim.balls().len() < self.ball_count && max_tries > 0 {
            max_tries -= 1;

            let rad = rng.random_range(1. .. 3.);
            let rad2 = rad*2.;
            if self.width <= rad2*2. || self.height <= rad2*2. {
                continue;
            }
            let pos = Vec2::new(rng.random_range(rad2..(self.width - rad2)), rng.random_range(rad2..(self.height - rad2)));
            let vel = Vec2::new(rng.random_range(-1. ..1.), rng.random_range(-1. ..1.))
                .normalize() * rng.random_range(3. .. 50.);

            for ball in sim.balls() {
                let dist = pos.distance_squared(ball.initial_pos);
                if dist < (ball.radius+rad).powi(2) {
                    continue 'ball_spawn;
                }
            }

            sim.push_ball(te::Ball {
                initial_pos: pos,
                initial_velocity: vel,
                radius: rad,
                ..Default::default()
            });
        }
        sim
    }
}

pub fn get_all_scenes() -> Vec<Arc<dyn Scene>> {
    vec![
        Arc::new(PolchinskiParadox { enable_time_travel: true }),
        Arc::new(PolchinskiParadox { enable_time_travel: false }),
        Arc::new(CollisionBehindPortal),
        Arc::new(SinglePortalScene { inverted: false }),
        Arc::new(SinglePortalScene { inverted: true }),
        Arc::new(InfiniteMovement),
        Arc::new(BasicTwoBallScene),
        Arc::new(BasicBouncingScene {
            seed: 4444,
            name: "Basic Bouncing 1",
            ball_count: 5,
            ..Default::default()
        }),
        Arc::new(BasicBouncingScene {
            seed: 4445,
            name: "Basic Bouncing 2",
            ball_count: 10,
            ..Default::default()
        }),
    ]
}
