use rand::{Rng, SeedableRng};
use time_engine as te;
use macroquad::prelude::{Affine2, Vec2};

pub trait Scene {
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
        sim.push_sphere(te::Sphere {
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
        sim.push_sphere(te::Sphere {
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

pub struct SinglePortalScene;

impl Scene for SinglePortalScene {
    fn name(&self) -> &'static str {
        "Ghost collision"
    }

    fn create_world_state(&self) -> te::WorldState {
        let mut sim = te::WorldState::new(100., 100.);
        sim.push_portal(te::Portal {
            height: 15.,
            in_transform: Affine2::from_angle_translation(
                std::f32::consts::PI,
                Vec2::new(25., 70.),
            ),
            out_transform: Affine2::from_angle_translation(
                0.,
                Vec2::new(25., 30.),
            ),
            time_offset: 0.,
        });
        sim.push_sphere(te::Sphere {
            initial_pos: Vec2::new(25.5, 70.), 
            initial_velocity: Vec2::new(0., 0.),
            radius: 3.,
            ..Default::default()
        });
        sim.push_sphere(te::Sphere {
            initial_pos: Vec2::new(29., 3.),
            initial_velocity: Vec2::new(0., 30.),
            radius: 3.,
            ..Default::default()
        });
        sim
    }
}

pub struct BasicBouncingScene {
    seed: u64,
    name: &'static str
}

impl Scene for BasicBouncingScene {
    fn name(&self) -> &'static str {
        self.name
    }

    fn create_world_state(&self) -> te::WorldState {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(self.seed);

        let mut sim = te::WorldState::new(100., 100.);
        for _ in 0..10 {
            let pos = Vec2::new(rng.random_range(0. ..100.), rng.random_range(0. ..100.));
            let vel = Vec2::new(rng.random_range(-1. ..1.), rng.random_range(-1. ..1.))
                .normalize() * rng.random_range(3. .. 50.);
            let rad = rng.random_range(1. .. 3.);

            sim.push_sphere(te::Sphere {
                initial_pos: pos,
                initial_velocity: vel,
                radius: rad,
                ..Default::default()
            });
        }
        sim
    }
}

pub fn get_all_scenes() -> Vec<Box<dyn Scene>> {
    vec![
        Box::new(PolchinskiParadox { enable_time_travel: true }),
        Box::new(PolchinskiParadox { enable_time_travel: false }),
        Box::new(CollisionBehindPortal),
        Box::new(SinglePortalScene),
        Box::new(BasicBouncingScene {
            seed: 4444,
            name: "Basic Bouncing 1",
        }),
        Box::new(BasicBouncingScene {
            seed: 4445,
            name: "Basic Bouncing 2",
        }),
    ]
}
