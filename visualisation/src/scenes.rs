use time_engine as te;
use macroquad::prelude::{Affine2, Vec2};

pub trait Scene {
    fn name(&self) -> &'static str;
    fn create_world_state(&self) -> te::WorldState;
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

pub struct BasicBouncingScene;

impl Scene for BasicBouncingScene {
    fn name(&self) -> &'static str {
        "Basic Bouncing"
    }

    fn create_world_state(&self) -> te::WorldState {
        let mut sim = te::WorldState::new(100., 100.);
        sim.push_sphere(te::Sphere {
            initial_pos: Vec2::new(20., 20.),
            initial_velocity: Vec2::new(35., 25.),
            radius: 2.,
            ..Default::default()
        });
        sim.push_sphere(te::Sphere {
            initial_pos: Vec2::new(80., 80.),
            initial_velocity: Vec2::new(-30., -20.),
            radius: 2.5,
            ..Default::default()
        });
        sim.push_sphere(te::Sphere {
            initial_pos: Vec2::new(50., 10.),
            initial_velocity: Vec2::new(0., 40.),
            radius: 1.5,
            ..Default::default()
        });
        sim.push_sphere(te::Sphere {
            initial_pos: Vec2::new(50., 50.),
            initial_velocity: Vec2::new(30., 40.),
            radius: 3.,
            ..Default::default()
        });
        sim.push_sphere(te::Sphere {
            initial_pos: Vec2::new(10., 50.),
            initial_velocity: Vec2::new(30., -40.),
            radius: 2.25,
            ..Default::default()
        });
        sim
    }
}

pub fn get_all_scenes() -> Vec<Box<dyn Scene>> {
    vec![
        Box::new(SinglePortalScene),
        Box::new(BasicBouncingScene),
    ]
}
