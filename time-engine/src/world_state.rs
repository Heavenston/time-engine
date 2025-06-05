use crate::{ default, Simulation, SimulationResult };

use glam::f32::{ Vec2, Affine2 };

pub struct Sphere {
    pub initial_time: f32,
    pub max_age: f32,
    pub initial_pos: Vec2,
    pub initial_velocity: Vec2,
    pub radius: f32,
}

impl Default for Sphere {
    fn default() -> Self {
        Self {
            initial_time: 0.,
            max_age: f32::INFINITY,
            initial_pos: Vec2::ZERO,
            initial_velocity: Vec2::ZERO,
            radius: 1.,
        }
    }
}

pub struct Portal {
    pub height: f32,
    pub initial_transform: Affine2,
    /// Index of the portal this portals links to
    pub link_to: usize,
}

pub struct WorldState {
    pub(crate) width: f32,
    pub(crate) height: f32,
    pub(crate) spheres: Vec<Sphere>,
    pub(crate) portals: Vec<Portal>,
}

impl WorldState {
    pub fn new(width: f32, height: f32) -> Self {
        Self {
            width,
            height,
            spheres: default(),
            portals: default(),
        }
    }

    pub fn width(&self) -> f32 {
        self.width
    }

    pub fn height(&self) -> f32 {
        self.height
    }

    pub fn spheres(&self) -> &[Sphere] {
        &self.spheres
    }

    pub fn portals(&self) -> &[Portal] {
        &self.portals
    }

    pub fn push_sphere(&mut self, sphere: Sphere) {
        self.spheres.push(sphere);
    }

    pub fn push_portal(&mut self, portal: Portal) {
        self.portals.push(portal);
    }

    pub fn simulate(&self, end_time: f32) -> SimulationResult {
        Simulation::new(self, end_time).run()
    }
}
