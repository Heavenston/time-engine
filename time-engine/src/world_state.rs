use crate::*;

use glam::f32::{ Vec2, Affine2 };
use i_overlay::i_shape::base::data::Shapes;

const PORTALS_WALLS_WIDTH: f32 = 0.1;
const PORTALS_WALLS_HEIGHT: f32 = 0.5;

#[derive(Debug, Clone, Copy)]
pub struct Sphere {
    pub initial_time: f32,
    pub initial_pos: Vec2,
    pub initial_velocity: Vec2,
    pub radius: f32,
}

impl Default for Sphere {
    fn default() -> Self {
        Self {
            initial_time: 0.,
            initial_pos: Vec2::ZERO,
            initial_velocity: Vec2::ZERO,
            radius: 1.,
        }
    }
}

#[derive(Default, Debug, Clone, Copy)]
pub struct Portal {
    pub height: f32,
    pub in_transform: Affine2,
    pub out_transform: Affine2,
    pub time_offset: f32,
}

impl Portal {
    pub fn swap(self) -> Self {
        Self {
            in_transform: self.out_transform,
            out_transform: self.in_transform,
            ..self
        }
    }
}

#[derive(Debug)]
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

    pub fn get_static_body_collision(&self) -> Shapes<Vec2> {
        // Vector used to add width to the walls
        let wall_width_splat = Vec2::splat(3.);
        let wall_width_angle = wall_width_splat * Vec2::new(-1., 1.);
        let mut shapes: Shapes<Vec2> = vec![
            vec![
                vec![
                    Vec2::new(0., 0.) - wall_width_splat,
                    Vec2::new(0., self.height) + wall_width_angle,
                    Vec2::new(self.width, self.height) + wall_width_splat,
                    Vec2::new(self.width, 0.) - wall_width_angle,
                ],
                vec![
                    Vec2::new(0., 0.),
                    Vec2::new(self.width, 0.),
                    Vec2::new(self.width, self.height),
                    Vec2::new(0., self.height),
                ],
            ]
        ];

        for (height, transform) in self.portals.iter()
            .flat_map(|portal| [(portal.height, portal.in_transform), (portal.height, portal.out_transform)])
        {
            let h2 = height / 2.;
            shapes.push(vec![vec![
                transform.transform_point2(Vec2::new(PORTALS_WALLS_HEIGHT, -h2)),
                transform.transform_point2(Vec2::new(-PORTALS_WALLS_HEIGHT, -h2)),
                transform.transform_point2(Vec2::new(-PORTALS_WALLS_HEIGHT, -h2 - PORTALS_WALLS_WIDTH)),
                transform.transform_point2(Vec2::new(PORTALS_WALLS_HEIGHT, -h2 - PORTALS_WALLS_WIDTH)),
            ]]);
            shapes.push(vec![vec![
                transform.transform_point2(Vec2::new(-PORTALS_WALLS_HEIGHT, h2)),
                transform.transform_point2(Vec2::new(PORTALS_WALLS_HEIGHT, h2)),
                transform.transform_point2(Vec2::new(PORTALS_WALLS_HEIGHT, h2 + PORTALS_WALLS_WIDTH)),
                transform.transform_point2(Vec2::new(-PORTALS_WALLS_HEIGHT, h2 + PORTALS_WALLS_WIDTH)),
            ]]);
        }

        shapes
    }

    pub fn create_simulator(&'_ self, max_time: f32) -> Simulator<'_> {
        Simulator::new(self, max_time)
    }

    pub fn simulate(&self, max_time: f32) {
        Simulator::new(self, max_time).run();
    }
}
