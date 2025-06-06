use glam::Vec2;
use i_overlay::{core::{fill_rule::FillRule, overlay_rule::OverlayRule}, float::single::SingleFloatOverlay, i_shape::base::data::{Shape, Shapes}};
use i_triangle::float::triangulatable::Triangulatable;
use itertools::Itertools;
use parry2d::shape;
use nalgebra as na;

use crate::{Portal, PortalDirection};

/// Size of the polygon created for clipping behind a portal
/// should aproximate a half space so be pretty big, but not too much as to
/// not endure precision penalty
const PORTAL_CLIPPING_EXTENT: Vec2 = Vec2::new(100., 100.);

pub fn circle_polygon(
    center: Vec2,
    radius: f32,
    num_segments: u32,
) -> Vec<Vec2> {
    let mut points = Vec::with_capacity(num_segments as usize);
    let angle_step = 2.0 * std::f32::consts::PI / num_segments as f32;
    for i in 0..num_segments {
        let angle = angle_step * i as f32;
        points.push(Vec2::new(
            center.x + radius * angle.cos(),
            center.y + radius * angle.sin(),
        ));
    }
    points
}

pub fn clip_shapes_on_portal(
    shapes: Shapes<Vec2>,
    portal: &Portal,
    direction: PortalDirection,
) -> Shapes<Vec2> {
    let h2 = PORTAL_CLIPPING_EXTENT.y / 2.;
    let start = portal.initial_transform.transform_point2(Vec2::new(0., -h2));
    let end = portal.initial_transform.transform_point2(Vec2::new(0., h2));

    let normal = portal.initial_transform.transform_vector2(Vec2::new(-1., 0.)) * PORTAL_CLIPPING_EXTENT.x;
    let normal = match direction {
        PortalDirection::Front => normal,
        PortalDirection::Back => -normal,
    };
    let clip_polygon = [
        start,
        end,
        end - normal,
        start - normal,
    ];

    shapes.overlay(&clip_polygon, OverlayRule::Difference, FillRule::EvenOdd)
}

pub fn i_shape_to_parry_shape(shapes: Shapes<Vec2>) -> impl shape::Shape {
    // Probably very ineficient to tringulate the mesh instead of
    // using something like convex hull but this is way easier and i know
    // it will work
    let triangulation = shapes.triangulate()
        .to_triangulation::<u32>();

    shape::TriMesh::new(
        triangulation.points.into_iter()
            .map(|point| na::point![point.x, point.y])
            .collect(),
        triangulation.indices.into_iter()
            .tuples::<(_, _, _)>()
            .map(|(a, b, c)| [a, b, c])
            .collect(),
    ).unwrap()
}
