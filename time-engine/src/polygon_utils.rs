use crate::*;

use glam::{Affine2, Vec2};
use i_overlay::{core::{fill_rule::FillRule, overlay_rule::OverlayRule}, float::single::SingleFloatOverlay, i_shape::base::data::Shapes};
use i_triangle::float::triangulatable::Triangulatable;
use itertools::Itertools;
use parry2d::shape;
use nalgebra as na;

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
    portal_transform: Affine2,
    direction: PortalDirection,
) -> Shapes<Vec2> {
    let h2 = PORTAL_CLIPPING_EXTENT.y / 2.;
    let start = portal_transform.transform_point2(Vec2::new(0., -h2));
    let end = portal_transform.transform_point2(Vec2::new(0., h2));

    let normal = portal_transform.transform_vector2(Vec2::new(-1., 0.)) * PORTAL_CLIPPING_EXTENT.x;
    let normal = if direction.is_back() { -normal } else { normal };

    let clip_polygon = [
        start,
        end,
        end - normal,
        start - normal,
    ];

    shapes.overlay(&clip_polygon, OverlayRule::Difference, FillRule::EvenOdd)
}

pub fn i_shape_to_parry_shape(shapes: Shapes<Vec2>) -> Option<impl shape::Shape> {
    let convexes = shapes.triangulate()
        .into_delaunay()
        .to_convex_polygons();

    if convexes.is_empty() {
        return None;
    }

    Some(shape::Compound::new(convexes.into_iter().map(|convex| {
        let shape = shape::ConvexPolygon::from_convex_polyline_unmodified(convex.iter()
            .map(|point| point.to_na())
            .collect_vec()
        ).expect("not empty");
        (na::Isometry2::identity(), shape::SharedShape::new(shape))
    }).collect()))
}
