use glam::Vec2;
use i_overlay::{core::{fill_rule::FillRule, overlay_rule::OverlayRule}, float::single::SingleFloatOverlay, i_shape::base::data::Shapes};

use crate::{Portal, PortalDirection};

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
    let start = portal.initial_transform.transform_point2(Vec2::new(0., -9999.));
    let end = portal.initial_transform.transform_point2(Vec2::new(0., 9999.));

    let normal = portal.initial_transform.transform_vector2(Vec2::new(-1., 0.)) * 10.;
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
