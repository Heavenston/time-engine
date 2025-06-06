use macroquad::prelude::*;
use i_overlay::{i_shape::base::data::Shapes, mesh::{outline::offset::OutlineOffset, style::OutlineStyle}};
use time_engine as te;
use crate::draw_polygon::draw_shapes;

pub fn render_simulation(
    world_state: &te::WorldState,
    simulation_result: &te::SimulationResult,
    sim_t: f32,
) {
    // Clear background
    clear_background(BLACK);

    // Draw the simulation bounding box
    draw_rectangle_lines(-2.5, -2.5, world_state.width() + 5., world_state.height() + 5., 5., WHITE);

    // Draw spheres
    for sphere in &simulation_result.spheres {
        let Some(snap) = sphere.interpolate_snapshot(sim_t)
        else { continue };

        let mut sphere_shapes: Shapes<Vec2> = vec![vec![te::circle_polygon(snap.pos, sphere.radius, 30)]];
        for traversal in snap.portal_traversals {
            if traversal.end_t < sim_t {
                continue
            }

            let portal_in = &world_state.portals()[traversal.portal_in_idx];
            let portal_ou = &world_state.portals()[traversal.portal_out_idx];

            // Make a new ball mesh at the output portal
            let in_relative_pos = portal_in.initial_transform.inverse().transform_point2(snap.pos);
            let out_pos = portal_ou.initial_transform.transform_point2(in_relative_pos);
            sphere_shapes.push(vec![te::circle_polygon(out_pos, sphere.radius, 30)]);

            sphere_shapes = te::clip_shapes_on_portal(sphere_shapes, portal_in, traversal.direction);
            sphere_shapes = te::clip_shapes_on_portal(sphere_shapes, portal_ou, traversal.direction.swap());
        }

        if !snap.portal_traversals.is_empty() {
            let outline = sphere_shapes.outline(&OutlineStyle {
                outer_offset: 0.5,
                inner_offset: 0.5,
                join: i_overlay::mesh::style::LineJoin::Round(1.),
            });
            let color = if snap.portal_traversals[0].direction.is_front() {
                RED
            } else {
                BLUE
            };
            draw_shapes(Vec2::ZERO, &outline, color);
        }
        draw_shapes(Vec2::ZERO, &sphere_shapes, WHITE);
        
        // Draw age text on sphere
        let text = &format!("{:.01}", snap.age);
        let size = 32;
        let scale: f32 = 0.075;
        let text_size = measure_text(text, None, size, scale);
        draw_text_ex(text, snap.pos.x - text_size.width / 2., snap.pos.y - text_size.height / 2., TextParams {
            font_size: size,
            font_scale: -scale,
            font_scale_aspect: -1.,
            color: BLACK,
            ..Default::default()
        });
    }

    // Draw portals
    for portal in world_state.portals() {
        let h2 = portal.height / 2.;
        let middle = portal.initial_transform.transform_point2(Vec2::new(0., 0.));
        let start = portal.initial_transform.transform_point2(Vec2::new(0., -h2));
        let end = portal.initial_transform.transform_point2(Vec2::new(0., h2));
        let normal = portal.initial_transform.transform_vector2(Vec2::new(-1., 0.)) * 10.;
        draw_line(middle.x, middle.y, middle.x + normal.x, middle.y + normal.y, 0.5, GREEN.with_alpha(0.25));
        draw_line(start.x, start.y, end.x, end.y, 1., GREEN);
    }
}