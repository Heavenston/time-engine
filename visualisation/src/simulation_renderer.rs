use macroquad::prelude::*;
use i_overlay::{i_shape::base::data::Shapes, mesh::outline::offset::OutlineOffset};
use time_engine::{self as te, clip_shapes_on_portal, Simulator};
use crate::draw_polygon::draw_shapes;

pub struct RenderSimulationArgs<'a> {
    pub world_state: &'a te::WorldState,
    pub enable_debug_rendering: bool,
    pub time: f32,
    pub simulator: &'a Simulator<'a>,
}

pub fn render_simulation(
    RenderSimulationArgs {
        world_state,
        enable_debug_rendering,
        time,
        simulator,
    }: RenderSimulationArgs<'_>
) {
    clear_background(BLACK);

    // Draw the simulation bounding box
    draw_rectangle_lines(-2.5, -2.5, world_state.width() + 5., world_state.height() + 5., 5., WHITE);

    // Draw spheres
    for snap in simulator.time_query(Some(time)).into_iter()
        .map(|link| simulator.snapshots()[link].extrapolate(time))
        .flat_map(move |snap| std::iter::once(snap).chain(snap.get_ghosts()))
    {
        let te::SimSnapshot {
            pos, vel, radius: rad,
            portal_traversals,
            ..
        } = snap;

        let sphere_shapes: Shapes<Vec2> = vec![vec![te::circle_polygon(pos, rad, 30)]];
        let sphere_shapes = portal_traversals.iter()
            .fold(sphere_shapes, |sphere_shapes, traversal|
                clip_shapes_on_portal(sphere_shapes, traversal.portal_in.transform, traversal.direction)
            );

        if enable_debug_rendering {
            // draw a velocity line
            draw_line(pos.x, pos.y, pos.x + vel.x, pos.y + vel.y, 0.5, ORANGE.with_alpha(0.25));

            if !portal_traversals.is_empty() {
                let outline = sphere_shapes.outline(&i_overlay::mesh::style::OutlineStyle {
                    outer_offset: 0.5,
                    inner_offset: 0.,
                    join: i_overlay::mesh::style::LineJoin::Bevel,
                });
                draw_shapes(Vec2::ZERO, &outline, RED);
            }
        }

        draw_shapes(Vec2::ZERO, &sphere_shapes, WHITE);
    }

    // Draw portals
    for (height, transform) in world_state.portals().iter()
        .flat_map(|portal| [
            (portal.height, portal.in_transform),
            (portal.height, portal.out_transform),
        ])
    {
        let h2 = height / 2.;
        let middle = transform.transform_point2(Vec2::new(0., 0.));
        let start = transform.transform_point2(Vec2::new(0., -h2));
        let end = transform.transform_point2(Vec2::new(0., h2));

        let mut portal_color = GREEN;
        if enable_debug_rendering {
            let normal = transform.transform_vector2(Vec2::new(-1., 0.)) * 10.;
            // draw the normal arrow
            draw_line(middle.x, middle.y, middle.x + normal.x, middle.y + normal.y, 0.5, GREEN.with_alpha(0.25));

            portal_color = portal_color.with_alpha(0.8);
        }
        // draw the actual portal surface
        draw_line(start.x, start.y, end.x, end.y, 1., portal_color);
    }

    // Draw the world's static body collision
    if enable_debug_rendering {
        draw_shapes(Vec2::ZERO, &world_state.get_static_body_collision(), ORANGE);
    }
}
