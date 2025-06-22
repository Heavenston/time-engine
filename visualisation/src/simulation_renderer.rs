use macroquad::prelude::*;
use i_overlay::{core::{fill_rule::FillRule, overlay_rule::OverlayRule}, float::single::SingleFloatOverlay as _, i_shape::base::data::Shapes, mesh::outline::offset::OutlineOffset};
use time_engine as te;
use crate::draw_polygon::draw_shapes;

pub struct RenderSimulationArgs<'a> {
    pub world_state: &'a te::WorldState,
    pub enable_debug_rendering: bool,
    pub time: f32,
    pub simulator: &'a te::Simulator,
}

const BALL_COLORS: [Color; 10] = [
    // 1) Pure white
    Color::new(1.0, 1.0, 1.0, 1.0),

    // Slightly more saturated “pastel” hues for contrast on black
    Color::new(1.0, 0.4, 0.4, 1.0), // red
    Color::new(1.0, 0.7, 0.4, 1.0), // orange
    Color::new(1.0, 1.0, 0.4, 1.0), // yellow
    Color::new(0.4, 1.0, 0.4, 1.0), // green
    Color::new(0.4, 1.0, 0.7, 1.0), // mint
    Color::new(0.4, 1.0, 1.0, 1.0), // cyan
    Color::new(0.4, 0.7, 1.0, 1.0), // sky blue
    Color::new(0.7, 0.4, 1.0, 1.0), // lavender
    Color::new(1.0, 0.4, 1.0, 1.0), // magenta
];

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

    // Draw balls
    for snap in simulator.time_query(time).into_iter()
        .filter_map(|(_, snap)| snap.extrapolate_to(time))
        .filter(|snap| snap.validity_time_range.start().is_none_or(|start| start <= time))
    {
        let is_ghost = false;
        let te::sg::Snapshot {
            pos, linvel,
            ..
        } = snap;
        let rad = world_state.balls()[snap.object_id].radius;

        let ball_shapes: Shapes<Vec2> = vec![vec![te::circle_polygon(pos, rad, 30)]];
        let cliped_ball_shapes = snap.portal_traversals.iter()
            // .filter(|traversal| traversal.duration.end >= time)
            .fold(ball_shapes.clone(), |ball_shapes, traversal| te::clip_shapes_on_portal(
                ball_shapes,
                simulator.half_portals()[traversal.half_portal_idx].transform,
                traversal.direction
            ))
        ;

        if enable_debug_rendering {
            // draw a velocity line
            draw_line(pos.x, pos.y, pos.x + linvel.x, pos.y + linvel.y, 0.5, ORANGE.with_alpha(0.25));

            let mut previous_shape = ball_shapes.clone();
            // rendering timeline color
            // if true {
            //     let color = BALL_COLORS[snap.timeline_id.to_usize() % (BALL_COLORS.len() - 1) + 1];
            //     let outline = previous_shape.outline(&i_overlay::mesh::style::OutlineStyle {
            //         outer_offset: 0.5,
            //         inner_offset: 0.,
            //         join: i_overlay::mesh::style::LineJoin::Bevel,
            //     });
            //     let outline2 = outline.overlay(&previous_shape, OverlayRule::Difference, FillRule::EvenOdd);
            //     previous_shape = outline;
            //     draw_shapes(Vec2::ZERO, &outline2, color.with_alpha(0.9));
            // }
            if !snap.portal_traversals.is_empty() {
                let is_real = snap.portal_traversals.iter().any(|traversal| traversal.duration.contains(&time));
                let color = if is_real { RED } else { PINK };

                let outline = previous_shape.outline(&i_overlay::mesh::style::OutlineStyle {
                    outer_offset: 0.5,
                    inner_offset: 0.,
                    join: i_overlay::mesh::style::LineJoin::Bevel,
                });
                let outline2 = outline.overlay(&previous_shape, OverlayRule::Difference, FillRule::EvenOdd);
                previous_shape = outline;
                draw_shapes(Vec2::ZERO, &outline2, color);
            }
            if is_ghost {
                let outline = previous_shape.outline(&i_overlay::mesh::style::OutlineStyle {
                    outer_offset: 0.5,
                    inner_offset: 0.,
                    join: i_overlay::mesh::style::LineJoin::Bevel,
                });
                let outline2 = outline.overlay(&previous_shape, OverlayRule::Difference, FillRule::EvenOdd);
                // previous_shape = outline;
                draw_shapes(Vec2::ZERO, &outline2, GREEN);
            }

            draw_shapes(Vec2::ZERO, &ball_shapes, WHITE.with_alpha(0.5));

            let color = BALL_COLORS[snap.object_id % BALL_COLORS.len()];
            draw_shapes(Vec2::ZERO, &cliped_ball_shapes, color.with_alpha(0.5));

            let text = format!("{}", snap.object_id);
            let params = TextParams {
                font: None,
                font_size: 16,
                font_scale: -0.2,
                font_scale_aspect: -1.,
                rotation: 0.,
                color: BLACK,
            };
            let size = measure_text(&text, None, params.font_size, params.font_scale);
            draw_text_ex(&text, pos.x + size.width / 2., pos.y + size.height / 2., params);
        }
        else {
            let color = BALL_COLORS[snap.object_id % BALL_COLORS.len()];
            draw_shapes(Vec2::ZERO, &cliped_ball_shapes, color);
        }
    }

    // Draw portals
    for half_portal in simulator.half_portals() {
        let h2 = half_portal.height / 2.;
        let middle = half_portal.transform.transform_point2(Vec2::new(0., 0.));
        let start = half_portal.transform.transform_point2(Vec2::new(0., -h2));
        let end = half_portal.transform.transform_point2(Vec2::new(0., h2));

        let mut portal_color = GREEN;
        if enable_debug_rendering {
            let normal = half_portal.transform.transform_vector2(Vec2::new(-1., 0.)) * 10.;
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
