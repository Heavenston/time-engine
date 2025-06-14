use macroquad::prelude::*;
use i_overlay::{core::{fill_rule::FillRule, overlay_rule::OverlayRule}, float::single::SingleFloatOverlay as _, i_shape::base::data::Shapes, mesh::outline::offset::OutlineOffset};
use time_engine::{self as te, clip_shapes_on_portal, Simulator};
use crate::draw_polygon::draw_shapes;

pub struct RenderSimulationArgs<'a> {
    pub world_state: &'a te::WorldState,
    pub enable_debug_rendering: bool,
    pub time: f32,
    pub simulator: &'a Simulator<'a>,
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

    // Draw spheres
    for (is_generated_ghost, snap) in simulator.time_query(time).into_iter()
        .map(|(range, link)| (range, simulator.snapshots()[link]))
        .flat_map(move |(range, snap)| std::iter::once((range.clone(), false, snap))
            .chain(snap.get_ghosts()
                .filter(move |ghost| time < ghost.expiration_time)
                .map(move |ghost| (range.offset(ghost.offset), true, ghost.snapshot))))
        .filter(|(range, _, _)| range.contains(time))
        .map(|(_, is_ghost, snap)| (is_ghost, snap.extrapolate(time)))
    {
        let is_ghost = snap.is_ghost();
        let te::SimSnapshot {
            pos, vel, radius: rad,
            portal_traversals,
            ..
        } = snap;

        let sphere_shapes: Shapes<Vec2> = vec![vec![te::circle_polygon(pos, rad, 30)]];
        let cliped_sphere_shapes = portal_traversals.iter()
            .fold(sphere_shapes.clone(), |sphere_shapes, traversal|
                clip_shapes_on_portal(sphere_shapes, traversal.portal_in.transform, traversal.direction)
            );

        if enable_debug_rendering {
            // draw a velocity line
            draw_line(pos.x, pos.y, pos.x + vel.x, pos.y + vel.y, 0.5, ORANGE.with_alpha(0.25));

            let mut previous_shape = sphere_shapes.clone();
            // rendering timeline color
            if true {
                let color = BALL_COLORS[snap.timeline_id.to_usize() % (BALL_COLORS.len() - 1) + 1];
                let outline = previous_shape.outline(&i_overlay::mesh::style::OutlineStyle {
                    outer_offset: 0.5,
                    inner_offset: 0.,
                    join: i_overlay::mesh::style::LineJoin::Bevel,
                });
                let outline2 = outline.overlay(&previous_shape, OverlayRule::Difference, FillRule::EvenOdd);
                previous_shape = outline;
                draw_shapes(Vec2::ZERO, &outline2, color.with_alpha(0.9));
            }
            if is_generated_ghost {
                let outline = previous_shape.outline(&i_overlay::mesh::style::OutlineStyle {
                    outer_offset: 0.5,
                    inner_offset: 0.,
                    join: i_overlay::mesh::style::LineJoin::Bevel,
                });
                let outline2 = outline.overlay(&previous_shape, OverlayRule::Difference, FillRule::EvenOdd);
                previous_shape = outline;
                draw_shapes(Vec2::ZERO, &outline2, ORANGE);
            }
            if !portal_traversals.is_empty() {
                let outline = previous_shape.outline(&i_overlay::mesh::style::OutlineStyle {
                    outer_offset: 0.5,
                    inner_offset: 0.,
                    join: i_overlay::mesh::style::LineJoin::Bevel,
                });
                let outline2 = outline.overlay(&previous_shape, OverlayRule::Difference, FillRule::EvenOdd);
                previous_shape = outline;
                draw_shapes(Vec2::ZERO, &outline2, RED);
            }
            if is_ghost {
                let outline = previous_shape.outline(&i_overlay::mesh::style::OutlineStyle {
                    outer_offset: 0.5,
                    inner_offset: 0.,
                    join: i_overlay::mesh::style::LineJoin::Bevel,
                });
                let outline2 = outline.overlay(&previous_shape, OverlayRule::Difference, FillRule::EvenOdd);
                previous_shape = outline;
                draw_shapes(Vec2::ZERO, &outline2, GREEN);
            }

            draw_shapes(Vec2::ZERO, &sphere_shapes, WHITE.with_alpha(0.5));

            let color = BALL_COLORS[snap.original_idx % BALL_COLORS.len()];
            draw_shapes(Vec2::ZERO, &cliped_sphere_shapes, color.with_alpha(0.5));
        }
        else {
            let color = BALL_COLORS[snap.original_idx % BALL_COLORS.len()];
            draw_shapes(Vec2::ZERO, &cliped_sphere_shapes, color);
        }
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
