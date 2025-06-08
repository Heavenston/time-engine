use std::{any::Any, collections::HashMap};

use macroquad::prelude::*;
use i_overlay::{i_shape::base::data::Shapes, mesh::{outline::offset::OutlineOffset, style::OutlineStyle}};
use time_engine as te;
use crate::draw_polygon::draw_shapes;
use itertools::Itertools;
use ordered_float::OrderedFloat as OF;

pub struct RenderSimulationArgs<'a> {
    pub world_state: &'a te::WorldState,
    pub simulation_result: &'a te::SimulationResult,
    pub sim_t: f32,
    pub enable_debug_rendering: bool,
    pub selected_timeline: Option<te::TimelineId>,
}

const TIMELINES_COLORS: [Color; 10] = [
    Color::from_hex(0xFF4500), // Orange Red
    Color::from_hex(0x00CED1), // Dark Turquoise
    Color::from_hex(0xFFD700), // Gold
    Color::from_hex(0x32CD32), // Lime Green
    Color::from_hex(0x8A2BE2), // Blue Violet
    Color::from_hex(0xFF69B4), // Hot Pink
    Color::from_hex(0x00FF7F), // Spring Green
    Color::from_hex(0x1E90FF), // Dodger Blue
    Color::from_hex(0xFF6347), // Tomato
    Color::from_hex(0xFF1493), // Deep Pink
];

const SPHERES_COLORS: [Color; 11] = [
    WHITE,
    Color::from_hex(0xFFB347), // Pastel Orange
    Color::from_hex(0xAEC6CF), // Pastel Blue
    Color::from_hex(0xFDFD96), // Pastel Yellow
    Color::from_hex(0x77DD77), // Pastel Green
    Color::from_hex(0xB19CD9), // Pastel Violet
    Color::from_hex(0xFFD1DC), // Pastel Pink
    Color::from_hex(0xA7F4B4), // Pastel Mint
    Color::from_hex(0x87CEEB), // Sky Blue
    Color::from_hex(0xFF9999), // Pastel Red
    Color::from_hex(0xF4C2C2), // Soft Pink
];

pub fn render_simulation(
    RenderSimulationArgs {
        world_state,
        simulation_result,
        sim_t,
        enable_debug_rendering,
        selected_timeline,
    }: RenderSimulationArgs<'_>
) {
    // Clear background
    clear_background(BLACK);

    // Draw the simulation bounding box
    draw_rectangle_lines(-2.5, -2.5, world_state.width() + 5., world_state.height() + 5., 5., WHITE);

    // assigns for each timeline a color
    let timeline_colors: HashMap<te::TimelineId, Color> = simulation_result.spheres.iter()
        .flat_map(|sphere| &sphere.snapshots)
        .sorted_by_key(|snap| OF(snap.t))
        .map(|snap| snap.tid)
        .unique()
        .enumerate()
        .map(|(idx, tid)| (tid, TIMELINES_COLORS[idx % TIMELINES_COLORS.len()]))
        .collect();

    let timelines_to_render = if enable_debug_rendering || selected_timeline.is_none() {
        timeline_colors.keys().copied().sorted().collect_vec()
    }
    else {
        vec![selected_timeline.unwrap()]
    };

    // Draw spheres
    for (sid, tid, sphere_idx, sphere) in simulation_result.spheres.iter()
        .enumerate()
        .flat_map(|(sphere_idx, sphere)| sphere.sids().map(move |sid| (sid, sphere_idx, sphere)))
        .flat_map(|(sid, sphere_idx, sphere)| timelines_to_render.iter().map(move |&tid| (sid, tid, sphere_idx, sphere)))
    {
        let Some(snap) = sphere.get_snapshot(&simulation_result.multiverse, sim_t, sid, tid)
        else { continue };
        if enable_debug_rendering && snap.tid != tid {
            continue;
        }

        let mut sphere_shapes: Shapes<Vec2> = vec![vec![te::circle_polygon(snap.pos, sphere.radius, 30)]];
        let mut sphere_ghost_shapes: Shapes<Vec2> = vec![];
        for traversal in snap.portal_traversals {
            let portal_in = &world_state.portals()[traversal.portal_in_idx];
            let portal_ou = &world_state.portals()[traversal.portal_out_idx];

            // Make a new ball mesh at the output portal
            let in_relative_pos = portal_in.initial_transform.inverse().transform_point2(snap.pos);
            let out_pos = portal_ou.initial_transform.transform_point2(in_relative_pos);
            let new_ghost_shape = vec![te::circle_polygon(out_pos, sphere.radius, 30)];

            sphere_shapes = te::clip_shapes_on_portal(sphere_shapes, portal_in, traversal.direction);
            sphere_ghost_shapes.extend(te::clip_shapes_on_portal(vec![new_ghost_shape], portal_ou, traversal.direction.swap()));
        }

        // TEMP: Ghosts when traveling in time are invalid
        // sphere_shapes.extend(sphere_ghost_shapes);

        // Renders an outline on the sphere showing its portal traversal state
        if enable_debug_rendering && !snap.portal_traversals.is_empty() {
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

        if enable_debug_rendering {
            let color = timeline_colors.get(&tid).copied()
                .expect("All timelines have a color");
            let outline = sphere_shapes.outline(&OutlineStyle {
                outer_offset: 0.5,
                inner_offset: 0.5,
                join: i_overlay::mesh::style::LineJoin::Round(1.),
            });
            draw_shapes(Vec2::ZERO, &outline, color);
        }

        // TEMP?: Better color each individual ball instead of timelines
        let color = SPHERES_COLORS[sphere_idx % SPHERES_COLORS.len()];

        // Drawing the actual sphere
        draw_shapes(Vec2::ZERO, &sphere_shapes, color);

        if enable_debug_rendering {
            // draw a velocity line
            draw_line(snap.pos.x, snap.pos.y, snap.pos.x + snap.vel.x, snap.pos.y + snap.vel.y, 0.5, ORANGE.with_alpha(0.25));

            // Draw age text on sphere
            let text = &format!("{sphere_idx}.{sid}");
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
    }

    // Draw portals
    for portal in world_state.portals() {
        let h2 = portal.height / 2.;
        let middle = portal.initial_transform.transform_point2(Vec2::new(0., 0.));
        let start = portal.initial_transform.transform_point2(Vec2::new(0., -h2));
        let end = portal.initial_transform.transform_point2(Vec2::new(0., h2));
        if enable_debug_rendering {
            let normal = portal.initial_transform.transform_vector2(Vec2::new(-1., 0.)) * 10.;
            // draw the normal arrow
            draw_line(middle.x, middle.y, middle.x + normal.x, middle.y + normal.y, 0.5, GREEN.with_alpha(0.25));
        }
        // draw the actual portal surface
        draw_line(start.x, start.y, end.x, end.y, 1., GREEN);
    }

    // Draw the world's static body collision
    if enable_debug_rendering {
        draw_shapes(Vec2::ZERO, &world_state.get_static_body_collision(), ORANGE);
    }
}
