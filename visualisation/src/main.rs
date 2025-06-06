mod draw_polygon;
use draw_polygon::*;
mod timeline_controls;
use timeline_controls::TimelineControls;

use i_overlay::{i_shape::base::data::Shapes, mesh::{outline::offset::OutlineOffset, style::OutlineStyle}};
use time_engine as te;
use macroquad::{ prelude::*, ui::{ self, root_ui } };

const CAMERA_ZOOM_SPEED: f32 = 1.25;

fn window_conf() -> Conf {
    Conf {
        window_title: "Window name".to_owned(),
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let sim_duration = 60f32;
    let sim = {
        let mut sim = te::WorldState::new(100., 100.);
        sim.push_portal(te::Portal {
            height: 20.,
            initial_transform: Affine2::from_angle_translation(
                std::f32::consts::FRAC_PI_2,
                Vec2::new(50., 98.),
            ),
            link_to: 1,
        });
        sim.push_portal(te::Portal {
            height: 20.,
            initial_transform: Affine2::from_angle_translation(
                std::f32::consts::PI,
                Vec2::new(70., 50.),
            ),
            link_to: 0,
        });
        sim.push_sphere(te::Sphere {
            initial_pos: glam::Vec2::new(50., 50.),
            initial_velocity: glam::Vec2::new(0., 30.),
            radius: 3.,
            ..Default::default()
        });
        // sim.push_sphere(te::Sphere {
        //     initial_pos: glam::Vec2::new(20., 6.),
        //     initial_velocity: glam::Vec2::new(30., 20.),
        //     radius: 3.,
        //     ..Default::default()
        // });
        // sim.push_sphere(te::Sphere {
        //     initial_pos: glam::Vec2::new(20., 20.),
        //     initial_velocity: glam::Vec2::new(10., 10.),
        //     radius: 3.,
        //     ..Default::default()
        // });
        sim
    };
    println!("Simulating...");
    let simulation_result = sim.simulate(sim_duration);
    println!("Finished simulation");

    let mut cam_offset = Vec2::ZERO;
    let mut zoom = 1.;
    let mut paused = true;
    let mut sim_t = 0.;
    let mut sim_speed = 1.;
    
    // Timeline controls
    let mut timeline_controls = TimelineControls::new();

    let mouse_pos = |camera: &Camera2D| {
        camera.screen_to_world(Vec2::new(mouse_position().0, mouse_position().1))
    };

    // Setup ui skin
    {
        let label_style = root_ui().style_builder()
            .font_size(32)
            .text_color(WHITE)
            .build();
        let skin = ui::Skin {
            label_style,
            ..root_ui().default_skin()
        };
        root_ui().push_skin(&skin);
    }

    loop {
        if !paused {
            sim_t += get_frame_time() * sim_speed;
        }
        if sim_t > sim_duration {
            sim_t = 0.;
        }

        // Setup camera
        let (cw, ch) = if screen_width() > screen_height() {
            ((screen_width() / screen_height()) * sim.height(), sim.height())
        } else {
            (sim.width(), (screen_height() / screen_width()) * sim.width())
        };
        let mut camera = Camera2D::from_display_rect(Rect {
            x: (sim.width() - cw) / 2.,
            y: (sim.height() - ch) / 2.,
            w: cw,
            h: ch,
        });
        let cam_centering_zoom = camera.zoom;
        let cam_centering_offset = camera.target;

        // Handle inputs
        if is_key_pressed(KeyCode::R) {
            cam_offset = Vec2::ZERO;
            zoom = 1.;
            sim_t = 0.;
        }

        if is_key_pressed(KeyCode::Space) {
            paused = !paused;
        }

        camera.target = cam_offset + cam_centering_offset;
        camera.zoom = cam_centering_zoom * zoom;

        // Handle timeline input
        let mouse_in_timeline = timeline_controls.handle_input(&mut sim_t, sim_duration, &mut paused, &mut sim_speed);

        if !mouse_in_timeline && is_mouse_button_down(MouseButton::Left) {
            cam_offset += mouse_delta_position() / camera.zoom;
        }

        camera.target = cam_offset + cam_centering_offset;

        if !mouse_in_timeline {
            let scroll = mouse_wheel().1;
            if scroll != 0. {
                let mouse_world_before = mouse_pos(&camera);
                
                zoom *= CAMERA_ZOOM_SPEED.powf(scroll);
                
                camera.zoom = cam_centering_zoom * zoom;
                
                let mouse_world_after = mouse_pos(&camera);
                cam_offset += mouse_world_before - mouse_world_after;
                camera.target = cam_offset + cam_centering_offset;
            }
        }

        camera.zoom = cam_centering_zoom * zoom;

        // Setup camera
        set_camera(&camera);

        // Drawing
        clear_background(BLACK);

        // Draw the simulation bounding box
        draw_rectangle_lines(-2.5, -2.5, sim.width() + 5., sim.height() + 5., 5., WHITE);

        for sphere in &simulation_result.spheres {
            let Some(snap) = sphere.interpolate_snapshot(sim_t)
            else { continue };

            let mut sphere_shapes: Shapes<Vec2> = vec![vec![te::circle_polygon(snap.pos, sphere.radius, 30)]];
            for traversal in snap.portal_traversals {
                if traversal.end_t < sim_t {
                    continue
                }

                let portal_in = &sim.portals()[traversal.portal_in_idx];
                let portal_ou = &sim.portals()[traversal.portal_out_idx];

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

        for portal in sim.portals() {
            let h2 = portal.height / 2.;
            let middle = portal.initial_transform.transform_point2(Vec2::new(0., 0.));
            let start = portal.initial_transform.transform_point2(Vec2::new(0., -h2));
            let end = portal.initial_transform.transform_point2(Vec2::new(0., h2));
            let normal = portal.initial_transform.transform_vector2(Vec2::new(-1., 0.)) * 10.;
            draw_line(middle.x, middle.y, middle.x + normal.x, middle.y + normal.y, 0.5, GREEN.with_alpha(0.25));
            draw_line(start.x, start.y, end.x, end.y, 1., GREEN);
        }

        root_ui().label(None, &format!("fps: {}", get_fps()));
        root_ui().label(None, &format!("time: {sim_t:.02}s/{sim_duration:.02}s"));
        if paused {
            root_ui().label(None, "PAUSED");
        }

        // Draw timeline controls
        set_default_camera();
        timeline_controls.draw(sim_t, sim_duration, paused, sim_speed);

        next_frame().await;
    }
}
