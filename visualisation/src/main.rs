mod draw_polygon;
mod simulation_renderer;
use std::ops::ControlFlow;

use simulation_renderer::{render_simulation, RenderSimulationArgs};

use time_engine as te;
use macroquad::{ prelude::{scene::clear, *}, ui::{ self, root_ui } };

const CAMERA_ZOOM_SPEED: f32 = 1.25;

fn window_conf() -> Conf {
    Conf {
        window_title: "Window name".to_owned(),
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let sim = {
        let mut sim = te::WorldState::new(100., 100.);
        // sim.push_portal(te::Portal {
        //     height: 20.,
        //     in_transform: Affine2::from_angle_translation(
        //         std::f32::consts::PI,
        //         Vec2::new(85., 50.),
        //     ),
        //     out_transform: Affine2::from_angle_translation(
        //         std::f32::consts::FRAC_PI_2,
        //         Vec2::new(50., 85.),
        //     ),
        //     time_offset: -2.3,
        // });
        sim.push_sphere(te::Sphere {
            initial_pos: glam::Vec2::new(10., 50.),
            initial_velocity: glam::Vec2::new(30., 0.),
            radius: 3.,
            ..Default::default()
        });
        sim.push_sphere(te::Sphere {
            initial_pos: glam::Vec2::new(50., 50.),
            initial_velocity: glam::Vec2::new(-30., 0.),
            radius: 3.,
            ..Default::default()
        });
        sim.push_sphere(te::Sphere {
            initial_pos: glam::Vec2::new(50., 70.), 
            initial_velocity: glam::Vec2::new(0., 30.),
            radius: 3.,
            ..Default::default()
        });
        sim
    };

    let mut simulator = sim.create_simulator(60.);
    let mut step_count = 0;

    let mut cam_offset = Vec2::ZERO;
    let mut zoom = 1.;
    let mut enable_debug_rendering = false;
    let mut is_paused = false;
    let mut is_finished_simulation = false;
    let mut time = 0.;

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
        }

        if is_key_pressed(KeyCode::D) {
            enable_debug_rendering = !enable_debug_rendering;
        }

        if is_key_pressed(KeyCode::Space) {
            is_paused = !is_paused;
        }

        camera.target = cam_offset + cam_centering_offset;
        camera.zoom = cam_centering_zoom * zoom;

        if is_mouse_button_down(MouseButton::Left) {
            cam_offset += mouse_delta_position() / camera.zoom;
        }

        camera.target = cam_offset + cam_centering_offset;

        {
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

        // Render simulation

        if !is_paused {
            time += get_frame_time();
        }

        let (min_time, max_time) = simulator.minmax_time();

        if time < min_time || time > simulator.max_time() {
            time = min_time;
        }
        else if time > max_time {
            if is_finished_simulation {
                time = min_time;
            }
            else {
                step_count += 1;
                println!("Step #{step_count}");
                if let ControlFlow::Break(reason) = simulator.step() {
                    println!("Finished simulation: {reason:?}");
                    is_finished_simulation = true;
                    simulator.extrapolate_to(simulator.max_time());
                }
                else {
                    time = time.min(simulator.minmax_time().1);
                }
            }
        }
        
        render_simulation(RenderSimulationArgs {
            world_state: &sim,
            enable_debug_rendering,
            time,
            simulator: &simulator,
        });

        root_ui().label(None, &format!("fps: {}", get_fps()));
        root_ui().label(None, &format!("time: {time:.02}s/{:.02}s", simulator.max_time()));
        root_ui().label(None, &format!("steps: {step_count}"));

        // Draw timeline controls
        set_default_camera();

        next_frame().await;
    }
}
