mod draw_polygon;
mod timeline_controls;
use timeline_controls::TimelineControls;
mod simulation_renderer;
use simulation_renderer::{render_simulation, RenderSimulationArgs};

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
    let sim = {
        let mut sim = te::WorldState::new(100., 100.);
        sim.push_portal(te::Portal {
            height: 20.,
            initial_transform: Affine2::from_angle_translation(
                std::f32::consts::PI,
                Vec2::new(85., 50.),
            ),
            link_to: 1,
            time_offset: 1.,
        });
        sim.push_portal(te::Portal {
            height: 20.,
            initial_transform: Affine2::from_angle_translation(
                std::f32::consts::FRAC_PI_2,
                Vec2::new(50., 85.),
            ),
            link_to: 0,
            time_offset: 0.,
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
    let simulation_result = sim.simulate(30f32);
    let sim_duration = simulation_result.max_t();
    println!("{simulation_result:#?}");
    println!("Finished simulation");

    let mut cam_offset = Vec2::ZERO;
    let mut zoom = 1.;
    let mut paused = true;
    let mut sim_t = 0.;
    let mut sim_speed = 1.;
    let mut enable_debug_rendering = false;
    
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

        if is_key_pressed(KeyCode::D) {
            enable_debug_rendering = !enable_debug_rendering;
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

        // Render simulation
        render_simulation(RenderSimulationArgs {
            world_state: &sim,
            simulation_result: &simulation_result,
            sim_t,
            enable_debug_rendering,
        });

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
