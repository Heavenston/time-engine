mod draw_polygon;
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
            in_transform: Affine2::from_angle_translation(
                std::f32::consts::PI,
                Vec2::new(90., 50.),
            ),
            out_transform: Affine2::from_angle_translation(
                std::f32::consts::FRAC_PI_2,
                Vec2::new(50., 90.),
            ),
            time_offset: 80./30.,
        });
        // sim.push_sphere(te::Sphere {
        //     initial_pos: glam::Vec2::new(10., 50.),
        //     initial_velocity: glam::Vec2::new(30., 0.),
        //     radius: 3.,
        //     ..Default::default()
        // });
        // sim.push_sphere(te::Sphere {
        //     initial_pos: glam::Vec2::new(50., 50.),
        //     initial_velocity: glam::Vec2::new(-30., 0.),
        //     radius: 3.,
        //     ..Default::default()
        // });
        sim.push_sphere(te::Sphere {
            initial_pos: glam::Vec2::new(50., 30.), 
            initial_velocity: glam::Vec2::new(0., 30.),
            radius: 3.,
            ..Default::default()
        });
        sim
    };

    let mut simulator = sim.create_simulator(5.);
    let mut step_count = 0;

    let mut cam_offset = Vec2::ZERO;
    let mut zoom = 1.;
    let mut enable_debug_rendering = false;
    let mut is_paused = false;
    let mut time = 0.;
    let mut speed = 1.;

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
        // Forward simulation
        let (min_time, max_time) = simulator.minmax_time();

        if !is_paused && time <= max_time {
            time += get_frame_time() * speed;
        }

        if time < min_time || time > simulator.max_time() {
            time = min_time;
        }
        else if time > max_time {
            if simulator.finished() {
                time = min_time;
            }
            else {
                step_count += 1;
                let _ = simulator.step();
                if simulator.finished() {
                    println!("{simulator:#?}");
                    simulator.extrapolate_to(simulator.max_time());
                }
            }
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

        let mut captured_pointer = false;
        let mut captured_keyboard = false;
        egui_macroquad::ui(|ctx| {
            let mut style = (*ctx.style()).clone();
            style.visuals.override_text_color = Some(egui::Color32::WHITE);
            // style.text_styles.get_mut(&egui::TextStyle::Body).expect("present").size = 16.;
            ctx.set_style(style);

            egui::Window::new("Informations")
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("FPS:");
                        ui.colored_label(egui::Color32::GRAY, format!("{}", get_fps()));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Total Simulation Steps:");
                        ui.colored_label(egui::Color32::GRAY, format!("{step_count}"));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Max time:");
                        ui.colored_label(egui::Color32::GRAY, format!("{max_time}"));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Finished simulating:");
                        if simulator.finished() {
                            ui.colored_label(egui::Color32::LIGHT_GREEN, "Yes");
                        }
                        else {
                            ui.colored_label(egui::Color32::LIGHT_RED, "No");
                        }
                    });
                });

            egui::Window::new("Controls")
                .show(ctx, |ui| {
                    let progress = max_time / simulator.max_time();
                    if progress < 1. {
                        ui.add(egui::ProgressBar::new(max_time / simulator.max_time()));
                    }

                    ui.horizontal(|ui| {
                        ui.add(egui::Slider::new(&mut time, 0. ..=simulator.max_time())
                            .suffix("s")
                        );
                        if ui.button(if is_paused { "Resume" } else { "Pause" }).clicked() {
                            is_paused = !is_paused;
                        }
                    });

                    ui.add(egui::Slider::new(&mut speed, 0.1 ..= 10.)
                        .prefix("x")
                        .logarithmic(true)
                        .clamping(egui::SliderClamping::Never)
                        .text("Speed"));
                });

            captured_pointer = ctx.wants_pointer_input();
            captured_keyboard = ctx.wants_keyboard_input();
        });

        if !captured_keyboard {
            if is_key_pressed(KeyCode::R) {
                cam_offset = Vec2::ZERO;
                zoom = 1.;
                time = 0.;
            }

            if is_key_pressed(KeyCode::D) {
                enable_debug_rendering = !enable_debug_rendering;
            }

            if is_key_pressed(KeyCode::Space) {
                is_paused = !is_paused;
            }
        }

        camera.target = cam_offset + cam_centering_offset;
        camera.zoom = cam_centering_zoom * zoom;

        if !captured_pointer {
            if is_mouse_button_down(MouseButton::Left) {
                cam_offset += mouse_delta_position() / camera.zoom;
            }
        }

        camera.target = cam_offset + cam_centering_offset;

        if !captured_pointer {
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
            enable_debug_rendering,
            time,
            simulator: &simulator,
        });

        // render ui
        egui_macroquad::draw();

        next_frame().await;
    }
}
