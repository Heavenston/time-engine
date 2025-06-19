mod draw_polygon;
mod simulation_renderer;
mod scenes;

use itertools::Itertools;
use simulation_renderer::{render_simulation, RenderSimulationArgs};
use scenes::get_all_scenes;

use macroquad::{ prelude::*, ui::{ self, root_ui } };
use ordered_float::OrderedFloat as OF;

const CAMERA_ZOOM_SPEED: f32 = 1.25;

fn window_conf() -> Conf {
    Conf {
        window_title: "Window name".to_owned(),
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let scenes = get_all_scenes();
    let mut current_scene_index = 0;
    let mut sim = scenes[current_scene_index].create_world_state();

    let mut simulator = sim.create_simulator(60.);
    let _ = simulator.step();
    let mut step_count = 1;
    let mut scene_changed = false;

    let mut cam_offset = Vec2::ZERO;
    let mut zoom = 1.;
    let mut enable_debug_rendering = false;
    let mut is_paused = true;
    let mut time = 0.;
    let mut speed = 1.;
    let mut max_t_text = format!("{:.02}", simulator.max_time());
    let mut simulator_finished = false;

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
        let min_time = simulator.snapshots().nodes()
            .map(|(_, node)| node.snapshot.time)
            .min_by_key(|&t| OF(t))
            .unwrap_or(0.);
        let max_time = simulator.snapshots().nodes()
            .filter(|(_, node)| node.age_children.is_empty())
            .map(|(_, node)| node.snapshot.time)
            .max_by_key(|&t| OF(t))
            .unwrap_or(0.);

        if !is_paused && time <= max_time {
            time += get_frame_time() * speed;
        }

        if time < min_time {
            time = min_time;
        }
        else if time > simulator.max_time() {
            time = simulator.max_time();
            is_paused = true;
        }
        else if time > max_time {
            if simulator_finished {
                time = max_time;
                is_paused = true;
            }
            else {
                step_count += 1;
                if simulator.step().is_break() {
                    simulator_finished = true;
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
                        ui.label("Number of timelines:");
                        let active = simulator.time_query(time).into_iter()
                            .map(|(_, link)| simulator.snapshots()[link].timeline_id)
                            .unique()
                            .count();
                        ui.colored_label(egui::Color32::GRAY, format!("{active}/{}", simulator.multiverse().len()));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Enabled debug rendering (d):");
                        if enable_debug_rendering {
                            ui.colored_label(egui::Color32::LIGHT_GREEN, "Yes");
                        }
                        else {
                            ui.colored_label(egui::Color32::LIGHT_RED, "No");
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Finished simulating:");
                        if simulator_finished {
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
                        if ui.button("Full Reset").clicked() {
                            simulator = sim.create_simulator(simulator.max_time());
                            let _ = simulator.step();
                            simulator_finished = false;
                            step_count = 1;
                        }
                        if ui.button("Full Simulate").clicked() {
                            simulator_finished = true;
                            simulator.run();
                            simulator.extrapolate_to(simulator.max_time());
                        }
                        let response = ui.add(egui::TextEdit::singleline(&mut max_t_text));
                        if response.lost_focus() && let Ok(max_t) = max_t_text.parse::<f32>() {
                            simulator = sim.create_simulator(max_t);
                            let _ = simulator.step();
                            simulator_finished = false;
                            step_count = 1;
                            max_t_text = format!("{max_t:.02}");
                        }
                    });

                    ui.horizontal(|ui| {
                        ui.add(egui::Slider::new(&mut time, min_time..=simulator.max_time())
                            .suffix("s")
                        );
                        if ui.button(if is_paused { "Resume" } else { "Pause" }).clicked() {
                            is_paused = !is_paused;
                            if time >= simulator.max_time() {
                                time = min_time;
                            }
                        }
                    });

                    ui.add(egui::Slider::new(&mut speed, 0.1 ..= 10.)
                        .prefix("x")
                        .logarithmic(true)
                        .clamping(egui::SliderClamping::Never)
                        .text("Speed"));
                });

            egui::Window::new("Scene Selector")
                .show(ctx, |ui| {
                    ui.label("Select Scene:");
                    for (i, scene) in scenes.iter().enumerate() {
                        if ui.selectable_label(i == current_scene_index, scene.name()).clicked() {
                            current_scene_index = i;
                            scene_changed = true;
                        }
                    }
                });

            egui::Window::new("Debug Info").default_open(false).show(ctx, |ui| {
                ui.label(format!("max_time: {max_time}"));
                ui.separator();
                ui.label(format!("Current time: {:?}", time));
                ui.separator();
                let str = simulator.snapshots().nodes().map(|(_, snap)| snap.snapshot.time)
                    .unique_by(|&t| OF(t))
                    .sorted_by_key(|&t| OF((t - time).abs()))
                    .take(5)
                    .sorted_by_key(|&t| OF(t))
                    .join(" -> \n   ");
                ui.label(format!("Times:\n   {str}"));
                ui.separator();
                let (q_min, q_max) = simulator.time_query(time).into_iter()
                    .map(|(_, link)| simulator.snapshots()[link].time)
                    .minmax_by_key(|&t| OF(t)).into_option().unwrap_or_default();
                ui.label(format!("Queryied: {q_min} - {q_max}"));
                ui.separator();
                ui.label(format!("{:#?}", simulator.multiverse()));
            });

            captured_pointer = ctx.wants_pointer_input();
            captured_keyboard = ctx.wants_keyboard_input();
        });

        if scene_changed {
            sim = scenes[current_scene_index].create_world_state();
            simulator = sim.create_simulator(60.);
            let _ = simulator.step();
            simulator_finished = false;
            step_count = 1;
            time = 0.;
            max_t_text = format!("{:.02}", simulator.max_time());
            scene_changed = false;
        }

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
                if time >= simulator.max_time() {
                    time = min_time;
                }
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
                
                zoom *= CAMERA_ZOOM_SPEED.powf(scroll.signum());
                
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
