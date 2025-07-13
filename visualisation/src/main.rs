#![feature(new_range_api)]

mod draw_polygon;
mod simulation_renderer;
mod scenes;

use simulation_renderer::{ render_simulation, RenderSimulationArgs };
use scenes::{ Scene, get_all_scenes };

use std::{ any::Any, sync::Arc };

use itertools::Itertools;
use macroquad::{ prelude::*, ui::{ self, root_ui } };
use ordered_float::OrderedFloat as OF;
use time_engine::{WorldState, Simulator};

use crate::scenes::BasicBouncingScene;

const CAMERA_ZOOM_SPEED: f32 = 1.25;

struct AppState {
    scenes: Vec<Arc<dyn Scene>>,
    current_scene: Arc<dyn Scene>,
    sim: Arc<WorldState>,
    simulator: Simulator,
    step_count: u32,
    scene_changed: bool,

    // ui state
    controls_opened: bool,
    informations_opened: bool,
    debug_info_opened: bool,
    random_scene_opened: bool,
    
    // Camera state
    cam_offset: Vec2,
    zoom: f32,
    
    // Simulation state
    enable_debug_rendering: bool,
    is_paused: bool,
    time: f32,
    speed: f32,
    max_t_text: String,
    simulator_finished: bool,
    auto_full_simulate: bool,
    auto_step: bool,
}

impl AppState {
    fn new() -> Self {
        let scenes = get_all_scenes();
        let scene = Arc::clone(&scenes[0]);
        let sim = Arc::new(scene.create_world_state());
        let simulator = sim.clone().create_simulator(scene.default_max_time());
        
        Self {
            max_t_text: format!("{:.02}", simulator.max_time()),
            scenes,
            current_scene: scene,
            sim,
            simulator,
            step_count: 1,
            scene_changed: false,

            controls_opened: true,
            informations_opened: true,
            debug_info_opened: cfg!(debug_assertions),
            random_scene_opened: false,

            cam_offset: Vec2::ZERO,
            zoom: 1.0,

            enable_debug_rendering: cfg!(debug_assertions),
            is_paused: true,
            time: 0.0,
            speed: 1.0,
            simulator_finished: false,
            auto_full_simulate: !cfg!(debug_assertions),
            auto_step: true,
        }
    }
    
    fn get_time_bounds(&self) -> (f32, f32) {
        let min_time = self.simulator.snapshots().nodes()
            .flat_map(|node| {
                let snaps = self.simulator.integrate_snapshot(node.handle());
                (0..snaps.len())
                    .map(move |i| snaps[i].time)
            })
            .min_by_key(|&t| OF(t))
            .unwrap_or(0.);
        let max_time = self.simulator.snapshots().nodes()
            .filter(|node| node.children().is_empty())
            .flat_map(|node| {
                let snaps = self.simulator.integrate_snapshot(node.handle());
                (0..snaps.len())
                    .map(move |i| snaps[i].time)
            })
            .max_by_key(|&t| OF(t))
            .unwrap_or(0.);
        (min_time, max_time)
    }
    
    fn simulation_step(&mut self) {
        println!("\n#### Start of step {} ####", self.step_count);
        self.step_count += 1;
        let mut simulator = std::mem::replace(&mut self.simulator, Simulator::empty());
        let max_time = simulator.max_time();
        let result = std::panic::catch_unwind(move || {
            (simulator.step(), simulator)
        });
        match result {
            Ok((control_flow, simulator)) => {
                self.simulator = simulator;
                self.simulator_finished |= control_flow.is_break();
                println!("#### End of step {} ###", self.step_count-1);
            },
            Err(_) => {
                println!("#### End of step {} ###: PANIC CATCHED", self.step_count-1);
                self.reset_simulator(Some(max_time));
                self.auto_step = false;
                self.auto_full_simulate = false;
                self.is_paused = true;
            },
        }
    }
    
    fn update_simulation(&mut self) {
        let (min_time, max_time) = self.get_time_bounds();
        
        if !self.is_paused {
            self.time += get_frame_time() * self.speed;
        }
        
        if self.time < min_time {
            self.time = min_time;
        } else if self.time > self.simulator.max_time() {
            self.time = self.simulator.max_time();
            self.is_paused = true;
        } else if !self.simulator_finished && (self.auto_full_simulate || (self.time > max_time && self.auto_step)) {
            self.simulation_step();
        }
    }
    
    fn create_camera(&self) -> Camera2D {
        let (cw, ch) = if screen_width() > screen_height() {
            ((screen_width() / screen_height()) * self.sim.height(), self.sim.height())
        } else {
            (self.sim.width(), (screen_height() / screen_width()) * self.sim.width())
        };
        
        let mut camera = Camera2D::from_display_rect(Rect {
            x: (self.sim.width() - cw) / 2.,
            y: (self.sim.height() - ch) / 2.,
            w: cw,
            h: ch,
        });
        
        let cam_centering_zoom = camera.zoom;
        let cam_centering_offset = camera.target;
        
        camera.target = self.cam_offset + cam_centering_offset;
        camera.zoom = cam_centering_zoom * self.zoom;
        
        camera
    }
    
    fn handle_keyboard_input(&mut self) {
        if is_key_pressed(KeyCode::R) {
            self.cam_offset = Vec2::ZERO;
            self.zoom = 1.;
            self.time = 0.;
        }
        
        if is_key_pressed(KeyCode::D) {
            self.enable_debug_rendering = !self.enable_debug_rendering;
        }
        
        if is_key_pressed(KeyCode::Space) {
            self.is_paused = !self.is_paused;
            if self.time >= self.simulator.max_time() {
                let (min_time, _) = self.get_time_bounds();
                self.time = min_time;
            }
        }
    }
    
    fn handle_mouse_input(&mut self, camera: &Camera2D) {
        if is_mouse_button_down(MouseButton::Left) {
            self.cam_offset += mouse_delta_position() / camera.zoom;
        }
        
        let scroll = mouse_wheel().1;
        if scroll != 0. {
            let mouse_world_before = camera.screen_to_world(Vec2::new(mouse_position().0, mouse_position().1));
            self.zoom *= CAMERA_ZOOM_SPEED.powf(scroll.signum());
            let new_camera = self.create_camera();
            let mouse_world_after = new_camera.screen_to_world(Vec2::new(mouse_position().0, mouse_position().1));
            self.cam_offset += mouse_world_before - mouse_world_after;
        }
    }
    
    fn reset_simulator(&mut self, max_t_override: Option<f32>) {
        self.simulator = self.sim.clone().create_simulator(
            max_t_override.unwrap_or(self.simulator.max_time())
        );
        self.max_t_text = format!("{:.02}", self.simulator.max_time());
        self.simulator_finished = false;
        self.step_count = 0;
    }
    
    fn full_simulate(&mut self) {
        let prev_full_simulate_option = self.auto_full_simulate;

        // set to true to detect when the simulation crashes
        self.auto_full_simulate = true;
        while !self.simulator_finished && self.auto_full_simulate {
            self.simulation_step();
        }

        self.auto_full_simulate = prev_full_simulate_option;
    }
    
    fn change_scene(&mut self, new_scene: Arc<dyn Scene>) {
        self.scene_changed = true;
        self.current_scene = new_scene;
    }
    
    fn handle_scene_change(&mut self) {
        if self.scene_changed {
            self.sim = Arc::new(self.current_scene.create_world_state());
            self.reset_simulator(None);
            self.time = 0.;
            self.scene_changed = false;
        }
    }
    
    fn render_ui(&mut self) -> (bool, bool) {
        let mut captured_pointer = false;
        let mut captured_keyboard = false;
        
        let (_, max_time) = self.get_time_bounds();
        
        egui_macroquad::ui(|ctx| {
            let mut style = (*ctx.style()).clone();
            style.visuals.override_text_color = Some(egui::Color32::WHITE);
            ctx.set_style(style);
            
            self.render_menu_bar(ctx);
            self.render_info_window(ctx);
            self.render_controls_window(ctx, max_time);
            self.render_debug_window(ctx);
            self.render_random_scene_window(ctx);
            
            captured_pointer = ctx.wants_pointer_input();
            captured_keyboard = ctx.wants_keyboard_input();
        });
        
        (captured_pointer, captured_keyboard)
    }

    fn render_menu_bar(&mut self, ctx: &egui::Context) {
        use egui::menu;

        let mut scene_to_change = None::<Arc<dyn Scene>>;

        egui::TopBottomPanel::top("menubar").show(ctx, |ui| {
            menu::bar(ui, |ui| {
                ui.menu_button("Change Scene", |ui| {
                    for scene in &self.scenes {
                        if ui.selectable_label(Arc::ptr_eq(&self.current_scene, scene), scene.name()).clicked() {
                            ui.close_menu();
                            scene_to_change = Some(Arc::clone(&scene));
                            self.random_scene_opened = false;
                        }
                    }
                    if ui.selectable_label(self.random_scene_opened, "Basic Bouncing Random").clicked() {
                        ui.close_menu();
                        self.random_scene_opened = true;
                    }
                });

                ui.menu_button("Window", |ui| {
                    if ui.selectable_label(self.controls_opened, "Controls").clicked() {
                        self.controls_opened = !self.controls_opened;
                    }
                    if ui.selectable_label(self.informations_opened, "Informations").clicked() {
                        self.informations_opened = !self.informations_opened;
                    }
                    if ui.selectable_label(self.debug_info_opened, "Debug Info").clicked() {
                        self.debug_info_opened = !self.debug_info_opened;
                    }
                });
            });
        });
        
        if let Some(scene) = scene_to_change {
            self.change_scene(scene);
        }
    }
    
    fn render_info_window(&mut self, ctx: &egui::Context) {
        let mut opened = self.informations_opened;
        egui::Window::new("Informations")
            .open(&mut opened)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label("FPS:");
                    ui.colored_label(egui::Color32::GRAY, format!("{}", get_fps()));
                });
                ui.horizontal(|ui| {
                    ui.label("Total Simulation Steps:");
                    ui.colored_label(egui::Color32::GRAY, format!("{}", self.step_count));
                });
                ui.horizontal(|ui| {
                    ui.label("Number of timelines:");
                    let active = self.simulator.time_query(self.time).into_iter()
                        .map(|s| s.timeline_id)
                        .unique()
                        .count();
                    ui.colored_label(egui::Color32::GRAY, format!("{active}/{}", self.simulator.multiverse().len()));
                });
                ui.horizontal(|ui| {
                    ui.label("Number of snapshots:");
                    let number = self.simulator.snapshots().nodes().count();
                    ui.colored_label(egui::Color32::GRAY, format!("{number}"));
                });
                ui.horizontal(|ui| {
                    ui.label("Enabled debug rendering (d):");
                    if self.enable_debug_rendering {
                        ui.colored_label(egui::Color32::LIGHT_GREEN, "Yes");
                    } else {
                        ui.colored_label(egui::Color32::LIGHT_RED, "No");
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Finished simulating:");
                    if self.simulator_finished {
                        ui.colored_label(egui::Color32::LIGHT_GREEN, "Yes");
                    } else {
                        ui.colored_label(egui::Color32::LIGHT_RED, "No");
                    }
                });
            });
        self.informations_opened = opened;
    }
    
    fn render_controls_window(&mut self, ctx: &egui::Context, max_time: f32) {
        let mut opened = self.controls_opened;
        egui::Window::new("Controls")
            .open(&mut opened)
            .show(ctx, |ui| {
                let progress = max_time / self.simulator.max_time();
                if progress < 1. && !self.simulator_finished {
                    ui.add(egui::ProgressBar::new(progress));
                }
            
                ui.horizontal(|ui| {
                    if ui.button("Full Reset").clicked() {
                        self.reset_simulator(None);
                    }
                    if !self.auto_full_simulate && ui.button("Full Simulate").clicked() {
                        self.full_simulate();
                    }
                    let response = ui.add(egui::TextEdit::singleline(&mut self.max_t_text));
                    if response.lost_focus() && let Ok(max_t) = self.max_t_text.parse::<f32>() {
                        self.reset_simulator(Some(max_t));
                    }
                });
            
                let (min_time, _) = self.get_time_bounds();
                ui.horizontal(|ui| {
                    ui.add(egui::Slider::new(&mut self.time, min_time..=self.simulator.max_time())
                        .suffix("s"));
                    if ui.button(if self.is_paused { "Resume" } else { "Pause" }).clicked() {
                        self.is_paused = !self.is_paused;
                        if self.time >= self.simulator.max_time() {
                            self.time = min_time;
                        }
                    }
                });
            
                ui.add(egui::Slider::new(&mut self.speed, 0.1..=10.)
                    .prefix("x")
                    .logarithmic(true)
                    .clamping(egui::SliderClamping::Never)
                    .text("Speed"));
            });
        self.controls_opened = opened;
    }
    
    fn render_debug_window(&mut self, ctx: &egui::Context) {
        let mut opened = self.debug_info_opened;
        egui::Window::new("Debug Info")
            .open(&mut opened)
            .show(ctx, |ui| {
                let (_, max_time) = self.get_time_bounds();
                ui.label(format!("max_time: {max_time}"));
                ui.label(format!("Current time: {:?}", self.time));
                ui.separator();
                let str = self.simulator.snapshots().nodes()
                    .flat_map(|node| {
                        let snaps = self.simulator.integrate_snapshot(node.handle());
                        (0..snaps.len())
                            .map(move |i| snaps[i].time)
                    })
                    .unique_by(|&t| OF(t))
                    .sorted_by_key(|&t| OF((t - self.time).abs()))
                    .take(5)
                    .sorted_by_key(|&t| OF(t))
                    .join(" -> \n   ");
                ui.label(format!("Times:\n   {str}"));
                let (q_min, q_max) = self.simulator.time_query(self.time)
                    .map(|snap| snap.time)
                    .minmax_by_key(|&t| OF(t)).into_option().unwrap_or_default();
                ui.label(format!("Queryied: {q_min} - {q_max}"));
                // ui.separator();
                // ui.label(format!("{:#?}", self.simulator.multiverse()));
                {
                    let pp = self.simulator.time_query(self.time)
                        .map(|snap| {format!(
                            "{snap} - {}",
                            snap.portal_traversals.iter()
                                .map(|traversal| format!(
                                    "{} {:?} {:?} {:?} {:?}",
                                    traversal.half_portal_idx,
                                    traversal.direction,
                                    traversal.time_range,
                                    traversal.direction,
                                    traversal.traversal_direction,
                                ))
                                .join(" - ")
                        )}).join("\n");
                    ui.separator();
                    ui.label(format!("{pp}"));
                }
                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button("Manual Step").clicked() {
                        self.simulation_step();
                    }
                    ui.checkbox(&mut self.auto_step, "Auto step");
                    ui.checkbox(&mut self.auto_full_simulate, "Auto full simulate");
                });
            });
        self.debug_info_opened = opened;
    }

    fn render_random_scene_window(&mut self, ctx: &egui::Context) {
        let mut opened = self.random_scene_opened;
        egui::Window::new("Random Scene")
            .open(&mut opened)
            .show(ctx, |ui| {
                let mut current_scene =
                    Arc::downcast::<BasicBouncingScene>(Arc::clone(&self.current_scene) as Arc<dyn Any + Send + Sync>)
                    .unwrap_or_else(|_| {
                        self.scene_changed = true;
                        Arc::new(BasicBouncingScene {
                            name: "Basic Bouncing Random",
                            ..Default::default()
                        })
                    })
                ;

                let mut changed = false;

                let mut seed_text = format!("{}", current_scene.seed);
                let mut ball_count_text = format!("{}", current_scene.ball_count);
                let mut width_text = format!("{}", current_scene.width);
                let mut height_text = format!("{}", current_scene.height);

                ui.horizontal(|ui| {
                    ui.label("Seed:");
                    changed |= ui.text_edit_singleline(&mut seed_text).changed();
                });
                ui.horizontal(|ui| {
                    ui.label("Ball count:");
                    changed |= ui.text_edit_singleline(&mut ball_count_text).changed();
                });
                ui.horizontal(|ui| {
                    ui.label("Size:");
                    changed |= ui.text_edit_singleline(&mut width_text).changed();
                    ui.label("x");
                    changed |= ui.text_edit_singleline(&mut height_text).changed();
                });

                if
                    changed &&
                    let Ok(seed) = seed_text.parse::<u64>() &&
                    let Ok(ball_count) = ball_count_text.parse::<usize>() &&
                    let Ok(width) = width_text.parse::<f32>() && width > 1. &&
                    let Ok(height) = height_text.parse::<f32>() && height > 1.
                {
                    self.scene_changed = true;
                    current_scene = Arc::new(BasicBouncingScene {
                        seed,
                        name: "Basic Bouncing Random",
                        ball_count: ball_count.clamp(0, 100),
                        width,
                        height,
                    });
                }

                self.current_scene = current_scene;
            });
        self.random_scene_opened = opened;
    }
    
    fn render_simulation(&self) {
        render_simulation(RenderSimulationArgs {
            world_state: &self.sim,
            enable_debug_rendering: self.enable_debug_rendering,
            time: self.time,
            simulator: &self.simulator,
        });
    }
}

fn window_conf() -> Conf {
    Conf {
        window_title: "Window name".to_owned(),
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut app_state = AppState::new();

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
        app_state.update_simulation();
        app_state.handle_scene_change();

        let camera = app_state.create_camera();
        let (captured_pointer, captured_keyboard) = app_state.render_ui();

        if !captured_keyboard {
            app_state.handle_keyboard_input();
        }

        if !captured_pointer {
            app_state.handle_mouse_input(&camera);
        }

        set_camera(&camera);
        app_state.render_simulation();

        egui_macroquad::draw();
        next_frame().await;
    }
}
