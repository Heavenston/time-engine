mod draw_polygon;
mod simulation_renderer;
mod scenes;

use itertools::Itertools;
use simulation_renderer::{render_simulation, RenderSimulationArgs};
use scenes::{get_all_scenes, Scene};

use macroquad::{ prelude::*, ui::{ self, root_ui } };
use ordered_float::OrderedFloat as OF;
use time_engine::{WorldState, Simulator};
use std::rc::Rc;

const CAMERA_ZOOM_SPEED: f32 = 1.25;

struct AppState {
    scenes: Vec<Box<dyn Scene>>,
    current_scene_index: usize,
    sim: Rc<WorldState>,
    simulator: Simulator,
    step_count: u32,
    scene_changed: bool,
    
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
}

impl AppState {
    fn new() -> Self {
        let scenes = get_all_scenes();
        let current_scene_index = 0;
        let sim = Rc::new(scenes[current_scene_index].create_world_state());
        let mut simulator = sim.clone().create_simulator(scenes[current_scene_index].default_max_time());
        let _ = simulator.step();
        
        Self {
            max_t_text: format!("{:.02}", simulator.max_time()),
            scenes,
            current_scene_index,
            sim,
            simulator,
            step_count: 1,
            scene_changed: false,
            cam_offset: Vec2::ZERO,
            zoom: 1.0,
            enable_debug_rendering: false,
            is_paused: true,
            time: 0.0,
            speed: 1.0,
            simulator_finished: false,
            auto_full_simulate: true,
        }
    }
    
    fn get_time_bounds(&self) -> (f32, f32) {
        let min_time = self.simulator.snapshots().nodes()
            .map(|(_, node)| node.snapshot.time)
            .min_by_key(|&t| OF(t))
            .unwrap_or(0.);
        let max_time = self.simulator.snapshots().nodes()
            .filter(|(_, node)| node.age_children.is_empty())
            .map(|(_, node)| node.snapshot.time)
            .max_by_key(|&t| OF(t))
            .unwrap_or(0.);
        (min_time, max_time)
    }
    
    fn simulation_step(&mut self) {
        self.step_count += 1;
        if self.simulator.step().is_break() {
            self.simulator_finished = true;
            self.simulator.extrapolate_to(self.simulator.max_time());
        }
    }
    
    fn update_simulation(&mut self) {
        let (min_time, max_time) = self.get_time_bounds();
        
        if !self.is_paused && self.time <= max_time {
            self.time += get_frame_time() * self.speed;
        }
        
        if self.time < min_time {
            self.time = min_time;
        } else if self.time > self.simulator.max_time() {
            self.time = self.simulator.max_time();
            self.is_paused = true;
        } else if self.auto_full_simulate && !self.simulator_finished {
            self.simulation_step();
        } else if self.time > max_time {
            if self.simulator_finished {
                self.time = max_time;
                self.is_paused = true;
            } else {
                self.simulation_step();
            }
        }
    }
    
    fn setup_camera(&self) -> Camera2D {
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
            let mouse_pos = |camera: &Camera2D| {
                camera.screen_to_world(Vec2::new(mouse_position().0, mouse_position().1))
            };
            
            let mouse_world_before = mouse_pos(camera);
            self.zoom *= CAMERA_ZOOM_SPEED.powf(scroll.signum());
            
            let cam_centering_zoom = camera.zoom / self.zoom;
            let new_camera = Camera2D {
                target: camera.target,
                offset: camera.offset,
                zoom: cam_centering_zoom * self.zoom,
                rotation: camera.rotation,
                render_target: None,
                viewport: camera.viewport,
            };
            
            let mouse_world_after = mouse_pos(&new_camera);
            self.cam_offset += mouse_world_before - mouse_world_after;
        }
    }
    
    fn reset_simulator(&mut self) {
        self.simulator = self.sim.clone().create_simulator(self.simulator.max_time());
        self.max_t_text = format!("{:.02}", self.simulator.max_time());
        self.simulator_finished = false;
        self.step_count = 0;

        // initial step to resolve collisions happening at t=0
        self.simulation_step();
    }
    
    fn full_simulate(&mut self) {
        while !self.simulator_finished {
            self.simulation_step();
        }
    }
    
    fn change_scene(&mut self, new_index: usize) {
        if new_index != self.current_scene_index {
            self.current_scene_index = new_index;
            self.scene_changed = true;
        }
    }
    
    fn handle_scene_change(&mut self) {
        if self.scene_changed {
            self.sim = Rc::new(self.scenes[self.current_scene_index].create_world_state());
            self.reset_simulator();
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
            
            self.render_info_window(ctx);
            self.render_controls_window(ctx, max_time);
            self.render_scene_selector_window(ctx);
            self.render_debug_window(ctx);
            
            captured_pointer = ctx.wants_pointer_input();
            captured_keyboard = ctx.wants_keyboard_input();
        });
        
        (captured_pointer, captured_keyboard)
    }
    
    fn render_info_window(&self, ctx: &egui::Context) {
        egui::Window::new("Informations").show(ctx, |ui| {
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
                    .map(|(_, link)| self.simulator.snapshots()[link].timeline_id)
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
            if !self.auto_full_simulate {
                ui.horizontal(|ui| {
                    ui.label("Finished simulating:");
                    if self.simulator_finished {
                        ui.colored_label(egui::Color32::LIGHT_GREEN, "Yes");
                    } else {
                        ui.colored_label(egui::Color32::LIGHT_RED, "No");
                    }
                });
            }
        });
    }
    
    fn render_controls_window(&mut self, ctx: &egui::Context, max_time: f32) {
        egui::Window::new("Controls").show(ctx, |ui| {
            let progress = max_time / self.simulator.max_time();
            if progress < 1. {
                ui.add(egui::ProgressBar::new(progress));
            }
            
            ui.horizontal(|ui| {
                if ui.button("Full Reset").clicked() {
                    self.reset_simulator();
                }
                if !self.auto_full_simulate && ui.button("Full Simulate").clicked() {
                    self.full_simulate();
                }
                let response = ui.add(egui::TextEdit::singleline(&mut self.max_t_text));
                if response.lost_focus() && let Ok(max_t) = self.max_t_text.parse::<f32>() {
                    self.simulator = self.sim.clone().create_simulator(max_t);
                    let _ = self.simulator.step();
                    self.simulator_finished = false;
                    self.step_count = 1;
                    self.max_t_text = format!("{max_t:.02}");
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
    }
    
    fn render_scene_selector_window(&mut self, ctx: &egui::Context) {
        let current_scene_index = self.current_scene_index;
        let mut scene_to_change = None;
        
        egui::Window::new("Scene Selector").show(ctx, |ui| {
            ui.label("Select Scene:");
            for (i, scene) in self.scenes.iter().enumerate() {
                if ui.selectable_label(i == current_scene_index, scene.name()).clicked() {
                    scene_to_change = Some(i);
                }
            }
        });
        
        if let Some(i) = scene_to_change {
            self.change_scene(i);
        }
    }
    
    fn render_debug_window(&mut self, ctx: &egui::Context) {
        egui::Window::new("Debug Info").default_open(false).show(ctx, |ui| {
            let (_, max_time) = self.get_time_bounds();
            ui.label(format!("max_time: {max_time}"));
            ui.label(format!("Current time: {:?}", self.time));
            ui.separator();
            let str = self.simulator.snapshots().nodes().map(|(_, snap)| snap.snapshot.time)
                .unique_by(|&t| OF(t))
                .sorted_by_key(|&t| OF((t - self.time).abs()))
                .take(5)
                .sorted_by_key(|&t| OF(t))
                .join(" -> \n   ");
            ui.label(format!("Times:\n   {str}"));
            let (q_min, q_max) = self.simulator.time_query(self.time).into_iter()
                .map(|(_, link)| self.simulator.snapshots()[link].time)
                .minmax_by_key(|&t| OF(t)).into_option().unwrap_or_default();
            ui.label(format!("Queryied: {q_min} - {q_max}"));
            ui.separator();
            ui.label(format!("{:#?}", self.simulator.multiverse()));
            ui.separator();
            if ui.button("Manual Step").clicked() {
                self.simulation_step();
            }
            ui.checkbox(&mut self.auto_full_simulate, "Auto full simulate");
        });
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

        let camera = app_state.setup_camera();
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
