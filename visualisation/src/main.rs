mod draw_polygon;
mod timeline_controls;
use itertools::Itertools;
use timeline_controls::TimelineControls;
mod simulation_renderer;
use simulation_renderer::{render_simulation, RenderSimulationArgs};

use time_engine as te;
use macroquad::{ prelude::*, ui::{ self, root_ui } };
use std::collections::HashMap;

const CAMERA_ZOOM_SPEED: f32 = 1.25;

fn collect_timeline_ranges(simulation_result: &te::SimulationResult) -> HashMap<te::TimelineId, (f32, f32)> {
    let mut ranges = HashMap::new();
    
    for sphere in &simulation_result.spheres {
        for snapshot in &sphere.snapshots {
            let entry = ranges.entry(snapshot.tid).or_insert((f32::INFINITY, f32::NEG_INFINITY));
            entry.0 = entry.0.min(snapshot.t);
            entry.1 = entry.1.max(snapshot.t);
        }
    }
    
    ranges
}

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
            time_offset: -2.2,
            // time_offset: -1.,
            // time_offset: 0.,
            // time_offset: 0.5,
            // time_offset: 1.,
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
            initial_pos: glam::Vec2::new(50., 10.),
            initial_velocity: glam::Vec2::new(0., 30.),
            radius: 3.,
            ..Default::default()
        });
        // sim.push_sphere(te::Sphere {
        //     initial_pos: glam::Vec2::new(20., 6.),
        //     initial_velocity: glam::Vec2::new(30., 30.),
        //     radius: 3.,
        //     ..Default::default()
        // });
        // sim.push_sphere(te::Sphere {
        //     initial_pos: glam::Vec2::new(20., 20.),
        //     initial_velocity: glam::Vec2::new(40., -30.),
        //     radius: 3.,
        //     ..Default::default()
        // });
        sim
    };
    println!("Simulating...");
    let simulation_result = sim.simulate(15f32);
    let sim_duration = simulation_result.max_t();
    println!("{simulation_result:#?}");
    println!("Finished simulation");

    let mut cam_offset = Vec2::ZERO;
    let mut zoom = 1.;
    let mut paused = true;
    let mut sim_t = 0.;
    let mut sim_speed = 1.;
    let mut enable_debug_rendering = false;
    let mut selected_timeline: Option<te::TimelineId> = None;
    
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

        // Collect timeline information
        let timelines = collect_timeline_ranges(&simulation_result);
        
        // Check for timelines that are newly active at current time
        let active_timelines_at_current_time: Vec<_> = timelines.iter()
            .filter(|(_, (start, end))| *start <= sim_t && sim_t <= *end)
            .map(|(tid, _)| *tid)
            .collect();
        
        // Find the deepest timeline that just became active (started exactly at current time or very recently)
        let newly_started_timeline = active_timelines_at_current_time.iter()
            .filter(|&&tid| {
                if let Some((start, _)) = timelines.get(&tid) {
                    // Check if this timeline started within the last frame (to catch newly started timelines)
                    (*start <= sim_t) && (*start > sim_t - get_frame_time() * sim_speed * 2.0)
                } else {
                    false
                }
            })
            .max(); // Uses Ord trait on TimelineId to get the deepest one
        
        if let Some(&new_timeline) = newly_started_timeline {
            // Only auto-switch if the user hasn't manually selected a timeline recently
            if selected_timeline.is_none_or(|current| new_timeline > current) {
                selected_timeline = Some(new_timeline);
                timeline_controls.set_selected_timeline(selected_timeline);
            }
        }
        
        // Handle timeline input
        let (mouse_in_timeline, timeline_change) = timeline_controls.handle_input(&mut sim_t, sim_duration, &mut paused, &mut sim_speed, &timelines);
        
        // Update selected timeline if user manually changed it
        if timeline_change.is_some() {
            selected_timeline = timeline_change;
        }
        
        // If no timeline is selected, default to the deepest one at current time
        if selected_timeline.is_none() {
            selected_timeline = timeline_controls.get_deepest_timeline_at_time(sim_t, &timelines);
            timeline_controls.set_selected_timeline(selected_timeline);
        }

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
            selected_timeline,
        });

        root_ui().label(None, &format!("fps: {}", get_fps()));
        root_ui().label(None, &format!("time: {sim_t:.02}s/{sim_duration:.02}s"));
        if paused {
            root_ui().label(None, "PAUSED");
        }

        // Draw timeline controls
        set_default_camera();
        timeline_controls.draw(sim_t, sim_duration, paused, sim_speed, &timelines);

        next_frame().await;
    }
}
