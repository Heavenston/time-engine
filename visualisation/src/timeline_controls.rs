use macroquad::prelude::*;
use time_engine::TimelineId;
use std::collections::HashMap;

pub struct TimelineControls {
    // State
    timeline_dragging: bool,
    speed_dragging: bool,
    selected_timeline: Option<TimelineId>,
    
    // Layout constants
    height: f32,
    timeline_row_height: f32,
    margin: f32,
    play_button_size: f32,
    bar_height: f32,
    scrubber_size: f32,
    speed_slider_width: f32,
    speed_handle_size: f32,
    
    // Speed range
    min_speed: f32,
    max_speed: f32,
}

impl TimelineControls {
    pub fn new() -> Self {
        Self {
            timeline_dragging: false,
            speed_dragging: false,
            selected_timeline: None,
            
            height: 100.0,
            timeline_row_height: 8.0,
            margin: 40.0,
            play_button_size: 32.0,
            bar_height: 6.0,
            scrubber_size: 16.0,
            speed_slider_width: 100.0,
            speed_handle_size: 12.0,
            
            min_speed: 0.1,
            max_speed: 3.0,
        }
    }
    
    pub fn handle_input(&mut self, sim_t: &mut f32, sim_duration: f32, paused: &mut bool, sim_speed: &mut f32, timelines: &HashMap<TimelineId, (f32, f32)>) -> (bool, Option<TimelineId>) {
        let timeline_y = screen_height() - self.height;
        let timeline_bar_y = timeline_y + 35.0;
        let mouse_x = mouse_position().0;
        let mouse_y = mouse_position().1;
        
        let play_button_x = self.margin;
        let play_button_y = timeline_bar_y;
        
        let speed_slider_x = screen_width() - self.margin - self.speed_slider_width;
        let speed_slider_y = timeline_bar_y;
        let speed_normalized = ((*sim_speed - self.min_speed) / (self.max_speed - self.min_speed)).clamp(0.0, 1.0);
        let speed_handle_x = speed_slider_x + speed_normalized * self.speed_slider_width;
        
        let timeline_bar_x = self.margin + self.play_button_size + 15.0;
        let timeline_bar_width = speed_slider_x - timeline_bar_x - 20.0;
        let scrubber_x = timeline_bar_x + (*sim_t / sim_duration) * timeline_bar_width;
        
        let mouse_in_timeline = mouse_y >= timeline_y;
        let mouse_on_play_button = (mouse_x - play_button_x).abs() <= self.play_button_size / 2.0 && 
                                  (mouse_y - play_button_y).abs() <= self.play_button_size / 2.0;
        let mouse_on_scrubber = (mouse_x - scrubber_x).abs() <= self.scrubber_size / 2.0 && 
                               (mouse_y - timeline_bar_y).abs() <= self.scrubber_size / 2.0;
        let mouse_on_bar = mouse_x >= timeline_bar_x && mouse_x <= timeline_bar_x + timeline_bar_width &&
                          (mouse_y - timeline_bar_y).abs() <= self.scrubber_size / 2.0;
        let mouse_on_speed_handle = (mouse_x - speed_handle_x).abs() <= self.speed_handle_size / 2.0 && 
                                   (mouse_y - speed_slider_y).abs() <= self.speed_handle_size / 2.0;
        let mouse_on_speed_slider = mouse_x >= speed_slider_x && mouse_x <= speed_slider_x + self.speed_slider_width &&
                                   (mouse_y - speed_slider_y).abs() <= self.speed_handle_size / 2.0;

        let mut timeline_changed = false;
        
        if mouse_in_timeline {
            if is_mouse_button_pressed(MouseButton::Left) {
                if mouse_on_play_button {
                    *paused = !*paused;
                } else if mouse_on_scrubber || mouse_on_bar {
                    self.timeline_dragging = true;
                    timeline_changed = true;
                } else if mouse_on_speed_handle || mouse_on_speed_slider {
                    self.speed_dragging = true;
                }
            }
            
            if self.timeline_dragging && is_mouse_button_down(MouseButton::Left) {
                let normalized_x = ((mouse_x - timeline_bar_x) / timeline_bar_width).clamp(0.0, 1.0);
                let new_t = normalized_x * sim_duration;
                *sim_t = new_t;
                
                // Find which timeline this time corresponds to
                let selected = self.find_timeline_at_time(new_t, timelines);
                if selected != self.selected_timeline {
                    self.selected_timeline = selected;
                    timeline_changed = true;
                }
            }
            
            if self.speed_dragging && is_mouse_button_down(MouseButton::Left) {
                let normalized_x = ((mouse_x - speed_slider_x) / self.speed_slider_width).clamp(0.0, 1.0);
                let raw_speed = self.min_speed + normalized_x * (self.max_speed - self.min_speed);
                
                // Snap to preset markers
                let preset_speeds = [0.25, 0.5, 1.0, 2.0];
                let snap_threshold = 0.03;
                
                let mut snapped_speed = raw_speed;
                for &preset in &preset_speeds {
                    if preset >= self.min_speed && preset <= self.max_speed {
                        let preset_normalized = (preset - self.min_speed) / (self.max_speed - self.min_speed);
                        if (normalized_x - preset_normalized).abs() < snap_threshold {
                            snapped_speed = preset;
                            break;
                        }
                    }
                }
                
                *sim_speed = snapped_speed;
            }
        }
        
        if is_mouse_button_released(MouseButton::Left) {
            self.timeline_dragging = false;
            self.speed_dragging = false;
        }

        (mouse_in_timeline, if timeline_changed { self.selected_timeline } else { None })
    }
    
    pub fn draw(&self, sim_t: f32, sim_duration: f32, paused: bool, sim_speed: f32, timelines: &HashMap<TimelineId, (f32, f32)>) {
        let timeline_y = screen_height() - self.height;
        let timeline_bar_y = timeline_y + 35.0;
        let mouse_x = mouse_position().0;
        let mouse_y = mouse_position().1;
        
        let play_button_x = self.margin;
        let play_button_y = timeline_bar_y;
        
        let speed_slider_x = screen_width() - self.margin - self.speed_slider_width;
        let speed_slider_y = timeline_bar_y;
        let speed_normalized = ((sim_speed - self.min_speed) / (self.max_speed - self.min_speed)).clamp(0.0, 1.0);
        let speed_handle_x = speed_slider_x + speed_normalized * self.speed_slider_width;
        
        let timeline_bar_x = self.margin + self.play_button_size + 15.0;
        let timeline_bar_width = speed_slider_x - timeline_bar_x - 20.0;
        let scrubber_x = timeline_bar_x + (sim_t / sim_duration) * timeline_bar_width;
        
        let mouse_on_play_button = (mouse_x - play_button_x).abs() <= self.play_button_size / 2.0 && 
                                  (mouse_y - play_button_y).abs() <= self.play_button_size / 2.0;
        let mouse_on_scrubber = (mouse_x - scrubber_x).abs() <= self.scrubber_size / 2.0 && 
                               (mouse_y - timeline_bar_y).abs() <= self.scrubber_size / 2.0;
        let mouse_on_speed_handle = (mouse_x - speed_handle_x).abs() <= self.speed_handle_size / 2.0 && 
                                   (mouse_y - speed_slider_y).abs() <= self.speed_handle_size / 2.0;
        
        // Timeline background
        draw_rectangle(0.0, timeline_y, screen_width(), self.height, Color::new(0.1, 0.1, 0.1, 0.9));
        
        // Play/Pause button
        let button_color = if mouse_on_play_button { 
            Color::new(0.8, 0.8, 0.8, 1.0) 
        } else { 
            Color::new(0.6, 0.6, 0.6, 1.0) 
        };
        draw_circle(play_button_x, play_button_y, self.play_button_size / 2.0, button_color);
        
        if paused {
            // Draw play triangle
            let triangle_size = 8.0;
            let v1 = Vec2::new(play_button_x - triangle_size / 2.0, play_button_y - triangle_size / 2.0);
            let v2 = Vec2::new(play_button_x - triangle_size / 2.0, play_button_y + triangle_size / 2.0);
            let v3 = Vec2::new(play_button_x + triangle_size / 2.0, play_button_y);
            draw_triangle(v1, v2, v3, BLACK);
        } else {
            // Draw pause bars
            let bar_width = 3.0;
            let bar_height = 10.0;
            let bar_spacing = 2.0;
            draw_rectangle(play_button_x - bar_spacing - bar_width, play_button_y - bar_height / 2.0, 
                          bar_width, bar_height, BLACK);
            draw_rectangle(play_button_x + bar_spacing, play_button_y - bar_height / 2.0, 
                          bar_width, bar_height, BLACK);
        }
        
        // Draw timeline segments
        self.draw_timeline_segments(timeline_bar_x, timeline_bar_y, timeline_bar_width, sim_duration, timelines);
        
        // Draw current time position
        let progress_x = timeline_bar_x + (sim_t / sim_duration) * timeline_bar_width;
        draw_line(progress_x, timeline_bar_y - 20.0, progress_x, timeline_bar_y + (timelines.len() as f32) * self.timeline_row_height + 10.0, 2.0, WHITE);
        
        // Scrubber handle
        let scrubber_color = if mouse_on_scrubber || self.timeline_dragging { 
            Color::new(1.0, 0.4, 0.4, 1.0)
        } else { 
            Color::new(0.9, 0.3, 0.3, 1.0)
        };
        draw_circle(scrubber_x, timeline_bar_y, self.scrubber_size / 2.0, scrubber_color);
        draw_circle(scrubber_x, timeline_bar_y, self.scrubber_size / 2.0 - 2.0, WHITE);
        
        // Time labels
        let start_text = "0:00";
        let end_text = format!("{}:{:02}", (sim_duration as i32) / 60, (sim_duration as i32) % 60);
        let current_text = format!("{}:{:02}", (sim_t as i32) / 60, (sim_t as i32) % 60);
        
        draw_text(start_text, timeline_bar_x, timeline_y + 50.0, 16.0, LIGHTGRAY);
        
        let end_text_size = measure_text(&end_text, None, 16, 1.0);
        draw_text(&end_text, timeline_bar_x + timeline_bar_width - end_text_size.width, timeline_y + 50.0, 16.0, LIGHTGRAY);
        
        let current_text_size = measure_text(&current_text, None, 18, 1.0);
        draw_text(&current_text, (screen_width() - current_text_size.width) / 2.0, timeline_y + 15.0, 18.0, WHITE);
        
        // Speed control slider
        draw_text("Speed:", speed_slider_x - 50.0, timeline_y + 15.0, 14.0, LIGHTGRAY);
        
        // Speed slider track
        draw_rectangle(speed_slider_x, speed_slider_y - self.bar_height / 2.0, 
                      self.speed_slider_width, self.bar_height, Color::new(0.4, 0.4, 0.4, 1.0));
        
        // Speed slider handle
        let speed_handle_color = if mouse_on_speed_handle || self.speed_dragging { 
            Color::new(0.9, 0.6, 0.2, 1.0)
        } else { 
            Color::new(0.7, 0.5, 0.2, 1.0)
        };
        draw_circle(speed_handle_x, speed_slider_y, self.speed_handle_size / 2.0, speed_handle_color);
        draw_circle(speed_handle_x, speed_slider_y, self.speed_handle_size / 2.0 - 2.0, WHITE);
        
        // Speed value display
        let speed_text = format!("{:.1}x", sim_speed);
        draw_text(&speed_text, speed_slider_x + self.speed_slider_width + 10.0, timeline_y + 15.0, 14.0, WHITE);
        
        // Speed preset markers
        let preset_speeds = [0.25, 0.5, 1.0, 2.0];
        for &preset in &preset_speeds {
            if preset >= self.min_speed && preset <= self.max_speed {
                let preset_normalized = (preset - self.min_speed) / (self.max_speed - self.min_speed);
                let preset_x = speed_slider_x + preset_normalized * self.speed_slider_width;
                let is_snapped = (sim_speed - preset).abs() < 0.01;
                let is_near_snap = self.speed_dragging && {
                    let current_normalized = (sim_speed - self.min_speed) / (self.max_speed - self.min_speed);
                    (current_normalized - preset_normalized).abs() < 0.03
                };
                
                let marker_color = if is_snapped { 
                    Color::new(1.0, 0.7, 0.3, 1.0)
                } else if is_near_snap {
                    Color::new(0.9, 0.6, 0.2, 1.0)
                } else { 
                    Color::new(0.6, 0.6, 0.6, 0.8)
                };
                
                let line_width = if is_snapped || is_near_snap { 2.0 } else { 1.0 };
                let line_height = if is_snapped || is_near_snap { 12.0 } else { 8.0 };
                
                draw_line(preset_x, speed_slider_y - line_height, preset_x, speed_slider_y + line_height, line_width, marker_color);
                
                if preset == 1.0 {
                    draw_text("1x", preset_x - 8.0, speed_slider_y - 18.0, 12.0, marker_color);
                }
            }
        }
    }
    
    fn find_timeline_at_time(&self, t: f32, timelines: &HashMap<TimelineId, (f32, f32)>) -> Option<TimelineId> {
        timelines.iter()
            .filter(|(_, (start, end))| t >= *start && t <= *end)
            .max_by_key(|(tid, _)| *tid)
            .map(|(tid, _)| *tid)
    }
    
    fn draw_timeline_segments(&self, bar_x: f32, bar_y: f32, bar_width: f32, sim_duration: f32, timelines: &HashMap<TimelineId, (f32, f32)>) {
        // Sort timelines by their ID for consistent ordering
        let mut sorted_timelines: Vec<_> = timelines.iter().collect();
        sorted_timelines.sort_by_key(|(tid, _)| *tid);
        
        // Timeline colors
        let timeline_colors = [
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
        
        for (row_idx, (tid, (start, end))) in sorted_timelines.iter().enumerate() {
            let row_y = bar_y + (row_idx as f32) * self.timeline_row_height;
            
            // Draw background track for this row
            draw_rectangle(bar_x, row_y - self.timeline_row_height / 2.0, 
                          bar_width, self.timeline_row_height, Color::new(0.3, 0.3, 0.3, 1.0));
            
            let start_x = bar_x + (start / sim_duration) * bar_width;
            let end_x = bar_x + (end / sim_duration) * bar_width;
            let width = end_x - start_x;
            
            let color = timeline_colors[row_idx % timeline_colors.len()];
            let final_color = if self.selected_timeline == Some(**tid) {
                Color::new(color.r, color.g, color.b, 1.0)
            } else {
                Color::new(color.r, color.g, color.b, 0.6)
            };
            
            draw_rectangle(start_x, row_y - self.timeline_row_height / 2.0, 
                          width, self.timeline_row_height, final_color);
            
            // Draw timeline split points
            if *start > 0.0 {
                draw_line(start_x, row_y - self.timeline_row_height / 2.0, start_x, row_y + self.timeline_row_height / 2.0, 1.0, WHITE);
            }
            
            // Draw timeline ID label
            let label = format!("T{}", tid);
            draw_text(&label, bar_x - 30.0, row_y + 3.0, 12.0, WHITE);
        }
    }
    
    pub fn get_selected_timeline(&self) -> Option<TimelineId> {
        self.selected_timeline
    }
    
    pub fn set_selected_timeline(&mut self, timeline: Option<TimelineId>) {
        self.selected_timeline = timeline;
    }
    
    pub fn get_deepest_timeline_at_time(&self, t: f32, timelines: &HashMap<TimelineId, (f32, f32)>) -> Option<TimelineId> {
        timelines.iter()
            .filter(|(_, (start, end))| t >= *start && t <= *end)
            .max_by_key(|(tid, _)| *tid)
            .map(|(tid, _)| *tid)
    }
}
