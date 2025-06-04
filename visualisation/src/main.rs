use time_engine as te;
use macroquad::prelude::{camera::mouse, *};

const CAMERA_ZOOM_SPEED: f32 = 1.25;

fn window_conf() -> Conf {
    Conf {
        window_title: "Window name".to_owned(),
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let sim = te::Simulation::new(100., 100.);

    let mut cam_offset = Vec2::ZERO;
    let mut zoom = 1.;

    let mouse_pos = |camera: &Camera2D| {
        camera.screen_to_world(Vec2::new(mouse_position().0, mouse_position().1))
    };

    loop {
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

        // Handle camera panning inputs
        if is_key_pressed(KeyCode::R) {
            cam_offset = Vec2::ZERO;
            zoom = 1.;
        }

        camera.target = cam_offset + cam_centering_offset;
        camera.zoom = cam_centering_zoom * zoom;

        if is_mouse_button_down(MouseButton::Left) {
            println!("camera.zoom = {}", camera.zoom);
            cam_offset += mouse_delta_position() / camera.zoom;
        }

        camera.target = cam_offset + cam_centering_offset;

        let scroll = mouse_wheel().1;
        if scroll != 0. {
            let mouse_world_before = mouse_pos(&camera);
            
            zoom *= CAMERA_ZOOM_SPEED.powf(scroll);
            
            camera.zoom = cam_centering_zoom * zoom;
            
            let mouse_world_after = mouse_pos(&camera);
            cam_offset += mouse_world_before - mouse_world_after;
            camera.target = cam_offset + cam_centering_offset;
        }

        camera.zoom = cam_centering_zoom * zoom;

        // Setup camera
        set_camera(&camera);

        // Drawing
        clear_background(BLACK);

        draw_rectangle_lines(-2.5, -2.5, sim.width() + 5., sim.height() + 5., 5., WHITE);

        draw_circle(screen_width() / 2., screen_height() / 2., 10., WHITE);

        next_frame().await;
    }
}
