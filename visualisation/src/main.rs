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
                Vec2::new(50., 85.),
            )
        });
        sim.push_portal(te::Portal {
            height: 20.,
            initial_transform: Affine2::from_angle_translation(
                0.,
                Vec2::new(85., 50.),
            )
        });
        sim.push_sphere(te::Sphere {
            max_age: 10.,
            initial_pos: glam::Vec2::new(50., 50.),
            initial_velocity: glam::Vec2::new(0., 30.),
            radius: 3.,
            ..Default::default()
        });
        sim.push_sphere(te::Sphere {
            initial_time: 2.5,
            initial_pos: glam::Vec2::new(20., 6.),
            initial_velocity: glam::Vec2::new(30., 20.),
            radius: 3.,
            ..Default::default()
        });
        sim.push_sphere(te::Sphere {
            initial_pos: glam::Vec2::new(20., 20.),
            initial_velocity: glam::Vec2::new(10., 10.),
            radius: 3.,
            ..Default::default()
        });
        sim
    };
    println!("Simulating...");
    let simulation_result = sim.simulate(sim_duration);
    println!("Finished simulation");

    let mut cam_offset = Vec2::ZERO;
    let mut zoom = 1.;
    let mut sim_start: f32 = 0.;

    let mouse_pos = |camera: &Camera2D| {
        camera.screen_to_world(Vec2::new(mouse_position().0, mouse_position().1))
    };

    // Setup ui skin
    {
        let label_style = root_ui().style_builder()
            .text_color(WHITE)
            .build();
        let skin = ui::Skin {
            label_style,
            ..root_ui().default_skin()
        };
        root_ui().push_skin(&skin);
    }

    loop {
        let mut t = get_time() as f32 - sim_start;
        if t > sim_duration {
            sim_start = get_time() as f32;
            t = 0.;
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

            sim_start = get_time() as f32;
            t = 0.;
        }

        camera.target = cam_offset + cam_centering_offset;
        camera.zoom = cam_centering_zoom * zoom;

        if is_mouse_button_down(MouseButton::Left) {
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

        for (idx, sphere) in sim.spheres().iter().enumerate() {
            let Some(pos) = simulation_result.get_sphere_pos(idx, t)
            else { continue };
            draw_circle(pos.x, pos.y, sphere.radius, WHITE);
        }

        for portal in sim.portals() {
            let h2 = portal.height / 2.;
            let start = portal.initial_transform.transform_point2(Vec2::new(0., -h2));
            let end = portal.initial_transform.transform_point2(Vec2::new(0., h2));
            draw_line(start.x, start.y, end.x, end.y, 1., GREEN);
        }

        root_ui().label(None, &format!("time: {t:.02}s/{sim_duration:.02}s"));

        next_frame().await;
    }
}
