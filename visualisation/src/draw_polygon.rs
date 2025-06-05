use macroquad::prelude::*;
use i_triangle::float::triangulatable::Triangulatable;

pub fn circle_polygon(
    center: Vec2,
    radius: f32,
    num_segments: u32,
) -> Vec<Vec2> {
    let mut points = Vec::with_capacity(num_segments as usize);
    let angle_step = 2.0 * std::f32::consts::PI / num_segments as f32;
    for i in 0..num_segments {
        let angle = angle_step * i as f32;
        points.push(Vec2::new(
            center.x + radius * angle.cos(),
            center.y + radius * angle.sin(),
        ));
    }
    points
}

pub fn draw_polygon(pos: Vec2, points: &[Vec2], color: Color) {
    let triangulation = points.triangulate().to_triangulation();

    let mesh = Mesh {
        vertices: triangulation.points.iter().map(|point| {
            Vertex {
                position: Vec3::new(pos.x + point.x, pos.y + point.y, 0.0),
                uv: Vec2::default(),
                color: color.into(),
                normal: Vec4::ZERO
            }
        }).collect(),
        indices: triangulation.indices,
        texture: None,
    };

    draw_mesh(&mesh);
}

pub fn draw_polygon_circle(center: Vec2, radius: f32, segments: u32, color: Color) {
    let polygon = circle_polygon(Vec2::ZERO, radius, segments);
    draw_polygon(center, &polygon, color);
}
