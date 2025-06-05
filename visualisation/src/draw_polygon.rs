use time_engine as te;

use macroquad::prelude::*;
use i_triangle::float::triangulatable::Triangulatable;

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
    let polygon = te::circle_polygon(Vec2::ZERO, radius, segments);
    draw_polygon(center, &polygon, color);
}
