use i_overlay::i_shape::base::data::Shapes;
use macroquad::prelude::*;
use i_triangle::float::triangulatable::Triangulatable;

pub fn draw_shapes(pos: Vec2, shapes: &Shapes<Vec2>, color: Color) {
    let triangulation = shapes.triangulate().to_triangulation();

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
