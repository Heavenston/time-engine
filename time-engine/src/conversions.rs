mod tofrom;
pub use tofrom::*;

use nalgebra as na;
use glam::{ Affine2, Vec2 };

pub fn angle_to_complex(angle: f32) -> na::Unit<na::Complex<f32>> {
    na::Unit::new_normalize(na::Complex::from_polar(1., angle))
}

pub fn affine_to_isometry(affine: Affine2) -> na::Isometry2<f32> {
    let (scale, angle, translation) = affine.to_scale_angle_translation();
    assert!(scale == Vec2::ONE);

    na::Isometry2::from_parts(
        translation.to_na(),
        angle_to_complex(angle)
    )
}
