use nalgebra as na;
use glam;

/// Convert from a glam type into a nalgebra type.
pub trait GlamToNalgebra<Output> {
    fn to_na(self) -> Output;
}

/// Convert from a nalgebra type into a glam type.
pub trait NalgebraToGlam<Output> {
    fn to_gl(self) -> Output;
}

// ================================================================
// Vec2 <-> Vector2<f32>
impl GlamToNalgebra<na::Vector2<f32>> for glam::Vec2 {
    fn to_na(self) -> na::Vector2<f32> {
        na::Vector2::new(self.x, self.y)
    }
}
impl NalgebraToGlam<glam::Vec2> for na::Vector2<f32> {
    fn to_gl(self) -> glam::Vec2 {
        glam::Vec2::new(self.x, self.y)
    }
}

impl GlamToNalgebra<na::Point2<f32>> for glam::Vec2 {
    fn to_na(self) -> na::Point2<f32> {
        na::Point2::new(self.x, self.y)
    }
}
impl NalgebraToGlam<glam::Vec2> for na::Point2<f32> {
    fn to_gl(self) -> glam::Vec2 {
        glam::Vec2::new(self.x, self.y)
    }
}

// Vec3 <-> Vector3<f32>
impl GlamToNalgebra<na::Vector3<f32>> for glam::Vec3 {
    fn to_na(self) -> na::Vector3<f32> {
        na::Vector3::new(self.x, self.y, self.z)
    }
}
impl NalgebraToGlam<glam::Vec3> for na::Vector3<f32> {
    fn to_gl(self) -> glam::Vec3 {
        glam::Vec3::new(self.x, self.y, self.z)
    }
}

// Vec4 <-> Vector4<f32>
impl GlamToNalgebra<na::Vector4<f32>> for glam::Vec4 {
    fn to_na(self) -> na::Vector4<f32> {
        na::Vector4::new(self.x, self.y, self.z, self.w)
    }
}
impl NalgebraToGlam<glam::Vec4> for na::Vector4<f32> {
    fn to_gl(self) -> glam::Vec4 {
        glam::Vec4::new(self.x, self.y, self.z, self.w)
    }
}

// DVec2 <-> Vector2<f64>
impl GlamToNalgebra<na::Vector2<f64>> for glam::DVec2 {
    fn to_na(self) -> na::Vector2<f64> {
        na::Vector2::new(self.x, self.y)
    }
}
impl NalgebraToGlam<glam::DVec2> for na::Vector2<f64> {
    fn to_gl(self) -> glam::DVec2 {
        glam::DVec2::new(self.x, self.y)
    }
}

// DVec3 <-> Vector3<f64>
impl GlamToNalgebra<na::Vector3<f64>> for glam::DVec3 {
    fn to_na(self) -> na::Vector3<f64> {
        na::Vector3::new(self.x, self.y, self.z)
    }
}
impl NalgebraToGlam<glam::DVec3> for na::Vector3<f64> {
    fn to_gl(self) -> glam::DVec3 {
        glam::DVec3::new(self.x, self.y, self.z)
    }
}

// DVec4 <-> Vector4<f64>
impl GlamToNalgebra<na::Vector4<f64>> for glam::DVec4 {
    fn to_na(self) -> na::Vector4<f64> {
        na::Vector4::new(self.x, self.y, self.z, self.w)
    }
}
impl NalgebraToGlam<glam::DVec4> for na::Vector4<f64> {
    fn to_gl(self) -> glam::DVec4 {
        glam::DVec4::new(self.x, self.y, self.z, self.w)
    }
}

// ================================================================
// Mat2 <-> Matrix2<f32>
impl GlamToNalgebra<na::Matrix2<f32>> for glam::Mat2 {
    fn to_na(self) -> na::Matrix2<f32> {
        na::Matrix2::from_column_slice(&self.to_cols_array())
    }
}
impl NalgebraToGlam<glam::Mat2> for na::Matrix2<f32> {
    fn to_gl(self) -> glam::Mat2 {
        let c0 = glam::Vec2::new(self[(0,0)], self[(1,0)]);
        let c1 = glam::Vec2::new(self[(0,1)], self[(1,1)]);
        glam::Mat2::from_cols(c0, c1)
    }
}

// Mat3 <-> Matrix3<f32>
impl GlamToNalgebra<na::Matrix3<f32>> for glam::Mat3 {
    fn to_na(self) -> na::Matrix3<f32> {
        na::Matrix3::from_column_slice(&self.to_cols_array())
    }
}
impl NalgebraToGlam<glam::Mat3> for na::Matrix3<f32> {
    fn to_gl(self) -> glam::Mat3 {
        let c0 = glam::Vec3::new(self[(0,0)], self[(1,0)], self[(2,0)]);
        let c1 = glam::Vec3::new(self[(0,1)], self[(1,1)], self[(2,1)]);
        let c2 = glam::Vec3::new(self[(0,2)], self[(1,2)], self[(2,2)]);
        glam::Mat3::from_cols(c0, c1, c2)
    }
}

// Mat4 <-> Matrix4<f32>
impl GlamToNalgebra<na::Matrix4<f32>> for glam::Mat4 {
    fn to_na(self) -> na::Matrix4<f32> {
        na::Matrix4::from_column_slice(&self.to_cols_array())
    }
}
impl NalgebraToGlam<glam::Mat4> for na::Matrix4<f32> {
    fn to_gl(self) -> glam::Mat4 {
        let c0 = glam::Vec4::new(self[(0,0)], self[(1,0)], self[(2,0)], self[(3,0)]);
        let c1 = glam::Vec4::new(self[(0,1)], self[(1,1)], self[(2,1)], self[(3,1)]);
        let c2 = glam::Vec4::new(self[(0,2)], self[(1,2)], self[(2,2)], self[(3,2)]);
        let c3 = glam::Vec4::new(self[(0,3)], self[(1,3)], self[(2,3)], self[(3,3)]);
        glam::Mat4::from_cols(c0, c1, c2, c3)
    }
}

// DMat2 <-> Matrix2<f64>
impl GlamToNalgebra<na::Matrix2<f64>> for glam::DMat2 {
    fn to_na(self) -> na::Matrix2<f64> {
        na::Matrix2::from_column_slice(&self.to_cols_array())
    }
}
impl NalgebraToGlam<glam::DMat2> for na::Matrix2<f64> {
    fn to_gl(self) -> glam::DMat2 {
        let c0 = glam::DVec2::new(self[(0,0)], self[(1,0)]);
        let c1 = glam::DVec2::new(self[(0,1)], self[(1,1)]);
        glam::DMat2::from_cols(c0, c1)
    }
}

// DMat3 <-> Matrix3<f64>
impl GlamToNalgebra<na::Matrix3<f64>> for glam::DMat3 {
    fn to_na(self) -> na::Matrix3<f64> {
        na::Matrix3::from_column_slice(&self.to_cols_array())
    }
}
impl NalgebraToGlam<glam::DMat3> for na::Matrix3<f64> {
    fn to_gl(self) -> glam::DMat3 {
        let c0 = glam::DVec3::new(self[(0,0)], self[(1,0)], self[(2,0)]);
        let c1 = glam::DVec3::new(self[(0,1)], self[(1,1)], self[(2,1)]);
        let c2 = glam::DVec3::new(self[(0,2)], self[(1,2)], self[(2,2)]);
        glam::DMat3::from_cols(c0, c1, c2)
    }
}

// DMat4 <-> Matrix4<f64>
impl GlamToNalgebra<na::Matrix4<f64>> for glam::DMat4 {
    fn to_na(self) -> na::Matrix4<f64> {
        na::Matrix4::from_column_slice(&self.to_cols_array())
    }
}
impl NalgebraToGlam<glam::DMat4> for na::Matrix4<f64> {
    fn to_gl(self) -> glam::DMat4 {
        let c0 = glam::DVec4::new(self[(0,0)], self[(1,0)], self[(2,0)], self[(3,0)]);
        let c1 = glam::DVec4::new(self[(0,1)], self[(1,1)], self[(2,1)], self[(3,1)]);
        let c2 = glam::DVec4::new(self[(0,2)], self[(1,2)], self[(2,2)], self[(3,2)]);
        let c3 = glam::DVec4::new(self[(0,3)], self[(1,3)], self[(2,3)], self[(3,3)]);
        glam::DMat4::from_cols(c0, c1, c2, c3)
    }
}

// ================================================================
// Quat <-> Quaternion<f32>
impl GlamToNalgebra<na::Quaternion<f32>> for glam::Quat {
    fn to_na(self) -> na::Quaternion<f32> {
        na::Quaternion::new(self.w, self.x, self.y, self.z)
    }
}
impl NalgebraToGlam<glam::Quat> for na::Quaternion<f32> {
    fn to_gl(self) -> glam::Quat {
        let v = self.imag();
        glam::Quat::from_xyzw(v.x, v.y, v.z, self.scalar())
    }
}

// DQuat <-> Quaternion<f64>
impl GlamToNalgebra<na::Quaternion<f64>> for glam::DQuat {
    fn to_na(self) -> na::Quaternion<f64> {
        na::Quaternion::new(self.w, self.x, self.y, self.z)
    }
}
impl NalgebraToGlam<glam::DQuat> for na::Quaternion<f64> {
    fn to_gl(self) -> glam::DQuat {
        let v = self.imag();
        glam::DQuat::from_xyzw(v.x, v.y, v.z, self.scalar())
    }
}

// ================================================================
// Vec2 <-> Translation2<f32>
impl GlamToNalgebra<na::Translation2<f32>> for glam::Vec2 {
    fn to_na(self) -> na::Translation2<f32> {
        na::Translation2::new(self.x, self.y)
    }
}
impl NalgebraToGlam<glam::Vec2> for na::Translation2<f32> {
    fn to_gl(self) -> glam::Vec2 {
        let v = self.vector;
        glam::Vec2::new(v.x, v.y)
    }
}

// Vec3 <-> Translation3<f32>
impl GlamToNalgebra<na::Translation3<f32>> for glam::Vec3 {
    fn to_na(self) -> na::Translation3<f32> {
        na::Translation3::new(self.x, self.y, self.z)
    }
}
impl NalgebraToGlam<glam::Vec3> for na::Translation3<f32> {
    fn to_gl(self) -> glam::Vec3 {
        let v = self.vector;
        glam::Vec3::new(v.x, v.y, v.z)
    }
}

// DVec2 <-> Translation2<f64>
impl GlamToNalgebra<na::Translation2<f64>> for glam::DVec2 {
    fn to_na(self) -> na::Translation2<f64> {
        na::Translation2::new(self.x, self.y)
    }
}
impl NalgebraToGlam<glam::DVec2> for na::Translation2<f64> {
    fn to_gl(self) -> glam::DVec2 {
        let v = self.vector;
        glam::DVec2::new(v.x, v.y)
    }
}

// DVec3 <-> Translation3<f64>
impl GlamToNalgebra<na::Translation3<f64>> for glam::DVec3 {
    fn to_na(self) -> na::Translation3<f64> {
        na::Translation3::new(self.x, self.y, self.z)
    }
}
impl NalgebraToGlam<glam::DVec3> for na::Translation3<f64> {
    fn to_gl(self) -> glam::DVec3 {
        let v = self.vector;
        glam::DVec3::new(v.x, v.y, v.z)
    }
}
