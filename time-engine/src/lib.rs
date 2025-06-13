#![feature(iter_collect_into)]

mod world_state;
pub use world_state::*;
mod simulation;
pub use simulation::*;
mod polygon_utils;
pub use polygon_utils::*;
mod timeline_id;
pub use timeline_id::*;
mod immutable_util;
pub(crate) use immutable_util::*;

// pub const DEFAULT_EPSILON: f32 = parry2d::math::DEFAULT_EPSILON;
pub const DEFAULT_EPSILON: f32 = parry2d::math::DEFAULT_EPSILON;

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortalDirection {
    #[default]
    Front,
    Back,
}

impl PortalDirection {
    pub fn is_front(self) -> bool {
        self == Self::Front
    }

    pub fn is_back(self) -> bool {
        self == Self::Back
    }

    pub fn swap(self) -> Self {
        match self {
            PortalDirection::Front => PortalDirection::Back,
            PortalDirection::Back => PortalDirection::Front,
        }
    }
}

pub(crate) fn default<T: Default>() -> T {
    T::default()
}
