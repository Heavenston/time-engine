mod world_state;
pub use world_state::*;
mod simulation;
pub use simulation::*;
mod polygon_utils;
pub use polygon_utils::*;
mod timeline_id;
pub use timeline_id::*;
mod physics_utils;
pub use physics_utils::*;

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
