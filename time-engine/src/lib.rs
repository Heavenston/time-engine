mod world_state;
pub use world_state::*;
mod simulation;
pub use simulation::*;
mod polygon_utils;
pub use polygon_utils::*;
mod timeline_id;
pub use timeline_id::*;

pub(crate) fn default<T: Default>() -> T {
    T::default()
}
