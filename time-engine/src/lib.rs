mod world_state;
pub use world_state::*;
mod simulation;
pub use simulation::*;
mod polygon_utils;
pub use polygon_utils::*;

pub(crate) fn default<T: Default>() -> T {
    T::default()
}
