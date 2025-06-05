mod world_state;
pub use world_state::*;
mod simulation;
pub use simulation::*;

pub(crate) fn default<T: Default>() -> T {
    T::default()
}
