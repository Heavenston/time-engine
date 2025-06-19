#![feature(iter_collect_into)]
#![feature(generic_const_exprs)]
#![feature(try_blocks)]

#![allow(incomplete_features)]

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

const fn n_elements_for_stack<T>() -> usize {
    std::mem::size_of::<T>() / std::mem::size_of::<Vec<T>>()
}

pub(crate) type TinyVec<T, const N: usize> = smallvec::SmallVec<T, N>;
pub(crate) type AutoSmallVec<T> = smallvec::SmallVec<T, { n_elements_for_stack::<T>() }>;

pub(crate) trait AutoSmallVecIterExt<T> {
    fn collect_smallvec(self) -> AutoSmallVec<T>;
}

impl<T, I: Iterator<Item = T>> AutoSmallVecIterExt<T> for I {
    fn collect_smallvec(self) -> AutoSmallVec<T> {
        self.collect()
    }
}

// pub const DEFAULT_EPSILON: f32 = parry2d::math::DEFAULT_EPSILON;
pub const DEFAULT_EPSILON: f32 = 0.001;

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
