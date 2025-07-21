#![feature(iter_collect_into)]
#![feature(generic_const_exprs)]
#![feature(try_blocks)]
#![feature(new_range_api)]
#![feature(new_range)]
#![feature(option_zip)]
#![feature(range_bounds_is_empty)]
#![feature(trait_alias)]
#![feature(associated_type_defaults)]

#![forbid(unsafe_code)]
#![expect(unused_imports)]
#![expect(incomplete_features)]
#![expect(dead_code)]

mod world_state;

pub use world_state::*;
mod simulation;
pub use simulation::*;
mod polygon_utils;
pub use polygon_utils::*;
mod timeline_id;
#[expect(deprecated)]
pub use timeline_id::*;
mod immutable_util;
pub(crate) use immutable_util::*;
mod conversions;
pub(crate) use conversions::*;
mod time_range;
pub use time_range::*;
pub mod delta_graph;
pub(crate) use delta_graph as dg;
pub mod rle_vec;
pub use rle_vec::*;
pub mod uid;
pub use uid::*;

use std::{range::RangeBounds, sync::Arc};

pub(crate) use typed_floats::tf32::*;

pub const fn n_elements_for_stack<T>() -> usize {
    std::mem::size_of::<T>() / std::mem::size_of::<Vec<T>>()
}

pub(crate) type SmallVec<T, const N: usize> = smallvec::SmallVec<T, N>;
pub(crate) type AutoSmallVec<T> = SmallVec<T, { n_elements_for_stack::<T>() }>;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortalVelocityDirection {
    NotMoving,
    IntoFront,
    IntoBack,
}

impl PortalVelocityDirection {
    pub fn is_not_moving(self) -> bool {
        matches!(self, Self::NotMoving)
    }

    pub fn is_into_front(self) -> bool {
        matches!(self, Self::IntoFront)
    }

    pub fn is_into_back(self) -> bool {
        matches!(self, Self::IntoBack)
    }
}

impl PartialEq<PortalDirection> for PortalVelocityDirection {
    fn eq(&self, other: &PortalDirection) -> bool {
        match self {
            PortalVelocityDirection::NotMoving => false,
            PortalVelocityDirection::IntoFront => other.is_front(),
            PortalVelocityDirection::IntoBack => other.is_back(),
        }
    }
}

pub(crate) fn default<T: Default>() -> T {
    T::default()
}

pub(crate) fn option_min(a: Option<f32>, b: Option<f32>) -> Option<f32> {
    a.zip_with(b, f32::min).or(a).or(b)
}

pub(crate) fn option_max(a: Option<f32>, b: Option<f32>) -> Option<f32> {
    a.zip_with(b, f32::max).or(a).or(b)
}

pub(crate) fn concat_arrays<T, const A: usize, const B: usize>(
    a: [T; A], b: [T; B]
) -> [T; A+B]
where
    T: Default,
{
    let mut ary: [T; A+B] = std::array::from_fn(|_| Default::default());
    for (idx, val) in a.into_iter().chain(b.into_iter()).enumerate() {
        ary[idx] = val;
    }
    ary
}

// FIXME: For some types where cloning is expensive being able to return
//        mapped Arc would probably be faster (https://crates.io/crates/mappable-rc)
pub(crate) fn cloned_arc_slice_iter<A: ?Sized, T, U>(
    ptr: Arc<A>,
    getter: impl Fn(&A) -> &[T],
    mapper: impl Fn(&T) -> U,
) -> impl Iterator<Item = U> {
    (0..getter(&*ptr).len()).into_iter()
        .map(move |i| mapper(&getter(&*ptr)[i]))
}
