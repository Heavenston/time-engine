#![feature(iter_collect_into)]
#![feature(generic_const_exprs)]
#![feature(try_blocks)]
#![feature(new_range_api)]
#![feature(new_range)]
#![feature(option_zip)]

#![expect(incomplete_features)]
#![expect(dead_code)]

mod world_state;
use std::{ops::BitAnd, range::{Range, RangeFrom, RangeTo, RangeFull, RangeBounds}};

pub use world_state::*;
mod simulation;
pub use simulation::*;
mod polygon_utils;
pub use polygon_utils::*;
mod timeline_id;
pub use timeline_id::*;
mod immutable_util;
pub(crate) use immutable_util::*;
mod conversions;
pub(crate) use conversions::*;

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

pub(crate) fn default<T: Default>() -> T {
    T::default()
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TimeRange {
    RangeFull(RangeFull),
    RangeFrom(RangeFrom<f32>),
    RangeTo(RangeTo<f32>),
    Range(Range<f32>),
}

impl TimeRange {
    pub fn new(start: Option<f32>, end: Option<f32>) -> Self {
        match (start, end) {
            (None, None) => Self::RangeFull(..),
            (None, Some(end)) => Self::RangeTo(..end),
            (Some(start), None) => Self::RangeFrom(start..),
            (Some(start), Some(end)) => Self::Range(start..end),
        }
    }

    pub fn start(&self) -> Option<f32> {
        match self {
            TimeRange::RangeFrom(range_from) => Some(range_from.start),
            TimeRange::Range(range) => Some(range.start),
            _ => None,
        }
    }

    pub fn end(&self) -> Option<f32> {
        match self {
            TimeRange::RangeTo(range_to) => Some(range_to.end),
            TimeRange::Range(range) => Some(range.end),
            _ => None,
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            TimeRange::RangeFull(_) => false,
            TimeRange::RangeFrom(_) => false,
            TimeRange::RangeTo(_) => false,
            TimeRange::Range(r) => r.is_empty(),
        }
    }

    pub fn offset(&self, offset: f32) -> Self {
        Self::new(self.start().map(|start| start + offset), self.end().map(|end| end + offset))
    }

    pub fn from(&self, from: f32) -> Self {
        Self::new(Some(self.start().unwrap_or(from).max(from)), self.end())
    }

    pub fn to(&self, to: f32) -> Self {
        Self::new(self.start(), Some(self.end().unwrap_or(to).min(to)))
    }
}

impl Default for TimeRange {
    fn default() -> Self {
        Self::RangeFull(..)
    }
}

impl BitAnd for TimeRange {
    type Output = TimeRange;

    /// Returns a range for values that are in both ranges
    fn bitand(self, rhs: Self) -> Self::Output {
        let start = self.start()
            .zip_with(rhs.start(), f32::max)
            .or(self.start()).or(rhs.start());
        let end = self.end()
            .zip_with(rhs.end(), f32::max)
            .or(self.end()).or(rhs.end());
        // Makes sure start <= end
        let start = end.zip_with(start, f32::max).or(start);

        Self::new(start, end)
    }
}

impl RangeBounds<f32> for TimeRange {
    fn start_bound(&self) -> std::ops::Bound<&f32> {
        match self {
            TimeRange::RangeFull(r) => r.start_bound(),
            TimeRange::RangeFrom(r) => r.start_bound(),
            TimeRange::RangeTo(r) => r.start_bound(),
            TimeRange::Range(r) => r.start_bound(),
        }
    }

    fn end_bound(&self) -> std::ops::Bound<&f32> {
        match self {
            TimeRange::RangeFull(r) => r.end_bound(),
            TimeRange::RangeFrom(r) => r.end_bound(),
            TimeRange::RangeTo(r) => r.end_bound(),
            TimeRange::Range(r) => r.end_bound(),
        }
    }
}

impl From<Range<f32>> for TimeRange {
    fn from(range: Range<f32>) -> Self {
        Self::Range(range)
    }
}

impl From<RangeFrom<f32>> for TimeRange {
    fn from(range_from: RangeFrom<f32>) -> Self {
        Self::RangeFrom(range_from)
    }
}
