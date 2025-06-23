use super::*;

use std::{ convert::identity, fmt::Debug, ops::BitAnd, range::{ Bound, Range, RangeFrom, RangeFull, RangeTo } };

pub trait RangeHelpers: RangeBounds<f32> {
    fn is_finished(&self, time: f32) -> bool {
        match self.end_bound() {
            Bound::Included(&i) => i < time,
            Bound::Excluded(&i) => i <= time,
            Bound::Unbounded => false,
        }
    }

    fn is_later(&self, time: f32) -> bool {
        match self.start_bound() {
            Bound::Included(&i) => i > time,
            Bound::Excluded(&i) => i >= time,
            Bound::Unbounded => false,
        }
    }
}

impl<T: RangeBounds<f32>> RangeHelpers for T { }

pub trait RangeModifiers: Sized {
    fn map_bounds(self, start: impl FnOnce(f32) -> f32, end: impl FnOnce(f32) -> f32) -> Self;

    fn map_all_bounds(self, f: impl Fn(f32) -> f32) -> Self {
        self.map_bounds(&f, &f)
    }

    fn offset(self, offset: f32) -> Self {
        self.map_all_bounds(|t| t + offset)
    }

    fn at_least_from(self, from: Option<f32>) -> Self {
        if let Some(from) = from {
            self.map_all_bounds(|t| f32::max(t, from))
        }
        else {
            self
        }
    }

    fn with_end(self, to: Option<f32>) -> Self {
        self.map_bounds(identity, |end| to.unwrap_or(end))
    }

    fn at_most_to(self, to: Option<f32>) -> Self {
        if let Some(to) = to {
            self.map_all_bounds(|t| f32::min(t, to))
        }
        else {
            self
        }
    }

    fn with_start(self, from: Option<f32>) -> Self {
        self.map_bounds(|start| from.unwrap_or(start), identity)
    }
}

impl RangeModifiers for RangeFull {
    fn map_bounds(self, _f1: impl FnOnce(f32) -> f32, _f2: impl FnOnce(f32) -> f32) -> Self {
        self
    }
}

impl RangeModifiers for RangeFrom<f32> {
    fn map_bounds(self, f1: impl FnOnce(f32) -> f32, _f2: impl FnOnce(f32) -> f32) -> Self {
        f1(self.start)..
    }
}

impl RangeModifiers for RangeTo<f32> {
    fn map_bounds(self, _f1: impl FnOnce(f32) -> f32, f2: impl FnOnce(f32) -> f32) -> Self {
        ..f2(self.end)
    }
}

impl RangeModifiers for Range<f32> {
    fn map_bounds(self, f1: impl FnOnce(f32) -> f32, f2: impl FnOnce(f32) -> f32) -> Self {
        f1(self.start)..f2(self.end)
    }
}

#[derive(Clone, Copy, PartialEq)]
pub enum TimeRange {
    RangeFull(RangeFull),
    RangeFrom(RangeFrom<f32>),
    RangeTo(RangeTo<f32>),
    Range(Range<f32>),
}

impl TimeRange {
    pub fn new(start: Option<f32>, end: Option<f32>) -> Self {
        // debug_assert!(start.zip(end).is_none_or(|(a, b)| a <= b), "{start:?} should be <= to {end:?}");
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
}

impl RangeModifiers for TimeRange {
    fn map_bounds(self, f1: impl FnOnce(f32) -> f32, f2: impl FnOnce(f32) -> f32) -> Self {
        match self {
            TimeRange::RangeFull(range_full) => TimeRange::RangeFull(range_full.map_bounds(f1, f2)),
            TimeRange::RangeFrom(range_from) => TimeRange::RangeFrom(range_from.map_bounds(f1, f2)),
            TimeRange::RangeTo(range_to) => TimeRange::RangeTo(range_to.map_bounds(f1, f2)),
            TimeRange::Range(range) => TimeRange::Range(range.map_bounds(f1, f2)),
        }
    }

    fn at_least_from(self, from: Option<f32>) -> Self {
        self & Self::new(from, None)
    }

    fn with_end(self, to: Option<f32>) -> Self {
        Self::new(self.start(), to)
    }

    fn at_most_to(self, to: Option<f32>) -> Self {
        self & Self::new(None, to)
    }

    fn with_start(self, from: Option<f32>) -> Self {
        Self::new(from, self.end())
    }
}

impl Default for TimeRange {
    fn default() -> Self {
        Self::RangeFull(..)
    }
}

impl Debug for TimeRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TimeRange::RangeFull(range_full) => write!(f, "{:?}", range_full),
            TimeRange::RangeFrom(range_from) => write!(f, "{:?}", range_from),
            TimeRange::RangeTo(range_to) => write!(f, "{:?}", range_to),
            TimeRange::Range(range) => write!(f, "{:?}", range),
        }
    }
}

impl BitAnd for TimeRange {
    type Output = TimeRange;

    /// Returns a range for values that are in both ranges
    fn bitand(self, rhs: Self) -> Self::Output {
        let start = option_max(self.start(), rhs.start());
        let end = option_min(self.end(), rhs.end());
        // // Makes sure start <= end
        // let end = option_max(start, end);

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

impl From<RangeFull> for TimeRange {
    fn from(range_full: RangeFull) -> Self {
        Self::RangeFull(range_full)
    }
}

