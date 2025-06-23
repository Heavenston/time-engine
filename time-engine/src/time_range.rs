use super::*;

use std::{ fmt::Debug, ops::BitAnd, range::{ Range, RangeFrom, RangeFull, RangeTo } };

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

    pub fn is_finished(&self, time: f32) -> bool {
        self.end().is_some_and(|end| end <= time)
    }

    pub fn is_later(&self, time: f32) -> bool {
        self.start().is_some_and(|start| start > time)
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

    pub fn starting_from(&self, from: Option<f32>) -> Self {
        *self & Self::new(from, None)
    }

    pub fn up_to(&self, to: Option<f32>) -> Self {
        *self & Self::new(None, to)
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

