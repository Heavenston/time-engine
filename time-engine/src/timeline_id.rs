use std::iter::FusedIterator;
use arbitrary_int::u1;

type TimelineIdInner = u64;

/// Represent a timeline path by packing a u1 array into a number with
/// a leading 1 bit as terminator
#[derive(Clone, Hash, PartialEq, Eq)]
pub struct TimelineId(TimelineIdInner);

impl TimelineId {
    pub const MAX_CAPACITY: u32 = TimelineIdInner::BITS - 1; // 63

    #[inline]
    pub fn new() -> Self {
        Self(1) // Just the terminator bit
    }

    #[inline]
    pub const fn capacity(&self) -> u32 {
        Self::MAX_CAPACITY
    }

    #[inline]
    const fn mark_bit_position(&self) -> u32 {
        debug_assert!(
            self.0.leading_zeros() < TimelineIdInner::BITS,
            "invalid inner value"
        );
        TimelineIdInner::BITS - self.0.leading_zeros() - 1
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.0 == 1
    }

    #[inline]
    pub const fn len(&self) -> u32 {
        self.mark_bit_position()
    }

    #[inline]
    pub fn push(&mut self, bit: u1) {
        assert!(self.len() < Self::MAX_CAPACITY);
        self.0 = (self.0 << 1) | TimelineIdInner::from(bit.value());
    }

    #[inline]
    pub const fn with_push(self, bit: u1) -> Self {
        Self((self.0 << 1) | (bit.value() as TimelineIdInner))
    }

    #[inline]
    pub fn peek(&self) -> Option<u1> {
        if self.is_empty() {
            return None;
        }
        Some(u1::new((self.0 & 1) as u8))
    }

    pub fn pop(&mut self) -> Option<u1> {
        if self.is_empty() {
            return None;
        }

        let bit = u1::new((self.0 & 1) as u8);
        self.0 >>= 1;
        Some(bit)
    }

    #[inline]
    pub fn parent(&self) -> Option<Self> {
        if self.is_empty() {
            return None;
        }
        Some(Self(self.0 >> 1))
    }

    /// Returns an iterator over all parents, from the deepest to the root
    /// excluding self
    #[inline]
    pub fn parents(&self) -> impl Iterator<Item = Self> {
        let mut current = self.clone();
        std::iter::from_fn(move || {
            current = current.parent()?;
            Some(current.clone())
        })
    }

    /// Returns an iterator over all timeline ids possible with the given length
    pub fn all_iter(length: u32) -> impl DoubleEndedIterator<Item = Self> {
        (0..(1 << length)).map(move |i| Self(i | (1 << length)))
    }

    /// Returns a number representation of the current timeline, unique
    /// for all timelines of the same length
    pub fn index(&self) -> usize {
        let marker_bit = self.mark_bit_position();
        usize::try_from(self.0 & !(TimelineIdInner::MAX << marker_bit)).unwrap()
    }

    pub fn from_index(index: TimelineIdInner, length: u32) -> Self {
        if length > Self::MAX_CAPACITY {
            panic!("Length higher than capacity");
        }
        Self(index | (1 << length))
    }

    /// Return a new TimelineId with only the first [length] elements of self
    pub fn take(&self, length: u32) -> Self {
        assert!(length <= self.len());
        let to_remove = self.len() - length;
        Self(self.0 >> to_remove)
    }

    /// Return a new TimelineId with the first [length_to_remove] elements removed
    pub fn reparent(self, length_to_remove: u32) -> Self {
        assert!(length_to_remove <= self.len());
        let new_length = self.len() - length_to_remove;
        let mask = (1 << new_length) - 1;
        Self((self.0 & mask) | (1 << new_length))
    }

    #[inline]
    pub fn extend(&mut self, other: &Self) {
        assert!(Self::MAX_CAPACITY >= self.len() + other.len());
        self.0 = (self.0 << other.len()) | other.index() as TimelineIdInner;
    }

    #[inline]
    pub fn extended(mut self, other: &Self) -> Self {
        self.extend(other);
        self
    }

    #[inline]
    pub fn is_prefix_of(&self, other: &Self) -> bool {
        let self_bit = self.mark_bit_position();
        let other_bit = other.mark_bit_position();
        if self_bit > other_bit {
            return false;
        }

        let smaller_other = other.0 >> (other_bit - self_bit);
        smaller_other == self.0
    }

    #[inline]
    pub const fn components() -> [u1; 2] {
        [u1::new(0), u1::new(1)]
    }

    pub fn children(&self) -> [Self; 2] {
        Self::components().map(|bit| self.clone().with_push(bit))
    }
}

impl Default for TimelineId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for TimelineId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("TimelineId(1")?;
        for bit in self.into_iter() {
            write!(f, "_{}", bit.value())?;
        }
        f.write_str(")")?;
        Ok(())
    }
}

impl IntoIterator for &TimelineId {
    type Item = u1;
    type IntoIter = TimelineIdIterator;

    fn into_iter(self) -> Self::IntoIter {
        TimelineIdIterator {
            timeline: self.clone(),
        }
    }
}

pub struct TimelineIdIterator {
    timeline: TimelineId,
}

impl TimelineIdIterator {
    pub fn new(timeline: TimelineId) -> Self {
        Self { timeline }
    }
}

impl From<TimelineId> for TimelineIdIterator {
    fn from(value: TimelineId) -> Self {
        Self::new(value)
    }
}

impl Iterator for TimelineIdIterator {
    type Item = u1;

    fn next(&mut self) -> Option<Self::Item> {
        // Pop from the back (most significant bit first)
        if self.timeline.is_empty() {
            return None;
        }

        let mbp = self.timeline.mark_bit_position();
        let bit = u1::new(((self.timeline.0 >> (mbp - 1)) & 1) as u8);

        // Remove the bit and adjust marker
        self.timeline.0 &= (1 << (mbp - 1)) - 1;
        if mbp > 1 {
            self.timeline.0 |= 1 << (mbp - 1);
        } else {
            self.timeline.0 = 1;
        }

        Some(bit)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.timeline.len() as usize;
        (len, Some(len))
    }
}

impl DoubleEndedIterator for TimelineIdIterator {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.timeline.pop()
    }
}

impl ExactSizeIterator for TimelineIdIterator {}
impl FusedIterator for TimelineIdIterator {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let mut timeline = TimelineId::new();
        assert!(timeline.is_empty());
        assert_eq!(timeline.len(), 0);

        timeline.push(u1::new(1));
        assert!(!timeline.is_empty());
        assert_eq!(timeline.len(), 1);
        assert_eq!(timeline.peek(), Some(u1::new(1)));

        timeline.push(u1::new(0));
        assert_eq!(timeline.len(), 2);
        assert_eq!(timeline.peek(), Some(u1::new(0)));

        assert_eq!(timeline.pop(), Some(u1::new(0)));
        assert_eq!(timeline.len(), 1);
        assert_eq!(timeline.peek(), Some(u1::new(1)));

        assert_eq!(timeline.pop(), Some(u1::new(1)));
        assert!(timeline.is_empty());
        assert_eq!(timeline.pop(), None);
    }

    #[test]
    fn test_with_push() {
        let timeline = TimelineId::new()
            .with_push(u1::new(1))
            .with_push(u1::new(0))
            .with_push(u1::new(1));

        assert_eq!(timeline.len(), 3);
        let mut iter = timeline.into_iter();
        assert_eq!(iter.next(), Some(u1::new(1)));
        assert_eq!(iter.next(), Some(u1::new(0)));
        assert_eq!(iter.next(), Some(u1::new(1)));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_parent() {
        let timeline = TimelineId::new()
            .with_push(u1::new(1))
            .with_push(u1::new(0))
            .with_push(u1::new(1));

        let parent = timeline.parent().unwrap();
        assert_eq!(parent.len(), 2);

        let grandparent = parent.parent().unwrap();
        assert_eq!(grandparent.len(), 1);

        let root = grandparent.parent().unwrap();
        assert!(root.is_empty());
        assert_eq!(root.parent(), None);
    }

    #[test]
    fn test_is_prefix_of() {
        let short = TimelineId::new().with_push(u1::new(1)).with_push(u1::new(0));
        let long = short.clone().with_push(u1::new(1)).with_push(u1::new(0));

        assert!(short.is_prefix_of(&long));
        assert!(!long.is_prefix_of(&short));
        assert!(short.is_prefix_of(&short));
    }

    #[test]
    fn test_children() {
        let timeline = TimelineId::new().with_push(u1::new(1));
        let [c0, c1] = timeline.children();
        
        assert_eq!(c0.len(), 2);
        assert_eq!(c1.len(), 2);
        assert_eq!(c0.peek(), Some(u1::new(0)));
        assert_eq!(c1.peek(), Some(u1::new(1)));
    }
}
