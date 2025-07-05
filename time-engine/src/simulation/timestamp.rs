use std::ops::Index;

use super::*;

// TODO: Implement in debug mode a check that each Timestamp belongs to the correct
// TimestampList on every usage with a global id counter

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Timestamp {
    idx: u32,
}

impl Timestamp {
    pub fn idx(self) -> usize {
        self.idx.try_into().expect("No overflow")
    }

    fn from_idx(idx: usize) -> Self {
        Self {
            idx: idx.try_into().expect("No overflow"),
        }
    }

    fn next(self) -> Self {
        Self {
            idx: self.idx + 1,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TimestampDelta {
    pub delta_t: Positive,
}

#[derive(Debug, Clone, Copy)]
pub struct TimestampData {
    pub delta: TimestampDelta,
    pub running_sum: Positive,
}

#[derive(Debug, Clone)]
pub struct TimestampList {
    datas: Vec<TimestampData>,
}

impl TimestampList {
    pub fn new() -> Self {
        Self {
            datas: vec![TimestampData {
                delta: TimestampDelta {
                    delta_t: Positive::new(0.).expect("positive"),
                },
                running_sum: Positive::new(0.).expect("positive"),
            }]
        }
    }

    pub fn first_timestamp(&self) -> Timestamp {
        assert!(self.datas.len() > 0);
        Timestamp { idx: 0 }
    }

    pub fn first(&self) -> &TimestampData {
        &self[self.first_timestamp()]
    }

    pub fn last_timestamp(&self) -> Timestamp {
        assert!(self.datas.len() > 0);
        Timestamp { idx: (self.datas.len() - 1).try_into().expect("No overflow")}
    }

    pub fn last(&self) -> &TimestampData {
        &self[self.last_timestamp()]
    }

    pub fn len(&self) -> usize {
        self.datas.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = (Timestamp, &'_ TimestampData)> {
        (0..self.len()).into_iter()
            .map(|idx| (Timestamp::from_idx(idx), &self.datas[idx]))
    }

    pub fn push(&mut self, delta: TimestampDelta) -> Timestamp {
        let ts = Timestamp { idx: self.datas.len().try_into().expect("No overflow") };

        let previous_sum = self.datas.last().expect("Not empty").running_sum;
        self.datas.push(TimestampData {
            delta,
            running_sum: previous_sum + delta.delta_t,
        });
        ts
    }
}

impl Default for TimestampList {
    fn default() -> Self {
        Self::new()
    }
}

impl Index<Timestamp> for TimestampList {
    type Output = TimestampData;

    fn index(&self, index: Timestamp) -> &Self::Output {
        &self.datas[index.idx()]
    }
}
