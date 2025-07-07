use std::ops::Index;
use std::sync::atomic::AtomicU64;

use super::*;

#[cfg(debug_assertions)]
static TIMESTAMP_LIST_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Timestamp {
    #[cfg(debug_assertions)]
    list_id: u64,
    idx: u32,
}

impl Timestamp {
    pub fn idx(self) -> usize {
        self.idx.try_into().expect("No overflow")
    }

    fn from_idx(idx: usize, list_id: u64) -> Self {
        Self {
            #[cfg(debug_assertions)]
            list_id,
            idx: idx.try_into().expect("No overflow"),
        }
    }

    fn next(self) -> Self {
        Self {
            #[cfg(debug_assertions)]
            list_id: self.list_id,
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
    #[cfg(debug_assertions)]
    id: u64,
    datas: Vec<TimestampData>,
}

impl TimestampList {
    pub fn new() -> Self {
        Self {
            #[cfg(debug_assertions)]
            id: TIMESTAMP_LIST_ID_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
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
        Timestamp::from_idx(0, self.id)
    }

    pub fn first(&self) -> &TimestampData {
        &self[self.first_timestamp()]
    }

    pub fn last_timestamp(&self) -> Timestamp {
        assert!(self.datas.len() > 0);
        Timestamp {
            #[cfg(debug_assertions)]
            list_id: self.id,
            idx: (self.datas.len() - 1).try_into().expect("No overflow"),
        }
    }

    pub fn last(&self) -> &TimestampData {
        &self[self.last_timestamp()]
    }

    pub fn len(&self) -> usize {
        self.datas.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = (Timestamp, &'_ TimestampData)> {
        (0..self.len()).into_iter()
            .map(|idx| (Timestamp::from_idx(idx, self.id), &self.datas[idx]))
    }

    pub fn push(&mut self, delta: TimestampDelta) -> Timestamp {
        let ts = Timestamp {
            #[cfg(debug_assertions)]
            list_id: self.id,
            idx: self.datas.len().try_into().expect("No overflow"),
        };

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
        #[cfg(debug_assertions)]
        debug_assert_eq!(index.list_id, self.id, "Using timestamp from another list");
        &self.datas[index.idx()]
    }
}
