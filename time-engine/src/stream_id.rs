use std::{fmt::Display, sync::atomic::AtomicU64};


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StreamId {
    val: u64,
}

impl StreamId {
    pub fn new() -> Self {
        static STREAM_COUNTER: AtomicU64 = AtomicU64::new(0);
        Self {
            val: STREAM_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        }
    }
}

impl Default for StreamId {
    fn default() -> Self {
        Self::new()
    }
}

impl Display for StreamId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.val)
    }
}
