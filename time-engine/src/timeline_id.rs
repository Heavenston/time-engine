use parking_lot::RwLock;

#[derive(Debug, Clone, Copy)]
pub struct TimelineId {
    idx: usize,
    start: f32,
}

impl TimelineId {
    pub fn start(&self) -> f32 {
        self.start
    }
}

impl PartialEq for TimelineId {
    fn eq(&self, other: &Self) -> bool {
        self.idx == other.idx
    }
}

impl Eq for TimelineId {}

impl PartialOrd for TimelineId {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TimelineId {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.idx.cmp(&other.idx)
    }
}

impl std::hash::Hash for TimelineId {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.idx.hash(state);
    }
}

#[derive(Debug, Default)]
pub struct TimelineMultiverse {
    /// For timelines with idx > 0, timeline_parents[idx-1] gives its parent
    timeline_parents: RwLock<Vec<usize>>,
    timeline_starts: RwLock<Vec<f32>>,
}

impl TimelineMultiverse {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn root(&self) -> TimelineId {
        TimelineId {
            idx: 0,
            start: 0.,
        }
    }

    fn timeline_from_idx(&self, idx: usize) -> TimelineId {
        TimelineId {
            idx,
            start: if idx == 0 { 0. } else { self.timeline_starts.read()[idx - 1] },
        }
    }

    pub fn is_root(&self, id: TimelineId) -> bool {
        id.idx == 0
    }

    pub fn try_parent_of(&self, of: TimelineId) -> Option<TimelineId> {
        if of.idx == 0 {
            None
        }
        else {
            Some(self.timeline_from_idx(self.timeline_parents.read()[of.idx - 1]))
        }
    }

    /// If the given timeline isn't the root, weturns its parent,
    /// otherwise returns the root
    pub fn parent_of(&self, of: TimelineId) -> TimelineId {
        self.try_parent_of(of).unwrap_or(of)
    }

    pub fn create_children(&self, of: TimelineId, start: f32) -> TimelineId {
        let mut parents = self.timeline_parents.write();
        parents.push(of.idx);
        self.timeline_starts.write().push(start);
        TimelineId { idx: parents.len(), start }
    }

    /// Returns true if parent is a parent or is equal to child
    pub fn is_parent(&self, parent: TimelineId, mut child: TimelineId) -> bool {
        while child != parent {
            let Some(new_child) = self.try_parent_of(child)
            else {
                return false;
            };
            child = new_child;
        }

        child == parent
    }

    pub fn is_related(&self, lhs: TimelineId, rhs: TimelineId) -> bool {
        self.is_parent(lhs, rhs) || self.is_parent(rhs, lhs)
    }
}
