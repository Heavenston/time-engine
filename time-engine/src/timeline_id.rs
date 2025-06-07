use parking_lot::RwLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TimelineId {
    idx: usize,
}

#[derive(Debug, Default)]
pub struct TimelineMultiverse {
    /// For timelines with idx > 0, timeline_parents[idx-1] gives its parent
    timeline_parents: RwLock<Vec<usize>>,
}

impl TimelineMultiverse {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn root(&self) -> TimelineId {
        TimelineId {
            idx: 0
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
            Some(TimelineId { idx: self.timeline_parents.read()[of.idx - 1] })
        }
    }

    /// If the given timeline isn't the root, weturns its parent,
    /// otherwise returns the root
    pub fn parent_of(&self, of: TimelineId) -> TimelineId {
        self.try_parent_of(of).unwrap_or(of)
    }

    pub fn create_children(&self, of: TimelineId) -> TimelineId {
        let mut parents = self.timeline_parents.write();
        parents.push(of.idx);
        TimelineId { idx: parents.len() }
    }

    /// Returns true if parent is a parent or is equal to child
    pub fn is_parent(&self, parent: TimelineId, mut child: TimelineId) -> bool {
        while child != parent {
            let Some(new_child) = self.try_parent_of(child)
            else { return false; };
            child = new_child;
        }

        child == parent
    }
}
