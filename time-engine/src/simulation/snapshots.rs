use super::*;

use std::ops::{Index, IndexMut};

use glam::Vec2;
use itertools::Itertools;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimSnapshot {
    /// Index of the sphere in the world state corresponding to this sphere
    /// never changes over time
    pub original_idx: usize,
    /// Radius of the sphere
    /// never changes over time
    pub radius: f32,
    /// Time as seen by the observer
    pub time: f32,
    pub timeline: TimelineId,
    /// Age, the time as seen by the sphere
    pub age: f32,
    pub pos: Vec2,
    pub vel: Vec2,
}

impl SimSnapshot {
    pub fn advanced(self, new_time: f32, new_pos: Vec2, new_vel: Vec2) -> Self {
        assert!(self.time <= new_time);
        let dt = new_time - self.time;
        Self {
            time: new_time,
            age: self.age + dt,
            pos: new_pos,
            vel: new_vel,
            ..self
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct SimSnapshotLink {
    idx: usize,
}

#[derive(Debug, Clone)]
pub struct SimSnapshotNode {
    pub snapshot: SimSnapshot,
    /// Link to the snapshot corresponding to the latest snapshot of that sphere
    /// that happens before this one in local time
    pub age_previous: Option<SimSnapshotLink>,
    pub age_children: Vec<SimSnapshotLink>,
}

impl From<SimSnapshot> for SimSnapshotNode {
    fn from(snapshot: SimSnapshot) -> Self {
        Self {
            snapshot,
            age_previous: default(),
            age_children: default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SimSnapshotContainer {
    nodes: Vec<SimSnapshotNode>,
}

impl SimSnapshotContainer {
    pub fn new() -> Self {
        Self {
            nodes: default(),
        }
    }

    pub fn get_node(&self, link: SimSnapshotLink) -> Option<&SimSnapshotNode> {
        self.nodes.get(link.idx)
    }

    pub fn get(&self, link: SimSnapshotLink) -> Option<&SimSnapshot> {
        self.nodes.get(link.idx).map(|node| &node.snapshot)
    }

    pub fn get_mut(&mut self, link: SimSnapshotLink) -> Option<&mut SimSnapshot> {
        self.nodes.get_mut(link.idx).map(|node| &mut node.snapshot)
    }

    pub fn age_previous(&self, link: SimSnapshotLink) -> Option<SimSnapshotLink> {
        self.get_node(link)?.age_previous
    }

    pub fn age_children(&self, link: SimSnapshotLink) -> &[SimSnapshotLink] {
        self.get_node(link)
            .map(|node| node.age_children.as_slice())
            .unwrap_or(&[])
    }

    pub fn insert(&mut self, multiverse: &TimelineMultiverse, snapshot: SimSnapshot, age_previous: Option<SimSnapshotLink>) -> SimSnapshotLink {
        let link = SimSnapshotLink { idx: self.nodes.len() };

        self.nodes.push(SimSnapshotNode {
            age_previous,
            ..SimSnapshotNode::from(snapshot)
        });

        if let Some(age_previous) = age_previous {
            let age_parent = &mut self.nodes[age_previous.idx];
            debug_assert_eq!(age_parent.snapshot.original_idx, snapshot.original_idx);
            let distance = multiverse.distance(age_parent.snapshot.timeline, snapshot.timeline);
            debug_assert!(distance == Some(0) || distance == Some(1));
            self.nodes[age_previous.idx].age_children.push(link);
            debug_assert!(
                self.nodes[age_previous.idx].age_children.iter()
                    .map(|&child_link| self[child_link].timeline)
                    .all_unique(),
                "Cannot branch multiple time for the same timeline!"
            );
        }

        link
    }

    /// Returns an iterator over all 'parents' of the given node, from the
    /// closest to the fursthest
    /// Starts with the given node itself
    pub fn iter_ancestry<'a>(&'a self, link: SimSnapshotLink) -> AncestryIterator<'a> {
        AncestryIterator::new(self, Some(link))
    }
}

impl Index<SimSnapshotLink> for SimSnapshotContainer {
    type Output = SimSnapshot;

    fn index(&self, index: SimSnapshotLink) -> &Self::Output {
        &self.nodes[index.idx].snapshot
    }
}

impl IndexMut<SimSnapshotLink> for SimSnapshotContainer {
    fn index_mut(&mut self, index: SimSnapshotLink) -> &mut Self::Output {
        &mut self.nodes[index.idx].snapshot
    }
}

impl Default for SimSnapshotContainer {
    fn default() -> Self {
        Self::new()
    }
}

pub struct AncestryIterator<'a> {
    container: &'a SimSnapshotContainer,
    latest: Option<SimSnapshotLink>,
}

impl<'a> AncestryIterator<'a> {
    pub fn new(container: &'a SimSnapshotContainer, latest: Option<SimSnapshotLink>) -> Self {
        Self {
            container,
            latest,
        }
    }
}

impl<'a> Iterator for AncestryIterator<'a> {
    type Item = (SimSnapshotLink, &'a SimSnapshotNode);

    fn next(&mut self) -> Option<Self::Item> {
        let link = self.latest?;
        let node = self.container.get_node(link)?;
        self.latest = node.age_previous;
        Some((link, node))
    }
}
