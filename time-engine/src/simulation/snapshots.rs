use super::*;

use std::ops::{ Index, IndexMut };

use glam::{ Affine2, Vec2 };
use itertools::Itertools;

#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct SimPortalTraversal {
    pub portal_in_idx: usize,
    pub portal_in: SimPortal,
    pub portal_out_idx: usize,
    pub portal_out: SimPortal,
    pub end_age: f32,
    pub direction: PortalDirection,
}

impl SimPortalTraversal {
    /// Finding when a sphere wont intersect with the portal anymore
    /// nothing in parry2d can help so we hardcode a sphere-line solution
    pub fn compute_end_dt(sphere_radius: f32, sphere_pos: Vec2, sphere_vel: Vec2, portal_transform: Affine2) -> f32 {
        let inv_portal_trans = portal_transform.inverse();
        let rel_vel = inv_portal_trans.transform_vector2(sphere_vel);
        let rel_pos = inv_portal_trans.transform_point2(sphere_pos);

        let t0 = (sphere_radius - rel_pos.x) / rel_vel.x;
        let t1 = (-sphere_radius - rel_pos.x) / rel_vel.x;

        // Not sure on the math of alaways taking the max but anyway it seems to work
        t0.max(t1)
    }

    pub fn time_offset(&self) -> f32 {
        self.portal_out.time_offset - self.portal_in.time_offset
    }

    pub fn swap(self) -> Self {
        Self {
            portal_in_idx: self.portal_out_idx,
            portal_in: self.portal_out,
            portal_out_idx: self.portal_in_idx,
            portal_out: self.portal_in,
            end_age: self.end_age,
            direction: self.direction.swap(),
        }
    }

    pub fn transform_point(&self, point: Vec2) -> Vec2 {
        let inv_input = self.portal_in.transform.inverse();
        self.portal_out.transform.transform_point2(inv_input.transform_point2(point))
    }

    pub fn transform_vector(&self, vector: Vec2) -> Vec2 {
        let inv_input = self.portal_in.transform.inverse();
        self.portal_out.transform.transform_vector2(inv_input.transform_vector2(vector))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GhostInfo {
    pub snapshot: SimSnapshot,
    pub offset: f32,
    pub index: usize,
    pub expiration_time: f32,
}

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
    pub timeline_id: TimelineId,
    /// Age, the time as seen by the sphere
    pub age: f32,
    pub pos: Vec2,
    pub vel: Vec2,
    /// List of portals this sphere is traversing
    /// Max length set to one because i dont think it will work with more than
    /// one for now...
    pub portal_traversals: tinyvec::ArrayVec<[SimPortalTraversal; 1]>,
}

impl SimSnapshot {
    pub fn advanced(self, new_time: f32, new_pos: Vec2, new_vel: Vec2) -> Self {
        assert!(self.time <= new_time);
        let dt = new_time - self.time;
        let new_age = self.age + dt;
        let changed_vel = new_vel != self.vel;
        Self {
            time: new_time,
            age: new_age,
            pos: new_pos,
            vel: new_vel,
            portal_traversals: self.portal_traversals.into_iter()
                .map(|traversal| {
                    if !changed_vel { return traversal; }

                    SimPortalTraversal {
                        end_age: SimPortalTraversal::compute_end_dt(self.radius, new_pos, new_vel, traversal.portal_in.transform),
                        ..traversal
                    }
                })
                .filter(|traversal| traversal.end_age > new_age)
                .collect(),
            ..self
        }
    }

    pub fn extrapolate(self, new_time: f32) -> Self {
        let dt = new_time - self.time;
        self.advanced(new_time, self.pos + self.vel * dt, self.vel)
    }

    pub fn with_portal_traversals(mut self, traversal: impl IntoIterator<Item = SimPortalTraversal>) -> Self {
        // FIXME: Only kinda obvious because there is only one maximum traversal
        let new: tinyvec::ArrayVec<[SimPortalTraversal; 1]> = traversal.into_iter().collect();
        if new.len() > 0 {
            self.portal_traversals = new;
        }
        self
    }

    pub fn offset_time(self, dt: f32) -> Self {
        Self {
            time: self.time + dt,
            ..self
        }
    }

    pub fn replace_timeline(self, new_timeline_id: TimelineId) -> Self {
        Self {
            timeline_id: new_timeline_id,
            ..self
        }
    }

    /// Returns true if this snapshot is behind a portal
    /// in other words if this snapshot is coming -out of the portals listend
    /// in portal_traversal
    pub fn is_ghost(&self) -> bool {
        // FIXME: probably incorrect for >1 portal traversals
        // (asserts just as reminder of this fact)
        assert!(self.portal_traversals.len() <= 1);

        self.portal_traversals.iter()
            .any(|traversal| {
                let rel_pos = traversal.portal_in.transform.inverse().transform_point2(self.pos);
                if traversal.direction.is_front() {
                    rel_pos.x > DEFAULT_EPSILON
                }
                else {
                    rel_pos.x < -DEFAULT_EPSILON
                }
            })
    }

    pub fn get_ghosts(self) -> impl Iterator<Item = GhostInfo> {
        // FIXME: probably incorrect for >1 portal traversals
        // (asserts just as reminder of this fact)
        assert!(self.portal_traversals.len() <= 1);

        self.portal_traversals.into_iter()
            .enumerate()
            .map(move |(index, traversal)| {
                let start = self.time + traversal.time_offset();
                let expiration_time = start + (traversal.end_age - self.age);
                let snapshot = Self {
                    time: start,
                    pos: traversal.transform_point(self.pos),
                    vel: traversal.transform_vector(self.vel),
                    portal_traversals: self.portal_traversals.iter()
                        .map(|traversal| traversal.swap())
                        .collect(),
                    ..self
                };
                GhostInfo {
                    snapshot,
                    offset: traversal.time_offset(),
                    index,
                    expiration_time,
                }
            })
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
        debug_assert!(!snapshot.is_ghost());
        let link = SimSnapshotLink { idx: self.nodes.len() };

        if let Some(age_previous) = age_previous {
            let age_parent = &mut self.nodes[age_previous.idx];
            debug_assert_eq!(age_parent.snapshot.original_idx, snapshot.original_idx);
            let distance = multiverse.distance(age_parent.snapshot.timeline_id, snapshot.timeline_id);
            debug_assert!(distance == Some(0) || distance == Some(1));
            let already_found = self.nodes[age_previous.idx].age_children.iter().copied()
                .find(|&child_link| self[child_link] == snapshot);
            if let Some(already_found) = already_found {
                return already_found;
            }
        }

        self.nodes.push(SimSnapshotNode {
            age_previous,
            ..SimSnapshotNode::from(snapshot)
        });

        if let Some(age_previous) = age_previous {
            self.nodes[age_previous.idx].age_children.push(link);
            debug_assert!(
                self.nodes[age_previous.idx].age_children.iter()
                    .map(|&child_link| self[child_link].timeline_id)
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

    pub fn nodes(&'_ self) -> impl Iterator<Item = (SimSnapshotLink, &'_ SimSnapshotNode)> {
        self.nodes.iter()
            .enumerate()
            .map(|(idx, node)| (SimSnapshotLink { idx }, node))
    }

    pub fn nodes_mut(&'_ mut self) -> impl Iterator<Item = (SimSnapshotLink, &'_ mut SimSnapshotNode)> {
        self.nodes.iter_mut()
            .enumerate()
            .map(|(idx, node)| (SimSnapshotLink { idx }, node))
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
