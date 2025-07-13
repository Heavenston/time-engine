use super::*;

use std::{ fmt::Display, range::RangeTo, sync::Arc };

use glam::{ Affine2, Vec2 };
use itertools::Itertools;
use smallvec::smallvec;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortalTraversalDirection {
    /// Special case for vecolity = 0
    NotMoving,
    GoingIn,
    GoingOut,
}

#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct RootSnapshot {
    pub object_id: usize,

    pub pos: Vec2,
    pub rot: f32,

    pub linvel: Vec2,
    pub angvel: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PartialPortalTraversal {
    pub half_portal_idx: usize,
    pub direction: PortalDirection,
    /// How much time does this portal traversal ends with the current
    /// velocity
    pub duration: Positive,
    pub sub_id: usize,
    pub traversal_direction: PortalTraversalDirection,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PartialSnapshotDelta {
    Impulse {
        /// Is in *local* space
        linear: Vec2,
        /// Is in *local* space... but because there is never any mirroring
        /// this is the same as global space
        angular: f32,
    },
    PortalTraversal {
        /// This is a new partial portal traversal that will stay until the
        /// snapshot does not collide with the portal anymore
        traversal: PartialPortalTraversal,
    },
    Ghostification {
        sub_id: usize,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct PartialSnapshot {
    /// How much time *after* the *previous* snapshot does this one
    /// happens in local time
    pub delta_age: Positive,
    pub delta: PartialSnapshotDelta,
}

impl PortalTraversalDirection {
    pub fn is_going_in(self) -> bool {
        matches!(self, Self::GoingIn)
    }

    pub fn is_going_out(self) -> bool {
        matches!(self, Self::GoingOut)
    }

    pub fn swap(self) -> Self {
        match self {
            Self::NotMoving => Self::NotMoving,
            Self::GoingIn => Self::GoingOut,
            Self::GoingOut => Self::GoingIn,
        }
    }

    pub fn from_velocity_direction(
        velocity_direction: PortalVelocityDirection,
        direction: PortalDirection,
    ) -> Self {
        match (velocity_direction, direction) {
            (PortalVelocityDirection::NotMoving, _) => Self::NotMoving,
            (a, b) if a == b => Self::GoingIn,
            _ => Self::GoingOut,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PortalTraversal {
    // TODO: Reduce size to maybe u8 or u16
    pub half_portal_idx: usize,
    pub direction: PortalDirection,
    pub traversal_direction: PortalTraversalDirection,
    pub time_range: RangeTo<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Snapshot {
    pub object_id: usize,
    /// Handle of the snapshot node that 'created' this snapshot
    pub handle: NodeHandle,
    /// Each snapshot graph node can create multiple snapshots, each indexed
    /// by their sub_id
    pub sub_id: usize,
    /// Time this snapshot was extrapolated by from its original time
    pub extrapolated_by: Positive,

    pub age: Positive,
    pub time: f32,

    pub linvel: Vec2,
    pub angvel: f32,

    pub pos: Vec2,
    pub rot: f32,

    pub portal_traversals: AutoSmallVec<PortalTraversal>,
    pub force_transform: Affine2,
}

impl Snapshot {
    /// Uniquely identity this snapshot (even across extroplation)
    pub fn id(&self) -> (NodeHandle, usize) {
        (self.handle, self.sub_id)
    }

    pub fn get_transform(&self) -> Affine2 {
        Affine2::from_angle_translation(self.rot, self.pos)
    }

    pub fn apply_portal_transormation(&mut self, transform: Affine2) {
        self.linvel = transform.transform_vector2(self.linvel);
        self.pos = transform.transform_point2(self.pos);
        // FIXME: Correct?, (too) slow?
        self.rot += transform.to_scale_angle_translation().1;
        self.force_transform *= transform;
    }

    pub fn integrate_by(&mut self, delta: Positive) {
        self.pos += self.linvel * delta.get();
        self.rot += self.angvel * delta.get();
        self.age += delta;
        self.time += delta.get();
    }

    pub fn extrapolate_to(&self, to: f32) -> Option<Self> {
        let to = if (to - self.time).abs() <= DEFAULT_EPSILON { self.time } else { to };
        let Ok(dt) = Positive::new(to - self.time)
        else {
            unreachable!("New time must be higher than current time ({to} is lower than {})", self.time);
        };

        Some(Self {
            extrapolated_by: self.extrapolated_by + dt,

            age: self.age + dt,
            time: to,

            pos: self.pos + self.linvel * dt.get(),
            rot: self.rot + self.angvel * dt.get(),

            portal_traversals: self.portal_traversals.clone(),

            ..*self
        })
    }
}

impl Display for Snapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "obj{}({}.{} {}+{}s)", self.object_id, self.handle, self.sub_id, self.time, self.extrapolated_by)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SnapgraphDataType;

impl dg::DeltaGraphDataType for SnapgraphDataType {
    type RootData = RootSnapshot;
    type PartialData = PartialSnapshot;
    type IntegratedData = [Snapshot];
    type Ctx<'a> = &'a Simulator;

    fn integrate_root(
        _ctx: &mut Self::Ctx<'_>,
        _graph: &dg::DeltaGraph<Self>,
        handle: RootNodeHandle,
        root_snap: &RootSnapshot,
    ) -> Arc<[Snapshot]> {
        let snapshot = sg::Snapshot {
            object_id: root_snap.object_id,
            handle: handle.into(),
            sub_id: 0,
            extrapolated_by: Positive::new(0.).expect("Positive"),

            // All objects starts with age 0 at time 0
            age: Positive::new(0.).expect("Positive"),
            time: 0.,

            linvel: root_snap.linvel,
            angvel: root_snap.angvel,

            pos: root_snap.pos,
            rot: root_snap.rot,

            // computed later
            portal_traversals: smallvec![],
            force_transform: Affine2::IDENTITY,
        };

        Arc::from(vec![snapshot].into_boxed_slice())
    }

    fn integrate_partial(
        ctx: &mut Self::Ctx<'_>,
        _graph: &dg::DeltaGraph<Self>,
        handle: InnerNodeHandle,
        snapshots: &[Snapshot],
        partial: &Self::PartialData,
    ) -> Arc<[Snapshot]> {
        let mut new_sub_id = {
            let mut counter = snapshots.iter().map(|snap| snap.sub_id)
                .max().expect("If new_sub_id is called this cannot be empty");
            move || {
                counter += 1;
                counter
            }
        };

        let new_snapshots = snapshots.iter()
        .flat_map(|snapshot| -> AutoSmallVec<_> {
            let mut new_snapshot = snapshot.clone();
            new_snapshot.integrate_by(partial.delta_age);
            new_snapshot.handle = handle.into();

            new_snapshot.portal_traversals.retain(|traversal| {
                !traversal.time_range.is_finished(new_snapshot.time)
            });
    
            // Portal traversals only affect a single sub_id
            // whereas impulses affect all sub_id s
            let ghost_snapshot: Option<sg::Snapshot> = match &partial.delta {
                sg::PartialSnapshotDelta::Impulse { linear, angular } => {
                    new_snapshot.linvel += snapshot.force_transform.transform_vector2(*linear);
                    new_snapshot.angvel += angular;

                    // Re-compute the end of each traversals as change in velocity
                    // means changes to when traversal ends
                    for i in 0..new_snapshot.portal_traversals.len() {
                        let traversal = &new_snapshot.portal_traversals[i];
                        let Some((_, new_delta_end, _, velocity_direction)) = ctx.cast_portal_traversal_start_end(&new_snapshot, traversal.half_portal_idx)
                        else { unreachable!("Should have been known/'detected' by the time_range before ?") };

                        let traversal = &mut new_snapshot.portal_traversals[i];
                        traversal.time_range = ..new_snapshot.time + new_delta_end;

                        traversal.traversal_direction = sg::PortalTraversalDirection::
                            from_velocity_direction(velocity_direction, traversal.direction);
                    }

                    None
                },
                sg::PartialSnapshotDelta::PortalTraversal { traversal } =>
                if traversal.sub_id == snapshot.sub_id {
                    let mut ghost_snapshot = new_snapshot.clone();
                    let half_portal = &ctx.half_portals[traversal.half_portal_idx];
                    let out_half_portal = &ctx.half_portals[half_portal.linked_to];

                    let transform = out_half_portal.transform * half_portal.transform.inverse();
                    
                    new_snapshot.portal_traversals.push(sg::PortalTraversal {
                        half_portal_idx: traversal.half_portal_idx,
                        direction: traversal.direction,
                        traversal_direction: traversal.traversal_direction,
                        time_range: ..new_snapshot.time + traversal.duration.get(),
                    });

                    ghost_snapshot.sub_id = new_sub_id();
                    ghost_snapshot.time += half_portal.time_offset;
                    ghost_snapshot.apply_portal_transormation(transform);
                    ghost_snapshot.portal_traversals.push(sg::PortalTraversal {
                        half_portal_idx: half_portal.linked_to,
                        direction: traversal.direction.swap(),
                        traversal_direction: traversal.traversal_direction.swap(),
                        time_range: ..ghost_snapshot.time + traversal.duration.get(),
                    });

                    Some(ghost_snapshot)
                } else { None },
                sg::PartialSnapshotDelta::Ghostification { sub_id } =>
                if snapshot.sub_id == *sub_id { return smallvec![]; } else { None },
            };

            if let Some(additional) = ghost_snapshot {
                smallvec![new_snapshot, additional]
            }
            else {
                smallvec![new_snapshot]
            }
        })
        .collect_vec();

        Arc::from(new_snapshots.into_boxed_slice())
    }

}

pub type NodeHandle = dg::NodeHandle<SnapgraphDataType>;
pub type InnerNodeHandle = dg::InnerNodeHandle<SnapgraphDataType>;
pub type RootNodeHandle = dg::RootNodeHandle<SnapgraphDataType>;
pub trait GenericNode = dg::GenericNode<SnapgraphDataType>;

pub type SnapshotGraph = dg::DeltaGraph<SnapgraphDataType>;
pub type InnerNode = dg::InnerNode<SnapgraphDataType>;
pub type RootNode = dg::RootNode<SnapgraphDataType>;
pub type RootNodeRef<'a> = dg::RootNodeRef<'a, SnapgraphDataType>;
pub type InnerNodeRef<'a> = dg::InnerNodeRef<'a, SnapgraphDataType>;
pub type NodeRef<'a> = dg::NodeRef<'a, SnapgraphDataType>;

// TEMP: TODO: REMOVE
pub trait DeprecatedTimelineIdDummy {
    #[deprecated]
    #[expect(deprecated)]
    fn timeline_id(&self) -> TimelineId {
        TimelineId::root()
    }
}

impl<T> DeprecatedTimelineIdDummy for T
    where T: GenericNode
{ }
