use std::{iter::once, sync::Arc};

use itertools::{chain, Itertools};

use super::*;

#[derive(Debug, Clone)]
pub struct TimestampRoot {
    pub time: f32,
    pub handles: Box<[sg::NodeHandle]>,
}

#[derive(Debug, Clone, Copy)]
pub struct TimestampPartial {
    pub delta_time: Positive,
    pub remove_snapshots: [Option<sg::NodeHandle>; 2],
    pub new_snapshots: [Option<sg::NodeHandle>; 2],
}

#[derive(Debug, Clone)]
pub struct TimestampSnapshotLink {
    pub snapshot_handle: sg::NodeHandle,
    pub delta_since_add: Positive,
}

#[derive(Debug, Clone)]
pub struct TimestampData {
    pub time: f32,
    // FIXME: Using RLEVec instead of Vec will definitely save memory here
    //        but may have other costs so benchmark before swiching
    //        Could also just be a linked list... but this goes kinda back to
    //        just traversing the graph every time so hard to see a big win
    pub path: Box<[usize]>,
    pub snapshot_links: Box<[TimestampSnapshotLink]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimestampDataType;

impl dg::DeltaGraphDataType for TimestampDataType {
    type RootData = TimestampRoot;
    type PartialData = TimestampPartial;
    type IntegratedData = TimestampData;
    type Ctx<'a> = ();

    fn integrate_root(
        _ctx: &mut Self::Ctx<'_>,
        _graph: &dg::DeltaGraph<Self>,
        _handle: dg::RootNodeHandle<Self>,
        root: &TimestampRoot,
    ) -> Arc<TimestampData> {
        Arc::new(TimestampData {
            time: root.time,
            path: default(),
            snapshot_links: default(),
        })
    }

    fn integrate_partial(
        _ctx: &mut Self::Ctx<'_>,
        graph: &dg::DeltaGraph<Self>,
        handle: dg::InnerNodeHandle<Self>,
        running: &TimestampData,
        partial: &TimestampPartial,
    ) -> Arc<TimestampData> {
        let mut links = Vec::from(&running.snapshot_links[..]);

        for link in &mut links {
            link.delta_since_add += partial.delta_time;
        }

        // check that Some are first in the array
        #[cfg(debug_assertions)]
        for arr in [partial.remove_snapshots, partial.new_snapshots] {
            debug_assert!(arr.is_sorted_by_key(Option::is_none));
        }

        let handles_changes = partial.remove_snapshots.iter().copied()
            .zip(partial.new_snapshots.iter().copied());
        for (remove, new) in handles_changes {
            let new = new.map(|new| TimestampSnapshotLink {
                snapshot_handle: new,
                delta_since_add: ZERO.into(),
            });

            if let Some(remove_link) = remove {
                let Some(pos) = links.iter().position(|l| l.snapshot_handle == remove_link)
                else { unreachable!("All timestamp partial should be valid") };
                if let Some(new_link) = new {
                    links[pos] = new_link;
                }
                else {
                    links.swap_remove(pos);
                }
            }
            else if let Some(link) = new {
                links.push(link);
            }
        }

        Arc::new(TimestampData {
            time: running.time + partial.delta_time.get(),
            path: running.path.iter().copied()
                .chain(once(graph.child_index(handle)))
                .collect(),
            snapshot_links: links.into_boxed_slice(),
        })
    }
}

pub type Timestamp = dg::NodeHandle<TimestampDataType>;
pub type TimestampGraph = dg::DeltaGraph<TimestampDataType>;

pub trait TimestampGraphExt {
    /// Returns a list of timestamps that is at most dt away
    /// along with how much time is still remaining
    fn advance(&self, timestamp: Timestamp, dt: Positive) -> Vec<(Timestamp, Positive)>;
}

impl TimestampGraphExt for TimestampGraph {
    fn advance(&self, timestamp: Timestamp, dt: Positive) -> Vec<(Timestamp, Positive)> {
        let mut result = vec![];

        let mut add_self = self[timestamp].children().is_empty();
        for child in self.children(timestamp) {
            let ndt = dt - child.data.delta_time;

            if let Ok(ndt) = Positive::new(ndt) /* dt >= 0. */ {
                let mut nws = self.advance(child.inner_handle.into(), ndt);
                nws.retain(|(h1, _)| !result.iter().any(|(h2, _)| h1 == h2));
                result.extend(nws);
            }
            else /* ndt < 0. */ {
                add_self = true;
            }
        }

        if add_self {
            result.push((timestamp, dt))
        }

        debug_assert!(result.iter().map(|(h, _)| h).all_unique());

        result
    }
}
