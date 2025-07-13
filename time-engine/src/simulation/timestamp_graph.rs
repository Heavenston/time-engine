use std::sync::Arc;

use itertools::Itertools;

use super::*;

#[derive(Debug, Clone, Copy)]
pub struct TimestampRoot {
    pub time: f32,
}

#[derive(Debug, Clone, Copy)]
pub enum TimestampPartialData {
    Forward { },
    BackwardBranch {
        /// Must be a parent
        target_timestamp: Timestamp,
    },
}

impl TimestampPartialData {
    pub fn is_forward(&self) -> bool {
        matches!(self, Self::Forward { .. })
    }

    pub fn is_backward_branch(&self) -> bool {
        matches!(self, Self::BackwardBranch { .. })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TimestampPartial {
    pub delta_time: Positive,
    pub delta: TimestampPartialData,
}

#[derive(Debug, Clone, Copy)]
pub struct TimestampBranch {
    /// The timestamp that did the branching
    pub branched_at: Timestamp,
    /// The timestamp to which the branching went
    pub branched_to: Timestamp,

    // TODO: Doc
    /// Is child of branched_to and parent of the current timestamp
    /// 
    /// uses delta times so may not be exact
    pub current_branch_timestamp: Timestamp,
    // TODO: Doc
    /// Should not be greater than the delta_time of the next timestamp after
    /// the current_branch_timestamp
    pub current_branch_delta_t: Positive,
}

#[derive(Debug, Clone)]
pub struct TimestampData {
    pub time: f32,
    pub branches: Vec<TimestampBranch>,
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
            branches: vec![],
        })
    }

    fn integrate_partial(
        ctx: &mut Self::Ctx<'_>,
        graph: &dg::DeltaGraph<Self>,
        handle: dg::InnerNodeHandle<Self>,
        running: &TimestampData,
        partial: &TimestampPartial,
    ) -> Arc<TimestampData> {
        let timestamp: Timestamp = handle.into();
        let mut branches = running.branches.clone();

        for branch in &mut branches {
            debug_assert!(graph.is_parent(branch.current_branch_timestamp, timestamp));
            let current_branch = &graph[branch.current_branch_timestamp];
            let Some(&next_timestamp) = current_branch.children().iter()
                .filter(|&&child_timestamp| graph.is_directly_related(child_timestamp.into(), timestamp))
                .at_most_one().expect("Cannot be multiple children of the same lineage")
            else {
                // Maybe just `continue;` ?
                todo!()
            };
            let next_partial = &graph[next_timestamp];

            branch.current_branch_delta_t += partial.delta_time;

            // if branch.current_branch_delta_t >= next_partial.data.delta_time
            // i.e. when we go further than the current_branch_timestamp
            if let Ok(new_delta_t) = Positive::new(branch.current_branch_delta_t - next_partial.data.delta_time) {
                branch.current_branch_timestamp = next_timestamp.into();
                branch.current_branch_delta_t = new_delta_t;
            }
        }

        match partial.delta {
            TimestampPartialData::Forward { } => {
                Arc::new(TimestampData {
                    time: running.time + partial.delta_time.get(),
                    branches,
                })
            },
            TimestampPartialData::BackwardBranch { target_timestamp } => {
                debug_assert!(graph.is_parent(target_timestamp, timestamp));

                #[cfg(debug_assertions)]
                'check_delta_t_is_not_too_high: {
                    let target_node = &graph[target_timestamp];
                    let Some(&next_timestamp) = target_node.children().iter()
                        .filter(|&&child_timestamp| graph.is_directly_related(child_timestamp.into(), timestamp))
                        .at_most_one().expect("Cannot be multiple children of the same lineage")
                    else { break 'check_delta_t_is_not_too_high; };
                    let next_partial = &graph[next_timestamp];
                    debug_assert!(partial.delta_time < next_partial.data.delta_time);
                }

                let target_data = graph.integrate(ctx, target_timestamp);
                branches.push(TimestampBranch {
                    branched_at: timestamp,
                    branched_to: target_timestamp,

                    current_branch_timestamp: timestamp,
                    current_branch_delta_t: partial.delta_time,
                });
                Arc::new(TimestampData {
                    time: target_data.time + partial.delta_time.get(),
                    branches,
                })
            },
        }
    }

    #[cfg(debug_assertions)]
    fn insert_hook(
        _ctx: &mut Self::InsertCtx<'_>,
        graph: &dg::DeltaGraph<Self>,
        timestamp: Timestamp,
    ) {
        // Just inserted so should be 0
        debug_assert_eq!(graph[timestamp].children().len(), 0);

        let dg::EitherHandle::Inner(handle) = timestamp.either()
        else { return };

        let parent = graph[handle].previous;

        let forward_count = graph.children(parent)
            .filter(|child| child.data.delta.is_forward())
            .count();

        debug_assert!(forward_count <= 1);
    }
}

pub type Timestamp = dg::NodeHandle<TimestampDataType>;
pub type TimestampGraph = dg::DeltaGraph<TimestampDataType>;
