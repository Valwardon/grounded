use std::sync::Arc;
use semantic_graph::prelude::*;

/// Tracks goal lifecycle: decomposition → execution → completion/failure.
pub struct GoalResolver {
    ctx: Arc<SemanticContext>,
    active_goals: Vec<NodeId>,
    max_active_goals: usize,
}

impl GoalResolver {
    pub fn new(ctx: Arc<SemanticContext>) -> Self {
        GoalResolver {
            ctx,
            active_goals: Vec::with_capacity(16),
            max_active_goals: 16,
        }
    }

    /// Register a new goal node in the graph.
    pub fn register_goal(&mut self, label: &str, priority: f64, deadline_tick: u64) -> NodeId {
        let mut graph = self.ctx.graph.write();
        let id = graph.insert(GroundedNode {
            id: NodeId::ZERO,
            label: label.to_string(),
            node_type: NodeType::Goal,
            grounding: Grounding::Goal {
                priority,
                deadline_tick,
                status: GoalStatus::Active,
            },
            decay: 0.99,
            threshold: 15.0,
            base_activation: 0.0,
            edges: vec![Edge::new(Relation::AssociatedWith, NodeId::SELF)],
            epistemic_status: EpistemicStatus::CoreConcept,
            valence: 0.0,
            mean_error: 0.0,
            variance: 0.0,
        });
        self.ctx.link_to_self(Relation::AssociatedWith, id);
        if self.active_goals.len() < self.max_active_goals {
            self.active_goals.push(id);
        }
        id
    }

    /// Mark a goal as completed (achieved).
    pub fn complete_goal(&mut self, goal_id: NodeId) {
        let mut graph = self.ctx.graph.write();
        if let Some(node) = graph.get(goal_id) {
            let mut n = node.write();
            if let Grounding::Goal { ref mut status, .. } = n.grounding {
                *status = GoalStatus::Completed;
            }
        }
        self.active_goals.retain(|&id| id != goal_id);
    }

    /// Mark a goal as failed (unachievable in current context).
    pub fn fail_goal(&mut self, goal_id: NodeId) {
        let mut graph = self.ctx.graph.write();
        if let Some(node) = graph.get(goal_id) {
            let mut n = node.write();
            if let Grounding::Goal { ref mut status, .. } = n.grounding {
                *status = GoalStatus::Failed;
            }
        }
        self.active_goals.retain(|&id| id != goal_id);
    }

    /// Return all active (non-terminal) goals, sorted by priority descending.
    pub fn active_goals(&self) -> Vec<(NodeId, f64)> {
        let graph = self.ctx.graph.read();
        let mut goals: Vec<(NodeId, f64)> = Vec::new();
        for i in 2..graph.len() {
            if let Some(n) = graph.get(NodeId::from_raw(i as u64)) {
                let node = n.read();
                if node.node_type == NodeType::Goal {
                    if let Grounding::Goal { priority, status, .. } = node.grounding {
                        if status == GoalStatus::Active || status == GoalStatus::InProgress {
                            goals.push((node.id, priority));
                        }
                    }
                }
            }
        }
        goals.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        goals
    }

    /// Check if a goal has been achieved by examining subgoal completion.
    pub fn check_goal_achieved(&self, goal_id: NodeId) -> bool {
        let graph = self.ctx.graph.read();
        let node = match graph.get(goal_id) {
            Some(n) => n.read(),
            None => return false,
        };
        // A goal is achieved when all its SubGoalOf children are Completed
        let mut all_completed = true;
        for edge in &node.edges {
            if edge.relation == Relation::SubGoalOf {
                if let Some(child) = graph.get(edge.target) {
                    let c = child.read();
                    if let Grounding::Goal { status, .. } = &c.grounding {
                        if *status != GoalStatus::Completed {
                            all_completed = false;
                        }
                    }
                }
            }
        }
        all_completed
    }

    /// Count of active (non-terminal) goals.
    pub fn active_count(&self) -> usize {
        self.active_goals.len()
    }
}
