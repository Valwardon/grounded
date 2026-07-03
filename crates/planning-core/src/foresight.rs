use std::sync::Arc;
use semantic_graph::prelude::*;
use crate::planner::PlanStep;

/// Symbolic foresight engine: evaluates action sequences by simulating
/// their cascading effects through the semantic graph.
///
/// Non-statistical: uses graph topology, valence propagation, and
/// deterministic activation rules to predict outcomes.
pub struct ForesightEngine {
    ctx: Arc<SemanticContext>,
    max_branches: usize,
    simulation_decay: f64,
}

/// Result of evaluating a plan or action sequence.
#[derive(Debug, Clone)]
pub struct ForesightResult {
    pub confidence: f64,
    pub predicted_valence_shift: f64,
    pub novelty_impact: f64,
    pub nodes_affected: Vec<(NodeId, f64)>,
    pub risk_score: f64,
}

impl ForesightEngine {
    pub fn new(ctx: Arc<SemanticContext>) -> Self {
        ForesightEngine {
            ctx,
            max_branches: 3,
            simulation_decay: 0.85,
        }
    }

    /// Evaluate a sequence of plan steps.
    /// Returns a confidence score (0.0–1.0) based on how well the
    /// steps chain together causally in the graph.
    pub fn evaluate_plan(&self, steps: &[PlanStep]) -> f64 {
        if steps.is_empty() {
            return 0.0;
        }

        let graph = self.ctx.graph.read();
        let mut chain_strength = 0.0;
        let mut valid_edges = 0;

        for window in steps.windows(2) {
            let prev = &window[0];
            let next = &window[1];

            let has_path = self.has_direct_path(&graph, prev.node_id, next.node_id);
            if has_path {
                chain_strength += 1.0;
            }
            valid_edges += 1;
        }

        if valid_edges == 0 {
            return 0.5;
        }

        chain_strength / valid_edges as f64
    }

    /// Check if there's a direct edge or 2-hop path between two nodes.
    fn has_direct_path(&self, graph: &GraphArena, from: NodeId, to: NodeId) -> bool {
        if let Some(node) = graph.get(from) {
            let n = node.read();
            for edge in &n.edges {
                if edge.target == to {
                    return true;
                }
                // Check 2-hop: does the target of this edge connect to `to`?
                if let Some(intermediate) = graph.get(edge.target) {
                    let inter = intermediate.read();
                    for e2 in &inter.edges {
                        if e2.target == to {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    /// Simulate the outcome of executing a specific action node.
    /// Returns the predicted activation levels of connected nodes
    /// after the action fires, using the graph's spread dynamics.
    pub fn simulate_action(&self, action_id: NodeId) -> ForesightResult {
        let graph = self.ctx.graph.read();
        let mut affected: Vec<(NodeId, f64)> = Vec::new();
        let mut total_valence = 0.0;
        let mut risk = 0.0;

        if let Some(node) = graph.get(action_id) {
            let n = node.read();
            let base_activation = n.base_activation + 0.5;

            for edge in &n.edges {
                let propagated = base_activation * edge.dynamic_weight * self.simulation_decay;
                if propagated.abs() > 0.01 {
                    affected.push((edge.target, propagated));
                    if let Some(target) = graph.get(edge.target) {
                        let t = target.read();
                        total_valence += t.valence * propagated;
                        if t.variance > 0.3 {
                            risk += propagated.abs() * t.variance;
                        }
                    }
                }
            }
        }

        let node_count = affected.len().max(1) as f64;
        let confidence = if risk > 0.0 {
            (1.0 - (risk / node_count).clamp(0.0, 1.0)) * 0.8 + 0.1
        } else {
            0.9
        };

        ForesightResult {
            confidence: confidence.clamp(0.0, 1.0),
            predicted_valence_shift: total_valence / node_count,
            novelty_impact: risk / node_count,
            nodes_affected: affected,
            risk_score: risk,
        }
    }

    /// Fork a simulation branch — creates a temporary simulation node
    /// in the graph linked to the action being evaluated.
    pub fn fork_branch(&self, action_label: &str) -> Option<NodeId> {
        let branches = self.active_branches();
        if branches >= self.max_branches {
            return None;
        }

        let mut graph = self.ctx.graph.write();
        let id = graph.insert(GroundedNode {
            id: NodeId::ZERO,
            label: format!("sim_{}", action_label),
            node_type: NodeType::Simulation,
            grounding: Grounding::Simulation {
                confidence: 0.5,
                horizon: 10,
            },
            decay: 0.9,
            threshold: 5.0,
            base_activation: 0.0,
            edges: vec![Edge::new(Relation::Simulates, NodeId::SELF)],
            epistemic_status: EpistemicStatus::CoreConcept,
            valence: 0.0,
            mean_error: 0.0,
            variance: 0.0,
        });
        // Link to SELF
        self.ctx.link_to_self(Relation::Simulates, id);
        Some(id)
    }

    /// Count active simulation branches in the graph.
    pub fn active_branches(&self) -> usize {
        self.ctx.graph.read().by_type(NodeType::Simulation).len()
    }

    /// Prune low-confidence simulation branches by relabeling them.
    pub fn prune_branches(&self, min_confidence: f64) -> usize {
        let graph = self.ctx.graph.read();
        let mut pruned = 0;
        for i in 2..graph.len() {
            if let Some(n) = graph.get(NodeId::from_raw(i as u64)) {
                let node = n.read();
                if node.node_type == NodeType::Simulation {
                    if let Grounding::Simulation { confidence, .. } = &node.grounding {
                        if *confidence < min_confidence {
                            // Soft-delete: clear edges so the node is ignored
                            // by downstream processing
                            drop(node);
                            // Can't mutate while holding read - need write lock
                            // We track counts but mark via valence as abandoned
                        }
                    }
                }
            }
        }
        // Second pass with write lock: clear edges on low-confidence branches
        let mut graph = self.ctx.graph.write();
        for i in 2..graph.len() {
            if let Some(n) = graph.get(NodeId::from_raw(i as u64)) {
                let mut node = n.write();
                if node.node_type == NodeType::Simulation {
                    if let Grounding::Simulation { confidence, .. } = &node.grounding {
                        if *confidence < min_confidence {
                            node.edges.clear();
                            pruned += 1;
                        }
                    }
                }
            }
        }
        pruned
    }
}
