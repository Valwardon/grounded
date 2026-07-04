use std::sync::Arc;
use semantic_graph::prelude::*;
use episodic_memory::query_by_node_label;

use crate::foresight::ForesightEngine;
use crate::planner::{HierarchicalPlanner, Plan, PlanStep};

/// ────────────────────────────────────────────────────────────
///  MCTS + Active Inference Strategic Planner
///
///  Extends HierarchicalPlanner with:
///    - Monte Carlo Tree Search over action sequences
///    - Active Inference Expected Free Energy for action selection
///    - Episodic memory rollouts (deterministic, no random sim)
///    - Pause-and-replan on high prediction error (DMN switch)
///
///  All scores are deterministic functions of graph topology,
///  activation levels, and episodic similarity — no ML, no RNG.
/// ────────────────────────────────────────────────────────────

/// Number of MCTS rollouts per planning cycle.
const MCTS_ROLLOUTS: usize = 8;

/// Maximum depth of a single MCTS rollout.
const MCTS_MAX_DEPTH: usize = 4;

/// Exploration constant for UCB1.
const UCB_C: f64 = 1.414;

/// Prediction error threshold that triggers DMN switch → pause-and-replan.
const DMN_REPLAN_THRESHOLD: f64 = 0.3;

/// Weight for expected prediction error in Expected Free Energy.
const EFE_ERROR_WEIGHT: f64 = 0.5;
/// Weight for expected novelty (information gain) in EFE.
const EFE_NOVELTY_WEIGHT: f64 = 0.3;
/// Weight for expected cost in EFE.
const EFE_COST_WEIGHT: f64 = 0.2;

/// A single node in the MCTS tree.
#[derive(Debug, Clone)]
pub struct MctsNode {
    /// The action node ID in the semantic graph.
    pub action_id: NodeId,
    /// Label for debugging.
    pub label: String,
    /// Number of times this node was visited during MCTS.
    pub visit_count: u64,
    /// Total accumulated value (sum of rollout scores).
    pub total_value: f64,
    /// Children indexed by (action_id, tree index).
    pub children: Vec<usize>,
    /// Parent index (None = root).
    pub parent: Option<usize>,
    /// Action cost (from graph).
    pub cost: f64,
}

impl MctsNode {
    pub fn new(action_id: NodeId, label: String, cost: f64) -> Self {
        MctsNode {
            action_id,
            label,
            visit_count: 0,
            total_value: 0.0,
            children: Vec::new(),
            parent: None,
            cost,
        }
    }

    /// UCB1 score for this node (higher = more promising).
    pub fn ucb_score(&self, parent_visits: u64) -> f64 {
        if self.visit_count == 0 {
            return f64::MAX; // Untried nodes are always explored first
        }
        let exploitation = self.total_value / self.visit_count as f64;
        let exploration = UCB_C * ((parent_visits as f64).ln() / self.visit_count as f64).sqrt();
        exploitation + exploration
    }
}

/// A single MCTS roll-out result — sequence of actions and their cumulative score.
#[derive(Debug, Clone)]
pub struct RolloutResult {
    pub action_ids: Vec<NodeId>,
    pub cumulative_score: f64,
    pub confidence: f64,
}

/// Strategic planner with MCTS + Active Inference.
///
/// Used by HierarchicalPlanner when a goal needs more than simple
/// single-pass decomposition. Runs MCTS rollouts over retrieved
/// episodic sequences, evaluates them via Expected Free Energy,
/// and returns the best action sequence.
pub struct StrategicPlanner {
    ctx: Arc<SemanticContext>,
    /// Tree nodes allocated during MCTS (grows across rollouts).
    tree: Vec<MctsNode>,
    /// Root index in tree (0 after build_tree).
    root_idx: usize,
    /// Reference to foresight engine for cost estimates.
    foresight: ForesightEngine,
    /// Prediction error threshold for DMN pause-and-replan.
    replan_threshold: f64,
}

impl StrategicPlanner {
    pub fn new(ctx: Arc<SemanticContext>) -> Self {

    /// Access the semantic context (for lifecycle/trait bridge use).
    pub fn ctx(&self) -> &Arc<SemanticContext> {
        &self.ctx
    }
        StrategicPlanner {
            foresight: ForesightEngine::new(ctx.clone()),
            ctx,
            tree: Vec::with_capacity(64),
            root_idx: 0,
            replan_threshold: DMN_REPLAN_THRESHOLD,
        }
    }

    /// Set the prediction error threshold at which the planner triggers replan.
    pub fn set_replan_threshold(&mut self, threshold: f64) {
        self.replan_threshold = threshold.clamp(0.05, 1.0);
    }

    /// Build an MCTS tree starting from a set of candidate action nodes
    /// that can achieve the given goal. Returns the root node index.
    fn build_tree(&mut self, goal_id: NodeId, candidates: &[(NodeId, String, f64)]) -> usize {
        self.tree.clear();
        let root = MctsNode::new(goal_id, format!("goal_{}", goal_id.0), 0.0);
        self.tree.push(root);
        let root_idx = 0;

        for (action_id, label, cost) in candidates {
            let child = MctsNode::new(*action_id, label.clone(), *cost);
            self.tree.push(child);
            let child_idx = self.tree.len() - 1;
            self.tree[child_idx].parent = Some(root_idx);
            self.tree[root_idx].children.push(child_idx);
        }

        root_idx
    }

    /// Select a leaf node using UCB1 (starting from root).
    fn select_leaf(&self) -> usize {
        let mut current = self.root_idx;
        loop {
            let node = &self.tree[current];
            if node.children.is_empty() {
                return current;
            }
            let parent_visits = node.visit_count.max(1);
            let best_child = node.children.iter()
                .max_by(|&&a, &&b| {
                    let score_a = self.tree[a].ucb_score(parent_visits);
                    let score_b = self.tree[b].ucb_score(parent_visits);
                    score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
                })
                .copied()
                .unwrap_or(node.children[0]);
            current = best_child;
        }
    }

    /// Expand a leaf node: add its children from the graph (action nodes that
    /// follow from this node's action via Precedes edges).
    fn expand(&mut self, leaf_idx: usize) {
        let leaf = &self.tree[leaf_idx];
        if leaf.visit_count > 0 || leaf_idx == self.root_idx {
            // Only expand if visited at least once (or it's the root)
            if leaf_idx != self.root_idx {
                return;
            }
        }

        let graph = self.ctx.graph.read();
        let action_id = leaf.action_id;

        // Find actions that can follow this action (via Precedes edges from target)
        let mut successors: Vec<(NodeId, String, f64)> = Vec::new();
        for i in 2..graph.len() {
            if let Some(n) = graph.get(NodeId::from_raw(i as u64)) {
                let node = n.read();
                if node.node_type == NodeType::Action {
                    for edge in &node.edges {
                        if edge.relation == Relation::Precedes && edge.target == action_id {
                            let cost = 1.0 - node.valence.clamp(0.0, 1.0);
                            successors.push((node.id, node.label.clone(), cost));
                        }
                    }
                }
                // Also check if this action has a Precedes edge toward another action
                if node.node_type == NodeType::Action {
                    for edge in &node.edges {
                        if edge.relation == Relation::Precedes && edge.target != action_id && edge.target.0 != 0 {
                            let cost = 1.0 - node.valence.clamp(0.0, 1.0);
                            successors.push((edge.target, format!("from_{}", node.label), cost));
                        }
                    }
                }
            }
        }

        // Deduplicate by action_id
        successors.sort_by(|a, b| a.0.cmp(&b.0));
        successors.dedup_by(|a, b| a.0 == b.0);

        // Add as children of leaf
        for (action_id, label, cost) in successors {
            let child = MctsNode::new(action_id, label, cost);
            self.tree.push(child);
            let child_idx = self.tree.len() - 1;
            self.tree[child_idx].parent = Some(leaf_idx);
            self.tree[leaf_idx].children.push(child_idx);
        }
    }

    /// Simulate a roll-out from the given leaf node.
    ///
    /// Instead of random simulation, we use episodic memory to retrieve
    /// past action sequences that followed from similar contexts. This
    /// makes the rollout deterministic and grounded in lived experience.
    fn simulate(&self, leaf_idx: usize) -> f64 {
        let leaf = &self.tree[leaf_idx];
        if leaf_idx == self.root_idx {
            return 0.0;
        }

        // Retrieve similar episodes from memory
        let graph = self.ctx.graph.read();
        let label = graph.label_of(leaf.action_id).unwrap_or_default();
        drop(graph);

        let episodes = query_by_node_label(&self.ctx.graph.read(), &label);
        let mut score = 0.0;
        let mut count = 0;

        // Score = average valence shift from similar past sequences
        for ep in episodes.iter().take(MCTS_MAX_DEPTH) {
            let valence_shift = ep.valence;
            score += valence_shift;
            count += 1;
        }

        if count > 0 {
            score / count as f64
        } else {
            // No episodic memory: use foresight simulation
            let foresight = self.foresight.simulate_action(leaf.action_id);
            foresight.predicted_valence_shift
        }
    }

    /// Backpropagate the simulation score up the tree.
    fn backpropagate(&mut self, leaf_idx: usize, score: f64) {
        let mut current = leaf_idx;
        loop {
            self.tree[current].visit_count += 1;
            self.tree[current].total_value += score;
            match self.tree[current].parent {
                Some(parent) => current = parent,
                None => break,
            }
        }
    }

    /// Compute Expected Free Energy for a candidate action.
    ///
    /// G = E[ prediction_error | action ] + E[ novelty | action ] - E[ cost | action ]
    ///
    /// Low EFE = preferred action (minimizes surprise, maximizes information gain).
    pub fn expected_free_energy(&self, action_id: NodeId) -> f64 {
        let graph = self.ctx.graph.read();
        let node = match graph.get(action_id) {
            Some(n) => n.read(),
            None => return 1.0,
        };

        // Expected prediction error: mean_error of the node
        let expected_error = (node.mean_error * EFE_ERROR_WEIGHT).clamp(0.0, 1.0);

        // Expected novelty: 1.0 - valence (unfamiliar actions are more novel)
        let novelty = ((1.0 - node.valence.clamp(0.0, 1.0)) * EFE_NOVELTY_WEIGHT).clamp(0.0, 1.0);

        // Expected cost: based on number of edges (more connections = more expensive)
        let edge_count = node.edges.len().min(32) as f64 / 32.0;
        let cost = (edge_count * EFE_COST_WEIGHT).clamp(0.0, 1.0);

        // Free Energy = surprise + novelty - cost
        expected_error + novelty - cost
    }

    /// Run a full MCTS planning cycle.
    ///
    /// Steps:
    ///   1. Find candidate action nodes for the goal
    ///   2. Build MCTS tree
    ///   3. Run N rollouts (select → expand → simulate → backpropagate)
    ///   4. Select best action sequence
    ///   5. Convert to Plan
    pub fn plan_for_goal(&mut self, goal_id: NodeId, hierarchical: &HierarchicalPlanner) -> Plan {
        let graph = self.ctx.graph.read();

        // Find candidate action nodes
        let mut candidates: Vec<(NodeId, String, f64)> = Vec::new();
        for i in 2..graph.len() {
            if let Some(n) = graph.get(NodeId::from_raw(i as u64)) {
                let node = n.read();
                if node.node_type == NodeType::Action {
                    for edge in &node.edges {
                        if edge.relation == Relation::Achieves && edge.target == goal_id {
                            let cost = 1.0 - node.valence.clamp(0.0, 1.0);
                            candidates.push((node.id, node.label.clone(), cost));
                        }
                    }
                    // Also check if this action is connected to the goal via SubGoalOf
                    for edge in &node.edges {
                        if edge.relation == Relation::SubGoalOf && edge.target == goal_id {
                            let cost = 0.8;
                            candidates.push((node.id, format!("subgoal:{}", node.label), cost));
                        }
                    }
                }
            }
        }

        // If no direct actions, use foresight to find candidates
        if candidates.is_empty() {
            // Get the initial plan from hierarchical planner
            let fallback = hierarchical.plan_for_goal(goal_id);
            for step in &fallback.steps {
                candidates.push((step.node_id, step.label.clone(), step.expected_cost));
            }
        }

        // Build MCTS tree
        self.build_tree(goal_id, &candidates);

        // Run MCTS rollouts
        for _ in 0..MCTS_ROLLOUTS {
            let leaf = self.select_leaf();
            self.expand(leaf);
            let score = self.simulate(leaf);
            self.backpropagate(leaf, score);
        }

        // Extract best path from root
        let mut best_path: Vec<(NodeId, f64)> = Vec::new();
        let mut current = self.root_idx;
        loop {
            let node = &self.tree[current];
            if node.children.is_empty() || node.visit_count == 0 {
                break;
            }
            let parent_visits = node.visit_count.max(1);
            let best_child = node.children.iter()
                .max_by(|&&a, &&b| {
                    let score_a = self.tree[a].ucb_score(parent_visits);
                    let score_b = self.tree[b].ucb_score(parent_visits);
                    score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
                })
                .copied();

            match best_child {
                Some(child_idx) => {
                    if child_idx != self.root_idx {
                        let child = &self.tree[child_idx];
                        let value = if child.visit_count > 0 {
                            child.total_value / child.visit_count as f64
                        } else {
                            0.0
                        };
                        best_path.push((child.action_id, value));
                        current = child_idx;
                    } else {
                        break;
                    }
                }
                None => break,
            }
            if best_path.len() >= MCTS_MAX_DEPTH {
                break;
            }
        }

        // Convert to Plan
        let mut steps: Vec<PlanStep> = Vec::with_capacity(best_path.len());
        let mut total_cost = 0.0;

        for (idx, (action_id, _)) in best_path.iter().enumerate() {
            let label = self.ctx.graph.read()
                .label_of(*action_id)
                .unwrap_or_else(|| format!("mcts_step_{}", idx));
            let cost = if idx < self.tree.len() && self.tree[idx].cost > 0.0 {
                self.tree[idx].cost
            } else {
                0.5
            };
            let outcome = if idx < self.tree.len() {
                self.tree[idx].total_value / self.tree[idx].visit_count.max(1) as f64
            } else {
                0.0
            };

            steps.push(PlanStep {
                label,
                node_id: *action_id,
                expected_cost: cost,
                expected_outcome: outcome,
                is_action: true,
            });
            total_cost += cost;
        }

        // If no steps found, create a fallback exploration step
        if steps.is_empty() {
            steps.push(PlanStep {
                label: format!("mcts_explore_goal_{}", goal_id.0),
                node_id: goal_id,
                expected_cost: 1.0,
                expected_outcome: 0.3,
                is_action: false,
            });
            total_cost = 1.0;
        }

        // Confidence: root visit count / total rollouts
        let root = &self.tree[self.root_idx];
        let confidence = if MCTS_ROLLOUTS > 0 {
            (root.visit_count as f64 / MCTS_ROLLOUTS as f64).clamp(0.0, 1.0)
        } else {
            0.5
        };

        Plan {
            goal_id,
            steps,
            total_cost,
            confidence,
        }
    }

    /// DMN switch: check if we should pause the current plan and replan.
    ///
    /// Returns true when prediction error exceeds threshold, indicating
    /// the Default Mode Network should take over for replanning.
    pub fn should_replan(&self, prediction_errors: &[PredictionError]) -> bool {
        if prediction_errors.is_empty() {
            return false;
        }
        let max_error = prediction_errors.iter()
            .map(|e| e.error_magnitude)
            .fold(0.0_f64, f64::max);
        max_error > self.replan_threshold
    }

    /// Run a single "imagination" rollout in counterfactual mode.
    ///
    /// Switches the cognitive engine to Counterfactual mode, executes
    /// the proposed action sequence in simulation, and returns the
    /// predicted outcome without modifying long-term memory.
    pub fn imagine(&self, actions: &[NodeId]) -> f64 {
        if actions.is_empty() {
            return 0.0;
        }

        let mut cumulative = 0.0;
        for action_id in actions {
            let result = self.foresight.simulate_action(*action_id);
            cumulative += result.predicted_valence_shift;
        }

        cumulative / actions.len() as f64
    }

    /// Active inference: choose the action that minimizes Expected Free Energy
    /// among a set of candidates.
    pub fn choose_action(&self, candidates: &[NodeId]) -> Option<NodeId> {
        candidates.iter()
            .min_by(|&&a, &&b| {
                let efe_a = self.expected_free_energy(a);
                let efe_b = self.expected_free_energy(b);
                efe_a.partial_cmp(&efe_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
    }
}
