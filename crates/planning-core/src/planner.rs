use std::sync::Arc;
use semantic_graph::prelude::*;
use crate::goal::GoalResolver;
use crate::foresight::ForesightEngine;

/// Hierarchical planner: decomposes goals into subgoals, builds action trees,
/// evaluates via foresight, and executes with mid-loop replanning.
pub struct HierarchicalPlanner {
    ctx: Arc<SemanticContext>,
    goals: GoalResolver,
    foresight: ForesightEngine,
    max_depth: u8,
    replan_threshold: f64,
}

/// A single planning step — one action or subgoal.
#[derive(Debug, Clone)]
pub struct PlanStep {
    pub label: String,
    pub node_id: NodeId,
    pub expected_cost: f64,
    pub expected_outcome: f64,
    pub is_action: bool,
}

/// A complete plan: ordered steps toward a goal.
#[derive(Debug, Clone)]
pub struct Plan {
    pub goal_id: NodeId,
    pub steps: Vec<PlanStep>,
    pub total_cost: f64,
    pub confidence: f64,
}

impl HierarchicalPlanner {
    pub fn new(ctx: Arc<SemanticContext>) -> Self {
        let foresight = ForesightEngine::new(ctx.clone());
        HierarchicalPlanner {
            goals: GoalResolver::new(ctx.clone()),
            ctx,
            foresight,
            max_depth: 5,
            replan_threshold: 0.3,
        }
    }

    pub fn goals(&self) -> &GoalResolver {
        &self.goals
    }

    pub fn foresight(&self) -> &ForesightEngine {
        &self.foresight
    }

    /// Set the maximum decomposition depth.
    pub fn set_max_depth(&mut self, depth: u8) {
        self.max_depth = depth;
    }

    /// Decompose a goal into a plan by traversing the semantic graph
    /// for relevant actions and subgoals.
    pub fn plan_for_goal(&self, goal_id: NodeId) -> Plan {
        let graph = self.ctx.graph.read();

        // Find action nodes linked to this goal (via Achieves edges)
        let mut steps: Vec<PlanStep> = Vec::new();
        let mut total_cost = 0.0;
        let mut goal_found = false;

        // Scan the graph for Action nodes that Achieve this goal
        for i in 2..graph.len() {
            if let Some(n) = graph.get(NodeId::from_raw(i as u64)) {
                let node = n.read();
                if node.node_type == NodeType::Action {
                    for edge in &node.edges {
                        if edge.relation == Relation::Achieves && edge.target == goal_id {
                            let cost = 1.0 - node.valence.clamp(0.0, 1.0);
                            steps.push(PlanStep {
                                label: node.label.clone(),
                                node_id: node.id,
                                expected_cost: cost,
                                expected_outcome: node.valence,
                                is_action: true,
                            });
                            total_cost += cost;
                            goal_found = true;
                        }
                    }
                }
            }
        }

        // If no direct action found, try subgoal decomposition
        if !goal_found {
            for i in 2..graph.len() {
                if let Some(n) = graph.get(NodeId::from_raw(i as u64)) {
                    let node = n.read();
                    if node.node_type == NodeType::Goal {
                        for edge in &node.edges {
                            if edge.relation == Relation::SubGoalOf && edge.target == goal_id {
                                let cost = 0.8;
                                steps.push(PlanStep {
                                    label: format!("subgoal:{}", node.label),
                                    node_id: node.id,
                                    expected_cost: cost,
                                    expected_outcome: 0.5,
                                    is_action: false,
                                });
                                total_cost += cost;
                            }
                        }
                    }
                }
            }
        }

        // If still nothing found, create a placeholder exploration step
        if steps.is_empty() {
            let explore_label = format!("explore_goal_{}", goal_id.0);
            steps.push(PlanStep {
                label: explore_label,
                node_id: goal_id,
                expected_cost: 1.0,
                expected_outcome: 0.3,
                is_action: false,
            });
            total_cost = 1.0;
        }

        // Evaluate plan via foresight
        let confidence = self.foresight.evaluate_plan(&steps);

        Plan {
            goal_id,
            steps,
            total_cost,
            confidence,
        }
    }

    /// Choose the best plan among alternatives.
    pub fn select_best_plan(&self, plans: Vec<Plan>) -> Option<Plan> {
        plans.into_iter()
            .max_by(|a, b| {
                let a_score = a.confidence - (a.total_cost * 0.1);
                let b_score = b.confidence - (b.total_cost * 0.1);
                a_score.partial_cmp(&b_score).unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Execute one step of the plan. Returns the step that was executed.
    /// The engine injects activation into the step's node to trigger firing.
    pub fn execute_step(&self, plan: &Plan, step_index: usize, engine: &parking_lot::Mutex<cognitive_core::ActivationEngine>) -> Option<PlanStep> {
        if step_index >= plan.steps.len() {
            return None;
        }
        let step = plan.steps[step_index].clone();
        let mut eng = engine.lock();
        eng.inject(step.node_id, 0.5);
        drop(eng);
        Some(step)
    }

    /// Check if a plan needs replanning based on prediction error.
    /// Returns true if the plan should be regenerated.
    pub fn should_replan(&self, plan: &Plan, engine: &parking_lot::Mutex<cognitive_core::ActivationEngine>) -> bool {
        let eng = engine.lock();
        let (n, a, _) = eng.read_modulators();
        // High novelty + high arousal = unexpected outcome → replan
        (n > self.replan_threshold && a > self.replan_threshold)
            || (1.0 - plan.confidence) > self.replan_threshold as f64
    }

    /// Create a plan node in the graph for persistence.
    pub fn materialize_plan(&self, plan: &Plan) -> NodeId {
        let mut graph = self.ctx.graph.write();
        let plan_id = graph.insert(GroundedNode {
            id: NodeId::ZERO,
            label: format!("plan_for_{}", plan.goal_id.0),
            node_type: NodeType::Plan,
            grounding: Grounding::Plan {
                status: PlanStatus::Pending,
                current_step: 0,
            },
            decay: 0.95,
            threshold: 10.0,
            base_activation: 0.0,
            edges: vec![Edge::new(Relation::Achieves, plan.goal_id)],
            epistemic_status: EpistemicStatus::CoreConcept,
            valence: 0.0,
            mean_error: 0.0,
            variance: 0.0,
        });
        // Link steps to plan
        for step in &plan.steps {
            if let Some(node) = graph.get(plan_id) {
                node.write().edges.push(Edge::new(Relation::StepInPlan, step.node_id));
            }
        }
        plan_id
    }
}
