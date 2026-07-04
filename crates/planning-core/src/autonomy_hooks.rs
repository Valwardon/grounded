use std::sync::Arc;
use semantic_graph::prelude::NodeId;
use cognitive_core::{
    GoalFormationHook, StrategicPlannerHook, ToolAbstractionHook,
    CrossDomainHook, PlanExecutionHook, PredictionError,
};

use crate::goal_formation::GoalFormationEngine;
use crate::strategic_planner::StrategicPlanner;
use crate::tool_abstraction::AffordanceRegistry;
use crate::cross_domain::{CrossDomainEngine, PlanExecutionEngine, ExecutionStatus};
use crate::planner::HierarchicalPlanner;

// ═══════════════════════════════════════════════════════════
//  Module A: GoalFormationHook for GoalFormationEngine
// ═══════════════════════════════════════════════════════════

impl GoalFormationHook for GoalFormationEngine {
    fn tick(
        &mut self,
        prediction_errors: &[PredictionError],
        activations: &[f64],
        novelty: f64,
        _arousal: f64,
        _reward: f64,
        drive_intensities: &[f64; 6],
        current_tick: u64,
    ) -> String {
        let result = GoalFormationEngine::tick(
            self,
            prediction_errors,
            activations,
            novelty,
            drive_intensities,
            current_tick,
        );
        match result {
            crate::goal_formation::GoalFormationResult::NewGoal { node_id, label, priority, reason } => {
                format!("[GOAL] Formed: {} (id={}, priority={:.2}, reason={:?})",
                    label, node_id.0, priority, reason)
            }
            crate::goal_formation::GoalFormationResult::None => String::new(),
        }
    }
}

// ═══════════════════════════════════════════════════════════
//  Module B: StrategicPlannerHook for StrategicPlanner
// ═══════════════════════════════════════════════════════════

impl StrategicPlannerHook for StrategicPlanner {
    fn should_replan(&self, prediction_errors: &[PredictionError]) -> bool {
        StrategicPlanner::should_replan(self, prediction_errors)
    }
    fn plan_cycle(&mut self, prediction_errors: &[PredictionError]) -> String {
        if !self.should_replan(prediction_errors) {
            return String::new();
        }

        // Scan the graph for active Goal nodes
        let graph = self.ctx().graph.read();
        let mut goal_ids: Vec<NodeId> = Vec::new();
        for i in 2..graph.len() {
            if let Some(n) = graph.get(NodeId::from_raw(i as u64)) {
                let node = n.read();
                if node.node_type == semantic_graph::prelude::NodeType::Goal {
                    goal_ids.push(node.id);
                }
            }
        }
        drop(graph);

        if goal_ids.is_empty() {
            return String::new();
        }

        // Plan for the highest-priority goal (first found)
        let goal_id = goal_ids[0];
        let hierarchical = HierarchicalPlanner::new(self.ctx().clone());
        let plan = self.plan_for_goal(goal_id, &hierarchical);

        format!(
            "[PLAN] MCTS replan for goal {}: {} steps, confidence={:.2}",
            goal_id.0, plan.steps.len(), plan.confidence
        )
    }
}

// ═══════════════════════════════════════════════════════════
//  Module C: ToolAbstractionHook for AffordanceRegistry
// ═══════════════════════════════════════════════════════════

impl ToolAbstractionHook for AffordanceRegistry {
    fn materialize_affordances(&mut self) -> usize {
        let before = self.count();
        AffordanceRegistry::materialize_affordances(self);
        self.count() - before
    }

    fn scan(&mut self) -> usize {
        let before = self.count();
        AffordanceRegistry::scan_graph(self);
        self.count() - before
    }

    fn count(&self) -> usize {
        AffordanceRegistry::count(self)
    }
}

// ═══════════════════════════════════════════════════════════
//  Module D: CrossDomainHook for CrossDomainEngine
// ═══════════════════════════════════════════════════════════

impl CrossDomainHook for CrossDomainEngine {
    fn learn_tick(&mut self) {
        CrossDomainEngine::learn_tick(self);
    }
}

// ═══════════════════════════════════════════════════════════
//  Module D: PlanExecutionHook for PlanExecutionEngine
// ═══════════════════════════════════════════════════════════

impl PlanExecutionHook for PlanExecutionEngine {
    fn is_active(&self) -> bool {
        PlanExecutionEngine::is_active(self)
    }

    fn injection_targets(&self) -> Vec<(u64, f64)> {
        PlanExecutionEngine::injection_targets(self)
            .into_iter()
            .map(|(id, energy)| (id.0, energy))
            .collect()
    }

    fn tick(&mut self, prediction_errors: &[PredictionError]) -> String {
        match PlanExecutionEngine::tick(self, prediction_errors) {
            Some(step) => {
                format!("[EXEC] Step: {} (cost={:.2})", step.label, step.expected_cost)
            }
            None => {
                match self.status() {
                    ExecutionStatus::Completed => "[EXEC] Plan completed successfully.".into(),
                    ExecutionStatus::Failed => "[EXEC] Plan execution failed.".into(),
                    ExecutionStatus::Waiting => "[EXEC] Waiting for conditions...".into(),
                    _ => String::new(),
                }
            }
        }
    }

    fn start_plan_for_goal(&mut self, _goal_id: u64) {
        // The plan execution engine needs a full Plan object.
        // In a real system, the strategic planner would provide it.
        // For the hook interface, this is a no-op — plans are started
        // by the strategic planner via start_plan().
    }

    fn pause(&mut self) {
        PlanExecutionEngine::pause(self);
    }

    fn resume(&mut self) {
        PlanExecutionEngine::resume(self);
    }

    fn status_description(&self) -> String {
        PlanExecutionEngine::status_description(self)
    }
}
