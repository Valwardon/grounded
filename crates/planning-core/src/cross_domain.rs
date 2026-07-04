use std::sync::Arc;
use semantic_graph::prelude::*;
use crate::planner::{Plan, PlanStep};

/// ────────────────────────────────────────────────────────────
///  Cross-Domain Runtime + Plan Execution Engine
///
///  Two subsystems in one file:
///
///  1. CrossDomainEngine — projects concepts between domain subspaces
///     using PrimitiveVector signatures. Enables zero-shot transfer:
///     "if I know how to push a cube in geometry, I can push a box
///      in physics" by projecting the action vector through the
///      geometric→kinematic DomainMapping matrix.
///
///  2. PlanExecutionEngine — executes plans step-by-step with
///     DMN-based pause/resume, parallel branch execution, and
///     automatic step advancement on low prediction error.
///
///  All operations are deterministic — matrix multiplication and
///  cosine similarity only. No ML, no RNG.
/// ────────────────────────────────────────────────────────────

// ═══════════════════════════════════════════════════════════
//  1. CROSS-DOMAIN ENGINE
// ═══════════════════════════════════════════════════════════

/// The 5 PrimitiveVector dimensions used as domain subspace basis.
pub const DOMAIN_DIMS: usize = 5;

/// A domain mapping — describes how concepts in the source domain
/// project into the target domain.
#[derive(Debug, Clone)]
pub struct DomainMapping {
    /// Human-readable name of the source domain (e.g., "geometry").
    pub source_domain: String,
    /// Human-readable name of the target domain (e.g., "kinematics").
    pub target_domain: String,
    /// 5×5 mapping matrix: target = matrix × source.
    /// Initialized as identity (no transformation) then learned
    /// from structural edge overlap between domains.
    pub matrix: [[f64; DOMAIN_DIMS]; DOMAIN_DIMS],
    /// Confidence in this mapping (0.0 = untrusted, 1.0 = validated).
    pub confidence: f64,
}

impl DomainMapping {
    pub fn new(source: &str, target: &str) -> Self {
        DomainMapping {
            source_domain: source.to_string(),
            target_domain: target.to_string(),
            matrix: Self::identity_matrix(),
            confidence: 0.5,
        }
    }

    /// Identity matrix — projects source→target without change.
    fn identity_matrix() -> [[f64; DOMAIN_DIMS]; DOMAIN_DIMS] {
        let mut m = [[0.0; DOMAIN_DIMS]; DOMAIN_DIMS];
        for i in 0..DOMAIN_DIMS {
            m[i][i] = 1.0;
        }
        m
    }

    /// Project a source PrimitiveVector through this mapping into the target domain.
    pub fn project(&self, source: &PrimitiveVector) -> PrimitiveVector {
        let src = [source.mass, source.velocity, source.spatial, source.valence, source.temporal];
        let mut tgt = [0.0; DOMAIN_DIMS];
        for i in 0..DOMAIN_DIMS {
            for j in 0..DOMAIN_DIMS {
                tgt[i] += self.matrix[i][j] * src[j];
            }
        }
        PrimitiveVector::new(
            tgt[0].clamp(-1.0, 1.0),
            tgt[1].clamp(-1.0, 1.0),
            tgt[2].clamp(-1.0, 1.0),
            tgt[3].clamp(-1.0, 1.0),
            tgt[4].clamp(-1.0, 1.0),
        )
    }

    /// Update the mapping matrix based on structural overlap between domains.
    ///
    /// For each pair of concepts that exist in both domains, the mapping should
    /// map source primitives to target primitives. This is computed by
    /// correlating the PrimitiveVectors of shared concepts.
    pub fn learn_from_overlap(&mut self, ctx: &SemanticContext) {
        let graph = ctx.graph.read();
        let mut src_vectors: Vec<PrimitiveVector> = Vec::new();
        let mut tgt_vectors: Vec<PrimitiveVector> = Vec::new();

        for i in 2..graph.len() {
            if let Some(n) = graph.get(NodeId::from_raw(i as u64)) {
                let node = n.read();
                // Check if node belongs to both domains via DomainOf edges
                let mut in_source = false;
                let mut in_target = false;
                for edge in &node.edges {
                    if edge.relation == Relation::DomainOf {
                        if let Some(domain_node) = graph.get(edge.target) {
                            let domain_label = domain_node.read().label.clone();
                            if domain_label == self.source_domain {
                                in_source = true;
                            }
                            if domain_label == self.target_domain {
                                in_target = true;
                            }
                        }
                    }
                }
                if in_source && in_target {
                    if let Some(pv) = primitive_for(&node.label) {
                        src_vectors.push(pv);
                        tgt_vectors.push(pv); // Same concept, different domain framing
                    }
                }
            }
        }

        if src_vectors.len() < 2 {
            return; // Not enough overlap data
        }

        // Compute average displacement between source and target vectors
        let mut displacement = [0.0; DOMAIN_DIMS];
        for (src, tgt) in src_vectors.iter().zip(tgt_vectors.iter()) {
            let s = [src.mass, src.velocity, src.spatial, src.valence, src.temporal];
            let t = [tgt.mass, tgt.velocity, tgt.spatial, tgt.valence, tgt.temporal];
            for i in 0..DOMAIN_DIMS {
                displacement[i] += t[i] - s[i];
            }
        }
        let n = src_vectors.len() as f64;
        for i in 0..DOMAIN_DIMS {
            displacement[i] /= n;
        }

        // Update matrix diagonal to reflect learned displacement
        for i in 0..DOMAIN_DIMS {
            self.matrix[i][i] = (1.0 + displacement[i]).clamp(0.0, 2.0);
        }

        // Update confidence based on data available
        self.confidence = (self.confidence + n * 0.01).clamp(0.0, 1.0);
    }
}

impl Default for DomainMapping {
    fn default() -> Self {
        DomainMapping::new("unknown", "unknown")
    }
}

/// The cross-domain engine — manages domain mappings and subspace projections.
///
/// Each concept in the semantic graph can be assigned a domain label via
/// DomainOf edges. The cross-domain engine maintains mappings between
/// domains and can project concepts from one domain to another.
pub struct CrossDomainEngine {
    ctx: Arc<SemanticContext>,
    /// Registered domain mappings.
    mappings: Vec<DomainMapping>,
}

impl CrossDomainEngine {
    pub fn new(ctx: Arc<SemanticContext>) -> Self {
        CrossDomainEngine {
            ctx,
            mappings: Vec::with_capacity(8),
        }
    }

    /// Register a domain mapping.
    pub fn register_mapping(&mut self, mapping: DomainMapping) {
        // Replace existing mapping for same domain pair
        if let Some(existing) = self.mappings.iter_mut()
            .find(|m| m.source_domain == mapping.source_domain && m.target_domain == mapping.target_domain)
        {
            *existing = mapping;
        } else {
            self.mappings.push(mapping);
        }
    }

    /// Find a mapping from source to target domain.
    pub fn find_mapping(&self, source: &str, target: &str) -> Option<&DomainMapping> {
        self.mappings.iter()
            .find(|m| m.source_domain == source && m.target_domain == target)
    }

    /// Get a mutable mapping for learning.
    pub fn find_mapping_mut(&mut self, source: &str, target: &str) -> Option<&mut DomainMapping> {
        self.mappings.iter_mut()
            .find(|m| m.source_domain == source && m.target_domain == target)
    }

    /// Project a concept from source domain to target domain.
    ///
    /// Returns the projected PrimitiveVector and the mapping confidence.
    pub fn project_concept(&self, concept_label: &str, source: &str, target: &str) -> Option<(PrimitiveVector, f64)> {
        let pv = primitive_for(concept_label)?;
        let mapping = self.find_mapping(source, target)?;
        let projected = mapping.project(&pv);
        Some((projected, mapping.confidence))
    }

    /// Compute domain similarity between two concept labels.
    /// Uses cosine similarity of their PrimitiveVectors.
    pub fn domain_similarity(&self, concept_a: &str, concept_b: &str) -> f64 {
        let pv_a = primitive_for(concept_a).unwrap_or(PrimitiveVector::zero());
        let pv_b = primitive_for(concept_b).unwrap_or(PrimitiveVector::zero());
        let dot = pv_a.mass * pv_b.mass
            + pv_a.velocity * pv_b.velocity
            + pv_a.spatial * pv_b.spatial
            + pv_a.valence * pv_b.valence
            + pv_a.temporal * pv_b.temporal;
        let norm_a = (pv_a.mass.powi(2) + pv_a.velocity.powi(2)
            + pv_a.spatial.powi(2) + pv_a.valence.powi(2)
            + pv_a.temporal.powi(2)).sqrt();
        let norm_b = (pv_b.mass.powi(2) + pv_b.velocity.powi(2)
            + pv_b.spatial.powi(2) + pv_b.valence.powi(2)
            + pv_b.temporal.powi(2)).sqrt();
        if norm_a < 1e-10 || norm_b < 1e-10 {
            return 0.0;
        }
        (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }

    /// Assign a domain label to a concept node in the graph.
    pub fn assign_domain(&mut self, concept_id: NodeId, domain_label: &str) {
        let mut graph = self.ctx.graph.write();
        // Find or create domain concept node
        let domain_id = match graph.find_by_label(domain_label) {
            Some(id) => id,
            None => {
                let id = graph.insert(GroundedNode {
                    id: NodeId::ZERO,
                    label: domain_label.to_string(),
                    node_type: NodeType::Concept,
                    grounding: Grounding::Abstract,
                    decay: 0.99,
                    threshold: 25.0,
                    base_activation: 0.0,
                    edges: Vec::new(),
                    epistemic_status: EpistemicStatus::CoreConcept,
                    valence: 0.0,
                    mean_error: 0.0,
                    variance: 0.0,
                });
                id
            }
        };
        // Add DomainOf edge
        if let Some(node) = graph.get(concept_id) {
            let already_linked = node.read().edges.iter()
                .any(|e| e.target == domain_id && e.relation == Relation::DomainOf);
            if !already_linked {
                node.write().edges.push(Edge::new(Relation::DomainOf, domain_id));
            }
        }
    }

    /// Run one learning tick — scan for concept overlap between
    /// domain pairs and update mapping matrices.
    pub fn learn_tick(&mut self) {
        let mapping_count = self.mappings.len();
        let mapping_indices: Vec<usize> = (0..mapping_count).collect();
        for idx in mapping_indices {
            if let Some(mapping) = self.mappings.get_mut(idx) {
                mapping.learn_from_overlap(&self.ctx);
            }
        }
    }

    /// Get all registered domain labels from the graph.
    pub fn known_domains(&self) -> Vec<String> {
        self.ctx.graph.read().nodes.iter()
            .filter(|n| {
                n.read().edges.iter().any(|e| e.relation == Relation::DomainOf)
            })
            .map(|n| n.read().label.clone())
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════
//  2. PLAN EXECUTION ENGINE
// ═══════════════════════════════════════════════════════════

/// Status of the plan execution engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionStatus {
    /// No active plan.
    Idle,
    /// Actively executing plan steps.
    Running,
    /// Waiting for a step's conditions to be met.
    Waiting,
    /// Paused (DMN override or prediction error spike).
    Paused,
    /// Plan completed successfully.
    Completed,
    /// Plan failed (irrecoverable step failure).
    Failed,
}

/// A single parallel branch in the execution tree.
#[derive(Debug, Clone)]
pub struct ExecutionBranch {
    /// Index into this branch's step sequence.
    pub step_index: usize,
    /// Steps unique to this branch.
    pub steps: Vec<PlanStep>,
    /// Whether this branch has completed.
    pub completed: bool,
    /// Whether this branch has failed.
    pub failed: bool,
}

/// Plan execution state machine — manages step-by-step advancement
/// with DMN pause/resume, parallel branch execution, and condition
/// checking (low prediction error → advance).
pub struct PlanExecutionEngine {
    /// The currently executing plan (None if idle).
    current_plan: Option<Plan>,
    /// Index of the current step in the main execution sequence.
    current_step: usize,
    /// Current execution status.
    status: ExecutionStatus,
    /// Parallel execution branches (for plans with SubGoalOf forks).
    branches: Vec<ExecutionBranch>,
    /// Tick when the current step started (for timeout detection).
    step_start_tick: u64,
    /// Maximum ticks to wait for a step before marking it failed.
    step_timeout_ticks: u64,
    /// Reference to ctx for graph operations.
    ctx: Arc<SemanticContext>,
}

impl PlanExecutionEngine {
    pub fn new(ctx: Arc<SemanticContext>) -> Self {
        PlanExecutionEngine {
            current_plan: None,
            current_step: 0,
            status: ExecutionStatus::Idle,
            branches: Vec::new(),
            step_start_tick: 0,
            step_timeout_ticks: 1000,
            ctx,
        }
    }

    /// Start executing a plan.
    pub fn start_plan(&mut self, plan: Plan) {
        self.current_plan = Some(plan);
        self.current_step = 0;
        self.status = ExecutionStatus::Running;
        self.step_start_tick = self.ctx.tick.load(std::sync::atomic::Ordering::Relaxed);
        self.branches.clear();
    }

    /// Pause execution (e.g., on high prediction error / DMN switch).
    pub fn pause(&mut self) {
        if self.status == ExecutionStatus::Running || self.status == ExecutionStatus::Waiting {
            self.status = ExecutionStatus::Paused;
        }
    }

    /// Resume execution after pause.
    pub fn resume(&mut self) {
        if self.status == ExecutionStatus::Paused {
            self.status = ExecutionStatus::Running;
        }
    }

    /// Cancel the current plan.
    pub fn cancel(&mut self) {
        self.current_plan = None;
        self.current_step = 0;
        self.status = ExecutionStatus::Idle;
        self.branches.clear();
    }

    /// Mark the current plan as completed.
    pub fn complete(&mut self) {
        self.status = ExecutionStatus::Completed;
    }

    /// Tick the execution engine — advance steps based on conditions.
    ///
    /// Called every cognitive tick. Returns the step to inject activation
    /// into, or None if no action is needed.
    ///
    /// Conditions for advancing to the next step:
    ///   - Current step has low prediction error (< 0.2)
    ///   - Current step hasn't timed out
    ///   - Engine is in Running state
    pub fn tick(&mut self, prediction_errors: &[PredictionError]) -> Option<PlanStep> {
        let plan = match &self.current_plan {
            Some(p) => p,
            None => return None,
        };

        if self.status != ExecutionStatus::Running {
            return None;
        }

        // Check for timeout on current step
        let current_tick = self.ctx.tick.load(std::sync::atomic::Ordering::Relaxed);
        if self.step_start_tick > 0
            && current_tick > self.step_start_tick + self.step_timeout_ticks
        {
            // Current step timed out — advance anyway
            self.current_step += 1;
            self.step_start_tick = current_tick;
            if self.current_step >= plan.steps.len() {
                self.status = ExecutionStatus::Completed;
                return None;
            }
        }

        // Check prediction errors for this step's node
        let step = &plan.steps[self.current_step];
        let step_has_error = prediction_errors.iter()
            .any(|e| e.node_id == step.node_id && e.error_magnitude > 0.2);

        if step_has_error {
            // Step has prediction error — pause and wait
            self.status = ExecutionStatus::Waiting;
            return None;
        }

        // Advance to next step if no errors
        if self.current_step < plan.steps.len() {
            let step = plan.steps[self.current_step].clone();
            self.current_step += 1;
            self.step_start_tick = current_tick;

            // Check if plan is complete
            if self.current_step >= plan.steps.len() {
                self.status = ExecutionStatus::Completed;
                // Return the last step for execution before completing
            }

            // Execute parallel branches if this step has SubGoalOf children
            self.check_branch_forks(step.node_id);

            Some(step)
        } else {
            None
        }
    }

    /// Scan for SubGoalOf relations from the given node and fork branches.
    fn check_branch_forks(&mut self, node_id: NodeId) {
        let graph = self.ctx.graph.read();
        if let Some(node) = graph.get(node_id) {
            let guard = node.read();
            let subgoals: Vec<NodeId> = guard.edges.iter()
                .filter(|e| e.relation == Relation::SubGoalOf)
                .map(|e| e.target)
                .collect();
            if !subgoals.is_empty() {
                // Fork a new branch for each subgoal
                for subgoal_id in subgoals {
                    let label = graph.label_of(subgoal_id).unwrap_or_default();
                    let branch_step = PlanStep {
                        label: format!("branch:{}", label),
                        node_id: subgoal_id,
                        expected_cost: 0.5,
                        expected_outcome: 0.5,
                        is_action: false,
                    };
                    self.branches.push(ExecutionBranch {
                        step_index: 0,
                        steps: vec![branch_step],
                        completed: false,
                        failed: false,
                    });
                }
            }
        }
    }

    /// Get the activation injection targets for the current tick.
    ///
    /// Returns the main step's node (if any) plus any active
    /// branch step nodes for parallel execution.
    pub fn injection_targets(&self) -> Vec<(NodeId, f64)> {
        let mut targets: Vec<(NodeId, f64)> = Vec::new();

        // Main plan step
        if let Some(plan) = &self.current_plan {
            if self.current_step < plan.steps.len() && self.status == ExecutionStatus::Running {
                targets.push((plan.steps[self.current_step].node_id, 0.5));
            }
        }

        // Parallel branches
        for branch in &self.branches {
            if !branch.completed && !branch.failed && branch.step_index < branch.steps.len() {
                targets.push((branch.steps[branch.step_index].node_id, 0.3));
            }
        }

        targets
    }

    /// Mark the current step as failed.
    pub fn fail_step(&mut self) {
        if let Some(plan) = &self.current_plan {
            if self.current_step < plan.steps.len() {
                self.current_step += 1;
                // If no more steps, mark plan as failed
                if self.current_step >= plan.steps.len() {
                    self.status = ExecutionStatus::Failed;
                }
            }
        }
    }

    /// Current execution status.
    pub fn status(&self) -> ExecutionStatus {
        self.status
    }

    /// The current plan and step index.
    pub fn current_plan(&self) -> Option<(&Plan, usize)> {
        self.current_plan.as_ref().map(|p| (p, self.current_step))
    }

    /// Check if the engine is actively executing.
    pub fn is_active(&self) -> bool {
        self.status == ExecutionStatus::Running || self.status == ExecutionStatus::Waiting
    }

    /// Human-readable status description.
    pub fn status_description(&self) -> String {
        match self.status {
            ExecutionStatus::Idle => "Idle".to_string(),
            ExecutionStatus::Running => {
                if let Some(plan) = &self.current_plan {
                    format!("Running: step {}/{}", self.current_step + 1, plan.steps.len())
                } else {
                    "Running (no plan)".to_string()
                }
            }
            ExecutionStatus::Waiting => "Waiting for conditions".to_string(),
            ExecutionStatus::Paused => "Paused".to_string(),
            ExecutionStatus::Completed => "Completed".to_string(),
            ExecutionStatus::Failed => "Failed".to_string(),
        }
    }
}
