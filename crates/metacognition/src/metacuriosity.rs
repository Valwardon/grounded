// ────────────────────────────────────────────────────────────
//  Recursive Metacognitive Curiosity Loop
//
//  When the DeficiencyScanner detects performance bottlenecks or
//  persistent prediction errors, the CuriosityBudget is diverted
//  away from external world exploration and routed inward to fuel
//  the Self-Healing Pipeline.
//
//  The core insight:
//    - External curiosity: resolve gaps in world knowledge.
//    - Internal (metacognitive) curiosity: resolve gaps in the
//      engine's own performance — why is this module slow? Why
//      does the parser fail on this specific user input pattern?
//
//  The extended budget formula:
//
//    E_consume = 0.3 · dist(SELF, concept)
//              + 0.4 · arousal
//              + 0.2 · error_rate
//              + 0.1 · deficiency_severity    ← NEW: metacognitive term
//
//    deficiency_severity:
//      0.0 = no internal deficiencies
//      >0.0 = internal module constraint violation detected
//
//  When deficiency_severity exceeds a threshold, the curiosity
//  daemon's budget is redirected from external gap resolution
//  to internal module optimization.
// ────────────────────────────────────────────────────────────

use std::time::Duration;
use semantic_graph::prelude::*;
use curiosity_core::gap::CuriosityBudget;

use crate::capability::*;
use crate::pipeline::SelfHealingPipeline;

/// Weight of the metacognitive term in the curiosity budget formula.
pub const METACOGNITIVE_WEIGHT: f64 = 0.1;

/// Threshold of deficiency severity that triggers internal routing.
/// When severity > this threshold, at least 30% of the budget goes
/// to internal optimization instead of external exploration.
pub const INTERNAL_ROUTING_THRESHOLD: f32 = 0.3;

/// Minimum budget fraction to route internally when deficiency is active.
pub const MIN_INTERNAL_BUDGET_FRACTION: f64 = 0.3;

// ── Budget Allocator ──────────────────────────────────────

/// Determines how the curiosity budget is split between external
/// world exploration and internal module optimization.
pub struct MetacognitiveBudgetAllocator {
    /// Reference to the self-healing pipeline for deficiency status.
    pipeline: Option<*const SelfHealingPipeline>,

    /// Cached deficiency severity from the last scan.
    last_deficiency_severity: f32,

    /// Total internal budget consumed so far.
    internal_budget_consumed: f64,

    /// Total external budget consumed so far.
    external_budget_consumed: f64,
}

// Safety: the raw pointer is only used within the same thread.
unsafe impl Send for MetacognitiveBudgetAllocator {}
unsafe impl Sync for MetacognitiveBudgetAllocator {}

impl MetacognitiveBudgetAllocator {
    pub fn new() -> Self {
        MetacognitiveBudgetAllocator {
            pipeline: None,
            last_deficiency_severity: 0.0,
            internal_budget_consumed: 0.0,
            external_budget_consumed: 0.0,
        }
    }

    /// Bind to a self-healing pipeline for deficiency status queries.
    pub fn bind(&mut self, pipeline: &SelfHealingPipeline) {
        self.pipeline = Some(pipeline as *const SelfHealingPipeline);
    }

    /// Current deficiency severity (0.0 = no deficiency, 1.0 = critical).
    pub fn deficiency_severity(&self) -> f32 {
        // Query the pipeline scanner for pending deficiencies
        if let Some(ptr) = self.pipeline {
            // SAFETY: single-threaded access during idle cycle
            let pipeline = unsafe { &*ptr };
            let reports = pipeline.scanner().pending_reports();
            if reports.is_empty() {
                return 0.0;
            }
            // Return the maximum severity across all pending deficiencies
            reports.iter()
                .map(|r| r.severity)
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(0.0)
        } else {
            0.0
        }
    }

    /// Compute the metacognitive term for the extended budget formula.
    ///
    ///   deficiency_term = deficiency_severity * METACOGNITIVE_WEIGHT
    ///
    /// This is added to the standard CuriosityBudget::step_cost() result.
    pub fn metacognitive_term(&self) -> f64 {
        self.last_deficiency_severity as f64 * METACOGNITIVE_WEIGHT
    }

    /// Extended step cost that includes the metacognitive deficiency term.
    ///
    /// Use this instead of `CuriosityBudget::step_cost()` when the
    /// metacognitive allocator is active.
    pub fn extended_step_cost(
        &self,
        semantic_distance: f64,
        arousal: f64,
        novelty: f64,
        error_rate: f64,
    ) -> f64 {
        0.3 * semantic_distance
            + 0.4 * arousal
            + 0.2 * error_rate
            + self.metacognitive_term()
    }

    /// Return true if the deficiency is severe enough to route budget inward.
    pub fn should_route_internal(&self) -> bool {
        self.last_deficiency_severity > INTERNAL_ROUTING_THRESHOLD
    }

    /// Determine how much of the remaining budget goes to internal
    /// (self-healing) vs external (world exploration).
    ///
    /// Returns (internal_fraction, external_fraction).
    pub fn allocation_split(&self) -> (f64, f64) {
        if self.should_route_internal() {
            // Route significant budget inward
            let internal = MIN_INTERNAL_BUDGET_FRACTION
                + (self.last_deficiency_severity as f64) * 0.3;
            let internal = internal.min(0.8); // cap at 80%
            (internal, 1.0 - internal)
        } else {
            // No deficiency: all budget goes to external exploration
            (0.0, 1.0)
        }
    }

    /// Consume budget for an internal (self-healing) operation.
    /// Returns true if there was enough budget remaining.
    pub fn consume_internal(&mut self, budget: &mut CuriosityBudget, amount: f64) -> bool {
        if budget.remaining >= amount {
            budget.remaining -= amount;
            self.internal_budget_consumed += amount;
            true
        } else {
            false
        }
    }

    /// Consume budget for an external (world exploration) operation.
    /// Returns true if there was enough budget remaining.
    pub fn consume_external(&mut self, budget: &mut CuriosityBudget, amount: f64) -> bool {
        // Apply the external fraction to the amount
        let (_, ext_frac) = self.allocation_split();
        let effective_amount = amount * ext_frac;
        if budget.remaining >= effective_amount {
            budget.remaining -= effective_amount;
            self.external_budget_consumed += effective_amount;
            true
        } else {
            false
        }
    }

    /// Refresh the cached deficiency severity from the pipeline.
    pub fn refresh(&mut self) {
        self.last_deficiency_severity = self.deficiency_severity();
    }

    /// Total budget consumed (internal + external).
    pub fn total_consumed(&self) -> f64 {
        self.internal_budget_consumed + self.external_budget_consumed
    }

    /// Ratio of budget spent internally (0.0–1.0).
    pub fn internal_ratio(&self) -> f64 {
        let total = self.total_consumed();
        if total == 0.0 {
            0.0
        } else {
            self.internal_budget_consumed / total
        }
    }
}

// ── Integrated Curiosity Daemon Extension ─────────────────

/// Extension that wraps an existing CuriosityBudget with metacognitive
/// awareness. This is the bridge between the cognitive daemon's
/// curiosity loop and the metacognitive self-healing system.
pub struct MetacognitiveCuriosity {
    /// Base budget for curiosity operations.
    pub budget: CuriosityBudget,

    /// The metacognitive allocator that splits budget.
    allocator: MetacognitiveBudgetAllocator,

    /// How many ticks to wait between deficiency scanner refreshes.
    scan_interval_ticks: u64,

    /// Ticks since last scan.
    ticks_since_scan: u64,

    /// Whether internal optimization is currently active.
    pub internal_optimization_active: bool,
}

impl MetacognitiveCuriosity {
    pub fn new(total_energy: f64) -> Self {
        MetacognitiveCuriosity {
            budget: CuriosityBudget::new(total_energy),
            allocator: MetacognitiveBudgetAllocator::new(),
            scan_interval_ticks: 100, // refresh every ~1.6 seconds
            ticks_since_scan: 0,
            internal_optimization_active: false,
        }
    }

    /// Bind to a self-healing pipeline.
    pub fn bind(&mut self, pipeline: &SelfHealingPipeline) {
        self.allocator.bind(pipeline);
    }

    /// Call every cognitive tick.
    /// Returns true if the budget should be routed to the self-healing
    /// pipeline this tick.
    pub fn tick(&mut self) -> bool {
        self.ticks_since_scan += 1;

        if self.ticks_since_scan >= self.scan_interval_ticks {
            self.allocator.refresh();
            self.ticks_since_scan = 0;

            if self.allocator.should_route_internal() {
                // Activate internal optimization mode
                self.internal_optimization_active = true;
                return true;
            }
        }

        // Deactivate internal mode if deficiency resolved
        if self.internal_optimization_active && !self.allocator.should_route_internal() {
            self.internal_optimization_active = false;
        }

        self.internal_optimization_active
    }

    /// Extended step cost including the metacognitive term.
    pub fn step_cost(&self, semantic_distance: f64, arousal: f64, novelty: f64, error_rate: f64) -> f64 {
        self.allocator.extended_step_cost(semantic_distance, arousal, novelty, error_rate)
    }

    /// Get the current allocation split.
    /// Returns (internal_fraction, external_fraction).
    pub fn allocation(&self) -> (f64, f64) {
        self.allocator.allocation_split()
    }

    /// Check if the budget is too low for the metacognitive pipeline to run.
    pub fn can_run_pipeline(&self) -> bool {
        let (int_frac, _) = self.allocation();
        let required = int_frac * 0.5; // 50% of internal allocation
        self.budget.remaining >= required
    }

    /// Allocate budget for a self-healing pipeline run.
    pub fn allocate_pipeline_run(&mut self) -> bool {
        let (int_frac, _) = self.allocation();
        let pipeline_cost = int_frac * 0.5;
        if self.budget.remaining >= pipeline_cost {
            self.budget.remaining -= pipeline_cost;
            true
        } else {
            false
        }
    }
}
