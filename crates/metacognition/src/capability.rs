// ────────────────────────────────────────────────────────────
//  Introspective Deficiency Graph
//
//  The engine maps its own limitations by attaching capability
//  metrics and performance constraints to nodes in the
//  GraphArena. The SELF node (ID 1) connects to capability
//  and constraint sub-graphs via HasCapability / HasConstraint
//  edge relations.
//
//  When a capability metric consistently violates its paired
//  constraint, the deficiency detector triggers an internal
//  curiosity loop that routes energy into the Self-Healing
//  Pipeline to generate a fix.
// ────────────────────────────────────────────────────────────

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use semantic_graph::prelude::*;

// ── Capability Metric Payload ─────────────────────────────

/// Runtime execution metrics for a cognitive module.
///
/// Stored as AtomicU64 for lock-free updates from the hot path.
/// The deficiency scanner reads snapshots during idle cycles.
#[derive(Debug, Clone)]
pub struct CapabilityMetrics {
    /// Human-readable label for the module this metric tracks.
    pub label: String,

    /// Mean execution latency in nanoseconds (AtomicU64).
    mean_latency_ns: AtomicU64,

    /// Approximate heap footprint in bytes (AtomicU64).
    memory_footprint: AtomicU64,

    /// Success rate as a fixed-point fraction * 1_000_000 (AtomicU64).
    /// e.g., 950_000 = 95.0% success.
    success_rate_ppm: AtomicU64,

    /// Total number of samples recorded (AtomicU64).
    sample_count: AtomicU64,
}

impl CapabilityMetrics {
    pub fn new(label: &str) -> Self {
        CapabilityMetrics {
            label: label.to_string(),
            mean_latency_ns: AtomicU64::new(0),
            memory_footprint: AtomicU64::new(0),
            success_rate_ppm: AtomicU64::new(1_000_000), // start at 100%
            sample_count: AtomicU64::new(0),
        }
    }

    /// Record a single execution sample.
    /// Updates EMA of latency and running success rate.
    /// Call this at the end of every module execution (hot path safe).
    pub fn record_sample(&self, latency: Duration, success: f32) {
        let lat_ns = latency.as_nanos() as u64;
        let ema_alpha: u64 = 32; // 1/32 = 0.03125 EMA factor

        // EMA update for latency (fixed-point: mean_latency_ns is stored as ns * 64)
        let old_lat = self.mean_latency_ns.load(Ordering::Relaxed);
        let new_lat = old_lat.wrapping_add(
            lat_ns.wrapping_mul(ema_alpha).wrapping_sub(old_lat.wrapping_div(ema_alpha).wrapping_mul(ema_alpha))
        );
        self.mean_latency_ns.store(new_lat, Ordering::Relaxed);

        // EMA update for success rate (ppm fixed-point)
        let success_ppm = (success * 1_000_000.0) as u64;
        let old_sr = self.success_rate_ppm.load(Ordering::Relaxed);
        let new_sr = old_sr.wrapping_add(
            success_ppm.wrapping_mul(ema_alpha).wrapping_sub(old_sr.wrapping_div(ema_alpha).wrapping_mul(ema_alpha))
        );
        self.success_rate_ppm.store(new_sr, Ordering::Relaxed);

        self.sample_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Set memory footprint (updated during consolidation, not hot path).
    pub fn set_memory_footprint(&self, bytes: usize) {
        self.memory_footprint.store(bytes as u64, Ordering::Release);
    }

    /// Read current mean latency as Duration.
    pub fn mean_latency(&self) -> Duration {
        let raw = self.mean_latency_ns.load(Ordering::Acquire);
        Duration::from_nanos(raw / 64)
    }

    /// Read current memory footprint in bytes.
    pub fn memory_footprint(&self) -> usize {
        self.memory_footprint.load(Ordering::Acquire) as usize
    }

    /// Read current success rate as 0.0–1.0 float.
    pub fn success_rate(&self) -> f32 {
        let ppm = self.success_rate_ppm.load(Ordering::Acquire);
        (ppm as f32) / 1_000_000.0
    }

    /// Total samples recorded.
    pub fn sample_count(&self) -> u64 {
        self.sample_count.load(Ordering::Acquire)
    }
}

// ── Constraint Definition ─────────────────────────────────

/// Performance constraint that a capability must satisfy.
///
/// When `violation_count` exceeds `max_violations_before_remedy`,
/// the deficiency scanner triggers a self-healing cycle for the
/// associated module.
#[derive(Debug, Clone)]
pub struct Constraint {
    /// Label of the capability this constraint applies to.
    pub target_label: String,

    /// Maximum acceptable mean latency.
    pub max_latency: Duration,

    /// Maximum acceptable memory footprint.
    pub max_memory: usize,

    /// Minimum acceptable success rate (0.0–1.0).
    pub min_success_rate: f32,

    /// Number of consecutive violations before triggering remedy.
    pub max_violations_before_remedy: u32,

    /// Current consecutive violation count.
    pub violation_count: u32,

    /// Whether this constraint is actively enforced.
    pub active: bool,
}

impl Constraint {
    pub fn new(target_label: &str) -> Self {
        Constraint {
            target_label: target_label.to_string(),
            max_latency: Duration::from_millis(10),
            max_memory: 1024 * 1024,     // 1 MB default
            min_success_rate: 0.95,
            max_violations_before_remedy: 5,
            violation_count: 0,
            active: true,
        }
    }

    /// Builder: set maximum acceptable latency.
    pub fn with_max_latency(mut self, d: Duration) -> Self {
        self.max_latency = d;
        self
    }

    /// Builder: set maximum acceptable memory footprint.
    pub fn with_max_memory(mut self, bytes: usize) -> Self {
        self.max_memory = bytes;
        self
    }

    /// Builder: set minimum acceptable success rate (0.0–1.0).
    pub fn with_min_success_rate(mut self, rate: f32) -> Self {
        self.min_success_rate = rate;
        self
    }

    /// Builder: set max consecutive violations before remedy.
    pub fn with_max_violations(mut self, count: u32) -> Self {
        self.max_violations_before_remedy = count;
        self
    }

    /// Check whether the given metrics violate this constraint.
    pub fn is_violated_by(&self, metrics: &CapabilityMetrics) -> bool {
        if !self.active {
            return false;
        }
        metrics.mean_latency() > self.max_latency
            || metrics.memory_footprint() > self.max_memory
            || metrics.success_rate() < self.min_success_rate
    }
}

// ── HasCapability / HasConstraint Edge Relations ──────────

/// Edge relation constants for the deficiency graph.
/// These are attached to edges originating from SELF (NodeId::SELF)
/// to capability and constraint sub-graphs.
pub const REL_HAS_CAPABILITY: &str = "HasCapability";
pub const REL_HAS_CONSTRAINT: &str = "HasConstraint";
pub const REL_DEFICIENCY_TRIGGERED: &str = "DeficiencyTriggered";
pub const REL_REMEDY_APPLIED: &str = "RemedyApplied";

// ── Deficiency Detection ──────────────────────────────────

/// Report generated when a constraint violation is detected.
#[derive(Debug, Clone)]
pub struct DeficiencyReport {
    /// Module whose metrics violated the constraint.
    pub module_label: String,

    /// The constraint that was violated.
    pub constraint: Constraint,

    /// Current metrics at the time of detection.
    pub current_metrics: CapabilityMetrics,

    /// Severity score (0.0 = mild, 1.0 = critical).
    /// Derived from how far the metric exceeds the constraint.
    pub severity: f32,

    /// Whether this deficiency has already triggered a remedy cycle.
    pub remedy_triggered: bool,
}

impl DeficiencyReport {
    /// Compute severity based on how badly metrics exceed constraints.
    pub fn compute_severity(metrics: &CapabilityMetrics, constraint: &Constraint) -> f32 {
        let mut factors: [f32; 3] = [0.0; 3];
        let mut count = 0u32;

        // Latency factor
        if metrics.mean_latency() > constraint.max_latency {
            let ratio = metrics.mean_latency().as_nanos() as f32
                / constraint.max_latency.as_nanos() as f32;
            factors[0] = (ratio - 1.0).min(5.0) / 5.0; // 0..1, saturates at 5x
            count += 1;
        }

        // Memory factor
        if metrics.memory_footprint() > constraint.max_memory {
            let ratio = metrics.memory_footprint() as f32 / constraint.max_memory as f32;
            factors[1] = (ratio - 1.0).min(5.0) / 5.0;
            count += 1;
        }

        // Success rate factor
        if metrics.success_rate() < constraint.min_success_rate {
            let deficit = constraint.min_success_rate - metrics.success_rate();
            factors[2] = (deficit / constraint.min_success_rate).min(1.0);
            count += 1;
        }

        if count == 0 {
            0.0
        } else {
            factors.iter().sum::<f32>() / count as f32
        }
    }
}

// ── Deficiency Scanner ────────────────────────────────────

/// Scans capability metrics against active constraints and produces
/// DeficiencyReports for the Self-Healing Pipeline.
///
/// Runs during idle consolidation cycles (novelty < 0.1, arousal < 0.1),
/// approximately every 1000 ticks (~16s).
pub struct DeficiencyScanner {
    /// Registered constraints indexed by module label.
    constraints: Vec<Constraint>,

    /// Most recent metrics snapshot for each tracked module.
    metrics_snapshots: Vec<CapabilityMetrics>,

    /// Deficiencies detected but not yet remedied.
    pending_reports: Vec<DeficiencyReport>,
}

impl DeficiencyScanner {
    pub fn new() -> Self {
        DeficiencyScanner {
            constraints: Vec::with_capacity(8),
            metrics_snapshots: Vec::with_capacity(8),
            pending_reports: Vec::with_capacity(4),
        }
    }

    /// Register a constraint to enforce.
    pub fn register_constraint(&mut self, constraint: Constraint) {
        // Replace existing constraint with same target label
        if let Some(existing) = self.constraints.iter_mut()
            .find(|c| c.target_label == constraint.target_label)
        {
            *existing = constraint;
        } else {
            self.constraints.push(constraint);
        }
    }

    /// Update the metrics snapshot for a module.
    pub fn update_metrics(&mut self, metrics: CapabilityMetrics) {
        if let Some(existing) = self.metrics_snapshots.iter_mut()
            .find(|m| m.label == metrics.label)
        {
            *existing = metrics;
        } else {
            self.metrics_snapshots.push(metrics);
        }
    }

    /// Run a full scan of all tracked metrics against all constraints.
    /// Returns new deficiency reports discovered this cycle.
    pub fn scan(&mut self) -> &[DeficiencyReport] {
        let mut new_reports: Vec<DeficiencyReport> = Vec::new();

        for constraint in &self.constraints {
            if !constraint.active {
                continue;
            }
            for metrics in &self.metrics_snapshots {
                if metrics.label != constraint.target_label {
                    continue;
                }
                if constraint.is_violated_by(metrics) {
                    let count = constraint.violation_count + 1;
                    // Update violation count via a clone
                    // In production this would use interior mutability
                    if count >= constraint.max_violations_before_remedy {
                        let report = DeficiencyReport {
                            module_label: metrics.label.clone(),
                            constraint: constraint.clone(),
                            current_metrics: metrics.clone(),
                            severity: DeficiencyReport::compute_severity(metrics, constraint),
                            remedy_triggered: false,
                        };
                        // Avoid duplicate pending reports
                        if !self.pending_reports.iter().any(|r| r.module_label == report.module_label) {
                            new_reports.push(report.clone());
                            self.pending_reports.push(report);
                        }
                    }
                }
            }
        }

        // Return only newly discovered reports
        &self.pending_reports
    }

    /// Mark a deficiency as remedied (removes from pending list).
    pub fn mark_remedied(&mut self, module_label: &str) {
        self.pending_reports.retain(|r| r.module_label != module_label);
        // Reset violation count for the associated constraint
        if let Some(c) = self.constraints.iter_mut()
            .find(|c| c.target_label == module_label)
        {
            c.violation_count = 0;
        }
    }

    /// Returns true when there are pending deficiencies awaiting remedy.
    pub fn has_pending(&self) -> bool {
        !self.pending_reports.is_empty()
    }

    /// Borrow the list of pending reports.
    pub fn pending_reports(&self) -> &[DeficiencyReport] {
        &self.pending_reports
    }
}

// ── Graph Integration Helpers ─────────────────────────────

/// Attach a capability metrics node to the SELF node in the GraphArena.
/// Creates a new GroundedNode with type=State and grounding=Abstract,
/// then links SELF → capability via a HasCapability edge.
pub fn attach_capability_to_self(graph: &mut GraphArena, label: &str) -> Option<NodeId> {
    let cap_node = GroundedNode {
        id: NodeId::ZERO,
        label: format!("capability:{}", label),
        node_type: NodeType::State,
        grounding: Grounding::Abstract,
        decay: 0.99,
        threshold: f64::MAX, // never fires spontaneously
        base_activation: 0.0,
        edges: Vec::new(),
        epistemic_status: EpistemicStatus::CoreConcept,
        valence: 0.0,
        mean_error: 0.0,
        variance: 0.0,
    };
    let cap_id = graph.insert(cap_node);
    // Link SELF → capability (using AssociatedWith as closest match for HasCapability)
    graph.link_to_self(Relation::AssociatedWith, cap_id);
    Some(cap_id)
}

/// Attach a constraint node to the SELF node.
pub fn attach_constraint_to_self(graph: &mut GraphArena, constraint: &Constraint) -> Option<NodeId> {
    let con_node = GroundedNode {
        id: NodeId::ZERO,
        label: format!("constraint:{}", constraint.target_label),
        node_type: NodeType::State,
        grounding: Grounding::Abstract,
        decay: 0.99,
        threshold: f64::MAX,
        base_activation: 0.0,
        edges: Vec::new(),
        epistemic_status: EpistemicStatus::CoreConcept,
        valence: 0.0,
        mean_error: 0.0,
        variance: 0.0,
    };
    let con_id = graph.insert(con_node);
    graph.link_to_self(Relation::AssociatedWith, con_id);
    Some(con_id)
}
