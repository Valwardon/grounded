use std::sync::Arc;
use semantic_graph::prelude::*;
use crate::goal::GoalResolver;

/// Size of the prediction error ring buffer per tracked node.
/// Only the most recent N errors are kept for trend detection.
const ERROR_HISTORY_SIZE: usize = 16;

/// Minimum number of consecutive high-error ticks before a goal is formed.
const ERROR_PERSISTENCE_THRESHOLD: u8 = 4;

/// Priority weight for prediction error magnitude (Eₚ).
const ALPHA: f64 = 0.35;
/// Priority weight for novelty level at detection (N).
const BETA: f64 = 0.25;
/// Priority weight for drive deprivation (D_dep).
const GAMMA: f64 = 0.25;
/// Priority weight for systemic inefficiency (S_ineff) — inverse of conservation drive.
const DELTA: f64 = 0.15;

/// A single node's prediction error history (fixed-size ring buffer).
#[derive(Debug, Clone)]
pub struct ErrorHistory {
    /// Ring buffer of recent error magnitudes.
    pub magnitudes: [f64; ERROR_HISTORY_SIZE],
    /// Write index into the ring buffer.
    pub write_idx: usize,
    /// How many samples have been recorded so far (saturates at ERROR_HISTORY_SIZE).
    pub count: usize,
    /// How many consecutive ticks this node has been above threshold.
    pub consecutive_high: u8,
}

impl ErrorHistory {
    pub fn new() -> Self {
        ErrorHistory {
            magnitudes: [0.0; ERROR_HISTORY_SIZE],
            write_idx: 0,
            count: 0,
            consecutive_high: 0,
        }
    }

    /// Record a new error magnitude and return the EMA of recent errors.
    pub fn record(&mut self, magnitude: f64) -> f64 {
        self.magnitudes[self.write_idx] = magnitude;
        self.write_idx = (self.write_idx + 1) % ERROR_HISTORY_SIZE;
        if self.count < ERROR_HISTORY_SIZE {
            self.count += 1;
        }

        // Track consecutive high-error ticks
        if magnitude > PREDICTION_ERROR_THRESHOLD {
            self.consecutive_high = (self.consecutive_high + 1).min(255);
        } else {
            self.consecutive_high = 0;
        }

        self.ema()
    }

    /// Exponential moving average of recent errors.
    fn ema(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        let mut sum = 0.0;
        let n = self.count.min(ERROR_HISTORY_SIZE);
        for i in 0..n {
            let idx = (self.write_idx + ERROR_HISTORY_SIZE - 1 - i) % ERROR_HISTORY_SIZE;
            // Weight recent errors more heavily: decay = 0.7 ^ i
            let weight = 0.7_f64.powi(i as i32);
            sum += self.magnitudes[idx] * weight;
        }
        sum / n as f64
    }

    /// True if this node shows persistent high prediction error.
    pub fn has_persistent_error(&self) -> bool {
        self.consecutive_high >= ERROR_PERSISTENCE_THRESHOLD
    }

    /// True if the trend is rising (last 3 errors increasing).
    pub fn rising_trend(&self) -> bool {
        if self.count < 4 {
            return false;
        }
        let i0 = (self.write_idx + ERROR_HISTORY_SIZE - 1) % ERROR_HISTORY_SIZE;
        let i1 = (self.write_idx + ERROR_HISTORY_SIZE - 2) % ERROR_HISTORY_SIZE;
        let i2 = (self.write_idx + ERROR_HISTORY_SIZE - 3) % ERROR_HISTORY_SIZE;
        self.magnitudes[i2] < self.magnitudes[i1] && self.magnitudes[i1] < self.magnitudes[i0]
    }
}

/// Tracks systemic efficiency — ratio of useful work to wasted energy.
#[derive(Debug, Clone)]
pub struct SystemicEfficiency {
    /// EMA of energy wasted on prediction errors (high = inefficient).
    pub energy_wasted_ema: f64,
    /// EMA of total activation energy in the system.
    pub total_energy_ema: f64,
}

impl SystemicEfficiency {
    pub fn new() -> Self {
        SystemicEfficiency {
            energy_wasted_ema: 0.0,
            total_energy_ema: 1.0,
        }
    }

    /// Update efficiency metrics from current tick data.
    pub fn update(&mut self, total_prediction_error: f64, total_activation_energy: f64) {
        let decay = 0.95;
        self.energy_wasted_ema = self.energy_wasted_ema * decay + total_prediction_error * (1.0 - decay);
        self.total_energy_ema = self.total_energy_ema * decay + total_activation_energy.max(0.01) * (1.0 - decay);
    }

    /// Systemic inefficiency (0.0 = perfectly efficient, 1.0 = all energy wasted).
    pub fn inefficiency(&self) -> f64 {
        (self.energy_wasted_ema / self.total_energy_ema).clamp(0.0, 1.0)
    }
}

/// Result of a goal formation cycle — what the engine decided to do.
#[derive(Debug, Clone)]
pub enum GoalFormationResult {
    /// A new goal was formed and registered in the graph.
    NewGoal {
        node_id: NodeId,
        label: String,
        priority: f64,
        reason: GoalReason,
    },
    /// No goal needed this tick.
    None,
}

/// Why a goal was formed — for logging and introspection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoalReason {
    PersistentPredictionError,
    RisingErrorTrend,
    DriveDeprivation,
    SystemicInefficiency,
    CuriosityDeficit,
}

impl GoalReason {
    pub fn description(&self) -> &'static str {
        match self {
            GoalReason::PersistentPredictionError => "Persistent prediction error on node",
            GoalReason::RisingErrorTrend => "Rising prediction error trend on node",
            GoalReason::DriveDeprivation => "Drive deprivation detected",
            GoalReason::SystemicInefficiency => "Systemic energy inefficiency",
            GoalReason::CuriosityDeficit => "Curiosity budget depleted",
        }
    }
}

/// Autonomous goal formation engine.
///
/// Monitors prediction errors from the cognitive daemon, tracks which nodes
/// have persistent high prediction error, and autonomously creates Goal nodes
/// in the semantic graph to "understand_<node>" or "fix_<system>".
///
/// Pure deterministic — no ML, no randomness.
pub struct GoalFormationEngine {
    /// Per-node prediction error history (indexed by node_id.0).
    error_histories: Vec<ErrorHistory>,
    /// Systemic efficiency tracker.
    efficiency: SystemicEfficiency,
    /// Cooldown: don't form the same goal type twice within N ticks.
    cooldown_ticks: u64,
    /// Last tick when a goal was formed (for cooldown).
    last_goal_tick: u64,
    /// Reference to the ctx for graph operations.
    ctx: Arc<SemanticContext>,
    /// Reference to the goal resolver for goal registration.
    goal_resolver: parking_lot::Mutex<GoalResolver>,
    /// Which nodes we've already formed "understand_<X>" goals for.
    goal_formations: Vec<u64>,
}

impl GoalFormationEngine {
    pub fn new(ctx: Arc<SemanticContext>) -> Self {
        GoalFormationEngine {
            error_histories: Vec::with_capacity(64),
            efficiency: SystemicEfficiency::new(),
            cooldown_ticks: 500,
            last_goal_tick: 0,
            goal_resolver: parking_lot::Mutex::new(GoalResolver::new(ctx.clone())),
            ctx,
            goal_formations: Vec::with_capacity(16),
        }
    }

    /// Access the goal resolver for inspecting active goals.
    pub fn goals(&self) -> &parking_lot::Mutex<GoalResolver> {
        &self.goal_resolver
    }

    /// Access systemic efficiency for monitoring.
    pub fn efficiency(&self) -> &SystemicEfficiency {
        &self.efficiency
    }

    /// Run one tick of the goal formation engine.
    ///
    /// Called after Phase 4 (prediction errors are available) in the cognitive
    /// daemon loop. Returns a GoalFormationResult if a new goal was formed.
    ///
    /// Steps:
    ///   1. Update prediction error histories for each node with errors
    ///   2. Update systemic efficiency metrics
    ///   3. Check each node for persistent high error → form "understand" goal
    ///   4. Check drive deprivation → form "satisfy" goal
    ///   5. Check systemic inefficiency → form "optimize" goal
    pub fn tick(
        &mut self,
        prediction_errors: &[PredictionError],
        activations: &[f64],
        novelty: f64,
        _arousal: f64,
        _reward: f64,
        // Drive intensities: [curiosity, safety, mastery, affiliation, exploration, conservation]
        drive_intensities: &[f64; 6],
        current_tick: u64,
    ) -> GoalFormationResult {
        let node_count = self.ctx.graph.read().len();
        if self.error_histories.len() < node_count {
            self.error_histories.resize(node_count, ErrorHistory::new());
        }

        // 1. Update prediction error histories
        let mut total_error_energy = 0.0;
        let mut total_activation = 0.0;

        for err in prediction_errors {
            let idx = err.node_id.0 as usize;
            if idx < self.error_histories.len() {
                let ema = self.error_histories[idx].record(err.error_magnitude);
                total_error_energy += ema;
            }
        }

        for &a in activations.iter().skip(1) {
            total_activation += a.abs();
        }

        // 2. Update systemic efficiency
        self.efficiency.update(total_error_energy, total_activation);

        // Skip if on cooldown
        if current_tick < self.last_goal_tick + self.cooldown_ticks {
            return GoalFormationResult::None;
        }

        // 3. Check for persistent prediction errors → understand goals
        for err in prediction_errors {
            let idx = err.node_id.0 as usize;
            if idx >= self.error_histories.len() {
                continue;
            }
            let history = &self.error_histories[idx];
            if !history.has_persistent_error() {
                continue;
            }

            let node_id = err.node_id;
            // Already formed a goal for this node?
            if self.goal_formations.contains(&node_id.0) {
                continue;
            }

            let label = self.ctx.graph.read()
                .label_of(node_id)
                .unwrap_or_else(|| format!("node_{}", node_id.0));
            let goal_label = format!("understand_{}", label);
            let ema = history.ema();

            // Priority: α·Eₚ + β·N + γ·D_dep(exploration) + δ·S_ineff
            let exploration_dep = 1.0 - drive_intensities[4];
            let priority = ALPHA * ema
                + BETA * novelty
                + GAMMA * exploration_dep
                + DELTA * self.efficiency.inefficiency();

            let priority = priority.clamp(0.05, 1.0);

            // Register goal
            let goal_id = self.goal_resolver.lock().register_goal(
                &goal_label, priority, current_tick + 10000,
            );

            self.goal_formations.push(node_id.0);
            self.last_goal_tick = current_tick;

            return GoalFormationResult::NewGoal {
                node_id: goal_id,
                label: goal_label,
                priority,
                reason: GoalReason::PersistentPredictionError,
            };
        }

        // 4. Check rising error trend on any node
        for err in prediction_errors {
            let idx = err.node_id.0 as usize;
            if idx >= self.error_histories.len() {
                continue;
            }
            if !self.error_histories[idx].rising_trend() {
                continue;
            }

            let node_id = err.node_id;
            if self.goal_formations.contains(&node_id.0) {
                continue;
            }

            let label = self.ctx.graph.read()
                .label_of(node_id)
                .unwrap_or_else(|| format!("node_{}", node_id.0));
            let goal_label = format!("investigate_{}", label);
            let ema = self.error_histories[idx].ema();
            let exploration_dep = 1.0 - drive_intensities[4];
            let priority = (ALPHA * ema + BETA * novelty + GAMMA * exploration_dep).clamp(0.05, 1.0);

            let goal_id = self.goal_resolver.lock().register_goal(
                &goal_label, priority, current_tick + 5000,
            );
            self.goal_formations.push(node_id.0);
            self.last_goal_tick = current_tick;

            return GoalFormationResult::NewGoal {
                node_id: goal_id,
                label: goal_label,
                priority,
                reason: GoalReason::RisingErrorTrend,
            };
        }

        // 5. Check drive deprivation (curiosity or exploration chronically low)
        let curiosity_dep = 1.0 - drive_intensities[0];
        let exploration_dep = 1.0 - drive_intensities[4];
        if curiosity_dep > 0.7 || exploration_dep > 0.7 {
            let goal_label = format!("satisfy_curiosity_drive");
            let priority = (GAMMA * curiosity_dep.max(exploration_dep)).clamp(0.05, 1.0);
            let goal_id = self.goal_resolver.lock().register_goal(
                &goal_label, priority, current_tick + 3000,
            );
            self.last_goal_tick = current_tick;

            return GoalFormationResult::NewGoal {
                node_id: goal_id,
                label: goal_label,
                priority,
                reason: GoalReason::DriveDeprivation,
            };
        }

        // 6. Check systemic inefficiency
        if self.efficiency.inefficiency() > 0.6 {
            let goal_label = "optimize_energy_efficiency".to_string();
            let priority = (DELTA * self.efficiency.inefficiency()).clamp(0.05, 1.0);
            let goal_id = self.goal_resolver.lock().register_goal(
                &goal_label, priority, current_tick + 5000,
            );
            self.last_goal_tick = current_tick;

            return GoalFormationResult::NewGoal {
                node_id: goal_id,
                label: goal_label,
                priority,
                reason: GoalReason::SystemicInefficiency,
            };
        }

        GoalFormationResult::None
    }

    /// Clear goal formation memory for a specific node (when its goal is resolved).
    pub fn clear_formation(&mut self, node_id: NodeId) {
        self.goal_formations.retain(|&id| id != node_id.0);
    }

    /// Reset all goal formation memory (e.g., on system reset).
    pub fn reset(&mut self) {
        self.error_histories.clear();
        self.efficiency = SystemicEfficiency::new();
        self.goal_formations.clear();
        self.last_goal_tick = 0;
    }
}
