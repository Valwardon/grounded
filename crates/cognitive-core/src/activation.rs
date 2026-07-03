use std::sync::Arc;

use semantic_graph::prelude::*;
use crate::{VerificationLoop, VerificationEvent};

// ────────────────────────────────────────────────────────────
//  Spreading activation engine — STDP + neuromodulation edition
//
//  This is the core "inference" mechanism. Every tick (~16ms):
//
//    Phase 1 — Neuromodulator decay:
//      novelty, arousal, reward leak toward baseline.
//      Compute global threshold_mod and plasticity_mod.
//
//    Phase 2 — Decay + Injection + Prediction Error:
//      For each node: a *= node.decay * DECAY_GLOBAL; a += injection.
//      Compare a against prediction from last tick → prediction error → novelty spike.
//
//    Phase 3 — Spreading activation + eligibility:
//      For each non-fired node, propagate energy along edges.
//      Track eligibility: boosted when source fires, decays each tick.
//      Compute next-tick prediction from resulting activations.
//
//    Phase 4 — STDP + pruning:
//      For each edge: drift toward default, LTP on co-firing, prune if too weak.
//
//  Deterministic: same graph + same event sequence → same activations.
//  Zero allocations in hot path: all buffers pre-sized at init.
// ────────────────────────────────────────────────────────────

pub const SPREAD_RATE: f64 = 0.15;
pub const DECAY_GLOBAL: f64 = 0.97;
pub const BASE_INJECT: f64 = 0.4;
pub const ACTIVATION_MIN: f64 = 0.001;

/// Result when a node crosses its threshold.
#[derive(Debug, Clone)]
pub struct FiredAction {
    pub node_id: NodeId,
    pub node_label: String,
    pub activation_level: f64,
    pub grounding: Grounding,
}

/// The activation engine.
pub struct ActivationEngine {
    ctx: Arc<SemanticContext>,
    injection_queue: Vec<f64>,
    pub fired: Vec<FiredAction>,

    // ── Neuromodulation ──
    pub modulators: Neuromodulator,

    // ── STDP history ──
    pub firing_history: FiringHistory,

    // ── Predictive coding ──
    /// Predicted activation for each node (computed during Phase 3 of previous tick).
    predictions: Vec<f64>,

    /// Prediction errors detected this tick (consumed by daemon after tick).
    pub prediction_errors: Vec<PredictionError>,

    // ── Structural verification ──
    /// Faults detected by the verification loop during the last tick.
    pub structural_faults: Vec<VerificationEvent>,

    // ── Staging arrays (pre-allocated, zero alloc in hot path) ──
    /// Which nodes fired this tick (bitset, parallel to firing_history words)
    fired_this_tick: Vec<u64>,
}

impl ActivationEngine {
    pub fn new(ctx: Arc<SemanticContext>) -> Self {
        let len = ctx.graph.read().len().max(64);
        let words = (len + 63) / 64;
        ActivationEngine {
            injection_queue: vec![0.0; len],
            fired: Vec::with_capacity(16),
            modulators: Neuromodulator::new(),
            firing_history: FiringHistory::new(len),
            predictions: vec![0.0; len],
            prediction_errors: Vec::with_capacity(4),
            structural_faults: Vec::with_capacity(2),
            fired_this_tick: vec![0; words.max(1)],
            ctx,
        }
    }

    /// Inject activation energy into a specific node.
    pub fn inject(&mut self, target: NodeId, energy: f64) {
        let idx = target.0 as usize;
        if idx < self.injection_queue.len() {
            self.injection_queue[idx] += energy;
        }
    }

    /// Inject energy into all nodes in a CD frame's role slots.
    pub fn inject_frame(&mut self, frame: &ConceptualFrame, base_energy: f64) {
        for (node_id, energy) in frame.injection_targets(base_energy) {
            self.inject(node_id, energy);
        }
    }

    /// Spike a neuromodulator channel (called from daemon).
    pub fn spike_novelty(&mut self, amount: f64) { self.modulators.spike_novelty(amount); }
    pub fn spike_arousal(&mut self, amount: f64) { self.modulators.spike_arousal(amount); }
    pub fn spike_reward(&mut self, amount: f64) { self.modulators.spike_reward(amount); }

    // ── Full 4-phase tick ──────────────────────────────────

    pub fn tick(&mut self) -> &[FiredAction] {
        self.fired.clear();
        self.prediction_errors.clear();
        self.structural_faults.clear();
        self.ctx.tick.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let graph = self.ctx.graph.read();
        let node_count = graph.len();

        // Ensure buffers are large enough (cold path if resized)
        if self.injection_queue.len() < node_count {
            self.injection_queue.resize(node_count, 0.0);
            self.predictions.resize(node_count, 0.0);
            self.firing_history.resize(node_count);
            let new_words = (node_count + 63) / 64;
            self.fired_this_tick.resize(new_words.max(1), 0);
        }

        let mut act_guard = self.ctx.activation.write();
        let activations = act_guard.back_mut();

        if activations.len() < node_count {
            drop(act_guard);
            drop(graph);
            let mut act_guard2 = self.ctx.activation.write();
            let activations2 = act_guard2.back_mut();
            for i in activations2.len()..node_count {
                if i < activations2.len() {
                    activations2[i] = 0.0;
                }
            }
            return &self.fired;
        }

        // ──────────────────────────────────────────────────
        //  Phase 1 — Neuromodulator decay
        // ──────────────────────────────────────────────────
        self.modulators.tick_decay();
        let threshold_mod = self.modulators.threshold_modifier();
        let plasticity_mod = self.modulators.plasticity_modifier();

        // ──────────────────────────────────────────────────
        //  Phase 2 — Decay + Injection + Prediction Error
        // ──────────────────────────────────────────────────
        for i in 1..node_count {
            let node = match graph.get(NodeId::from_raw(i as u64)) {
                Some(n) => n.read(),
                None => continue,
            };

            let mut a = activations[i];

            // Node-specific decay followed by global decay
            a *= node.decay * DECAY_GLOBAL;

            // Add injected energy from external events
            a += self.injection_queue[i];

            // Floor to prevent noise accumulation
            if a < ACTIVATION_MIN {
                a = 0.0;
            }

            // ── Prediction error check ──
            let predicted = self.predictions[i];
            if predicted > ACTIVATION_MIN {
                let error = (a - predicted).abs() / predicted.max(ACTIVATION_MIN);
                if error > PREDICTION_ERROR_THRESHOLD {
                    self.prediction_errors.push(PredictionError {
                        node_id: NodeId::from_raw(i as u64),
                        expected: predicted,
                        actual: a,
                        error_magnitude: error,
                    });
                    // Spike novelty proportional to prediction error
                    self.modulators.spike_novelty(error * 0.15);
                }
            }

            activations[i] = a;
        }

        // ── Measure total energy before spread (for conservation check) ──
        let mut energy_before_spread: f64 = 0.0;
        for i in 1..node_count {
            energy_before_spread += activations[i].abs();
        }

        // ──────────────────────────────────────────────────
        //  Phase 3 — Spreading activation + eligibility
        // ──────────────────────────────────────────────────
        //
        //  Clear fired_this_tick bitset for the current tick
        self.clear_fired_bitset();

        for i in 1..node_count {
            let a_i = activations[i];
            if a_i.abs() < ACTIVATION_MIN {
                continue;
            }

            let node = match graph.get(NodeId::from_raw(i as u64)) {
                Some(n) => n.read(),
                None => continue,
            };

            // ── Threshold check (modulated by neuromodulators) ──
            let effective_threshold = node.threshold * threshold_mod;
            if a_i > effective_threshold {
                // Node fires this tick
                self.fired.push(FiredAction {
                    node_id: NodeId::from_raw(i as u64),
                    node_label: node.label.clone(),
                    activation_level: a_i,
                    grounding: node.grounding.clone(),
                });

                // Record in bitset
                self.set_fired_bit(i);

                // Record in ring buffer
                self.firing_history.record_fired(NodeId::from_raw(i as u64));

                // Consume all energy on fire (no spread from fired node)
                activations[i] = 0.0;
                continue;
            }

            // ── Spread to neighbors ──
            for edge in &node.edges {
                let target_idx = edge.target.0 as usize;
                if target_idx >= node_count {
                    continue;
                }
                let spread_energy = a_i * edge.effective_weight() * SPREAD_RATE;
                activations[target_idx] += spread_energy;
                activations[i] -= spread_energy; // conservation
            }
        }

        // ── Measure total energy after spread (for conservation check) ──
        let mut energy_after_spread: f64 = 0.0;
        for i in 1..node_count {
            energy_after_spread += activations[i].abs();
        }

        // ── Compute next-tick predictions from current activation levels ──
        // Prediction = what activation we expect next tick (after decay + injection)
        for i in 1..node_count {
            self.predictions[i] = activations[i];
        }

        // ──────────────────────────────────────────────────
        //  Phase 4 — STDP + pruning
        // ──────────────────────────────────────────────────
        //
        //  Iterate all edges (O(E)). For each:
        //    1. Eligibility decay
        //    2. Eligibility boost if source fired this tick
        //    3. LTP if target fired this tick → consume eligibility
        //    4. LTD drift toward default weight
        //    5. Mark for pruning if |dynamic_weight| < PRUNE_THRESHOLD

        let mut prune_targets: Vec<(usize, usize)> = Vec::with_capacity(16);

        for i in 1..node_count {
            let node = match graph.get(NodeId::from_raw(i as u64)) {
                Some(n) => n.write(),
                None => continue,
            };

            for (edge_idx, edge) in node.edges.iter_mut().enumerate() {
                // Eligibility decay
                edge.eligibility *= ELIGIBILITY_DECAY;

                // Boost if source (i) fired this tick
                if self.is_fired_this_tick(i) {
                    edge.eligibility += 1.0;
                }

                // LTP: if target fired this tick, consume eligibility
                let target_idx = edge.target.0 as usize;
                if target_idx < node_count && self.is_fired_this_tick(target_idx) {
                    let delta = edge.eligibility * LTP_RATE * plasticity_mod;
                    edge.dynamic_weight = (edge.dynamic_weight + delta).clamp(-1.0, 1.0);
                    edge.eligibility *= 0.5; // consumed
                }

                // LTD drift toward default weight
                let default_w = edge.weight_override.unwrap_or_else(|| edge.relation.spread_weight());
                edge.dynamic_weight += (default_w - edge.dynamic_weight) * DRIFT_RATE;
                edge.dynamic_weight = edge.dynamic_weight.clamp(-1.0, 1.0);

                // Mark for pruning if weight fell below threshold
                if edge.dynamic_weight.abs() < PRUNE_THRESHOLD {
                    prune_targets.push((i, edge_idx));
                    edge.dynamic_weight = 0.0;
                }
            }
        }

        // Prune edges with |weight| < threshold (set weight to 0, will be GC'd later)
        // We mark them by zeroing dynamic_weight so they become effectively disconnected
        for (node_idx, edge_idx) in &prune_targets {
            if let Some(node_lock) = graph.get(NodeId::from_raw(*node_idx as u64)) {
                let mut node = node_lock.write();
                if *edge_idx < node.edges.len() {
                    node.edges[*edge_idx].dynamic_weight = 0.0;
                }
            }
        }

        // ──────────────────────────────────────────────────
        //  Phase 5 — Valence update (preference formation)
        // ──────────────────────────────────────────────────
        //
        //  Nodes that fire without prediction error → positive valence (familiar = good).
        //  Nodes involved in prediction errors → negative valence (surprise = aversive).
        //  SELF drifts slowly upward (baseline contentment).
        //  Over time, this creates genuine preferences — the system "likes" what it
        //  can predict and "dislikes" what surprises it.

        let reward_level = self.modulators.reward;

        // Build set of node IDs with prediction errors this tick
        let mut error_nodes: Vec<usize> = self.prediction_errors.iter()
            .map(|e| e.node_id.0 as usize)
            .collect();
        error_nodes.sort();
        error_nodes.dedup();

        for action in &self.fired {
            let idx = action.node_id.0 as usize;
            let in_error = error_nodes.binary_search(&idx).is_ok();
            if in_error {
                // Negative: surprise is aversive
                graph.update_valence(action.node_id, -0.3, 0.05);
            } else {
                // Positive: familiar patterns feel good
                let pos_target = 0.2 + reward_level * 0.3;
                graph.update_valence(action.node_id, pos_target, 0.02);
            }
        }

        // Nodes that received injection but didn't fire: mild positive (being noticed)
        for i in 1..node_count {
            if self.injection_queue[i] > 0.0 {
                let id = NodeId::from_raw(i as u64);
                let already_fired = self.fired.iter().any(|f| f.node_id == id);
                if !already_fired {
                    graph.update_valence(id, 0.1, 0.01);
                }
            }
        }

        // SELF baseline: slow drift toward positive
        graph.update_valence(NodeId::SELF, 0.5, 0.001);

        // ── Phase 6 — Structural verification ──
        let fired_chain: Vec<NodeId> = self.fired.iter().map(|f| f.node_id).collect();
        self.structural_faults = VerificationLoop::verify(
            &self.ctx,
            &mut self.modulators,
            energy_before_spread,
            energy_after_spread,
            &fired_chain,
        );

        // Penalize valence on nodes involved in structural faults
        for fault in &self.structural_faults {
            if let VerificationEvent::StructuralFault { ref error, ref penalties } = fault {
                for action in &self.fired {
                    graph.update_valence(action.node_id, -penalties.valence_drop, 0.1);
                }
                match error {
                    StructuralError::ContractMismatch { source, target, .. } => {
                        // Mark the offending edge for pruning
                        if let Some(node) = graph.get(*source) {
                            let mut n = node.write();
                            for edge in &mut n.edges {
                                if edge.target == *target {
                                    edge.dynamic_weight = 0.0;
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        // ── Advance firing history ring buffer ──
        self.firing_history.advance_tick();

        // ── Zero out injection queue for next tick ──
        for i in 1..node_count {
            self.injection_queue[i] = 0.0;
        }

        // ── Flip double buffer ──
        act_guard.flip();

        &self.fired
    }

    // ── Bitset helpers ──────────────────────────────────

    #[inline]
    fn set_fired_bit(&mut self, node_idx: usize) {
        let word = node_idx / 64;
        let bit = node_idx % 64;
        if word < self.fired_this_tick.len() {
            self.fired_this_tick[word] |= 1u64 << bit;
        }
    }

    #[inline]
    fn is_fired_this_tick(&self, node_idx: usize) -> bool {
        let word = node_idx / 64;
        let bit = node_idx % 64;
        word < self.fired_this_tick.len() && (self.fired_this_tick[word] & (1u64 << bit)) != 0
    }

    #[inline]
    fn clear_fired_bitset(&mut self) {
        for w in &mut self.fired_this_tick {
            *w = 0;
        }
    }

    /// Get the current activation levels (for monitoring / debug UI).
    pub fn read_activations(&self) -> Vec<(NodeId, f64)> {
        let act = self.ctx.activation.read();
        let snapshot = act.read();
        snapshot.iter().enumerate().skip(1).map(|(i, &v)| (NodeId::from_raw(i as u64), v)).collect()
    }

    /// Current neuromodulator levels (for debug/bridge).
    pub fn read_modulators(&self) -> (f64, f64, f64) {
        (self.modulators.novelty, self.modulators.arousal, self.modulators.reward)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_graph() -> GraphArena {
        let mut g = GraphArena::with_capacity(16);

        g.insert(GroundedNode {
            id: NodeId::ZERO,
            label: "sensor_motion".into(),
            node_type: NodeType::Sensor,
            grounding: Grounding::Sensor {
                sensor_type: "accelerometer".into(),
                channel: 0,
                norm: SensorNorm::Linear { scale: 0.1, offset: 0.0 },
            },
            decay: 0.8,
            threshold: 2.0,
            base_activation: 0.0,
            edges: vec![Edge::new(Relation::Activates, NodeId::from_raw(2))],
            valence: 0.0,
        });

        g.insert(GroundedNode {
            id: NodeId::ZERO,
            label: "concept_movement".into(),
            node_type: NodeType::Concept,
            grounding: Grounding::Abstract,
            decay: 0.9,
            threshold: 1.5,
            base_activation: 0.0,
            edges: vec![Edge::new(Relation::Implies, NodeId::from_raw(3))],
            valence: 0.0,
        });

        g.insert(GroundedNode {
            id: NodeId::ZERO,
            label: "action_notify".into(),
            node_type: NodeType::Action,
            grounding: Grounding::Action {
                intent_template: r#"{"action":"notify_movement","params":{}}"#.into(),
            },
            decay: 0.5,
            threshold: 1.0,
            base_activation: 0.0,
            edges: Vec::new(),
            valence: 0.0,
        });

        g
    }

    #[test]
    fn activation_propagates_and_fires() {
        let graph = test_graph();
        let ctx = SemanticContext::new(graph);
        let mut engine = ActivationEngine::new(ctx.clone());

        engine.inject(NodeId::from_raw(1), 2.5);

        for _ in 0..10 {
            let fired = engine.tick();
            if !fired.is_empty() {
                assert_eq!(fired[0].node_label, "action_notify");
                return;
            }
        }
        panic!("No action fired within 10 ticks");
    }

    #[test]
    fn energy_decays_to_zero() {
        let graph = test_graph();
        let ctx = SemanticContext::new(graph);
        let mut engine = ActivationEngine::new(ctx.clone());

        engine.inject(NodeId::from_raw(1), 0.3);

        let mut max_seen = 0.0;
        for _ in 0..30 {
            let _fired = engine.tick();
            let activations = engine.read_activations();
            let max = activations.iter().map(|(_, v)| *v).fold(0.0_f64, f64::max);
            max_seen = max_seen.max(max);
            if max < ACTIVATION_MIN {
                return;
            }
        }
        panic!("Energy did not decay to zero within 30 ticks (last max: {:.6})", max_seen);
    }

    #[test]
    fn activation_reads_after_tick() {
        let graph = test_graph();
        let ctx = SemanticContext::new(graph);
        let mut engine = ActivationEngine::new(ctx.clone());
        engine.inject(NodeId::from_raw(1), 0.5);
        let _fired = engine.tick();
        let activations = engine.read_activations();
        assert!(!activations.is_empty(), "should have activation readings");
        assert_eq!(activations[0].0, NodeId::from_raw(1));
    }

    #[test]
    fn stdp_strengthens_cofiring_edge() {
        let mut graph = GraphArena::with_capacity(16);
        // Node 2 and 3 where 2→3 has an edge
        let mut n2 = GroundedNode {
            id: NodeId::ZERO,
            label: "trigger".into(),
            node_type: NodeType::Concept,
            grounding: Grounding::Abstract,
            decay: 0.9,
            threshold: 0.5,
            base_activation: 0.0,
            edges: vec![Edge::new(Relation::Activates, NodeId::from_raw(3))],
            valence: 0.0,
        };
        let n3 = GroundedNode {
            id: NodeId::ZERO,
            label: "target".into(),
            node_type: NodeType::Concept,
            grounding: Grounding::Abstract,
            decay: 0.9,
            threshold: 1.5,
            base_activation: 0.0,
            edges: Vec::new(),
            valence: 0.0,
        };
        graph.insert(n3);

        let ctx = SemanticContext::new(graph);
        let mut engine = ActivationEngine::new(ctx.clone());

        let initial_weight = engine.ctx.graph.read()
            .get(NodeId::from_raw(2)).unwrap().read()
            .edges[0].dynamic_weight;

        // Inject so both fire: trigger gets 2.0, target gets 2.0
        engine.inject(NodeId::from_raw(2), 2.0);
        engine.inject(NodeId::from_raw(3), 2.0);

        for _ in 0..3 {
            engine.tick();
        }

        let post_weight = engine.ctx.graph.read()
            .get(NodeId::from_raw(2)).unwrap().read()
            .edges[0].dynamic_weight;

        // Edge should be reinforced from co-firing
        assert!(post_weight > initial_weight,
            "STDP should strengthen co-firing edge: {:.6} -> {:.6}", initial_weight, post_weight);
    }

    #[test]
    fn prediction_error_spikes_novelty() {
        let graph = test_graph();
        let ctx = SemanticContext::new(graph);
        let mut engine = ActivationEngine::new(ctx.clone());

        // First tick establishes a prediction baseline
        engine.inject(NodeId::from_raw(1), 2.5);
        engine.tick();

        let novelty_before = engine.modulators.novelty;

        // Second tick with zero injection → prediction will be wrong (expects energy, gets none)
        engine.inject(NodeId::from_raw(1), 0.0);
        engine.tick();

        let novelty_after = engine.modulators.novelty;
        assert!(novelty_after > novelty_before,
            "Prediction error should increase novelty: {:.6} -> {:.6}", novelty_before, novelty_after);
    }

    #[test]
    fn neuromodulator_threshold_modulation() {
        let mut graph = GraphArena::with_capacity(16);
        let node = GroundedNode {
            id: NodeId::ZERO,
            label: "hard_to_fire".into(),
            node_type: NodeType::Concept,
            grounding: Grounding::Abstract,
            decay: 0.9,
            threshold: 5.0,  // high threshold
            base_activation: 0.0,
            edges: Vec::new(),
            valence: 0.0,
        };
        graph.insert(node);
        let ctx = SemanticContext::new(graph);
        let mut engine = ActivationEngine::new(ctx.clone());

        // Inject below threshold (4.0 < 5.0), should not fire without neuromodulation
        engine.inject(NodeId::from_raw(2), 4.0);
        let fired = engine.tick();
        assert!(fired.is_empty(), "Should not fire without modulation");

        // Now spike novelty to lower effective threshold
        engine.spike_novelty(1.0);
        engine.inject(NodeId::from_raw(2), 4.0);
        let fired2 = engine.tick();

        // With novelty=1.0, threshold_mod = 1.0 - 0.35 = 0.65
        // effective_threshold = 5.0 * 0.65 = 3.25
        // 4.0 > 3.25 → should fire
        assert!(!fired2.is_empty(), "Should fire with high novelty modulation");
    }
}
