use std::sync::atomic::Ordering;
use std::sync::Arc;

use semantic_graph::prelude::*;

// ────────────────────────────────────────────────────────────
//  Spreading activation engine
//
//  This is the core "inference" mechanism. It is:
//   - Deterministic: same input → same output, always.
//   - Not probabilistic: no random samples, no temperature.
//   - Grounded: energy flows between nodes along explicit edges.
//
//  Algorithm per tick:
//
//    For each node i:
//      a) Decay:      activation[i] *= node.decay
//      b) Inject:     activation[i] += injected_energy[i] (from external events)
//      c) Diffuse:    for each edge (i → j):
//                        delta = activation[i] * edge.weight() * SPREAD_RATE
//                        activation[j] += delta
//                        activation[i] -= delta   (conservation, optional)
//      d) Threshold:  if activation[i] > node.threshold:
//                        fire_action(node)
//                        activation[i] = 0.0
//
//  Constants (deterministic, tuned for 16ms tick):
//      SPREAD_RATE = 0.15   (15% of node's energy propagates per tick)
//      BASE_INJECT = 0.4    (default energy for external event injection)
// ────────────────────────────────────────────────────────────

pub const SPREAD_RATE: f64 = 0.15;
pub const DECAY_GLOBAL: f64 = 0.97;       // global multiplier applied after node.decay
pub const BASE_INJECT: f64 = 0.4;
pub const ACTIVATION_MIN: f64 = 0.001;     // floor to prevent floating-point noise accumulation

/// Result when a node crosses its threshold.
#[derive(Debug, Clone)]
pub struct FiredAction {
    pub node_id: NodeId,
    pub node_label: String,
    pub activation_level: f64,
    pub grounding: Grounding,
}

/// The activation engine. Owns references to the graph arena and the
/// double-buffered activation array.
pub struct ActivationEngine {
    /// Shared graph reference
    ctx: Arc<SemanticContext>,
    /// Accumulated injected energy from external events since last tick.
    /// Indexed by NodeId.0.
    injection_queue: Vec<f64>,
    /// Nodes that fired this tick.
    pub fired: Vec<FiredAction>,
}

impl ActivationEngine {
    pub fn new(ctx: Arc<SemanticContext>) -> Self {
        let len = ctx.graph.read().len();
        ActivationEngine {
            ctx,
            injection_queue: vec![0.0; len.max(64)],
            fired: Vec::with_capacity(16),
        }
    }

    /// Inject activation energy into a specific node. Called by external
    /// event handlers (parser, sensor bridge, timer) — not by the spread loop.
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

    // ── Single tick of the spreading activation algorithm ────
    //
    //  Allocates zero new memory in the hot path — works on pre-allocated
    //  buffers inside the ActivationBuffer double buffer.
    //
    //  Returns: Vec<FiredAction> — list of nodes that crossed threshold
    //           this tick.

    pub fn tick(&mut self) -> &[FiredAction] {
        self.fired.clear();
        self.ctx.tick.fetch_add(1, Ordering::Relaxed);

        let graph = self.ctx.graph.read();
        let node_count = graph.len();

        // Ensure buffers are large enough
        if self.injection_queue.len() < node_count {
            self.injection_queue.resize(node_count, 0.0);
        }

        // Acquire write access to the activation double buffer.
        // We write into `back_mut()`, then flip at the end.
        let mut act_guard = self.ctx.activation.write();
        let activations = act_guard.back_mut();

        // Ensure activations buffer is sized correctly
        if activations.len() < node_count {
            // This path is cold — only hits on graph resize
            drop(act_guard);
            drop(graph);
            // Re-acquire after potential resize
            let mut act_guard2 = self.ctx.activation.write();
            let activations2 = act_guard2.back_mut();
            // Pad with zeros if needed
            for i in activations2.len()..node_count {
                if i < activations2.len() {
                    activations2[i] = 0.0;
                }
            }
            return &self.fired;
        }

        // ── Phase 1: Decay + Injection ─────────────────────
        for i in 1..node_count {
            let node = graph.get(NodeId::from_raw(i as u64));
            let node = match node {
                Some(n) => n.read(),
                None => continue,
            };

            let mut a = activations[i];

            // Apply node-specific decay
            a *= node.decay;
            // Apply global decay floor
            a *= DECAY_GLOBAL;

            // Add injected energy from external events
            a += self.injection_queue[i];

            // Floor to prevent noise accumulation
            if a < ACTIVATION_MIN {
                a = 0.0;
            }

            activations[i] = a;
        }

        // ── Phase 2: Spreading activation along edges ───────
        //
        //  For each node with activation > threshold, fire.
        //  For each edge, propagate energy to target.
        //
        //  NOTE: This is an O(E) pass. E is bounded by graph size at init time.
        //  For large graphs (>10K nodes), partition into subgraphs and process
        //  in chunks across ticks.

        for i in 1..node_count {
            let a_i = activations[i];
            if a_i.abs() < ACTIVATION_MIN {
                continue;
            }

            let node = graph.get(NodeId::from_raw(i as u64));
            let node = match node {
                Some(n) => n.read(),
                None => continue,
            };

            // ── Threshold check ──
            if a_i > node.threshold {
                self.fired.push(FiredAction {
                    node_id: NodeId::from_raw(i as u64),
                    node_label: node.label.clone(),
                    activation_level: a_i,
                    grounding: node.grounding.clone(),
                });
                activations[i] = 0.0; // consume all energy on fire
                continue; // don't spread from a just-fired node
            }

            // ── Spread to neighbors ──
            for edge in &node.edges {
                let target_idx = edge.target.0 as usize;
                if target_idx >= node_count {
                    continue;
                }
                let spread_energy = a_i * edge.effective_weight() * SPREAD_RATE;
                activations[target_idx] += spread_energy;
                // Conservation: subtract what we gave
                activations[i] -= spread_energy;
            }
        }

        // ── Zero out injection queue for next tick ──
        for i in 1..node_count {
            self.injection_queue[i] = 0.0;
        }

        // ── Flip the double buffer — readers now see this tick's state ──
        act_guard.flip();

        &self.fired
    }

    /// Get the current activation levels (for monitoring / debug UI).
    pub fn read_activations(&self) -> Vec<(NodeId, f64)> {
        let act = self.ctx.activation.read();
        let snapshot = act.read();
        snapshot
            .iter()
            .enumerate()
            .skip(1)
            .map(|(i, &v)| (NodeId::from_raw(i as u64), v))
            .collect()
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
            edges: vec![
                Edge {
                    relation: Relation::Activates,
                    target: NodeId::from_raw(2),
                    weight_override: None,
                },
            ],
        });

        g.insert(GroundedNode {
            id: NodeId::ZERO,
            label: "concept_movement".into(),
            node_type: NodeType::Concept,
            grounding: Grounding::Abstract,
            decay: 0.9,
            threshold: 1.5,
            base_activation: 0.0,
            edges: vec![
                Edge {
                    relation: Relation::Implies,
                    target: NodeId::from_raw(3),
                    weight_override: None,
                },
            ],
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
        });

        g
    }

    #[test]
    fn activation_propagates_and_fires() {
        let graph = test_graph();
        let ctx = SemanticContext::new(graph);
        let mut engine = ActivationEngine::new(ctx.clone());

        // Inject energy into sensor_motion node (id=1)
        engine.inject(NodeId::from_raw(1), 2.5);

        // Run ticks until something fires or we hit limit
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
            let max = activations
                .iter()
                .map(|(_, v)| *v)
                .fold(0.0_f64, f64::max);
            max_seen = max_seen.max(max);

            if max < ACTIVATION_MIN {
                return; // decayed to zero — success
            }
        }

        panic!(
            "Energy did not decay to zero within 30 ticks (last max: {:.6})",
            max_seen
        );
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
}
