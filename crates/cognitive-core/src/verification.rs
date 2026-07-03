use std::sync::Arc;

use semantic_graph::prelude::*;

// ────────────────────────────────────────────────────────────
//  Verification Loop — Runtime structural integrity checks
//
//  Runs as Phase 6 after the activation tick's valence update.
//  Checks:
//    1. Energy conservation: total activation after spread must
//       equal total before spread (within floating-point epsilon).
//    2. Structural path integrity: any fired action chain must
//       pass `GraphArena::verify_path()`.
//
//  On failure:
//    - Spikes novelty (prediction error penalty)
//    - Drops valence of all involved nodes
//    - Marks faulty edges for pruning
//    - Logs the structural error for the daemon's output channel
//
//  Zero allocations in the happy path (no errors).
// ────────────────────────────────────────────────────────────

/// Result of a single verification pass.
#[derive(Debug, Clone)]
pub enum VerificationEvent {
    Pass,
    StructuralFault {
        error: StructuralError,
        penalties: PenaltySummary,
    },
}

#[derive(Debug, Clone)]
pub struct PenaltySummary {
    pub novelty_spike: f64,
    pub valence_drop: f64,
    pub edges_marked: usize,
}

/// The verification loop — called after each activation tick.
///
/// Operates on the post-tick graph snapshot and neuromodulator state.
/// Returns any structural faults detected so the daemon can log them.
pub struct VerificationLoop;

impl VerificationLoop {
    /// Run all verification checks for this tick.
    ///
    /// `total_before` and `total_after` are the sum of absolute activation
    /// values before and after the spread phases (measured by the caller).
    /// `fired_chain` is the ordered list of node IDs that fired this tick.
    ///
    /// Returns a vector of faults (empty = all checks passed).
    pub fn verify(
        ctx: &Arc<SemanticContext>,
        modulators: &mut Neuromodulator,
        total_before: f64,
        total_after: f64,
        fired_chain: &[NodeId],
    ) -> Vec<VerificationEvent> {
        let mut faults: Vec<VerificationEvent> = Vec::new();

        // ── Check 1: Energy conservation ──
        if let Some(event) = Self::check_energy_conservation(modulators, total_before, total_after) {
            faults.push(event);
        }

        if faults.is_empty() && fired_chain.len() >= 2 {
            // ── Check 2: Structural path integrity ──
            if let Some(event) = Self::check_path_integrity(ctx, modulators, fired_chain) {
                faults.push(event);
            }
        }

        faults
    }

    /// Verify that total activation energy is conserved during the spread phase.
    /// Discrepancy > 1% of the larger magnitude triggers a penalty.
    fn check_energy_conservation(
        modulators: &mut Neuromodulator,
        total_before: f64,
        total_after: f64,
    ) -> Option<VerificationEvent> {
        let max = total_before.abs().max(total_after.abs()).max(1e-10);
        let discrepancy = (total_after - total_before).abs() / max;

        if discrepancy > 0.01 {
            // 1% tolerance for floating-point drift
            let novelty_spike = (discrepancy * 0.5).clamp(0.0, 0.8);
            modulators.spike_novelty(novelty_spike);

            return Some(VerificationEvent::StructuralFault {
                error: StructuralError::EnergyNonConservation {
                    total_before,
                    total_after,
                    discrepancy,
                },
                penalties: PenaltySummary {
                    novelty_spike,
                    valence_drop: 0.2,
                    edges_marked: 0,
                },
            });
        }
        None
    }

    /// Verify that the chain of fired nodes forms a structurally valid path.
    /// If verification fails, penalize every node in the chain.
    fn check_path_integrity(
        ctx: &Arc<SemanticContext>,
        modulators: &mut Neuromodulator,
        fired_chain: &[NodeId],
    ) -> Option<VerificationEvent> {
        let graph = ctx.graph.read();

        if let Err(err) = graph.verify_path(fired_chain) {
            let novelty_spike = 0.4;
            modulators.spike_novelty(novelty_spike);

            // Drop valence on every node in the faulty chain
            for &node_id in fired_chain {
                graph.update_valence(node_id, -0.4, 0.15);
            }

            // Mark edge between the fault point for pruning
            let mut edges_marked = 0;
            let fault_pair: Option<(NodeId, NodeId)> = match &err {
                StructuralError::ContractMismatch { source, target, .. } => {
                    Some((*source, *target))
                }
                _ => None,
            };

            if let Some((src, tgt)) = fault_pair {
                if let Some(node) = graph.get(src) {
                    let mut n = node.write();
                    for edge in &mut n.edges {
                        if edge.target == tgt {
                            edge.dynamic_weight = 0.0;
                            edges_marked += 1;
                        }
                    }
                }
            }

            return Some(VerificationEvent::StructuralFault {
                error: err,
                penalties: PenaltySummary {
                    novelty_spike,
                    valence_drop: 0.4,
                    edges_marked,
                },
            });
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Node layout after with_capacity(16):
    //   Null(0), SELF(1), sensor_accel(2), concept_motion(3), action_notify(4)
    const SENSOR_ID: u64 = 2;
    const CONCEPT_ID: u64 = 3;
    const ACTION_ID: u64 = 4;

    fn test_graph() -> GraphArena {
        let mut g = GraphArena::with_capacity(16);

        g.insert(GroundedNode {
            id: NodeId::ZERO,
            label: "sensor_accel".into(),
            node_type: NodeType::Sensor,
            grounding: Grounding::Sensor {
                sensor_type: "accelerometer".into(),
                channel: 0,
                norm: SensorNorm::Clamp { min: 0.0, max: 1.0 },
            },
            decay: 0.9, threshold: 2.0, base_activation: 0.0,
            edges: vec![Edge::new(Relation::Activates, NodeId::from_raw(CONCEPT_ID))],
            valence: 0.0,
            mean_error: 0.0,
            variance: 0.0,
        });
        g.insert(GroundedNode {
            id: NodeId::ZERO,
            label: "concept_motion".into(),
            node_type: NodeType::Concept,
            grounding: Grounding::Abstract,
            decay: 0.9, threshold: 1.5, base_activation: 0.0,
            edges: vec![Edge::new(Relation::Implies, NodeId::from_raw(ACTION_ID))],
            valence: 0.0,
            mean_error: 0.0,
            variance: 0.0,
        });
        g.insert(GroundedNode {
            id: NodeId::ZERO,
            label: "action_notify".into(),
            node_type: NodeType::Action,
            grounding: Grounding::Action {
                intent_template: r#"{"action":"notify"}"#.into(),
            },
            decay: 0.5, threshold: 1.0, base_activation: 0.0,
            edges: vec![],
            valence: 0.0,
            mean_error: 0.0,
            variance: 0.0,
        });
        g
    }

    #[test]
    fn energy_conservation_passes_on_match() {
        let graph = test_graph();
        let ctx = Arc::new(SemanticContext::new(graph));
        let mut mods = Neuromodulator::new();

        let faults = VerificationLoop::verify(&ctx, &mut mods, 100.0, 99.9, &[]);
        assert!(faults.is_empty(), "Should pass with <1% discrepancy");
    }

    #[test]
    fn energy_conservation_fails_on_large_discrepancy() {
        let graph = test_graph();
        let ctx = Arc::new(SemanticContext::new(graph));
        let mut mods = Neuromodulator::new();
        let novelty_before = mods.novelty;

        let faults = VerificationLoop::verify(&ctx, &mut mods, 100.0, 50.0, &[]);
        assert!(!faults.is_empty(), "Should detect large energy discrepancy");
        assert!(mods.novelty > novelty_before, "Should spike novelty on fault");
    }

    #[test]
    fn valid_path_passes() {
        let graph = test_graph();
        let ctx = Arc::new(SemanticContext::new(graph));
        let mut mods = Neuromodulator::new();

        // Valid chain: sensor → concept → action
        let path = vec![
            NodeId::from_raw(SENSOR_ID),
            NodeId::from_raw(CONCEPT_ID),
            NodeId::from_raw(ACTION_ID),
        ];
        let faults = VerificationLoop::verify(&ctx, &mut mods, 100.0, 100.0, &path);
        let path_faults: Vec<_> = faults.iter().filter(|f| matches!(f, VerificationEvent::StructuralFault { .. })).collect();
        assert!(path_faults.is_empty(), "Valid structural path should pass");
    }

    #[test]
    fn structural_fault_penalizes_nodes() {
        let graph = test_graph();
        let ctx = Arc::new(SemanticContext::new(graph));
        let mut mods = Neuromodulator::new();

        // Initially neutral valences
        assert!((ctx.graph.read().get_valence(NodeId::from_raw(SENSOR_ID)).unwrap() - 0.0).abs() < 0.01);

        // Invalid path: action → sensor — reverse data flow (no edge exists)
        let path = vec![
            NodeId::from_raw(ACTION_ID),
            NodeId::from_raw(SENSOR_ID),
        ];
        let faults = VerificationLoop::verify(&ctx, &mut mods, 100.0, 100.0, &path);
        assert!(!faults.is_empty(), "Should detect invalid path");

        // Action node should have negative valence after penalty
        let v4 = ctx.graph.read().get_valence(NodeId::from_raw(ACTION_ID)).unwrap();
        assert!(v4 < 0.0, "Node in faulty chain should have negative valence, got {}", v4);
    }
}
