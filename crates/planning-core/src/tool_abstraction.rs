use std::sync::Arc;
use semantic_graph::prelude::*;

/// ────────────────────────────────────────────────────────────
///  Tool Abstraction Layer
///
///  Maps high-level goals ("reach high shelf") to concrete motor
///  chains ("use extendable arm") via functional affordance matching.
///
///  An affordance is a *functional signature* that describes what a
///  tool does in terms of sensorimotor properties — not what it's
///  called. A step-stool and a stick both have `reach_extension > 0.3`,
///  so either can be selected for a reach task regardless of label.
///
///  All matching is deterministic: cosine similarity on affordance
///  signature vectors. No ML, no NLP, no randomness.
/// ────────────────────────────────────────────────────────────

/// The 3 functional affordance dimensions:
///   0: reach_extension  — how far the tool extends reach (0..1)
///   1: energy_cost      — activation energy cost to use (0..1)
///   2: precision        — how precisely the tool can be controlled (0..1)
///   3: force            — how much force it can apply (0..1)
///   4: temporal_dur     — how long the effect lasts (0..1)
pub const AFFORDANCE_DIMS: usize = 5;

/// A functional affordance signature — pure sensorimotor vector.
#[derive(Debug, Clone, Copy)]
pub struct AffordanceSignature {
    pub reach_extension: f64,
    pub energy_cost: f64,
    pub precision: f64,
    pub force: f64,
    pub temporal_duration: f64,
}

impl AffordanceSignature {
    pub fn new(reach: f64, cost: f64, precision: f64, force: f64, duration: f64) -> Self {
        AffordanceSignature {
            reach_extension: reach.clamp(0.0, 1.0),
            energy_cost: cost.clamp(0.0, 1.0),
            precision: precision.clamp(0.0, 1.0),
            force: force.clamp(0.0, 1.0),
            temporal_duration: duration.clamp(0.0, 1.0),
        }
    }

    /// Zero signature (no affordance).
    pub fn zero() -> Self {
        AffordanceSignature {
            reach_extension: 0.0,
            energy_cost: 0.0,
            precision: 0.0,
            force: 0.0,
            temporal_duration: 0.0,
        }
    }

    /// Convert to a 5-element array for distance computation.
    pub fn as_array(&self) -> [f64; AFFORDANCE_DIMS] {
        [
            self.reach_extension,
            self.energy_cost,
            self.precision,
            self.force,
            self.temporal_duration,
        ]
    }

    /// Cosine similarity between this and another signature.
    pub fn similarity(&self, other: &AffordanceSignature) -> f64 {
        let a = self.as_array();
        let b = other.as_array();
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm_a < 1e-10 || norm_b < 1e-10 {
            return 0.0;
        }
        (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }

    /// Manhattan distance to another signature.
    pub fn distance(&self, other: &AffordanceSignature) -> f64 {
        let a = self.as_array();
        let b = other.as_array();
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum::<f64>() / AFFORDANCE_DIMS as f64
    }

    /// Check if this signature satisfies a requirement signature.
    /// Returns true if all required dimensions are met or exceeded.
    pub fn satisfies(&self, requirement: &AffordanceSignature) -> bool {
        self.reach_extension >= requirement.reach_extension * 0.8
            && self.energy_cost <= requirement.energy_cost * 1.2
            && self.precision >= requirement.precision * 0.8
            && self.force >= requirement.force * 0.8
    }

    /// Create from a Grounding::Affordance.
    pub fn from_grounding(grounding: &Grounding) -> Option<Self> {
        if let Grounding::Affordance { reach_extension, energy_cost, .. } = grounding {
            Some(AffordanceSignature {
                reach_extension: *reach_extension,
                energy_cost: *energy_cost,
                precision: 0.5,
                force: 0.5,
                temporal_duration: 0.5,
            })
        } else {
            None
        }
    }

    /// Derive a tool's affordance signature from its PrimitiveVector.
    /// Maps the 5 physical dimensions to the 5 affordance dimensions.
    pub fn from_primitive(pv: &PrimitiveVector) -> Self {
        AffordanceSignature {
            reach_extension: (pv.spatial * 0.6 + pv.velocity * 0.4).clamp(0.0, 1.0),
            energy_cost: (pv.mass * 0.7 + pv.velocity * 0.3).clamp(0.0, 1.0),
            precision: (1.0 - pv.velocity.abs() + pv.valence.abs() * 0.5).clamp(0.0, 1.0) * 0.5,
            force: (pv.mass * 0.8).clamp(0.0, 1.0),
            temporal_duration: pv.temporal.clamp(0.0, 1.0),
        }
    }
}

impl Default for AffordanceSignature {
    fn default() -> Self {
        Self::zero()
    }
}

/// A registered affordance — maps a concept node to its functional signature.
#[derive(Debug, Clone)]
pub struct AffordanceEntry {
    pub node_id: NodeId,
    pub label: String,
    pub signature: AffordanceSignature,
    pub manipulator_type: AffordanceType,
}

/// The affordance registry — maintains functional signatures for tool-like nodes.
///
/// Populated at startup by scanning the semantic graph for nodes with
/// Grounding::Affordance or NodeType::Tool. Can also be updated dynamically.
pub struct AffordanceRegistry {
    /// Registered affordances.
    entries: Vec<AffordanceEntry>,
    ctx: Arc<SemanticContext>,
}

impl AffordanceRegistry {
    pub fn new(ctx: Arc<SemanticContext>) -> Self {
        AffordanceRegistry {
            entries: Vec::with_capacity(16),
            ctx,
        }
    }

    /// Scan the semantic graph and register all tool/affordance nodes.
    pub fn scan_graph(&mut self) {
        let graph = self.ctx.graph.read();
        self.entries.clear();

        for i in 2..graph.len() {
            if let Some(n) = graph.get(NodeId::from_raw(i as u64)) {
                let node = n.read();

                // Check for Tool nodes
                if node.node_type == NodeType::Tool {
                    // Derive affordance from PrimitiveVector or use defaults
                    let pv = primitive_for(&node.label).unwrap_or(PrimitiveVector::zero());
                    let signature = AffordanceSignature::from_primitive(&pv);
                    let manip_type = self.infer_affordance_type(&signature);
                    self.entries.push(AffordanceEntry {
                        node_id: node.id,
                        label: node.label.clone(),
                        signature,
                        manipulator_type: manip_type,
                    });
                    continue;
                }

                // Check for nodes with Affordance grounding
                if let Grounding::Affordance { reach_extension, energy_cost, manipulator_type, .. } = &node.grounding {
                    let signature = AffordanceSignature {
                        reach_extension: *reach_extension,
                        energy_cost: *energy_cost,
                        precision: 0.5,
                        force: 0.5,
                        temporal_duration: 0.5,
                    };
                    self.entries.push(AffordanceEntry {
                        node_id: node.id,
                        label: node.label.clone(),
                        signature,
                        manipulator_type: *manipulator_type,
                    });
                    continue;
                }

                // Derive affordance from concept nodes with high spatial or mass
                if node.node_type == NodeType::Concept || node.node_type == NodeType::Entity {
                    let pv = primitive_for(&node.label).unwrap_or(PrimitiveVector::zero());
                    if pv.spatial > 0.3 || pv.mass > 0.3 {
                        let signature = AffordanceSignature::from_primitive(&pv);
                        let manip_type = self.infer_affordance_type(&signature);
                        self.entries.push(AffordanceEntry {
                            node_id: node.id,
                            label: node.label.clone(),
                            signature,
                            manipulator_type: manip_type,
                        });
                    }
                }
            }
        }
    }

    /// Infer the affordance type from a signature.
    fn infer_affordance_type(&self, sig: &AffordanceSignature) -> AffordanceType {
        if sig.reach_extension > 0.6 && sig.force < 0.3 {
            AffordanceType::Reach
        } else if sig.force > 0.6 && sig.precision < 0.3 {
            AffordanceType::Manipulate
        } else if sig.energy_cost > 0.7 {
            AffordanceType::Support
        } else if sig.precision > 0.6 {
            AffordanceType::Surface
        } else if sig.force > 0.4 && sig.reach_extension > 0.3 {
            AffordanceType::Connect
        } else if sig.temporal_duration > 0.6 {
            AffordanceType::Contain
        } else {
            AffordanceType::Manipulate
        }
    }

    /// Register a tool node manually.
    pub fn register(&mut self, node_id: NodeId, label: &str, signature: AffordanceSignature, manip_type: AffordanceType) {
        // Update or insert
        if let Some(entry) = self.entries.iter_mut().find(|e| e.node_id == node_id) {
            entry.signature = signature;
            entry.manipulator_type = manip_type;
        } else {
            self.entries.push(AffordanceEntry {
                node_id,
                label: label.to_string(),
                signature,
                manipulator_type: manip_type,
            });
        }
    }

    /// Unregister a tool node.
    pub fn unregister(&mut self, node_id: NodeId) {
        self.entries.retain(|e| e.node_id != node_id);
    }

    /// Find tools that can satisfy a given requirement signature.
    ///
    /// Matches by functional signature similarity, not by label.
    /// Returns results sorted by similarity score descending.
    pub fn affordance_match(&self, requirement: &AffordanceSignature, top_k: usize) -> Vec<AffordanceEntry> {
        let mut scored: Vec<(f64, &AffordanceEntry)> = self.entries.iter()
            .filter(|e| e.signature.satisfies(requirement))
            .map(|e| (e.signature.similarity(requirement), e))
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        scored.into_iter().map(|(_, e)| e.clone()).collect()
    }

    /// Find the best tool for a given goal by matching the goal's
    /// required affordance to registered tools.
    pub fn best_tool_for_goal(&self, goal_affordance: &AffordanceSignature) -> Option<AffordanceEntry> {
        self.affordance_match(goal_affordance, 1).into_iter().next()
    }

    /// Given a high-level goal label (e.g., "reach_high", "grasp_object"),
    /// derive the required affordance signature and find matching tools.
    pub fn plan_tool_chain(&self, goal_label: &str) -> Vec<AffordanceEntry> {
        let required = self.required_affordance_for(goal_label);
        self.affordance_match(&required, 3)
    }

    /// Derive the required affordance signature for a goal label.
    fn required_affordance_for(&self, goal_label: &str) -> AffordanceSignature {
        // Match against known goal patterns
        if goal_label.contains("reach") || goal_label.contains("grasp") {
            AffordanceSignature::new(0.6, 0.3, 0.5, 0.2, 0.1)
        } else if goal_label.contains("lift") || goal_label.contains("move") {
            AffordanceSignature::new(0.2, 0.5, 0.3, 0.7, 0.1)
        } else if goal_label.contains("contain") || goal_label.contains("store") {
            AffordanceSignature::new(0.1, 0.2, 0.3, 0.2, 0.8)
        } else if goal_label.contains("sense") || goal_label.contains("detect") {
            AffordanceSignature::new(0.5, 0.1, 0.8, 0.0, 0.3)
        } else if goal_label.contains("support") || goal_label.contains("hold") {
            AffordanceSignature::new(0.1, 0.1, 0.2, 0.8, 0.6)
        } else if goal_label.contains("connect") || goal_label.contains("attach") {
            AffordanceSignature::new(0.5, 0.3, 0.4, 0.5, 0.5)
        } else {
            AffordanceSignature::new(0.3, 0.3, 0.5, 0.3, 0.3)
        }
    }

    /// Get all registered affordances.
    pub fn entries(&self) -> &[AffordanceEntry] {
        &self.entries
    }

    /// Count of registered affordances.
    pub fn count(&self) -> usize {
        self.entries.len()
    }

    /// Inject affordance info into the semantic graph as edges.
    /// Creates Affords edges from registered tools to their
    /// inferred action types.
    pub fn materialize_affordances(&self) {
        let mut graph = self.ctx.graph.write();
        for entry in &self.entries {
            // Find action nodes that match this affordance type
            for i in 2..graph.len() {
                if let Some(n) = graph.get(NodeId::from_raw(i as u64)) {
                    let node = n.read();
                    if node.node_type == NodeType::Action {
                        // Check if action label is related to the affordance type
                        let action_label = node.label.to_lowercase();
                        let should_link = match entry.manipulator_type {
                            AffordanceType::Reach => action_label.contains("reach") || action_label.contains("extend"),
                            AffordanceType::Contain => action_label.contains("contain") || action_label.contains("store"),
                            AffordanceType::Surface => action_label.contains("surface") || action_label.contains("touch"),
                            AffordanceType::Sense => action_label.contains("sense") || action_label.contains("read"),
                            AffordanceType::Support => action_label.contains("support") || action_label.contains("hold"),
                            AffordanceType::Connect => action_label.contains("connect") || action_label.contains("tie"),
                            AffordanceType::Manipulate => action_label.contains("move") || action_label.contains("use"),
                        };
                        if should_link {
                            drop(node);
                            if let Some(tool_node) = graph.get(entry.node_id) {
                                let existing = tool_node.read().edges.iter()
                                    .any(|e| e.target == node.id && e.relation == Relation::Affords);
                                if !existing {
                                    tool_node.write().edges.push(Edge::new(Relation::Affords, node.id));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Clear all registered affordances.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}
