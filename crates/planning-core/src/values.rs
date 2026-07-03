use std::sync::Arc;
use cognitive_core::DriveHook;
use semantic_graph::prelude::*;

/// Hardwired intrinsic motivation system.
///
/// Each drive is a persistent source of activation bias that shapes
/// the cognitive engine's behavior beyond raw curiosity. Drives
/// are implemented as persistent Value nodes in the semantic graph
/// with fixed activation injection during each tick.
pub struct ValueSystem {
    ctx: Arc<SemanticContext>,
    drives: Vec<DriveDef>,
    long_term_goals: Vec<LongTermGoal>,
}

/// Definition of a hardwired drive.
#[derive(Debug, Clone)]
pub struct DriveDef {
    pub drive_type: DriveType,
    pub label: &'static str,
    pub base_intensity: f64,
    pub current_intensity: f64,
}

/// A persistent long-term goal that survives restarts.
#[derive(Debug, Clone)]
pub struct LongTermGoal {
    pub label: String,
    pub priority: f64,
    pub category: ValueCategory,
    pub node_id: Option<NodeId>,
}

impl ValueSystem {
    pub fn new(ctx: Arc<SemanticContext>) -> Self {
        let drives = vec![
            DriveDef {
                drive_type: DriveType::Curiosity,
                label: "curiosity",
                base_intensity: 0.4,
                current_intensity: 0.4,
            },
            DriveDef {
                drive_type: DriveType::Safety,
                label: "safety",
                base_intensity: 0.6,
                current_intensity: 0.6,
            },
            DriveDef {
                drive_type: DriveType::Mastery,
                label: "mastery",
                base_intensity: 0.3,
                current_intensity: 0.3,
            },
            DriveDef {
                drive_type: DriveType::Affiliation,
                label: "affiliation",
                base_intensity: 0.2,
                current_intensity: 0.2,
            },
            DriveDef {
                drive_type: DriveType::Exploration,
                label: "exploration",
                base_intensity: 0.5,
                current_intensity: 0.5,
            },
            DriveDef {
                drive_type: DriveType::Conservation,
                label: "conservation",
                base_intensity: 0.3,
                current_intensity: 0.3,
            },
        ];

        let mut vs = ValueSystem {
            ctx,
            drives,
            long_term_goals: Vec::new(),
        };
        vs.initialize_drive_nodes();
        vs
    }

    /// Create Drive and Value nodes in the semantic graph.
    fn initialize_drive_nodes(&mut self) {
        let mut graph = self.ctx.graph.write();
        for drive in &self.drives {
            let id = graph.insert(GroundedNode {
                id: NodeId::ZERO,
                label: format!("drive_{}", drive.label),
                node_type: NodeType::Value,
                grounding: Grounding::Drive {
                    drive_type: drive.drive_type,
                    intensity: drive.base_intensity,
                },
                decay: 0.99,
                threshold: 20.0,
                base_activation: drive.base_intensity,
                edges: vec![Edge::new(Relation::Drives, NodeId::SELF)],
                epistemic_status: EpistemicStatus::CoreConcept,
                valence: 0.3,
                mean_error: 0.0,
                variance: 0.0,
            });
            drop(graph);
            self.ctx.link_to_self(Relation::Drives, id);
            graph = self.ctx.graph.write();
        }

        // Create global Value nodes for each value category
        let categories = [
            (ValueCategory::Knowledge, 0.8, "value_knowledge"),
            (ValueCategory::Safety, 0.9, "value_safety"),
            (ValueCategory::Efficiency, 0.5, "value_efficiency"),
            (ValueCategory::Novelty, 0.6, "value_novelty"),
            (ValueCategory::Stability, 0.7, "value_stability"),
            (ValueCategory::Growth, 0.4, "value_growth"),
        ];

        for (cat, weight, label) in &categories {
            let id = graph.insert(GroundedNode {
                id: NodeId::ZERO,
                label: label.to_string(),
                node_type: NodeType::Value,
                grounding: Grounding::Value {
                    weight: *weight,
                    category: *cat,
                },
                decay: 1.0,
                threshold: 25.0,
                base_activation: *weight,
                edges: vec![Edge::new(Relation::Drives, NodeId::SELF)],
                epistemic_status: EpistemicStatus::CoreConcept,
                valence: 0.5,
                mean_error: 0.0,
                variance: 0.0,
            });
            drop(graph);
            self.ctx.link_to_self(Relation::Drives, id);
            graph = self.ctx.graph.write();
        }
    }

    /// Register a long-term goal.
    pub fn add_long_term_goal(&mut self, label: &str, priority: f64, category: ValueCategory) {
        // Create a Goal node in the graph
        let mut graph = self.ctx.graph.write();
        let id = graph.insert(GroundedNode {
            id: NodeId::ZERO,
            label: format!("ltg_{}", label),
            node_type: NodeType::Goal,
            grounding: Grounding::Goal {
                priority,
                deadline_tick: 0,
                status: GoalStatus::Active,
            },
            decay: 1.0,
            threshold: 30.0,
            base_activation: priority * 0.8,
            edges: vec![Edge::new(Relation::AssociatedWith, NodeId::SELF)],
            epistemic_status: EpistemicStatus::CoreConcept,
            valence: 0.5,
            mean_error: 0.0,
            variance: 0.0,
        });
        self.ctx.link_to_self(Relation::AssociatedWith, id);

        self.long_term_goals.push(LongTermGoal {
            label: label.to_string(),
            priority,
            category,
            node_id: Some(id),
        });
    }

    /// Compute drive intensities based on current neuromodulator state.
    /// Called every tick to adjust drive influence on the activation landscape.
    pub fn update_drive_intensities(&mut self, novelty: f64, arousal: f64, reward: f64) {
        for drive in &mut self.drives {
            match drive.drive_type {
                DriveType::Curiosity => {
                    // Novelty increases curiosity
                    drive.current_intensity = (drive.base_intensity + novelty * 0.3).clamp(0.0, 1.0);
                }
                DriveType::Safety => {
                    // High arousal increases safety drive (caution)
                    drive.current_intensity = (drive.base_intensity + arousal * 0.4).clamp(0.0, 1.0);
                }
                DriveType::Mastery => {
                    // Low prediction error = mastery growing
                    drive.current_intensity = (drive.base_intensity + reward * 0.2).clamp(0.0, 1.0);
                }
                DriveType::Affiliation => {
                    // Reward increases affiliation seeking
                    drive.current_intensity = (drive.base_intensity + reward * 0.15).clamp(0.0, 1.0);
                }
                DriveType::Exploration => {
                    // Low novelty + low arousal = boredom → explore
                    let boredom = (1.0 - novelty) * (1.0 - arousal);
                    drive.current_intensity = (drive.base_intensity + boredom * 0.3).clamp(0.0, 1.0);
                }
                DriveType::Conservation => {
                    // High arousal = conserve energy
                    drive.current_intensity = (drive.base_intensity + arousal * 0.2).clamp(0.0, 1.0);
                }
            }
        }

        // Update drive node base activations in the graph
        let mut graph = self.ctx.graph.write();
        for drive in &self.drives {
            let label = format!("drive_{}", drive.label);
            if let Some(id) = graph.lookup(&label) {
                if let Some(node) = graph.get(id) {
                    let mut n = node.write();
                    n.base_activation = drive.current_intensity;
                    if let Grounding::Drive { ref mut intensity, .. } = n.grounding {
                        *intensity = drive.current_intensity;
                    }
                }
            }
        }
    }

    /// Get the activation bias for a specific drive type.
    /// Used by the cognitive daemon to inject drive-based activation.
    pub fn drive_intensity(&self, drive_type: DriveType) -> f64 {
        self.drives
            .iter()
            .find(|d| d.drive_type == drive_type)
            .map(|d| d.current_intensity)
            .unwrap_or(0.0)
    }

    /// Return the currently dominant drive (highest intensity).
    pub fn dominant_drive(&self) -> Option<DriveType> {
        self.drives
            .iter()
            .max_by(|a, b| a.current_intensity.partial_cmp(&b.current_intensity).unwrap_or(std::cmp::Ordering::Equal))
            .map(|d| d.drive_type)
    }

    /// Get all active long-term goals.
    pub fn long_term_goals(&self) -> &[LongTermGoal] {
        &self.long_term_goals
    }

    /// Drive activation bias to apply during the tick loop.
    /// Returns a list of (node_id, activation_amount) pairs.
    pub fn drive_biases(&self) -> Vec<(NodeId, f64)> {
        let graph = self.ctx.graph.read();
        let mut biases: Vec<(NodeId, f64)> = Vec::new();
        for i in 2..graph.len() {
            if let Some(n) = graph.get(NodeId::from_raw(i as u64)) {
                let node = n.read();
                if node.node_type == NodeType::Value {
                    if let Grounding::Drive { intensity, .. } = node.grounding {
                        if intensity > 0.01 {
                            biases.push((node.id, intensity * 0.1));
                        }
                    } else if let Grounding::Value { weight, .. } = node.grounding {
                        if weight > 0.01 {
                            biases.push((node.id, weight * 0.05));
                        }
                    }
                }
            }
        }
        biases
    }
}

impl DriveHook for ValueSystem {
    fn drive_biases(&mut self, novelty: f64, arousal: f64, reward: f64) -> Vec<(NodeId, f64)> {
        self.update_drive_intensities(novelty, arousal, reward);
        // The trait returns the biases directly (calls our inherent method)
        ValueSystem::drive_biases(self)
    }
}
