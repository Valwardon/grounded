use semantic_graph::prelude::*;

// ────────────────────────────────────────────────────────────
//  CuriosityBudget — energy-aware curiosity model
//
//  Replaces the hard recursion depth cap (MAX_RECURSION_DEPTH=10)
//  with an energy pool $E_{curious}$ that depletes as the system
//  explores. Each recursive step consumes energy proportional to:
//
//    E_consume = alpha * dist(SELF, concept) + beta * arousal + gamma * error_rate
//
//  Where:
//    alpha = 0.3 (semantic distance weight)
//    beta  = 0.4 (arousal weight — high arousal clamps search)
//    gamma = 0.2 (structural error weight)
//
//  Halt when E_remaining falls below dynamic threshold:
//    threshold = novelty * 0.15  (high novelty = more exploration)
//
//  This allows infinite depth for novel/rewarding concepts while
//  naturally truncating low-value branches.
// ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct CuriosityBudget {
    /// Total energy pool allocated for this curiosity chain.
    pub total_energy: f64,
    /// Remaining energy.
    pub remaining: f64,
    /// Accumulated structural errors in this chain.
    pub error_count: u64,
}

impl CuriosityBudget {
    /// Create a new budget with a given total energy.
    pub fn new(total_energy: f64) -> Self {
        CuriosityBudget {
            total_energy,
            remaining: total_energy,
            error_count: 0,
        }
    }

    /// Default budget: 10.0 units (enough for ~10-20 steps).
    pub fn default() -> Self {
        CuriosityBudget::new(10.0)
    }

    /// Compute energy cost for one recursive step.
    ///
    /// Parameters:
    ///   semantic_distance: distance from SELF to the target concept (0.0-5.0)
    ///   arousal: global arousal level (0.0-1.0)
    ///   novelty: global novelty level (0.0-1.0) — acts as discount
    pub fn step_cost(&self, semantic_distance: f64, arousal: f64, novelty: f64) -> f64 {
        let alpha = 0.3;
        let beta = 0.4;
        let gamma = 0.2;

        let error_rate = if self.total_energy > 0.0 {
            self.error_count as f64 / self.total_energy
        } else {
            0.0
        };

        let base = alpha * semantic_distance.min(5.0) / 5.0
            + beta * arousal
            + gamma * error_rate.min(1.0);

        // Novelty discounts energy cost — high novelty = cheaper exploration
        let discount = 1.0 - novelty * 0.3;

        (base * discount * 0.5).clamp(0.05, 2.0)
    }

    /// Dynamic halt threshold: the minimum energy required to continue.
    /// High novelty lowers the threshold (allows deeper search).
    pub fn halt_threshold(&self, novelty: f64) -> f64 {
        (novelty * 0.15).clamp(0.01, 0.5)
    }

    /// Consume energy for one step. Returns false if budget is exhausted.
    pub fn consume(&mut self, semantic_distance: f64, arousal: f64, novelty: f64) -> bool {
        let cost = self.step_cost(semantic_distance, arousal, novelty);
        if self.remaining - cost < self.halt_threshold(novelty) {
            return false;
        }
        self.remaining -= cost;
        true
    }

    /// Record an error that increases future step costs.
    pub fn record_error(&mut self) {
        self.error_count += 1;
    }

    /// Ratio of remaining energy (0.0 = empty, 1.0 = full).
    pub fn ratio(&self) -> f64 {
        if self.total_energy > 0.0 {
            (self.remaining / self.total_energy).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

// ────────────────────────────────────────────────────────────
//  Knowledge Gap detection
//
//  The core principle: when a token has zero relational edges in
//  the semantic graph, that's a structural instability. The engine
//  cannot close its reasoning loop until every ungrounded identifier
//  is resolved down to fundamental physical primitives.
// ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct KnowledgeGap {
    /// The ungrounded token (e.g., "Cat", "Pirate", "Bipedal")
    pub token: String,
    /// The node that referenced this token (if any)
    pub parent_node_id: Option<NodeId>,
    /// Curiosity budget — energy-aware exploration limit
    pub budget: CuriosityBudget,
    /// How this gap was discovered
    pub source: GapSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GapSource {
    UserInput,
    SensorFrame,
    RecursiveResolution,
    FileIngestion,
    AssetDecomposition,
}

impl KnowledgeGap {
    pub fn new(token: &str, source: GapSource) -> Self {
        KnowledgeGap {
            token: token.to_string(),
            parent_node_id: None,
            budget: CuriosityBudget::default(),
            source,
        }
    }

    pub fn with_parent(mut self, parent: NodeId) -> Self {
        self.parent_node_id = Some(parent);
        self
    }

    pub fn with_budget(mut self, budget: CuriosityBudget) -> Self {
        self.budget = budget;
        self
    }
}

/// The gap detector. Scans tokens against the semantic graph
/// and emits KnowledgeGap events for anything ungrounded.
pub struct GapDetector {
    ctx: Arc<SemanticContext>,
    min_edges_for_grounded: usize,
}

impl GapDetector {
    pub fn new(ctx: Arc<SemanticContext>) -> Self {
        GapDetector {
            ctx,
            min_edges_for_grounded: 1,
        }
    }

    /// Inspect a set of tokens against the graph. Any token that
    /// has no node, or whose node has zero edges, is a gap.
    pub fn detect(&self, tokens: &[String], source: GapSource) -> Vec<KnowledgeGap> {
        let mut gaps = Vec::new();
        let graph = self.ctx.graph.read();

        for token in tokens {
            let normalized = token.to_lowercase();
            let node_id = graph.lookup(&normalized);

            match node_id {
                Some(id) => {
                    // Node exists — check if it's grounded (has edges)
                    if let Some(node) = graph.get(id) {
                        let n = node.read();
                        if n.edges.len() < self.min_edges_for_grounded
                            && n.node_type != NodeType::Sensor
                            && n.node_type != NodeType::State
                        {
                            // Found a node with no relational grounding
                            gaps.push(KnowledgeGap {
                                token: normalized,
                                parent_node_id: Some(id),
                                budget: CuriosityBudget::default(),
                                source,
                            });
                        }
                    }
                }
                None => {
                    // No node at all — complete gap
                    gaps.push(KnowledgeGap {
                        token: normalized,
                        parent_node_id: None,
                        budget: CuriosityBudget::default(),
                        source,
                    });
                }
            }
        }

        gaps
    }

    /// Check a single concept for grounding. Returns true if the
    /// token is fully resolved down to primitives.
    pub fn is_grounded(&self, token: &str) -> bool {
        let graph = self.ctx.graph.read();
        let node_id = match graph.lookup(&token.to_lowercase()) {
            Some(id) => id,
            None => return false,
        };

        // Check that the node has edges and each connected node
        // also has edges (recursive grounding check to depth 1)
        let node = match graph.get(node_id) {
            Some(n) => n.read(),
            None => return false,
        };

        if node.edges.is_empty() && node.node_type != NodeType::Sensor {
            return false;
        }

        // For Entity and Concept nodes, verify they connect to grounded children
        if node.node_type == NodeType::Entity || node.node_type == NodeType::Concept {
            for edge in &node.edges {
                if let Some(child) = graph.get(edge.target) {
                    let c = child.read();
                    if c.edges.is_empty() && c.node_type != NodeType::Sensor {
                        return false; // child is ungrounded
                    }
                }
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_context() -> Arc<SemanticContext> {
        let mut g = GraphArena::with_capacity(16);

        // Grounded: cat with edges
        let cat_id = g.insert(GroundedNode {
            id: NodeId::ZERO,
            label: "cat".into(),
            node_type: NodeType::Entity,
            grounding: Grounding::Abstract,
            decay: 0.9,
            threshold: 10.0,
            base_activation: 0.0,
            edges: vec![
                Edge::new(Relation::IsA, NodeId::from_raw(2)), // "animal"
                Edge::new(Relation::HasProperty, NodeId::from_raw(3)), // "quadruped"
            ],
            valence: 0.0,
            mean_error: 0.0,
            variance: 0.0,
        });

        // Grounded: animal
        g.insert(GroundedNode {
            id: NodeId::ZERO,
            label: "animal".into(),
            node_type: NodeType::Concept,
            grounding: Grounding::Abstract,
            decay: 0.9,
            threshold: 10.0,
            base_activation: 0.0,
            edges: vec![Edge::new(Relation::HasProperty, NodeId::from_raw(3))],
            valence: 0.0,
            mean_error: 0.0,
            variance: 0.0,
        });

        // Grounded: quadruped (primitive — has edges)
        g.insert(GroundedNode {
            id: NodeId::ZERO,
            label: "quadruped".into(),
            node_type: NodeType::Concept,
            grounding: Grounding::Abstract,
            decay: 0.9,
            threshold: 10.0,
            base_activation: 0.0,
            edges: vec![Edge::new(Relation::HasProperty, cat_id)],
            valence: 0.0,
            mean_error: 0.0,
            variance: 0.0,
        });

        SemanticContext::new(g)
    }

    #[test]
    fn detects_missing_token() {
        let ctx = test_context();
        let detector = GapDetector::new(ctx);
        let tokens = vec!["pirate".to_string()];
        let gaps = detector.detect(&tokens, GapSource::UserInput);
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0].token, "pirate");
    }

    #[test]
    fn known_token_is_not_a_gap() {
        let ctx = test_context();
        let detector = GapDetector::new(ctx);
        let tokens = vec!["cat".to_string()];
        let gaps = detector.detect(&tokens, GapSource::UserInput);
        assert_eq!(gaps.len(), 0);
    }

    #[test]
    fn grounded_check() {
        let ctx = test_context();
        let detector = GapDetector::new(ctx);
        assert!(detector.is_grounded("cat"));
        assert!(!detector.is_grounded("pirate"));
    }
}
