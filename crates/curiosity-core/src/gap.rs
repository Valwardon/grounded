use semantic_graph::prelude::*;

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
    /// Recursion depth — circuit breaker at 10
    pub recursion_depth: u8,
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
    pub fn new(token: &str, depth: u8, source: GapSource) -> Self {
        KnowledgeGap {
            token: token.to_string(),
            parent_node_id: None,
            recursion_depth: depth,
            source,
        }
    }

    pub fn with_parent(mut self, parent: NodeId) -> Self {
        self.parent_node_id = Some(parent);
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
                                recursion_depth: 0,
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
                        recursion_depth: 0,
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
                Edge {
                    relation: Relation::IsA,
                    target: NodeId::from_raw(2), // "animal"
                    weight_override: None,
                },
                Edge {
                    relation: Relation::HasProperty,
                    target: NodeId::from_raw(3), // "quadruped"
                    weight_override: None,
                },
            ],
            valence: 0.0,
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
            edges: vec![Edge {
                relation: Relation::HasProperty,
                target: NodeId::from_raw(3),
                weight_override: None,
            }],
            valence: 0.0,
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
            edges: vec![Edge {
                relation: Relation::HasProperty,
                target: cat_id,
                weight_override: None,
            }],
            valence: 0.0,
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
