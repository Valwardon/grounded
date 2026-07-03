use semantic_graph::prelude::*;

// ────────────────────────────────────────────────────────────
//  Deterministic predicate parser
//
//  When the harvester fetches a raw definition string for a
//  concept (e.g., "cat is an animal, cat has fur, cat has tail"),
//  this parser extracts relational predicates using a grammar
//  table — no ML, no embedding similarity.
//
//  Supported grammar patterns:
//    "<X> is a <Y>"       → X IsA Y
//    "<X> is an <Y>"      → X IsA Y
//    "<X> has <Y>"        → X HasProperty Y
//    "<X> can <Y>"        → X Activates Y (capability)
//    "<X> needs <Y>"      → X Requires Y
//    "<X> causes <Y>"     → X CausedBy Y (inverse)
//    "<X> is like <Y>"    → X AssociatedWith Y
// ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Predicate {
    pub subject: String,
    pub relation: Relation,
    pub object: String,
}

#[derive(Debug, Clone)]
pub struct ResolvedDefinition {
    /// The main node that was created
    pub main_node_id: NodeId,
    /// All predicates extracted from the raw definition
    pub predicates: Vec<Predicate>,
    /// New tokens discovered that need their own resolution
    pub dependencies: Vec<String>,
}

/// Resolves a raw text definition into structured graph nodes.
pub struct DefinitionResolver {
    ctx: Arc<SemanticContext>,
}

impl DefinitionResolver {
    pub fn new(ctx: Arc<SemanticContext>) -> Self {
        DefinitionResolver { ctx }
    }

    /// Parse a raw definition string and commit all discovered
    /// predicates into the semantic graph.
    pub fn resolve(
        &self,
        token: &str,
        raw_definition: &str,
        parent_node: Option<NodeId>,
    ) -> ResolvedDefinition {
        let predicates = self.parse_predicates(raw_definition);
        let mut dependencies = Vec::new();
        let mut graph = self.ctx.graph.write();

        // Create or find the main node for this token
        let main_id = match graph.lookup(&token.to_lowercase()) {
            Some(id) => id,
            None => {
                let id = graph.insert(GroundedNode {
                    id: NodeId::ZERO,
                    label: token.to_lowercase(),
                    node_type: NodeType::Concept,
                    grounding: Grounding::Abstract,
                    decay: 0.9,
                    threshold: 10.0,
                    base_activation: 0.0,
                    edges: Vec::with_capacity(predicates.len()),
                    valence: 0.0,
                });
                id
            }
        };

        // Link to parent if provided
        if let Some(parent) = parent_node {
            if let Some(parent_node) = graph.get(parent) {
                parent_node
                    .write()
                    .edges
                    .push(Edge {
                        relation: Relation::AssociatedWith,
                        target: main_id,
                        weight_override: None,
                    });
            }
        }

        // Process each predicate — create child nodes as needed
        for pred in &predicates {
            let object_id = match graph.lookup(&pred.object.to_lowercase()) {
                Some(id) => id,
                None => {
                    // The object doesn't exist yet — create a placeholder
                    // and record it as a dependency for recursive resolution
                    let id = graph.insert(GroundedNode {
                        id: NodeId::ZERO,
                        label: pred.object.to_lowercase(),
                        node_type: NodeType::Concept,
                        grounding: Grounding::Abstract,
                        decay: 0.9,
                        threshold: 10.0,
                        base_activation: 0.0,
                        edges: Vec::new(),
                        valence: 0.0,
                    });
                    if !dependencies.contains(&pred.object.to_lowercase()) {
                        dependencies.push(pred.object.to_lowercase());
                    }
                    id
                }
            };

            // Add edge from main node to object
            if let Some(main_node) = graph.get(main_id) {
                main_node.write().edges.push(Edge {
                    relation: pred.relation,
                    target: object_id,
                    weight_override: None,
                });
            }
        }

        ResolvedDefinition {
            main_node_id: main_id,
            predicates,
            dependencies,
        }
    }

    /// Parse a raw text string into structured predicates using
    /// deterministic grammar patterns.
    fn parse_predicates(&self, raw: &str) -> Vec<Predicate> {
        let mut predicates = Vec::new();
        let lower = raw.to_lowercase();

        // Split on sentence boundaries: '.', ';', '\n'
        for sentence in lower.split(|c: char| c == '.' || c == ';' || c == '\n') {
            let sentence = sentence.trim();
            if sentence.is_empty() {
                continue;
            }

            // Pattern: "<subject> is a[n] <object>"
            if let Some(rest) = sentence.strip_prefix("is a ") {
                let parts: Vec<&str> = rest.splitn(2, |c: char| c == ' ' || c == ',').collect();
                if !parts.is_empty() {
                    predicates.push(Predicate {
                        subject: String::new(), // filled by caller
                        relation: Relation::IsA,
                        object: parts[0].trim().to_string(),
                    });
                }
            } else if let Some(rest) = sentence.strip_prefix("is an ") {
                let parts: Vec<&str> = rest.splitn(2, |c: char| c == ' ' || c == ',').collect();
                if !parts.is_empty() {
                    predicates.push(Predicate {
                        subject: String::new(),
                        relation: Relation::IsA,
                        object: parts[0].trim().to_string(),
                    });
                }
            }
            // Pattern: "<subject> has <object>"
            else if let Some(rest) = sentence.strip_prefix("has ") {
                let obj = rest
                    .split(|c: char| c == ' ' || c == ',')
                    .next()
                    .unwrap_or(rest)
                    .trim();
                predicates.push(Predicate {
                    subject: String::new(),
                    relation: Relation::HasProperty,
                    object: obj.to_string(),
                });
            }
            // Pattern: "<subject> can <object>"
            else if let Some(rest) = sentence.strip_prefix("can ") {
                let obj = rest
                    .split(|c: char| c == ' ' || c == ',')
                    .next()
                    .unwrap_or(rest)
                    .trim();
                predicates.push(Predicate {
                    subject: String::new(),
                    relation: Relation::Activates,
                    object: obj.to_string(),
                });
            }
            // Pattern: "<subject> needs <object>"
            else if let Some(rest) = sentence.strip_prefix("needs ") {
                let obj = rest
                    .split(|c: char| c == ' ' || c == ',')
                    .next()
                    .unwrap_or(rest)
                    .trim();
                predicates.push(Predicate {
                    subject: String::new(),
                    relation: Relation::Requires,
                    object: obj.to_string(),
                });
            }
            // Pattern: "<subject> causes <object>"
            else if let Some(rest) = sentence.strip_prefix("causes ") {
                let obj = rest
                    .split(|c: char| c == ' ' || c == ',')
                    .next()
                    .unwrap_or(rest)
                    .trim();
                predicates.push(Predicate {
                    subject: String::new(),
                    relation: Relation::CausedBy,
                    object: obj.to_string(),
                });
            }
            // Pattern: "<subject> is like <object>"
            else if let Some(rest) = sentence.strip_prefix("is like ") {
                let obj = rest
                    .split(|c: char| c == ' ' || c == ',')
                    .next()
                    .unwrap_or(rest)
                    .trim();
                predicates.push(Predicate {
                    subject: String::new(),
                    relation: Relation::AssociatedWith,
                    object: obj.to_string(),
                });
            }
        }

        predicates
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_is_a() {
        let ctx = SemanticContext::new(GraphArena::with_capacity(8));
        let resolver = DefinitionResolver::new(ctx);
        let result = resolver.resolve("cat", "is a feline. has fur. can climb.", None);
        assert_eq!(result.predicates.len(), 3);
        assert_eq!(result.predicates[0].relation, Relation::IsA);
        assert_eq!(result.predicates[0].object, "feline");
        assert_eq!(result.predicates[1].relation, Relation::HasProperty);
        assert_eq!(result.predicates[1].object, "fur");
        assert_eq!(result.predicates[2].relation, Relation::Activates);
        assert_eq!(result.predicates[2].object, "climb");
    }

    #[test]
    fn dependencies_discovered() {
        let ctx = SemanticContext::new(GraphArena::with_capacity(8));
        let resolver = DefinitionResolver::new(ctx);
        let result = resolver.resolve("cat", "is an animal. has tail.", None);
        assert!(result.dependencies.contains(&"animal".to_string()));
        assert!(result.dependencies.contains(&"tail".to_string()));
    }

    #[test]
    fn circuit_breaker_not_reached() {
        let ctx = SemanticContext::new(GraphArena::with_capacity(8));
        let resolver = DefinitionResolver::new(ctx);
        let result = resolver.resolve("x", "is a y. y is a z. z is a w.", None);
        assert_eq!(result.predicates.len(), 1);
        assert_eq!(result.dependencies.len(), 1);
        assert_eq!(result.dependencies[0], "y");
    }
}
