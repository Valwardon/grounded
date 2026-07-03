use semantic_graph::prelude::*;
use semantic_parser::relational::RelationalParser;

// ────────────────────────────────────────────────────────────
//  CCG-based definition resolver
//
//  Replaces the 6-grammar-rule DefinitionResolver with a
//  stateless CCG RelationalParser. Raw definitions are parsed
//  by the RelationalParser into relation triples, which are
//  then materialized into the semantic graph.
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

/// Resolves a raw text definition into structured graph nodes
/// using the CCG RelationalParser.
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
        // Use CCG RelationalParser to extract relations
        let parse = RelationalParser::resolve_definition(token, raw_definition);
        let mut dependencies = Vec::new();
        let mut predicates = Vec::new();
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
                    edges: Vec::with_capacity(parse.relations.len()),
                    epistemic_status: EpistemicStatus::CoreConcept,
                    valence: 0.0,
                    mean_error: 0.0,
                    variance: 0.0,
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
                    .push(Edge::new(Relation::AssociatedWith, main_id));
            }
        }

        // Process each relation from the CCG parse
        for (source, rel, target) in &parse.relations {
            let object_id = match graph.lookup(&target.to_lowercase()) {
                Some(id) => id,
                None => {
                    let id = graph.insert(GroundedNode {
                        id: NodeId::ZERO,
                        label: target.to_lowercase(),
                        node_type: NodeType::Concept,
                        grounding: Grounding::Abstract,
                        decay: 0.9,
                        threshold: 10.0,
                        base_activation: 0.0,
                        edges: Vec::new(),
                        valence: 0.0,
                        mean_error: 0.0,
                        variance: 0.0,
                    });
                    if !dependencies.contains(&target.to_lowercase()) {
                        dependencies.push(target.to_lowercase());
                    }
                    id
                }
            };

            // Add edge from source (or main node) to target
            let source_id = if source.to_lowercase() == token.to_lowercase() {
                main_id
            } else {
                match graph.lookup(&source.to_lowercase()) {
                    Some(id) => id,
                    None => {
                        let id = graph.insert(GroundedNode {
                            id: NodeId::ZERO,
                            label: source.to_lowercase(),
                            node_type: NodeType::Concept,
                            grounding: Grounding::Abstract,
                            decay: 0.9,
                            threshold: 10.0,
                            base_activation: 0.0,
                            edges: Vec::new(),
                            valence: 0.0,
                            mean_error: 0.0,
                            variance: 0.0,
                        });
                        if !dependencies.contains(&source.to_lowercase()) {
                            dependencies.push(source.to_lowercase());
                        }
                        id
                    }
                }
            };

            if let Some(node) = graph.get(source_id) {
                node.write().edges.push(Edge::new(*rel, object_id));
            }

            predicates.push(Predicate {
                subject: source.clone(),
                relation: *rel,
                object: target.clone(),
            });
        }

        ResolvedDefinition {
            main_node_id: main_id,
            predicates,
            dependencies,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_simple_definition() {
        let ctx = SemanticContext::new(GraphArena::with_capacity(8));
        let resolver = DefinitionResolver::new(ctx);
        let result = resolver.resolve("cat", "cat is a feline. cat has fur. cat can climb.", None);
        assert!(!result.predicates.is_empty(), "should produce predicates");
        assert!(result.dependencies.contains(&"feline".to_string()));
    }

    #[test]
    fn dependencies_discovered() {
        let ctx = SemanticContext::new(GraphArena::with_capacity(8));
        let resolver = DefinitionResolver::new(ctx);
        let result = resolver.resolve("cat", "cat is an animal. cat has tail.", None);
        assert!(result.dependencies.contains(&"animal".to_string()));
    }
}
