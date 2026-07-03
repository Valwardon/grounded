use semantic_graph::prelude::*;

// ────────────────────────────────────────────────────────────
//  Combinatory Categorial Grammar (CCG) Relational Parser
//
//  Replaces the 6-grammar-rule DefinitionResolver with a
//  stateless CCG parser. Tokenizes input concepts into
//  semantic categories, reduces via combinatory rules, and
//  produces Relation edges.
//
//  Unrecognized structures degrade gracefully to proximity
//  edges rather than failing.
// ────────────────────────────────────────────────────────────

/// Semantic category in the CCG lexicon.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SemanticCategory {
    /// Entity / noun (e.g., "cat", "pirate", "ship")
    Entity,
    /// Action / verb (e.g., "walk", "give", "eat")
    Action,
    /// Property / adjective (e.g., "big", "red", "fast")
    Property,
    /// Relation / preposition (e.g., "to", "from", "with")
    Relation,
    /// Category modifier / adverb (e.g., "very", "quickly")
    Modifier,
    /// Unknown — will fallback to proximity
    Unknown,
}

/// A tokenized concept with its CCG category.
#[derive(Debug, Clone)]
pub struct TokenizedConcept {
    pub label: String,
    pub category: SemanticCategory,
}

/// A parse tree node (result of reduction).
#[derive(Debug, Clone)]
pub enum ParseNode {
    /// A single concept token
    Token(TokenizedConcept),
    /// A reduced relation: [Entity] -[Relation]-> [Entity]
    Relation {
        source_label: String,
        rel: Relation,
        target_label: String,
    },
    /// A property attribution: [Entity] has property [Property]
    Attribution {
        entity_label: String,
        property_label: String,
    },
    /// Action frame: [Entity] performs [Action] on [Entity]
    ActionFrame {
        actor: String,
        action: String,
        object: Option<String>,
    },
    /// Unresolved token cluster — fallback to proximity edges
    Cluster(Vec<String>),
}

/// The output of a CCG parse.
#[derive(Debug, Clone)]
pub struct RelationalParse {
    /// Successfully parsed relations
    pub relations: Vec<(String, Relation, String)>,
    /// Unresolved tokens that need proximity fallback
    pub unresolved: Vec<String>,
    /// Confidence (1.0 = all parsed, 0.0 = none)
    pub confidence: f64,
}

/// The CCG RelationalParser — stateless, deterministic.
pub struct RelationalParser;

impl RelationalParser {
    /// Classify a concept label into a semantic category using
    /// the existing verb table and known entity types.
    pub fn classify(label: &str) -> SemanticCategory {
        let lower = label.to_lowercase();

        // Check verb table first
        if verb_to_cd_action(&lower).is_some() {
            return SemanticCategory::Action;
        }

        // Known entity suffixes
        if lower.ends_with("er") || lower.ends_with("or")
            || lower.ends_with("man") || lower.ends_with("woman")
            || lower.ends_with("ist") || lower.ends_with("ant")
        {
            return SemanticCategory::Entity;
        }

        // Property markers
        if lower.ends_with("ful") || lower.ends_with("less")
            || lower.ends_with("ous") || lower.ends_with("able")
            || lower.ends_with("ive") || lower.ends_with("al")
            || lower.ends_with("ic") || lower.ends_with("ish")
            || lower.ends_with("like") || lower.ends_with("y")
            || lower.ends_with("ed") || lower.ends_with("ing")
        {
            return SemanticCategory::Property;
        }

        // Relation markers
        if lower == "to" || lower == "from" || lower == "with"
            || lower == "by" || lower == "at" || lower == "in"
            || lower == "on" || lower == "for" || lower == "of"
            || lower == "into" || lower == "onto" || lower == "toward"
            || lower == "through" || lower == "between" || lower == "under"
        {
            return SemanticCategory::Relation;
        }

        // Modifier markers
        if lower.ends_with("ly") || lower == "very" || lower == "quite"
            || lower == "extremely" || lower == "slightly"
            || lower == "almost" || lower == "barely"
        {
            return SemanticCategory::Modifier;
        }

        // Default: check if it's a known primitive — if so, it's an Entity
        if primitive_for(&lower).is_some() {
            return SemanticCategory::Entity;
        }

        // Default to Unknown (will get proximity fallback)
        SemanticCategory::Unknown
    }

    /// Parse a sequence of concept labels into a RelationalParse.
    ///
    /// Uses shift-reduce CCG:
    ///   - Shift: push token onto stack
    ///   - Reduce: apply combinatory rules to top of stack
    ///   - Repeat until all tokens consumed
    pub fn parse(labels: &[String]) -> RelationalParse {
        if labels.is_empty() {
            return RelationalParse {
                relations: Vec::new(),
                unresolved: Vec::new(),
                confidence: 1.0,
            };
        }

        // Tokenize
        let tokens: Vec<TokenizedConcept> = labels.iter()
            .map(|l| TokenizedConcept {
                label: l.to_lowercase(),
                category: Self::classify(l),
            })
            .collect();

        // Parse stack: Vec<ParseNode>
        let mut stack: Vec<ParseNode> = Vec::with_capacity(tokens.len());
        let total = tokens.len();

        for token in tokens {
            // Shift
            stack.push(ParseNode::Token(token));

            // Reduce loop
            Self::reduce_all(&mut stack);
        }

        // Extract relations and unresolved
        let mut relations: Vec<(String, Relation, String)> = Vec::new();
        let mut unresolved: Vec<String> = Vec::new();

        for node in &stack {
            match node {
                ParseNode::Relation { source_label, rel, target_label } => {
                    relations.push((source_label.clone(), *rel, target_label.clone()));
                }
                ParseNode::Attribution { entity_label, property_label } => {
                    relations.push((entity_label.clone(), Relation::HasProperty, property_label.clone()));
                }
                ParseNode::ActionFrame { actor, action, object } => {
                    // Map action to CD action to get relation
                    let rel = verb_to_cd_action(action)
                        .map(|cd| match cd {
                            CDAction::Atrans => Relation::Requires,
                            CDAction::Ptrans => Relation::Implies,
                            CDAction::Mtrans => Relation::Activates,
                            CDAction::Ingest => Relation::Requires,
                            _ => Relation::AssociatedWith,
                        })
                        .unwrap_or(Relation::AssociatedWith);
                    relations.push((actor.clone(), rel, action.clone()));
                    if let Some(obj) = object {
                        relations.push((action.clone(), Relation::HasProperty, obj));
                    }
                }
                ParseNode::Token(t) => {
                    if t.category == SemanticCategory::Unknown || t.category == SemanticCategory::Modifier {
                        unresolved.push(t.label.clone());
                    } else {
                        unresolved.push(t.label.clone());
                    }
                }
                ParseNode::Cluster(_) => {
                    // Fallback: spread all into unresolved for proximity
                    // (clusters should have been reduced already)
                }
            }
        }

        // Generate proximity edges between all unresolved tokens
        if unresolved.len() >= 2 {
            for i in 0..unresolved.len() - 1 {
                relations.push((
                    unresolved[i].clone(),
                    Relation::AssociatedWith,
                    unresolved[i + 1].clone(),
                ));
            }
        }

        let parsed_count = relations.len();
        let confidence = if total == 0 {
            1.0
        } else {
            (parsed_count as f64 / total as f64).min(1.0)
        };

        RelationalParse { relations, unresolved, confidence }
    }

    /// Reduce the stack as much as possible using CCG rules.
    fn reduce_all(stack: &mut Vec<ParseNode>) {
        loop {
            let len = stack.len();
            if len < 2 {
                break;
            }

            // Try each reduction rule in priority order
            let mut reduced = false;

            // Rule 1: [Entity] [Action] [Entity] → ActionFrame(actor, action, object)
            if len >= 3 {
                if let (
                    ParseNode::Token(e1),
                    ParseNode::Token(a),
                    ParseNode::Token(e2),
                ) = (&stack[len - 3], &stack[len - 2], &stack[len - 1])
                {
                    if e1.category == SemanticCategory::Entity
                        && a.category == SemanticCategory::Action
                        && (e2.category == SemanticCategory::Entity || e2.category == SemanticCategory::Unknown)
                    {
                        let actor = e1.label.clone();
                        let action = a.label.clone();
                        let object = e2.label.clone();
                        stack.truncate(len - 3);
                        stack.push(ParseNode::ActionFrame {
                            actor,
                            action: action.clone(),
                            object: Some(object),
                        });
                        reduced = true;
                        continue;
                    }
                }
            }

            // Rule 2: [Property] [Entity] → Attribution(Entity, Property)
            if len >= 2 {
                let last = &stack[len - 2];
                let last2 = &stack[len - 1];
                if let (ParseNode::Token(p), ParseNode::Token(e)) = (last, last2) {
                    if p.category == SemanticCategory::Property
                        && (e.category == SemanticCategory::Entity || e.category == SemanticCategory::Unknown)
                    {
                        let prop = p.label.clone();
                        let entity = e.label.clone();
                        stack.truncate(len - 2);
                        stack.push(ParseNode::Attribution {
                            entity_label: entity,
                            property_label: prop,
                        });
                        reduced = true;
                        continue;
                    }
                }
            }

            // Rule 3: [Entity] [Action] → ActionFrame(actor=Entity, action, object=None)
            if len >= 2 {
                let last = &stack[len - 2];
                let last2 = &stack[len - 1];
                if let (ParseNode::Token(e), ParseNode::Token(a)) = (last, last2) {
                    if e.category == SemanticCategory::Entity
                        && a.category == SemanticCategory::Action
                    {
                        let actor = e.label.clone();
                        let action = a.label.clone();
                        stack.truncate(len - 2);
                        stack.push(ParseNode::ActionFrame {
                            actor,
                            action: action.clone(),
                            object: None,
                        });
                        reduced = true;
                        continue;
                    }
                }
            }

            // Rule 4: [Entity] [Relation] [Entity] → Relation
            if len >= 3 {
                if let (
                    ParseNode::Token(e1),
                    ParseNode::Token(r),
                    ParseNode::Token(e2),
                ) = (&stack[len - 3], &stack[len - 2], &stack[len - 1])
                {
                    if (e1.category == SemanticCategory::Entity || e1.category == SemanticCategory::Unknown)
                        && r.category == SemanticCategory::Relation
                        && (e2.category == SemanticCategory::Entity || e2.category == SemanticCategory::Unknown)
                    {
                        let src = e1.label.clone();
                        let rel = Relation::AssociatedWith;
                        let tgt = e2.label.clone();
                        stack.truncate(len - 3);
                        stack.push(ParseNode::Relation {
                            source_label: src,
                            rel,
                            target_label: tgt,
                        });
                        reduced = true;
                        continue;
                    }
                }
            }

            // Rule 5: [Entity] [Unknown] → proximity (if no better rule matched)
            if len >= 2 {
                let last = &stack[len - 2];
                let last2 = &stack[len - 1];
                if let (ParseNode::Token(e), ParseNode::Token(u)) = (last, last2) {
                    if e.category == SemanticCategory::Entity
                        && u.category == SemanticCategory::Unknown
                    {
                        let src = e.label.clone();
                        let tgt = u.label.clone();
                        stack.truncate(len - 2);
                        stack.push(ParseNode::Relation {
                            source_label: src,
                            rel: Relation::AssociatedWith,
                            target_label: tgt,
                        });
                        reduced = true;
                        continue;
                    }
                }
            }

            // Rule 6: [Unknown] [Entity] → proximity (reverse direction)
            if len >= 2 {
                let last = &stack[len - 2];
                let last2 = &stack[len - 1];
                if let (ParseNode::Token(u), ParseNode::Token(e)) = (last, last2) {
                    if u.category == SemanticCategory::Unknown
                        && e.category == SemanticCategory::Entity
                    {
                        let src = u.label.clone();
                        let tgt = e.label.clone();
                        stack.truncate(len - 2);
                        stack.push(ParseNode::Relation {
                            source_label: src,
                            rel: Relation::AssociatedWith,
                            target_label: tgt,
                        });
                        reduced = true;
                        continue;
                    }
                }
            }

            // Rule 7: [ActionFrame] [Entity] → add object to action frame
            if len >= 2 {
                let second_last = stack.len() - 2;
                if let ParseNode::ActionFrame { actor, action, object } = &stack[second_last] {
                    if object.is_none() {
                        if let ParseNode::Token(e) = &stack[len - 1] {
                            if e.category == SemanticCategory::Entity || e.category == SemanticCategory::Unknown {
                                let obj = e.label.clone();
                                let a = action.clone();
                                let act = actor.clone();
                                stack.truncate(len - 1); // pop the entity
                                stack[second_last] = ParseNode::ActionFrame {
                                    actor: act,
                                    action: a,
                                    object: Some(obj),
                                };
                                reduced = true;
                                continue;
                            }
                        }
                    }
                }
            }

            if !reduced {
                break;
            }
        }
    }

    /// Utility to resolve a raw definition string into a RelationalParse.
    /// Splits on sentence boundaries, tokenizes each sentence, and parses.
    pub fn resolve_definition(token: &str, raw_definition: &str) -> RelationalParse {
        let lower = raw_definition.to_lowercase();
        let mut all_relations: Vec<(String, Relation, String)> = Vec::new();

        for sentence in lower.split(|c: char| c == '.' || c == ';' || c == '\n') {
            let sentence = sentence.trim();
            if sentence.is_empty() {
                continue;
            }

            // Build labels: [token] + [rest of sentence words]
            let mut labels: Vec<String> = vec![token.to_string()];
            labels.extend(
                sentence.split_whitespace()
                    .map(|w| w.trim_matches(|c: char| c == ',' || c == '!' || c == '?'))
                    .filter(|w| !w.is_empty())
                    .map(|w| w.to_string())
            );

            let parse = Self::parse(&labels);
            all_relations.extend(parse.relations);
        }

        RelationalParse {
            relations: all_relations,
            unresolved: Vec::new(),
            confidence: if all_relations.is_empty() { 0.0 } else { 1.0 },
        }
    }
}

// Re-use the verb_to_cd_action from parent module (imported via lib.rs)
fn verb_to_cd_action(verb: &str) -> Option<CDAction> {
    match verb.to_lowercase().as_str() {
        "give" | "transfer" | "send" | "pay" => Some(CDAction::Atrans),
        "take" | "steal" | "receive" | "buy" => Some(CDAction::Atrans),
        "go" | "move" | "walk" | "run" | "come" | "bring" => Some(CDAction::Ptrans),
        "tell" | "show" | "notify" | "inform" => Some(CDAction::Mtrans),
        "see" | "watch" | "read" | "scan" => Some(CDAction::Mtrans),
        "decide" | "conclude" | "plan" | "choose" => Some(CDAction::Mbuild),
        "think" | "consider" | "remember" => Some(CDAction::Mbuild),
        "push" | "pull" | "throw" | "hit" | "propel" => Some(CDAction::Propel),
        "eat" | "drink" | "consume" | "ingest" => Some(CDAction::Ingest),
        "exhale" | "spit" | "exhaust" => Some(CDAction::Expel),
        "wave" | "point" | "lift" | "lower" => Some(CDAction::Move),
        "hold" | "grasp" | "grip" | "grab" | "catch" => Some(CDAction::Grasp),
        "say" | "ask" | "speak" | "announce" | "reply" => Some(CDAction::Speak),
        "listen" | "hear" | "attend" | "focus" | "observe" => Some(CDAction::Attend),
        "open" | "close" | "start" | "stop" | "enable" | "disable" | "launch" | "lock" => {
            Some(CDAction::SystemAction)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_entity() {
        assert_eq!(RelationalParser::classify("pirate"), SemanticCategory::Entity);
        assert_eq!(RelationalParser::classify("cat"), SemanticCategory::Unknown);
    }

    #[test]
    fn classify_action() {
        assert_eq!(RelationalParser::classify("walk"), SemanticCategory::Action);
        assert_eq!(RelationalParser::classify("give"), SemanticCategory::Action);
        assert_eq!(RelationalParser::classify("eat"), SemanticCategory::Action);
    }

    #[test]
    fn classify_property() {
        assert_eq!(RelationalParser::classify("colorful"), SemanticCategory::Property);
        assert_eq!(RelationalParser::classify("useless"), SemanticCategory::Property);
    }

    #[test]
    fn classify_relation() {
        assert_eq!(RelationalParser::classify("to"), SemanticCategory::Relation);
        assert_eq!(RelationalParser::classify("from"), SemanticCategory::Relation);
    }

    #[test]
    fn parse_entity_action_entity() {
        let labels = vec!["pirate".into(), "buys".into(), "hat".into()];
        let result = RelationalParser::parse(&labels);
        assert!(!result.relations.is_empty(), "should produce relations");
        // "pirate buys hat" → pirate requires buys, buys has_property hat
        assert!(result.relations.iter().any(|(s, _, _)| s == "pirate"));
    }

    #[test]
    fn parse_property_entity() {
        let labels = vec!["big".into(), "ship".into()];
        let result = RelationalParser::parse(&labels);
        assert!(!result.relations.is_empty());
        assert!(result.relations.iter().any(|(s, r, _)| s == "ship" && *r == Relation::HasProperty));
    }

    #[test]
    fn parse_empty_returns_empty() {
        let result = RelationalParser::parse(&[]);
        assert!(result.relations.is_empty());
        assert!((result.confidence - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn fallback_proximity_for_unknown() {
        let labels = vec!["bogus".into(), "thing".into()];
        let result = RelationalParser::parse(&labels);
        assert!(!result.relations.is_empty(), "should have proximity fallback");
        assert!(result.relations.iter().any(|(_, r, _)| *r == Relation::AssociatedWith));
    }

    #[test]
    fn resolve_definition_creates_relations() {
        let result = RelationalParser::resolve_definition("cat", "cat is a feline. cat has fur.");
        assert!(!result.relations.is_empty());
    }
}
