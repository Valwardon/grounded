use semantic_graph::prelude::*;

// ────────────────────────────────────────────────────────────
//  Multi-Modal Asset Decomposition
//
//  This is the operational tree builder for compound prompts.
//  Given "cat dressed as a pirate walking on two legs":
//
//  1. Lexical pass → token sequence
//  2. Role assignment → map tokens to CD frame slots
//  3. Asset extraction → identify visual components needed
//  4. Transform cascade → determine geometric transforms
//  5. Operational tree → ordered pipeline of rendering ops
//
//  No embeddings. No LLM. Pure structural grammar matching.
// ────────────────────────────────────────────────────────────

/// A single extracted visual component that needs to be rendered.
#[derive(Debug, Clone)]
pub struct VisualComponent {
    pub label: String,
    pub component_type: ComponentType,
    /// Spatial relationship to parent
    pub attachment: Attachment,
    /// Geometric transform to apply
    pub transform: GeometryTransform,
    /// Child components (recursive)
    pub children: Vec<VisualComponent>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComponentType {
    Skeleton,
    Mesh,
    Texture,
    Attachment,
    Decoration,
}

/// How a component attaches to its parent.
#[derive(Debug, Clone)]
pub struct Attachment {
    pub parent_slot: String,   // "head", "torso", "left_arm", "spine"
    pub offset: [f32; 3],      // x, y, z offset
    pub rotation: [f32; 3],    // pitch, yaw, roll
}

/// A geometric transform that changes a component's structure.
#[derive(Debug, Clone)]
pub struct GeometryTransform {
    pub translate: [f32; 3],
    pub rotate: [f32; 3],
    pub scale: [f32; 3],
    /// If set, this is a structural change (e.g., quadruped → biped)
    pub structural_mutation: Option<StructuralMutation>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructuralMutation {
    QuadrupedToBiped,
    MirrorHorizontal,
    RemoveComponent,
    AddComponent,
}

/// A single node in the operational rendering tree.
#[derive(Debug, Clone)]
pub struct RenderOp {
    pub op_type: RenderOpType,
    pub component: VisualComponent,
    pub shader_params: ShaderParams,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderOpType {
    DrawSkeleton,
    ApplyMesh,
    OverlayTexture,
    CompositeLayer,
    Transform,
}

#[derive(Debug, Clone)]
pub struct ShaderParams {
    pub color_palette: [f32; 12],
    pub wireframe: bool,
    pub opacity: f32,
    pub blend_mode: BlendMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMode {
    Over,
    Multiply,
    Add,
}

/// The complete prompt decomposition result.
#[derive(Debug, Clone)]
pub struct DecomposedPrompt {
    /// Original input text
    pub raw: String,
    /// Tokens extracted from lexical pass
    pub tokens: Vec<String>,
    /// Core subjects (actors)
    pub subjects: Vec<String>,
    /// Actions or states
    pub predicates: Vec<String>,
    /// Modifiers (adjectives, descriptors)
    pub modifiers: Vec<String>,
    /// Extracted visual components with transformations
    pub components: Vec<VisualComponent>,
    /// Ordered rendering operations
    pub render_ops: Vec<RenderOp>,
}

// ────────────────────────────────────────────────────────────
//  Component extraction logic
// ────────────────────────────────────────────────────────────

pub const BASE_SKELETONS: &[(&str, &[&str])] = &[
    (
        "human",
        &["head", "torso", "left_arm", "right_arm", "left_leg", "right_leg"],
    ),
    (
        "quadruped",
        &["head", "torso", "front_left_leg", "front_right_leg", "back_left_leg", "back_right_leg", "tail"],
    ),
    (
        "bird",
        &["head", "torso", "left_wing", "right_wing", "left_leg", "right_leg", "tail"],
    ),
    ("fish", &["head", "torso", "dorsal_fin", "tail_fin", "left_pectoral", "right_pectoral"]),
];

pub struct ComponentExtractor {
    ctx: Arc<SemanticContext>,
}

impl ComponentExtractor {
    pub fn new(ctx: Arc<SemanticContext>) -> Self {
        ComponentExtractor { ctx }
    }

    /// Decompose a compound prompt into an operational tree.
    pub fn decompose(&self, prompt: &str) -> DecomposedPrompt {
        let raw = prompt.to_string();
        let tokens = self.lexical_analysis(prompt);

        let (subjects, predicates, modifiers) = self.role_assignment(&tokens);

        let components = self.extract_components(&subjects, &predicates, &modifiers);

        let render_ops = self.build_render_ops(&components);

        DecomposedPrompt {
            raw,
            tokens,
            subjects,
            predicates,
            modifiers,
            components,
            render_ops,
        }
    }

    /// Tokenize and tag parts of speech using a deterministic grammar.
    fn lexical_analysis(&self, prompt: &str) -> Vec<String> {
        prompt
            .split(|c: char| c.is_whitespace() || c == ',')
            .map(|s| s.trim().to_lowercase())
            .filter(|s| !s.is_empty())
            .collect()
    }

    /// Assign roles based on position and known word classes.
    ///
    /// Pattern: "[subject] [verb] [modifier] [object] [preposition] [modifier] [object]"
    ///   "cat dressed as a pirate on two legs"
    ///   → subjects: ["cat"]
    ///   → predicates: ["dressed", "walking"]
    ///   → modifiers: ["two"]
    fn role_assignment(&self, tokens: &[String]) -> (Vec<String>, Vec<String>, Vec<String>) {
        let mut subjects = Vec::new();
        let mut predicates = Vec::new();
        let mut modifiers = Vec::new();

        // Known verbs (deterministic lookup)
        let verbs = [
            "dressed", "walking", "running", "sitting", "standing",
            "wearing", "holding", "carrying", "riding", "flying",
            "swimming", "eating", "drinking", "reading", "writing",
        ];

        // Known prepositions (signal role boundaries)
        let prepositions = [
            "as", "on", "in", "at", "with", "by", "for", "of",
            "to", "from", "under", "over", "through", "beside",
        ];

        let mut i = 0;
        while i < tokens.len() {
            let token = &tokens[i];

            if verbs.contains(&token.as_str()) {
                predicates.push(token.clone());
            } else if prepositions.contains(&token.as_str()) {
                // preposition — skip, next token may be a modifier or object
            } else if subjects.is_empty() && predicates.is_empty() {
                // First noun-like token → subject
                subjects.push(token.clone());
            } else if !predicates.is_empty() {
                // After a verb, check if this is a number (modifier)
                if token.parse::<f64>().is_ok() {
                    modifiers.push(token.clone());
                } else if i + 1 < tokens.len()
                    && !verbs.contains(&tokens[i + 1].as_str())
                    && !prepositions.contains(&tokens[i + 1].as_str())
                {
                    // Current token modifies the next one
                    modifiers.push(token.clone());
                } else {
                    // Or it's a standalone object/subject
                    subjects.push(token.clone());
                }
            }

            i += 1;
        }

        (subjects, predicates, modifiers)
    }

    /// Extract visual components by looking up each subject in
    /// the semantic graph and building a skeleton tree.
    fn extract_components(
        &self,
        subjects: &[String],
        predicates: &[String],
        modifiers: &[String],
    ) -> Vec<VisualComponent> {
        let mut components = Vec::new();
        let graph = self.ctx.graph.read();

        for subject in subjects {
            let skeleton = self.build_skeleton(subject, &graph);

            // Apply predicate-driven transformations
            let transformed = self.apply_predicate_transforms(skeleton, predicates, modifiers);

            components.push(transformed);
        }

        components
    }

    /// Build a skeleton tree for a subject.
    fn build_skeleton(&self, label: &str, graph: &GraphArena) -> VisualComponent {
        let node_id = graph.lookup(&label.to_lowercase());
        let guessed_skeleton = self.guess_skeleton_type(label);

        let mut children = Vec::new();
        for part in guessed_skeleton {
            children.push(VisualComponent {
                label: part.to_string(),
                component_type: ComponentType::Skeleton,
                attachment: Attachment {
                    parent_slot: "torso".into(),
                    offset: [0.0; 3],
                    rotation: [0.0; 3],
                },
                transform: GeometryTransform {
                    translate: [0.0; 3],
                    rotate: [0.0; 3],
                    scale: [1.0; 3],
                    structural_mutation: None,
                },
                children: Vec::new(),
            });
        }

        VisualComponent {
            label: label.to_string(),
            component_type: ComponentType::Skeleton,
            attachment: Attachment {
                parent_slot: "root".into(),
                offset: [0.0; 3],
                rotation: [0.0; 3],
            },
            transform: GeometryTransform {
                translate: [0.0; 3],
                rotate: [0.0; 3],
                scale: [1.0; 3],
                structural_mutation: None,
            },
            children,
        }
    }

    /// Guess the skeleton type from the label using keyword matching.
    fn guess_skeleton_type(&self, label: &str) -> &[&str] {
        let lower = label.to_lowercase();
        if lower.contains("cat") || lower.contains("dog") || lower.contains("horse")
            || lower.contains("cow") || lower.contains("lion") || lower.contains("tiger")
        {
            BASE_SKELETONS[1].1 // quadruped
        } else if lower.contains("bird") || lower.contains("eagle") || lower.contains("crow") {
            BASE_SKELETONS[2].1 // bird
        } else if lower.contains("fish") || lower.contains("shark") {
            BASE_SKELETONS[3].1 // fish
        } else {
            BASE_SKELETONS[0].1 // human (default biped)
        }
    }

    /// Apply predicate-driven transformations to a skeleton.
    fn apply_predicate_transforms(
        &self,
        mut component: VisualComponent,
        predicates: &[String],
        modifiers: &[String],
    ) -> VisualComponent {
        for pred in predicates {
            match pred.as_str() {
                "walking" | "running" | "standing" if component.label != "human" => {
                    // Non-human walking → requires bipedal transformation
                    component.transform.structural_mutation =
                        Some(StructuralMutation::QuadrupedToBiped);

                    // Add leg animation params
                    for child in &mut component.children {
                        if child.label.contains("leg") {
                            child.component_type = ComponentType::Attachment;
                        }
                    }
                }
                "dressed" | "wearing" => {
                    // Clothing attachment — the next predicate/modifier
                    // specifies the clothing type (handled by caller)
                }
                _ => {}
            }
        }

        component
    }

    /// Build the ordered rendering operation tree.
    fn build_render_ops(&self, components: &[VisualComponent]) -> Vec<RenderOp> {
        let mut ops = Vec::new();

        for component in components {
            // 1. Draw skeleton
            ops.push(RenderOp {
                op_type: RenderOpType::DrawSkeleton,
                component: component.clone(),
                shader_params: ShaderParams {
                    color_palette: [0.8, 0.5, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    wireframe: true,
                    opacity: 1.0,
                    blend_mode: BlendMode::Over,
                },
            });

            // 2. Apply any structural mutations
            if component.transform.structural_mutation.is_some() {
                ops.push(RenderOp {
                    op_type: RenderOpType::Transform,
                    component: component.clone(),
                    shader_params: ShaderParams {
                        color_palette: [0.0; 12],
                        wireframe: false,
                        opacity: 1.0,
                        blend_mode: BlendMode::Over,
                    },
                });
            }

            // 3. Apply meshes/textures for child components
            for child in &component.children {
                ops.push(RenderOp {
                    op_type: RenderOpType::ApplyMesh,
                    component: child.clone(),
                    shader_params: ShaderParams {
                        color_palette: [0.5, 0.3, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        wireframe: false,
                        opacity: 1.0,
                        blend_mode: BlendMode::Over,
                    },
                });
            }
        }

        ops
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pirate_cat_decomposition() {
        let g = GraphArena::with_capacity(8);
        let ctx = Arc::new(SemanticContext::new(g));
        let extractor = ComponentExtractor::new(ctx);

        let prompt = "cat dressed as a pirate walking on two legs";
        let result = extractor.decompose(prompt);

        assert!(result.subjects.contains(&"cat".to_string()));
        assert!(result.predicates.contains(&"dressed".to_string()));
        assert!(result.predicates.contains(&"walking".to_string()));
        assert!(result.modifiers.contains(&"two".to_string()));
    }

    #[test]
    fn quadruped_to_biped_transform() {
        let g = GraphArena::with_capacity(8);
        let ctx = Arc::new(SemanticContext::new(g));
        let extractor = ComponentExtractor::new(ctx);

        let result = extractor.decompose("cat walking on two legs");
        assert!(result.subjects.contains(&"cat".to_string()));

        let cat = &result.components[0];
        assert_eq!(
            cat.transform.structural_mutation,
            Some(StructuralMutation::QuadrupedToBiped)
        );
    }

    #[test]
    fn render_ops_generated() {
        let g = GraphArena::with_capacity(8);
        let ctx = Arc::new(SemanticContext::new(g));
        let extractor = ComponentExtractor::new(ctx);

        let result = extractor.decompose("cat");
        assert!(!result.render_ops.is_empty());

        let draw_count = result
            .render_ops
            .iter()
            .filter(|op| op.op_type == RenderOpType::DrawSkeleton)
            .count();
        assert_eq!(draw_count, 1);
    }
}
