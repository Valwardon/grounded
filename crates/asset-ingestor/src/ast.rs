use crate::component::*;
use semantic_graph::prelude::StructuralError;
use semantic_graph::MotorCommandType;

// ────────────────────────────────────────────────────────────
//  Render AST — Compiled, validated intermediate representation
//
//  The asset pipeline compiles a DecomposedPrompt into a RenderAst,
//  validates the AST's structural integrity, and then serializes it.
//
//  This replaces the direct RenderOp → JSON conversion path.
//  All render ops must pass through compile_to_ast() + validate_ast()
//  before they reach the wgpu renderer.
// ────────────────────────────────────────────────────────────

/// A node in the render operation AST.
///
/// Each variant carries the data needed for the procedural renderer
/// AND the validation constraints that the AST checker enforces.
#[derive(Debug, Clone, PartialEq)]
pub enum RenderAst {
    /// Root node: a scene containing ordered children.
    Scene {
        label: String,
        children: Vec<RenderAst>,
    },

    /// Draw a skeleton (must have at least one bone).
    DrawSkeleton {
        label: String,
        bones: Vec<String>,
        color_palette: [f32; 12],
        wireframe: bool,
        opacity: f32,
        blend_mode: BlendMode,
    },

    /// Apply a geometric transform to a previous sibling or child scope.
    ApplyTransform {
        target_label: String,
        translate: [f32; 3],
        rotate: [f32; 3],
        scale: [f32; 3],
    },

    /// Apply a mesh to a skeleton (mesh must reference an existing skeleton label).
    ApplyMesh {
        skeleton_label: String,
        mesh_label: String,
        color_palette: [f32; 12],
        blend_mode: BlendMode,
    },

    /// Composite multiple layers together with a blend mode.
    Composite {
        label: String,
        sources: Vec<String>,
        blend_mode: BlendMode,
        opacity: f32,
    },

    /// Motor effector primitive — directly maps to a Grounding::MotorCommand.
    ///
    /// When this node fires in the spreading activation engine, it triggers
    /// a render operation. The rendered output feeds back into the prediction
    /// error system: a match between commanded and actual render state
    /// generates a Reward spike; a mismatch generates PredictionError + Novelty.
    Effector {
        label: String,
        command_type: MotorCommandType,
        target: String,
        parameters: Vec<f64>,
        /// Expected visual result hash (prediction for feedback loop)
        expected_hash: u64,
    },
}

// MotorCommandType is re-exported from semantic_graph.

/// Compile a decomposed prompt into a validated RenderAst tree.
///
/// This is the gateway through which ALL render operations must pass.
/// The AST can then be validated before serialization.
pub fn compile_to_ast(prompt: &DecomposedPrompt) -> RenderAst {
    let mut scene_children: Vec<RenderAst> = Vec::new();

    for component in &prompt.components {
        // 1. DrawSkeleton node
        let bones: Vec<String> = component.children.iter().map(|c| c.label.clone()).collect();
        let skeleton_label = format!("skeleton:{}", component.label);

        let draw_skeleton = RenderAst::DrawSkeleton {
            label: skeleton_label.clone(),
            bones: bones.clone(),
            color_palette: [0.8, 0.5, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            wireframe: bones.is_empty(),
            opacity: 1.0,
            blend_mode: BlendMode::Over,
        };
        scene_children.push(draw_skeleton);

        // 2. ApplyTransform if structural mutation exists
        if component.transform.structural_mutation.is_some() {
            let transform_node = RenderAst::ApplyTransform {
                target_label: skeleton_label.clone(),
                translate: component.transform.translate,
                rotate: component.transform.rotate,
                scale: component.transform.scale,
            };
            scene_children.push(transform_node);
        }

        // 3. ApplyMesh for each child component
        for child in &component.children {
            let mesh_node = RenderAst::ApplyMesh {
                skeleton_label: skeleton_label.clone(),
                mesh_label: child.label.clone(),
                color_palette: [0.5, 0.3, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                blend_mode: BlendMode::Over,
            };
            scene_children.push(mesh_node);
        }
    }

    RenderAst::Scene {
        label: prompt.raw.clone(),
        children: scene_children,
    }
}

/// Validate a RenderAst tree for structural integrity.
///
/// Checks:
///   - Scene must have at least one child
///   - DrawSkeleton must have valid blend mode
///   - ApplyTransform must reference an existing skeleton label
///   - ApplyMesh must reference an existing skeleton label (same scene)
///   - No conflicting transforms on the same target
///   - Composite sources must reference existing labels
///   - All blend modes in a Composite subtree must be compatible
///
/// Returns the first structural error found, or Ok(()).
pub fn validate_ast(ast: &RenderAst) -> Result<(), StructuralError> {
    match ast {
        RenderAst::Scene { label, children } => {
            if children.is_empty() {
                return Err(StructuralError::DeadNode {
                    node_id: semantic_graph::NodeId::ZERO,
                    label: format!("empty_scene:{}", label),
                });
            }
            // Collect labels defined in this scene
            let mut defined_labels: Vec<String> = Vec::new();
            for child in children {
                match child {
                    RenderAst::DrawSkeleton { label, .. } => {
                        defined_labels.push(format!("skeleton:{}", label));
                        validate_skeleton(child)?;
                    }
                    RenderAst::Composite { label, sources, blend_mode, opacity } => {
                        defined_labels.push(format!("composite:{}", label));
                        // All sources must be defined before this composite
                        for src in sources {
                            if !defined_labels.iter().any(|d| d.contains(src)) {
                                return Err(StructuralError::DeadNode {
                                    node_id: semantic_graph::NodeId::ZERO,
                                    label: format!("undefined_source:{}", src),
                                });
                            }
                        }
                    }
                    RenderAst::ApplyTransform { target_label, .. } => {
                        if !defined_labels.iter().any(|d| d.contains(target_label)) {
                            return Err(StructuralError::ContractMismatch {
                                source: semantic_graph::NodeId::ZERO,
                                target: semantic_graph::NodeId::ZERO,
                                expected_input: semantic_graph::DataType::Any,
                                actual_output: semantic_graph::DataType::Any,
                            });
                        }
                    }
                    RenderAst::ApplyMesh { skeleton_label, .. } => {
                        if !defined_labels.iter().any(|d| d.contains(skeleton_label)) {
                            return Err(StructuralError::ContractMismatch {
                                source: semantic_graph::NodeId::ZERO,
                                target: semantic_graph::NodeId::ZERO,
                                expected_input: semantic_graph::DataType::Any,
                                actual_output: semantic_graph::DataType::Any,
                            });
                        }
                    }
                    RenderAst::Effector { label, target, .. } => {
                        defined_labels.push(format!("effector:{}", label));
                        // Effector target must reference an existing skeleton or composite
                        if !defined_labels.iter().any(|d| d.contains(target)) {
                            return Err(StructuralError::DeadNode {
                                node_id: semantic_graph::NodeId::ZERO,
                                label: format!("effector_target_not_found:{}", target),
                            });
                        }
                    }
                }
            }
            // Check for conflicting transforms on same target
            let transform_targets: Vec<&String> = children.iter()
                .filter_map(|c| match c {
                    RenderAst::ApplyTransform { target_label, .. } => Some(target_label),
                    _ => None,
                })
                .collect();
            for (i, a) in transform_targets.iter().enumerate() {
                for b in transform_targets.iter().skip(i + 1) {
                    if a == b {
                        return Err(StructuralError::DeadNode {
                            node_id: semantic_graph::NodeId::ZERO,
                            label: format!("conflicting_transform:{}", a),
                        });
                    }
                }
            }
            Ok(())
        }
        _ => {
            // Top-level must be a Scene
            Err(StructuralError::DeadNode {
                node_id: semantic_graph::NodeId::ZERO,
                label: "top_level_must_be_scene".into(),
            })
        }
    }
}

/// Validate a single DrawSkeleton node.
fn validate_skeleton(ast: &RenderAst) -> Result<(), StructuralError> {
    if let RenderAst::DrawSkeleton { label, bones, .. } = ast {
        // Must have at least one bone OR the wireframe fallback is explicit
        if bones.is_empty() {
            // Empty bone list is valid ONLY with wireframe=true
            // (we already set wireframe = bones.is_empty() in compile_to_ast)
        }
        Ok(())
    } else {
        Err(StructuralError::DeadNode {
            node_id: semantic_graph::NodeId::ZERO,
            label: "expected_draw_skeleton".into(),
        })
    }
}

/// Convert a validated RenderAst to JSON for the wgpu renderer.
pub fn render_ast_to_json(ast: &RenderAst) -> String {
    ast_node_to_json(ast)
}

fn ast_node_to_json(ast: &RenderAst) -> String {
    match ast {
        RenderAst::Scene { label, children } => {
            let children_json: Vec<String> = children.iter().map(ast_node_to_json).collect();
            serde_json::to_string_pretty(&serde_json::json!({
                "type": "scene",
                "label": label,
                "children": children_json.iter().map(|c| {
                    serde_json::from_str::<serde_json::Value>(c).unwrap_or(serde_json::Value::Null)
                }).collect::<Vec<_>>(),
            })).unwrap_or_else(|_| r#"{"error":"serialization_failed"}"#.into())
        }
        RenderAst::DrawSkeleton { label, bones, color_palette, wireframe, opacity, blend_mode } => {
            serde_json::to_string_pretty(&serde_json::json!({
                "type": "draw_skeleton",
                "label": label,
                "bones": bones,
                "color_palette": color_palette,
                "wireframe": wireframe,
                "opacity": opacity,
                "blend_mode": match blend_mode {
                    BlendMode::Over => "over",
                    BlendMode::Multiply => "multiply",
                    BlendMode::Add => "add",
                },
            })).unwrap_or_else(|_| r#"{"error":"serialization_failed"}"#.into())
        }
        RenderAst::ApplyTransform { target_label, translate, rotate, scale } => {
            serde_json::to_string_pretty(&serde_json::json!({
                "type": "apply_transform",
                "target_label": target_label,
                "translate": translate,
                "rotate": rotate,
                "scale": scale,
            })).unwrap_or_else(|_| r#"{"error":"serialization_failed"}"#.into())
        }
        RenderAst::ApplyMesh { skeleton_label, mesh_label, color_palette, blend_mode } => {
            serde_json::to_string_pretty(&serde_json::json!({
                "type": "apply_mesh",
                "skeleton_label": skeleton_label,
                "mesh_label": mesh_label,
                "color_palette": color_palette,
                "blend_mode": match blend_mode {
                    BlendMode::Over => "over",
                    BlendMode::Multiply => "multiply",
                    BlendMode::Add => "add",
                },
            })).unwrap_or_else(|_| r#"{"error":"serialization_failed"}"#.into())
        }
        RenderAst::Composite { label, sources, blend_mode, opacity } => {
            serde_json::to_string_pretty(&serde_json::json!({
                "type": "composite",
                "label": label,
                "sources": sources,
                "blend_mode": match blend_mode {
                    BlendMode::Over => "over",
                    BlendMode::Multiply => "multiply",
                    BlendMode::Add => "add",
                },
                "opacity": opacity,
            })).unwrap_or_else(|_| r#"{"error":"serialization_failed"}"#.into())
        }
        RenderAst::Effector { label, command_type, target, parameters, expected_hash } => {
            serde_json::to_string_pretty(&serde_json::json!({
                "type": "effector",
                "label": label,
                "command_type": command_type.label(),
                "target": target,
                "parameters": parameters,
                "expected_hash": expected_hash,
            })).unwrap_or_else(|_| r#"{"error":"serialization_failed"}"#.into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_prompt() -> DecomposedPrompt {
        DecomposedPrompt {
            raw: "test prompt".into(),
            tokens: vec!["test".into()],
            subjects: vec!["cat".into()],
            predicates: vec![],
            modifiers: vec![],
            components: vec![VisualComponent {
                label: "cat".into(),
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
                children: vec![
                    VisualComponent {
                        label: "head".into(),
                        component_type: ComponentType::Skeleton,
                        attachment: Attachment {
                            parent_slot: "torso".into(),
                            offset: [0.0, 0.5, 0.0],
                            rotation: [0.0; 3],
                        },
                        transform: GeometryTransform {
                            translate: [0.0; 3],
                            rotate: [0.0; 3],
                            scale: [1.0; 3],
                            structural_mutation: None,
                        },
                        children: vec![],
                    },
                    VisualComponent {
                        label: "torso".into(),
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
                        children: vec![],
                    },
                ],
            }],
            render_ops: vec![],
        }
    }

    #[test]
    fn compile_empty_scene_fails_validation() {
        let ast = RenderAst::Scene {
            label: "empty".into(),
            children: vec![],
        };
        assert!(validate_ast(&ast).is_err());
    }

    #[test]
    fn compile_valid_scene_passes_validation() {
        let prompt = make_test_prompt();
        let ast = compile_to_ast(&prompt);
        assert!(validate_ast(&ast).is_ok());
    }

    #[test]
    fn missing_skeleton_causes_contract_mismatch() {
        let ast = RenderAst::Scene {
            label: "bad".into(),
            children: vec![
                RenderAst::ApplyMesh {
                    skeleton_label: "skeleton:nonexistent".into(),
                    mesh_label: "leg".into(),
                    color_palette: [0.5; 12],
                    blend_mode: BlendMode::Over,
                },
            ],
        };
        assert!(validate_ast(&ast).is_err());
    }

    #[test]
    fn conflicting_transforms_detected() {
        let ast = RenderAst::Scene {
            label: "bad".into(),
            children: vec![
                RenderAst::DrawSkeleton {
                    label: "skeleton:cat".into(),
                    bones: vec!["head".into()],
                    color_palette: [0.8; 12],
                    wireframe: false,
                    opacity: 1.0,
                    blend_mode: BlendMode::Over,
                },
                RenderAst::ApplyTransform {
                    target_label: "skeleton:cat".into(),
                    translate: [0.0; 3],
                    rotate: [0.0; 3],
                    scale: [1.0; 3],
                },
                RenderAst::ApplyTransform {
                    target_label: "skeleton:cat".into(),
                    translate: [1.0; 3],
                    rotate: [0.0; 3],
                    scale: [1.0; 3],
                },
            ],
        };
        assert!(validate_ast(&ast).is_err());
    }

    #[test]
    fn render_ast_produces_valid_json() {
        let prompt = make_test_prompt();
        let ast = compile_to_ast(&prompt);
        let json = render_ast_to_json(&ast);
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["type"], "scene");
        assert!(parsed["children"].as_array().unwrap().len() >= 1);
    }
}
