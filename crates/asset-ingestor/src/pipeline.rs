use crate::component::*;
use crate::transform::*;

// ────────────────────────────────────────────────────────────
//  Asset Ingestion Pipeline
//
//  Coordinates the full pipeline from raw prompt → decomposed
//  components → transformed skeleton → render ops.
//
//  Acts as the public API for the multi-modal asset system.
// ────────────────────────────────────────────────────────────

pub struct AssetPipeline {
    extractor: ComponentExtractor,
    ctx: std::sync::Arc<semantic_graph::SemanticContext>,
}

impl AssetPipeline {
    pub fn new(ctx: std::sync::Arc<semantic_graph::SemanticContext>) -> Self {
        AssetPipeline {
            extractor: ComponentExtractor::new(ctx.clone()),
            ctx,
        }
    }

    /// Process a compound prompt through the full pipeline.
    ///
    /// Returns the decomposed prompt with all render ops ready
    /// for the wgpu procedural renderer.
    pub fn process_prompt(&self, prompt: &str) -> DecomposedPrompt {
        let mut result = self.extractor.decompose(prompt);

        // Apply structural mutations
        for component in &mut result.components {
            if let Some(mutation) = component.transform.structural_mutation {
                let transformed = TransformEngine::apply(component, mutation);
                *component = transformed;
            }
        }

        // Rebuild render ops post-transformation
        let mut new_ops = Vec::with_capacity(result.render_ops.len());
        for component in &result.components {
            new_ops.push(RenderOp {
                op_type: RenderOpType::DrawSkeleton,
                component: component.clone(),
                shader_params: ShaderParams {
                    color_palette: [0.8, 0.5, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    wireframe: false,
                    opacity: 1.0,
                    blend_mode: BlendMode::Over,
                },
            });
        }
        result.render_ops = new_ops;
        result
    }
}

/// Realize a decomposed prompt into a JSON structure that
/// the wgpu procedural renderer can consume directly.
pub fn realize_to_render_json(prompt: &DecomposedPrompt) -> String {
    let ops: Vec<serde_json::Value> = prompt
        .render_ops
        .iter()
        .map(|op| {
            serde_json::json!({
                "op": match op.op_type {
                    RenderOpType::DrawSkeleton => "draw_skeleton",
                    RenderOpType::ApplyMesh => "apply_mesh",
                    RenderOpType::OverlayTexture => "overlay_texture",
                    RenderOpType::CompositeLayer => "composite",
                    RenderOpType::Transform => "transform",
                },
                "label": op.component.label,
                "attachment": {
                    "parentSlot": op.component.attachment.parent_slot,
                    "offset": op.component.attachment.offset,
                    "rotation": op.component.attachment.rotation,
                },
                "transform": {
                    "translate": op.component.transform.translate,
                    "rotate": op.component.transform.rotate,
                    "scale": op.component.transform.scale,
                },
                "shader": {
                    "opacity": op.shader_params.opacity,
                    "wireframe": op.shader_params.wireframe,
                },
                "children": op.component.children.iter().map(|c| {
                    serde_json::json!({
                        "label": c.label,
                        "parentSlot": c.attachment.parent_slot,
                    })
                }).collect::<Vec<_>>(),
            })
        })
        .collect();

    serde_json::to_string_pretty(&serde_json::json!({
        "prompt": prompt.raw,
        "subjects": prompt.subjects,
        "predicates": prompt.predicates,
        "modifiers": prompt.modifiers,
        "render_ops": ops,
    }))
    .unwrap_or_else(|_| r#"{"error":"serialization_failed"}"#.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use semantic_graph::prelude::*;

    #[test]
    fn full_pipeline_pirate_cat() {
        let g = GraphArena::with_capacity(8);
        let ctx = std::sync::Arc::new(semantic_graph::SemanticContext::new(g));
        let pipeline = AssetPipeline::new(ctx);

        let result = pipeline.process_prompt("cat dressed as a pirate walking on two legs");

        assert_eq!(result.subjects[0], "cat");
        assert!(result.predicates.contains(&"dressed".to_string()));
        assert!(result.predicates.contains(&"walking".to_string()));

        // Verify quadruped-to-biped transform was applied
        let cat_component = &result.components[0];
        if cat_component.transform.structural_mutation.is_some() {
            let arm_count = cat_component
                .children
                .iter()
                .filter(|c| c.label.contains("arm"))
                .count();
            assert_eq!(arm_count, 2);
        }
    }

    #[test]
    fn render_json_produces_valid_output() {
        let g = GraphArena::with_capacity(8);
        let ctx = std::sync::Arc::new(semantic_graph::SemanticContext::new(g));
        let pipeline = AssetPipeline::new(ctx);

        let result = pipeline.process_prompt("pirate cat");
        let json = realize_to_render_json(&result);
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed.get("render_ops").is_some());
        assert_eq!(parsed["subjects"][0], "pirate");
    }
}
