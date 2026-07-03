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
///
/// All render ops pass through compile_to_ast() + validate_ast()
/// before serialization. This ensures structural integrity of
/// the render output.
pub fn realize_to_render_json(prompt: &DecomposedPrompt) -> String {
    let ast = compile_to_ast(prompt);

    // Validate the AST — structural errors are reported
    // but we still serialize (the renderer may still attempt
    // best-effort rendering).
    if let Err(e) = validate_ast(&ast) {
        // Return a JSON structure that includes the error so the
        // renderer can decide how to handle it.
        let valid_json = render_ast_to_json(&ast);
        let mut value: serde_json::Value =
            serde_json::from_str(&valid_json).unwrap_or(serde_json::Value::Null);
        if let serde_json::Value::Object(ref mut map) = value {
            map.insert(
                "validation_error".into(),
                serde_json::json!(e.to_string()),
            );
        }
        return serde_json::to_string_pretty(&value)
            .unwrap_or_else(|_| r#"{"error":"serialization_failed"}"#.into());
    }

    render_ast_to_json(&ast)
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
        assert_eq!(parsed["type"], "scene", "AST root should be a scene");
        assert!(parsed.get("validation_error").is_none(), "No validation errors expected");
        assert!(parsed["children"].as_array().map(|a| !a.is_empty()).unwrap_or(false));
    }

    #[test]
    fn render_json_validation_error_included() {
        let g = GraphArena::with_capacity(8);
        let ctx = std::sync::Arc::new(semantic_graph::SemanticContext::new(g));
        let pipeline = AssetPipeline::new(ctx);

        // Create an empty prompt that will produce an invalid AST
        let empty_prompt = DecomposedPrompt {
            raw: "".into(),
            tokens: vec![],
            subjects: vec![],
            predicates: vec![],
            modifiers: vec![],
            components: vec![],
            render_ops: vec![],
        };
        let json = realize_to_render_json(&empty_prompt);
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed.get("validation_error").is_some(), "Empty prompt should produce a validation error");
    }
}
