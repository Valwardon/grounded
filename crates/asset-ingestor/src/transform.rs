use crate::component::*;

// ────────────────────────────────────────────────────────────
//  Geometric Transform Engine
//
//  Applies deterministic mathematical transformations to
//  skeleton coordinates for structural mutations like
//  quadruped → bipedal.
//
//  The quadruped skeleton has 4 legs planted horizontally.
//  The bipedal skeleton has 2 legs planted vertically.
//  The transformation:
//    1. Rear front legs → arms (rotate 90° up, attach to torso)
//    2. Rear back legs → legs (rotate -90° down, attach to pelvis)
//    3. Rotate spine 90° to vertical
//    4. Adjust tail to hang down
//
//  All transforms are pure matrix math — no learning.
// ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SkeletonTransform {
    /// Per-joint transformation matrix as [rotation, translation, scale]
    pub joint_transforms: Vec<JointTransform>,
}

#[derive(Debug, Clone)]
pub struct JointTransform {
    pub joint_name: String,
    pub rotate: [f32; 3],
    pub translate: [f32; 3],
    pub scale: [f32; 3],
}

/// Applies a structural mutation to a VisualComponent tree.
pub struct TransformEngine;

impl TransformEngine {
    /// Apply a structural mutation to a component tree, returning
    /// a new transformed tree.
    pub fn apply(
        component: &VisualComponent,
        mutation: StructuralMutation,
    ) -> VisualComponent {
        match mutation {
            StructuralMutation::QuadrupedToBiped => Self::quadruped_to_biped(component),
            StructuralMutation::MirrorHorizontal => Self::mirror_horizontal(component),
            _ => component.clone(),
        }
    }

    /// Quadruped → biped transformation.
    ///
    /// Joint mapping:
    ///   Quadruped          → Biped
    ///   ─────────             ─────
    ///   front_left_leg     → left_arm
    ///   front_right_leg    → right_arm
    ///   back_left_leg      → left_leg
    ///   back_right_leg     → right_leg
    ///   spine              → vertical (rotate 90°)
    ///   head               → head (elevate)
    ///   tail               → tail (rotate down)
    fn quadruped_to_biped(component: &VisualComponent) -> VisualComponent {
        let mut transformed = component.clone();

        // Mark the transform
        let mut current_transform = GeometryTransform {
            translate: [0.0, 0.0, 0.0],
            rotate: [0.0, -90.0, 0.0], // rotate entire skeleton 90° on Z
            scale: [1.0, 1.2, 1.0],    // stretch vertically
            structural_mutation: None,
        };

        // Transform each child joint
        for child in &mut transformed.children {
            match child.label.as_str() {
                "front_left_leg" | "front_right_leg" => {
                    // Front legs become arms: rotate up 90°, attach to upper torso
                    child.label = if child.label == "front_left_leg" {
                        "left_arm".into()
                    } else {
                        "right_arm".into()
                    };
                    child.attachment.parent_slot = "upper_torso".into();
                    child.transform = GeometryTransform {
                        translate: [0.0, 0.5, 0.0], // lift to shoulder height
                        rotate: [0.0, 0.0, 90.0],  // rotate to point down
                        scale: [0.8, 0.8, 0.8],    // arms shorter than legs
                        structural_mutation: None,
                    };
                }
                "back_left_leg" | "back_right_leg" => {
                    // Back legs remain legs, rotate to point down
                    child.label = if child.label == "back_left_leg" {
                        "left_leg".into()
                    } else {
                        "right_leg".into()
                    };
                    child.attachment.parent_slot = "pelvis".into();
                    child.transform = GeometryTransform {
                        translate: [0.0, -0.5, 0.0],
                        rotate: [0.0, 0.0, 0.0],
                        scale: [1.2, 1.2, 1.2], // legs slightly longer
                        structural_mutation: None,
                    };
                }
                "tail" => {
                    // Tail rotates to hang down
                    child.transform = GeometryTransform {
                        translate: [0.0, -0.2, 0.0],
                        rotate: [0.0, 0.0, -45.0],
                        scale: [1.0, 1.0, 1.0],
                        structural_mutation: None,
                    };
                }
                "head" => {
                    // Head elevated
                    child.transform = GeometryTransform {
                        translate: [0.0, 0.8, 0.0],
                        rotate: [0.0, 0.0, 0.0],
                        scale: [1.0, 1.0, 1.0],
                        structural_mutation: None,
                    };
                }
                _ => {}
            }
        }

        transformed.transform = current_transform;
        transformed
    }

    /// Mirror a component horizontally.
    fn mirror_horizontal(component: &VisualComponent) -> VisualComponent {
        let mut mirrored = component.clone();
        mirrored.transform.rotate[1] += 180.0; // flip on Y axis
        mirrored
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_quadruped() -> VisualComponent {
        VisualComponent {
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
                    label: "front_left_leg".into(),
                    component_type: ComponentType::Skeleton,
                    attachment: Attachment { parent_slot: "torso".into(), offset: [0.0; 3], rotation: [0.0; 3] },
                    transform: GeometryTransform { translate: [0.0; 3], rotate: [0.0; 3], scale: [1.0; 3], structural_mutation: None },
                    children: Vec::new(),
                },
                VisualComponent {
                    label: "front_right_leg".into(),
                    component_type: ComponentType::Skeleton,
                    attachment: Attachment { parent_slot: "torso".into(), offset: [0.0; 3], rotation: [0.0; 3] },
                    transform: GeometryTransform { translate: [0.0; 3], rotate: [0.0; 3], scale: [1.0; 3], structural_mutation: None },
                    children: Vec::new(),
                },
                VisualComponent {
                    label: "back_left_leg".into(),
                    component_type: ComponentType::Skeleton,
                    attachment: Attachment { parent_slot: "torso".into(), offset: [0.0; 3], rotation: [0.0; 3] },
                    transform: GeometryTransform { translate: [0.0; 3], rotate: [0.0; 3], scale: [1.0; 3], structural_mutation: None },
                    children: Vec::new(),
                },
                VisualComponent {
                    label: "back_right_leg".into(),
                    component_type: ComponentType::Skeleton,
                    attachment: Attachment { parent_slot: "torso".into(), offset: [0.0; 3], rotation: [0.0; 3] },
                    transform: GeometryTransform { translate: [0.0; 3], rotate: [0.0; 3], scale: [1.0; 3], structural_mutation: None },
                    children: Vec::new(),
                },
                VisualComponent {
                    label: "head".into(),
                    component_type: ComponentType::Skeleton,
                    attachment: Attachment { parent_slot: "spine".into(), offset: [0.0; 3], rotation: [0.0; 3] },
                    transform: GeometryTransform { translate: [0.0; 3], rotate: [0.0; 3], scale: [1.0; 3], structural_mutation: None },
                    children: Vec::new(),
                },
                VisualComponent {
                    label: "tail".into(),
                    component_type: ComponentType::Skeleton,
                    attachment: Attachment { parent_slot: "spine".into(), offset: [0.0; 3], rotation: [0.0; 3] },
                    transform: GeometryTransform { translate: [0.0; 3], rotate: [0.0; 3], scale: [1.0; 3], structural_mutation: None },
                    children: Vec::new(),
                },
            ],
        }
    }

    #[test]
    fn quadruped_legs_become_arms() {
        let cat = make_quadruped();
        let biped = TransformEngine::apply(&cat, StructuralMutation::QuadrupedToBiped);

        let arm_count = biped
            .children
            .iter()
            .filter(|c| c.label.contains("arm"))
            .count();
        let leg_count = biped
            .children
            .iter()
            .filter(|c| c.label.contains("leg"))
            .count();

        assert_eq!(arm_count, 2, "should have 2 arms");
        assert_eq!(leg_count, 2, "should have 2 legs");
    }

    #[test]
    fn spine_rotated_vertical() {
        let cat = make_quadruped();
        let biped = TransformEngine::apply(&cat, StructuralMutation::QuadrupedToBiped);
        assert!((biped.transform.rotate[2] - (-90.0)).abs() < f32::EPSILON);
    }
}
