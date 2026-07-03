use crate::effector::*;
use semantic_graph::prelude::*;

// ────────────────────────────────────────────────────────────
//  TransformEngine — deterministic sensor→image transforms
//
//  Runs on the cognitive daemon thread (not in wgpu). Reads raw
//  sensor values and produces VisualEffectorState modifications
//  that are pushed atomically into the VisualEffectorBuffer for
//  the wgpu RenderBridge to consume.
//
//  All transforms are deterministic functions of:
//    - Raw sensor values (lux, accelerometer gx/gy/gz)
//    - Neuromodulator levels (arousal, novelty)
//    - Fired node activation levels (KineticEnergy, SpatialBound)
// ────────────────────────────────────────────────────────────

/// The transform engine — one per cognitive daemon.
pub struct TransformEngine {
    pub palette: PaletteInterpolator,
    prev_gravity: GravityVector,
}

impl TransformEngine {
    pub fn new() -> Self {
        TransformEngine {
            palette: PaletteInterpolator::new(),
            prev_gravity: GravityVector::new(0.0, 9.81, 0.0),
        }
    }

    /// Process a light sensor reading and produce palette coefficients.
    ///
    /// Returns (pal_coeff_0, pal_coeff_1, wireframe_flag_as_f32).
    pub fn process_light(&mut self, lux: f32, arousal: f32) -> (f32, f32, f32) {
        let (pc0, pc1) = self.palette.compute(lux, arousal);
        let wf = if self.palette.wireframe_active() { 1.0 } else { 0.0 };
        (pc0, pc1, wf)
    }

    /// Process an accelerometer reading and produce skeletal transform deltas.
    ///
    /// Returns (rotation_angles_radians, scale_adjustment).
    pub fn process_accelerometer(
        &mut self,
        gx: f32,
        gy: f32,
        gz: f32,
        kinetic_activation: f32,
    ) -> ([f32; 6], [f32; 3]) {
        let g = GravityVector::new(gx, gy, gz);
        let matrix = SkeletalTransformMatrix::from_gravity(&g, kinetic_activation);
        let (sx, sy, sz) = SkeletalTransformMatrix::rest_pose_adjustment(&g);

        // Extract rotation angles from matrix (simplified: fill rot0..rot5)
        let mut rotations = [0.0f32; 6];
        rotations[0] = matrix.data[5].atan2(matrix.data[10]); // rot_x
        rotations[1] = (-matrix.data[2]).atan2(
            (matrix.data[6] * matrix.data[6] + matrix.data[10] * matrix.data[10]).sqrt(),
        ); // rot_y
        rotations[2] = matrix.data[6].atan2(matrix.data[0]); // rot_z
        // rot3..rot5 aren't driven by gravity alone; set proportional to rotational energy
        let rot_energy = (rotations[0].abs() + rotations[1].abs() + rotations[2].abs()) / 3.0;
        rotations[3] = rot_energy * kinetic_activation * 0.2;
        rotations[4] = rot_energy * kinetic_activation * 0.15;
        rotations[5] = rot_energy * kinetic_activation * 0.1;

        self.prev_gravity = g;
        (rotations, [sx, sy, sz])
    }

    /// Apply a full sensor update to the effector state array,
    /// returning the updated state based on sensor readings and
    /// neuromodulator levels.
    ///
    /// This is the main entry point called from the cognitive engine.
    pub fn update_effector_state(
        &mut self,
        state: &mut [f32; EFFECTOR_STATE_FLOATS],
        lux: f32,
        arousal: f32,
        gx: f32,
        gy: f32,
        gz: f32,
        kinetic_activation: f32,
        spatial_activation: f32,
    ) {
        // Light → palette
        let (pc0, pc1, wf) = self.process_light(lux, arousal);
        effector_state::set_palette_coeffs(state, pc0, pc1);
        effector_state::set_wireframe(state, wf);

        // Accel → skeletal rotations + scale
        let (rotations, scale) = self.process_accelerometer(gx, gy, gz, kinetic_activation);
        for j in 0..6 {
            effector_state::set_rotation(state, j, rotations[j]);
        }
        effector_state::set_scale(state, scale[0], scale[1], scale[2]);

        // Spatial activation → color intensity
        let color_intensity = (spatial_activation * 0.8 + 0.2).clamp(0.0, 1.0);
        effector_state::set_color0(state, color_intensity, color_intensity * 0.5, 0.3, 1.0);
        effector_state::set_color1(state, 0.3, 0.5, color_intensity, 0.8);

        // Blend weight from arousal
        let blend = (arousal * 0.5 + 0.5).clamp(0.0, 1.0);
        effector_state::set_blend(state, blend);
    }
}

/// Convert a light sensor value (lux) to a wireframe/color-palette
/// interpolation vector. Pure function, no side effects.
pub fn light_to_palette_matrix(lux: f32, arousal: f32) -> (f32, f32, f32, f32, f32) {
    let norm_lux = (lux.max(0.0) / 10000.0).sqrt().clamp(0.0, 1.0);
    let warm = (arousal * 0.4).clamp(0.0, 0.4);
    let base = norm_lux * 0.7 + 0.3;
    let pc0 = (base + warm).clamp(0.0, 1.0);
    let pc1 = (base - warm * 0.5).clamp(0.0, 1.0);
    let wireframe = if lux < 10.0 && arousal > 0.5 { 1.0 } else { 0.0 };
    (pc0, pc1, wireframe, base, warm)
}

/// Convert accelerometer vector + activation to skeletal rotation
/// matrix components. Pure function.
pub fn accel_to_skeletal_rotation(gx: f32, gy: f32, gz: f32, activation: f32) -> [f32; 6] {
    let g = GravityVector::new(gx, gy, gz);
    let matrix = SkeletalTransformMatrix::from_gravity(&g, activation);
    let mut rotations = [0.0f32; 6];
    rotations[0] = matrix.data[5].atan2(matrix.data[10]);
    rotations[1] = (-matrix.data[2]).atan2(
        (matrix.data[6] * matrix.data[6] + matrix.data[10] * matrix.data[10]).sqrt(),
    );
    rotations[2] = matrix.data[6].atan2(matrix.data[0]);
    let rot_energy = (rotations[0].abs() + rotations[1].abs() + rotations[2].abs()) / 3.0;
    rotations[3] = rot_energy * activation * 0.2;
    rotations[4] = rot_energy * activation * 0.15;
    rotations[5] = rot_energy * activation * 0.1;
    rotations
}

/// Convert gravitational vector to rest-pose scale modifiers.
pub fn gravitational_to_rest_pose(gx: f32, gy: f32, gz: f32) -> (f32, f32, f32) {
    let g = GravityVector::new(gx, gy, gz);
    SkeletalTransformMatrix::rest_pose_adjustment(&g)
}
