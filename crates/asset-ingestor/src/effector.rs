use semantic_graph::prelude::*;

// ────────────────────────────────────────────────────────────
//  Effector math types
//
//  These types transform sensor readings into deterministic
//  visual effector parameters. No random numbers — all derived
//  from activation-weighted sensor deltas and neuromodulator
//  levels.
// ────────────────────────────────────────────────────────────

/// 3D gravity vector from accelerometer (m/s²).
#[derive(Debug, Clone, Copy)]
pub struct GravityVector {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl GravityVector {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        GravityVector { x, y, z }
    }

    /// Magnitude of the gravity vector.
    pub fn magnitude(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Normalize to unit vector.
    pub fn normalize(&self) -> Self {
        let m = self.magnitude();
        if m < 0.001 {
            GravityVector::new(0.0, 1.0, 0.0)
        } else {
            GravityVector::new(self.x / m, self.y / m, self.z / m)
        }
    }

    /// Determine if the skeleton is supine (laying on back).
    /// Supine when gravity vector points through the back (z > 0.8).
    pub fn is_supine(&self) -> bool {
        self.z > 8.0
    }

    /// Determine if the skeleton is upright (|g| < 0.5 means free-fall / no dominant axis).
    pub fn is_upright(&self) -> bool {
        self.magnitude() < 0.5
    }
}

/// Palette interpolator: maps light sensor lux values and arousal
/// to palette color coefficients deterministically.
#[derive(Debug, Clone)]
pub struct PaletteInterpolator {
    prev_lux: f32,
    /// Spikes to 1.0 on sudden light drop and decays at 0.92/tick
    dark_flash: f32,
}

impl PaletteInterpolator {
    pub fn new() -> Self {
        PaletteInterpolator {
            prev_lux: 500.0,
            dark_flash: 0.0,
        }
    }

    /// Compute palette coefficients from lux value and arousal.
    ///
    /// Returns (pal_coeff_0, pal_coeff_1) in [0, 1].
    ///   - Bright light → coeffs near 1.0 (full color)
    ///   - Dim light → coeffs near 0.3 (muted)
    ///   - Sudden drop >50% → dark_flash spikes, forces wireframe high-contrast
    ///   - High arousal shifts toward warm palette (pal_coeff_0 > pal_coeff_1)
    pub fn compute(&mut self, lux: f32, arousal: f32) -> (f32, f32) {
        // Normalize lux: 0..10000 → 0..1 with log scaling
        let norm_lux = (lux.max(0.0) / 10000.0).sqrt().clamp(0.0, 1.0);

        // Detect sudden drop
        let drop_ratio = if self.prev_lux > 1.0 {
            (lux / self.prev_lux).clamp(0.0, 1.0)
        } else {
            1.0
        };
        if drop_ratio < 0.5 {
            // Sudden drop > 50% → spike flash
            self.dark_flash = 1.0;
        }
        self.dark_flash *= 0.92;

        self.prev_lux = lux;

        // Base palette: dim light → muted, bright → full
        let base = if self.dark_flash > 0.1 {
            // Dark flash: high contrast
            1.0
        } else {
            norm_lux * 0.7 + 0.3
        };

        // Arousal shift: warm palette under arousal
        let warm_shift = (arousal * 0.4).clamp(0.0, 0.4);
        let pal_coeff_0 = (base + warm_shift).clamp(0.0, 1.0);
        let pal_coeff_1 = (base - warm_shift * 0.5).clamp(0.0, 1.0);

        (pal_coeff_0, pal_coeff_1)
    }

    /// Whether wireframe mode should be active (high contrast).
    pub fn wireframe_active(&self) -> bool {
        self.dark_flash > 0.1
    }
}

/// A 4×4 skeletal transform matrix stored as [f32; 16] (column-major).
///
/// This is the deterministic mapping from sensor data + activation
/// levels to skeletal bone transforms.
#[derive(Debug, Clone, Copy)]
pub struct SkeletalTransformMatrix {
    /// Column-major 4×4 matrix (16 elements)
    pub data: [f32; 16],
}

impl SkeletalTransformMatrix {
    pub fn identity() -> Self {
        let mut data = [0.0f32; 16];
        data[0] = 1.0;
        data[5] = 1.0;
        data[10] = 1.0;
        data[15] = 1.0;
        SkeletalTransformMatrix { data }
    }

    /// Build a rotation matrix around the X axis.
    pub fn rotation_x(angle: f32) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        let mut m = Self::identity();
        m.data[5] = c;
        m.data[6] = s;
        m.data[9] = -s;
        m.data[10] = c;
        m
    }

    /// Build a rotation matrix around the Y axis.
    pub fn rotation_y(angle: f32) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        let mut m = Self::identity();
        m.data[0] = c;
        m.data[2] = -s;
        m.data[8] = s;
        m.data[10] = c;
        m
    }

    /// Build a rotation matrix around the Z axis.
    pub fn rotation_z(angle: f32) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        let mut m = Self::identity();
        m.data[0] = c;
        m.data[1] = s;
        m.data[4] = -s;
        m.data[5] = c;
        m
    }

    /// Multiply two 4×4 matrices.
    pub fn multiply(&self, other: &SkeletalTransformMatrix) -> Self {
        let mut result = [0.0f32; 16];
        for col in 0..4 {
            for row in 0..4 {
                result[col * 4 + row] =
                    self.data[0 * 4 + row] * other.data[col * 4 + 0]
                        + self.data[1 * 4 + row] * other.data[col * 4 + 1]
                        + self.data[2 * 4 + row] * other.data[col * 4 + 2]
                        + self.data[3 * 4 + row] * other.data[col * 4 + 3];
            }
        }
        SkeletalTransformMatrix { data: result }
    }

    /// Apply to effector state: write rotation angles into rot0..rot5.
    pub fn write_rotations(&self, state: &mut [f32; EFFECTOR_STATE_FLOATS]) {
        // Decompose matrix to Euler angles (XYZ order)
        let sy = (self.data[2] * self.data[2] + self.data[6] * self.data[6]).sqrt();
        let singular = sy < 1e-6;
        if !singular {
            effector_state::set_rotation(state, 0, self.data[10].atan2(self.data[14]));
            effector_state::set_rotation(state, 1, (-self.data[2]).atan2(sy));
            effector_state::set_rotation(state, 2, self.data[6].atan2(self.data[0]));
        } else {
            effector_state::set_rotation(state, 0, (-self.data[10]).atan2(self.data[5]));
            effector_state::set_rotation(state, 1, (-self.data[2]).atan2(sy));
            effector_state::set_rotation(state, 2, 0.0);
        }
    }

    /// Derive gravity-based rotation from accelerometer vector.
    pub fn from_gravity(g: &GravityVector, activation: f32) -> Self {
        let n = g.normalize();
        // Map gravity direction → rotation: upright = identity, tilted = rotate
        let angle_x = n.y.atan2(n.z) * activation;
        let angle_y = n.x.atan2(n.z) * activation * 0.5;
        let angle_z = n.x.atan2(n.y) * activation * 0.3;
        let rx = Self::rotation_x(angle_x);
        let ry = Self::rotation_y(angle_y);
        let rz = Self::rotation_z(angle_z);
        rx.multiply(&ry).multiply(&rz)
    }

    /// Derive rest-pose modifier from gravitational magnitude.
    /// Upright (|g| < 0.5) → scale = 1.0, supine (|g| > 8) → scale compressed
    pub fn rest_pose_adjustment(g: &GravityVector) -> (f32, f32, f32) {
        let mag = g.magnitude();
        if g.is_supine() {
            // Supine: compressed vertical scale
            (1.0, 0.6, 1.0)
        } else if g.is_upright() {
            // Upright: full scale
            (1.0, 1.0, 1.0)
        } else {
            // Tilted: proportional vertical compression
            let compression = 1.0 - (mag / 9.81).clamp(0.0, 0.4);
            (1.0, compression, 1.0)
        }
    }
}
