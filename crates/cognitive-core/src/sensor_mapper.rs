use semantic_graph::prelude::*;

// ────────────────────────────────────────────────────────────
//  SensorMapper — stateless fixed-point sensor→activation mapper
//
//  Converts raw hardware sensor readings into activation injection
//  targets for visual primitive nodes. Used during Phase 2 (Injection)
//  of the cognitive tick.
//
//  All math is deterministic fixed-point (f64). No random numbers.
//  No heap allocations.
// ────────────────────────────────────────────────────────────

/// Maximum injection energy from a single sensor reading.
const MAX_SENSOR_INJECT: f64 = 0.8;

/// Scale factor for accelerometer → rotation mapping.
/// Maps 0..20 m/s² to 0..1 activation range.
const ACCEL_SCALE: f64 = 0.05;

/// Scale factor for light lux → chroma mapping.
/// Maps 0..10000 lux to 0..1 activation range.
const LIGHT_SCALE: f64 = 0.001;

/// Variance threshold for prediction error detection (>30%).
const VARIANCE_THRESHOLD: f64 = 0.30;

/// SensorMapper is a stateless namespace of mapping functions.
/// No state, no allocations, no side effects.
pub struct SensorMapper;

impl SensorMapper {
    /// Map accelerometer X component to RotationX injection.
    ///
    /// Fixed-point math: gx * ACCEL_SCALE, clamped to [-MAX_SENSOR_INJECT, MAX_SENSOR_INJECT].
    /// Positive gx (rightward acceleration) → positive rotation activation.
    pub fn map_accel_x(gx: f64) -> f64 {
        (gx * ACCEL_SCALE).clamp(-MAX_SENSOR_INJECT, MAX_SENSOR_INJECT)
    }

    /// Map accelerometer Y component to RotationY injection.
    pub fn map_accel_y(gy: f64) -> f64 {
        (gy * ACCEL_SCALE).clamp(-MAX_SENSOR_INJECT, MAX_SENSOR_INJECT)
    }

    /// Map accelerometer Z component to RotationZ injection.
    pub fn map_accel_z(gz: f64) -> f64 {
        (gz * ACCEL_SCALE).clamp(-MAX_SENSOR_INJECT, MAX_SENSOR_INJECT)
    }

    /// Map all three accelerometer axes to their respective rotation
    /// node injection targets. Returns [(NodeId, activation), ...].
    pub fn map_accelerometer(gx: f64, gy: f64, gz: f64) -> [(NodeId, f64); 3] {
        [
            (NodeId::from_raw(VISUAL_ROTATION_X), Self::map_accel_x(gx)),
            (NodeId::from_raw(VISUAL_ROTATION_Y), Self::map_accel_y(gy)),
            (NodeId::from_raw(VISUAL_ROTATION_Z), Self::map_accel_z(gz)),
        ]
    }

    /// Map ambient light value to ColorChroma injection.
    ///
    /// Fixed-point math: lux * LIGHT_SCALE, clamped to [0, MAX_SENSOR_INJECT].
    /// Higher light → more chroma saturation activation.
    pub fn map_light(lux: f64) -> f64 {
        (lux * LIGHT_SCALE).clamp(0.0, MAX_SENSOR_INJECT)
    }

    /// Map ambient light to SpatialScale injection.
    /// Dim light → lower spatial scale (reduced detail).
    pub fn map_light_to_scale(lux: f64) -> f64 {
        let norm = (lux / 1000.0).clamp(0.0, 1.0);
        (norm * 0.5 + 0.2).clamp(0.0, MAX_SENSOR_INJECT)
    }

    /// Map all light sensor information to visual primitive injections.
    pub fn map_light_sensor(lux: f64) -> [(NodeId, f64); 2] {
        [
            (NodeId::from_raw(VISUAL_COLOR_CHROMA), Self::map_light(lux)),
            (NodeId::from_raw(VISUAL_SPATIAL_SCALE), Self::map_light_to_scale(lux)),
        ]
    }

    /// Detect sensor variance > 30% and compute prediction error magnitude.
    ///
    /// Returns `Some(error_magnitude)` if the absolute ratio of
    /// `|curr - prev| / max(|prev|, 0.001)` exceeds the threshold.
    /// The magnitude is clamped to [0, 1] for use as a novelty spike.
    pub fn variance_error(prev: f64, curr: f64) -> Option<f64> {
        let denom = prev.abs().max(0.001);
        let ratio = (curr - prev).abs() / denom;
        if ratio > VARIANCE_THRESHOLD {
            Some(ratio.clamp(0.0, 1.0))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accelerometer_maps_to_rotation() {
        let targets = SensorMapper::map_accelerometer(9.81, 0.0, 0.0);
        assert_eq!(targets.len(), 3);
        // X-axis: 9.81 * 0.05 = 0.4905
        assert!((targets[0].1 - 0.4905).abs() < 0.001);
        assert!((targets[1].1).abs() < 0.001);
        assert!((targets[2].1).abs() < 0.001);
    }

    #[test]
    fn light_maps_to_chroma() {
        let targets = SensorMapper::map_light_sensor(500.0);
        assert!((targets[0].1 - 0.5).abs() < 0.001);
        assert!(targets[0].1 <= 0.8);
    }

    #[test]
    fn variance_detects_threshold() {
        assert!(SensorMapper::variance_error(10.0, 15.0).is_none()); // 50% diff
        assert!(SensorMapper::variance_error(10.0, 13.1).is_some()); // 31% diff > 30%
        assert!(SensorMapper::variance_error(10.0, 11.0).is_none()); // 10% diff < 30%
    }

    #[test]
    fn map_accel_zero_is_zero() {
        assert!((SensorMapper::map_accel_x(0.0)).abs() < 1e-10);
        assert!((SensorMapper::map_accel_y(0.0)).abs() < 1e-10);
        assert!((SensorMapper::map_accel_z(0.0)).abs() < 1e-10);
    }
}
