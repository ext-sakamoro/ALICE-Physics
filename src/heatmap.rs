//! Stress and Temperature Heatmap Visualization
//!
//! Generates 2D heatmap data from 3D physics state by sampling along axis-aligned
//! slices. Supports stress visualization from contact forces and temperature fields.
//!
//! The output `Heatmap` contains raw `Fix128` values plus min/max range, and can
//! be converted to RGBA pixels using a viridis-like colormap.
//!
//! All sampling uses deterministic 128-bit fixed-point arithmetic.
//! RGBA conversion uses `f64` intermediates for color interpolation only.

use crate::math::{Fix128, Vec3Fix};

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ============================================================================
// Heatmap Types
// ============================================================================

/// Axis along which to take a 2D slice of the 3D domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SliceAxis {
    /// Slice perpendicular to the X axis (YZ plane)
    X,
    /// Slice perpendicular to the Y axis (XZ plane)
    Y,
    /// Slice perpendicular to the Z axis (XY plane)
    Z,
}

/// Configuration for heatmap generation.
#[derive(Clone, Debug)]
pub struct HeatmapConfig {
    /// Horizontal resolution (pixels/cells)
    pub resolution_x: usize,
    /// Vertical resolution (pixels/cells)
    pub resolution_y: usize,
    /// Axis perpendicular to the slice plane
    pub slice_axis: SliceAxis,
    /// Position along the slice axis
    pub slice_offset: Fix128,
    /// Minimum corner of the sampling region
    pub bounds_min: Vec3Fix,
    /// Maximum corner of the sampling region
    pub bounds_max: Vec3Fix,
}

/// A 2D heatmap of scalar values.
#[derive(Clone, Debug)]
pub struct Heatmap {
    /// Row-major scalar data (width * height elements)
    pub data: Vec<Fix128>,
    /// Width (horizontal resolution)
    pub width: usize,
    /// Height (vertical resolution)
    pub height: usize,
    /// Minimum value in the data
    pub min_value: Fix128,
    /// Maximum value in the data
    pub max_value: Fix128,
}

// ============================================================================
// Heatmap Generation
// ============================================================================

/// Build a 3D sample point from 2D grid coordinates and the slice configuration.
fn sample_point(config: &HeatmapConfig, u_frac: Fix128, v_frac: Fix128) -> Vec3Fix {
    let bmin = config.bounds_min;
    let bmax = config.bounds_max;

    match config.slice_axis {
        SliceAxis::X => Vec3Fix::new(
            config.slice_offset,
            bmin.y + (bmax.y - bmin.y) * u_frac,
            bmin.z + (bmax.z - bmin.z) * v_frac,
        ),
        SliceAxis::Y => Vec3Fix::new(
            bmin.x + (bmax.x - bmin.x) * u_frac,
            config.slice_offset,
            bmin.z + (bmax.z - bmin.z) * v_frac,
        ),
        SliceAxis::Z => Vec3Fix::new(
            bmin.x + (bmax.x - bmin.x) * u_frac,
            bmin.y + (bmax.y - bmin.y) * v_frac,
            config.slice_offset,
        ),
    }
}

/// Generate a stress heatmap from rigid body contacts.
///
/// Samples a 2D slice and accumulates contact force density at each pixel
/// using a radial falloff kernel from nearby contact points.
///
/// # Arguments
///
/// - `body_positions`: Positions of all rigid bodies
/// - `contacts`: Contact data as `(point, normal, force_magnitude)` triples
/// - `config`: Heatmap sampling configuration
#[must_use]
pub fn generate_stress_heatmap(
    body_positions: &[Vec3Fix],
    contacts: &[(Vec3Fix, Vec3Fix, Fix128)],
    config: &HeatmapConfig,
) -> Heatmap {
    let w = config.resolution_x.max(1);
    let h = config.resolution_y.max(1);
    let mut data = vec![Fix128::ZERO; w * h];

    let _ = body_positions; // Used for spatial reference; contacts carry positions

    let influence_radius = Fix128::from_int(2);
    let radius_sq = influence_radius * influence_radius;

    for iy in 0..h {
        for ix in 0..w {
            let u = Fix128::from_ratio(ix as i64, w as i64);
            let v = Fix128::from_ratio(iy as i64, h as i64);
            let sample = sample_point(config, u, v);

            let mut stress = Fix128::ZERO;

            for &(ref contact_point, ref _normal, force) in contacts {
                let diff = sample - *contact_point;
                let dist_sq = diff.dot(diff);

                if dist_sq < radius_sq {
                    // Linear falloff
                    let dist = dist_sq.sqrt();
                    let falloff = Fix128::ONE - dist / influence_radius;
                    stress = stress + force * falloff;
                }
            }

            data[iy * w + ix] = stress;
        }
    }

    compute_heatmap_minmax(data, w, h)
}

/// Generate a temperature heatmap by sampling a temperature function.
///
/// Evaluates the provided temperature function at each sample point on
/// the 2D slice.
///
/// # Arguments
///
/// - `temperature_fn`: Function mapping a 3D point to a temperature value
/// - `config`: Heatmap sampling configuration
#[must_use]
pub fn generate_temperature_heatmap<F>(temperature_fn: F, config: &HeatmapConfig) -> Heatmap
where
    F: Fn(Vec3Fix) -> Fix128,
{
    let w = config.resolution_x.max(1);
    let h = config.resolution_y.max(1);
    let mut data = vec![Fix128::ZERO; w * h];

    for iy in 0..h {
        for ix in 0..w {
            let u = Fix128::from_ratio(ix as i64, w as i64);
            let v = Fix128::from_ratio(iy as i64, h as i64);
            let sample = sample_point(config, u, v);
            data[iy * w + ix] = temperature_fn(sample);
        }
    }

    compute_heatmap_minmax(data, w, h)
}

/// Compute min/max and build the `Heatmap` struct.
fn compute_heatmap_minmax(data: Vec<Fix128>, width: usize, height: usize) -> Heatmap {
    let mut min_val = data.first().copied().unwrap_or(Fix128::ZERO);
    let mut max_val = min_val;

    for &v in &data {
        if v < min_val {
            min_val = v;
        }
        if v > max_val {
            max_val = v;
        }
    }

    Heatmap {
        data,
        width,
        height,
        min_value: min_val,
        max_value: max_val,
    }
}

// ============================================================================
// RGBA Conversion (viridis-like colormap)
// ============================================================================

/// Convert a heatmap to RGBA pixels using a viridis-like colormap.
///
/// Maps `min_value` to dark purple and `max_value` to bright yellow.
/// Returns one `[R, G, B, A]` per pixel in row-major order.
///
/// This function uses `f64` intermediates for color interpolation.
/// The heatmap values themselves remain in `Fix128`.
#[must_use]
pub fn heatmap_to_rgba(heatmap: &Heatmap) -> Vec<[u8; 4]> {
    let range = heatmap.max_value - heatmap.min_value;
    let has_range = !range.is_zero();

    heatmap
        .data
        .iter()
        .map(|&val| {
            let t = if has_range {
                ((val - heatmap.min_value) / range).to_f64().clamp(0.0, 1.0)
            } else {
                0.5
            };
            viridis_color(t)
        })
        .collect()
}

/// Viridis-like colormap: 5-stop linear interpolation.
///
/// Stops: 0.0 -> dark purple, 0.25 -> blue, 0.5 -> teal,
///        0.75 -> green-yellow, 1.0 -> bright yellow.
fn viridis_color(t: f64) -> [u8; 4] {
    // Viridis key colors (simplified 5-stop gradient)
    const STOPS: [(f64, f64, f64); 5] = [
        (0.267, 0.004, 0.329), // dark purple
        (0.282, 0.140, 0.458), // blue-purple
        (0.127, 0.566, 0.551), // teal
        (0.544, 0.774, 0.247), // green-yellow
        (0.993, 0.906, 0.144), // bright yellow
    ];

    let t = t.clamp(0.0, 1.0);
    let segment = (t * 4.0).min(3.999);
    let idx = segment as usize;
    let frac = segment - idx as f64;

    let (r0, g0, b0) = STOPS[idx];
    let (r1, g1, b1) = STOPS[idx + 1];

    let r = r0 + (r1 - r0) * frac;
    let g = g0 + (g1 - g0) * frac;
    let b = b0 + (b1 - b0) * frac;

    [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8, 255]
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> HeatmapConfig {
        HeatmapConfig {
            resolution_x: 4,
            resolution_y: 4,
            slice_axis: SliceAxis::Z,
            slice_offset: Fix128::ZERO,
            bounds_min: Vec3Fix::from_int(-5, -5, -5),
            bounds_max: Vec3Fix::from_int(5, 5, 5),
        }
    }

    #[test]
    fn test_stress_heatmap_no_contacts() {
        let config = default_config();
        let heatmap = generate_stress_heatmap(&[], &[], &config);
        assert_eq!(heatmap.width, 4);
        assert_eq!(heatmap.height, 4);
        assert_eq!(heatmap.data.len(), 16);
        // All values should be zero
        for &v in &heatmap.data {
            assert!(v.is_zero());
        }
    }

    #[test]
    fn test_stress_heatmap_with_contact() {
        let config = default_config();
        let contacts = vec![(Vec3Fix::ZERO, Vec3Fix::UNIT_Y, Fix128::from_int(100))];
        let heatmap = generate_stress_heatmap(&[], &contacts, &config);
        assert!(heatmap.max_value > Fix128::ZERO);
    }

    #[test]
    fn test_temperature_heatmap_uniform() {
        let config = default_config();
        let temp = Fix128::from_int(300); // 300K
        let heatmap = generate_temperature_heatmap(|_| temp, &config);
        assert_eq!(heatmap.min_value, temp);
        assert_eq!(heatmap.max_value, temp);
    }

    #[test]
    fn test_temperature_heatmap_gradient() {
        let config = HeatmapConfig {
            resolution_x: 10,
            resolution_y: 1,
            slice_axis: SliceAxis::Z,
            slice_offset: Fix128::ZERO,
            bounds_min: Vec3Fix::from_int(0, 0, 0),
            bounds_max: Vec3Fix::from_int(10, 10, 10),
        };
        // Temperature increases with X
        let heatmap = generate_temperature_heatmap(|p| p.x, &config);
        assert!(heatmap.max_value > heatmap.min_value);
    }

    #[test]
    fn test_heatmap_to_rgba_length() {
        let config = default_config();
        let heatmap = generate_temperature_heatmap(|_| Fix128::ONE, &config);
        let rgba = heatmap_to_rgba(&heatmap);
        assert_eq!(rgba.len(), 16); // 4x4
    }

    #[test]
    fn test_heatmap_to_rgba_alpha_255() {
        let config = default_config();
        let heatmap = generate_temperature_heatmap(|p| p.x, &config);
        let rgba = heatmap_to_rgba(&heatmap);
        for pixel in &rgba {
            assert_eq!(pixel[3], 255);
        }
    }

    #[test]
    fn test_viridis_endpoints() {
        let dark = viridis_color(0.0);
        let bright = viridis_color(1.0);
        // Dark purple should have low R, G, moderate B
        assert!(dark[0] < 100);
        // Bright yellow should have high R, G
        assert!(bright[0] > 200);
        assert!(bright[1] > 200);
    }

    #[test]
    fn test_slice_axis_x() {
        let config = HeatmapConfig {
            resolution_x: 2,
            resolution_y: 2,
            slice_axis: SliceAxis::X,
            slice_offset: Fix128::from_int(5),
            bounds_min: Vec3Fix::from_int(0, 0, 0),
            bounds_max: Vec3Fix::from_int(10, 10, 10),
        };
        // Temperature = X coordinate
        let heatmap = generate_temperature_heatmap(|p| p.x, &config);
        // All samples should have X=5 (the slice offset)
        for &v in &heatmap.data {
            assert_eq!(v.hi, 5);
        }
    }

    #[test]
    fn test_slice_axis_y() {
        let config = HeatmapConfig {
            resolution_x: 2,
            resolution_y: 2,
            slice_axis: SliceAxis::Y,
            slice_offset: Fix128::from_int(3),
            bounds_min: Vec3Fix::from_int(0, 0, 0),
            bounds_max: Vec3Fix::from_int(10, 10, 10),
        };
        let heatmap = generate_temperature_heatmap(|p| p.y, &config);
        for &v in &heatmap.data {
            assert_eq!(v.hi, 3);
        }
    }

    #[test]
    fn test_heatmap_dimensions() {
        let config = HeatmapConfig {
            resolution_x: 7,
            resolution_y: 3,
            slice_axis: SliceAxis::Z,
            slice_offset: Fix128::ZERO,
            bounds_min: Vec3Fix::from_int(-1, -1, -1),
            bounds_max: Vec3Fix::from_int(1, 1, 1),
        };
        let heatmap = generate_temperature_heatmap(|_| Fix128::ZERO, &config);
        assert_eq!(heatmap.width, 7);
        assert_eq!(heatmap.height, 3);
        assert_eq!(heatmap.data.len(), 21);
    }
}
