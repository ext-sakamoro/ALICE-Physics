//! Analytic-oracle tests for `heatmap`'s wiring pass.
//!
//! `heatmap` had 6 items (`Heatmap`/`HeatmapConfig`/`SliceAxis`/
//! `generate_stress_heatmap`/`generate_temperature_heatmap`/
//! `heatmap_to_rgba`) that production never called outside the module's own
//! `#[cfg(test)]` block. `examples/heatmap_visualization.rs` is the
//! production entry point; this file pins the closed forms independently of
//! that example and of the implementation under test.
//!
//! Expected values are hand-derived from the formulas documented in
//! `src/heatmap.rs` (the private `sample_point` per-axis mapping, the
//! `influence_radius = 2` radial-falloff kernel in `generate_stress_heatmap`,
//! and the 5-stop viridis lerp in `viridis_color`), never by calling the
//! functions under test for the expected side. The viridis STOPS table is
//! transcribed verbatim from the doc comment above `viridis_color` and
//! re-interpolated independently in each test.

#![cfg(feature = "std")]

use alice_physics::heatmap::{
    generate_stress_heatmap, generate_temperature_heatmap, heatmap_to_rgba, Heatmap, HeatmapConfig,
    SliceAxis,
};
use alice_physics::math::{Fix128, Vec3Fix};
use std::panic::{catch_unwind, AssertUnwindSafe};

/// STOPS table copied verbatim from `viridis_color`'s doc comment
/// (`src/heatmap.rs`), re-interpolated here independently of the
/// implementation under test.
const STOPS: [(f64, f64, f64); 5] = [
    (0.267, 0.004, 0.329),
    (0.282, 0.140, 0.458),
    (0.127, 0.566, 0.551),
    (0.544, 0.774, 0.247),
    (0.993, 0.906, 0.144),
];

fn expected_viridis(t: f64) -> [u8; 4] {
    let t = t.clamp(0.0, 1.0);
    let segment = (t * 4.0).min(3.999);
    let idx = segment as usize;
    let frac = segment - idx as f64;
    let (r0, g0, b0) = STOPS[idx];
    let (r1, g1, b1) = STOPS[idx + 1];
    let r = (r1 - r0) * frac + r0;
    let g = (g1 - g0) * frac + g0;
    let b = (b1 - b0) * frac + b0;
    [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8, 255]
}

// ============================================================================
// HeatmapConfig / SliceAxis: per-axis slice-plane mapping (`sample_point`)
// ============================================================================

/// SliceAxis::X fixes x = slice_offset; y,z vary with u,v over the bounds box.
#[test]
fn slice_axis_x_fixes_x_varies_y_and_z() {
    let config = HeatmapConfig {
        resolution_x: 4,
        resolution_y: 2,
        slice_axis: SliceAxis::X,
        slice_offset: Fix128::from_int(5),
        bounds_min: Vec3Fix::from_int(0, 0, 0),
        bounds_max: Vec3Fix::from_int(8, 8, 8),
    };
    // u = ix/4 in {0,.25,.5,.75} -> y = 8u in {0,2,4,6}
    // v = iy/2 in {0,.5}         -> z = 8v in {0,4}
    let y_map = generate_temperature_heatmap(|p| p.y, &config);
    assert_eq!(y_map.data, [0, 2, 4, 6, 0, 2, 4, 6].map(Fix128::from_int));
    let z_map = generate_temperature_heatmap(|p| p.z, &config);
    assert_eq!(z_map.data, [0, 0, 0, 0, 4, 4, 4, 4].map(Fix128::from_int));
    let x_map = generate_temperature_heatmap(|p| p.x, &config);
    assert!(x_map.data.iter().all(|&v| v == Fix128::from_int(5)));
}

/// SliceAxis::Y fixes y = slice_offset; x,z vary.
#[test]
fn slice_axis_y_fixes_y_varies_x_and_z() {
    let config = HeatmapConfig {
        resolution_x: 2,
        resolution_y: 2,
        slice_axis: SliceAxis::Y,
        slice_offset: Fix128::from_int(3),
        bounds_min: Vec3Fix::from_int(0, 0, 0),
        bounds_max: Vec3Fix::from_int(4, 4, 4),
    };
    // u = ix/2 in {0,.5} -> x = 4u in {0,2}
    // v = iy/2 in {0,.5} -> z = 4v in {0,2}
    let x_map = generate_temperature_heatmap(|p| p.x, &config);
    assert_eq!(x_map.data, [0, 2, 0, 2].map(Fix128::from_int));
    let z_map = generate_temperature_heatmap(|p| p.z, &config);
    assert_eq!(z_map.data, [0, 0, 2, 2].map(Fix128::from_int));
    let y_map = generate_temperature_heatmap(|p| p.y, &config);
    assert!(y_map.data.iter().all(|&v| v == Fix128::from_int(3)));
}

/// SliceAxis::Z fixes z = slice_offset; x,y vary.
#[test]
fn slice_axis_z_fixes_z_varies_x_and_y() {
    let config = HeatmapConfig {
        resolution_x: 2,
        resolution_y: 2,
        slice_axis: SliceAxis::Z,
        slice_offset: Fix128::from_int(-7),
        bounds_min: Vec3Fix::from_int(0, 0, 0),
        bounds_max: Vec3Fix::from_int(4, 4, 4),
    };
    let x_map = generate_temperature_heatmap(|p| p.x, &config);
    assert_eq!(x_map.data, [0, 2, 0, 2].map(Fix128::from_int));
    let y_map = generate_temperature_heatmap(|p| p.y, &config);
    assert_eq!(y_map.data, [0, 0, 2, 2].map(Fix128::from_int));
    let z_map = generate_temperature_heatmap(|p| p.z, &config);
    assert!(z_map.data.iter().all(|&v| v == Fix128::from_int(-7)));
}

// ============================================================================
// generate_stress_heatmap: radial-falloff kernel (influence_radius = 2)
// ============================================================================

fn stress_config_4x1() -> HeatmapConfig {
    HeatmapConfig {
        resolution_x: 4,
        resolution_y: 1,
        slice_axis: SliceAxis::Z,
        slice_offset: Fix128::ZERO,
        bounds_min: Vec3Fix::from_int(0, 0, 0),
        bounds_max: Vec3Fix::from_int(4, 0, 0),
    }
}

/// Closed form: stress(p) = sum over contacts with dist_sq < 4 of
/// force * (1 - sqrt(dist_sq) / 2). Samples land at x = 0,1,2,3.
///
/// Mutation-testing note (2026-10 wiring pass): flipping the kernel's
/// boundary comparison from `dist_sq < radius_sq` to `dist_sq <= radius_sq`
/// survives against this test (and every other test in this file) — this is
/// a *proven* equivalent mutant, not an oracle gap. At the exact boundary
/// (dist == influence_radius), `falloff = 1 - dist/influence_radius = 1 - 1
/// = 0` identically (confirmed directly against `Fix128` arithmetic: dist=2,
/// influence_radius=2 gives `falloff.is_zero() == true` for any force
/// magnitude), so `force * falloff` is exactly zero whether or not that
/// boundary point is included by the comparison. No input can make this
/// mutation observable through `generate_stress_heatmap`'s output.
#[test]
fn generate_stress_heatmap_radial_falloff_closed_form() {
    let contacts = [(Vec3Fix::ZERO, Vec3Fix::UNIT_Y, Fix128::from_int(100))];
    let config = stress_config_4x1();
    let map = generate_stress_heatmap(&[], &contacts, &config);
    assert_eq!(map.width, 4);
    assert_eq!(map.height, 1);
    assert_eq!(
        map.data,
        [
            Fix128::from_int(100),                            // x=0: dist=0, falloff=1.0
            Fix128::from_ratio(1, 2) * Fix128::from_int(100), // x=1: dist=1, falloff=0.5
            Fix128::ZERO,                                     // x=2: dist_sq=4, NOT < 4
            Fix128::ZERO,                                     // x=3: dist_sq=9, NOT < 4
        ]
    );
    assert_eq!(map.min_value, Fix128::ZERO);
    assert_eq!(map.max_value, Fix128::from_int(100));
}

/// Multiple contacts at the same point sum their force contributions rather
/// than overwriting (the accumulator is `stress = stress + force * falloff`,
/// never a plain assignment).
#[test]
fn generate_stress_heatmap_accumulates_multiple_contacts() {
    let contacts = [
        (Vec3Fix::ZERO, Vec3Fix::UNIT_Y, Fix128::from_int(100)),
        (Vec3Fix::ZERO, Vec3Fix::UNIT_Y, Fix128::from_int(10)),
    ];
    let config = stress_config_4x1();
    let map = generate_stress_heatmap(&[], &contacts, &config);
    assert_eq!(map.data[0], Fix128::from_int(110)); // (100+10) * falloff(0)=1.0
    assert_eq!(
        map.data[1],
        Fix128::from_ratio(1, 2) * Fix128::from_int(110)
    ); // (100+10) * falloff(1)=0.5
}

/// Degenerate: empty contacts (and empty body_positions) yields an all-zero
/// field of the configured dimensions, not an empty/undersized buffer.
#[test]
fn generate_stress_heatmap_empty_contacts_is_all_zero() {
    let config = HeatmapConfig {
        resolution_x: 3,
        resolution_y: 2,
        ..stress_config_4x1()
    };
    let map = generate_stress_heatmap(&[], &[], &config);
    assert_eq!(map.width, 3);
    assert_eq!(map.height, 2);
    assert_eq!(map.data.len(), 6);
    assert!(map.data.iter().all(|v| v.is_zero()));
    assert_eq!(map.min_value, Fix128::ZERO);
    assert_eq!(map.max_value, Fix128::ZERO);
}

/// Degenerate: single-cell field (1x1). u=v=0 so the one sample lands
/// exactly at the bounds_min corner.
#[test]
fn generate_stress_heatmap_single_cell_resolution() {
    let config = HeatmapConfig {
        resolution_x: 1,
        resolution_y: 1,
        slice_axis: SliceAxis::Z,
        slice_offset: Fix128::ZERO,
        bounds_min: Vec3Fix::from_int(2, 2, 2),
        bounds_max: Vec3Fix::from_int(6, 6, 6),
    };
    // Contact placed exactly at the bounds_min corner (2,2,0) with slice
    // z=0, dist=0, falloff=1.0.
    let contacts = [(
        Vec3Fix::from_int(2, 2, 0),
        Vec3Fix::UNIT_Y,
        Fix128::from_int(42),
    )];
    let map = generate_stress_heatmap(&[], &contacts, &config);
    assert_eq!(map.data.len(), 1);
    assert_eq!(map.data[0], Fix128::from_int(42));
}

/// Degenerate: zero-configured resolution clamps to 1 (`.max(1)` in the
/// source), it does not produce an empty buffer or panic.
#[test]
fn generate_stress_heatmap_zero_resolution_clamps_to_one() {
    let config = HeatmapConfig {
        resolution_x: 0,
        resolution_y: 0,
        slice_axis: SliceAxis::Z,
        slice_offset: Fix128::ZERO,
        bounds_min: Vec3Fix::from_int(0, 0, 0),
        bounds_max: Vec3Fix::from_int(10, 10, 10),
    };
    let map = generate_stress_heatmap(&[], &[], &config);
    assert_eq!(map.width, 1);
    assert_eq!(map.height, 1);
    assert_eq!(map.data.len(), 1);
}

/// Extreme magnitude: Fix128 arithmetic is wrapping (confirmed against
/// `src/math.rs`'s `overflowing_add`/`wrapping_add` in `impl Add for
/// Fix128`), so summing two `i64::MAX`-magnitude forces at the same point
/// (falloff=1.0 exactly, since dist=0) wraps the integer half exactly like
/// plain `i64::MAX.wrapping_add(i64::MAX)` = -2, rather than panicking or
/// saturating. Verified not to panic via `catch_unwind`.
#[test]
fn generate_stress_heatmap_extreme_magnitude_wraps_without_panic() {
    let huge = Fix128::from_int(i64::MAX);
    let contacts = [
        (Vec3Fix::ZERO, Vec3Fix::UNIT_Y, huge),
        (Vec3Fix::ZERO, Vec3Fix::UNIT_Y, huge),
    ];
    let config = HeatmapConfig {
        resolution_x: 1,
        resolution_y: 1,
        slice_axis: SliceAxis::Z,
        slice_offset: Fix128::ZERO,
        bounds_min: Vec3Fix::ZERO,
        bounds_max: Vec3Fix::ZERO,
    };
    let result = catch_unwind(AssertUnwindSafe(|| {
        generate_stress_heatmap(&[], &contacts, &config)
    }));
    let map = result.expect("extreme-magnitude accumulation must not panic (Fix128 wraps)");
    assert_eq!(map.data[0], Fix128::from_raw(-2, 0));
}

// ============================================================================
// generate_temperature_heatmap
// ============================================================================

/// Closed form over a 2D gradient field; cross-checks the per-axis mapping
/// used by `slice_axis_x_fixes_x_varies_y_and_z` with a combined field.
#[test]
fn generate_temperature_heatmap_closed_form_gradient() {
    let config = HeatmapConfig {
        resolution_x: 4,
        resolution_y: 2,
        slice_axis: SliceAxis::X,
        slice_offset: Fix128::from_int(5),
        bounds_min: Vec3Fix::from_int(0, 0, 0),
        bounds_max: Vec3Fix::from_int(8, 8, 8),
    };
    let map = generate_temperature_heatmap(|p| p.y + p.z, &config);
    assert_eq!(map.data, [0, 2, 4, 6, 4, 6, 8, 10].map(Fix128::from_int));
    assert_eq!(map.min_value, Fix128::ZERO);
    assert_eq!(map.max_value, Fix128::from_int(10));
}

/// Degenerate: zero resolution clamps to 1x1, the single sample lands at the
/// bounds_min corner (u=v=0), not an average over the box.
#[test]
fn generate_temperature_heatmap_zero_resolution_clamps_to_one() {
    let config = HeatmapConfig {
        resolution_x: 0,
        resolution_y: 0,
        slice_axis: SliceAxis::Z,
        slice_offset: Fix128::ZERO,
        bounds_min: Vec3Fix::from_int(3, 3, 3),
        bounds_max: Vec3Fix::from_int(9, 9, 9),
    };
    let map = generate_temperature_heatmap(|p| p.x, &config);
    assert_eq!(map.width, 1);
    assert_eq!(map.height, 1);
    assert_eq!(map.data[0], Fix128::from_int(3));
}

/// Degenerate: a slice_offset far outside bounds_min/bounds_max is not
/// clamped anywhere in `sample_point` — the slice plane simply sits outside
/// the box without panicking. Verified via `catch_unwind`.
#[test]
fn generate_temperature_heatmap_slice_offset_outside_bounds_does_not_panic() {
    let config = HeatmapConfig {
        resolution_x: 2,
        resolution_y: 1,
        slice_axis: SliceAxis::Y,
        slice_offset: Fix128::from_int(1_000),
        bounds_min: Vec3Fix::from_int(0, 0, 0),
        bounds_max: Vec3Fix::from_int(10, 10, 10),
    };
    let result = catch_unwind(AssertUnwindSafe(|| {
        generate_temperature_heatmap(|p| p.y, &config)
    }));
    let map = result.expect("out-of-bounds slice_offset must not panic");
    assert!(map.data.iter().all(|&v| v == Fix128::from_int(1_000)));
}

/// Degenerate: inverted bounds (bounds_min > bounds_max on the varying
/// axes) is not an error either — the lerp just runs in the opposite
/// direction, since `bmin + (bmax - bmin) * u` makes no ordering
/// assumption.
#[test]
fn generate_temperature_heatmap_inverted_bounds_runs_in_reverse() {
    let config = HeatmapConfig {
        resolution_x: 4,
        resolution_y: 1,
        slice_axis: SliceAxis::Z,
        slice_offset: Fix128::ZERO,
        bounds_min: Vec3Fix::from_int(4, 0, 0),
        bounds_max: Vec3Fix::from_int(0, 0, 0),
    };
    // u in {0,.25,.5,.75} -> x = 4 + (0-4)*u = 4 - 4u in {4,3,2,1}
    let map = generate_temperature_heatmap(|p| p.x, &config);
    assert_eq!(map.data, [4, 3, 2, 1].map(Fix128::from_int));
}

// ============================================================================
// heatmap_to_rgba / viridis colormap
// ============================================================================

/// Exact colormap stop fractions (0, .25, .5, .75, 1.0) via a manually
/// constructed `Heatmap` (every field is `pub`, no generator function
/// needed). Also pins the `segment.min(3.999)` undershoot at t=1.0.
#[test]
fn heatmap_to_rgba_exact_stops() {
    let map = Heatmap {
        data: [0, 1, 2, 3, 4].map(Fix128::from_int).to_vec(),
        width: 5,
        height: 1,
        min_value: Fix128::from_int(0),
        max_value: Fix128::from_int(4),
    };
    let rgba = heatmap_to_rgba(&map);
    assert_eq!(rgba.len(), 5);
    for (i, t) in [0.0, 0.25, 0.5, 0.75, 1.0].into_iter().enumerate() {
        assert_eq!(rgba[i], expected_viridis(t), "stop {i} (t={t})");
    }
    // The t=1.0 byte triple undershoots a naive lerp straight to
    // STOPS[4]=(0.993,0.906,0.144) because `segment` is clamped to 3.999,
    // not 4.0, to keep `idx+1` in bounds for a 5-element table.
    assert_eq!(rgba[4], [253, 230, 36, 255]);
    let naive_stop4 = [
        (STOPS[4].0 * 255.0) as u8,
        (STOPS[4].1 * 255.0) as u8,
        (STOPS[4].2 * 255.0) as u8,
        255,
    ];
    assert_ne!(
        rgba[4], naive_stop4,
        "the 3.999 clamp must produce a value distinct from the literal STOPS[4] byte triple"
    );
}

/// Alpha is always 255 regardless of value.
#[test]
fn heatmap_to_rgba_alpha_always_255() {
    let map = Heatmap {
        data: [0, 1, 2, 3, 4].map(Fix128::from_int).to_vec(),
        width: 5,
        height: 1,
        min_value: Fix128::from_int(0),
        max_value: Fix128::from_int(4),
    };
    for px in heatmap_to_rgba(&map) {
        assert_eq!(px[3], 255);
    }
}

/// Degenerate: a value outside [min_value, max_value] (constructible only
/// because every `Heatmap` field is `pub` — a generator function could
/// never produce this, since min/max are derived from the same data) is
/// clamped to t in [0,1], not panicking or wrapping.
#[test]
fn heatmap_to_rgba_out_of_range_value_clamps() {
    let map = Heatmap {
        data: vec![Fix128::from_int(-50), Fix128::from_int(200)],
        width: 2,
        height: 1,
        min_value: Fix128::ZERO,
        max_value: Fix128::from_int(100),
    };
    let result = catch_unwind(AssertUnwindSafe(|| heatmap_to_rgba(&map)));
    let rgba = result.expect("out-of-range value must not panic");
    // (-50-0)/100 = -0.5 -> clamp(0,1) -> 0.0
    assert_eq!(rgba[0], expected_viridis(0.0));
    // (200-0)/100 = 2.0 -> clamp(0,1) -> 1.0
    assert_eq!(rgba[1], expected_viridis(1.0));
}

/// Degenerate: a uniform field (min_value == max_value, `has_range` false)
/// maps every pixel to t=0.5 (the source's explicit fallback), not a
/// division-by-zero panic.
#[test]
fn heatmap_to_rgba_uniform_field_pins_half() {
    let map = Heatmap {
        data: vec![Fix128::from_int(7); 3],
        width: 3,
        height: 1,
        min_value: Fix128::from_int(7),
        max_value: Fix128::from_int(7),
    };
    let rgba = heatmap_to_rgba(&map);
    for px in rgba {
        assert_eq!(px, expected_viridis(0.5));
    }
}

/// Degenerate: empty heatmap returns an empty Vec, not a panic.
#[test]
fn heatmap_to_rgba_empty_heatmap_returns_empty() {
    let map = Heatmap {
        data: vec![],
        width: 0,
        height: 0,
        min_value: Fix128::ZERO,
        max_value: Fix128::ZERO,
    };
    let result = catch_unwind(AssertUnwindSafe(|| heatmap_to_rgba(&map)));
    let rgba = result.expect("empty heatmap must not panic");
    assert!(rgba.is_empty());
}

/// Negative-valued field (min and max both negative) is handled the same
/// way as a positive-valued one: `t` depends only on the relative position
/// within [min,max], not on sign.
#[test]
fn heatmap_to_rgba_negative_range() {
    let map = Heatmap {
        data: vec![
            Fix128::from_int(-10),
            Fix128::from_int(-5),
            Fix128::from_int(0),
        ],
        width: 3,
        height: 1,
        min_value: Fix128::from_int(-10),
        max_value: Fix128::from_int(0),
    };
    let rgba = heatmap_to_rgba(&map);
    assert_eq!(rgba[0], expected_viridis(0.0)); // (-10-(-10))/10 = 0.0
    assert_eq!(rgba[1], expected_viridis(0.5)); // (-5-(-10))/10 = 0.5
    assert_eq!(rgba[2], expected_viridis(1.0)); // (0-(-10))/10 = 1.0
}
