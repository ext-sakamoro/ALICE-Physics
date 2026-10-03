//! Heatmap visualization wiring: stress/temperature slice sampling + RGBA
//!
//! Exercises every previously-unwired item in `heatmap`:
//! - `HeatmapConfig` / `SliceAxis` to describe a 2D sampling plane through
//!   the 3D domain
//! - `generate_stress_heatmap`, run over a known set of contact points, each
//!   pixel checked against a hand-derived radial-falloff sum (read from
//!   `src/heatmap.rs`'s `influence_radius = 2` kernel, not from running the
//!   function and trusting its own output)
//! - `generate_temperature_heatmap`, run over a synthetic `Fn(Vec3Fix) ->
//!   Fix128` field, each pixel checked against the closed-form slice-plane
//!   formula (`sample_point`'s per-axis mapping, re-derived by hand here)
//! - `Heatmap` constructed directly (all fields `pub`) to probe
//!   `heatmap_to_rgba`'s clamp behavior on out-of-range and degenerate data
//! - `heatmap_to_rgba`, checked against an independently re-implemented
//!   5-stop viridis lerp (not by calling `heatmap_to_rgba` for the expected
//!   side)
//!
//! `generate_stress_heatmap` takes raw `(point, normal, force)` contact
//! triples, not a `FemSolution`/per-element stress tensor — so there is no
//! existing FEM solve to feed it directly (`linear_elastic_fem::FemSolution`
//! stores a stress tensor *per tetrahedral element*, not per world-space
//! point, and `heatmap`'s own module doc says the stress path is "contact
//! forces", not FEM). A synthetic, hand-placed contact set is the only
//! input shape the signature accepts, matching the house style of
//! `debug_render_primitives.rs` Part 1 (direct, closed-form-checked
//! construction rather than driving a solver).
//!
//! ```bash
//! cargo run --example heatmap_visualization --features std
//! ```

use alice_physics::heatmap::{
    generate_stress_heatmap, generate_temperature_heatmap, heatmap_to_rgba, Heatmap, HeatmapConfig,
    SliceAxis,
};
use alice_physics::math::{Fix128, Vec3Fix};

fn main() {
    // --- Part 1: generate_stress_heatmap -----------------------------
    //
    // 4x1 slice along Z, sampling x in {0,1,2,3} (bounds 0..4, u = ix/4).
    // One contact sits exactly at the origin with force = 100.
    //
    // Closed form (read from `src/heatmap.rs`): for each sample point p,
    //   dist_sq = |p - contact_point|^2
    //   if dist_sq < radius_sq (radius = 2, radius_sq = 4):
    //       stress += force * (1 - sqrt(dist_sq) / radius)
    //
    //   x=0: dist=0 -> falloff=1-0/2=1.0   -> 100 * 1.0  = 100
    //   x=1: dist=1 -> falloff=1-1/2=0.5   -> 100 * 0.5  = 50
    //   x=2: dist_sq=4, NOT < radius_sq=4 (strict <)        -> 0
    //   x=3: dist_sq=9, NOT < radius_sq=4                   -> 0
    let stress_config = HeatmapConfig {
        resolution_x: 4,
        resolution_y: 1,
        slice_axis: SliceAxis::Z,
        slice_offset: Fix128::ZERO,
        bounds_min: Vec3Fix::from_int(0, 0, 0),
        bounds_max: Vec3Fix::from_int(4, 0, 0),
    };
    let contacts = [(Vec3Fix::ZERO, Vec3Fix::UNIT_Y, Fix128::from_int(100))];
    let stress_map = generate_stress_heatmap(&[], &contacts, &stress_config);

    let expected_stress = [
        Fix128::from_int(100),
        Fix128::from_ratio(1, 2) * Fix128::from_int(100), // 50
        Fix128::ZERO,
        Fix128::ZERO,
    ];
    assert_eq!(stress_map.width, 4);
    assert_eq!(stress_map.height, 1);
    assert_eq!(stress_map.data.len(), 4);
    for (i, (&got, &want)) in stress_map
        .data
        .iter()
        .zip(expected_stress.iter())
        .enumerate()
    {
        assert_eq!(got, want, "stress pixel {i} mismatch");
    }
    assert_eq!(stress_map.min_value, Fix128::ZERO);
    assert_eq!(stress_map.max_value, Fix128::from_int(100));
    println!(
        "[heatmap] stress: {:?} (min={:?} max={:?}) — radial falloff at exact boundary (x=2, dist_sq==radius_sq) excluded by strict `<`",
        stress_map.data.iter().map(|v| v.to_f64()).collect::<Vec<_>>(),
        stress_map.min_value.to_f64(),
        stress_map.max_value.to_f64()
    );

    // A second contact at the same point accumulates (does not overwrite).
    let two_contacts = [
        (Vec3Fix::ZERO, Vec3Fix::UNIT_Y, Fix128::from_int(100)),
        (Vec3Fix::ZERO, Vec3Fix::UNIT_Y, Fix128::from_int(10)),
    ];
    let stress_map2 = generate_stress_heatmap(&[], &two_contacts, &stress_config);
    // x=0: (100+10)*1.0 = 110, x=1: (100+10)*0.5 = 55
    assert_eq!(stress_map2.data[0], Fix128::from_int(110));
    assert_eq!(
        stress_map2.data[1],
        Fix128::from_ratio(1, 2) * Fix128::from_int(110)
    );
    println!(
        "[heatmap] stress accumulation: x=0 -> {} (100+10 at full falloff)",
        stress_map2.data[0].to_f64()
    );

    // --- Part 2: generate_temperature_heatmap ------------------------
    //
    // 4x2 slice along X at slice_offset=5, bounds (0,0,0)..(8,8,8).
    // temperature_fn = |p| p.y + p.z.
    //
    // Closed form (SliceAxis::X fixes x = slice_offset; y,z vary):
    //   u = ix/4 in {0, .25, .5, .75} -> y = 8*u in {0,2,4,6}
    //   v = iy/2 in {0, .5}           -> z = 8*v in {0,4}
    //   temp = y + z
    //
    // row iy=0 (z=0): [0, 2, 4, 6]
    // row iy=1 (z=4): [4, 6, 8, 10]
    let temp_config = HeatmapConfig {
        resolution_x: 4,
        resolution_y: 2,
        slice_axis: SliceAxis::X,
        slice_offset: Fix128::from_int(5),
        bounds_min: Vec3Fix::from_int(0, 0, 0),
        bounds_max: Vec3Fix::from_int(8, 8, 8),
    };
    let temp_map = generate_temperature_heatmap(|p| p.y + p.z, &temp_config);
    let expected_temp: [i64; 8] = [0, 2, 4, 6, 4, 6, 8, 10];
    for (i, (&got, &want)) in temp_map.data.iter().zip(expected_temp.iter()).enumerate() {
        assert_eq!(
            got,
            Fix128::from_int(want),
            "temperature pixel {i} mismatch"
        );
    }
    assert_eq!(temp_map.min_value, Fix128::ZERO);
    assert_eq!(temp_map.max_value, Fix128::from_int(10));
    println!(
        "[heatmap] temperature: {:?} (min={:?} max={:?})",
        temp_map.data.iter().map(|v| v.to_f64()).collect::<Vec<_>>(),
        temp_map.min_value.to_f64(),
        temp_map.max_value.to_f64(),
    );

    // Independent check (separate call, `temperature_fn = |p| p.x`) that
    // every sample sits exactly on the X=5 slice plane: SliceAxis::X fixes
    // x = slice_offset, independent of u/v.
    let x_plane_map = generate_temperature_heatmap(|p| p.x, &temp_config);
    assert!(x_plane_map.data.iter().all(|&x| x == Fix128::from_int(5)));
    println!(
        "[heatmap] all {} samples pinned to slice plane x=5 (SliceAxis::X)",
        x_plane_map.data.len()
    );

    // --- Part 3: heatmap_to_rgba --------------------------------------
    //
    // Independent re-implementation of the documented 5-stop viridis lerp
    // (STOPS table copied verbatim from the doc comment in `heatmap.rs`,
    // not obtained by calling `heatmap_to_rgba`/`viridis_color`):
    //   STOPS = [(.267,.004,.329), (.282,.140,.458), (.127,.566,.551),
    //            (.544,.774,.247), (.993,.906,.144)]
    //   segment = min(t*4, 3.999); idx = floor(segment); frac = segment-idx
    //   channel = lerp(STOPS[idx], STOPS[idx+1], frac) * 255 (truncated to u8)
    fn expected_viridis(t: f64) -> [u8; 4] {
        const STOPS: [(f64, f64, f64); 5] = [
            (0.267, 0.004, 0.329),
            (0.282, 0.140, 0.458),
            (0.127, 0.566, 0.551),
            (0.544, 0.774, 0.247),
            (0.993, 0.906, 0.144),
        ];
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

    // stress_map has data [100, 50, 0, 0], min=0, max=100 -> t in {1.0, 0.5, 0.0, 0.0}.
    let stress_rgba = heatmap_to_rgba(&stress_map);
    assert_eq!(stress_rgba.len(), 4);
    assert_eq!(stress_rgba[0], expected_viridis(1.0)); // val=100 -> t=1.0
    assert_eq!(stress_rgba[1], expected_viridis(0.5)); // val=50  -> t=0.5
    assert_eq!(stress_rgba[2], expected_viridis(0.0)); // val=0   -> t=0.0
    assert_eq!(stress_rgba[3], expected_viridis(0.0));
    // t=1.0 undershoots the literal STOPS[4] value because `segment` is
    // clamped to 3.999 (not 4.0) to keep `idx+1` in bounds: the exact byte
    // triple is (253, 230, 36), one less in G than a naive lerp to
    // STOPS[4]=(0.993,0.906,0.144) (0.906*255 truncates to 231) would give.
    assert_eq!(stress_rgba[0], [253, 230, 36, 255]);
    println!(
        "[heatmap] rgba(t=1.0)={:?} (STOPS[4] undershoot from the 3.999 clamp)",
        stress_rgba[0]
    );
    println!("[heatmap] rgba: stress pixels -> {stress_rgba:?}");

    // Manually constructed Heatmap (every field is `pub`) to probe
    // heatmap_to_rgba's clamp/degenerate paths independent of either
    // generator function:
    //   (a) exact stop fractions 0, .25, .5, .75, 1.0 via data [0,1,2,3,4]
    //   (b) a value *outside* [min_value,max_value] (inconsistent by
    //       construction) clamps rather than panicking or wrapping
    //   (c) a uniform field (min==max, has_range=false) maps to t=0.5
    //   (d) an empty heatmap returns an empty Vec, no panic
    let exact_stops = Heatmap {
        data: vec![
            Fix128::from_int(0),
            Fix128::from_int(1),
            Fix128::from_int(2),
            Fix128::from_int(3),
            Fix128::from_int(4),
        ],
        width: 5,
        height: 1,
        min_value: Fix128::from_int(0),
        max_value: Fix128::from_int(4),
    };
    let exact_rgba = heatmap_to_rgba(&exact_stops);
    for (i, t) in [0.0, 0.25, 0.5, 0.75, 1.0].into_iter().enumerate() {
        assert_eq!(exact_rgba[i], expected_viridis(t), "exact-stop pixel {i}");
    }
    println!("[heatmap] rgba: exact-stop pixels -> {exact_rgba:?}");

    let out_of_range = Heatmap {
        data: vec![Fix128::from_int(-50), Fix128::from_int(200)],
        width: 2,
        height: 1,
        min_value: Fix128::ZERO,
        max_value: Fix128::from_int(100),
    };
    let oor_rgba = heatmap_to_rgba(&out_of_range);
    // (-50 - 0) / 100 = -0.5 -> clamp(0,1) -> 0.0
    // (200 - 0) / 100 = 2.0  -> clamp(0,1) -> 1.0
    assert_eq!(
        oor_rgba[0],
        expected_viridis(0.0),
        "below-range value clamps to t=0"
    );
    assert_eq!(
        oor_rgba[1],
        expected_viridis(1.0),
        "above-range value clamps to t=1"
    );
    println!("[heatmap] rgba: out-of-range inputs clamp (no panic) -> {oor_rgba:?}");

    let uniform = Heatmap {
        data: vec![Fix128::from_int(7); 3],
        width: 3,
        height: 1,
        min_value: Fix128::from_int(7),
        max_value: Fix128::from_int(7),
    };
    let uniform_rgba = heatmap_to_rgba(&uniform);
    for px in &uniform_rgba {
        assert_eq!(*px, expected_viridis(0.5), "zero-range field pins t=0.5");
    }
    println!("[heatmap] rgba: uniform field (min==max) -> t=0.5 everywhere -> {uniform_rgba:?}");

    let empty = Heatmap {
        data: vec![],
        width: 0,
        height: 0,
        min_value: Fix128::ZERO,
        max_value: Fix128::ZERO,
    };
    let empty_rgba = heatmap_to_rgba(&empty);
    assert!(empty_rgba.is_empty());
    println!("[heatmap] rgba: empty heatmap -> empty Vec, no panic");

    // --- Part 4: degenerate sampling configs --------------------------
    //
    // resolution_x/resolution_y of 0 clamp to 1 (`.max(1)` in the source),
    // so a "zero-resolution" request still returns a single sample at
    // u=v=0 (the bounds_min corner) rather than an empty/degenerate buffer.
    let zero_res_config = HeatmapConfig {
        resolution_x: 0,
        resolution_y: 0,
        slice_axis: SliceAxis::Z,
        slice_offset: Fix128::ZERO,
        bounds_min: Vec3Fix::from_int(3, 3, 3),
        bounds_max: Vec3Fix::from_int(9, 9, 9),
    };
    let zero_res_map = generate_temperature_heatmap(|p| p.x, &zero_res_config);
    assert_eq!(zero_res_map.width, 1);
    assert_eq!(zero_res_map.height, 1);
    assert_eq!(zero_res_map.data.len(), 1);
    // u=0/1=0 -> x = bounds_min.x = 3 exactly (the single sample is the
    // bounds_min corner, not an average or a midpoint).
    assert_eq!(zero_res_map.data[0], Fix128::from_int(3));
    println!(
        "[heatmap] zero-resolution config clamps to 1x1, single sample at bounds_min corner: {}",
        zero_res_map.data[0].to_f64()
    );

    // slice_offset far outside bounds_min/bounds_max: nothing in the source
    // clamps slice_offset to the bounds box, so the slice plane simply sits
    // outside the visually-intended region without panicking.
    let outside_slice_config = HeatmapConfig {
        resolution_x: 2,
        resolution_y: 1,
        slice_axis: SliceAxis::Y,
        slice_offset: Fix128::from_int(1_000),
        bounds_min: Vec3Fix::from_int(0, 0, 0),
        bounds_max: Vec3Fix::from_int(10, 10, 10),
    };
    let outside_slice_map = generate_temperature_heatmap(|p| p.y, &outside_slice_config);
    assert!(outside_slice_map
        .data
        .iter()
        .all(|&v| v == Fix128::from_int(1_000)));
    println!(
        "[heatmap] slice_offset outside bounds (1000 vs bounds [0,10]) is not clamped: sampled y={}",
        outside_slice_map.data[0].to_f64()
    );

    // --- Part 5: extreme-magnitude wraparound (no panic, deterministic) --
    //
    // Fix128's Add/Sub/Mul are all wrapping (confirmed against
    // `src/math.rs`'s `wrapping_add`/`wrapping_mul` impls, not assumed):
    // two contacts at the same point each contributing force=i64::MAX at
    // falloff=1.0 wrap the i64 `hi` half exactly like plain i64 wrapping
    // addition: MAX + MAX = (2^63-1)*2 = 2^64-2, which as a signed i64
    // wraps to -2. No panic, no NaN, no saturation.
    let huge_force = Fix128::from_int(i64::MAX);
    let extreme_contacts = [
        (Vec3Fix::ZERO, Vec3Fix::UNIT_Y, huge_force),
        (Vec3Fix::ZERO, Vec3Fix::UNIT_Y, huge_force),
    ];
    let extreme_config = HeatmapConfig {
        resolution_x: 1,
        resolution_y: 1,
        slice_axis: SliceAxis::Z,
        slice_offset: Fix128::ZERO,
        bounds_min: Vec3Fix::ZERO,
        bounds_max: Vec3Fix::ZERO,
    };
    let extreme_map = generate_stress_heatmap(&[], &extreme_contacts, &extreme_config);
    assert_eq!(extreme_map.data[0], Fix128::from_raw(-2, 0));
    println!(
        "[heatmap] extreme magnitude (i64::MAX + i64::MAX) wraps deterministically to {:?} (no panic)",
        extreme_map.data[0]
    );

    println!("[heatmap] all 6 wiring targets exercised: Heatmap, HeatmapConfig, SliceAxis, generate_stress_heatmap, generate_temperature_heatmap, heatmap_to_rgba");
}
