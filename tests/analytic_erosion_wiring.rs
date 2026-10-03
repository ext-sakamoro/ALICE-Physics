//! Oracles for `ErosionModifier::{set_exposure_at, compute_exposure_from_normals,
//! erosion_at}` (previously unwired) and the rate law the module documents:
//!
//! ```text
//! depth += rate (1 - hardness) v^n * exposure * dt      capped at max_depth
//! Wind n = 1 (x1)   Water n = 1 (x1.5)   Chemical n = 0   Ablation n = 2
//! exposure decays as exp(-5 dt) after every update
//! splat weight w(d) = s^2 (3 - 2 s), s = 1 - d / radius                  (smoothstep)
//! windward exposure = max(-n . flow_hat, 0) on the surface band
//! ```
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::erosion::{
    ErosionConfig, ErosionModifier, ErosionType, EXPOSURE_DECAY_PER_S, WATER_PREFACTOR,
};
use alice_physics::sdf_collider::ClosureSdf;
use alice_physics::sim_modifier::PhysicsModifier;

fn config(kind: ErosionType) -> ErosionConfig {
    ErosionConfig {
        erosion_type: kind,
        rate: 0.2,
        hardness: 0.5,
        max_depth: 2.0,
        smoothing: 0.0,
        flow_direction: (1.0, 0.0, 0.0),
        flow_speed: 3.0,
    }
}

/// 9^3 nodes on (-4, 4)^3: node spacing exactly 1
fn modifier(cfg: ErosionConfig) -> ErosionModifier {
    ErosionModifier::new(cfg, 9, (-4.0, -4.0, -4.0), (4.0, 4.0, 4.0))
}

fn weight(d: f32, r: f32) -> f32 {
    if d >= r {
        return 0.0;
    }
    let s = 1.0 - d / r;
    s * s * (3.0 - 2.0 * s)
}

fn sphere(radius: f32) -> ClosureSdf {
    ClosureSdf::new(
        move |x, y, z| (x * x + y * y + z * z).sqrt() - radius,
        |x, y, z| {
            let l = (x * x + y * y + z * z).sqrt();
            if l < 1e-10 {
                (0.0, 1.0, 0.0)
            } else {
                (x / l, y / l, z / l)
            }
        },
    )
}

// ---------------------------------------------------------------- exposure

#[test]
fn set_exposure_at_splats_a_smoothstep_blob_and_accumulates() {
    let mut m = modifier(config(ErosionType::Wind));
    m.set_exposure_at(0.0, 0.0, 0.0, 0.8, 2.0);
    for (x, y, z) in [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (1, 1, 1),
        (2, 0, 0),
        (0, 3, 0),
    ] {
        let d = ((x * x + y * y + z * z) as f32).sqrt();
        let got = m
            .exposure
            .get((x + 4) as usize, (y + 4) as usize, (z + 4) as usize);
        assert!(
            (got - 0.8 * weight(d, 2.0)).abs() < 1e-5,
            "({x},{y},{z}): {got}"
        );
    }
    // a second call adds
    m.set_exposure_at(0.0, 0.0, 0.0, 0.2, 2.0);
    assert!((m.exposure.get(4, 4, 4) - 1.0).abs() < 1e-5);
    // negative exposure removes (the field is a plain additive splat)
    m.set_exposure_at(0.0, 0.0, 0.0, -1.0, 2.0);
    assert!(m.exposure.get(4, 4, 4).abs() < 1e-5);
}

#[test]
fn windward_faces_of_a_sphere_get_exposure_one_and_leeward_faces_zero() {
    let mut m = modifier(config(ErosionType::Wind)); // flow +x
    m.compute_exposure_from_normals(&sphere(2.0), 0.5);
    let at = |x: i32, y: i32, z: i32| {
        m.exposure
            .get((x + 4) as usize, (y + 4) as usize, (z + 4) as usize)
    };
    assert!(
        (at(-2, 0, 0) - 1.0).abs() < 1e-6,
        "windward pole faces the flow: {}",
        at(-2, 0, 0)
    );
    assert_eq!(at(2, 0, 0), 0.0, "leeward pole");
    assert_eq!(
        at(0, 2, 0),
        0.0,
        "grazing: normal perpendicular to the flow"
    );
    assert_eq!(at(0, 0, 2), 0.0);
    // far from the surface (|phi| = 2 > threshold): untouched
    assert_eq!(at(0, 0, 0), 0.0);
    assert_eq!(at(4, 4, 4), 0.0);
    // exposure = max(-n.f, 0): at (-1,-1,?) not on a node's surface; use node (-2, 0, 0) only
    // every node's exposure lies in [0, 1] and nonzero ones are windward
    for z in -4..=4 {
        for y in -4i32..=4 {
            for x in -4i32..=4 {
                let e = at(x, y, z);
                assert!((0.0..=1.0 + 1e-6).contains(&e));
                if e > 0.0 {
                    assert!(
                        x < 0,
                        "exposed node ({x},{y},{z}) must be on the upwind side"
                    );
                }
            }
        }
    }
}

#[test]
fn flow_direction_sets_which_side_is_exposed_and_is_normalised() {
    let mut cfg = config(ErosionType::Wind);
    cfg.flow_direction = (-3.0, 0.0, 0.0); // not unit length
    let mut m = modifier(cfg);
    m.compute_exposure_from_normals(&sphere(2.0), 0.5);
    assert!(
        (m.exposure.get(6, 4, 4) - 1.0).abs() < 1e-6,
        "flow -x exposes the +x pole"
    );
    assert_eq!(m.exposure.get(2, 4, 4), 0.0);
    // a y flow exposes the bottom pole
    let mut cfg_y = config(ErosionType::Wind);
    cfg_y.flow_direction = (0.0, 2.0, 0.0);
    let mut my = modifier(cfg_y);
    my.compute_exposure_from_normals(&sphere(2.0), 0.5);
    assert!((my.exposure.get(4, 2, 4) - 1.0).abs() < 1e-6);
    assert_eq!(my.exposure.get(4, 6, 4), 0.0);
}

#[test]
fn z_and_oblique_flows_expose_the_matching_poles_and_recomputation_overwrites() {
    let mut cfg = config(ErosionType::Wind);
    cfg.flow_direction = (0.0, 0.0, 1.0);
    let mut m = modifier(cfg);
    m.exposure.set(4, 4, 4, 0.0);
    m.exposure.set(4, 4, 2, 5.0); // stale value on a band node
    m.compute_exposure_from_normals(&sphere(2.0), 0.5);
    assert!(
        (m.exposure.get(4, 4, 2) - 1.0).abs() < 1e-6,
        "z = -2 pole faces a +z flow; the old 5.0 is overwritten"
    );
    assert_eq!(m.exposure.get(4, 4, 6), 0.0);
    // recomputing is idempotent (assignment, not accumulation)
    m.compute_exposure_from_normals(&sphere(2.0), 0.5);
    assert!((m.exposure.get(4, 4, 2) - 1.0).abs() < 1e-6);
    // an oblique flow (1, 0, 1)/sqrt2 weights the node by n . f: node (-2, 0, 0) has n = -x
    let mut cfg = config(ErosionType::Wind);
    cfg.flow_direction = (1.0, 0.0, 1.0);
    let mut o = modifier(cfg);
    o.compute_exposure_from_normals(&sphere(2.0), 0.5);
    assert!((o.exposure.get(2, 4, 4) - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
    assert!((o.exposure.get(4, 4, 2) - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
}

#[test]
fn cells_without_exposure_are_left_alone_even_above_the_cap() {
    let mut m = modifier(config(ErosionType::Wind));
    m.erosion_depth.set(1, 1, 1, 5.0); // above max_depth = 2, no exposure here
    m.update(0.1);
    assert_eq!(m.erosion_depth.get(1, 1, 1), 5.0);
}

#[test]
fn zero_flow_direction_falls_back_to_plus_x() {
    let mut cfg = config(ErosionType::Wind);
    cfg.flow_direction = (0.0, 0.0, 0.0);
    let mut m = modifier(cfg);
    m.compute_exposure_from_normals(&sphere(2.0), 0.5);
    assert!((m.exposure.get(2, 4, 4) - 1.0).abs() < 1e-6);
}

#[test]
fn threshold_is_strict_and_nodes_outside_it_keep_their_old_exposure() {
    let mut m = modifier(config(ErosionType::Wind));
    m.exposure.set(4, 4, 4, 5.0); // deep inside the sphere, outside any band
    m.compute_exposure_from_normals(&sphere(2.0), 0.5);
    assert_eq!(
        m.exposure.get(4, 4, 4),
        5.0,
        "stale exposure outside the band is left alone"
    );
    // |phi| == threshold is excluded: sphere radius 2.5, node at x = -2 has |phi| = 0.5
    let mut s = modifier(config(ErosionType::Wind));
    s.compute_exposure_from_normals(&sphere(2.5), 0.5);
    assert_eq!(s.exposure.get(2, 4, 4), 0.0);
    // a slightly wider band includes it, with the exact alignment (normal -x => 1)
    let mut w = modifier(config(ErosionType::Wind));
    w.compute_exposure_from_normals(&sphere(2.5), 0.5001);
    assert!((w.exposure.get(2, 4, 4) - 1.0).abs() < 1e-6);
}

// --------------------------------------------------------------- rate law

fn one_step(kind: ErosionType, speed: f32, hardness: f32, exposure: f32, dt: f32) -> (f32, f32) {
    let mut cfg = config(kind);
    cfg.flow_speed = speed;
    cfg.hardness = hardness;
    let mut m = modifier(cfg);
    m.exposure.set(4, 4, 4, exposure);
    m.update(dt);
    (m.erosion_at(0.0, 0.0, 0.0), m.exposure.get(4, 4, 4))
}

#[test]
fn rate_laws_per_type_match_the_documented_table() {
    let (rate, h, dt) = (0.2f32, 0.5f32, 0.1f32);
    let base = rate * (1.0 - h);
    for (kind, speed, expected) in [
        (ErosionType::Wind, 3.0, base * 3.0 * 1.0),
        (ErosionType::Water, 3.0, base * 3.0 * WATER_PREFACTOR),
        (ErosionType::Chemical, 3.0, base),
        (ErosionType::Chemical, 30.0, base),
        (ErosionType::Ablation, 3.0, base * 9.0),
        (ErosionType::Ablation, 6.0, base * 36.0),
    ] {
        let (depth, _) = one_step(kind, speed, h, 1.0, dt);
        assert!(
            (depth - expected * dt).abs() < 1e-6,
            "{kind:?} v = {speed}: {depth} vs {}",
            expected * dt
        );
    }
    assert_eq!(WATER_PREFACTOR, 1.5);
}

#[test]
fn rate_is_linear_in_exposure_and_proportional_to_one_minus_hardness() {
    let dt = 0.1;
    let (half, _) = one_step(ErosionType::Wind, 3.0, 0.5, 0.5, dt);
    let (full, _) = one_step(ErosionType::Wind, 3.0, 0.5, 1.0, dt);
    assert!((full - 2.0 * half).abs() < 1e-7);
    let (soft, _) = one_step(ErosionType::Wind, 3.0, 0.0, 1.0, dt);
    let (hard, _) = one_step(ErosionType::Wind, 3.0, 0.75, 1.0, dt);
    assert!((soft - 2.0 * full).abs() < 1e-6, "hardness 0 vs 0.5");
    assert!((hard - 0.5 * full).abs() < 1e-6, "hardness 0.75 vs 0.5");
    let (rock, _) = one_step(ErosionType::Wind, 3.0, 1.0, 1.0, dt);
    assert_eq!(rock, 0.0, "hardness 1 does not erode");
    let (none, _) = one_step(ErosionType::Wind, 3.0, 0.5, 0.0, dt);
    assert_eq!(none, 0.0, "no exposure, no erosion");
}

#[test]
fn depth_is_capped_at_max_depth_and_exposure_decays_exponentially() {
    let (depth, _) = one_step(ErosionType::Ablation, 1000.0, 0.0, 1.0, 1.0);
    assert_eq!(depth, 2.0);
    let (_, e) = one_step(ErosionType::Wind, 3.0, 0.5, 1.0, 0.1);
    assert!(
        (e - (-EXPOSURE_DECAY_PER_S * 0.1).exp()).abs() < 1e-5,
        "{e}"
    );
    assert_eq!(EXPOSURE_DECAY_PER_S, 5.0);
}

#[test]
fn erosion_accumulates_over_steps_and_follows_the_decaying_exposure() {
    // exposure e0 once, then n steps: depth_n = sum_k rate_base v e0 exp(-5 k dt) dt
    let mut cfg = config(ErosionType::Wind);
    cfg.flow_speed = 2.0;
    let mut m = modifier(cfg);
    m.exposure.set(4, 4, 4, 1.0);
    let dt = 0.05f32;
    let mut want = 0.0f32;
    for k in 0..10 {
        want += 0.2 * 0.5 * 2.0 * (-5.0 * dt * k as f32).exp() * dt;
        m.update(dt);
    }
    assert!(
        (m.erosion_at(0.0, 0.0, 0.0) - want).abs() < 1e-5,
        "{} vs {want}",
        m.erosion_at(0.0, 0.0, 0.0)
    );
}

#[test]
fn smoothing_spreads_erosion_without_changing_its_total() {
    let mut cfg = config(ErosionType::Wind);
    cfg.smoothing = 0.2;
    let mut m = modifier(cfg);
    m.exposure.set(4, 4, 4, 1.0);
    m.update(0.1);
    let total: f32 = m.erosion_depth.data.iter().sum();
    assert!(
        (total - 0.2 * 0.5 * 3.0 * 0.1).abs() < 1e-6,
        "total {total}"
    );
    assert!(m.erosion_at(0.0, 0.0, 0.0) < 0.03, "peak is smoothed");
    assert!(
        m.erosion_at(1.0, 0.0, 0.0) > 0.0,
        "and spread to the neighbours"
    );
}

#[test]
fn erosion_at_samples_trilinearly_and_modify_distance_adds_it() {
    let mut m = modifier(config(ErosionType::Wind));
    m.erosion_depth.set(4, 4, 4, 1.0);
    m.erosion_depth.set(5, 4, 4, 0.5);
    assert_eq!(m.erosion_at(0.0, 0.0, 0.0), 1.0);
    assert!((m.erosion_at(0.5, 0.0, 0.0) - 0.75).abs() < 1e-6);
    assert!((m.modify_distance(0.5, 0.0, 0.0, -2.0) - (-2.0 + 0.75)).abs() < 1e-6);
    m.enabled = false;
    assert_eq!(m.modify_distance(0.5, 0.0, 0.0, -2.0), -2.0);
    assert!(!m.is_active());
    m.exposure.set(4, 4, 4, 1.0);
    m.update(1.0);
    assert_eq!(
        m.erosion_at(0.0, 0.0, 0.0),
        1.0,
        "disabled: update is a no-op"
    );
    assert_eq!(
        m.exposure.get(4, 4, 4),
        1.0,
        "and the exposure does not decay"
    );
    assert_eq!(m.name(), "erosion");
}
