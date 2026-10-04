//! Audit oracles for `alice_physics::pressure`.
//!
//! Grid: 5^3 nodes over [-2, 2]^3, so the cell size is exactly 1.0 and the
//! world origin is grid node (2, 2, 2) with six interior face neighbours.
//! Expected values are hand-derived: one explicit-Euler diffusion step of a
//! point spike, the exponential decay factor, and the yield law
//! `delta = (p - threshold) * rate * dt`.

#![allow(clippy::disallowed_methods)]

use alice_physics::pressure::{PressureConfig, PressureModifier};
use alice_physics::sim_modifier::PhysicsModifier;

fn modifier(config: PressureConfig) -> PressureModifier {
    PressureModifier::new(config, 5, (-2.0, -2.0, -2.0), (2.0, 2.0, 2.0))
}

fn rel_ok(got: f32, want: f64, tol: f64) -> bool {
    let g = f64::from(got);
    (g - want).abs() <= tol * want.abs().max(1.0e-12)
}

/// Pressure the spike leaves in the centre node and each face neighbour after
/// one `diffuse(dt, rate)` step: centre `P (1 - 6k)`, neighbour `P k`.
fn spike_config() -> PressureConfig {
    PressureConfig {
        diffusion_rate: 0.2,
        decay_rate: 1.5,
        yield_threshold: 10.0,
        deformation_rate: 0.5,
        max_deformation: 100.0,
        internal_pressure: 0.0,
        expansion_rate: 0.01,
    }
}

#[test]
fn update_of_a_point_spike_matches_hand_derived_diffusion_decay_and_yield() {
    let mut m = modifier(spike_config());
    m.apply_pressure_at(0.0, 0.0, 0.0, 100.0, 0.5);
    assert_eq!(m.pressure_at(0.0, 0.0, 0.0), 100.0);
    let dt = 0.1_f32;
    m.update(dt);

    // yield uses the pre-diffusion field: (100 - 10) * 0.5 * 0.1 = 4.5 at the
    // centre node only
    assert!(rel_ok(m.deformation_at(0.0, 0.0, 0.0), 4.5, 1.0e-6));
    for p in [
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    ] {
        assert_eq!(m.deformation_at(p.0, p.1, p.2), 0.0, "neighbour {p:?}");
    }

    // diffusion: k = rate * dt / h^2 = 0.02; then decay exp(-1.5 * 0.1)
    let k = 0.2_f64 * 0.1;
    let decay = (-1.5_f64 * 0.1).exp();
    let centre = 100.0 * (1.0 - 6.0 * k) * decay;
    let neigh = 100.0 * k * decay;
    assert!(
        rel_ok(m.pressure_at(0.0, 0.0, 0.0), centre, 2.0e-6),
        "centre {} vs {centre}",
        m.pressure_at(0.0, 0.0, 0.0)
    );
    for p in [
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    ] {
        assert!(
            rel_ok(m.pressure_at(p.0, p.1, p.2), neigh, 2.0e-6),
            "neighbour {p:?}: {} vs {neigh}",
            m.pressure_at(p.0, p.1, p.2)
        );
    }
    // diagonal nodes are two hops away and stay empty after one step
    assert_eq!(m.pressure_at(1.0, 1.0, 0.0), 0.0);
}

#[test]
fn uniform_pressure_follows_the_closed_form_yield_and_decay() {
    for &(p0, thr, rate, dt, max_d, decay) in &[
        (50.0_f32, 10.0_f32, 0.05_f32, 0.016_f32, 1.0_f32, 0.5_f32),
        (30.0, 5.0, 0.2, 0.25, 1.0, 2.0),
        (200.0, 10.0, 0.5, 0.5, 3.0, 0.0), // reaches the cap
    ] {
        let cfg = PressureConfig {
            decay_rate: decay,
            yield_threshold: thr,
            deformation_rate: rate,
            max_deformation: max_d,
            ..PressureConfig::default()
        };
        let mut m = modifier(cfg);
        m.pressure.data.fill(p0);
        m.update(dt);
        let want_def =
            (f64::from(p0 - thr) * f64::from(rate) * f64::from(dt)).min(f64::from(max_d));
        let want_p = f64::from(p0) * (-f64::from(decay) * f64::from(dt)).exp();
        let got_def = m.deformation_at(0.3, -0.7, 1.1);
        let got_p = m.pressure_at(0.3, -0.7, 1.1);
        assert!(
            rel_ok(got_def, want_def, 1.0e-5),
            "def {got_def} vs {want_def}"
        );
        assert!(rel_ok(got_p, want_p, 1.0e-4), "p {got_p} vs {want_p}");
    }
}

#[test]
fn pressure_at_or_below_the_yield_threshold_leaves_no_permanent_dent() {
    for p0 in [0.0_f32, 4.0, 10.0] {
        let mut m = modifier(PressureConfig::default());
        m.pressure.data.fill(p0);
        for _ in 0..20 {
            m.update(0.05);
        }
        assert!(m.deformation.data.iter().all(|&d| d == 0.0), "p0 {p0}");
    }
}

#[test]
fn deformation_is_permanent_after_the_pressure_has_decayed() {
    let cfg = PressureConfig {
        decay_rate: 20.0,
        ..spike_config()
    };
    let mut m = modifier(cfg);
    m.apply_pressure_at(0.0, 0.0, 0.0, 100.0, 0.5);
    m.update(0.1);
    let dent = m.deformation_at(0.0, 0.0, 0.0);
    assert!(dent > 0.0);
    for _ in 0..200 {
        m.update(0.1);
    }
    assert!(m.pressure_at(0.0, 0.0, 0.0) < 1.0e-3);
    // Pressure is far below the threshold now; the dent must be unchanged.
    assert!(m.deformation_at(0.0, 0.0, 0.0) >= dent);
}

#[test]
fn deformation_after_update_stays_within_zero_and_max() {
    let cfg = PressureConfig {
        max_deformation: 0.75,
        ..spike_config()
    };
    let mut m = modifier(cfg);
    m.pressure.data.fill(1.0e4);
    m.deformation.data.fill(-3.0);
    m.update(0.1);
    assert!(
        m.deformation.data.iter().all(|&d| d == 0.75),
        "capped at max"
    );
    m.deformation.data.fill(-3.0);
    m.pressure.data.fill(0.0);
    m.update(0.1);
    assert!(m.deformation.data.iter().all(|&d| d == 0.0), "floored at 0");
}

#[test]
fn modify_distance_adds_dent_and_subtracts_uniform_bulge() {
    let cfg = PressureConfig {
        internal_pressure: 40.0,
        expansion_rate: 0.025,
        ..PressureConfig::default()
    };
    let mut m = modifier(cfg);
    m.deformation.data.fill(0.2);
    // d' = d + deformation - internal * expansion = d + 0.2 - 1.0
    let d = m.modify_distance(0.3, 0.1, -0.4, 5.0);
    assert!(rel_ok(d, 4.2, 1.0e-6), "got {d}");
}

#[test]
fn disabled_modifier_is_transparent_and_frozen() {
    let mut m = modifier(spike_config());
    m.apply_pressure_at(0.0, 0.0, 0.0, 100.0, 0.5);
    m.deformation.data.fill(0.3);
    m.config.internal_pressure = 10.0;
    m.enabled = false;
    let before_p = m.pressure.data.clone();
    let before_d = m.deformation.data.clone();
    m.update(0.1);
    assert_eq!(m.pressure.data, before_p);
    assert_eq!(m.deformation.data, before_d);
    assert_eq!(m.modify_distance(0.0, 0.0, 0.0, 1.25), 1.25);
    assert!(!m.is_active());
    m.enabled = true;
    assert!(m.is_active());
    assert_eq!(m.name(), "pressure");
}

#[test]
fn negative_deformation_values_never_shrink_the_distance() {
    let mut m = modifier(PressureConfig::default());
    m.deformation.data.fill(-0.5);
    assert_eq!(m.modify_distance(0.0, 0.0, 0.0, 2.0), 2.0);
}

#[test]
fn pressure_between_nodes_is_trilinear_and_clamped_outside_the_domain() {
    let mut m = modifier(spike_config());
    m.apply_pressure_at(0.0, 0.0, 0.0, 100.0, 0.5);
    assert!(rel_ok(m.pressure_at(0.5, 0.0, 0.0), 50.0, 1.0e-6));
    assert!(rel_ok(m.pressure_at(0.5, 0.5, 0.5), 12.5, 1.0e-6));
    assert_eq!(m.pressure_at(0.0, 0.0, 0.0), 100.0);
    let mut u = modifier(spike_config());
    u.pressure.data.fill(7.0);
    assert_eq!(u.pressure_at(1.0e6, -1.0e6, 3.0), 7.0);
}

#[test]
fn diffusion_alone_conserves_total_pressure() {
    let cfg = PressureConfig {
        decay_rate: 0.0,
        yield_threshold: 1.0e9,
        ..spike_config()
    };
    let mut m = modifier(cfg);
    m.apply_pressure_at(1.0, -1.0, 0.0, 80.0, 0.5);
    m.apply_pressure_at(-2.0, 2.0, 2.0, 40.0, 0.5); // corner node
    let total0: f64 = m.pressure.data.iter().map(|&v| f64::from(v)).sum();
    for _ in 0..30 {
        m.update(0.1);
    }
    let total1: f64 = m.pressure.data.iter().map(|&v| f64::from(v)).sum();
    assert!(
        (total1 - total0).abs() < 1.0e-3 * total0,
        "{total0} -> {total1}"
    );
}

/// `apply_impact` clamps each impact to `max_deformation`, but impacts add into
/// the field, so two impacts at one node leave 2 * max until `update` clamps.
/// `deformation_at` / `modify_distance` read the field directly and see it.
#[test]
#[ignore = "known defect: AUD-A-S6W1-003: two apply_impact calls at one node give deformation 2.0 with max_deformation 1.0 before update() clamps"]
fn repeated_impacts_never_exceed_max_deformation_before_update() {
    let cfg = PressureConfig {
        deformation_rate: 1.0,
        max_deformation: 1.0,
        ..PressureConfig::default()
    };
    let mut m = modifier(cfg);
    m.apply_impact(0.0, 0.0, 0.0, 5.0, 0.5);
    m.apply_impact(0.0, 0.0, 0.0, 5.0, 0.5);
    assert!(m.deformation_at(0.0, 0.0, 0.0) <= 1.0);
    assert!(m.modify_distance(0.0, 0.0, 0.0, 0.0) <= 1.0);
}

/// `diffuse` is explicit Euler with no stability limit. With
/// `rate * dt / h^2 = 0.5 > 1/6` a positive spike produces negative pressure.
#[test]
#[ignore = "known defect: AUD-A-S6W1-004: update() with diffusion_rate*dt/h^2 = 0.5 turns a +100 spike into negative pressure (centre -200 before decay)"]
fn diffusion_step_never_produces_negative_pressure() {
    let cfg = PressureConfig {
        diffusion_rate: 5.0,
        decay_rate: 0.0,
        yield_threshold: 1.0e9,
        ..PressureConfig::default()
    };
    let mut m = modifier(cfg);
    m.apply_pressure_at(0.0, 0.0, 0.0, 100.0, 0.5);
    m.update(0.1);
    assert!(m.pressure.data.iter().all(|&p| p >= 0.0));
}

/// `PressureModifier::new` accepts resolution 0 and the first read panics
/// (`nx - 1` underflows).
#[test]
fn zero_resolution_does_not_panic_on_read() {
    let r = std::panic::catch_unwind(|| {
        let m = PressureModifier::new(
            PressureConfig::default(),
            0,
            (-1.0, -1.0, -1.0),
            (1.0, 1.0, 1.0),
        );
        m.pressure_at(0.0, 0.0, 0.0)
    });
    assert!(r.is_ok());
}

/// A negative `internal_pressure` is ignored, although the field doc reads
/// "positive = outward expansion".
#[test]
#[ignore = "known defect: AUD-A-S6W1-006: negative internal_pressure has no effect (no inward contraction), doc says only positive = outward"]
fn negative_internal_pressure_contracts_the_surface() {
    let cfg = PressureConfig {
        internal_pressure: -40.0,
        expansion_rate: 0.025,
        ..PressureConfig::default()
    };
    let m = modifier(cfg);
    assert!(m.modify_distance(0.0, 0.0, 0.0, 5.0) > 5.0);
}
