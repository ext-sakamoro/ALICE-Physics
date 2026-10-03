//! Audit oracles for `rope::Rope` (XPBD distance chain).
//!
//! Existing oracles (`analytic_rope_wiring`, `analytic_rope_attach_wiring`)
//! cover pins, `current_length` and the SDF no-op paths; none checks the
//! dynamics of `step` or the SDF contact response. Closed forms used here:
//!
//! * symplectic Euler over `n` substeps of size `h`: `v_k = g h k`,
//!   `x_k = x_0 + g h^2 k (k+1)/2`; frame damping once: `v <- d v`
//! * one XPBD iteration of one distance constraint:
//!   `lambda = e / (w0 + w1 + alpha/dt^2)`, `p0 += w0 lambda u`, `p1 -= w1 lambda u`

use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::rope::Rope;
use alice_physics::sdf_collider::{ClosureSdf, SdfCollider};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn arr(v: Vec3Fix) -> [f64; 3] {
    [v.x.to_f64(), v.y.to_f64(), v.z.to_f64()]
}

fn ground() -> SdfCollider {
    let field = ClosureSdf::new(|_x, y, _z| y, |_x, _y, _z| (0.0, 1.0, 0.0));
    SdfCollider::new_static(Box::new(field), Vec3Fix::ZERO, QuatFix::IDENTITY)
}

/// `new`: particles lie on the segment at `i/N`, rest length is `L/N`, `total_length = L`.
#[test]
fn new_places_particles_evenly_with_rest_length_l_over_n() {
    let r = Rope::new(v3(1.0, 2.0, 3.0), v3(4.0, 6.0, 3.0), 5, fx(0.5));
    assert_eq!(r.particle_count(), 6);
    assert_eq!(r.segment_count(), 5);
    assert!(
        (r.total_length.to_f64() - 5.0).abs() < 1e-12,
        "3-4-5 length"
    );
    for i in 0..6 {
        let t = i as f64 / 5.0;
        let p = arr(r.positions[i]);
        let want = [1.0 + 3.0 * t, 2.0 + 4.0 * t, 3.0];
        for k in 0..3 {
            assert!((p[k] - want[k]).abs() < 1e-12, "particle {i} axis {k}");
        }
    }
    for l in &r.rest_lengths {
        assert!((l.to_f64() - 1.0).abs() < 1e-12);
    }
    // the last particle is the end point exactly
    assert_eq!(r.positions[5], v3(4.0, 6.0, 3.0));
    assert_eq!(r.prev_positions, r.positions);
}

#[test]
#[should_panic(expected = "at least one segment")]
fn zero_segments_panics_as_documented() {
    let _ = Rope::new(Vec3Fix::ZERO, v3(1.0, 0.0, 0.0), 0, Fix128::ONE);
}

/// `mass_per_unit` is a mass per unit length: the rope's total mass should be
/// `mass_per_unit * L`.  `new` gives every one of the `N+1` particles a full
/// segment mass `mass_per_unit * L / N`, so the total is `mass_per_unit * L * (N+1)/N`
/// (2x for a one-segment rope).
#[test]
#[ignore = "known defect: AUD-A-S3W1-003: Rope::new total particle mass = mpu*L*(N+1)/N, not mpu*L (measured N=1: 12 vs 6); end particles are not half mass"]
fn total_particle_mass_is_mass_per_unit_times_length() {
    for (n, mpu, len) in [(1usize, 2.0, 3.0), (4, 0.5, 8.0), (10, 1.0, 5.0)] {
        let r = Rope::new(Vec3Fix::ZERO, v3(len, 0.0, 0.0), n, fx(mpu));
        let total: f64 = r.inv_masses.iter().map(|w| 1.0 / w.to_f64()).sum();
        assert!(
            (total - mpu * len).abs() < 1e-9,
            "n={n}: total mass {total}, expected {}",
            mpu * len
        );
    }
}

/// Inverse mass is `1/(mpu * L/N)` for every particle (the convention `new` uses today).
#[test]
fn inverse_mass_is_one_over_segment_mass() {
    let r = Rope::new(Vec3Fix::ZERO, v3(6.0, 0.0, 0.0), 3, fx(0.25));
    // segment length 2, particle mass 0.5, inv mass 2
    for w in &r.inv_masses {
        assert!((w.to_f64() - 2.0).abs() < 1e-12);
    }
}

fn free_fall_rope(substeps: usize) -> Rope {
    // unpinned horizontal rope, constraints already satisfied: falls rigidly
    let mut r = Rope::new(Vec3Fix::ZERO, v3(4.0, 0.0, 0.0), 4, Fix128::ONE);
    r.config.substeps = substeps;
    r
}

/// Free fall of a satisfied chain: every particle moves identically,
/// `y = g h^2 n(n+1)/2` after `n` substeps and `v = d * g * dt` after the frame
/// (damping once per frame, independent of the number of substeps).
#[test]
fn free_fall_follows_symplectic_euler_with_damping_once_per_frame() {
    let dt = 1.0 / 16.0; // dyadic
    let g = -10.0;
    let d = 0.9375; // dyadic (15/16)
    for substeps in [1usize, 2, 4] {
        let mut r = free_fall_rope(substeps);
        r.config.damping = fx(d);
        r.config.gravity = v3(0.0, g, 0.0);
        r.step(fx(dt));
        let h = dt / substeps as f64;
        let n = substeps as f64;
        let want_y = g * h * h * n * (n + 1.0) / 2.0;
        let want_v = d * g * dt;
        for (i, p) in r.positions.iter().enumerate() {
            assert!(
                (p.y.to_f64() - want_y).abs() < 1e-9,
                "substeps {substeps} particle {i}: y {}",
                p.y.to_f64()
            );
            assert!(
                (p.x.to_f64() - i as f64).abs() < 1e-9,
                "x drift particle {i}"
            );
        }
        for v in &r.velocities {
            assert!(
                (v.y.to_f64() - want_v).abs() < 1e-9,
                "substeps {substeps}: v {}",
                v.y.to_f64()
            );
        }
    }
}

/// One iteration of one distance constraint, equal masses, no gravity: the stretched
/// pair is pulled together symmetrically to the rest length.
#[test]
fn one_distance_iteration_restores_rest_length_symmetrically() {
    let mut r = Rope::new(Vec3Fix::ZERO, v3(1.0, 0.0, 0.0), 1, Fix128::ONE);
    r.config.gravity = Vec3Fix::ZERO;
    r.config.iterations = 1;
    r.config.substeps = 1;
    r.config.damping = Fix128::ONE;
    r.positions[0] = v3(0.0, 0.0, 0.0);
    r.positions[1] = v3(3.0, 0.0, 0.0);
    r.prev_positions = r.positions.clone();
    r.step(fx(0.25));
    // rest 1, error 2, w0 = w1 -> each moves 1 toward the other
    assert!((r.positions[0].x.to_f64() - 1.0).abs() < 1e-12);
    assert!((r.positions[1].x.to_f64() - 2.0).abs() < 1e-12);
    // velocities = (x - prev)/dt
    assert!((r.velocities[0].x.to_f64() - 4.0).abs() < 1e-9);
    assert!((r.velocities[1].x.to_f64() + 4.0).abs() < 1e-9);
}

/// Unequal masses: the correction splits in proportion to the inverse masses (w0 : w1).
#[test]
fn constraint_correction_splits_by_inverse_mass() {
    let mut r = Rope::new(Vec3Fix::ZERO, v3(1.0, 0.0, 0.0), 1, Fix128::ONE);
    r.config.gravity = Vec3Fix::ZERO;
    r.config.iterations = 1;
    r.config.substeps = 1;
    r.inv_masses[0] = fx(1.0);
    r.inv_masses[1] = fx(3.0);
    r.positions[0] = v3(0.0, 0.0, 0.0);
    r.positions[1] = v3(5.0, 0.0, 0.0);
    r.prev_positions = r.positions.clone();
    r.step(fx(0.25));
    // error 4, total move 4 split 1:3 -> p0 += 1, p1 -= 3
    assert!((r.positions[0].x.to_f64() - 1.0).abs() < 1e-12);
    assert!((r.positions[1].x.to_f64() - 2.0).abs() < 1e-12);
}

/// A pinned (w = 0) end does not move; the free end takes the whole correction.
#[test]
fn pinned_end_takes_no_correction() {
    let mut r = Rope::new(Vec3Fix::ZERO, v3(1.0, 0.0, 0.0), 1, Fix128::ONE);
    r.config.gravity = Vec3Fix::ZERO;
    r.config.iterations = 1;
    r.config.substeps = 1;
    r.pin_start();
    r.positions[1] = v3(5.0, 0.0, 0.0);
    r.prev_positions = r.positions.clone();
    r.step(fx(0.25));
    assert_eq!(r.positions[0], Vec3Fix::ZERO);
    assert!((r.positions[1].x.to_f64() - 1.0).abs() < 1e-12);
}

/// XPBD compliance: `lambda = e / (w_sum + alpha/dt^2)`; each particle moves `w*lambda`.
#[test]
fn compliance_softens_the_correction_by_alpha_over_dt_squared() {
    let mut r = Rope::new(Vec3Fix::ZERO, v3(1.0, 0.0, 0.0), 1, Fix128::ONE);
    r.config.gravity = Vec3Fix::ZERO;
    r.config.iterations = 1;
    r.config.substeps = 1;
    let alpha = 0.0625;
    r.config.compliance = fx(alpha);
    r.positions[0] = v3(0.0, 0.0, 0.0);
    r.positions[1] = v3(3.0, 0.0, 0.0);
    r.prev_positions = r.positions.clone();
    let dt = 0.25;
    r.step(fx(dt));
    // w0 = w1 = 1; error 2; alpha/dt^2 = 1.0; lambda = 2/(2+1) = 2/3
    let lambda = 2.0 / (2.0 + alpha / (dt * dt));
    assert!(
        (r.positions[0].x.to_f64() - lambda).abs() < 1e-9,
        "{}",
        r.positions[0].x.to_f64()
    );
    assert!((r.positions[1].x.to_f64() - (3.0 - lambda)).abs() < 1e-9);
}

/// A compressed segment (shorter than rest) is pushed apart (error < 0).
#[test]
fn compressed_segment_is_pushed_apart() {
    let mut r = Rope::new(Vec3Fix::ZERO, v3(2.0, 0.0, 0.0), 1, Fix128::ONE);
    r.config.gravity = Vec3Fix::ZERO;
    r.config.iterations = 1;
    r.config.substeps = 1;
    r.positions[0] = v3(0.0, 0.0, 0.0);
    r.positions[1] = v3(0.5, 0.0, 0.0);
    r.prev_positions = r.positions.clone();
    r.step(fx(0.25));
    // rest 2, dist 0.5, error -1.5: p0 -> -0.75, p1 -> 1.25
    assert!((r.positions[0].x.to_f64() + 0.75).abs() < 1e-12);
    assert!((r.positions[1].x.to_f64() - 1.25).abs() < 1e-12);
}

/// Direction of the correction follows the segment direction in 3D (not just x).
#[test]
fn constraint_correction_acts_along_the_segment_direction() {
    let mut r = Rope::new(Vec3Fix::ZERO, v3(5.0, 0.0, 0.0), 1, Fix128::ONE);
    r.config.gravity = Vec3Fix::ZERO;
    r.config.iterations = 1;
    r.config.substeps = 1;
    r.positions[0] = v3(0.0, 0.0, 0.0);
    r.positions[1] = v3(6.0, 8.0, 0.0); // length 10, rest 5
    r.prev_positions = r.positions.clone();
    r.step(fx(0.25));
    // each moves 2.5 toward the other along (0.6, 0.8)
    let p0 = arr(r.positions[0]);
    let p1 = arr(r.positions[1]);
    assert!(
        (p0[0] - 1.5).abs() < 1e-9 && (p0[1] - 2.0).abs() < 1e-9,
        "{p0:?}"
    );
    assert!(
        (p1[0] - 4.5).abs() < 1e-9 && (p1[1] - 6.0).abs() < 1e-9,
        "{p1:?}"
    );
}

fn contact_rope(substeps: usize, friction: f64) -> Rope {
    // one free particle (second pinned far away is avoided: use a 1-segment rope with
    // zero iterations so constraints do not interfere)
    let mut r = Rope::new(v3(0.0, -0.01, 0.0), v3(1.0, -0.01, 0.0), 1, Fix128::ONE);
    r.config.iterations = 0;
    r.config.substeps = substeps;
    r.config.sdf_friction = fx(friction);
    r.config.damping = Fix128::ONE;
    r.config.gravity = Vec3Fix::ZERO;
    r
}

/// Closed form of one SDF contact for an inbound particle: after the push-out
/// `v_n' = -0.1 v_n` (restitution 0.1) and `v_t' = (1 - mu) v_t`.
#[test]
fn inbound_contact_scales_normal_by_minus_tenth_and_tangent_by_one_minus_mu() {
    let mut r = contact_rope(1, 0.25);
    r.velocities[0] = v3(2.0, -3.0, 0.0);
    r.velocities[1] = v3(2.0, -3.0, 0.0);
    r.positions[0] = v3(0.0, -0.01, 0.0);
    r.positions[1] = v3(1.0, -0.01, 0.0);
    r.prev_positions = r.positions.clone();
    r.step_with_sdf(fx(0.125), &[ground()]);
    // moved by v*dt = (0.25, -0.375): y = -0.385 < 0 -> pushed to y = 0
    assert!(
        r.positions[0].y.to_f64().abs() < 1e-5,
        "pushed to the surface: {}",
        r.positions[0].y.to_f64()
    );
    let v = arr(r.velocities[0]);
    assert!((v[0] - 2.0 * 0.75).abs() < 1e-5, "tangent {v:?}");
    assert!((v[1] - 0.3).abs() < 1e-5, "normal {v:?}");
}

/// A particle still inside the surface but moving *out* of it keeps its outward
/// velocity; restitution applies only to the approaching component.
#[test]
#[ignore = "known defect: AUD-A-S3W1-004: resolve_sdf_collisions sets v_n' = -0.1 v_n unconditionally, reversing a separating (outward) normal velocity into the surface (vy +3 -> -0.3 measured)"]
fn separating_normal_velocity_is_not_reversed_by_the_contact() {
    let mut r = contact_rope(1, 0.0);
    r.positions[0] = v3(0.0, -0.5, 0.0);
    r.positions[1] = v3(1.0, -0.5, 0.0);
    r.prev_positions = r.positions.clone();
    r.velocities[0] = v3(0.0, 3.0, 0.0);
    r.velocities[1] = v3(0.0, 3.0, 0.0);
    // dt 1/8: y -> -0.5 + 0.375 = -0.125, still inside, moving outward
    r.step_with_sdf(fx(0.125), &[ground()]);
    assert!(
        r.velocities[0].y.to_f64() >= 0.0,
        "outward velocity reversed to {}",
        r.velocities[0].y.to_f64()
    );
}

/// `sdf_friction` should not depend on the substep count (the 1.2.0 note on
/// `damping` fixes the same kind of dependence): the tangential speed lost over a
/// frame for a rope pressed on the ground is `mu * g * dt` (Coulomb), not
/// `(1-mu)^substeps`.
#[test]
#[ignore = "known defect: AUD-A-S3W1-005: sdf_friction is a per-contact velocity retention (1-mu) applied each substep and independent of the normal force; tangential speed 5.0 -> 3.5 (substeps=1) vs 1.2005 (substeps=4) over one frame, i.e. 0.7^substeps"]
fn tangential_loss_per_frame_is_independent_of_the_substep_count() {
    let mut speeds = Vec::new();
    for substeps in [1usize, 4] {
        let mut r = contact_rope(substeps, 0.3);
        r.config.gravity = v3(0.0, -10.0, 0.0);
        r.positions[0] = v3(0.0, 0.0, 0.0);
        r.positions[1] = v3(1.0, 0.0, 0.0);
        r.prev_positions = r.positions.clone();
        r.velocities[0] = v3(5.0, 0.0, 0.0);
        r.velocities[1] = v3(5.0, 0.0, 0.0);
        r.step_with_sdf(fx(0.0625), &[ground()]);
        speeds.push(r.velocities[0].x.to_f64());
    }
    let rel = (speeds[0] - speeds[1]).abs() / speeds[0].abs().max(1e-9);
    assert!(
        rel < 0.05,
        "tangential speed after one frame: substeps=1 {} vs substeps=4 {}",
        speeds[0],
        speeds[1]
    );
}

/// A rope resting entirely above the SDF: `step_with_sdf` is bit-identical to `step`
/// (already covered for an empty collider list; here with a collider that is not touched).
#[test]
fn untouched_collider_does_not_change_the_step() {
    let mk = || {
        let mut r = Rope::new(v3(0.0, 5.0, 0.0), v3(2.0, 5.0, 0.0), 4, Fix128::ONE);
        r.pin_start();
        r
    };
    let mut a = mk();
    let mut b = mk();
    let g = [ground()];
    for _ in 0..5 {
        a.step_with_sdf(fx(1.0 / 64.0), &g);
        b.step(fx(1.0 / 64.0));
    }
    assert_eq!(a.positions, b.positions);
}

/// `iterations` is honoured: for a pinned, stretched chain more Gauss-Seidel sweeps leave
/// a strictly smaller length residual, and zero sweeps leave the stretch untouched.
#[test]
fn more_iterations_leave_a_smaller_length_residual() {
    let residual = |iters: usize| {
        let mut r = Rope::new(Vec3Fix::ZERO, v3(3.0, 0.0, 0.0), 3, Fix128::ONE);
        r.config.gravity = Vec3Fix::ZERO;
        r.config.substeps = 1;
        r.config.iterations = iters;
        r.pin_start();
        r.positions[3] = v3(6.0, 0.0, 0.0);
        r.prev_positions = r.positions.clone();
        r.step(fx(0.25));
        (r.current_length().to_f64() - 3.0).abs()
    };
    let (r0, r1, r8) = (residual(0), residual(1), residual(8));
    assert!(
        (r0 - 3.0).abs() < 1e-12,
        "zero sweeps: stretch untouched, residual {r0}"
    );
    assert!(r1 < r0 && r8 < r1 * 0.6, "residuals {r0} {r1} {r8}");
}

/// A particle exactly on the surface (distance 0) is not in contact: its velocity is
/// left alone (contact needs `dist < 0`).
#[test]
fn a_particle_exactly_on_the_surface_keeps_its_velocity() {
    let mut r = contact_rope(1, 0.5);
    r.positions[0] = v3(0.0, 0.0, 0.0);
    r.positions[1] = v3(1.0, 0.0, 0.0);
    r.prev_positions = r.positions.clone();
    r.velocities[0] = v3(2.0, 0.0, 0.0);
    r.velocities[1] = v3(2.0, 0.0, 0.0);
    r.step_with_sdf(fx(0.125), &[ground()]);
    assert!(
        (r.velocities[0].x.to_f64() - 2.0).abs() < 1e-12,
        "{}",
        r.velocities[0].x.to_f64()
    );
}
