//! Oracles for `molecular_dynamics`: velocity Verlet, the periodic cell list
//! and the observables, entered through `VelocityVerlet::new` / `step` (the
//! production path), compared with closed forms.
//!
//! # Where the expected values come from
//!
//! - LJ dimer, small amplitude: harmonic frequency `ω = √(k/μ)` with
//!   `k = U''(r_min) = 72 ε / r_min² = 36·2^{2/3} ε/σ²` and the reduced mass
//!   `μ = m₁m₂/(m₁+m₂)`. Velocity Verlet on a harmonic oscillator is exactly
//!   periodic with `cos(ω_d h) = 1 − (ωh)²/2` (the leapfrog dispersion
//!   relation; Hairer, Lubich & Wanner, *Geometric Numerical Integration*,
//!   §I.5), so the period of the discrete trajectory is
//!   `T_d = 2π h / arccos(1 − (ωh)²/2)`. The anharmonic shift at amplitude
//!   `a` is `Δω/ω = a² (U''''/(16k) − 5U'''²/(48k²)) ≈ −18 a²/σ²` (Landau &
//!   Lifshitz, *Mechanics*, §28), 1.8e-7 at `a = 1e-4 σ`, included in the
//!   expected period.
//! - NVE: velocity Verlet is symplectic and time reversible, so the energy
//!   error stays bounded and scales as `h²` (no secular drift).
//! - Momentum: pair forces are applied as `+F` / `−F` of the same value, so
//!   `Σ F = 0` exactly in fixed point; `Σ m v` changes only by the rounding
//!   of the per-particle kick `v += F·h/(2m)`.
//! - Cell list: fixed-point addition is exact, so the force sum does not
//!   depend on the pair order and the cell list equals the O(N²) sum bit for
//!   bit when it finds the same pairs.
//! - Periodic boundary: the dynamics depends on minimum-image separations
//!   only, so a configuration translated by any vector evolves into the same
//!   configuration translated (fixed-point addition is exact).
//! - Temperature: `T = 2 K / (k_B (3N − 3))`.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
// f64 `sqrt` / `acos` compute closed-form references, not state.
#![allow(clippy::disallowed_methods)]

use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::molecular_dynamics::{
    pair_forces_all_pairs, pair_forces_cell_list, MdError, PeriodicBox, VelocityVerlet,
};
use alice_physics::pair_potential::{
    LennardJones, PairPotential, PairPotentialError, ShiftMode, Truncated,
};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn lj_unit(shift: ShiftMode, rc: f64) -> Truncated<LennardJones> {
    Truncated::new(
        LennardJones::new(Fix128::ONE, Fix128::ONE).unwrap(),
        fx(rc),
        shift,
    )
    .unwrap()
}

/// Deterministic generator for test inputs (64-bit LCG, Knuth MMIX).
struct Lcg(u64);
impl Lcg {
    fn next_unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Simple-cubic lattice of `n³` sites with spacing `a`, jittered by up to
/// `jitter` per component.
fn lattice(n: usize, a: f64, jitter: f64, rng: &mut Lcg) -> Vec<Vec3Fix> {
    let mut out = Vec::new();
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let mut c =
                    |b: usize| (b as f64 + 0.5) * a + jitter * (2.0 * rng.next_unit() - 1.0);
                let (x, y, z) = (c(i), c(j), c(k));
                out.push(Vec3Fix::new(fx(x), fx(y), fx(z)));
            }
        }
    }
    out
}

fn dimer_separation(md: &VelocityVerlet<LennardJones>) -> f64 {
    let p = md.positions();
    md.periodic_box()
        .minimum_image(p[1] - p[0])
        .length()
        .to_f64()
}

// ---------------------------------------------------------------------------
// LJ dimer: period of the small oscillation
// ---------------------------------------------------------------------------

#[test]
fn lj_dimer_small_oscillation_period_is_two_pi_sqrt_mu_over_k() {
    let (m1, m2) = (1.0, 3.0);
    let mu = m1 * m2 / (m1 + m2);
    let r_min = 2f64.powf(1.0 / 6.0);
    let k = 72.0 / (r_min * r_min);
    assert!((k - 57.146_437_5).abs() < 1e-5);
    let omega = (k / mu).sqrt();
    let h = 1e-3;
    let t_d = 2.0 * std::f64::consts::PI * h / (1.0 - (omega * h).powi(2) / 2.0).acos();
    let t_cont = 2.0 * std::f64::consts::PI / omega;
    // the discrete correction (ωh)²/24 ≈ 3.2e-6 is resolved by the tolerance
    assert!((t_cont - t_d) / t_cont > 2e-6);

    let a = 1e-4;
    let bx = PeriodicBox::cubic(Fix128::from_int(10)).unwrap();
    let mut md = VelocityVerlet::new(
        lj_unit(ShiftMode::EnergyShift, 2.5),
        bx,
        vec![
            Vec3Fix::new(fx(4.0), fx(5.0), fx(5.0)),
            Vec3Fix::new(fx(4.0 + r_min + a), fx(5.0), fx(5.0)),
        ],
        vec![Vec3Fix::ZERO; 2],
        vec![fx(m1), fx(m2)],
    )
    .unwrap();

    let mut prev = dimer_separation(&md) - r_min;
    let mut crossings = Vec::new();
    let mut amplitude = 0f64;
    for step in 1..=8000 {
        md.step(fx(h)).unwrap();
        let cur = dimer_separation(&md) - r_min;
        amplitude = amplitude.max(cur.abs());
        if prev < 0.0 && cur >= 0.0 {
            let frac = -prev / (cur - prev);
            crossings.push((step as f64 - 1.0 + frac) * h);
        }
        prev = cur;
    }
    assert!(crossings.len() >= 10, "{} crossings", crossings.len());
    let period = (crossings[crossings.len() - 1] - crossings[0]) / (crossings.len() - 1) as f64;
    // Expected: T_d with the anharmonic correction of Landau & Lifshitz §28,
    // T = T_d (1 − a² (U⁗/(16k) − 5U‴²/(48k²))), U‴ = −1512 ε/r_min³,
    // U⁗ = 26712 ε/r_min⁴ (σ = ε = 1). The correction is 1.8e-7; the next
    // order is O(a⁴) ~ 1e-13 and the crossing interpolation error is
    // O((ωh)³) per crossing and cancels between same-direction crossings,
    // so 2e-8 relative separates "correction included" from "omitted".
    let u3 = -1512.0 / r_min.powi(3);
    let u4 = 26712.0 / r_min.powi(4);
    let shift = a * a * (u4 / (16.0 * k) - 5.0 * u3 * u3 / (48.0 * k * k));
    assert!((shift + 1.8e-7).abs() < 0.1e-7, "shift {shift:e}");
    let expected = t_d * (1.0 - shift);
    assert!(
        (period - expected).abs() <= 2e-8 * expected,
        "period {period}, closed form {expected}, rel {:e}",
        (period - expected) / expected
    );
    // the amplitude stays a (released from rest at r_min + a; the
    // anharmonic asymmetry of the turning points is O(a²|U'''|/k) ≈ 2e-7)
    assert!((amplitude - a).abs() <= 1e-2 * a, "amplitude {amplitude}");
}

// ---------------------------------------------------------------------------
// NVE: bounded O(h²) energy error, momentum
// ---------------------------------------------------------------------------

fn lj_gas(seed: u64) -> (Vec<Vec3Fix>, Vec<Vec3Fix>, PeriodicBox) {
    let mut rng = Lcg(seed);
    let n = 3;
    let a = 1.8;
    let pos = lattice(n, a, 0.1, &mut rng);
    let mut vel: Vec<[f64; 3]> = (0..pos.len())
        .map(|_| {
            [
                2.0 * rng.next_unit() - 1.0,
                2.0 * rng.next_unit() - 1.0,
                2.0 * rng.next_unit() - 1.0,
            ]
        })
        .collect();
    let nn = vel.len() as f64;
    for c in 0..3 {
        let mean: f64 = vel.iter().map(|v| v[c]).sum::<f64>() / nn;
        for v in &mut vel {
            v[c] -= mean;
        }
    }
    let vel = vel
        .into_iter()
        .map(|v| Vec3Fix::new(fx(v[0]), fx(v[1]), fx(v[2])))
        .collect();
    (pos, vel, PeriodicBox::cubic(fx(n as f64 * a)).unwrap())
}

struct NveRun {
    first_half: f64,
    second_half: f64,
    momentum: f64,
}

fn run_nve(start: &VelocityVerlet<LennardJones>, h: f64, steps: usize) -> NveRun {
    let mut md = start.clone();
    let e0 = md.total_energy().to_f64();
    let p0 = md.momentum();
    let mut run = NveRun {
        first_half: 0.0,
        second_half: 0.0,
        momentum: 0.0,
    };
    for s in 0..steps {
        md.step(fx(h)).unwrap();
        // Σ F = 0 exactly
        let fsum = md.forces().iter().fold(Vec3Fix::ZERO, |acc, f| acc + *f);
        assert_eq!(fsum, Vec3Fix::ZERO, "step {s}: sum of pair forces");
        let dev = (md.total_energy().to_f64() - e0).abs();
        if s < steps / 2 {
            run.first_half = run.first_half.max(dev);
        } else {
            run.second_half = run.second_half.max(dev);
        }
        run.momentum = run.momentum.max((md.momentum() - p0).length().to_f64());
    }
    run
}

#[test]
fn nve_energy_error_is_bounded_and_second_order() {
    // 27 particles from a jittered lattice; 300 steps of equilibration so
    // the comparison starts from a state that no longer converts lattice
    // potential energy into heat (the error amplitude follows the
    // temperature)
    let (pos, vel, bx) = lj_gas(7);
    let n = pos.len();
    let mut md = VelocityVerlet::new(
        lj_unit(ShiftMode::ForceShift, 2.5),
        bx,
        pos,
        vel,
        vec![Fix128::ONE; n],
    )
    .unwrap();
    for _ in 0..300 {
        md.step(fx(0.002)).unwrap();
    }
    let e0 = md.total_energy().to_f64();
    let coarse = run_nve(&md, 0.004, 300);
    let fine = run_nve(&md, 0.002, 600);
    let dev_h = coarse.first_half.max(coarse.second_half);
    let dev_h2 = fine.first_half.max(fine.second_half);
    // the energy error is resolved and small against the energy scale
    assert!(dev_h > 1e-7, "dev_h {dev_h}");
    assert!(
        dev_h < 1e-3 * (e0.abs() + md.kinetic_energy().to_f64()),
        "dev_h {dev_h}, E0 {e0}"
    );
    // second order: halving h divides the error by ≈ 4
    let ratio = dev_h / dev_h2;
    assert!((3.0..=5.5).contains(&ratio), "ratio {ratio}");
    // no drift: the second half does not exceed the first by more than the
    // fluctuation itself (a secular drift would grow linearly in time)
    for r in [&coarse, &fine] {
        assert!(
            r.second_half <= 2.0 * r.first_half,
            "first {}, second {}",
            r.first_half,
            r.second_half
        );
    }
    // momentum: only the per-particle kick rounding, 27 particles × 600
    // steps × 2 kicks × 2⁻⁶⁴ ≈ 2e-15
    assert!(
        coarse.momentum < 1e-13 && fine.momentum < 1e-13,
        "momentum drift {:e} {:e}",
        coarse.momentum,
        fine.momentum
    );
}

// ---------------------------------------------------------------------------
// Periodic boundary
// ---------------------------------------------------------------------------

#[test]
fn crossing_the_boundary_is_the_translated_motion_bit_for_bit() {
    let l = 8.0;
    let bx = PeriodicBox::cubic(fx(l)).unwrap();
    let r0 = 1.15;
    // A: the dimer straddles the x = 0 face from the start and drifts in −x
    // B: the same dimer translated by +L/2 in x (in the middle of the box)
    let a_pos = vec![
        Vec3Fix::new(fx(0.3), fx(2.0), fx(3.0)),
        Vec3Fix::new(fx(l + 0.3 - r0), fx(2.1), fx(3.0)), // across the x = 0 face
    ];
    let shift = Vec3Fix::new(fx(l / 2.0), Fix128::ZERO, Fix128::ZERO);
    let b_pos: Vec<Vec3Fix> = a_pos.iter().map(|p| bx.wrap(*p + shift)).collect();
    let vel = vec![
        Vec3Fix::new(fx(-1.3), fx(0.2), Fix128::ZERO),
        Vec3Fix::new(fx(-0.7), fx(-0.1), fx(0.05)),
    ];
    let mk = |pos: Vec<Vec3Fix>| {
        VelocityVerlet::new(
            lj_unit(ShiftMode::ForceShift, 2.5),
            bx,
            pos,
            vel.clone(),
            vec![Fix128::ONE, fx(2.0)],
        )
        .unwrap()
    };
    let mut a = mk(a_pos);
    let mut b = mk(b_pos);
    // the pair interacts through the face: non-zero force at the start
    assert!(a.forces()[0].length() > fx(1e-3));
    let mut crossed = 0;
    for _ in 0..1500 {
        let before = a.positions()[0].x;
        a.step(fx(0.002)).unwrap();
        b.step(fx(0.002)).unwrap();
        if a.positions()[0].x > before {
            crossed += 1; // wrapped from 0 to L
        }
        for i in 0..2 {
            assert_eq!(a.positions()[i], bx.wrap(b.positions()[i] + shift));
            assert_eq!(a.velocities()[i], b.velocities()[i]);
            assert_eq!(a.forces()[i], b.forces()[i]);
        }
    }
    assert!(crossed >= 1, "particle 0 never crossed the face");
    // positions stay in [0, L)
    for p in a.positions() {
        for c in [p.x, p.y, p.z] {
            assert!(c >= Fix128::ZERO && c < fx(l));
        }
    }
}

#[test]
fn minimum_image_and_wrap_closed_form() {
    let bx = PeriodicBox::new(Vec3Fix::new(fx(4.0), fx(6.0), fx(10.0))).unwrap();
    let d = bx.minimum_image(Vec3Fix::new(fx(2.4), fx(-3.5), fx(19.0)));
    assert_eq!(d, Vec3Fix::new(fx(-1.6), fx(2.5), fx(-1.0)));
    // the tie d = L/2 maps to −L/2: the range is [−L/2, L/2)
    let t = bx.minimum_image(Vec3Fix::new(fx(2.0), fx(-3.0), Fix128::ZERO));
    assert_eq!(t, Vec3Fix::new(fx(-2.0), fx(-3.0), Fix128::ZERO));
    let w = bx.wrap(Vec3Fix::new(fx(-0.5), fx(13.0), fx(10.0)));
    assert_eq!(w, Vec3Fix::new(fx(3.5), fx(1.0), Fix128::ZERO));
    assert_eq!(bx.lengths(), Vec3Fix::new(fx(4.0), fx(6.0), fx(10.0)));
}

// ---------------------------------------------------------------------------
// Cell list = all-pairs sum
// ---------------------------------------------------------------------------

fn cell_list_matches_all_pairs(n: usize, a: f64, seed: u64) {
    let mut rng = Lcg(seed);
    let pos = lattice(n, a, 0.3 * a, &mut rng);
    let bx = PeriodicBox::cubic(fx(n as f64 * a)).unwrap();
    let pot = lj_unit(ShiftMode::ForceShift, 2.5);
    let c = pair_forces_cell_list(&pot, &bx, &pos).unwrap();
    let o = pair_forces_all_pairs(&pot, &bx, &pos).unwrap();
    assert_eq!(c.forces, o.forces);
    assert_eq!(c.potential_energy, o.potential_energy);
    // the scene has pairs inside and beyond the cutoff, and across the faces
    let mut inside = 0;
    let mut across = 0;
    for i in 0..pos.len() {
        for j in (i + 1)..pos.len() {
            let raw = pos[j] - pos[i];
            let d = bx.minimum_image(raw);
            if d.length() < fx(2.5) {
                inside += 1;
                if d != raw {
                    across += 1;
                }
            }
        }
    }
    let total = pos.len() * (pos.len() - 1) / 2;
    assert!(inside > 0 && inside < total, "inside {inside} of {total}");
    assert!(across > 0, "no pair across a face");
    assert!(c.forces.iter().all(|f| *f != Vec3Fix::ZERO));
}

#[test]
fn cell_list_matches_all_pairs_bit_for_bit() {
    // L / r_c = 2.08 (2 cells per axis), 3.12 (3 cells), 5.2 (5 cells)
    cell_list_matches_all_pairs(4, 1.3, 11);
    cell_list_matches_all_pairs(6, 1.3, 12);
    cell_list_matches_all_pairs(10, 1.3, 13);
}

#[test]
fn cell_list_handles_empty_and_single_particle() {
    let bx = PeriodicBox::cubic(Fix128::from_int(6)).unwrap();
    let pot = lj_unit(ShiftMode::EnergyShift, 2.5);
    let e = pair_forces_cell_list(&pot, &bx, &[]).unwrap();
    assert!(e.forces.is_empty());
    assert_eq!(e.potential_energy, Fix128::ZERO);
    let one = [Vec3Fix::new(fx(1.0), fx(2.0), fx(3.0))];
    let s = pair_forces_cell_list(&pot, &bx, &one).unwrap();
    assert_eq!(s.forces, vec![Vec3Fix::ZERO]);
    assert_eq!(s.potential_energy, Fix128::ZERO);

    // N = 1: free flight across the face, no self-image interaction (L ≥ 2 r_c)
    let mut md = VelocityVerlet::new(
        pot,
        bx,
        one.to_vec(),
        vec![Vec3Fix::new(fx(-2.0), Fix128::ZERO, Fix128::ONE)],
        vec![Fix128::ONE],
    )
    .unwrap();
    for _ in 0..4 {
        md.step(fx(0.25)).unwrap();
    }
    assert_eq!(md.positions()[0], Vec3Fix::new(fx(5.0), fx(2.0), fx(4.0)));
    assert_eq!(md.potential_energy(), Fix128::ZERO);
    assert_eq!(md.kinetic_energy(), fx(2.5));
    assert_eq!(
        md.instantaneous_temperature(Fix128::ONE),
        Err(MdError::TooFewParticles)
    );

    // N = 0
    let mut md0 = VelocityVerlet::new(pot, bx, vec![], vec![], vec![]).unwrap();
    md0.step(fx(0.1)).unwrap();
    assert_eq!(md0.total_energy(), Fix128::ZERO);
    assert_eq!(md0.momentum(), Vec3Fix::ZERO);
}

// ---------------------------------------------------------------------------
// Observables
// ---------------------------------------------------------------------------

#[test]
fn temperature_and_kinetic_energy_closed_form() {
    let bx = PeriodicBox::cubic(Fix128::from_int(10)).unwrap();
    // ε = 0: no interaction, only the kinetic part
    let pot = Truncated::new(
        LennardJones::new(Fix128::ZERO, Fix128::ONE).unwrap(),
        fx(2.5),
        ShiftMode::None,
    )
    .unwrap();
    let md = VelocityVerlet::new(
        pot,
        bx,
        vec![
            Vec3Fix::new(fx(1.0), fx(1.0), fx(1.0)),
            Vec3Fix::new(fx(5.0), fx(5.0), fx(5.0)),
        ],
        vec![
            Vec3Fix::new(fx(1.0), Fix128::ZERO, Fix128::ZERO),
            Vec3Fix::new(fx(-0.5), Fix128::ZERO, Fix128::ZERO),
        ],
        vec![Fix128::ONE, Fix128::from_int(2)],
    )
    .unwrap();
    // K = ½·1·1 + ½·2·0.25 = 0.75, dof = 3·2 − 3 = 3, T = 2K/(3 k_B)
    assert_eq!(md.kinetic_energy(), fx(0.75));
    assert_eq!(md.momentum(), Vec3Fix::ZERO);
    let t = md.instantaneous_temperature(fx(0.5)).unwrap().to_f64();
    assert!((t - 1.0).abs() < 1e-15, "T {t}");
    assert_eq!(
        md.instantaneous_temperature(Fix128::ZERO),
        Err(MdError::NonPositiveBoltzmannConstant)
    );
    assert_eq!(md.masses(), &[Fix128::ONE, Fix128::from_int(2)]);
    assert_eq!(md.potential().cutoff(), fx(2.5));
}

// ---------------------------------------------------------------------------
// Degenerate input
// ---------------------------------------------------------------------------

#[test]
fn degenerate_inputs_are_errors() {
    let pot = lj_unit(ShiftMode::EnergyShift, 2.5);
    assert_eq!(
        PeriodicBox::cubic(Fix128::ZERO).err(),
        Some(MdError::NonPositiveBoxLength)
    );
    assert_eq!(
        PeriodicBox::new(Vec3Fix::new(fx(5.0), fx(-1.0), fx(5.0))).err(),
        Some(MdError::NonPositiveBoxLength)
    );
    // box < 2 r_c on one axis: the minimum image no longer holds
    let small = PeriodicBox::new(Vec3Fix::new(fx(6.0), fx(4.99), fx(6.0))).unwrap();
    let p = vec![Vec3Fix::new(fx(1.0), fx(1.0), fx(1.0))];
    assert_eq!(
        VelocityVerlet::new(
            pot,
            small,
            p.clone(),
            vec![Vec3Fix::ZERO],
            vec![Fix128::ONE]
        )
        .err(),
        Some(MdError::BoxTooSmall)
    );
    assert_eq!(
        pair_forces_cell_list(&pot, &small, &p).err(),
        Some(MdError::BoxTooSmall)
    );
    assert_eq!(
        pair_forces_all_pairs(&pot, &small, &p).err(),
        Some(MdError::BoxTooSmall)
    );
    // exactly 2 r_c is allowed
    let edge = PeriodicBox::cubic(fx(5.0)).unwrap();
    assert!(pair_forces_cell_list(&pot, &edge, &p).is_ok());

    let bx = PeriodicBox::cubic(fx(6.0)).unwrap();
    assert_eq!(
        VelocityVerlet::new(pot, bx, p.clone(), vec![], vec![Fix128::ONE]).err(),
        Some(MdError::LengthMismatch)
    );
    assert_eq!(
        VelocityVerlet::new(pot, bx, p.clone(), vec![Vec3Fix::ZERO], vec![Fix128::ZERO]).err(),
        Some(MdError::NonPositiveMass)
    );
    let mut md =
        VelocityVerlet::new(pot, bx, p.clone(), vec![Vec3Fix::ZERO], vec![Fix128::ONE]).unwrap();
    assert_eq!(md.step(Fix128::ZERO), Err(MdError::NonPositiveTimestep));
    assert_eq!(md.step(fx(-0.1)), Err(MdError::NonPositiveTimestep));

    // coincident particles at construction
    let two = vec![p[0], p[0]];
    assert_eq!(
        VelocityVerlet::new(pot, bx, two, vec![Vec3Fix::ZERO; 2], vec![Fix128::ONE; 2]).err(),
        Some(MdError::Potential {
            i: 0,
            j: 1,
            error: PairPotentialError::NonPositiveDistance
        })
    );
}

#[test]
fn failed_step_leaves_the_state_unchanged() {
    // ε = 0 (no force): the two particles meet exactly at x = 1.5 after one
    // step; the distance 0 is an error and the step is not applied
    let pot = Truncated::new(
        LennardJones::new(Fix128::ZERO, Fix128::ONE).unwrap(),
        fx(2.5),
        ShiftMode::None,
    )
    .unwrap();
    let bx = PeriodicBox::cubic(fx(6.0)).unwrap();
    let pos = vec![
        Vec3Fix::new(fx(1.0), fx(1.0), fx(1.0)),
        Vec3Fix::new(fx(2.0), fx(1.0), fx(1.0)),
    ];
    let vel = vec![
        Vec3Fix::new(fx(0.5), Fix128::ZERO, Fix128::ZERO),
        Vec3Fix::new(fx(-0.5), Fix128::ZERO, Fix128::ZERO),
    ];
    let mut md =
        VelocityVerlet::new(pot, bx, pos.clone(), vel.clone(), vec![Fix128::ONE; 2]).unwrap();
    assert_eq!(
        md.step(Fix128::ONE),
        Err(MdError::Potential {
            i: 0,
            j: 1,
            error: PairPotentialError::NonPositiveDistance
        })
    );
    assert_eq!(md.positions(), &pos[..]);
    assert_eq!(md.velocities(), &vel[..]);
}

#[test]
fn md_error_messages_are_distinct() {
    use std::collections::BTreeSet;
    let all = [
        MdError::NonPositiveBoxLength,
        MdError::BoxTooSmall,
        MdError::LengthMismatch,
        MdError::NonPositiveMass,
        MdError::NonPositiveTimestep,
        MdError::NonPositiveBoltzmannConstant,
        MdError::TooFewParticles,
        MdError::Potential {
            i: 0,
            j: 1,
            error: PairPotentialError::Overflow,
        },
    ];
    let msgs: BTreeSet<String> = all.iter().map(ToString::to_string).collect();
    assert_eq!(msgs.len(), all.len());
    // the potential trait is usable through the truncation
    let pot = lj_unit(ShiftMode::EnergyShift, 2.5);
    assert!(pot.energy(fx(1.0)).is_ok());
}
