//! Second, independent set of oracles for `molecular_dynamics`.
//!
//! `tests/analytic_molecular_dynamics.rs` uses the unit LJ dimer's
//! zero-crossing period (with the Landau–Lifshitz anharmonic shift), a
//! 27-particle LJ gas for the `h²` energy error, translation by a lattice
//! vector, and cell list vs all pairs. This file uses other systems and other
//! derivations:
//!
//! - the dimer frequency from the **discrete quadratic invariant** of velocity
//!   Verlet on a harmonic mode: `μ v_n² + k (1 − (ωh)²/4) x_n²` is exactly
//!   conserved (`x_n`, `v_n` at whole steps; follows from the one-step map
//!   `v_{n+1/2} = v_n − (h/2) ω² x_n`, `x_{n+1} = x_n + h v_{n+1/2}`,
//!   `v_{n+1} = v_{n+1/2} − (h/2) ω² x_{n+1}`), so a regression of `μ v²`
//!   on `x²` over the trajectory has slope `−k (1 − (ωh)²/4)`. Done for an
//!   unequal-mass LJ dimer (`ε = 0.8, σ = 1.2, m = (2, 5)`,
//!   `k = 72ε/(2^{1/3}σ²)`) and a Morse dimer (`k = 2Da²`);
//! - the `h²` energy error and momentum on a 5-particle **Morse** cluster with
//!   unequal masses (the module is generic over the potential);
//! - time reversibility (forward, negate velocities, forward);
//! - Galilean invariance: a uniform velocity added to every particle leaves
//!   the relative motion unchanged and translates the centre of mass by
//!   `u t` (modulo the box);
//! - the shifted-force cutoff moves the dimer's equilibrium: at rest at
//!   `2^{1/6}σ` the force is `−F_LJ(r_c)` along the bond, which the
//!   energy-shifted form does not have.
//!
//! No expected value is produced by calling the code under test.

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::molecular_dynamics::{MdError, PeriodicBox, VelocityVerlet};
use alice_physics::pair_potential::{
    LennardJones, Morse, PairPotential, PairPotentialError, ShiftMode, Truncated,
};

fn fx(x: f64) -> Fix128 {
    Fix128::from_f64(x)
}

fn f(x: Fix128) -> f64 {
    if x.is_negative() {
        -(-x).to_f64()
    } else {
        x.to_f64()
    }
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn norm(v: Vec3Fix) -> f64 {
    (f(v.x).powi(2) + f(v.y).powi(2) + f(v.z).powi(2)).sqrt()
}

// ---------------------------------------------------------------------------
// Dimer frequency from the discrete invariant
// ---------------------------------------------------------------------------

/// Run a dimer along `x` released at `r_eq + amp`, return the fitted slope
/// `K` of `μ v_rel² = C − K x²`, `x = r − r_eq`.
fn fitted_stiffness<P: PairPotential + Clone>(
    potential: Truncated<P>,
    r_eq: f64,
    amp: f64,
    masses: (f64, f64),
    h: f64,
    steps: usize,
) -> f64 {
    let (m1, m2) = masses;
    let mu = m1 * m2 / (m1 + m2);
    let mut md = VelocityVerlet::new(
        potential,
        PeriodicBox::cubic(Fix128::from_int(12)).unwrap(),
        vec![v3(3.0, 6.0, 6.0), v3(3.0 + r_eq + amp, 6.0, 6.0)],
        vec![Vec3Fix::ZERO; 2],
        vec![fx(m1), fx(m2)],
    )
    .unwrap();
    let (mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0);
    for _ in 0..steps {
        md.step(fx(h)).unwrap();
        let p = md.positions();
        let d = md.periodic_box().minimum_image(p[1] - p[0]);
        let x = norm(d) - r_eq;
        let v = md.velocities();
        let vrel = f(v[1].x) - f(v[0].x);
        let (xx, y) = (x * x, mu * vrel * vrel);
        sx += xx;
        sy += y;
        sxx += xx * xx;
        sxy += xx * y;
    }
    let n = steps as f64;
    -(n * sxy - sx * sy) / (n * sxx - sx * sx)
}

/// oracle: the LJ dimer's fitted stiffness is `k (1 − (ωh)²/4)` with
/// `k = 72ε/(2^{1/3}σ²)`, `ω = √(k/μ)`. At `ωh ≈ 0.063` the correction is
/// `1.0·10⁻³`; the anharmonic bias of the fit is second order in the
/// amplitude, `(U''' a / k)² ≈ 3·10⁻⁶` at `a = 10⁻⁴σ`. Tolerance `5e-5`
/// separates "dispersion included" from both "omitted" (`1e-3`) and a wrong
/// curvature. Halving `h` divides the correction by 4 (accepted
/// `[3.5, 4.5]`).
#[test]
fn lj_dimer_stiffness_from_discrete_invariant() {
    let (eps, sig) = (0.8, 1.2);
    let (m1, m2) = (2.0, 5.0);
    let mu = m1 * m2 / (m1 + m2);
    let k = 72.0 * eps / (2f64.powf(1.0 / 3.0) * sig * sig);
    let omega = (k / mu).sqrt();
    let r_eq = 2f64.powf(1.0 / 6.0) * sig;
    let lj = || {
        Truncated::new(
            LennardJones::new(fx(eps), fx(sig)).unwrap(),
            fx(3.0 * sig),
            ShiftMode::EnergyShift,
        )
        .unwrap()
    };
    let h = 0.0632 / omega;
    let k1 = fitted_stiffness(lj(), r_eq, 1e-4 * sig, (m1, m2), h, 3000);
    let k2 = fitted_stiffness(lj(), r_eq, 1e-4 * sig, (m1, m2), h / 2.0, 6000);
    let want1 = 1.0 - (omega * h).powi(2) / 4.0;
    let want2 = 1.0 - (omega * h / 2.0).powi(2) / 4.0;
    assert!(
        (k1 / k - want1).abs() < 5e-5,
        "K/k = {} vs 1 − (ωh)²/4 = {want1}",
        k1 / k
    );
    assert!(
        (k2 / k - want2).abs() < 5e-5,
        "K/k = {} vs {want2} (h/2)",
        k2 / k
    );
    let ratio = (1.0 - k1 / k) / (1.0 - k2 / k);
    assert!((3.5..4.5).contains(&ratio), "dispersion ratio {ratio}");
}

/// oracle: the same for a Morse dimer, `k = 2Da²` at `r_e`
/// (`D = 1.3, a = 1.7, r_e = 0.95`, `m = (1, 1)`, `μ = 1/2`).
#[test]
fn morse_dimer_stiffness_from_discrete_invariant() {
    let (d, a, re) = (1.3, 1.7, 0.95);
    let k = 2.0 * d * a * a;
    let omega = (k / 0.5_f64).sqrt();
    let morse = Truncated::new(
        Morse::new(fx(d), fx(a), fx(re)).unwrap(),
        fx(4.0),
        ShiftMode::EnergyShift,
    )
    .unwrap();
    let h = 0.05 / omega;
    let got = fitted_stiffness(morse, re, 1e-4, (1.0, 1.0), h, 4000);
    let want = 1.0 - (omega * h).powi(2) / 4.0;
    assert!((got / k - want).abs() < 3e-5, "K/k = {} vs {want}", got / k);
}

/// oracle: a dimer at rest at `2^{1/6}σ` feels no force under the
/// energy-shifted cutoff (`|F| < 1e-13`), and under the shifted-force cutoff
/// it feels `F_FS = F_LJ(r_min) − F_LJ(r_c) = −F_LJ(r_c)`, which is
/// repulsive (`F_LJ(r_c) < 0`) with magnitude
/// `|24ε(2(σ/r_c)¹² − (σ/r_c)⁶)/r_c|` (`r_c = 2.5σ`), equal and opposite on
/// the two particles: the shifted-force minimum lies outside `2^{1/6}σ`.
#[test]
fn force_shift_moves_the_dimer_equilibrium() {
    let (eps, sig) = (1.1, 0.9);
    let rc = 2.5 * sig;
    let r_eq = 2f64.powf(1.0 / 6.0) * sig;
    let s6 = (sig / rc).powi(6);
    let f_rc = 24.0 * eps * (2.0 * s6 * s6 - s6) / rc;
    assert!(f_rc < 0.0);
    for (mode, want) in [
        (ShiftMode::EnergyShift, 0.0),
        (ShiftMode::ForceShift, -f_rc),
    ] {
        let md = VelocityVerlet::new(
            Truncated::new(LennardJones::new(fx(eps), fx(sig)).unwrap(), fx(rc), mode).unwrap(),
            PeriodicBox::cubic(Fix128::from_int(6)).unwrap(),
            vec![v3(1.0, 2.0, 3.0), v3(1.0 + r_eq, 2.0, 3.0)],
            vec![Vec3Fix::ZERO; 2],
            vec![fx(1.0), fx(1.0)],
        )
        .unwrap();
        let fr = md.forces();
        // radial force on particle 0 (d = x0 − x1 = −r x̂): F_r d/r → −F_r on x
        let f_on_0 = f(fr[0].x);
        assert!(
            (f_on_0 - (-want)).abs() < 1e-12,
            "{mode:?}: F on 0 = {f_on_0} vs {}",
            -want
        );
        assert!(
            (f(fr[1].x) + f_on_0).abs() < 1e-17,
            "{mode:?}: not opposite"
        );
        assert!(f(fr[0].y).abs() < 1e-17 && f(fr[0].z).abs() < 1e-17);
    }
}

// ---------------------------------------------------------------------------
// Morse cluster: energy order, momentum, reversibility, Galilean invariance
// ---------------------------------------------------------------------------

fn morse_cluster(drift: [f64; 3]) -> VelocityVerlet<Morse> {
    let pot = Truncated::new(
        Morse::new(fx(1.0), fx(2.0), fx(1.1)).unwrap(),
        fx(2.5),
        ShiftMode::ForceShift,
    )
    .unwrap();
    let pos = vec![
        v3(2.0, 2.0, 2.0),
        v3(3.1, 2.1, 1.9),
        v3(2.4, 3.05, 2.2),
        v3(2.6, 2.5, 3.1),
        v3(3.4, 3.2, 2.9),
    ];
    let base = [
        [0.3, -0.1, 0.2],
        [-0.2, 0.25, -0.05],
        [0.1, -0.3, 0.15],
        [-0.15, 0.05, -0.25],
        [0.05, 0.2, 0.1],
    ];
    let vel = base
        .iter()
        .map(|v| v3(v[0] + drift[0], v[1] + drift[1], v[2] + drift[2]))
        .collect();
    let masses = vec![fx(1.0), fx(2.5), fx(0.6), fx(1.4), fx(3.0)];
    VelocityVerlet::new(
        pot,
        PeriodicBox::new(v3(6.0, 6.5, 7.0)).unwrap(),
        pos,
        vel,
        masses,
    )
    .unwrap()
}

/// Max `|E − E₀|` and max `|P − P₀|` over a run of total time `2.0`.
fn run_cluster(h: f64) -> (f64, f64) {
    let mut md = morse_cluster([0.0; 3]);
    let e0 = f(md.total_energy());
    let p0 = md.momentum();
    let steps = (2.0 / h).round() as usize;
    let (mut de, mut dp) = (0.0_f64, 0.0_f64);
    for _ in 0..steps {
        md.step(fx(h)).unwrap();
        de = de.max((f(md.total_energy()) - e0).abs());
        dp = dp.max(norm(md.momentum() - p0));
    }
    (de, dp)
}

/// oracle: velocity Verlet's energy error is `O(h²)`: the ratio of the
/// maximum deviations at `h = 0.01` and `0.005` is in `[3.5, 4.5]`; and
/// momentum is conserved to rounding (`< 1e-15`) with unequal masses.
#[test]
fn morse_cluster_energy_error_is_second_order_and_momentum_conserved() {
    let (e1, p1) = run_cluster(0.01);
    let (e2, p2) = run_cluster(0.005);
    assert!(e1 > 0.0 && e1 < 1e-2, "energy error {e1:e}");
    let ratio = e1 / e2;
    assert!(
        (3.5..4.5).contains(&ratio),
        "ratio {ratio} ({e1:e}, {e2:e})"
    );
    assert!(p1 < 1e-15 && p2 < 1e-15, "momentum drift {p1:e} {p2:e}");
}

/// oracle: forward 600 steps, negate velocities, 600 steps: back at the
/// start (minimum-image distance `< 1e-12`) with negated velocities.
#[test]
fn md_is_time_reversible() {
    let start = morse_cluster([0.0; 3]);
    let mut md = start.clone();
    for _ in 0..600 {
        md.step(fx(0.01)).unwrap();
    }
    let bx = *md.periodic_box();
    let moved = norm(bx.minimum_image(md.positions()[0] - start.positions()[0]));
    assert!(moved > 0.1, "barely moved: {moved}");
    let reversed: Vec<Vec3Fix> = md.velocities().iter().map(|v| -*v).collect();
    let mut back = VelocityVerlet::new(
        *md.potential(),
        bx,
        md.positions().to_vec(),
        reversed,
        md.masses().to_vec(),
    )
    .unwrap();
    for _ in 0..600 {
        back.step(fx(0.01)).unwrap();
    }
    for k in 0..5 {
        let dx = norm(bx.minimum_image(back.positions()[k] - start.positions()[k]));
        let dv = norm(back.velocities()[k] + start.velocities()[k]);
        assert!(dx < 1e-12, "particle {k}: Δx {dx:e}");
        assert!(dv < 1e-12, "particle {k}: Δv {dv:e}");
    }
}

/// oracle: Galilean invariance. With `u = (0.7, −0.4, 0.25)` added to every
/// velocity, after 300 steps of `h = 0.01` every particle sits at the
/// undrifted position plus `u·3` (minimum image, `< 1e-12`), with velocity
/// `+u`; the potential energy is the same.
#[test]
fn uniform_drift_does_not_change_relative_motion() {
    let u = [0.7, -0.4, 0.25];
    let mut a = morse_cluster([0.0; 3]);
    let mut b = morse_cluster(u);
    for _ in 0..300 {
        a.step(fx(0.01)).unwrap();
        b.step(fx(0.01)).unwrap();
    }
    let bx = *a.periodic_box();
    let shift = v3(3.0 * u[0], 3.0 * u[1], 3.0 * u[2]);
    for k in 0..5 {
        let dx = norm(bx.minimum_image(b.positions()[k] - a.positions()[k] - shift));
        assert!(dx < 1e-12, "particle {k}: Δx {dx:e}");
        let dv = norm(b.velocities()[k] - a.velocities()[k] - v3(u[0], u[1], u[2]));
        assert!(dv < 1e-12, "particle {k}: Δv {dv:e}");
    }
    assert!((f(a.potential_energy()) - f(b.potential_energy())).abs() < 1e-12);
}

// ---------------------------------------------------------------------------
// Degenerate input
// ---------------------------------------------------------------------------

fn lj_cut() -> Truncated<LennardJones> {
    Truncated::new(
        LennardJones::new(Fix128::ONE, Fix128::ONE).unwrap(),
        fx(2.5),
        ShiftMode::EnergyShift,
    )
    .unwrap()
}

/// Two particles at `x = 0` and `x = L` are the same point of the periodic
/// box: construction fails with `Potential { i: 0, j: 1, NonPositiveDistance }`;
/// a zero mass, `L < 2 r_c` and mismatched lengths have their own errors.
#[test]
fn coincident_through_the_boundary_and_invalid_systems() {
    let bx = PeriodicBox::cubic(Fix128::from_int(10)).unwrap();
    let err = VelocityVerlet::new(
        lj_cut(),
        bx,
        vec![v3(0.0, 4.0, 4.0), v3(5.0, 5.0, 5.0), v3(10.0, 4.0, 4.0)],
        vec![Vec3Fix::ZERO; 3],
        vec![Fix128::ONE; 3],
    )
    .err();
    assert_eq!(
        err,
        Some(MdError::Potential {
            i: 0,
            j: 2,
            error: PairPotentialError::NonPositiveDistance
        })
    );
    let err = VelocityVerlet::new(
        lj_cut(),
        bx,
        vec![v3(1.0, 1.0, 1.0), v3(5.0, 5.0, 5.0)],
        vec![Vec3Fix::ZERO; 2],
        vec![Fix128::ONE, Fix128::ZERO],
    )
    .err();
    assert_eq!(err, Some(MdError::NonPositiveMass));
    let err = VelocityVerlet::new(
        lj_cut(),
        PeriodicBox::cubic(fx(4.99)).unwrap(),
        vec![v3(1.0, 1.0, 1.0)],
        vec![Vec3Fix::ZERO],
        vec![Fix128::ONE],
    )
    .err();
    assert_eq!(err, Some(MdError::BoxTooSmall));
    let err = VelocityVerlet::new(
        lj_cut(),
        bx,
        vec![v3(1.0, 1.0, 1.0)],
        vec![],
        vec![Fix128::ONE],
    )
    .err();
    assert_eq!(err, Some(MdError::LengthMismatch));
    assert_eq!(
        PeriodicBox::new(v3(1.0, 0.0, 1.0)).err(),
        Some(MdError::NonPositiveBoxLength)
    );
}

/// `dt = 0` and `dt < 0` are `NonPositiveTimestep` and leave the state
/// untouched; one particle flies freely and wraps exactly
/// (`9.5 + 1.0·1.0 → 0.5` in a box of 10); `N = 0` steps without error; a
/// pair exactly at `r_c` has zero force and zero energy; temperature of one
/// particle is `TooFewParticles`.
#[test]
fn time_step_and_trivial_systems() {
    let bx = PeriodicBox::cubic(Fix128::from_int(10)).unwrap();
    let mut md = VelocityVerlet::new(
        lj_cut(),
        bx,
        vec![v3(9.5, 3.0, 3.0)],
        vec![v3(1.0, 0.0, -0.25)],
        vec![fx(2.0)],
    )
    .unwrap();
    let before = (md.positions().to_vec(), md.velocities().to_vec());
    for dt in [Fix128::ZERO, fx(-0.1)] {
        assert_eq!(md.step(dt), Err(MdError::NonPositiveTimestep));
        assert_eq!((md.positions().to_vec(), md.velocities().to_vec()), before);
    }
    md.step(Fix128::ONE).unwrap();
    assert_eq!(md.positions()[0], v3(0.5, 3.0, 2.75));
    assert_eq!(md.velocities()[0], v3(1.0, 0.0, -0.25));
    assert_eq!(
        md.instantaneous_temperature(Fix128::ONE),
        Err(MdError::TooFewParticles)
    );

    let mut empty = VelocityVerlet::new(lj_cut(), bx, vec![], vec![], vec![]).unwrap();
    empty.step(fx(0.1)).unwrap();
    assert_eq!(empty.total_energy(), Fix128::ZERO);

    let at_cut = VelocityVerlet::new(
        lj_cut(),
        bx,
        vec![v3(1.0, 1.0, 1.0), v3(3.5, 1.0, 1.0)],
        vec![Vec3Fix::ZERO; 2],
        vec![Fix128::ONE; 2],
    )
    .unwrap();
    assert_eq!(at_cut.forces(), &[Vec3Fix::ZERO, Vec3Fix::ZERO]);
    assert_eq!(at_cut.potential_energy(), Fix128::ZERO);
    // the same pair through the boundary: 9.0 and 1.5 are 2.5 apart
    let through = VelocityVerlet::new(
        lj_cut(),
        bx,
        vec![v3(9.0, 1.0, 1.0), v3(1.5, 1.0, 1.0)],
        vec![Vec3Fix::ZERO; 2],
        vec![Fix128::ONE; 2],
    )
    .unwrap();
    assert_eq!(through.forces(), &[Vec3Fix::ZERO, Vec3Fix::ZERO]);
    // and just inside the cutoff through the boundary: attractive, toward
    // the image (particle 0 at 9.0 is pulled in +x toward the image of 1.25 at 11.25)
    let inside = VelocityVerlet::new(
        lj_cut(),
        bx,
        vec![v3(9.0, 1.0, 1.0), v3(1.25, 1.0, 1.0)],
        vec![Vec3Fix::ZERO; 2],
        vec![Fix128::ONE; 2],
    )
    .unwrap();
    let s6 = (1.0_f64 / 2.25).powi(6);
    let f_r = 24.0 * (2.0 * s6 * s6 - s6) / 2.25;
    assert!(f_r < 0.0);
    assert!(
        (f(inside.forces()[0].x) + f_r).abs() < 1e-14,
        "F_x on 0 = {} vs {}",
        f(inside.forces()[0].x),
        -f_r
    );
}
