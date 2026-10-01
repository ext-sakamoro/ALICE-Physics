//! Analytic oracles for the deterministic coupling channel.
//!
//! Every expected value below comes from a closed form, never from calling the
//! implementation. The three field operations have exact references:
//!
//! - **trilinear interpolation** reproduces an affine function exactly, so the
//!   expectation is the affine function itself, evaluated directly;
//! - **splat** is the transpose of sample, so the expectation is
//!   `sample(g, p)` computed through a different code path, and the deposited
//!   total is one times the splatted value because the trilinear weights sum
//!   to one;
//! - **diffusion** of `cos(k x)` on a grid with reflective boundaries is an
//!   exact eigenmode of the discrete Laplacian, so the amplitude after `M`
//!   explicit-Euler steps is `(1 - ν dt λ_h)^M` with
//!   `λ_h = 2 (1 - cos(k h)) / h²`, which converges to the analytic
//!   `exp(-ν k² t)` at second order in `h`.
//!
//! Each null-ish result carries a vacuity guard: a decay test also asserts the
//! amplitude actually moved, a "they now agree" test also asserts they
//! disagreed beforehand, and the bit-exact tests also assert the probes are
//! pairwise distinct. A test that would pass on an all-zero field measures
//! nothing.
//!
//! Author: Moroya Sakamoto

#![allow(clippy::float_cmp)]

use alice_physics::coupled_field::{reconcile_mean, CoupledField, CoupledScalar};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::phase_change::{PhaseChangeConfig, PhaseChangeModifier};
use alice_physics::sim_field::ScalarField3D;
use alice_physics::sim_modifier::PhysicsModifier;
use alice_physics::thermal::{ThermalConfig, ThermalModifier};

// ============================================================================
// Helpers
// ============================================================================

/// Raw I64F64 bit pattern as a signed integer, so differences count ulps.
fn raw(v: Fix128) -> i128 {
    ((v.hi as i128) << 64) | (v.lo as i128)
}

/// Distance between two fixed-point values in units of the last place (2⁻⁶⁴).
fn ulps(a: Fix128, b: Fix128) -> i128 {
    (raw(a) - raw(b)).abs()
}

fn q(num: i64, den: i64) -> Fix128 {
    Fix128::from_ratio(num, den)
}

fn p3(x: Fix128, y: Fix128, z: Fix128) -> Vec3Fix {
    Vec3Fix::new(x, y, z)
}

/// `5 x 5 x 5` nodes over `[0, 4]³`, so the node spacing is exactly one and
/// every coordinate below is a dyadic rational — all the arithmetic in the
/// interpolation is then exact and the oracles can assert on bits.
fn dyadic_grid() -> CoupledField {
    CoupledField::try_new(
        5,
        5,
        5,
        (Fix128::ZERO, Fix128::ZERO, Fix128::ZERO),
        (
            Fix128::from_int(4),
            Fix128::from_int(4),
            Fix128::from_int(4),
        ),
    )
    .expect("5x5x5 is a valid grid")
}

/// Coefficients of the affine oracle `f(x, y, z) = a + b x + c y + d z`.
/// All dyadic, all different, none zero — a coefficient that is zero or shared
/// with another axis would let an axis swap pass unnoticed.
const A: (i64, i64) = (3, 4);
const B: (i64, i64) = (1, 2);
const C: (i64, i64) = (-1, 4);
const D: (i64, i64) = (1, 8);

fn affine(x: Fix128, y: Fix128, z: Fix128) -> Fix128 {
    q(A.0, A.1) + q(B.0, B.1) * x + q(C.0, C.1) * y + q(D.0, D.1) * z
}

/// Fill a grid with the affine oracle evaluated at its nodes.
fn fill_affine(f: &mut CoupledField) {
    for iz in 0..f.nz() {
        for iy in 0..f.ny() {
            for ix in 0..f.nx() {
                let v = affine(
                    Fix128::from_int(ix as i64),
                    Fix128::from_int(iy as i64),
                    Fix128::from_int(iz as i64),
                );
                f.set(ix, iy, iz, v);
            }
        }
    }
}

/// Probe points inside `[0, 4]³`, all dyadic, with the three fractional parts
/// different from each other at every probe so that swapping two of the
/// interpolation weights changes the answer.
fn probes() -> [Vec3Fix; 4] {
    [
        p3(q(3, 8), q(13, 8), q(23, 8)),
        p3(q(9, 8), q(7, 2), q(1, 4)),
        p3(q(11, 4), q(1, 8), q(27, 8)),
        p3(q(63, 16), q(33, 16), q(7, 4)),
    ]
}

// ============================================================================
// 1. Trilinear interpolation against the affine closed form
// ============================================================================

/// Trilinear interpolation reproduces an affine function exactly.
///
/// The oracle is the affine function itself. Trilinear interpolation of nodal
/// samples of `a + b x + c y + d z` equals that function at every point of the
/// cell — the bilinear and trilinear cross terms all carry a zero coefficient.
/// On this grid the node spacing is one and every coordinate and coefficient
/// is a dyadic rational with few significant bits, so every intermediate
/// product is exactly representable in I64F64 and the agreement is on bits,
/// not within a tolerance.
#[test]
fn trilinear_reproduces_an_affine_function_bit_exactly() {
    let mut field = dyadic_grid();
    fill_affine(&mut field);

    let mut seen = Vec::new();
    for p in probes() {
        let want = affine(p.x, p.y, p.z);
        let got = field.sample(p);
        assert_eq!(
            got, want,
            "trilinear did not reproduce the affine oracle at \
             ({:?}, {:?}, {:?}): got {} want {} ({} ulps)",
            p.x, p.y, p.z, got, want, 0
        );
        seen.push(got);
    }

    // Vacuity: the probes must not all land on the same value, or "it matched"
    // would only say that a constant matched a constant.
    for i in 0..seen.len() {
        for j in (i + 1)..seen.len() {
            assert_ne!(
                seen[i], seen[j],
                "probes {i} and {j} sample the same value; the oracle above is \
                 not distinguishing anything"
            );
        }
    }
    // Vacuity: the field itself is not constant.
    assert_ne!(
        field.get(0, 0, 0),
        field.get(4, 4, 4),
        "the affine field is constant; nothing is being interpolated"
    );
}

/// The gradient of an affine field is its coefficient vector.
///
/// Central differences over one cell are exact for an affine field, so the
/// oracle is `(b, c, d)` directly. Probes are kept one half-cell away from
/// every face, where the stencil is symmetric.
#[test]
fn gradient_of_an_affine_field_is_the_coefficient_vector_bit_exactly() {
    let mut field = dyadic_grid();
    fill_affine(&mut field);

    let want = p3(q(B.0, B.1), q(C.0, C.1), q(D.0, D.1));
    // Vacuity: the three coefficients differ, so an axis swap cannot pass.
    assert_ne!(want.x, want.y);
    assert_ne!(want.y, want.z);
    assert_ne!(want.x, want.z);

    for p in [
        p3(q(11, 8), q(21, 8), q(9, 8)),
        p3(q(1, 2), q(7, 2), q(5, 2)),
        p3(q(27, 8), q(3, 4), q(25, 8)),
    ] {
        // Every arm of the one-cell stencil has to stay inside `[0, 4]`; the
        // clamp in `sample` otherwise shortens one side and the difference is
        // no longer central. (A probe at z = 29/8 fails here for that reason,
        // which is the intended behaviour, not a defect.)
        for c in [p.x, p.y, p.z] {
            assert!(
                c >= q(1, 2) && c <= q(7, 2),
                "probe {c} is too close to a face"
            );
        }
        let g = field.gradient(p);
        assert_eq!(g.x, want.x, "d/dx at {:?}", p.x);
        assert_eq!(g.y, want.y, "d/dy at {:?}", p.y);
        assert_eq!(g.z, want.z, "d/dz at {:?}", p.z);
    }
}

// ============================================================================
// 2. Splat: conservation and adjointness
// ============================================================================

/// Rounding envelope for the total deposited by one `splat` of `value`.
///
/// Each of the eight weights is a product of three factors in `[0, 1]`, built
/// with two multiplications; `Fix128` multiplication truncates downward, so a
/// weight is short of its exact value by at most 2 ulp. Each contribution
/// `value * w` truncates once more (≤ 1 ulp) and inherits `|value| * 2` ulp
/// from the weight. Addition into the grid is exact. Summing the eight:
///
/// ```text
/// |deposited - value| <= 8 * (1 + 2 * ceil(|value|)) ulp
/// ```
fn splat_bound(value: Fix128) -> i128 {
    let magnitude = value.abs().ceil().hi.max(1) as i128;
    8 * (1 + 2 * magnitude)
}

/// A splat deposits exactly the value it was given, up to the derived bound.
///
/// The oracle is the identity `Σ w_i = 1` for trilinear weights, which holds
/// for any point; the only departure is fixed-point truncation, bounded by
/// [`splat_bound`].
#[test]
fn splat_conserves_the_total_within_the_derived_rounding_bound() {
    // Deliberately non-dyadic fractions, so every weight needs rounding.
    let point = p3(q(4, 3), q(11, 7), q(23, 11));

    for value in [Fix128::ONE, q(7, 2), q(-5, 3), Fix128::from_int(12)] {
        let mut field = dyadic_grid();
        field.splat(point, value);

        let total = field.sum();
        let bound = splat_bound(value);
        let err = ulps(total, value);
        assert!(
            err <= bound,
            "splat of {value} deposited {total} ({err} ulps off, bound {bound})"
        );

        // Vacuity: the mass really was spread over a whole cell. If the point
        // had landed on a node, or the weights had collapsed, "the total is
        // right" would be true of a single-node write as well.
        let touched = field.as_slice().iter().filter(|v| !v.is_zero()).count();
        assert_eq!(
            touched, 8,
            "expected all eight corners of one cell to receive mass, got \
             {touched}"
        );
    }
}

/// `splat` is the transpose of `sample`.
///
/// Splatting one unit at `p` into an empty field produces the vector of
/// trilinear weights; its inner product with any field `g` is by definition
/// the trilinear interpolation of `g` at `p`. The oracle is therefore
/// `sample(g, p)`, reached through an unrelated code path (nested lerps rather
/// than eight weighted terms).
///
/// Bound: the dot product costs ≤ 1 ulp of truncation per term plus
/// `|g_i| * 2` ulp inherited from each weight (8 terms, `|g| <= 8` here, so
/// ≤ 136 ulp), and the nested-lerp `sample` accumulates ≤ 7 ulp over its three
/// levels. 256 ulp (≈ 1.4e-17) leaves room for the interaction of the two
/// without loosening the test in any way that matters: a transposition error
/// in the weights moves the answer by order 0.1, seventeen decades away.
#[test]
fn splat_is_the_adjoint_of_sample() {
    // A field with no symmetry, so that swapped weights cannot cancel out.
    let mut g = dyadic_grid();
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    for iz in 0..g.nz() {
        for iy in 0..g.ny() {
            for ix in 0..g.nx() {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let hi = (state >> 61) as i64 - 4; // -4 ..= 3
                g.set(ix, iy, iz, Fix128::from_raw(hi, state));
            }
        }
    }
    // Vacuity: |g| really is within the bound's assumption, and g is not flat.
    assert!(g.as_slice().iter().all(|v| v.abs().hi.abs() <= 8));
    assert_ne!(g.get(0, 0, 0), g.get(1, 0, 0));

    for point in [
        p3(q(4, 3), q(11, 7), q(23, 11)),
        p3(q(1, 5), q(19, 6), q(7, 9)),
        p3(q(29, 8), q(2, 13), q(15, 4)),
    ] {
        let mut weights = dyadic_grid();
        weights.splat(point, Fix128::ONE);

        let dot = weights
            .as_slice()
            .iter()
            .zip(g.as_slice().iter())
            .fold(Fix128::ZERO, |acc, (w, v)| acc + *w * *v);

        let want = g.sample(point);
        let err = ulps(dot, want);
        assert!(
            err <= 256,
            "adjointness broken at {point:?}: <splat(p), g> = {dot}, \
             sample(g, p) = {want}, {err} ulps apart"
        );
    }
}

// ============================================================================
// 3. Diffusion against the discrete and the continuous closed forms
// ============================================================================

/// Grid, time step and step count for one refinement level of the 1D
/// diffusion study. `n` nodes over `[0, 1]`, `dt = 2 h²`, `T = 1/2`.
struct Level {
    n: usize,
    h: f64,
    dt: Fix128,
    steps: usize,
}

fn level(n: usize) -> Level {
    let h = 1.0 / (n - 1) as f64;
    // dt = 2 h² keeps ν dt / h² = 1/4 at every level (ν = 1/8), inside the
    // explicit-Euler 1D stability limit of 1/2, and makes the time error O(h²)
    // like the space error.
    let denom = (n - 1) * (n - 1) / 2; // 2h² = 2 / (n-1)² = 1 / ((n-1)²/2)
    let dt = Fix128::from_ratio(1, denom as i64);
    let steps = denom / 2; // T = 1/2 = steps * dt
    Level { n, h, dt, steps }
}

const NU: (i64, i64) = (1, 8);

/// `cos(x)` for `0 <= x <= pi`, from its Maclaurin series.
///
/// `det_math` exposes no deterministic `f64` cosine, and `clippy.toml` bans
/// `f64::cos` because the platform `libm` is not bit-exact across targets.
/// This series uses only `+`, `-`, `*` and `/`, which IEEE 754 requires to be
/// correctly rounded, so the reference is identical on every platform — which
/// is exactly the property the ban exists to protect.
///
/// The argument is first reflected into `[0, pi/2]`, where the alternating
/// series has no significant cancellation (largest term 1.24 against a result
/// in `[0, 1]`); at that range 20 terms are past `f64` resolution, the last
/// being `(pi/2)^40 / 40! ~ 5e-41`.
fn cos_series(x: f64) -> f64 {
    assert!(
        (0.0..=core::f64::consts::PI + 1e-12).contains(&x),
        "cos_series is only conditioned for [0, pi], got {x}"
    );
    let (theta, sign) = if x > core::f64::consts::FRAC_PI_2 {
        (core::f64::consts::PI - x, -1.0)
    } else {
        (x, 1.0)
    };
    let t2 = theta * theta;
    let mut term = 1.0;
    let mut sum = 1.0;
    for n in 1..=20 {
        let k = (2 * n) as f64;
        term *= -t2 / (k * (k - 1.0));
        sum += term;
    }
    sign * sum
}

/// The series reference is itself checked against cosines that are known in
/// closed form, so the diffusion oracles do not rest on an unverified helper.
#[test]
fn cos_series_reproduces_the_closed_form_cosines() {
    use core::f64::consts::PI;
    for (x, want) in [
        (0.0, 1.0),
        (PI / 3.0, 0.5),
        (PI / 2.0, 0.0),
        (2.0 * PI / 3.0, -0.5),
        (PI, -1.0),
    ] {
        let got = cos_series(x);
        assert!(
            (got - want).abs() < 1e-15,
            "cos({x}) = {got}, closed form {want}"
        );
    }
    // Vacuity: the helper is not returning a constant.
    assert!(cos_series(0.0) - cos_series(PI) > 1.9);
}

/// One node row along X with reflective ends, initialised to `cos(k x)`.
fn cosine_row(n: usize) -> CoupledField {
    let mut f = CoupledField::try_new(
        n,
        1,
        1,
        (Fix128::ZERO, Fix128::ZERO, Fix128::ZERO),
        (Fix128::ONE, Fix128::ZERO, Fix128::ZERO),
    )
    .expect("a row of nodes is a valid grid");
    let h = 1.0 / (n - 1) as f64;
    for ix in 0..n {
        let x = ix as f64 * h;
        f.set(
            ix,
            0,
            0,
            Fix128::from_f64(cos_series(core::f64::consts::PI * x)),
        );
    }
    f
}

/// Amplitude factor of the first Neumann mode after `steps` explicit-Euler
/// steps, in closed form: `(1 - ν dt λ_h)^steps`, `λ_h = 2(1-cos(k h))/h²`.
///
/// The power is a plain multiplication loop rather than `powi`, which is
/// banned for the same cross-platform reason as `cos`; repeated multiplication
/// is correctly rounded at every step.
fn discrete_decay(l: &Level) -> f64 {
    let k = core::f64::consts::PI;
    let lambda = 2.0 * (1.0 - cos_series(k * l.h)) / (l.h * l.h);
    let nu = NU.0 as f64 / NU.1 as f64;
    let dt = l.dt.to_f64();
    let per_step = 1.0 - nu * dt * lambda;
    let mut factor = 1.0;
    for _ in 0..l.steps {
        factor *= per_step;
    }
    factor
}

/// Amplitude factor of the same mode in the continuum: `exp(-ν k² T)`.
///
/// `det_math::exp64` is the crate's deterministic `f64` exponential; the
/// inherent `f64::exp` is banned by `clippy.toml`.
fn analytic_decay(total_time: f64) -> f64 {
    let k = core::f64::consts::PI;
    let nu = NU.0 as f64 / NU.1 as f64;
    alice_physics::det_math::exp64(-nu * k * k * total_time)
}

fn run_row(l: &Level) -> CoupledField {
    let mut f = cosine_row(l.n);
    let rate = q(NU.0, NU.1);
    for _ in 0..l.steps {
        f.diffuse(l.dt, rate);
    }
    f
}

/// The implemented stencil is the one the closed form describes.
///
/// `cos(π x)` sampled at the nodes is an exact eigenvector of the seven-point
/// Laplacian with reflective ghost nodes, so the whole profile is multiplied
/// by one scalar per step. The oracle is that scalar raised to the step count,
/// evaluated independently in `f64`; it pins the spacing, the stencil and the
/// boundary treatment at once, because getting any of the three wrong changes
/// the eigenvalue by a large factor rather than by rounding.
#[test]
fn diffusion_matches_the_closed_form_discrete_decay() {
    let l = level(17);
    let want = discrete_decay(&l);

    // Vacuity: the run must actually decay, and not to nothing. A factor of 1
    // (nothing happened) or 0 (everything died) would make the comparison
    // below meaningless.
    assert!(
        (0.1..0.9).contains(&want),
        "the closed-form factor {want} is degenerate; pick another T"
    );

    let before = cosine_row(l.n);
    let after = run_row(&l);

    let got = after.get(0, 0, 0).to_f64();
    assert!(
        (got - want).abs() < 1e-9,
        "amplitude after {} steps was {got}, closed form says {want}",
        l.steps
    );

    // The mode is preserved: every node decays by the same factor. Node 8 is
    // the zero crossing of cos(π x) on this grid and is skipped.
    for ix in 0..l.n {
        if ix == (l.n - 1) / 2 {
            continue;
        }
        let ratio = after.get(ix, 0, 0).to_f64() / before.get(ix, 0, 0).to_f64();
        assert!(
            (ratio - want).abs() < 1e-6,
            "node {ix} decayed by {ratio}, the mode decays by {want}; the \
             profile is not being preserved, so it is not an eigenmode"
        );
    }

    // Vacuity: the profile was not flat to begin with.
    assert!(
        before.get(0, 0, 0).to_f64() - before.get(l.n - 1, 0, 0).to_f64() > 1.5,
        "the initial profile is flat; nothing was diffusing"
    );
}

/// Refining the grid drives the result to the analytic exponential at second
/// order.
///
/// The oracle is `exp(-ν k² T)`, the solution of `∂T/∂t = ν ∂²T/∂x²` for the
/// mode `cos(k x)`. With `dt ∝ h²` both the space error `O(h²)` and the time
/// error `O(dt)` scale as `h²`, so halving `h` must quarter the error. The
/// The window is stated as the error *ratio* between consecutive levels
/// (`f64::log2` is banned for the same cross-platform reason as `cos`):
/// `[2^1.8, 2^2.2] = [3.482, 4.595]`. It is loose enough for the neglected
/// higher-order terms at these coarse levels and far too tight for a
/// first-order (ratio 2) or inconsistent (ratio 1) scheme.
#[test]
fn diffusion_converges_to_the_analytic_exponential_at_second_order() {
    let want = analytic_decay(0.5);
    let mut errors = Vec::new();

    for n in [9usize, 17, 33] {
        let l = level(n);
        assert_eq!(
            l.dt.to_f64() * l.steps as f64,
            0.5,
            "level {n} does not integrate to T = 1/2"
        );
        let got = run_row(&l).get(0, 0, 0).to_f64();
        errors.push((n, (got - want).abs()));
    }

    // Vacuity: the coarse level must be visibly wrong, or "it converged" is a
    // statement about rounding rather than about discretisation.
    assert!(
        errors[0].1 > 1e-4,
        "the coarsest level is already accurate to {:.3e}; there is no error \
         left to converge",
        errors[0].1
    );
    // Vacuity: the error is monotonically shrinking.
    assert!(
        errors[0].1 > errors[1].1 && errors[1].1 > errors[2].1,
        "errors are not decreasing under refinement: {errors:?}"
    );

    // 2^1.8 .. 2^2.2, i.e. "the error is between 3.48x and 4.60x smaller when
    // the grid is refined by two" — second order is exactly 4x.
    const LO: f64 = 3.482_202_253_184_6;
    const HI: f64 = 4.594_793_419_988_2;
    for w in errors.windows(2) {
        let (coarse_n, coarse) = w[0];
        let (fine_n, fine) = w[1];
        let ratio = coarse / fine;
        assert!(
            (LO..=HI).contains(&ratio),
            "error shrank by {ratio:.3}x between n={coarse_n} ({coarse:.3e}) \
             and n={fine_n} ({fine:.3e}); second order (4x, window \
             {LO:.3}..{HI:.3}) was expected"
        );
    }
}

// ============================================================================
// 4. The two temperature owners reconcile through the channel
// ============================================================================

const RES: usize = 6;
const MIN: (f32, f32, f32) = (-2.0, -2.0, -2.0);
const MAX: (f32, f32, f32) = (2.0, 2.0, 2.0);
const DT: f32 = 0.016;

fn diverged_pair() -> (ThermalModifier, PhaseChangeModifier) {
    let thermal_cfg = ThermalConfig {
        ambient_temperature: 20.0,
        ..ThermalConfig::default()
    };
    let phase_cfg = PhaseChangeConfig {
        ambient_temperature: 20.0,
        ..PhaseChangeConfig::default()
    };
    let mut thermal = ThermalModifier::new(thermal_cfg, RES, MIN, MAX);
    let mut phase = PhaseChangeModifier::new(phase_cfg, RES, MIN, MAX);
    // Heat only one of them, then let both run: they start from the same
    // ambient and end up disagreeing purely because nothing relates them.
    thermal.apply_heat_at(0.0, 0.0, 0.0, 3_000.0, 1.5);
    for _ in 0..20 {
        thermal.update(DT);
        phase.update(DT);
    }
    (thermal, phase)
}

/// After reconciling, a point has one temperature instead of two.
///
/// The oracle is the arithmetic mean of the two published fields, which is
/// what `reconcile_mean` is defined to produce; it is checked cell by cell
/// against a mean computed outside the channel.
#[test]
fn reconcile_gives_the_two_temperature_owners_one_field() {
    let (mut thermal, mut phase) = diverged_pair();
    let probe = (0.0_f32, 0.0_f32, 0.0_f32);

    let before_t = thermal.temperature_at(probe.0, probe.1, probe.2);
    let before_p = phase.temperature_at(probe.0, probe.1, probe.2);

    // Vacuity: the gap is real and large. Reconciling two equal fields proves
    // nothing.
    assert!(
        (before_t - before_p).abs() > 100.0,
        "the two owners already agree to within {} degrees before \
         reconciling; there is no gap to close",
        (before_t - before_p).abs()
    );

    // Independent expectation: the per-cell mean of the two f32 fields.
    let expected: Vec<f32> = thermal
        .temperature
        .data
        .iter()
        .zip(phase.temperature.data.iter())
        .map(|(a, b)| (Fix128::from_f32(*a) + Fix128::from_f32(*b)) / Fix128::from_int(2))
        .map(Fix128::to_f32)
        .collect();

    let mut channel = thermal.coupled_channel().expect("matching channel");
    {
        let mut participants: [&mut dyn CoupledScalar; 2] = [&mut thermal, &mut phase];
        reconcile_mean(&mut participants, &mut channel).expect("same grid");
    }

    assert_eq!(
        thermal.temperature.data, phase.temperature.data,
        "the two temperature fields still differ after reconciling"
    );
    assert_eq!(
        thermal.temperature.data, expected,
        "the agreed field is not the mean of what the two owners published"
    );

    let after_t = thermal.temperature_at(probe.0, probe.1, probe.2);
    let after_p = phase.temperature_at(probe.0, probe.1, probe.2);
    assert_eq!(
        after_t.to_bits(),
        after_p.to_bits(),
        "the probe still reads two different temperatures"
    );
    // The agreed value sits between what the two owners brought, which the
    // mean must and a stale copy of either field would not.
    let lo = before_t.min(before_p);
    let hi = before_t.max(before_p);
    assert!(
        after_t > lo && after_t < hi,
        "agreed temperature {after_t} is not strictly between the two inputs \
         ({lo}, {hi})"
    );
}

/// The agreed field does not depend on the order of the participants.
///
/// `Fix128` addition is exact, so the sum of the published fields is
/// order-independent and the single trailing division cannot reintroduce an
/// order dependence. Three participants with three different fields make the
/// division inexact (by three), which is where a running-average
/// implementation would diverge between orders.
#[test]
fn reconcile_is_independent_of_participant_order() {
    let build = || {
        let (thermal, phase) = diverged_pair();
        let mut third = ThermalModifier::new(
            ThermalConfig {
                ambient_temperature: 20.0,
                ..ThermalConfig::default()
            },
            RES,
            MIN,
            MAX,
        );
        third.apply_heat_at(1.0, 0.5, -0.5, 900.0, 1.0);
        for _ in 0..20 {
            third.update(DT);
        }
        (thermal, phase, third)
    };

    let (mut a1, mut b1, mut c1) = build();
    let (mut a2, mut b2, mut c2) = build();

    // Vacuity: the three participants must disagree, or any order gives the
    // same answer for trivial reasons.
    assert_ne!(a1.temperature.data, b1.temperature.data);
    assert_ne!(a1.temperature.data, c1.temperature.data);
    assert_ne!(b1.temperature.data, c1.temperature.data);

    let mut ch1 = a1.coupled_channel().expect("matching channel");
    {
        let mut order1: [&mut dyn CoupledScalar; 3] = [&mut a1, &mut b1, &mut c1];
        reconcile_mean(&mut order1, &mut ch1).expect("same grid");
    }

    let mut ch2 = a2.coupled_channel().expect("matching channel");
    {
        let mut order2: [&mut dyn CoupledScalar; 3] = [&mut c2, &mut b2, &mut a2];
        reconcile_mean(&mut order2, &mut ch2).expect("same grid");
    }

    assert_eq!(
        ch1.as_slice(),
        ch2.as_slice(),
        "the agreed field changed when the participants were reordered"
    );
    assert_eq!(a1.temperature.data, a2.temperature.data);
    assert_eq!(b1.temperature.data, b2.temperature.data);
    assert_eq!(c1.temperature.data, c2.temperature.data);
}

// ---------------------------------------------------------------------------
// What each ghost convention conserves
//
// The two `diffuse` implementations in the crate conserve *different* sums, and
// the difference only shows on a field that is non-zero **on** the boundary.
// Each test below asserts the invariant of its own convention and, as its
// vacuity guard, that the other sum moved: a scheme that froze the field, or
// one whose two sums happened to coincide, would fail the guard rather than
// pass the invariant.
//
// The weight is derived in the `coupled_field` module docs: the mirror ghost
// puts the boundary at the node, so an end node owns half a dual cell per end
// axis and the three-dimensional weight is `2⁻ᵇ`; the copy ghost puts it half a
// cell outside, so every node owns a full cell and the weight is one.
// ---------------------------------------------------------------------------

/// `2⁻ᵇ` for a node, `b` counting the axes on which it is an end.
///
/// A degenerate axis (`n == 1`) contributes no flux, so it is not an end for
/// this purpose and does not halve anything.
fn dual_weight(i: usize, n: usize) -> Fix128 {
    if n == 1 || (i != 0 && i != n - 1) {
        Fix128::ONE
    } else {
        q(1, 2)
    }
}

/// `Σ 2⁻ᵇ Tᵢ` over a [`CoupledField`], in `Fix128` so the sum is exact.
///
/// Halving is exact in binary and the additions are the same `Fix128` additions
/// `diffuse` performs, so this reference carries no error of its own and the
/// comparison below can be `assert_eq!` rather than a tolerance.
fn dual_weighted_sum(f: &CoupledField) -> Fix128 {
    let mut total = Fix128::ZERO;
    for iz in 0..f.nz() {
        for iy in 0..f.ny() {
            for ix in 0..f.nx() {
                let w = dual_weight(ix, f.nx()) * dual_weight(iy, f.ny()) * dual_weight(iz, f.nz());
                total = total + f.get(ix, iy, iz) * w;
            }
        }
    }
    total
}

/// A `n³` grid spanning `[0, 4]³`, so `h = 4 / (n − 1)` is dyadic for `n = 5`.
fn cube_grid(n: usize) -> CoupledField {
    CoupledField::try_new(
        n,
        n,
        n,
        (Fix128::ZERO, Fix128::ZERO, Fix128::ZERO),
        (
            Fix128::from_int(4),
            Fix128::from_int(4),
            Fix128::from_int(4),
        ),
    )
    .expect("a cube of at least two nodes per axis")
}

/// `CoupledField::diffuse` holds `Σ 2⁻ᵇ T` exactly and does **not** hold `Σ T`.
///
/// The mirror ghost is what makes this the conserved sum, and the plain sum is
/// the guard: if the scheme conserved both, or moved neither, one of the two
/// assertions fails. The heat starts on a face node, which is the only place
/// the two sums can disagree.
#[test]
fn the_mirror_ghost_conserves_the_dual_weighted_sum_and_not_the_plain_one() {
    let n = 5;
    let mut f = cube_grid(n);
    // Dyadic amplitude on a face node: every subsequent value stays dyadic, so
    // the exact comparison below is a statement about the scheme, not about
    // where the rounding happened to fall.
    f.set(0, n / 2, n / 2, Fix128::from_int(8));

    let dual_before = dual_weighted_sum(&f);
    let plain_before = f.sum();
    assert_eq!(
        dual_before,
        Fix128::from_int(4),
        "a face node owns half a cell"
    );
    assert_eq!(plain_before, Fix128::from_int(8));

    let dt = q(1, 16);
    for _ in 0..6 {
        f.diffuse(dt, Fix128::ONE);
    }

    assert_eq!(
        dual_weighted_sum(&f),
        dual_before,
        "the mirror ghost must hold the dual-weighted sum exactly; it moved by {} ulps",
        ulps(dual_weighted_sum(&f), dual_before)
    );
    assert_ne!(
        f.sum(),
        plain_before,
        "the plain sum did not move, so this scene cannot tell the two sums \
         apart and the invariant above is vacuous"
    );
}

/// `ScalarField3D::diffuse` holds the plain sum and does **not** hold
/// `Σ 2⁻ᵇ T` — the mirror of the test above.
///
/// `f32` cannot be exact, so the invariant is bounded rather than asserted
/// equal. The bound is derived, not fitted: one step touches each of the `n³`
/// nodes with a handful of `mul_add`s, so the sum of a positive field carries
/// at most a few `f32::EPSILON` of relative error per step, and `8 ε · steps`
/// is a generous envelope for that. The guard then asks the *other* sum to move
/// by a thousand times the same envelope, so the two cannot be confused by
/// rounding alone (measured here: drift at 1 % of the envelope, the other sum
/// at over three thousand times it).
#[test]
fn the_copy_ghost_conserves_the_plain_sum_and_not_the_dual_weighted_one() {
    let n = 5usize;
    let steps = 6usize;
    let mut f = ScalarField3D::new(n, n, n, (0.0, 0.0, 0.0), (4.0, 4.0, 4.0));
    // Non-dyadic amplitude, so the `f32` arithmetic really rounds and the bound
    // is doing work rather than describing an exact case.
    let amplitude = 7.3_f32;
    f.data[n * ((n / 2) + n * (n / 2))] = amplitude;

    let weights = |i: usize| if i == 0 || i == n - 1 { 0.5_f64 } else { 1.0 };
    let sums = |g: &ScalarField3D| -> (f64, f64) {
        let mut plain = 0.0;
        let mut dual = 0.0;
        for iz in 0..n {
            for iy in 0..n {
                for ix in 0..n {
                    let v = f64::from(g.data[ix + n * (iy + n * iz)]);
                    plain += v;
                    dual += v * weights(ix) * weights(iy) * weights(iz);
                }
            }
        }
        (plain, dual)
    };

    let (plain_before, dual_before) = sums(&f);
    assert!(plain_before > 0.0 && dual_before > 0.0);

    for _ in 0..steps {
        f.diffuse(1.0 / 64.0, 1.0);
    }
    let (plain_after, dual_after) = sums(&f);

    let envelope = 8.0 * f64::from(f32::EPSILON) * steps as f64;
    let plain_drift = (plain_after - plain_before).abs() / plain_before;
    let dual_change = (dual_after - dual_before).abs() / dual_before;

    assert!(
        plain_drift <= envelope,
        "the copy ghost must hold the plain sum to rounding: drift {plain_drift:.3e} \
         exceeds the envelope {envelope:.3e}"
    );
    assert!(
        dual_change >= 1000.0 * envelope,
        "the dual-weighted sum moved by only {dual_change:.3e}, under a thousand \
         envelopes ({:.3e}); this scene cannot separate the two conventions, so \
         the invariant above is vacuous",
        1000.0 * envelope
    );
}
