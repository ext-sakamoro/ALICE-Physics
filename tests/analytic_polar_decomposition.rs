//! Oracle for the polar decomposition `F = R·U`.
//!
//! This is the one primitive a co-rotational formulation needs and the crate
//! does not have: `Mat3Fix` carries `determinant` / `inverse` / `transpose` /
//! `mul_mat`, but nothing that separates a deformation gradient's rotation from
//! its stretch. `modal` is a closed form, not an eigensolver, and there is no
//! SVD anywhere.
//!
//! # Why this file exists before the implementation
//!
//! `tests/analytic_large_rotation.rs` is the **integration** gate: it says a
//! rigid rotation must carry no stress. If the polar factor were wrong it would
//! still red, but it would not say *where*. These are the unit-level oracles, so
//! a red here points at the decomposition and a red there points at the
//! co-rotational assembly.
//!
//! # The method being pinned
//!
//! Higham's Newton iteration for the orthogonal polar factor:
//!
//! ```text
//! R₀ = F,   R_{k+1} = ½ (R_k + R_k⁻ᵀ)
//! ```
//!
//! It needs only inverse, transpose, scale and addition — no square root, no
//! trigonometry, no eigensolver — so every operation is one `Fix128` is exact
//! in, and the result is bit-identical on every target. Convergence is
//! quadratic, so the number of correct bits doubles per step.
//!
//! # Properties, and which of them are definitional
//!
//! Each test below states a property that follows from the *definition* of the
//! polar decomposition rather than from a numeric coincidence, because a
//! coincidence can hold for a wrong implementation. The
//! sharpest is `shear_is_where_gram_schmidt_and_polar_disagree`: orthonormalising
//! the columns also produces an orthogonal factor, and on a rotation it produces
//! the *same* one, so a Gram-Schmidt implementation would pass most of these
//! tests. What separates them is that only the polar factor leaves a
//! **symmetric** remainder.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
// The oracle values are closed-form f64 evaluations, not simulation state.
#![allow(clippy::disallowed_methods)]

use alice_physics::math::{Fix128, Mat3Fix, PolarError, Vec3Fix};

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

/// Matrix from rows, which is how the closed forms below are written.
fn rows(r0: [f64; 3], r1: [f64; 3], r2: [f64; 3]) -> Mat3Fix {
    Mat3Fix::from_cols(
        Vec3Fix::new(fx(r0[0]), fx(r1[0]), fx(r2[0])),
        Vec3Fix::new(fx(r0[1]), fx(r1[1]), fx(r2[1])),
        Vec3Fix::new(fx(r0[2]), fx(r1[2]), fx(r2[2])),
    )
}

fn col(m: Mat3Fix, i: usize) -> Vec3Fix {
    match i {
        0 => m.col0,
        1 => m.col1,
        _ => m.col2,
    }
}

fn at(m: Mat3Fix, row: usize, column: usize) -> f64 {
    let c = col(m, column);
    match row {
        0 => c.x.to_f64(),
        1 => c.y.to_f64(),
        _ => c.z.to_f64(),
    }
}

/// Largest absolute component difference between two matrices, **through `f64`**.
///
/// ⚠️ **This instrument cannot see a difference smaller than `2⁻⁵³`.** `f64` carries
/// 53 mantissa bits and `Fix128` carries 64 fractional bits, so `1 − 2⁻⁶⁴` and `1.0`
/// are the *same* `f64` and this function reports `0.0` for them. During development
/// a probe built on it reported "idempotent to 0.00 ulp" for a pair that
/// `assert_eq!` rejected, and that wrong reading was reported once before being
/// retracted.
///
/// Use it only for tolerances at or above `1e-15`. **Any claim of bit equality, or
/// any bound counted in units in the last place, must go through
/// [`max_diff_ulps`] or `assert_eq!` instead.**
fn max_diff(a: Mat3Fix, b: Mat3Fix) -> f64 {
    let mut worst = 0.0_f64;
    for r in 0..3 {
        for c in 0..3 {
            worst = worst.max((at(a, r, c) - at(b, r, c)).abs());
        }
    }
    worst
}

/// Largest absolute component.
fn max_abs(a: Mat3Fix) -> f64 {
    let mut worst = 0.0_f64;
    for r in 0..3 {
        for c in 0..3 {
            worst = worst.max(at(a, r, c).abs());
        }
    }
    worst
}

/// Raw two's-complement value of a `Fix128`, in units of `2⁻⁶⁴`.
fn raw(v: Fix128) -> i128 {
    ((v.hi as i128) << 64) | v.lo as i128
}

/// Largest component difference between two matrices, **in units in the last
/// place**, computed on the raw representation.
///
/// This is the instrument `max_diff` cannot be: it sees every bit `Fix128` holds.
fn max_diff_ulps(a: Mat3Fix, b: Mat3Fix) -> u128 {
    let mut worst = 0_u128;
    for i in 0..3 {
        let (ca, cb) = (col(a, i), col(b, i));
        for (x, y) in [(ca.x, cb.x), (ca.y, cb.y), (ca.z, cb.z)] {
            worst = worst.max(raw(x).abs_diff(raw(y)));
        }
    }
    worst
}

/// Largest deviation from symmetry, `max |mᵢⱼ − mⱼᵢ|`.
fn asymmetry(m: Mat3Fix) -> f64 {
    let mut worst = 0.0_f64;
    for r in 0..3 {
        for c in (r + 1)..3 {
            worst = worst.max((at(m, r, c) - at(m, c, r)).abs());
        }
    }
    worst
}

/// Step budget handed to [`Mat3Fix::polar_rotation`].
///
/// Measured worst case over the gradients in this file is **21** steps
/// (`diag(1, 1, 1e-4)`), with `rotation ∘ diag(1.3, 0.9, 1.05)` at 20; the
/// implementation's doc carries the full table. 32 leaves margin without hiding a
/// non-convergence — a budget of 16 was tried first and rejected those two, which
/// is how the table came to be measured rather than assumed.
const MAX_ITERS: u32 = 32;

/// The settling bound the implementation uses, in units of `2⁻⁶⁴`.
///
/// Mirrored here rather than exported, so the oracle states the number it is
/// checking instead of importing it and agreeing with itself by construction.
const SETTLING_ULPS: u128 = 4;

/// The determinant floor the implementation documents for a caller that wants
/// only the intrinsic numerical criterion: `ε · ‖F‖³` with `ε = 2⁻¹⁹`.
///
/// The band was fixed before the value: `1e-6` at the bottom (below it the
/// determinant's own three multiplications stop being the limiting error) and
/// `1e-3` at the top (above it a legitimately flattened element would be
/// refused). `2⁻¹⁹ ≈ 1.907e-6` is the smallest power of two inside that band, so
/// it is exact in `Fix128` and was not chosen to fit a measurement.
fn intrinsic_floor(f: Mat3Fix) -> Fix128 {
    let scale = f.max_abs_component();
    Fix128::from_raw(0, 1 << 45) * scale * scale * scale
}

/// [`Mat3Fix::polar_rotation`] with the documented intrinsic floor and budget.
fn polar(f: Mat3Fix) -> Result<Mat3Fix, PolarError> {
    f.polar_rotation(intrinsic_floor(f), MAX_ITERS)
}

/// The stretch factor implied by a rotation: `U = Rᵀ F`.
fn stretch(f: Mat3Fix, r: Mat3Fix) -> Mat3Fix {
    r.transpose().mul_mat(f)
}

/// Gram-Schmidt orthonormalisation of the columns — the *wrong* answer this
/// file has to distinguish the polar factor from.
///
/// This is the QR factor, not the polar factor. It is written here rather than
/// in the crate precisely because it must never be mistaken for one.
fn gram_schmidt(f: Mat3Fix) -> Mat3Fix {
    let norm = |v: Vec3Fix| {
        (v.x.to_f64() * v.x.to_f64() + v.y.to_f64() * v.y.to_f64() + v.z.to_f64() * v.z.to_f64())
            .sqrt()
    };
    let dot = |a: Vec3Fix, b: Vec3Fix| {
        a.x.to_f64() * b.x.to_f64() + a.y.to_f64() * b.y.to_f64() + a.z.to_f64() * b.z.to_f64()
    };
    let scale_f = |v: Vec3Fix, s: f64| {
        Vec3Fix::new(
            fx(v.x.to_f64() * s),
            fx(v.y.to_f64() * s),
            fx(v.z.to_f64() * s),
        )
    };
    let sub = |a: Vec3Fix, b: Vec3Fix| Vec3Fix::new(a.x - b.x, a.y - b.y, a.z - b.z);

    let q0 = scale_f(f.col0, 1.0 / norm(f.col0));
    let p1 = sub(f.col1, scale_f(q0, dot(q0, f.col1)));
    let q1 = scale_f(p1, 1.0 / norm(p1));
    let p2a = sub(f.col2, scale_f(q0, dot(q0, f.col2)));
    let p2 = sub(p2a, scale_f(q1, dot(q1, f.col2)));
    let q2 = scale_f(p2, 1.0 / norm(p2));
    Mat3Fix::from_cols(q0, q1, q2)
}

// ---------------------------------------------------------------------------
// exactly representable rotations
// ---------------------------------------------------------------------------

/// The 24 rotations of the cube, whose entries are all `0` or `±1` and so are
/// exact in `Fix128`.
///
/// Built as signed axis permutations with determinant `+1`, which is the
/// definition of the octahedral rotation group.
fn octahedral_rotations() -> Vec<Mat3Fix> {
    let mut out = Vec::new();
    const PERMS: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    for perm in PERMS {
        for signs in 0..8u8 {
            let s = [
                if signs & 1 == 0 { 1.0 } else { -1.0 },
                if signs & 2 == 0 { 1.0 } else { -1.0 },
                if signs & 4 == 0 { 1.0 } else { -1.0 },
            ];
            // row i of the matrix is ±e_{perm[i]}
            let mut r = [[0.0_f64; 3]; 3];
            for i in 0..3 {
                r[i][perm[i]] = s[i];
            }
            let m = rows(r[0], r[1], r[2]);
            if m.determinant().to_f64() > 0.5 {
                out.push(m);
            }
        }
    }
    assert_eq!(
        out.len(),
        24,
        "the octahedral rotation group has 24 elements"
    );
    out
}

/// A rotation whose entries are exactly representable is already its own polar
/// factor, and the iteration must return it **bit for bit**.
///
/// For an orthogonal `F`, `F⁻ᵀ = F`, so the first Higham step is `½(F + F) = F`
/// with no rounding anywhere. Anything less than bit equality means the
/// iteration is perturbing an exact answer.
#[test]
fn exactly_representable_rotations_are_returned_unchanged() {
    for (n, f) in octahedral_rotations().into_iter().enumerate() {
        let r = polar(f).unwrap_or_else(|e| panic!("rotation {n} was rejected: {e:?}"));
        assert_eq!(
            r, f,
            "rotation {n}: an exactly representable rotation is its own polar factor, so the \
             iteration must not move it. F⁻ᵀ = F holds exactly here, so ½(F + F⁻ᵀ) = F needs no \
             rounding"
        );
    }
}

/// A rotation that is *not* exactly representable is recovered to the
/// arithmetic's precision.
///
/// `cos θ = 3/5`, `sin θ = 4/5` about z: the ratios are exact rationals but not
/// dyadic, so `Fix128` holds them to `2⁻⁶⁴` and `F` is orthogonal only to that
/// accuracy. The polar factor must come back within the same order.
#[test]
fn a_general_rotation_is_recovered() {
    let (c, s) = (3.0 / 5.0, 4.0 / 5.0);
    for f in [
        rows([c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]),
        rows([c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]),
        rows([1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]),
    ] {
        let r = polar(f).expect("a rotation must decompose");
        let d = max_diff(r, f);
        eprintln!("  general rotation: max |R − F| = {d:.3e}");
        assert!(
            d < 1e-15,
            "a rotation must be its own polar factor; got max |R − F| = {d:.3e}"
        );
    }
}

// ---------------------------------------------------------------------------
// the defining properties
// ---------------------------------------------------------------------------

/// Deformation gradients to run every property against.
///
/// Chosen to cover the cases that separate correct from nearly correct: pure
/// stretch, stretch plus rotation, simple shear (where Gram-Schmidt diverges),
/// and a strongly anisotropic stretch (where the iteration has to work).
fn deformation_gradients() -> Vec<(&'static str, Mat3Fix)> {
    let (c, s) = (3.0 / 5.0, 4.0 / 5.0);
    vec![
        (
            "pure stretch diag(1.5, 0.8, 1.1)",
            rows([1.5, 0.0, 0.0], [0.0, 0.8, 0.0], [0.0, 0.0, 1.1]),
        ),
        (
            "simple shear γ = 0.4 in xy",
            rows([1.0, 0.4, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
        ),
        (
            "rotation ∘ stretch",
            rows([c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]).mul_mat(rows(
                [1.3, 0.0, 0.0],
                [0.0, 0.9, 0.0],
                [0.0, 0.0, 1.05],
            )),
        ),
        (
            "anisotropic diag(4, 1, 0.25)",
            rows([4.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.25]),
        ),
        (
            "general full matrix",
            rows([1.2, 0.3, -0.1], [-0.2, 1.1, 0.25], [0.05, -0.15, 0.95]),
        ),
    ]
}

/// `R·U` must give `F` back, with `U = Rᵀ F`. This is the decomposition itself.
#[test]
fn the_factors_reconstruct_the_original() {
    for (name, f) in deformation_gradients() {
        let r = polar(f).unwrap_or_else(|e| panic!("{name}: rejected with {e:?}"));
        let u = stretch(f, r);
        let back = r.mul_mat(u);
        let d = max_diff(back, f);
        eprintln!("  {name}: max |R·U − F| = {d:.3e}");
        assert!(
            d < 1e-14,
            "{name}: R·(Rᵀ F) must be F; got max difference {d:.3e}"
        );
    }
}

/// `R` must be orthogonal with determinant `+1` — a rotation, not a reflection.
#[test]
fn the_rotation_factor_is_a_rotation() {
    for (name, f) in deformation_gradients() {
        let r = polar(f).unwrap_or_else(|e| panic!("{name}: rejected with {e:?}"));
        let should_be_identity = r.transpose().mul_mat(r);
        let d = max_diff(should_be_identity, Mat3Fix::IDENTITY);
        let det = r.determinant().to_f64();
        eprintln!("  {name}: max |RᵀR − I| = {d:.3e}, det R = {det:.15}");
        assert!(
            d < 1e-14,
            "{name}: RᵀR must be I; got max deviation {d:.3e}"
        );
        assert!(
            (det - 1.0).abs() < 1e-14,
            "{name}: det R must be +1, not {det:.15}. A determinant of −1 is a reflection, \
             which would flip the sign of every stress component"
        );
    }
}

/// `U` must be symmetric and positive definite. **This is the property that
/// defines the polar decomposition** and the one an orthonormalisation does not
/// have.
///
/// Positive definiteness is checked by Sylvester's criterion on the leading
/// minors, which needs no eigenvalues.
#[test]
fn the_stretch_factor_is_symmetric_positive_definite() {
    for (name, f) in deformation_gradients() {
        let r = polar(f).unwrap_or_else(|e| panic!("{name}: rejected with {e:?}"));
        let u = stretch(f, r);

        let a = asymmetry(u);
        eprintln!("  {name}: max |Uᵢⱼ − Uⱼᵢ| = {a:.3e}");
        assert!(
            a < 1e-14,
            "{name}: U = Rᵀ F must be symmetric; got max asymmetry {a:.3e}. Asymmetry means R \
             is not the polar factor — an orthonormalisation of the columns gives an \
             orthogonal R with a non-symmetric remainder"
        );

        // Sylvester: the three leading principal minors must be positive.
        let m1 = at(u, 0, 0);
        let m2 = at(u, 0, 0) * at(u, 1, 1) - at(u, 0, 1) * at(u, 1, 0);
        let m3 = u.determinant().to_f64();
        eprintln!("  {name}: leading minors {m1:.6}, {m2:.6}, {m3:.6}");
        for (k, minor) in [m1, m2, m3].into_iter().enumerate() {
            assert!(
                minor > 0.0,
                "{name}: leading minor {} of U is {minor:.6e}, so U is not positive definite",
                k + 1
            );
        }
    }
}

/// A symmetric positive definite `F` has no rotation in it, so `R` must be the
/// identity.
#[test]
fn symmetric_positive_definite_input_gives_the_identity() {
    for (name, f) in [
        (
            "diag",
            rows([2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.5]),
        ),
        (
            "general spd",
            rows([2.0, 0.3, 0.1], [0.3, 1.5, -0.2], [0.1, -0.2, 1.8]),
        ),
    ] {
        let r = polar(f).unwrap_or_else(|e| panic!("{name}: rejected with {e:?}"));
        let d = max_diff(r, Mat3Fix::IDENTITY);
        eprintln!("  {name}: max |R − I| = {d:.3e}");
        assert!(
            d < 1e-14,
            "{name}: a symmetric positive definite F carries no rotation, so R must be I; got \
             max |R − I| = {d:.3e}"
        );
    }
}

// ---------------------------------------------------------------------------
// the discriminator
// ---------------------------------------------------------------------------

/// **The test that separates the polar factor from an orthonormalisation.**
///
/// Gram-Schmidt on the columns also yields an orthogonal matrix, and on a
/// rotation it yields the *same* one, so a Gram-Schmidt implementation would
/// pass every test above except the symmetry of `U`. Simple shear is where the
/// two answers part company, and the separation is definitional rather than
/// numeric:
///
/// - the polar factor leaves `Rᵀ F` **symmetric**;
/// - Gram-Schmidt leaves `Qᵀ F` upper triangular and **not** symmetric, because
///   it never rotates the first column at all.
///
/// So this asserts both sides: the polar remainder is symmetric, the
/// Gram-Schmidt remainder is not, and the two rotations differ by a stated
/// margin. If someone replaces the implementation with an orthonormalisation,
/// this reds and the others do not.
#[test]
fn shear_is_where_gram_schmidt_and_polar_disagree() {
    let gamma = 0.4_f64;
    let f = rows([1.0, gamma, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);

    let r = polar(f).expect("shear must decompose");
    let q = gram_schmidt(f);

    let u_polar = stretch(f, r);
    let u_qr = stretch(f, q);
    let a_polar = asymmetry(u_polar);
    let a_qr = asymmetry(u_qr);
    let separation = max_diff(r, q);

    eprintln!("  polar        : asymmetry of Rᵀ F = {a_polar:.3e}");
    eprintln!("  gram-schmidt : asymmetry of Qᵀ F = {a_qr:.6}");
    eprintln!("  max |R − Q|  = {separation:.6}");

    assert!(
        a_polar < 1e-14,
        "the polar remainder must be symmetric; got {a_polar:.3e}"
    );
    assert!(
        a_qr > 0.1,
        "Gram-Schmidt on a shear of γ = {gamma} must leave a visibly non-symmetric remainder, \
         or this test is not separating the two factorisations; got {a_qr:.3e}"
    );
    assert!(
        separation > 0.1,
        "the polar factor and the Gram-Schmidt factor must differ on a shear, or the \
         implementation may be an orthonormalisation; got max |R − Q| = {separation:.3e}"
    );
}

// ---------------------------------------------------------------------------
// rejection, on a relative criterion
// ---------------------------------------------------------------------------

/// An inverted deformation has no rotation factor, and must be rejected rather
/// than answered with a reflection.
///
/// `det F < 0` means the element has been turned inside out. The nearest
/// orthogonal matrix is then a reflection with `det = −1`, and returning it
/// would silently flip every stress component — the same failure mode as an
/// `.abs()` that hides a sign, one layer up.
#[test]
fn inverted_deformation_is_rejected() {
    for (name, f) in [
        (
            "single axis flip",
            rows([-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
        ),
        (
            "flip with stretch",
            rows([1.2, 0.0, 0.0], [0.0, -0.9, 0.0], [0.0, 0.0, 1.1]),
        ),
        (
            "reflection composed with shear",
            rows([1.0, 0.3, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]),
        ),
    ] {
        let det = f.determinant().to_f64();
        assert!(
            det < 0.0,
            "{name}: the scene must have det F < 0, got {det}"
        );
        assert_eq!(
            polar(f),
            Err(PolarError::Inverted),
            "{name}: det F = {det:.6} < 0, so there is no rotation factor and the refusal must \
             say so. Returning the nearest orthogonal matrix would return a reflection \
             (det = −1) and flip every stress component; reporting it as Degenerate or \
             NotConverged would send the caller to the floor or the budget instead of to the \
             mesh"
        );
    }
}

/// The degeneracy criterion must be **relative**, so scaling the whole matrix
/// cannot change the verdict.
///
/// This is the test an absolute floor fails, and the one `det.is_zero()` fails
/// too: `Mat3Fix::inverse` only rejects an exactly zero determinant, so a nearly
/// flat `F` comes back with an enormous inverse and the iteration diverges
/// (a fixed absolute threshold always breaks somewhere).
///
/// `det` scales as the cube of the matrix, so the scale-free quantity is
/// `|det F| / ‖F‖³`. Scaling `F` by `k` must leave accept/reject alone.
#[test]
fn the_degeneracy_criterion_is_scale_invariant() {
    for flatness in [1.0_f64, 1e-1, 1e-2, 1e-4, 1e-6, 1e-8, 1e-12] {
        let base = rows([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, flatness]);
        let verdict = polar(base).is_ok();
        if !verdict {
            assert_eq!(
                polar(base),
                Err(PolarError::Degenerate),
                "flatness {flatness:.0e} must be refused by the floor, not by the budget or \
                 an intermediate singularity"
            );
        }
        for k in [1e-3_f64, 1.0, 1e3] {
            let scaled = base.scale(fx(k));
            let scaled_verdict = polar(scaled).is_ok();
            assert_eq!(
                scaled_verdict, verdict,
                "flatness {flatness:.0e} scaled by {k:.0e}: the verdict changed from \
                 {verdict} to {scaled_verdict}. |det F| / ‖F‖³ is scale free, so a relative \
                 criterion cannot depend on k — an absolute floor can"
            );
        }
        eprintln!(
            "  flatness {flatness:.0e}: |det|/‖F‖³ = {:.3e}, accepted = {verdict}",
            base.determinant().to_f64().abs() / max_abs(base).powi(3)
        );
    }
}

/// Applying the decomposition to its own output must not move it by more than the
/// settling bound.
///
/// Per the analytic-oracle rule a precision parameter must not change the answer,
/// and the iteration count is that parameter. Idempotence measures it without a
/// second entry point: a converged `R` is already a rotation, so decomposing it
/// again should return it.
///
/// ⚠️ **The bound is `SETTLING_ULPS`, not bit equality, and that is a property of
/// the arithmetic rather than a concession.** `Fix128` multiplication truncates, so
/// the iteration has no exact fixed point — it creeps by one unit in the last place
/// every one or two steps for ever (the implementation's doc carries the 24-step
/// trace). Requiring `again == r` asks for something the map cannot provide, and
/// trying it burned the whole budget and surfaced as a spurious refusal to
/// decompose.
///
/// The measured deviation is printed, and
/// `a_wrong_iteration_exceeds_the_settling_bound` is the destruction test that
/// shows the bound is not wide enough to admit a wrong implementation.
#[test]
fn the_decomposition_is_idempotent() {
    for (name, f) in deformation_gradients() {
        let r = polar(f).unwrap_or_else(|e| panic!("{name}: rejected with {e:?}"));
        let again = polar(r).unwrap_or_else(|e| panic!("{name}: the rotation factor gave {e:?}"));
        let ulps = max_diff_ulps(again, r);
        eprintln!("  {name}: |polar(R) − R| = {ulps} ulp");
        assert!(
            ulps <= SETTLING_ULPS,
            "{name}: decomposing the rotation again moved it by {ulps} ulp, above the settling \
             bound of {SETTLING_ULPS}. Either the iteration stopped early or the drift of the \
             truncating map is larger than measured"
        );
    }
}

// ---------------------------------------------------------------------------
// destruction test for the settling bound
// ---------------------------------------------------------------------------

/// Smallest power of two at or above a positive `v` — the divisor the
/// implementation uses, replicated so the reference iteration below starts from
/// the same place.
fn ceil_power_of_two(v: Fix128) -> Fix128 {
    let r = raw(v);
    assert!(r > 0, "the divisor is only defined for a positive scale");
    let raw_u = r as u128;
    let bits = 128 - raw_u.leading_zeros();
    let floor_pow = 1_u128 << (bits - 1);
    let pow = if raw_u == floor_pow || bits >= 127 {
        floor_pow
    } else {
        floor_pow << 1
    };
    Fix128 {
        hi: (pow >> 64) as i64,
        lo: pow as u64,
    }
}

fn add_mat(a: Mat3Fix, b: Mat3Fix) -> Mat3Fix {
    Mat3Fix::from_cols(a.col0 + b.col0, a.col1 + b.col1, a.col2 + b.col2)
}

/// Higham's iteration, written here so it can be written **wrong** on purpose.
///
/// `coefficient` is `½` in the correct scheme and `settle_bound` is the change at
/// which it stops. Returns the iterate and the number of steps taken.
fn reference_higham(f: Mat3Fix, coefficient: Fix128, settle_bound: Fix128) -> (Mat3Fix, u32) {
    let divisor = ceil_power_of_two(f.max_abs_component());
    let mut r = if divisor == Fix128::ONE {
        f
    } else {
        Mat3Fix::from_cols(f.col0 / divisor, f.col1 / divisor, f.col2 / divisor)
    };
    let mut steps = 0_u32;
    while steps < MAX_ITERS {
        let inverse = r.inverse().expect("the reference scenes are invertible");
        let next = add_mat(r, inverse.transpose()).scale(coefficient);
        steps += 1;
        let change = add_mat(next, r.scale(Fix128::NEG_ONE)).max_abs_component();
        r = next;
        if change <= settle_bound {
            return (r, steps);
        }
    }
    (r, steps)
}

/// **The destruction test for the settling bound.**
///
/// A bound of four units in the last place is only worth writing if a wrong
/// iteration misses it. Two deliberate faults are injected into a reference
/// implementation and each must land **outside** the bound:
///
/// Two faults are injected into a reference implementation:
///
/// 1. **a settling bound 2⁴⁴ times too loose** (`2⁻²⁰` instead of `4·2⁻⁶⁴`) — the
///    shape a plausible-looking "relative tolerance" would take;
/// 2. **the averaging coefficient perturbed from `½` to `0.51`** — the smallest
///    error a transcription can introduce.
///
/// The faithful reference must land inside the bound on every gradient, which is
/// what makes the failures attributable to the faults rather than to the reference
/// being a different algorithm.
///
/// ⚠️ **The loose bound is caught on some gradients and not others, and that
/// scene dependence is the reason this test asserts across the set rather than per
/// scene.** Measured: `diag(1.5, 0.8, 1.1)` loses one step and comes out **0 ulp**
/// away, while `simple shear γ = 0.4` loses two steps and comes out **3089 ulp**
/// away. Quadratic convergence is why the first is insensitive — the step that
/// first brings the change under `2⁻²⁰` has already produced an iterate the next
/// step barely moves — and whether that happens depends on where the scene's
/// change sequence straddles the loose bound.
///
/// ⚠️ Two wrong conclusions were drawn from single scenes before this form: first
/// that the loose bound "cannot be caught", then that the bound "is not
/// load-bearing for accuracy". Both came from reading gradient one and stopping.
/// **A destruction test needs the fault caught somewhere in the set, and the set
/// has to be looked at before the conclusion is written**.
///
/// ⚠️ **"Drop the final step" is not a usable fault at all**, measured at 0 ulp
/// everywhere and necessarily so: the iteration stops *because* the last step moved
/// it by no more than the bound, so the iterate before it is inside the bound by
/// definition. "The last step did not matter" is what the property asserts, not a
/// violation of it.
#[test]
fn a_wrong_iteration_exceeds_the_settling_bound() {
    let half = Fix128 { hi: 0, lo: 1 << 63 };
    let correct_bound = Fix128 {
        hi: 0,
        lo: SETTLING_ULPS as u64,
    };
    // 2^-20 — a bound that reads like a reasonable relative tolerance and is
    // 2^44 times too loose for this arithmetic.
    let loose_bound = Fix128 { hi: 0, lo: 1 << 44 };
    // 0.51 = 51/100, the smallest plausible transcription error in the coefficient
    let wrong_coefficient = Fix128::from_ratio(51, 100);

    let mut worst_loose = 0_u128;
    for (name, f) in deformation_gradients() {
        let good = polar(f).unwrap_or_else(|e| panic!("{name}: rejected with {e:?}"));

        let (faithful, steps) = reference_higham(f, half, correct_bound);
        let d_faithful = max_diff_ulps(faithful, good);
        let (loose, loose_steps) = reference_higham(f, half, loose_bound);
        let d_loose = max_diff_ulps(loose, good);
        let (skewed, _) = reference_higham(f, wrong_coefficient, correct_bound);
        let d_skewed = max_diff_ulps(skewed, good);

        eprintln!(
            "  {name}: {steps} steps | faithful {d_faithful} ulp | bound 2⁻²⁰ \
             ({loose_steps} steps) {d_loose} ulp | coefficient 0.51 {d_skewed} ulp"
        );

        assert!(
            d_faithful <= SETTLING_ULPS,
            "{name}: the faithful reference differs from the implementation by {d_faithful} ulp, \
             above the bound of {SETTLING_ULPS}. The two are then not the same algorithm and the \
             faults below prove nothing"
        );
        worst_loose = worst_loose.max(d_loose);
        assert!(
            d_skewed > SETTLING_ULPS,
            "{name}: perturbing the averaging coefficient from 0.5 to 0.51 changed the answer by \
             only {d_skewed} ulp, inside the bound of {SETTLING_ULPS}. The bound is then not \
             measuring the iteration at all"
        );
    }

    assert!(
        worst_loose > SETTLING_ULPS,
        "no gradient in the set distinguished a settling bound of 2⁻²⁰ from {SETTLING_ULPS}·2⁻⁶⁴ \
         (worst difference {worst_loose} ulp). The bound would then be unobservable in the answer \
         on every scene here, and its width could only be justified by termination — add a scene \
         whose change sequence straddles the loose bound before trusting the tight one"
    );
    eprintln!("  worst loose-bound deviation across the set: {worst_loose} ulp");
}
