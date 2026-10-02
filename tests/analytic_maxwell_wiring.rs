//! Oracles for the wiring of the Yee lattice's sources, Gauss's law readers,
//! `∇·B` reader, absorber constructor and Courant bookkeeping — the twenty
//! items `examples/maxwell_sources_and_absorber.rs` drives.
//!
//! The solver core (dispersion relation, cavity modes, PML absorption, the
//! split wiring) is pinned by `tests/analytic_maxwell_fdtd.rs` and is **not**
//! re-derived here. This file pins what that one reads through but never
//! states: the stencils themselves, the accessors, the constants, and what
//! each reader does on a degenerate lattice.
//!
//! # Closed forms (none calls the implementation for its expectation)
//!
//! * **Node divergence** `div_e(i,j,k) = (Ex[i]−Ex[i−1]) + (Ey[j]−Ey[j−1]) +
//!   (Ez[k]−Ez[k−1])`, six hand-seeded edges; `gauss_residual = div_e − ρ`,
//!   so a zero field reads `−ρ` and a placement with `ρ = div_e` reads the
//!   zero bit pattern.
//! * **Gauss's law under the update**: `∇·(∇×H)` telescopes, so `∇·E − ρ` is
//!   a constant of the motion. With `S = 1/2` and an integer seed every
//!   product is a shift and the constant holds **to the bit**; an
//!   inconsistent placement carries its residual unchanged.
//! * **Cell divergence** `div_b(i,j,k) = (Hx[i+1]−Hx[i]) + (Hy[j+1]−Hy[j]) +
//!   (Hz[k+1]−Hz[k])`, by hand; and under the update it is the zero bit
//!   pattern while the arithmetic is exact (`S = 1/2`, integer seed) — the
//!   same lattice at `S = 9/16` shows the truncation residual the module
//!   header documents, which is what proves the zero is not vacuous.
//! * **`total_charge`** is `Σρ` with exact addition: dyadic placements
//!   `3/8 − 5/16 + 7/32 = 9/32` to the bit; writing a node twice replaces.
//! * **`COURANT_3D = 9/16`** bit for bit; **`cfl_limit_3d() = 1/√3`**, so
//!   `limit·√3` and `3·limit²` are `1` within the ULP the two truncations
//!   can lose, `9/16·√3 < 1` and the next dyadic `37/64·√3 > 1`.
//! * **`loss_coefficients`** with `σ = 2`, `S = 9/16`: `a = 9/16`,
//!   `ca = 7/25`, `cb = 9/25` (within 1 ULP, one truncating division each);
//!   with `σ = 4`, `S = 1/2`: `a = 1`, `(0, 1/4)` exactly; `S = 0` gives
//!   `(1, 0)` exactly.
//! * **`theoretical_pml_reflection`**: midpoint rule over the cubic profile,
//!   `depth = 3`, `σ_max = 5/2`: `Σt³ = (1+27+125)/216 = 17/24`,
//!   `R = exp(−2·5/2·17/24) = exp(−85/24)`; checked both bit-exactly against
//!   `Fix128::exp` of that rational and against `det_math::exp64` within the
//!   documented `Fix128::exp` accuracy; `depth = 1`: `R = exp(−σ/4)`.
//! * **`is_absorbing`** with a layer exactly half the axis: integer
//!   coordinates `0, 1` and `3, 4` are lossy and `2` is not, so the
//!   non-absorbing `Ex` samples are exactly `nx × 1 × 1`.
//!
//! # Degenerate input (each with the result the doc names)
//!
//! * zero dimension → panic in `new` and `new_with_absorber`;
//! * a lattice one cell thin on some axis has **no** interior node:
//!   `interior_node_dims` reports `0` there, `set_charge` / `charge` /
//!   `div_e` / `gauss_residual` panic, `max_abs_gauss_residual` and
//!   `total_charge` return zero (nothing to sum), `set_current` on a `1³`
//!   lattice panics because every edge is tangential to a PEC wall;
//! * charge index out of range (`0` or `n`) → panic, `div_b` out of range →
//!   panic, `is_absorbing` out of range → panic;
//! * `S = 0` freezes the field bit for bit;
//! * PML depth `0` on every axis with a non-zero `σ_max`, and non-zero depth
//!   with `σ_max = 0`, both leave no absorbing sample and step bit-identically
//!   to a plain lattice; depth exactly half the axis is accepted;
//! * `σ_max` large enough that the exponent is below `−44` returns exactly
//!   `0` (`Fix128::exp` contract); a negative `σ_max` reflects **more** than
//!   one, by the same formula;
//! * `σ·S/2 = −1` hits the pole of `loss_coefficients`: `Fix128`'s `/` by
//!   zero is `0`, so the pair is `(0, 0)` — pinned as measured, see the
//!   note on the test;
//! * a `Fix128` at the top of its range steps without panicking and wraps
//!   modulo `2¹²⁸`, with the wrapped bit pattern predicted by hand;
//! * `max_abs_field` over a sample at `Fix128`'s minimum reports `0`,
//!   because `|MIN|` is not representable — pinned as measured.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]

use alice_physics::det_math::{exp64, sqrt64};
use alice_physics::math::Fix128;
use alice_physics::maxwell_fdtd::{
    cfl_limit_3d, loss_coefficients, theoretical_pml_reflection, Absorber, Component, YeeGrid,
    COURANT_3D,
};
use std::panic::{catch_unwind, AssertUnwindSafe};

fn q(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn int(n: i64) -> Fix128 {
    Fix128::from_int(n)
}

fn raw(x: Fix128) -> i128 {
    (i128::from(x.hi) << 64) | i128::from(x.lo)
}

fn ulp_gap(a: Fix128, b: Fix128) -> i128 {
    (raw(a) - raw(b)).abs()
}

/// `assert!`/`panic!` with a plain string literal (no format arguments) boxes
/// a `&'static str`, not a `String` — only an interpolated message boxes a
/// `String`. A caller that only tries one of the two silently reads an empty
/// message on whichever kind it did not try.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> &str {
    payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("<non-string panic payload>")
}

const EVERY: [Component; 6] = [
    Component::Ex,
    Component::Ey,
    Component::Ez,
    Component::Hx,
    Component::Hy,
    Component::Hz,
];

fn assert_bit_identical(a: &YeeGrid, b: &YeeGrid, what: &str) {
    for c in EVERY {
        let (ni, nj, nk) = a.component_dims(c);
        for i in 0..ni {
            for j in 0..nj {
                for k in 0..nk {
                    assert_eq!(
                        a.get(c, i, j, k),
                        b.get(c, i, j, k),
                        "{what}: {c:?}[{i}][{j}][{k}]"
                    );
                }
            }
        }
    }
}

/// Largest `|H|`, so a "zero divergence" claim can be shown to be about a
/// field that is actually there.
fn max_abs_h(grid: &YeeGrid) -> Fix128 {
    let mut worst = Fix128::ZERO;
    for c in [Component::Hx, Component::Hy, Component::Hz] {
        let (ni, nj, nk) = grid.component_dims(c);
        for i in 0..ni {
            for j in 0..nj {
                for k in 0..nk {
                    worst = worst.max(grid.get(c, i, j, k).abs());
                }
            }
        }
    }
    worst
}

/// The `(a = 4, m = 2) ⊗ (a = 4, m = 2)` `Ez` cavity mode from the oracle
/// table of `analytic_maxwell_fdtd.rs`, an integer seed on a 4×4×4 box.
fn seed_integer_mode(grid: &mut YeeGrid) {
    let v = [0i64, 1, 0, -1, 0];
    for (i, &vi) in v.iter().enumerate() {
        for (j, &vj) in v.iter().enumerate() {
            for k in 0..4 {
                grid.set(Component::Ez, i, j, k, int(vi * vj));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// W1 — accessors and constants
// ---------------------------------------------------------------------------

/// `dims` and `courant` return the constructor's inputs, with and without an
/// absorber, and `component_dims` is the module-header table.
#[test]
fn dims_and_courant_return_the_constructor_inputs() {
    let plain = YeeGrid::new(3, 5, 7, q(3, 8));
    assert_eq!(plain.dims(), (3, 5, 7));
    assert_eq!(plain.courant(), q(3, 8));
    let lined = YeeGrid::new_with_absorber(
        6,
        4,
        8,
        q(5, 16),
        Absorber::GradedPml {
            depth: [1, 2, 3],
            sigma_max: int(4),
        },
    );
    assert_eq!(
        lined.dims(),
        (6, 4, 8),
        "the absorber must not change the cell count"
    );
    assert_eq!(lined.courant(), q(5, 16));
    // The staggering table: Ex has nx × (ny+1) × (nz+1), and so on.
    assert_eq!(plain.component_dims(Component::Ex), (3, 6, 8));
    assert_eq!(plain.component_dims(Component::Hx), (4, 5, 7));
}

/// `COURANT_3D` is `9/16` to the bit and `cfl_limit_3d` is `1/√3`.
///
/// `limit·√3` and `3·limit²` are each `1` up to the ULP that two truncating
/// operations can lose (measured: −1 and −2 ULP respectively, 2026-10-03;
/// the budget is 8). The bracket `9/16·√3 < 1 < 37/64·√3` is the statement
/// that `COURANT_3D` is the largest dyadic with denominator 16 under the
/// limit; `37/64` is the next dyadic up on a finer scale and is already over.
#[test]
fn courant_constant_and_cfl_limit_are_the_closed_forms() {
    assert_eq!(COURANT_3D, q(9, 16), "COURANT_3D must be exactly 9/16");
    let limit = cfl_limit_3d();
    let sqrt3 = int(3).sqrt();
    let gap_product = ulp_gap(limit * sqrt3, Fix128::ONE);
    assert!(
        gap_product <= 8,
        "limit * sqrt(3) is {gap_product} ULP from 1"
    );
    let gap_square = ulp_gap(limit * limit * int(3), Fix128::ONE);
    assert!(gap_square <= 8, "3 * limit^2 is {gap_square} ULP from 1");
    let as_f64 = limit.to_f64();
    let reference = 1.0 / sqrt64(3.0);
    assert!(
        (as_f64 - reference).abs() < 1e-15,
        "cfl_limit_3d = {as_f64}, 1/sqrt64(3) = {reference}"
    );
    assert!(COURANT_3D * sqrt3 < Fix128::ONE, "9/16 is under the limit");
    assert!(q(37, 64) * sqrt3 > Fix128::ONE, "37/64 is over the limit");
    assert!(COURANT_3D < limit && limit < q(37, 64));
}

/// `max_abs_field` is the largest seeded magnitude over all six components,
/// and zero on a fresh lattice.
#[test]
fn max_abs_field_is_the_largest_seeded_magnitude() {
    let mut grid = YeeGrid::new(4, 4, 4, COURANT_3D);
    assert_eq!(grid.max_abs_field(), Fix128::ZERO);
    // One sample per component, the largest magnitude negative and magnetic
    // so that a probe that only looked at E, or only at positives, fails.
    grid.set(Component::Ex, 1, 1, 1, q(3, 7));
    grid.set(Component::Ey, 1, 1, 1, q(-1, 3));
    grid.set(Component::Ez, 1, 2, 1, q(1, 2));
    grid.set(Component::Hx, 2, 2, 2, q(2, 9));
    grid.set(Component::Hy, 2, 2, 2, q(-11, 5));
    grid.set(Component::Hz, 2, 2, 2, q(7, 4));
    assert_eq!(grid.max_abs_field(), q(11, 5), "largest |value| is |-11/5|");
    grid.set(Component::Hy, 2, 2, 2, Fix128::ZERO);
    assert_eq!(
        grid.max_abs_field(),
        q(7, 4),
        "then 7/4 once -11/5 is cleared"
    );
}

// ---------------------------------------------------------------------------
// W2 — the stencils by hand
// ---------------------------------------------------------------------------

/// `div_e` is the six-edge node stencil and `gauss_residual` is `div_e − ρ`.
///
/// Six distinct non-dyadic values so that every term, its sign and its
/// position are observable; the hand sum is formed with the same exact
/// `Fix128` additions the stencil uses, so the comparison is to the bit.
#[test]
fn div_e_and_gauss_residual_are_the_six_edge_stencil_by_hand() {
    let mut grid = YeeGrid::new(4, 4, 4, COURANT_3D);
    let (i, j, k) = (2usize, 2usize, 2usize);
    let [exp, exm, eyp, eym, ezp, ezm] =
        [q(3, 7), q(-5, 11), q(2, 13), q(7, 3), q(-1, 17), q(4, 19)];
    grid.set(Component::Ex, i, j, k, exp);
    grid.set(Component::Ex, i - 1, j, k, exm);
    grid.set(Component::Ey, i, j, k, eyp);
    grid.set(Component::Ey, i, j - 1, k, eym);
    grid.set(Component::Ez, i, j, k, ezp);
    grid.set(Component::Ez, i, j, k - 1, ezm);
    let by_hand = (exp - exm) + (eyp - eym) + (ezp - ezm);
    assert_eq!(grid.div_e(i, j, k), by_hand, "div E at the node");
    assert_eq!(
        grid.gauss_residual(i, j, k),
        by_hand,
        "rho = 0 so the residual is div E"
    );

    let rho = q(9, 23);
    grid.set_charge(i, j, k, rho);
    assert_eq!(grid.charge(i, j, k), rho, "set then read");
    assert_eq!(grid.gauss_residual(i, j, k), by_hand - rho, "div E - rho");
    assert!(grid.max_abs_gauss_residual() > Fix128::ZERO);

    // Zero field, placed charge: the residual is exactly -rho.
    let mut empty = YeeGrid::new(4, 4, 4, COURANT_3D);
    empty.set_charge(1, 3, 2, rho);
    assert_eq!(empty.div_e(1, 3, 2), Fix128::ZERO);
    assert_eq!(empty.gauss_residual(1, 3, 2), -rho, "E = 0 reads -rho");
    assert_eq!(
        empty.max_abs_gauss_residual(),
        rho,
        "|−rho| over the lattice"
    );
    assert_eq!(
        empty.gauss_residual(2, 2, 2),
        Fix128::ZERO,
        "other nodes are untouched"
    );
}

/// `div_b` is the six-face cell stencil, by hand.
#[test]
fn div_b_is_the_six_face_stencil_by_hand() {
    let mut grid = YeeGrid::new(4, 4, 4, COURANT_3D);
    let (i, j, k) = (1usize, 2usize, 0usize);
    let [hxp, hxm, hyp, hym, hzp, hzm] = [q(5, 3), q(-2, 7), q(1, 9), q(4, 5), q(-8, 11), q(3, 13)];
    grid.set(Component::Hx, i + 1, j, k, hxp);
    grid.set(Component::Hx, i, j, k, hxm);
    grid.set(Component::Hy, i, j + 1, k, hyp);
    grid.set(Component::Hy, i, j, k, hym);
    grid.set(Component::Hz, i, j, k + 1, hzp);
    grid.set(Component::Hz, i, j, k, hzm);
    let by_hand = (hxp - hxm) + (hyp - hym) + (hzp - hzm);
    assert_eq!(grid.div_b(i, j, k), by_hand);
    // The neighbouring cell shares one face with the opposite sign and the
    // others are zero, so its divergence is -hxp from the shared x face.
    assert_eq!(
        grid.div_b(i + 1, j, k),
        -hxp,
        "shared face enters the neighbour negated"
    );
    assert_eq!(grid.max_abs_div_b(), by_hand.abs().max(hxp.abs()));
}

/// `current` reads back exactly what `set_current` stored, zero on every
/// other edge, and zero everywhere before any source exists.
#[test]
fn current_reads_back_exactly_what_was_set() {
    let mut grid = YeeGrid::new(4, 4, 4, COURANT_3D);
    assert_eq!(
        grid.current(Component::Ex, 1, 2, 2),
        Fix128::ZERO,
        "no source allocated yet"
    );
    let j = q(3, 7);
    grid.set_current(Component::Ex, 1, 2, 2, j);
    assert_eq!(
        grid.current(Component::Ex, 1, 2, 2),
        j,
        "set then get, same edge"
    );
    assert_eq!(
        grid.current(Component::Ey, 2, 1, 2),
        Fix128::ZERO,
        "an untouched edge of a different component must stay zero"
    );
    assert_eq!(
        grid.current(Component::Ex, 2, 2, 2),
        Fix128::ZERO,
        "an untouched Ex edge must stay zero"
    );
    let k = q(-5, 11);
    grid.set_current(Component::Ey, 2, 1, 2, k);
    assert_eq!(
        grid.current(Component::Ex, 1, 2, 2),
        j,
        "the first write must survive the second"
    );
    assert_eq!(grid.current(Component::Ey, 2, 1, 2), k);
}

/// `total_charge` is the exact sum of what was placed; a second write to the
/// same node replaces rather than accumulates.
#[test]
fn total_charge_is_the_exact_sum_of_placed_charges() {
    let mut grid = YeeGrid::new(4, 4, 4, COURANT_3D);
    assert_eq!(
        grid.total_charge(),
        Fix128::ZERO,
        "nothing placed, nothing allocated"
    );
    grid.set_charge(1, 1, 1, q(3, 8));
    grid.set_charge(2, 3, 1, q(-5, 16));
    grid.set_charge(3, 2, 3, q(7, 32));
    assert_eq!(
        grid.total_charge(),
        q(9, 32),
        "3/8 - 5/16 + 7/32 = 9/32, all dyadic"
    );
    grid.set_charge(1, 1, 1, q(1, 8));
    assert_eq!(
        grid.total_charge(),
        q(1, 32),
        "1/8 - 5/16 + 7/32 = 1/32 after the overwrite"
    );
    assert_eq!(grid.charge(1, 1, 1), q(1, 8));
}

// ---------------------------------------------------------------------------
// W3 — Gauss's law and ∇·B under the update, with exact arithmetic
// ---------------------------------------------------------------------------

/// A consistent charge placement keeps `∇·E − ρ` at the zero bit pattern
/// under the source-free update; an inconsistent one carries its residual
/// unchanged. Both because `∇·(∇×H)` telescopes and the arithmetic is exact.
///
/// ⚠️ 30 steps is an arithmetic limit, not a physical one: `S = 1/2` adds one
/// fractional bit per step to an integer seed, and the zero bit pattern is
/// only guaranteed while no product truncates.
#[test]
fn a_consistent_charge_keeps_gauss_exact_and_an_inconsistent_one_is_carried() {
    let s = q(1, 2);
    let e = int(3);
    let mut grid = YeeGrid::new(4, 4, 4, s);
    grid.set(Component::Ex, 1, 2, 2, e);
    grid.set_charge(1, 2, 2, e);
    grid.set_charge(2, 2, 2, -e);
    assert_eq!(
        grid.max_abs_gauss_residual(),
        Fix128::ZERO,
        "consistent at step 0"
    );
    assert_eq!(grid.total_charge(), Fix128::ZERO);

    let mut off = YeeGrid::new(4, 4, 4, s);
    off.set(Component::Ex, 1, 2, 2, e);
    off.set_charge(2, 2, 2, int(5));
    let carried = -e - int(5);
    assert_eq!(off.gauss_residual(2, 2, 2), carried);
    assert_eq!(
        off.gauss_residual(1, 2, 2),
        e,
        "the other end has no charge, so it reads +3"
    );

    for n in 1..=30 {
        grid.step();
        off.step();
        assert_eq!(
            grid.max_abs_gauss_residual(),
            Fix128::ZERO,
            "step {n}: consistent placement must stay exact"
        );
        assert_eq!(
            off.gauss_residual(2, 2, 2),
            carried,
            "step {n}: inconsistent residual is carried"
        );
        assert_eq!(
            off.gauss_residual(1, 2, 2),
            e,
            "step {n}: and so is the other end"
        );
    }
    assert!(
        max_abs_h(&grid) > Fix128::ZERO,
        "H must have been driven, or the zero is vacuous"
    );
    assert_eq!(
        grid.total_charge(),
        Fix128::ZERO,
        "no current, so rho does not move"
    );
    assert_eq!(
        grid.charge(1, 2, 2),
        e,
        "rho itself is untouched by the field update"
    );
}

/// `∇·B` is the zero bit pattern under the update while the arithmetic is
/// exact, and the same lattice at a non-dyadic `S` shows the truncation
/// residual — which is what makes the zero a statement and not a vacuity.
///
/// Measured 2026-10-03 on this seed: `S = 1/2` gives 0 ULP over 40 steps,
/// `S = 9/16` gives a worst of 26 ULP over the same 40 steps.
#[test]
fn div_b_is_bit_exactly_zero_while_the_arithmetic_is_exact() {
    let mut exact = YeeGrid::new(4, 4, 4, q(1, 2));
    seed_integer_mode(&mut exact);
    assert_eq!(exact.max_abs_div_b(), Fix128::ZERO, "H starts at zero");
    for n in 1..=40 {
        exact.step();
        assert_eq!(
            exact.max_abs_div_b(),
            Fix128::ZERO,
            "step {n}: with S = 1/2 and an integer seed div B must be the zero bit pattern"
        );
    }
    assert!(
        max_abs_h(&exact) > Fix128::ZERO,
        "H is non-zero, so the zero divergence is not vacuous"
    );

    let mut truncating = YeeGrid::new(4, 4, 4, COURANT_3D);
    seed_integer_mode(&mut truncating);
    let mut worst = 0i128;
    for _ in 0..40 {
        truncating.step();
        worst = worst.max(raw(truncating.max_abs_div_b()));
    }
    assert!(
        worst > 0,
        "S = 9/16 must show a truncation residual, or the exact run above proves nothing"
    );
    assert!(
        worst <= 2 * 40 + 16,
        "and it stays within the documented linear budget (got {worst} ULP)"
    );
}

// ---------------------------------------------------------------------------
// W4 — absorber: constructor, map, coefficients, reflection
// ---------------------------------------------------------------------------

/// `loss_coefficients` for a non-dyadic pair, an exact pair, and `S = 0`.
#[test]
fn loss_coefficients_match_the_closed_form_for_a_non_dyadic_pair() {
    // sigma = 2, S = 9/16: a = 9/16, ca = (7/16)/(25/16) = 7/25, cb = (9/16)/(25/16) = 9/25.
    let (ca, cb) = loss_coefficients(int(2), COURANT_3D);
    assert!(
        ulp_gap(ca, q(7, 25)) <= 1,
        "ca = 7/25 within one truncating division"
    );
    assert!(
        ulp_gap(cb, q(9, 25)) <= 1,
        "cb = 9/25 within one truncating division"
    );
    // sigma = 4, S = 1/2: a = 1, ca = 0, cb = 1/4 — every value dyadic.
    assert_eq!(loss_coefficients(int(4), q(1, 2)), (Fix128::ZERO, q(1, 4)));
    // S = 0: a = 0, (1/1, 0/1) = (1, 0) exactly, so a frozen lattice stays frozen.
    assert_eq!(
        loss_coefficients(int(3), Fix128::ZERO),
        (Fix128::ONE, Fix128::ZERO)
    );
}

/// `theoretical_pml_reflection` is the midpoint rule over the cubic profile.
///
/// `depth = 3`, `σ_max = 5/2`: `Σt³ = 17/24`, exponent `−2·(5/2)·(17/24) =
/// −85/24`. Bit-exact against `Fix128::exp` of that rational, and against
/// `det_math::exp64` within the `1e-6` relative accuracy `Fix128::exp`
/// documents (measured gap 2026-10-03: see the printed value).
#[test]
fn theoretical_reflection_is_the_midpoint_closed_form_through_det_math() {
    let r = theoretical_pml_reflection(3, q(5, 2));
    assert_eq!(r, q(-85, 24).exp(), "R = exp(-85/24)");
    let reference = exp64(-85.0 / 24.0);
    let rel = ((r.to_f64() - reference) / reference).abs();
    assert!(
        rel < 2e-6,
        "R = {} vs exp64 = {reference}, rel {rel}",
        r.to_f64()
    );
    println!(
        "theoretical R(3, 5/2) = {}, exp64 = {reference}, rel gap {rel:e}",
        r.to_f64()
    );

    // depth = 1: the single centre is t = 1/2, so R = exp(-2 * sigma / 8) = exp(-sigma/4).
    assert_eq!(theoretical_pml_reflection(1, int(4)), int(-1).exp());
    assert_eq!(theoretical_pml_reflection(1, int(2)), q(-1, 2).exp());
}

/// A layer exactly half the axis is accepted, and its map is the closed form:
/// coordinates `0, 1` and `3, 4` are lossy, `2` is not, so the `Ex` samples
/// the layer leaves alone are exactly those at `(j, k) = (2, 2)`: `nx` of them.
#[test]
fn a_layer_exactly_half_the_axis_is_accepted_and_has_the_closed_form_map() {
    let grid = YeeGrid::new_with_absorber(
        4,
        4,
        4,
        COURANT_3D,
        Absorber::GradedPml {
            depth: [2, 2, 2],
            sigma_max: int(4),
        },
    );
    let (ni, nj, nk) = grid.component_dims(Component::Ex);
    let mut lossless = Vec::new();
    for i in 0..ni {
        for j in 0..nj {
            for k in 0..nk {
                if !grid.is_absorbing(Component::Ex, i, j, k) {
                    lossless.push((i, j, k));
                }
            }
        }
    }
    assert_eq!(lossless, vec![(0, 2, 2), (1, 2, 2), (2, 2, 2), (3, 2, 2)]);
    // H sits at half-integer coordinates ½ … 3½, none of which is the centre,
    // so every H sample is absorbing.
    for c in [Component::Hx, Component::Hy, Component::Hz] {
        let (ni, nj, nk) = grid.component_dims(c);
        for i in 0..ni {
            for j in 0..nj {
                for k in 0..nk {
                    assert!(grid.is_absorbing(c, i, j, k), "{c:?}[{i}][{j}][{k}]");
                }
            }
        }
    }
}

/// `σ_max = 0` with a non-zero depth allocates the split storage but marks no
/// sample lossy, so the lattice must step bit-identically to a plain one and
/// `is_absorbing` must be false everywhere.
#[test]
fn a_zero_conductivity_layer_is_inert() {
    let mut plain = YeeGrid::new(6, 6, 6, COURANT_3D);
    let mut lined = YeeGrid::new_with_absorber(
        6,
        6,
        6,
        COURANT_3D,
        Absorber::GradedPml {
            depth: [2, 1, 3],
            sigma_max: Fix128::ZERO,
        },
    );
    for c in EVERY {
        let (ni, nj, nk) = lined.component_dims(c);
        for i in 0..ni {
            for j in 0..nj {
                for k in 0..nk {
                    assert!(!lined.is_absorbing(c, i, j, k), "{c:?}[{i}][{j}][{k}]");
                }
            }
        }
    }
    for g in [&mut plain, &mut lined] {
        g.set(Component::Ez, 2, 2, 2, q(3, 7));
        g.set(Component::Ex, 1, 3, 2, q(-2, 5));
    }
    for n in 0..60 {
        plain.step();
        lined.step();
        assert_bit_identical(&plain, &lined, &format!("step {n}"));
    }
    assert!(plain.max_abs_field() > Fix128::ZERO);
}

// ---------------------------------------------------------------------------
// W5 — degenerate input
// ---------------------------------------------------------------------------

#[test]
#[should_panic(expected = "lattice must have a cell in each axis")]
fn a_zero_dimension_is_rejected_by_new() {
    let _ = YeeGrid::new(0, 4, 4, COURANT_3D);
}

#[test]
#[should_panic(expected = "lattice must have a cell in each axis")]
fn a_zero_dimension_is_rejected_by_new_with_absorber() {
    let _ = YeeGrid::new_with_absorber(4, 0, 4, COURANT_3D, Absorber::None);
}

/// A lattice one cell thin on some axis has no interior node: the readers that
/// need one panic, and the lattice-wide sums have nothing to sum.
#[test]
fn a_thin_lattice_has_no_interior_nodes() {
    let mut thin = YeeGrid::new(4, 4, 1, COURANT_3D);
    assert_eq!(thin.interior_node_dims(), (3, 3, 0));
    assert_eq!(
        thin.max_abs_gauss_residual(),
        Fix128::ZERO,
        "no node, nothing to read"
    );
    assert_eq!(thin.total_charge(), Fix128::ZERO, "no node, nothing to sum");
    for (what, r) in [
        (
            "set_charge",
            catch_unwind(AssertUnwindSafe(|| thin.set_charge(1, 1, 1, Fix128::ONE))),
        ),
        (
            "charge",
            catch_unwind(AssertUnwindSafe(|| {
                let _ = thin.charge(1, 1, 1);
            })),
        ),
        (
            "div_e",
            catch_unwind(AssertUnwindSafe(|| {
                let _ = thin.div_e(1, 1, 1);
            })),
        ),
        (
            "gauss_residual",
            catch_unwind(AssertUnwindSafe(|| {
                let _ = thin.gauss_residual(1, 1, 1);
            })),
        ),
    ] {
        let err = r.expect_err(what);
        let msg = panic_message(&*err);
        assert!(msg.contains("not an interior node"), "{what}: {msg}");
    }
    // A 1×1×1 lattice: every E edge is tangential to a PEC wall, so a current
    // has nowhere to go; div B of its one cell is defined and zero; stepping
    // runs the H loops over a zero E and changes nothing.
    let mut cube = YeeGrid::new(1, 1, 1, COURANT_3D);
    let r = catch_unwind(AssertUnwindSafe(|| {
        cube.set_current(Component::Ex, 0, 0, 0, Fix128::ONE)
    }));
    let err = r.expect_err("set_current on a 1^3 lattice");
    let msg = panic_message(&*err);
    assert!(msg.contains("tangential to a PEC wall"), "{msg}");
    assert_eq!(cube.div_b(0, 0, 0), Fix128::ZERO);
    assert_eq!(cube.max_abs_div_b(), Fix128::ZERO);
    assert_eq!(cube.max_abs_field(), Fix128::ZERO);
    let before = cube.clone();
    cube.step();
    assert_eq!(cube, before, "nothing can move on a 1^3 PEC box");
}

#[test]
#[should_panic(expected = "not an interior node")]
fn a_charge_on_the_low_wall_is_rejected() {
    let mut grid = YeeGrid::new(4, 4, 4, COURANT_3D);
    grid.set_charge(0, 1, 1, Fix128::ONE);
}

#[test]
#[should_panic(expected = "not an interior node")]
fn a_charge_on_the_high_wall_is_rejected() {
    let grid = YeeGrid::new(4, 4, 4, COURANT_3D);
    let _ = grid.charge(1, 4, 1);
}

#[test]
#[should_panic(expected = "cell index out of range")]
fn div_b_outside_the_lattice_is_rejected() {
    let grid = YeeGrid::new(4, 4, 4, COURANT_3D);
    let _ = grid.div_b(4, 0, 0);
}

#[test]
#[should_panic(expected = "field index out of range")]
fn is_absorbing_outside_the_lattice_is_rejected() {
    let grid = YeeGrid::new(4, 4, 4, COURANT_3D);
    // Ex has nx = 4 samples along x, so index 4 is one past the end.
    let _ = grid.is_absorbing(Component::Ex, 4, 0, 0);
}

#[test]
#[should_panic(expected = "lives on electric edges")]
fn current_on_a_magnetic_face_is_rejected_by_the_reader() {
    let grid = YeeGrid::new(4, 4, 4, COURANT_3D);
    let _ = grid.current(Component::Hy, 1, 1, 1);
}

/// `S = 0` freezes the field: every product is zero, so a step is the identity.
#[test]
fn a_zero_courant_number_freezes_the_field() {
    let mut grid = YeeGrid::new(4, 4, 4, Fix128::ZERO);
    assert_eq!(grid.courant(), Fix128::ZERO);
    seed_integer_mode(&mut grid);
    grid.set(Component::Hx, 2, 1, 1, q(5, 7));
    grid.set_current(Component::Ey, 2, 1, 2, q(2, 3));
    let before = grid.clone();
    for _ in 0..10 {
        grid.step();
    }
    assert_eq!(
        grid, before,
        "S = 0: fields, sources and charge all stay put"
    );
    assert_eq!(
        grid.total_charge(),
        Fix128::ZERO,
        "rho -= S * div J with S = 0"
    );
}

/// `σ_max` big enough that `−2∫σ` is below `−44` returns exactly `0`
/// (`Fix128::exp` contract), and a negative `σ_max` is a gain by the same
/// formula: `depth = 4`, `σ_max = −4` gives `exp(+31/4)`.
#[test]
fn extreme_conductivities_follow_the_exp_contract() {
    // depth 4: sum t^3 = 31/32; sigma_max = 100 gives exponent -2*100*31/32 = -193.75 < -44.
    assert_eq!(theoretical_pml_reflection(4, int(100)), Fix128::ZERO);
    assert_eq!(theoretical_pml_reflection(4, int(-4)), q(31, 4).exp());
    assert!(
        theoretical_pml_reflection(4, int(-4)) > Fix128::ONE,
        "a negative layer reflects more than all"
    );
}

/// `σ·S/2 = −1` is the pole of `(1 − a)/(1 + a)`. `Fix128`'s `/` returns `0`
/// for a zero divisor (documented on `checked_div`), so the pair comes back
/// `(0, 0)` — a coefficient pair that annihilates a sample in one step.
///
/// ⚠️ Pinned as **measured** (2026-10-03). There is no documented contract
/// for a negative conductivity; this test exists so that a change here is
/// visible, and the finding is reported rather than resolved.
#[test]
fn the_loss_coefficient_pole_returns_the_zero_pair() {
    assert_eq!(
        loss_coefficients(int(-4), q(1, 2)),
        (Fix128::ZERO, Fix128::ZERO)
    );
    // One step short of the pole is still finite: sigma = -2, S = 1/2, a = -1/2,
    // ca = (3/2)/(1/2) = 3, cb = (1/2)/(1/2) = 1 — a growing update.
    assert_eq!(loss_coefficients(int(-2), q(1, 2)), (int(3), Fix128::ONE));
}

/// A sample at the top of the `Fix128` range steps without panicking and the
/// overflowing difference wraps modulo `2¹²⁸` to a bit pattern predicted by
/// hand.
///
/// `Ez[1] = MAX` (raw `i128::MAX`), `Ez[2] = −MAX` (raw `i128::MIN + 1`, the
/// negation of `MAX` does not hit the self-wrapping `i128::MIN`). `Hy` reads
/// `(∇×E)_y = −∂Ez/∂x` at three `i`:
///
/// * `i = 0`: `b = Ez[1] − Ez[0] = MAX − 0 = MAX`, `curl = −MAX`,
///   `Hy = 0 − 1·curl = −(−MAX) = MAX` (negating `raw(−MAX) = i128::MIN+1`
///   does not touch the wrap point, so this is exact, not wrapped).
/// * `i = 1`: `b = Ez[2] − Ez[1] = (−MAX) − MAX`; in raw 128-bit integers
///   `(i128::MIN+1) − i128::MAX = −2¹²⁸ + 2 ≡ +2` (mod `2¹²⁸`) — **this is
///   the wrap**, two near-extremal values subtracting to a tiny one.
///   `curl = 0 − 2 ULP = −2 ULP`, `Hy = 0 − 1·(−2 ULP) = +2 ULP`.
/// * `i = 2`: by the mirror of the `i = 0` case, `b = Ez[3] − Ez[2] = 0 −
///   (−MAX) = MAX`, so `Hy = MAX` again, without wrapping.
#[test]
fn a_full_range_sample_wraps_modulo_two_to_the_128() {
    let max = Fix128::from_raw(i64::MAX, u64::MAX);
    assert_eq!(raw(max), i128::MAX, "MAX must be the full positive range");
    assert_eq!(
        raw(-max),
        i128::MIN + 1,
        "negating MAX must not hit the self-wrapping MIN"
    );
    let mut grid = YeeGrid::new(4, 4, 4, Fix128::ONE);
    grid.set(Component::Ez, 1, 2, 2, max);
    grid.set(Component::Ez, 2, 2, 2, -max);
    let r = catch_unwind(AssertUnwindSafe(|| grid.step()));
    assert!(r.is_ok(), "Fix128 arithmetic wraps; a step must not panic");
    assert_eq!(
        grid.get(Component::Hy, 0, 2, 2),
        max,
        "exact: 0 - 1*(MAX - 0) negated back to MAX"
    );
    assert_eq!(
        grid.get(Component::Hy, 1, 2, 2),
        Fix128::from_raw(0, 2),
        "wrapped: (-MAX) - MAX overflows to +2 ULP"
    );
    assert_eq!(
        grid.get(Component::Hy, 2, 2, 2),
        max,
        "exact: mirror of i = 0"
    );
}

/// `max_abs_field` over a sample at `Fix128`'s minimum reports `0`: `|MIN|`
/// is not representable, `abs()` wraps back to `MIN`, and a negative value
/// never wins the maximum.
///
/// ⚠️ Pinned as **measured** (2026-10-03) and reported as a finding: the
/// probe's contract "largest |value|" cannot be met at that one input, and
/// it is silent about it.
#[test]
fn max_abs_field_drops_a_sample_at_the_fixed_point_minimum() {
    let min = Fix128::from_raw(i64::MIN, 0);
    assert_eq!(min.abs(), min, "|MIN| wraps to MIN");
    let mut grid = YeeGrid::new(4, 4, 4, COURANT_3D);
    grid.set(Component::Ez, 1, 1, 1, min);
    assert_eq!(grid.max_abs_field(), Fix128::ZERO);
    // One ULP above the minimum is representable and is reported.
    let almost = Fix128::from_raw(i64::MIN, 1);
    grid.set(Component::Ez, 1, 1, 1, almost);
    assert_eq!(grid.max_abs_field(), -almost);
}
