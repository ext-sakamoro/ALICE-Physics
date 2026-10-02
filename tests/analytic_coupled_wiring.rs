//! Oracles for the wiring of `coupled_field::reconcile_weighted` (built on
//! `add_assign` / `scale_div` / `same_grid_as`), of the checked L2 residual
//! norm inside the FEM conjugate gradient, and the closed forms of the
//! equilibration scale that is still waiting for its consumer.
//!
//! # Closed forms
//!
//! * **Weighted reconcile** is `Σ wᵢ fᵢ / Σ wᵢ` with integer weights: the
//!   products and the sum are exact in `Fix128`, and the one division
//!   truncates toward zero, so each cell equals the integer division of the
//!   raw sum by `Σ wᵢ` **to the bit**. Equal weights give `reconcile_mean`'s
//!   result bit for bit (same sum, same division), and the order of the
//!   participants cannot matter (the sum is associative).
//! * **Checked L2 in CG**: `|r|·|r|` and `r·r` are the same `Fix128` product
//!   (the exact product does not depend on the sign and the truncation acts
//!   on the exact value), so a faithful norm is bit-identical to the former
//!   `dot(r, r).sqrt()`; a residual whose square leaves the faithful range is
//!   refused with `FemError::ResidualNormUnfaithful` instead of being
//!   compared as a wrapped number.
//! * **Equilibration scale**: `covering(m)` is the smallest power of two
//!   `≥ m` (an exact power maps to itself), `round_trip_bound(e)` is
//!   `(2^e − 1)·2⁻⁶⁴`, a `scale_down` then `scale_up` round trip errs by at
//!   most that bound and the other order is exact.
//!
//! # Degenerate input
//!
//! `reconcile_weighted` refuses an empty slice, a weight slice of the wrong
//! length, all-zero weights and a grid mismatch, every time with no
//! participant touched; a zero weight abstains but still adopts the result.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]

use alice_physics::coupled_field::{
    reconcile_mean, reconcile_weighted, CoupledField, CoupledFieldError, CoupledScalar,
};
use alice_physics::coupled_iteration::{ConfigFault, EquilibrationScale};
use alice_physics::linear_elastic_fem::{
    solve, Axis, BoundaryConditions, ElasticMaterial, FemError, SolverConfig,
};
use alice_physics::math::Fix128;
use alice_physics::sdf_fem_mesh::{SdfTetMesh, Tetrahedron};

fn q(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn int(n: i64) -> Fix128 {
    Fix128::from_int(n)
}

fn raw(x: Fix128) -> i128 {
    (i128::from(x.hi) << 64) | i128::from(x.lo)
}

fn from_raw_i128(r: i128) -> Fix128 {
    Fix128::from_raw((r >> 64) as i64, r as u64)
}

const N: usize = 3;

fn grid() -> CoupledField {
    CoupledField::try_new(
        N,
        N,
        N,
        (Fix128::ZERO, Fix128::ZERO, Fix128::ZERO),
        (int(2), int(2), int(2)),
    )
    .expect("non-degenerate grid")
}

/// A participant that owns one field on the shared grid.
struct Owner {
    name: &'static str,
    field: CoupledField,
}

impl Owner {
    fn with(name: &'static str, f: impl Fn(usize, usize, usize) -> Fix128) -> Self {
        let mut field = grid();
        for k in 0..N {
            for j in 0..N {
                for i in 0..N {
                    field.set(i, j, k, f(i, j, k));
                }
            }
        }
        Self { name, field }
    }
}

impl CoupledScalar for Owner {
    fn coupled_name(&self) -> &'static str {
        self.name
    }
    fn coupled_channel(&self) -> Result<CoupledField, CoupledFieldError> {
        CoupledField::try_new(
            N,
            N,
            N,
            (Fix128::ZERO, Fix128::ZERO, Fix128::ZERO),
            (int(2), int(2), int(2)),
        )
    }
    fn publish(&self, out: &mut CoupledField) -> Result<(), CoupledFieldError> {
        if !out.same_grid_as(&self.field) {
            return Err(CoupledFieldError::BoundsMismatch);
        }
        for (dst, src) in out.as_mut_slice().iter_mut().zip(self.field.as_slice()) {
            *dst = *src;
        }
        Ok(())
    }
    fn adopt(&mut self, src: &CoupledField) -> Result<(), CoupledFieldError> {
        if !self.field.same_grid_as(src) {
            return Err(CoupledFieldError::BoundsMismatch);
        }
        for (dst, s) in self.field.as_mut_slice().iter_mut().zip(src.as_slice()) {
            *dst = *s;
        }
        Ok(())
    }
}

fn a_value(i: usize, j: usize, k: usize) -> Fix128 {
    q(7 * i as i64 - 3 * j as i64 + 11 * k as i64 + 5, 8)
}

fn b_value(i: usize, j: usize, k: usize) -> Fix128 {
    q(-(i as i64) + 9 * j as i64 - 2 * k as i64 + 1, 16)
}

fn c_value(i: usize, j: usize, k: usize) -> Fix128 {
    q(3 * i as i64 + 3 * j as i64 + 3 * k as i64 - 20, 4)
}

// ===========================================================================
// Oracle 1 — the weighted mean to the bit, order-independent
// ===========================================================================

#[test]
fn reconcile_weighted_is_the_integer_division_of_the_exact_weighted_sum() {
    let weights = [3u32, 1, 2];
    let mut a = Owner::with("a", a_value);
    let mut b = Owner::with("b", b_value);
    let mut c = Owner::with("c", c_value);
    let mut channel = grid();
    {
        let mut participants: [&mut dyn CoupledScalar; 3] = [&mut a, &mut b, &mut c];
        reconcile_weighted(&mut participants, &weights, &mut channel).expect("same grid");
    }
    for k in 0..N {
        for j in 0..N {
            for i in 0..N {
                let sum =
                    3 * raw(a_value(i, j, k)) + raw(b_value(i, j, k)) + 2 * raw(c_value(i, j, k));
                let want = from_raw_i128(sum / 6);
                assert_eq!(channel.get(i, j, k), want, "cell ({i}, {j}, {k})");
                assert_eq!(a.field.get(i, j, k), want, "a adopted");
                assert_eq!(b.field.get(i, j, k), want, "b adopted");
                assert_eq!(c.field.get(i, j, k), want, "c adopted");
            }
        }
    }

    // The same participants in another order, with the weights permuted
    // alongside: bit-identical.
    let mut a2 = Owner::with("a", a_value);
    let mut b2 = Owner::with("b", b_value);
    let mut c2 = Owner::with("c", c_value);
    let mut other = grid();
    {
        let mut participants: [&mut dyn CoupledScalar; 3] = [&mut c2, &mut a2, &mut b2];
        reconcile_weighted(&mut participants, &[2, 3, 1], &mut other).expect("same grid");
    }
    assert_eq!(
        other.as_slice(),
        channel.as_slice(),
        "order leaked into the weighted mean"
    );
}

#[test]
fn equal_weights_reproduce_reconcile_mean_bit_for_bit_and_one_participant_is_identity() {
    let mut a = Owner::with("a", a_value);
    let mut b = Owner::with("b", b_value);
    let mut weighted = grid();
    {
        let mut participants: [&mut dyn CoupledScalar; 2] = [&mut a, &mut b];
        reconcile_weighted(&mut participants, &[5, 5], &mut weighted).expect("same grid");
    }
    let mut a2 = Owner::with("a", a_value);
    let mut b2 = Owner::with("b", b_value);
    let mut mean = grid();
    {
        let mut participants: [&mut dyn CoupledScalar; 2] = [&mut a2, &mut b2];
        reconcile_mean(&mut participants, &mut mean).expect("same grid");
    }
    // Not the bit the arithmetic mean gives for weight 5: `5a + 5b` over 10
    // and `a + b` over 2 are the same rational, but the truncating division
    // sees different numerators. Pin what the closed form says instead.
    for k in 0..N {
        for j in 0..N {
            for i in 0..N {
                let sum = 5 * (raw(a_value(i, j, k)) + raw(b_value(i, j, k)));
                assert_eq!(weighted.get(i, j, k), from_raw_i128(sum / 10));
                let plain = raw(a_value(i, j, k)) + raw(b_value(i, j, k));
                assert_eq!(mean.get(i, j, k), from_raw_i128(plain / 2));
            }
        }
    }
    // Unit weights are the arithmetic mean to the bit.
    let mut a3 = Owner::with("a", a_value);
    let mut b3 = Owner::with("b", b_value);
    let mut unit = grid();
    {
        let mut participants: [&mut dyn CoupledScalar; 2] = [&mut a3, &mut b3];
        reconcile_weighted(&mut participants, &[1, 1], &mut unit).expect("same grid");
    }
    assert_eq!(unit.as_slice(), mean.as_slice());

    // One participant: its own field back, bit for bit, whatever the weight.
    let mut solo = Owner::with("solo", c_value);
    let before = solo.field.clone();
    let mut channel = grid();
    {
        let mut participants: [&mut dyn CoupledScalar; 1] = [&mut solo];
        reconcile_weighted(&mut participants, &[7], &mut channel).expect("same grid");
    }
    assert_eq!(solo.field.as_slice(), before.as_slice());
    assert_eq!(channel.as_slice(), before.as_slice());
}

#[test]
fn a_zero_weight_abstains_but_adopts() {
    let mut a = Owner::with("a", a_value);
    let mut b = Owner::with("b", b_value);
    let mut channel = grid();
    {
        let mut participants: [&mut dyn CoupledScalar; 2] = [&mut a, &mut b];
        reconcile_weighted(&mut participants, &[0, 4], &mut channel).expect("same grid");
    }
    for k in 0..N {
        for j in 0..N {
            for i in 0..N {
                // 4b / 4 = b exactly.
                assert_eq!(channel.get(i, j, k), b_value(i, j, k));
                assert_eq!(
                    a.field.get(i, j, k),
                    b_value(i, j, k),
                    "the abstaining owner adopts"
                );
            }
        }
    }
}

// ===========================================================================
// Refusals
// ===========================================================================

#[test]
fn reconcile_weighted_refuses_bad_input_with_every_participant_untouched() {
    let mut a = Owner::with("a", a_value);
    let mut b = Owner::with("b", b_value);
    let (a0, b0) = (a.field.clone(), b.field.clone());
    let mut channel = grid();

    let mut none: [&mut dyn CoupledScalar; 0] = [];
    assert_eq!(
        reconcile_weighted(&mut none, &[], &mut channel).unwrap_err(),
        CoupledFieldError::NoParticipants
    );
    {
        let mut participants: [&mut dyn CoupledScalar; 2] = [&mut a, &mut b];
        assert_eq!(
            reconcile_weighted(&mut participants, &[1], &mut channel).unwrap_err(),
            CoupledFieldError::NoParticipants,
            "a weight per participant, or nothing"
        );
        assert_eq!(
            reconcile_weighted(&mut participants, &[0, 0], &mut channel).unwrap_err(),
            CoupledFieldError::NoParticipants,
            "everybody abstaining is nobody taking part"
        );
    }
    let mut wrong = CoupledField::try_new(
        N + 1,
        N,
        N,
        (Fix128::ZERO, Fix128::ZERO, Fix128::ZERO),
        (int(2), int(2), int(2)),
    )
    .unwrap();
    {
        let mut participants: [&mut dyn CoupledScalar; 2] = [&mut a, &mut b];
        let err = reconcile_weighted(&mut participants, &[1, 1], &mut wrong).unwrap_err();
        assert!(
            matches!(err, CoupledFieldError::ResolutionMismatch { .. }),
            "a channel on another grid is refused: {err:?}"
        );
    }
    assert_eq!(
        a.field.as_slice(),
        a0.as_slice(),
        "a refusal must not adopt"
    );
    assert_eq!(
        b.field.as_slice(),
        b0.as_slice(),
        "a refusal must not adopt"
    );
}

// ===========================================================================
// Oracle 2 — the checked L2 norm inside the conjugate gradient
// ===========================================================================

#[test]
fn a_product_of_a_value_with_itself_does_not_depend_on_its_sign() {
    // The claim behind "bit-identical to dot(r, r)": |r|·|r| == r·r.
    let probes = [
        q(-3, 7),
        q(-1, 3),
        Fix128::from_raw(-5, 0x8000_0000_0000_0001),
        Fix128::from_raw(-1, 1),
        Fix128::from_raw(0, 1 << 32),
        q(-12345, 1024),
    ];
    for r in probes {
        let a = r.abs();
        assert_eq!(a * a, r * r, "{r:?}");
    }
}

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn node_index(nx: usize, ny: usize, i: usize, j: usize, k: usize) -> u32 {
    (i + (nx + 1) * (j + (ny + 1) * k)) as u32
}

/// A Kuhn-split box of `nx × ny × nz` cells of edge `h` (mm).
fn kuhn_box(nx: usize, ny: usize, nz: usize, h: f32) -> SdfTetMesh {
    let mut mesh = SdfTetMesh::default();
    for k in 0..=nz {
        for j in 0..=ny {
            for i in 0..=nx {
                mesh.vertices
                    .push([i as f32 * h, j as f32 * h, k as f32 * h]);
            }
        }
    }
    let n = |i, j, k| node_index(nx, ny, i, j, k);
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let c = [
                    n(i, j, k),
                    n(i + 1, j, k),
                    n(i, j + 1, k),
                    n(i + 1, j + 1, k),
                    n(i, j, k + 1),
                    n(i + 1, j, k + 1),
                    n(i, j + 1, k + 1),
                    n(i + 1, j + 1, k + 1),
                ];
                for t in [
                    [0, 1, 3, 7],
                    [0, 1, 5, 7],
                    [0, 2, 3, 7],
                    [0, 2, 6, 7],
                    [0, 4, 5, 7],
                    [0, 4, 6, 7],
                ] {
                    mesh.tets.push(Tetrahedron {
                        vertices: [c[t[0]], c[t[1]], c[t[2]], c[t[3]]],
                    });
                }
            }
        }
    }
    mesh
}

fn stretched_bar(
    youngs_mpa: f64,
    stretch_mm: f64,
) -> (SdfTetMesh, ElasticMaterial, BoundaryConditions) {
    let mesh = kuhn_box(2, 1, 1, 1.0);
    let material = ElasticMaterial::new(fx(youngs_mpa), fx(0.3)).expect("valid material");
    let mut bc = BoundaryConditions::new();
    for k in 0..=1 {
        for j in 0..=1 {
            bc.fix(node_index(2, 1, 0, j, k));
            bc.prescribe(node_index(2, 1, 2, j, k), Axis::X, fx(stretch_mm));
        }
    }
    (mesh, material, bc)
}

#[test]
fn an_ordinary_bar_solves_and_a_residual_past_the_faithful_range_is_refused_not_wrapped() {
    let config = SolverConfig::default();
    let (mesh, material, bc) = stretched_bar(3500.0, 0.01);
    let solution = solve(&mesh, &material, &bc, &config).expect("PLA bar solves");
    assert!(solution.iterations > 0);

    // E = 1e13 MPa with a prescribed stretch of 10 mm: the initial residual
    // components are of order 1e14 N, whose squares (1e28) are far past the
    // 2^63 the squared norm can hold. The old `dot(r, r).sqrt()` wrapped here
    // and compared a garbage number against the tolerance.
    let (mesh, material, bc) = stretched_bar(1.0e13, 10.0);
    let err = solve(&mesh, &material, &bc, &config).unwrap_err();
    assert!(
        matches!(err, FemError::ResidualNormUnfaithful { .. }),
        "expected the unfaithful-norm refusal, got {err:?}"
    );
}

// ===========================================================================
// Oracle 3 — the equilibration scale's closed forms (consumer pending)
// ===========================================================================

#[test]
fn covering_is_the_next_power_of_two_and_the_round_trip_bound_is_exact() {
    for (magnitude, exponent) in [
        (q(1, 2), 0u32),
        (Fix128::ONE, 0),
        (q(3, 2), 1),
        (int(2), 1),
        (q(9, 4), 2),
        (int(1000), 10),
        (int(1 << 40), 40),
    ] {
        let s = EquilibrationScale::covering(magnitude).unwrap();
        assert_eq!(s.exponent(), exponent, "covering({magnitude:?})");
        assert_eq!(s.factor(), int(1 << exponent));
        // (2^e − 1)·2⁻⁶⁴ exactly.
        let want = if exponent == 0 {
            Fix128::ZERO
        } else {
            Fix128::from_raw(0, (1u64 << exponent) - 1)
        };
        assert_eq!(s.round_trip_bound(), want);
        // Down then up loses at most the bound; up then down is exact.
        let value = q(-123_457, 977);
        let back = s.scale_up(s.scale_down(value));
        let loss = value - back;
        assert!(
            loss >= Fix128::ZERO && loss <= want,
            "loss {loss:?} > bound {want:?}"
        );
        assert_eq!(s.scale_down(s.scale_up(value)), value);
    }
    // Zero and negative magnitudes are the identity; past 2^62 is refused.
    assert_eq!(
        EquilibrationScale::covering(Fix128::ZERO).unwrap(),
        EquilibrationScale::IDENTITY
    );
    assert_eq!(
        EquilibrationScale::covering(int(-5)).unwrap(),
        EquilibrationScale::IDENTITY
    );
    assert_eq!(
        EquilibrationScale::covering(Fix128::from_raw(i64::MAX, 0)).unwrap_err(),
        ConfigFault::ScaleOutOfRange
    );
    assert_eq!(EquilibrationScale::MAX_EXPONENT, 62);
}
