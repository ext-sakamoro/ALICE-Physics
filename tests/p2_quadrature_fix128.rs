//! Can `Fix128` carry the quadrature a P2 tetrahedron needs?
//!
//! A P2 tetrahedron's shape functions are quadratic, so their gradients are
//! linear, so the integrand of the element stiffness `BᵀDB` is **quadratic** in
//! the barycentric coordinates on a straight-edged element. Integrating it
//! exactly needs a degree-2 rule. The rule that every textbook reaches for is
//! Hammer–Stroud's four points, and its abscissae are irrational —
//! `(5 ± √5)/20` — which is where the question "does the fixed-point type
//! support this?" comes from.
//!
//! ⚠️ **The premise of that question is wrong, and this file measures why.** The
//! irrational rule is one option among several, and it is the worst of them
//! here. Two others reach the same degree with **rational** data:
//!
//! - **Keast's five points** (degree 3): weights `−4/5` and `9/20`, abscissae
//!   `1/2` and `1/6`. No irrational anywhere, and it is exact one degree higher
//!   than a P2 stiffness needs.
//! - **The closed form** — the barycentric monomial moments themselves,
//!   `∫λ₁^a λ₂^b λ₃^c λ₄^d / V = a!b!c!d!·3!/(a+b+c+d+3)!`. On an affine element
//!   the Jacobian is constant, so the whole element matrix is a fixed rational
//!   combination of these and no sampling is involved at all.
//!
//! What this file pins is not an opinion about which to pick but the numbers
//! that decide it: each candidate is applied to every barycentric monomial a P2
//! stiffness can produce, and the error against the exact rational value is
//! reported **in ulps of the `Fix128` representation**, not in decimal. Decimal
//! rounds, and the whole question is about the last bits.
//!
//! # ⚠️ The measured answer contradicts the reason the file was written
//!
//! The expectation going in was that avoiding `sqrt` would be the safer route,
//! and that the rational rule would therefore be at least as accurate. **It is
//! not.** On the degrees a P2 stiffness actually needs (≤ 2) the measurement is:
//!
//! | rule | worst error within its exactness degree |
//! |---|---|
//! | Hammer–Stroud 4pt (irrational, uses `sqrt`) | **−2 ulp** |
//! | Keast 5pt (rational, four operations only) | −5 ulp |
//!
//! The irrational rule wins because its *weights* are `1/4`, which is exactly
//! representable in a binary fixed-point type, while Keast's `1/6`, `4/5` and
//! `9/20` are not — and Keast's centroid weight is negative, so its five terms
//! cancel. `√5` costs nothing by comparison: `Fix128::sqrt` returns **exactly**
//! `floor(√x · 2⁶⁴)`, measured against `isqrt(5 << 128)` with zero ulp of
//! error.
//!
//! So the conclusion for a P2 element on this crate is that **quadrature is not
//! a constraint**. Both routes are open, the irrational one is the more accurate
//! of the two at the degree that matters, and neither touches CORDIC.
//!
//! # Why ulps and not `to_f64`
//!
//! `f64` carries 53 bits of mantissa and `Fix128` carries 64 bits of fraction,
//! so a difference of a few ulps is *invisible* through `to_f64`. Reporting
//! these errors as decimals would print `0.000e0` for every row and prove
//! nothing (the same
//! mistake was made in this crate ten commits ago and produced a retracted
//! "bit-identical" claim).

#![cfg(feature = "std")]

use alice_physics::math::Fix128;

/// Raw two's-complement 128-bit integer behind a `Fix128`, as a signed integer.
///
/// `value = hi + lo·2⁻⁶⁴`, so this encoding is monotone in the value and
/// differences in it are exactly differences in ulps.
fn raw(x: Fix128) -> i128 {
    ((x.hi as i128) << 64) | (x.lo as i128)
}

fn ulps(a: Fix128, b: Fix128) -> i128 {
    raw(a) - raw(b)
}

fn int(n: i64) -> Fix128 {
    Fix128::from_int(n)
}

/// `p / q`, rounded by whatever `Fix128::Div` does.
fn ratio(p: i64, q: i64) -> Fix128 {
    int(p) / int(q)
}

// ---------------------------------------------------------------------------
// exact moments
// ---------------------------------------------------------------------------

/// A barycentric monomial `λ₁^a λ₂^b λ₃^c λ₄^d`.
#[derive(Clone, Copy)]
struct Monomial {
    powers: [u32; 4],
    /// `∫ monomial dV / V`, rounded to nearest in the `Fix128` encoding.
    ///
    /// Computed off-line from `a!b!c!d!·3!/(a+b+c+d+3)!` in exact rational
    /// arithmetic; the pair is the raw `(hi, lo)`. Deriving it here with
    /// `Fix128` division would make the reference share the rounding of the
    /// thing being measured.
    exact_raw: (i64, u64),
    label: &'static str,
}

/// Every monomial a P2 element stiffness can produce (degree ≤ 2), plus the
/// degree-3 ones, which are what distinguishes Keast's rule from Hammer–Stroud's
/// and are needed if the element is ever used with a non-constant Jacobian.
const MONOMIALS: [Monomial; 7] = [
    Monomial {
        powers: [0, 0, 0, 0],
        exact_raw: (1, 0),
        label: "1",
    },
    Monomial {
        powers: [1, 0, 0, 0],
        exact_raw: (0, 4_611_686_018_427_387_904),
        label: "l1",
    },
    Monomial {
        powers: [2, 0, 0, 0],
        exact_raw: (0, 1_844_674_407_370_955_162),
        label: "l1^2",
    },
    Monomial {
        powers: [1, 1, 0, 0],
        exact_raw: (0, 922_337_203_685_477_581),
        label: "l1 l2",
    },
    Monomial {
        powers: [3, 0, 0, 0],
        exact_raw: (0, 922_337_203_685_477_581),
        label: "l1^3",
    },
    Monomial {
        powers: [2, 1, 0, 0],
        exact_raw: (0, 307_445_734_561_825_860),
        label: "l1^2 l2",
    },
    Monomial {
        powers: [1, 1, 1, 0],
        exact_raw: (0, 153_722_867_280_912_930),
        label: "l1 l2 l3",
    },
];

impl Monomial {
    fn exact(&self) -> Fix128 {
        Fix128::from_raw(self.exact_raw.0, self.exact_raw.1)
    }
    fn degree(&self) -> u32 {
        self.powers.iter().sum()
    }
}

// ---------------------------------------------------------------------------
// rules
// ---------------------------------------------------------------------------

struct Rule {
    name: &'static str,
    /// Highest polynomial degree the rule integrates exactly in real arithmetic.
    exact_to_degree: u32,
    points: Vec<[Fix128; 4]>,
    weights: Vec<Fix128>,
}

impl Rule {
    /// Sum in index order. `Fix128` addition is not associative at the last bit,
    /// so a rule that summed in a different order would be a different rule.
    fn integrate(&self, m: &Monomial) -> Fix128 {
        let mut acc = Fix128::ZERO;
        for (p, w) in self.points.iter().zip(&self.weights) {
            let mut term = *w;
            for (k, coord) in p.iter().enumerate() {
                for _ in 0..m.powers[k] {
                    term = term * *coord;
                }
            }
            acc = acc + term;
        }
        acc
    }
}

/// Hammer–Stroud, four points, exact to degree 2. `b = (5+3√5)/20` on one
/// coordinate and `a = (5−√5)/20` on the other three; weight `1/4` each.
fn hammer_stroud() -> Rule {
    let root5 = int(5).sqrt();
    let a = (int(5) - root5) / int(20);
    let b = (int(5) + int(3) * root5) / int(20);
    let w = ratio(1, 4);
    let mut points = Vec::new();
    for i in 0..4 {
        let mut p = [a; 4];
        p[i] = b;
        points.push(p);
    }
    Rule {
        name: "Hammer-Stroud 4pt (irrational, needs sqrt)",
        exact_to_degree: 2,
        points,
        weights: vec![w; 4],
    }
}

/// Keast, five points, exact to degree 3. Centroid with weight `−4/5`, and the
/// four permutations of `(1/2, 1/6, 1/6, 1/6)` with weight `9/20`.
fn keast() -> Rule {
    let quarter = ratio(1, 4);
    let half = ratio(1, 2);
    let sixth = ratio(1, 6);
    let mut points = vec![[quarter; 4]];
    let mut weights = vec![-ratio(4, 5)];
    for i in 0..4 {
        let mut p = [sixth; 4];
        p[i] = half;
        points.push(p);
        weights.push(ratio(9, 20));
    }
    Rule {
        name: "Keast 5pt (rational, four operations only)",
        exact_to_degree: 3,
        points,
        weights,
    }
}

// ---------------------------------------------------------------------------
// the measurements
// ---------------------------------------------------------------------------

/// Both rules must be partitions of the reference simplex before their errors
/// mean anything: weights summing to one, barycentric coordinates summing to
/// one at every point. A mistyped constant fails here rather than showing up as
/// a plausible-looking error further down.
#[test]
fn both_rules_are_partitions_of_the_simplex() {
    for rule in [hammer_stroud(), keast()] {
        let mut wsum = Fix128::ZERO;
        for w in &rule.weights {
            wsum = wsum + *w;
        }
        let dw = ulps(wsum, Fix128::ONE);
        eprintln!("[p2quad] {}: weight sum error {dw} ulp", rule.name);
        assert!(
            dw.abs() <= 8,
            "{}: weights must sum to one; off by {dw} ulp",
            rule.name
        );
        for (i, p) in rule.points.iter().enumerate() {
            let mut s = Fix128::ZERO;
            for c in p {
                s = s + *c;
            }
            let d = ulps(s, Fix128::ONE);
            assert!(
                d.abs() <= 8,
                "{}: point {i} barycentric coordinates must sum to one; off by {d} ulp",
                rule.name
            );
        }
    }
}

/// The headline table: error per monomial, per rule, in ulps.
#[test]
fn quadrature_error_per_monomial_in_ulps() {
    let rules = [hammer_stroud(), keast()];
    eprintln!("[p2quad] ---- error vs the exact rational moment, in ulps of Fix128 ----");
    eprintln!(
        "[p2quad] {:>10}  {:>4}  {:>14}  {:>14}",
        "monomial", "deg", "Hammer-Stroud", "Keast"
    );
    let mut worst = [0i128; 2];
    for m in &MONOMIALS {
        let e: Vec<i128> = rules
            .iter()
            .map(|r| ulps(r.integrate(m), m.exact()))
            .collect();
        eprintln!(
            "[p2quad] {:>10}  {:>4}  {:>14}  {:>14}",
            m.label,
            m.degree(),
            e[0],
            e[1]
        );
        for (k, r) in rules.iter().enumerate() {
            if m.degree() <= r.exact_to_degree && e[k].abs() > worst[k].abs() {
                worst[k] = e[k];
            }
        }
    }
    eprintln!(
        "[p2quad] worst error within each rule's exactness degree: Hammer-Stroud {} ulp, \
         Keast {} ulp",
        worst[0], worst[1]
    );

    // Within the degree each rule is exact to, the only error left is the
    // rounding of its own constants. Both must land within a handful of ulps —
    // a rule that is off by more than that has a wrong constant, not a rounding
    // problem.
    for (k, r) in rules.iter().enumerate() {
        assert!(
            worst[k].abs() <= 64,
            "{}: exact to degree {} in real arithmetic, so the residual error must be \
             rounding only; got {} ulp",
            r.name,
            r.exact_to_degree,
            worst[k]
        );
    }

    // ⚠️ Measured, against the expectation that wrote this file: the irrational
    // rule is the *more* accurate of the two on degree ≤ 2. Its weight is 1/4,
    // which a binary fixed-point type holds exactly; Keast pays for 1/6, 4/5
    // and 9/20, and for the cancellation its negative centroid weight creates.
    // Pinned in this direction so that a future change which makes the rational
    // rule win has to come and say so here.
    assert!(
        worst[0].abs() <= worst[1].abs(),
        "avoiding sqrt was expected to cost nothing, and the measurement is that it \
         costs accuracy: Hammer-Stroud {} ulp against Keast {} ulp. If this has \
         reversed, the note in the module header is now wrong",
        worst[0],
        worst[1]
    );

    // Hammer-Stroud is exact only to degree 2, and the degree-3 rows show it
    // plainly rather than leaving the reader to infer it: a rule used one degree
    // past its exactness is not slightly worse, it is wrong by ~10^16 ulp.
    let cubic = MONOMIALS
        .iter()
        .find(|m| m.label == "l1^3")
        .expect("present");
    let hs_cubic = ulps(rules[0].integrate(cubic), cubic.exact());
    assert!(
        hs_cubic.abs() > 1_000_000,
        "the degree-3 row exists to show the cliff past a rule's exactness degree; \
         Hammer-Stroud was off by only {hs_cubic} ulp on l1^3, so either the rule or \
         the reference moved"
    );
}

/// `sqrt` is usable, and this pins exactly what it produces.
///
/// `Fix128::sqrt` is an exact integer square root over a fixed 96 steps — not
/// CORDIC, not Newton with a tolerance — so it is bit-identical on every target.
///
/// ⚠️ **The defining property has to be stated in the raw representation, not
/// through `Fix128` multiplication.** The first version of this test asserted
/// `(r + 1 ulp)² > 5` and failed, and the failure was the test's, not `sqrt`'s:
/// `Fix128::Mul` truncates its 256-bit product back to 64 fractional bits, so
/// the strict inequality that holds over the integers is rounded away. In exact
/// integer arithmetic `r² ≤ 5·2¹²⁸ < (r+1)²` does hold — verified off-line
/// against `isqrt(5 << 128)`, which the golden below equals with **zero** ulp of
/// error. This is the same shape as an earlier retracted "bit-identical"
/// claim: a bit-level statement
/// measured through an instrument coarser than the bits.
#[test]
fn sqrt_five_and_the_irrational_abscissae_are_pinned() {
    let root5 = int(5).sqrt();
    let a = (int(5) - root5) / int(20);
    let b = (int(5) + int(3) * root5) / int(20);
    eprintln!(
        "[p2quad] sqrt(5) raw = ({}, {}), a=(5-sqrt5)/20 raw = ({}, {}), \
         b=(5+3sqrt5)/20 raw = ({}, {})",
        root5.hi, root5.lo, a.hi, a.lo, b.hi, b.lo
    );

    // Golden: this is `isqrt(5 << 128)` exactly, i.e. `floor(√5 · 2⁶⁴)`.
    // A change here is a change in `Fix128::sqrt`, and it invalidates any
    // determinism claim made about a P2 element built on this rule.
    assert!(
        (root5.hi, root5.lo) == (2, 4_354_685_564_936_845_355),
        "sqrt(5) moved: got ({}, {}), want (2, 4354685564936845355) = floor(sqrt(5)*2^64)",
        root5.hi,
        root5.lo
    );

    // What the truncating multiply can still say: r² lands within a few ulps of
    // 5 from below. Reported as a number because "close enough" is the whole
    // question.
    let sq = ulps(root5 * root5, int(5));
    eprintln!("[p2quad] sqrt(5)^2 - 5 = {sq} ulp (negative: Mul truncates)");
    assert!(
        (-16..=0).contains(&sq),
        "sqrt(5)^2 must sit just below 5; got {sq} ulp"
    );

    // The abscissae are a partition: 3a + b = 1.
    let d = 3 * raw(a) + raw(b) - raw(Fix128::ONE);
    eprintln!("[p2quad] 3a + b - 1 = {d} ulp");
    assert!(d.abs() <= 8, "3a + b must be one: off by {d} ulp");
}

/// Neither rule needs anything beyond the four operations and `sqrt`, and the
/// rational one does not need `sqrt` either.
///
/// This is a statement about the *source*, and the grep that backs it lives in
/// the report rather than here. What is checkable in a test is the consequence:
/// `Fix128` has `sin`, `cos`, `atan`, `atan2` and `exp`, and those are CORDIC or
/// series — fixed-iteration and so deterministic, but without the exact
/// algebraic identities the four operations have. A P2 element that avoided
/// them keeps the stronger guarantee. The assertion records that the rational
/// rule's constants are reachable by division alone.
#[test]
fn the_rational_rule_reaches_its_constants_by_division_alone() {
    for (p, q) in [(1i64, 2i64), (1, 6), (1, 4), (4, 5), (9, 20)] {
        let v = ratio(p, q);
        // v·q must return to p, up to the rounding of one division.
        let back = v * int(q);
        let d = ulps(back, int(p));
        eprintln!(
            "[p2quad] {p}/{q}: raw ({}, {}), round-trip error {d} ulp",
            v.hi, v.lo
        );
        assert!(
            d.abs() <= 32,
            "{p}/{q} must round-trip through multiplication by {q}; off by {d} ulp"
        );
    }
}
