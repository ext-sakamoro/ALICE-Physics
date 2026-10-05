//! Can `Fix128` carry the quadrature a **P3** (20-node cubic) tetrahedron needs?
//!
//! This is the P3 counterpart of `tests/p2_quadrature_fix128.rs`, and it is a
//! harder question for the same reason the element is a bigger one. A P3
//! tetrahedron's shape functions are cubic, so their gradients are quadratic,
//! so the integrand of the element stiffness `BᵀDB` is **quartic** in the
//! barycentric coordinates on a straight-edged element. Integrating it exactly
//! needs a degree-4 rule, where P2 needed degree 2.
//!
//! # ⚠️ No degree-2-or-higher rule can be fully dyadic, so "pick the rational
//! rule" is not available
//!
//! What a binary fixed-point type holds exactly is the **dyadic** rationals —
//! denominator a power of two — and that `1/6` is in the same position as `√5`.
//! For quadrature on a tetrahedron that observation has a sharp consequence,
//! which `no_tetrahedron_rule_of_degree_two_or_more_can_be_fully_dyadic` below
//! measures:
//!
//! > The exact moments are `∫λ₁^a λ₂^b λ₃^c λ₄^d dV / V = a!b!c!d!·3!/(a+b+c+d+3)!`.
//! > At degree 2 that gives `1/10`; at degree 4 it gives `1/35`, `1/140`,
//! > `1/210`, `1/420`, `1/840`. **None of these is dyadic.** A rule with dyadic
//! > points and dyadic weights produces a sum of products of dyadics, which is
//! > dyadic, and so can never equal `1/10` or `1/35`.
//!
//! Exactness in this arithmetic is therefore unreachable in principle, and the
//! question is only **how many roundings there are and whether they cancel**.
//! The answer is not the same at degree 4 as it was at degree 2.
//!
//! # What is measured, and what it decided
//!
//! Three rules, every barycentric monomial up to degree 5, error reported in
//! **ulps of the `Fix128` encoding** (see "why ulps" below):
//!
//! | rule | degree | points | worst ulp within its degree |
//! |---|---|---|---|
//! | Hammer–Stroud 4pt (what P2 landed with) | 2 | 4 | **−2** |
//! | Keast 11pt (the textbook degree-4 rule) | 4 | 11 | **−11** |
//! | the dyadic-abscissa rule used by `cubic_elastic_fem` | 4 (5 in fact) | 24 | **−7** |
//!
//! So moving from P2 to P3 costs a factor of 3.5 in the quadrature constants,
//! not a factor of a thousand: **`Fix128` carries a P3 element.** That was the
//! open question before this file existed.
//!
//! ⚠️ **Two expectations going in were wrong, and both are pinned below.**
//!
//! 1. *"Prefer an all-positive rule; negative weights cancel and cost digits."*
//!    That heuristic is in the memo above and it does not survive here. Over a
//!    search of degree-4 rules with dyadic abscissae, the all-positive ones
//!    measured **−14 ulp** and the best rule found has one small negative
//!    weight (`−16/315`) and measures **−7**. The reason is that forcing every
//!    weight positive pushes the weights onto much larger odd denominators
//!    (`25279/694575` and the like), and each of those rounds; one weight of
//!    `−16/315` rounds once and cancels against terms of comparable size.
//!    **At equal degree, count the odd denominators before counting the signs.**
//! 2. *"The textbook rule is the one to use."* Keast's 11 points are fewer and
//!    measure worse (−11 against −7), because its abscissae are irrational
//!    (`(1 ± √(5/14))/4`) **and** its weights have denominators 1875, 7500 and
//!    375. The rule this crate uses instead puts every abscissa on a dyadic
//!    value, which costs points (24 against 11) and buys two things: the
//!    abscissae themselves are exact, and — measured in
//!    `the_cubic_shape_functions_sum_to_exactly_one_at_the_dyadic_abscissae` —
//!    **the cubic shape functions evaluate at those points with no rounding at
//!    all**, because their coefficients (`1/2`, `9/2`, `27`) are dyadic too.
//!    That last property is invisible in a table of moments and is the reason
//!    for the choice.
//!
//! # Why ulps and not `to_f64`
//!
//! `f64` carries 53 bits of mantissa and `Fix128` carries 64 bits of fraction,
//! so a difference of a few ulps is *invisible* through `to_f64`; every row of
//! the table above would print `0.000e0`. Same instrument mistake as in
//! `tests/p2_quadrature_fix128.rs`.
//!
//! The reference moments are rounded to nearest from exact rational arithmetic
//! **off-line**, not derived here with `Fix128` division — otherwise the
//! rounding being measured would also sit in the reference and cancel out.
//!
//! Author: Moroya Sakamoto

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

/// `numer / 2^shift`, built from the bit pattern so that no division is
/// involved and the value is exact by construction.
fn dyadic(numer: i64, shift: u32) -> Fix128 {
    let r = (numer as i128) << (64 - shift);
    Fix128::from_raw((r >> 64) as i64, r as u64)
}

// ---------------------------------------------------------------------------
// exact moments, rounded to nearest off-line from a!b!c!d!·3!/(a+b+c+d+3)!
// ---------------------------------------------------------------------------

/// `∫ λ₁^a λ₂^b λ₃^c λ₄^d dV / V`, rounded to nearest in the `Fix128` encoding.
///
/// The value depends only on the multiset of exponents, so the table is keyed by
/// the exponents sorted descending. Degrees 0..=6 are covered: 0..=4 is what a
/// P3 stiffness needs, 5 shows that the rule in use has one degree of headroom,
/// and 6 shows the cliff on the other side of it.
fn exact_moment(powers: [u32; 4]) -> Fix128 {
    let mut p = powers;
    p.sort_unstable();
    p.reverse();
    let (hi, lo): (i64, u64) = match p {
        [0, 0, 0, 0] => (1, 0),
        // degree 1: 1/4
        [1, 0, 0, 0] => (0, 4_611_686_018_427_387_904),
        // degree 2: 1/10, 1/20
        [2, 0, 0, 0] => (0, 1_844_674_407_370_955_162),
        [1, 1, 0, 0] => (0, 922_337_203_685_477_581),
        // degree 3: 1/20, 1/60, 1/120
        [3, 0, 0, 0] => (0, 922_337_203_685_477_581),
        [2, 1, 0, 0] => (0, 307_445_734_561_825_860),
        [1, 1, 1, 0] => (0, 153_722_867_280_912_930),
        // degree 4: 1/35, 1/140, 1/210, 1/420, 1/840
        [4, 0, 0, 0] => (0, 527_049_830_677_415_760),
        [3, 1, 0, 0] => (0, 131_762_457_669_353_940),
        [2, 2, 0, 0] => (0, 87_841_638_446_235_960),
        [2, 1, 1, 0] => (0, 43_920_819_223_117_980),
        [1, 1, 1, 1] => (0, 21_960_409_611_558_990),
        // degree 5: 1/56, 1/280, 1/560, 1/1120, 1/1680, 1/3360
        [5, 0, 0, 0] => (0, 329_406_144_173_384_850),
        [4, 1, 0, 0] => (0, 65_881_228_834_676_970),
        [3, 2, 0, 0] => (0, 32_940_614_417_338_485),
        [3, 1, 1, 0] => (0, 16_470_307_208_669_243),
        [2, 2, 1, 0] => (0, 10_980_204_805_779_495),
        [2, 1, 1, 1] => (0, 5_490_102_402_889_748),
        // degree 6: 1/84 … 1/15120
        [6, 0, 0, 0] => (0, 219_604_096_115_589_900),
        [5, 1, 0, 0] => (0, 36_600_682_685_931_650),
        [4, 2, 0, 0] => (0, 14_640_273_074_372_660),
        [4, 1, 1, 0] => (0, 7_320_136_537_186_330),
        [3, 3, 0, 0] => (0, 10_980_204_805_779_495),
        [3, 2, 1, 0] => (0, 3_660_068_268_593_165),
        [3, 1, 1, 1] => (0, 1_830_034_134_296_583),
        [2, 2, 2, 0] => (0, 2_440_045_512_395_443),
        [2, 2, 1, 1] => (0, 1_220_022_756_197_722),
        other => panic!("no reference moment for exponents {other:?} (degree > 6?)"),
    };
    Fix128::from_raw(hi, lo)
}

/// Every exponent tuple of total degree `d`, in a fixed order.
fn monomials(d: u32) -> Vec<[u32; 4]> {
    let mut out = Vec::new();
    for a in 0..=d {
        for b in 0..=d - a {
            for c in 0..=d - a - b {
                out.push([a, b, c, d - a - b - c]);
            }
        }
    }
    out
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
    fn integrate(&self, powers: [u32; 4]) -> Fix128 {
        let mut acc = Fix128::ZERO;
        for (p, w) in self.points.iter().zip(&self.weights) {
            let mut term = *w;
            for (k, coord) in p.iter().enumerate() {
                for _ in 0..powers[k] {
                    term = term * *coord;
                }
            }
            acc = acc + term;
        }
        acc
    }

    /// Worst error over every monomial of degree `0..=exact_to_degree`.
    fn worst_within_degree(&self) -> i128 {
        let mut worst = 0i128;
        for d in 0..=self.exact_to_degree {
            for m in monomials(d) {
                let e = ulps(self.integrate(m), exact_moment(m));
                if e.abs() > worst.abs() {
                    worst = e;
                }
            }
        }
        worst
    }
}

/// The four points of an `S31` orbit: `a` on one coordinate, `b = (1−a)/3` on
/// the other three.
fn orbit_s31(a: Fix128, b: Fix128) -> Vec<[Fix128; 4]> {
    (0..4)
        .map(|i| {
            let mut p = [b; 4];
            p[i] = a;
            p
        })
        .collect()
}

/// The six points of an `S22` orbit: `a` twice and `b = 1/2 − a` twice, in a
/// fixed pattern order.
fn orbit_s22(a: Fix128, b: Fix128) -> Vec<[Fix128; 4]> {
    const PATTERN: [[u8; 4]; 6] = [
        [0, 0, 1, 1],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 0],
        [1, 1, 0, 0],
    ];
    PATTERN
        .iter()
        .map(|pat| {
            let mut p = [a; 4];
            for (slot, &s) in p.iter_mut().zip(pat.iter()) {
                *slot = if s == 0 { a } else { b };
            }
            p
        })
        .collect()
}

/// The rule `cubic_elastic_fem` uses: five symmetric orbits, **every abscissa
/// dyadic**, exact to degree 5 in real arithmetic (one more than a P3 stiffness
/// needs), 24 points.
///
/// | orbit | points | barycentric | weight |
/// |---|---|---|---|
/// | `S31(1)` | 4 | the vertices `(1,0,0,0)` | `1/300` |
/// | `S31(5/8)` | 4 | `(5/8, 1/8, 1/8, 1/8)` | `8/63` |
/// | `S31(1/16)` | 4 | `(1/16, 5/16, 5/16, 5/16)` | `256/1575` |
/// | `S22(1/2)` | 6 | the edge midpoints `(1/2, 1/2, 0, 0)` | `1/45` |
/// | `S22(3/8)` | 6 | `(3/8, 3/8, 1/8, 1/8)` | `−16/315` |
///
/// Found by fixing the abscissae to dyadic values and solving the five moment
/// conditions for degree 4 in exact rational arithmetic, then checking the
/// result against every monomial up to degree 7 (exact through 5, inexact from
/// 6). It is not a rule from the literature; its exactness is measured here
/// rather than cited.
fn cubic_dyadic_24() -> Rule {
    let zero = Fix128::ZERO;
    let mut points = Vec::with_capacity(24);
    let mut weights = Vec::with_capacity(24);
    let orbits: [(Vec<[Fix128; 4]>, Fix128); 5] = [
        (orbit_s31(dyadic(1, 0), zero), ratio(1, 300)),
        (orbit_s31(dyadic(5, 3), dyadic(1, 3)), ratio(8, 63)),
        (orbit_s31(dyadic(1, 4), dyadic(5, 4)), ratio(256, 1575)),
        (orbit_s22(dyadic(1, 1), zero), ratio(1, 45)),
        (orbit_s22(dyadic(3, 3), dyadic(1, 3)), ratio(-16, 315)),
    ];
    for (pts, w) in orbits {
        for p in pts {
            points.push(p);
            weights.push(w);
        }
    }
    Rule {
        name: "dyadic-abscissa 24pt (deg 4, exact to 5)",
        exact_to_degree: 4,
        points,
        weights,
    }
}

/// Keast's eleven points, the textbook degree-4 rule: centroid with weight
/// `−148/1875`, the `S31(11/14)` orbit with `343/7500`, and the `S22` orbit at
/// `(1 ± √(5/14))/4` with `56/375`.
fn keast_11() -> Rule {
    let mut points = vec![[ratio(1, 4); 4]];
    let mut weights = vec![ratio(-148, 1875)];
    for p in orbit_s31(ratio(11, 14), ratio(1, 14)) {
        points.push(p);
        weights.push(ratio(343, 7500));
    }
    let r = int(5).sqrt() / int(14).sqrt();
    let a = (Fix128::ONE + r) / int(4);
    let b = (Fix128::ONE - r) / int(4);
    for p in orbit_s22(a, b) {
        points.push(p);
        weights.push(ratio(56, 375));
    }
    Rule {
        name: "Keast 11pt (deg 4, textbook)",
        exact_to_degree: 4,
        points,
        weights,
    }
}

/// Hammer–Stroud's four points, the rule `quadratic_elastic_fem` landed with.
/// Carried here as the scale against which the P3 numbers are read.
fn hammer_stroud_4() -> Rule {
    let root5 = int(5).sqrt();
    let a = (int(5) - root5) / int(20);
    let b = (int(5) + int(3) * root5) / int(20);
    let mut points = Vec::with_capacity(4);
    for i in 0..4 {
        let mut p = [a; 4];
        p[i] = b;
        points.push(p);
    }
    Rule {
        name: "Hammer-Stroud 4pt (deg 2, the P2 baseline)",
        exact_to_degree: 2,
        points,
        weights: vec![ratio(1, 4); 4],
    }
}

// ---------------------------------------------------------------------------
// the cubic shape functions, written out here so this file does not depend on
// the element implementation it justifies
// ---------------------------------------------------------------------------

const EDGES: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
const FACES: [(usize, usize, usize); 4] = [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)];

/// The twenty cubic Lagrange shape functions at one barycentric point.
///
/// Corner `i`: `½ λᵢ(3λᵢ−1)(3λᵢ−2)`. Edge node two thirds of the way to `i`
/// along `(i,j)`: `9/2 λᵢλⱼ(3λᵢ−1)`. Face node of `(i,j,k)`: `27 λᵢλⱼλₖ`.
///
/// ⚠️ **Every coefficient here — `1/2`, `9/2`, `27`, `3`, `1`, `2` — is
/// dyadic.** That is why a rule with dyadic abscissae evaluates these with no
/// rounding whatever, and it is the property the element is built around.
fn cubic_shape_values(l: &[Fix128; 4]) -> [Fix128; 20] {
    let half = dyadic(1, 1);
    let nine_halves = dyadic(9, 1);
    let three = int(3);
    let two = int(2);
    let twenty_seven = int(27);
    let mut n = [Fix128::ZERO; 20];
    for (slot, &li) in n.iter_mut().zip(l.iter()) {
        *slot = half * li * (three * li - Fix128::ONE) * (three * li - two);
    }
    for (e, &(i, j)) in EDGES.iter().enumerate() {
        let (li, lj) = (l[i], l[j]);
        n[4 + 2 * e] = nine_halves * li * lj * (three * li - Fix128::ONE);
        n[5 + 2 * e] = nine_halves * lj * li * (three * lj - Fix128::ONE);
    }
    for (f, &(i, j, k)) in FACES.iter().enumerate() {
        n[16 + f] = twenty_seven * l[i] * l[j] * l[k];
    }
    n
}

// ---------------------------------------------------------------------------
// the measurements
// ---------------------------------------------------------------------------

/// What a binary fixed-point type holds exactly, stated as a measurement rather
/// than as a remark.
///
/// A dyadic rational round-trips through multiplication by its denominator with
/// **zero** ulp of error; a rational with an odd factor in the denominator does
/// not. That is the reason
/// the rule above fixes its abscissae on `1/16`, `1/8`, `5/16`, `3/8`, `1/2` and
/// `5/8` rather than on the values a textbook rule would use.
#[test]
fn dyadic_constants_are_exact_and_others_are_not() {
    eprintln!("[p3quad] --- round-trip error of p/q through multiplication by q ---");
    for (p, q, want_exact) in [
        (1i64, 2i64, true),
        (1, 4, true),
        (1, 8, true),
        (1, 16, true),
        (5, 16, true),
        (3, 8, true),
        (5, 8, true),
        (1, 3, false),
        (1, 300, false),
        (8, 63, false),
        (256, 1575, false),
        (16, 315, false),
    ] {
        let v = ratio(p, q);
        let d = ulps(v * int(q), int(p));
        eprintln!(
            "[p3quad] {p:>4}/{q:<5} raw ({}, {}) round-trip {d} ulp  (dyadic: {want_exact})",
            v.hi, v.lo
        );
        if want_exact {
            assert_eq!(
                d, 0,
                "{p}/{q} has a power-of-two denominator and must be exact; off by {d} ulp"
            );
        } else {
            assert_ne!(
                d, 0,
                "{p}/{q} has an odd factor in its denominator, so it cannot be exact in a \
                 binary fixed-point type; a zero here means the instrument stopped working"
            );
        }
    }

    // And the same statement about the abscissae actually used: built from the
    // bit pattern, they agree with the division to the last bit.
    for (p, shift) in [(1i64, 1u32), (1, 3), (3, 3), (5, 3), (1, 4), (5, 4)] {
        let by_bits = dyadic(p, shift);
        let by_division = ratio(p, 1 << shift);
        assert_eq!(
            ulps(by_bits, by_division),
            0,
            "{p}/2^{shift} must be the same value however it is reached"
        );
    }
}

/// ⚠️ Exactness in this arithmetic is impossible, and this is why.
///
/// The degree-2 moment is `1/10` and the degree-4 moments are `1/35`, `1/140`,
/// `1/210`, `1/420` and `1/840`. Every one of them has an odd factor in its
/// denominator, so **none of them is representable at all**. A rule whose points
/// and weights were all dyadic would compute a sum of products of dyadics, which
/// is dyadic, and could therefore never land on these values.
///
/// The consequence is the shape of the whole investigation: the question for a
/// P3 element is not "is there an exact rule" but "how few roundings can it be
/// done in", and that is what the ulp table below answers.
#[test]
fn no_tetrahedron_rule_of_degree_two_or_more_can_be_fully_dyadic() {
    // Representable: the moments a degree-1 rule has to match.
    for (p, q) in [(1i64, 1i64), (1, 4)] {
        let d = ulps(ratio(p, q) * int(q), int(p));
        assert_eq!(d, 0, "{p}/{q} is dyadic and must be exact");
    }
    // Not representable: the first moment a degree-2 rule has to match, and
    // every moment a degree-4 rule has to match.
    let mut unrepresentable = 0;
    for (p, q, deg) in [
        (1i64, 10i64, 2),
        (1, 20, 2),
        (1, 35, 4),
        (1, 140, 4),
        (1, 210, 4),
        (1, 420, 4),
        (1, 840, 4),
    ] {
        let d = ulps(ratio(p, q) * int(q), int(p));
        eprintln!("[p3quad] degree-{deg} moment {p}/{q}: round-trip {d} ulp");
        assert_ne!(
            d, 0,
            "{p}/{q} must NOT be representable; if it is, the impossibility \
             argument in this file's header is wrong"
        );
        unrepresentable += 1;
    }
    assert_eq!(
        unrepresentable, 7,
        "the count is part of the claim: every degree-2 and degree-4 moment is \
         outside the dyadic rationals"
    );
}

/// All three rules must be partitions of the reference simplex before their
/// errors mean anything. A mistyped constant fails here rather than showing up
/// as a plausible-looking error further down.
#[test]
fn every_rule_is_a_partition_of_the_simplex() {
    for rule in [hammer_stroud_4(), keast_11(), cubic_dyadic_24()] {
        let mut wsum = Fix128::ZERO;
        for w in &rule.weights {
            wsum = wsum + *w;
        }
        let dw = ulps(wsum, Fix128::ONE);
        eprintln!(
            "[p3quad] {}: {} points, weight sum error {dw} ulp",
            rule.name,
            rule.points.len()
        );
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
    let hs = hammer_stroud_4();
    let keast = keast_11();
    let dyad = cubic_dyadic_24();

    eprintln!("[p3quad] ---- error vs the exact rational moment, in ulps of Fix128 ----");
    eprintln!(
        "[p3quad] {:>5}  {:>14}  {:>14}  {:>14}",
        "deg", "Hammer-Stroud", "Keast 11pt", "dyadic 24pt"
    );
    for d in 0..=5u32 {
        let mut worst = [0i128; 3];
        for m in monomials(d) {
            for (k, r) in [&hs, &keast, &dyad].into_iter().enumerate() {
                let e = ulps(r.integrate(m), exact_moment(m));
                if e.abs() > worst[k].abs() {
                    worst[k] = e;
                }
            }
        }
        eprintln!(
            "[p3quad] {:>5}  {:>14}  {:>14}  {:>14}",
            d, worst[0], worst[1], worst[2]
        );
    }

    let (w_hs, w_keast, w_dyad) = (
        hs.worst_within_degree(),
        keast.worst_within_degree(),
        dyad.worst_within_degree(),
    );
    eprintln!(
        "[p3quad] worst within each rule's exactness degree: \
         Hammer-Stroud(2) {w_hs}, Keast(4) {w_keast}, dyadic(4) {w_dyad}"
    );

    // Within the degree each rule is exact to, the only error left is the
    // rounding of its own constants. A rule off by more than a handful of ulps
    // has a wrong constant, not a rounding problem.
    for (name, worst) in [(hs.name, w_hs), (keast.name, w_keast), (dyad.name, w_dyad)] {
        assert!(
            worst.abs() <= 64,
            "{name}: exact in real arithmetic over its degree, so the residual must be \
             rounding only; got {worst} ulp"
        );
    }

    // ⚠️ The gate this file was written to answer: is degree 4 in `Fix128`
    // *qualitatively* worse than degree 2, or only quantitatively? Measured: a
    // factor of 3.5, so the element is viable. Pinned at 8x so that a change
    // which makes P3 quadrature an order of magnitude worse than P2's has to
    // come here and say so.
    assert!(
        w_dyad.abs() <= 8 * w_hs.abs(),
        "P3 quadrature must stay within an order of magnitude of P2's: dyadic 24pt \
         {w_dyad} ulp against Hammer-Stroud {w_hs} ulp"
    );

    // ⚠️ Measured against the expectation that the textbook rule would win: it
    // does not. Keast's eleven points pay for irrational abscissae *and* for
    // denominators 1875 / 7500 / 375; the dyadic rule pays only for its five
    // weights. Pinned in this direction so that a future change making the
    // textbook rule the better one has to update the module header.
    assert!(
        w_dyad.abs() <= w_keast.abs(),
        "the dyadic-abscissa rule was chosen because it measures better than the \
         textbook one; got dyadic {w_dyad} ulp against Keast {w_keast} ulp"
    );
}

/// The rule in use is exact to degree 5, one past what a P3 stiffness needs, and
/// is wrong by ~10¹⁵ ulp at degree 6.
///
/// Both halves matter. The headroom is what makes the rule safe for the
/// consistent load vector of a constant body force (`Nᵢ·f` is cubic) and for a
/// P2 mass matrix (degree 4) if either is ever wanted. The cliff is spelled out
/// with a number so that the next reader does not reach for it at degree 6: a
/// rule used past its exactness degree is not slightly worse, it is wrong.
#[test]
fn the_dyadic_rule_is_exact_to_degree_five_and_falls_off_a_cliff_at_six() {
    let rule = cubic_dyadic_24();
    for d in 0..=5u32 {
        let mut worst = 0i128;
        for m in monomials(d) {
            let e = ulps(rule.integrate(m), exact_moment(m));
            if e.abs() > worst.abs() {
                worst = e;
            }
        }
        eprintln!("[p3quad] degree {d}: worst {worst} ulp");
        assert!(
            worst.abs() <= 64,
            "the rule is exact to degree 5 in real arithmetic, so degree {d} must be \
             rounding only; got {worst} ulp"
        );
    }

    let mut worst_six = 0i128;
    for m in monomials(6) {
        let e = ulps(rule.integrate(m), exact_moment(m));
        if e.abs() > worst_six.abs() {
            worst_six = e;
        }
    }
    eprintln!("[p3quad] degree 6: worst {worst_six} ulp  <- past the exactness degree");
    assert!(
        worst_six.abs() > 1_000_000,
        "the degree-6 row exists to show the cliff; the rule was off by only \
         {worst_six} ulp, so either the rule or the reference moved"
    );
}

/// ⚠️ The measurement that actually decided the rule, and the one a table of
/// moments cannot show.
///
/// The twenty cubic shape functions sum to one identically. Evaluated at the
/// dyadic abscissae every intermediate product is a dyadic rational that fits
/// in 64 fractional bits, so the identity holds **to the bit**; evaluated at
/// Hammer–Stroud's or Keast's irrational abscissae it does not. A P3 element
/// built on this rule therefore starts its stiffness assembly from exact shape
/// function values, and the only rounding left in the integrand comes from the
/// element geometry.
#[test]
fn the_cubic_shape_functions_sum_to_exactly_one_at_the_dyadic_abscissae() {
    let dyad = cubic_dyadic_24();
    let mut worst_dyadic = 0i128;
    for (i, p) in dyad.points.iter().enumerate() {
        let n = cubic_shape_values(p);
        let mut s = Fix128::ZERO;
        for v in &n {
            s = s + *v;
        }
        let d = ulps(s, Fix128::ONE);
        if d.abs() > worst_dyadic.abs() {
            worst_dyadic = d;
        }
        assert_eq!(
            d, 0,
            "dyadic point {i} must evaluate the cubic basis exactly; off by {d} ulp"
        );
    }
    eprintln!("[p3quad] partition of unity at the 24 dyadic points: worst {worst_dyadic} ulp");

    let mut worst_irrational = 0i128;
    for rule in [hammer_stroud_4(), keast_11()] {
        for p in &rule.points {
            let n = cubic_shape_values(p);
            let mut s = Fix128::ZERO;
            for v in &n {
                s = s + *v;
            }
            let d = ulps(s, Fix128::ONE);
            if d.abs() > worst_irrational.abs() {
                worst_irrational = d;
            }
        }
    }
    eprintln!(
        "[p3quad] partition of unity at the irrational-abscissa points: worst \
         {worst_irrational} ulp"
    );
    assert_ne!(
        worst_irrational, 0,
        "the comparison is the point: if the irrational abscissae also evaluate the \
         cubic basis exactly, then dyadic abscissae bought nothing and the rule \
         should be reconsidered"
    );
}
