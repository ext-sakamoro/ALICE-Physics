//! Oracles for [`polygon_offset::erode`].
//!
//! # The independent truth: a scanline measurement
//!
//! `support/eroded_area.rs` measures the eroded area without the code under
//! test and without any formula for it: per scan row, the inside intervals
//! minus the union of the row's intersections with every segment's capsule of
//! radius `d` (each an exact interval), integrated by the midpoint rule at `n`
//! and `2n` rows with their difference as the bound. It holds for any outline,
//! including the logo letters whose bends erode away.
//!
//! # The eroded area in closed form
//!
//! Let `P` be a polygon (holes allowed) with area `A`, perimeter `L`, and erode
//! it by `d`. Cover the removed band by one `d`-wide strip per edge, cut at the
//! normals through the edge's ends: together they hold `L d`. At a vertex with
//! interior angle `θ`:
//!
//! - convex (`θ < π`): the two strips overlap in a kite with corners at the
//!   vertex, at the eroded corner (distance `d / sin(θ/2)` along the bisector)
//!   and at the two strip edges; it is counted twice, so add it back:
//!   area `d² cot(θ/2)`.
//! - reflex (`θ > π`): the strips leave a gap, the sector of radius `d` and
//!   angle `θ - π` between the two normals, which is removed too: subtract
//!   `d² (θ - π) / 2`.
//!
//! `A(d) = A - L d + d² Σ_convex cot(θ/2) - d² Σ_reflex (θ - π) / 2`, as long as
//! no edge vanishes (every edge longer than its convex corners consume,
//! `d (cot(θ₁/2) + cot(θ₂/2))`) and no two non-neighbouring strips meet. The
//! test checks the first condition explicitly before using the formula, and the
//! formula against the scanline truth, on shapes where both hold.
//!
//! # The eroded polygon
//!
//! Every point of it is within the configured sagitta of the true eroded
//! boundary (arc chords; an arc one chord covers is its midpoint, joined to its
//! shifted edges by segments that stay within the same sagitta), so its shoelace
//! area differs from the truth by at most `sagitta x (eroded perimeter)`.
#![cfg(feature = "std")]
#![allow(
    clippy::disallowed_methods,
    reason = "analytic reference: in f64, angles and areas are computed from the input, never by the code under test"
)]

#[path = "support/eroded_area.rs"]
mod eroded_area;

use alice_physics_core::polygon_offset::{erode, OffsetConfig, OffsetError, OffsetNormal};
use eroded_area::eroded_area;

const SKIN: f64 = 0.012;

fn shoelace(c: &[[f64; 2]]) -> f64 {
    let n = c.len();
    0.5 * (0..n)
        .map(|i| c[i][0] * c[(i + 1) % n][1] - c[(i + 1) % n][0] * c[i][1])
        .sum::<f64>()
}

fn perimeter(c: &[Vec<[f64; 2]>]) -> f64 {
    c.iter()
        .map(|r| {
            (0..r.len())
                .map(|i| {
                    let (p, q) = (r[i], r[(i + 1) % r.len()]);
                    ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt()
                })
                .sum::<f64>()
        })
        .sum()
}

fn oriented(c: &[Vec<[f64; 2]>]) -> Vec<Vec<[f64; 2]>> {
    let outer = c
        .iter()
        .map(|r| shoelace(r))
        .fold(0.0f64, |m, a| if a.abs() > m.abs() { a } else { m });
    c.iter()
        .map(|r| {
            if outer < 0.0 {
                r.iter().rev().copied().collect()
            } else {
                r.clone()
            }
        })
        .collect()
}

/// `A(d)` by the closed form, or `None` when an edge vanishes (the formula's
/// precondition fails).
fn closed_form(c: &[Vec<[f64; 2]>], d: f64) -> Option<f64> {
    let c = oriented(c);
    let mut a = 0.0;
    for r in &c {
        a += shoelace(r);
        let n = r.len();
        let angle = |i: usize| -> f64 {
            let (p, v, q) = (r[(i + n - 1) % n], r[i], r[(i + 1) % n]);
            let e = [v[0] - p[0], v[1] - p[1]];
            let f = [q[0] - v[0], q[1] - v[1]];
            std::f64::consts::PI - (e[0] * f[1] - e[1] * f[0]).atan2(e[0] * f[0] + e[1] * f[1])
        };
        for i in 0..n {
            let (v, q) = (r[i], r[(i + 1) % n]);
            let len = ((q[0] - v[0]).powi(2) + (q[1] - v[1]).powi(2)).sqrt();
            let eat = |theta: f64| {
                if theta < std::f64::consts::PI {
                    d / (theta / 2.0).tan()
                } else {
                    0.0
                }
            };
            if len <= eat(angle(i)) + eat(angle((i + 1) % n)) {
                return None;
            }
            a -= d * len;
            let theta = angle(i);
            if theta < std::f64::consts::PI {
                a += d * d / (theta / 2.0).tan();
            } else if theta > std::f64::consts::PI {
                a -= d * d * (theta - std::f64::consts::PI) / 2.0;
            }
        }
    }
    Some(a)
}

fn logo() -> Vec<(char, Vec<Vec<[f64; 2]>>)> {
    include_str!("data/logo_outlines.txt")
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            let (ch, rest) = l.split_once(' ').expect("char and contours");
            let contours = rest
                .split(';')
                .map(|c| {
                    c.split_whitespace()
                        .map(|p| {
                            let (x, y) = p.split_once(',').expect("x,y");
                            [x.parse().expect("x"), y.parse().expect("y")]
                        })
                        .collect()
                })
                .collect();
            (ch.chars().next().expect("char"), contours)
        })
        .collect()
}

fn cfg() -> OffsetConfig {
    OffsetConfig { sagitta: 1.0e-5 }
}

/// erode's polygon against the scanline truth; the formula too when it applies.
/// Returns erode's polygon area.
fn check(name: &str, c: &[Vec<[f64; 2]>]) -> f64 {
    let (truth, err) = eroded_area(c, SKIN, 4000);
    let e = erode(c, SKIN, &cfg()).unwrap_or_else(|x| panic!("{name}: {x:?}"));
    let got: f64 = e.points().iter().map(|r| shoelace(r)).sum();
    let chord = cfg().sagitta * perimeter(&e.points());
    assert!(
        (got - truth).abs() <= chord + err,
        "{name}: eroded polygon {got:.9e} vs scanline {truth:.9e} (chord {chord:.2e} + scanline {err:.2e})"
    );
    if let Some(formula) = closed_form(c, SKIN) {
        assert!(
            (formula - truth).abs() <= err,
            "{name}: closed form {formula:.9e} vs scanline {truth:.9e} (scanline bound {err:.2e})"
        );
    }
    got
}

#[test]
fn a_square_erodes_to_the_inner_square() {
    let c = vec![vec![[0.0, 0.0], [0.2, 0.0], [0.2, 0.2], [0.0, 0.2]]];
    let a = check("square", &c);
    // a square's eroded polygon is the exact inner square (no arcs, no chords)
    assert!((a - (0.2 - 2.0 * SKIN).powi(2)).abs() < 1e-15);
}

#[test]
fn an_l_erodes_with_an_arc_at_its_inside_corner() {
    let c = vec![vec![
        [0.0, 0.0],
        [0.2, 0.0],
        [0.2, 0.08],
        [0.08, 0.08],
        [0.08, 0.2],
        [0.0, 0.2],
    ]];
    check("L", &c);
    let e = erode(&c, SKIN, &cfg()).expect("L");
    let arc: Vec<_> = e.contours[0]
        .iter()
        .filter(|v| matches!(v.normal, OffsetNormal::Arc { .. }))
        .collect();
    assert!(arc.len() >= 3);
    for v in arc {
        let r = ((v.p[0] - 0.08).powi(2) + (v.p[1] - 0.08).powi(2)).sqrt();
        // a junction of the arc's tangent polygon: outside the circle by at most
        // the sagitta, never inside it
        assert!(
            r >= SKIN - 1e-15 && r <= SKIN + cfg().sagitta,
            "arc point at {r}"
        );
    }
}

#[test]
fn a_square_hole_grows_by_the_offset() {
    let c = vec![
        vec![[0.0, 0.0], [0.3, 0.0], [0.3, 0.3], [0.0, 0.3]],
        vec![[0.1, 0.1], [0.1, 0.2], [0.2, 0.2], [0.2, 0.1]],
    ];
    check("holed square", &c);
}

#[test]
fn a_notch_shorter_than_the_offset_closes_on_a_straight_edge() {
    // a 4 mm deep, 6 mm wide notch in the bottom edge: the erosion closes it and
    // the bottom is one straight line again
    let c = vec![vec![
        [0.0, 0.0],
        [0.097, 0.0],
        [0.097, 0.004],
        [0.103, 0.004],
        [0.103, 0.0],
        [0.2, 0.0],
        [0.2, 0.2],
        [0.0, 0.2],
    ]];
    check("notch", &c);
}

/// A tab 4 mm out and 6 mm wide on a straight edge erodes away entirely: the
/// two halves of the edge are one line again (merged, not refused as parallel).
#[test]
fn a_tab_thinner_than_twice_the_offset_erodes_away() {
    let c = vec![vec![
        [0.0, 0.0],
        [0.097, 0.0],
        [0.097, -0.004],
        [0.103, -0.004],
        [0.103, 0.0],
        [0.2, 0.0],
        [0.2, 0.2],
        [0.0, 0.2],
    ]];
    check("tab", &c);
    let e = erode(&c, SKIN, &cfg()).expect("tab");
    // the bottom of the eroded square is the straight line y = SKIN
    let bottom: Vec<_> = e.contours[0].iter().filter(|v| v.p[1] < 0.05).collect();
    assert_eq!(bottom.len(), 2, "{bottom:?}");
    for v in bottom {
        assert!((v.p[1] - SKIN).abs() < 1e-15, "{:?}", v.p);
    }
}

/// erode's polygon area per letter against the scanline truth (which needs no
/// precondition: `r` and `a` lose edges at 12 mm, the closed form does not
/// apply to them), and erode's areas pinned as a regression check of erode
/// itself (m², SKIN = 12 mm, sagitta 10 µm).
fn letter(ch: char, pinned: f64) {
    let (_, c) = logo().into_iter().find(|(x, _)| *x == ch).expect("letter");
    let got = check(&ch.to_string(), &c);
    assert!(
        (got - pinned).abs() <= 1e-12 * pinned,
        "{ch}: erode's area {got:.17} vs pinned {pinned:.17}"
    );
}

#[test]
fn letter_e_erodes_to_the_scanline_area() {
    letter('E', PIN_E);
}
#[test]
fn letter_x_erodes_to_the_scanline_area() {
    letter('x', PIN_X);
}
#[test]
fn letter_t_erodes_to_the_scanline_area() {
    letter('t', PIN_T);
}
#[test]
fn letter_o_erodes_to_the_scanline_area() {
    letter('o', PIN_O);
}
#[test]
fn letter_r_erodes_to_the_scanline_area() {
    letter('r', PIN_R);
}
#[test]
fn letter_i_erodes_to_the_scanline_area() {
    letter('i', PIN_I);
}
#[test]
fn letter_a_erodes_to_the_scanline_area() {
    letter('a', PIN_A);
}

const PIN_E: f64 = 0.124_203_919_797_288_18;
const PIN_X: f64 = 0.079_827_876_527_074_88;
const PIN_T: f64 = 0.069_320_212_246_059_22;
const PIN_O: f64 = 0.103_705_751_825_986_85;
const PIN_R: f64 = 0.052_494_728_555_283_53;
const PIN_I: f64 = 0.045_645_659_999_999_81;
const PIN_A: f64 = 0.105_168_753_296_534_33;

#[test]
fn degenerate_and_impossible_erosions_are_refused() {
    let sq = vec![vec![[0.0, 0.0], [0.02, 0.0], [0.02, 0.02], [0.0, 0.02]]];
    assert!(matches!(
        erode(&sq, SKIN, &cfg()),
        Err(OffsetError::Vanished { contour: 0 })
    ));
    assert!(matches!(
        erode(&sq, 0.0, &cfg()),
        Err(OffsetError::BadConfig)
    ));
    assert!(matches!(
        erode(&sq, f64::NAN, &cfg()),
        Err(OffsetError::BadConfig)
    ));
    assert!(matches!(
        erode(&[], SKIN, &cfg()),
        Err(OffsetError::NoContours)
    ));
    let nan = vec![vec![[0.0, 0.0], [f64::NAN, 0.0], [0.2, 0.2]]];
    assert!(matches!(
        erode(&nan, SKIN, &cfg()),
        Err(OffsetError::NonFinite { contour: 0 })
    ));
    let line = vec![vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]];
    assert!(matches!(
        erode(&line, SKIN, &cfg()),
        Err(OffsetError::DegenerateContour { contour: 0 })
    ));
    // a zero-width slit into the bottom edge: a half turn
    let slit = vec![vec![
        [0.0, 0.0],
        [0.1, 0.0],
        [0.1, 0.05],
        [0.1, 0.0],
        [0.2, 0.0],
        [0.2, 0.2],
        [0.0, 0.2],
    ]];
    assert!(matches!(
        erode(&slit, SKIN, &cfg()),
        Err(OffsetError::DegenerateContour { contour: 0 })
    ));
    // a hole wound like the outside: refused, not shrunk as an island
    let same = vec![
        vec![[0.0, 0.0], [0.3, 0.0], [0.3, 0.3], [0.0, 0.3]],
        vec![[0.1, 0.1], [0.2, 0.1], [0.2, 0.2], [0.1, 0.2]],
    ];
    assert!(matches!(
        erode(&same, SKIN, &cfg()),
        Err(OffsetError::WindingMismatch { contour: 1 })
    ));
    // a hole that grows into the outer boundary: a wall thinner than 2 SKIN
    let thin = vec![
        vec![[0.0, 0.0], [0.3, 0.0], [0.3, 0.3], [0.0, 0.3]],
        vec![[0.015, 0.1], [0.015, 0.2], [0.2, 0.2], [0.2, 0.1]],
    ];
    assert!(matches!(
        erode(&thin, SKIN, &cfg()),
        Err(OffsetError::SelfIntersection)
    ));
}

/// An island beside the outside (the dot of an `i`) runs the same way and
/// erodes inward; it is not a hole.
#[test]
fn an_island_runs_with_the_outside_and_shrinks() {
    let c = vec![
        vec![[0.0, 0.0], [0.1, 0.0], [0.1, 0.3], [0.0, 0.3]],
        vec![[0.0, 0.4], [0.1, 0.4], [0.1, 0.5], [0.0, 0.5]],
    ];
    check("island", &c);
}
