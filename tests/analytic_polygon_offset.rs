//! Oracles for [`polygon_offset::erode`].
//!
//! # The eroded area in closed form
//!
//! Let `P` be a polygon (holes allowed) with area `A`, perimeter `L`, and erode
//! it by `d` (keep the points at least `d` from the boundary). Cover the removed
//! band by one `d`-wide strip per edge, cut at the normals through the edge's
//! ends: together they hold `L d`. At a vertex with interior angle `θ`:
//!
//! - convex (`θ < π`): the two strips overlap in a kite with corners at the
//!   vertex, at the eroded corner (distance `d / sin(θ/2)` along the bisector)
//!   and at the two strip edges; it is counted twice, so add it back:
//!   area `d² cot(θ/2)`.
//! - reflex (`θ > π`): the strips leave a gap, the sector of radius `d` and
//!   angle `θ - π` between the two normals, which is removed too: subtract
//!   `d² (θ - π) / 2`.
//!
//! `A(d) = A - L d + d² Σ_convex cot(θ/2) - d² Σ_reflex (θ - π) / 2`, as long
//! as no edge vanishes (every edge longer than its two corners consume).
//!
//! # Independent check of the formula
//!
//! The formula is the oracle's own derivation, so it is checked against a
//! measurement that shares nothing with it: the polygon's exact signed distance
//! (nearest segment, sign by crossing parity), sampled on a fine grid, counting
//! the cells with distance `≤ -d`. The count's error is bounded by the cells the
//! eroded boundary passes through: `(eroded perimeter) x h x sqrt(2)` with `h`
//! the grid step.
//!
//! # The eroded polygon
//!
//! Every point of it is within the configured sagitta of the true eroded
//! boundary (the arcs at reflex vertices are chords; an arc one chord covers is
//! its midpoint), so its shoelace area differs from `A(d)` by at most
//! `sagitta x perimeter`, and the code's output is held to that.
#![cfg(feature = "std")]
#![allow(
    clippy::disallowed_methods,
    reason = "analytic reference in f64: angles, areas and the sampled distance are computed from the input, never by the code under test"
)]

use alice_physics_core::polygon_offset::{erode, OffsetConfig, OffsetError};

const SKIN: f64 = 0.012;

fn shoelace(c: &[[f64; 2]]) -> f64 {
    let n = c.len();
    0.5 * (0..n)
        .map(|i| c[i][0] * c[(i + 1) % n][1] - c[(i + 1) % n][0] * c[i][1])
        .sum::<f64>()
}

/// Contours oriented so the inside is on the left (outer by the largest area).
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

/// `A(d)` by the closed form, and the total arc angle (for the chord bound).
fn closed_form(c: &[Vec<[f64; 2]>], d: f64) -> (f64, f64) {
    let c = oriented(c);
    let mut a = 0.0;
    let mut arc = 0.0;
    for r in &c {
        a += shoelace(r);
        let n = r.len();
        for i in 0..n {
            let (p, v, q) = (r[(i + n - 1) % n], r[i], r[(i + 1) % n]);
            let e = [v[0] - p[0], v[1] - p[1]];
            let f = [q[0] - v[0], q[1] - v[1]];
            a -= d * (e[0] * e[0] + e[1] * e[1]).sqrt();
            let turn = (e[0] * f[1] - e[1] * f[0]).atan2(e[0] * f[0] + e[1] * f[1]);
            // interior angle = pi - turn (left turn = convex)
            let theta = std::f64::consts::PI - turn;
            if turn > 0.0 {
                a += d * d / (theta / 2.0).tan();
            } else if turn < 0.0 {
                a -= d * d * (theta - std::f64::consts::PI) / 2.0;
                arc += theta - std::f64::consts::PI;
            }
        }
    }
    (a, arc * d)
}

fn seg_dist(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let e = [b[0] - a[0], b[1] - a[1]];
    let l = e[0] * e[0] + e[1] * e[1];
    let s = (((p[0] - a[0]) * e[0] + (p[1] - a[1]) * e[1]) / l).clamp(0.0, 1.0);
    ((p[0] - a[0] - s * e[0]).powi(2) + (p[1] - a[1] - s * e[1]).powi(2)).sqrt()
}

/// Area of `{signed distance ≤ -d}` by grid sampling, and the count's error bound.
fn sampled(c: &[Vec<[f64; 2]>], d: f64, h: f64) -> (f64, f64) {
    let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
    for r in c {
        for p in r {
            for k in 0..2 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
    }
    let segs: Vec<([f64; 2], [f64; 2])> = c
        .iter()
        .flat_map(|r| (0..r.len()).map(move |i| (r[i], r[(i + 1) % r.len()])))
        .collect();
    let (nx, ny) = (
        ((hi[0] - lo[0]) / h).ceil() as usize,
        ((hi[1] - lo[1]) / h).ceil() as usize,
    );
    let mut count = 0usize;
    for iy in 0..ny {
        for ix in 0..nx {
            let p = [lo[0] + (ix as f64 + 0.5) * h, lo[1] + (iy as f64 + 0.5) * h];
            let mut inside = false;
            let mut dist = f64::INFINITY;
            for &(a, b) in &segs {
                dist = dist.min(seg_dist(p, a, b));
                if (a[1] > p[1]) != (b[1] > p[1])
                    && p[0] < a[0] + (p[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
                {
                    inside = !inside;
                }
            }
            if inside && dist >= d {
                count += 1;
            }
        }
    }
    let per: f64 = segs
        .iter()
        .map(|(a, b)| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt())
        .sum();
    (count as f64 * h * h, per * h * 2.0f64.sqrt())
}

fn logo() -> Vec<(char, Vec<Vec<[f64; 2]>>)> {
    let text = include_str!("data/logo_outlines.txt");
    text.lines()
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

fn check(name: &str, c: &[Vec<[f64; 2]>], d: f64, h: f64) -> f64 {
    let (formula, arc_len) = closed_form(c, d);
    let (grid, grid_err) = sampled(c, d, h);
    assert!(
        (formula - grid).abs() <= grid_err,
        "{name}: closed form {formula:.6e} vs sampled {grid:.6e} (bound {grid_err:.2e})"
    );
    let e = erode(c, d, &cfg()).unwrap_or_else(|err| panic!("{name}: {err:?}"));
    let got: f64 = e.points().iter().map(|r| shoelace(r)).sum();
    // every point of the eroded polygon is within the sagitta of the true eroded
    // boundary (arc chords; an arc one chord covers is its midpoint, joined to
    // its shifted edges by segments that stay within the same sagitta), so the
    // areas differ by at most sagitta x perimeter
    let per: f64 = e
        .points()
        .iter()
        .map(|r| {
            (0..r.len())
                .map(|i| {
                    let (p, q) = (r[i], r[(i + 1) % r.len()]);
                    ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt()
                })
                .sum::<f64>()
        })
        .sum();
    let _ = arc_len;
    let chord = cfg().sagitta * per + 1e-12;
    assert!(
        (got - formula).abs() <= chord,
        "{name}: eroded polygon {got:.9e} vs closed form {formula:.9e} (chord bound {chord:.2e})"
    );
    formula
}

#[test]
fn a_square_erodes_to_the_inner_square() {
    let c = vec![vec![[0.0, 0.0], [0.2, 0.0], [0.2, 0.2], [0.0, 0.2]]];
    let a = check("square", &c, SKIN, 5e-4);
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
    check("L", &c, SKIN, 3e-4);
    let e = erode(&c, SKIN, &cfg()).expect("L");
    // every vertex of the arc is SKIN from the reflex corner (0.08, 0.08)
    let arc: Vec<_> = e.contours[0]
        .iter()
        .filter(|v| {
            matches!(
                v.normal,
                alice_physics_core::polygon_offset::OffsetNormal::Arc { .. }
            )
        })
        .collect();
    assert!(arc.len() >= 3);
    for v in arc {
        let r = ((v.p[0] - 0.08).powi(2) + (v.p[1] - 0.08).powi(2)).sqrt();
        assert!((r - SKIN).abs() < 1e-12, "arc point at {r}");
    }
}

#[test]
fn a_square_hole_grows_by_the_offset() {
    let c = vec![
        vec![[0.0, 0.0], [0.3, 0.0], [0.3, 0.3], [0.0, 0.3]],
        vec![[0.1, 0.1], [0.1, 0.2], [0.2, 0.2], [0.2, 0.1]],
    ];
    check("holed square", &c, SKIN, 4e-4);
}

/// The logo letters: closed form against the sampled distance for every letter,
/// and the eroded areas pinned (m², SKIN = 12 mm) so a change in the outlines
/// or the formula shows here.
#[test]
fn the_logo_letters_erode_to_their_closed_form_areas() {
    let mut areas = Vec::new();
    for (ch, c) in logo() {
        let a = check(&ch.to_string(), &c, SKIN, 4e-4);
        areas.push((ch, a));
    }
    let pinned = [
        ('E', 0.124_204_010_657_883_13),
        ('x', 0.079_828_058_133_363_58),
        ('t', 0.069_320_358_387_653_09),
        ('o', 0.103_705_878_329_492_37),
        ('r', 0.052_495_423_953_714_45),
        ('i', 0.045_645_660_000_000_04),
        ('a', 0.105_168_998_567_507_36),
    ];
    assert_eq!(areas.len(), pinned.len());
    for ((ch, a), (pc, pa)) in areas.iter().zip(pinned) {
        assert_eq!(*ch, pc);
        assert!(
            (a - pa).abs() <= 1e-12 * pa,
            "{ch}: eroded area {a:.17} vs pinned {pa:.17}"
        );
    }
}

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
    let line = vec![vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]];
    assert!(matches!(
        erode(&line, SKIN, &cfg()),
        Err(OffsetError::DegenerateContour { contour: 0 })
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
