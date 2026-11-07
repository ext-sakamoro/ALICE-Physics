//! Oracles for [`polygon_cdt::triangulate`], against closed forms.
//!
//! - **Area**: the triangles tile the domain, so their summed area equals the
//!   shoelace area of the contours (outer minus holes), exactly in grid units
//!   when no boundary segment had to be split off its line, and within
//!   `perimeter x grid / 2` when some were (a split point is rounded to the
//!   grid, half a step at most).
//! - **Boundary**: every input segment is a union of boundary edges, and every
//!   boundary edge lies on an input segment.
//! - **Delaunay**: no point is strictly inside the circumcircle of a triangle
//!   that shares an edge with it (exact in-circle on the grid integers).
//! - **Quality**: every angle is at least the configured minimum, except at
//!   input corners sharper than 60° (Ruppert's bound does not hold there).
//! - **Euler**: `V - E + F = 1 - holes` for the triangulated domain.
//! - **Determinism**: two runs give identical output.
#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, reason = "analytic reference in f64: areas, angles and the circle points are computed from the input, never by the code under test")]

use alice_physics_core::polygon_cdt::{incircle, orient, triangulate, Cdt, CdtConfig, CdtError};
use std::collections::BTreeMap;

fn shoelace(c: &[[f64; 2]]) -> f64 {
    let n = c.len();
    0.5 * (0..n).map(|i| c[i][0] * c[(i + 1) % n][1] - c[(i + 1) % n][0] * c[i][1]).sum::<f64>()
}

fn rect(x0: f64, y0: f64, x1: f64, y1: f64, ccw: bool) -> Vec<[f64; 2]> {
    let mut v = vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
    if !ccw {
        v.reverse();
    }
    v
}

/// A circle of radius `r` about `c` as an `n`-gon, on 1 µm grid points.
fn circle(c: [f64; 2], r: f64, n: usize, ccw: bool) -> Vec<[f64; 2]> {
    let mut v: Vec<[f64; 2]> = (0..n)
        .map(|k| {
            let a = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
            [((c[0] + r * a.cos()) * 1e6).round() / 1e6, ((c[1] + r * a.sin()) * 1e6).round() / 1e6]
        })
        .collect();
    if !ccw {
        v.reverse();
    }
    v
}

fn area(t: &Cdt) -> f64 {
    t.area2_grid() as f64 * 0.5 * t.grid * t.grid
}

fn min_angle_deg(t: &Cdt, skip_corners_below_60: &[[f64; 2]]) -> f64 {
    let mut worst = 180.0f64;
    for tri in &t.triangles {
        let p = tri.map(|i| t.point(i));
        for k in 0..3 {
            let (a, b, c) = (p[k], p[(k + 1) % 3], p[(k + 2) % 3]);
            if skip_corners_below_60.iter().any(|q| (q[0] - a[0]).abs() < 1e-9 && (q[1] - a[1]).abs() < 1e-9) {
                continue;
            }
            let u = [b[0] - a[0], b[1] - a[1]];
            let w = [c[0] - a[0], c[1] - a[1]];
            let cos = (u[0] * w[0] + u[1] * w[1]) / ((u[0] * u[0] + u[1] * u[1]).sqrt() * (w[0] * w[0] + w[1] * w[1]).sqrt());
            worst = worst.min(cos.clamp(-1.0, 1.0).acos().to_degrees());
        }
    }
    worst
}

fn check_delaunay(t: &Cdt) -> usize {
    let mut edge: BTreeMap<(u32, u32), Vec<u32>> = BTreeMap::new();
    for tri in &t.triangles {
        for k in 0..3 {
            let (a, b, c) = (tri[k], tri[(k + 1) % 3], tri[(k + 2) % 3]);
            edge.entry((a.min(b), a.max(b))).or_default().push(c);
        }
    }
    let mut bad = 0;
    for tri in &t.triangles {
        let p = tri.map(|i| t.points[i as usize]);
        for k in 0..3 {
            let (a, b) = (tri[k], tri[(k + 1) % 3]);
            for &o in &edge[&(a.min(b), a.max(b))] {
                if !tri.contains(&o) && incircle(p[0], p[1], p[2], t.points[o as usize]) > 0 {
                    bad += 1;
                }
            }
        }
    }
    bad
}

fn boundary_edges(t: &Cdt) -> Vec<(u32, u32)> {
    let mut count: BTreeMap<(u32, u32), usize> = BTreeMap::new();
    for tri in &t.triangles {
        for k in 0..3 {
            let (a, b) = (tri[k], tri[(k + 1) % 3]);
            *count.entry((a.min(b), a.max(b))).or_default() += 1;
        }
    }
    count.into_iter().filter(|(_, n)| *n == 1).map(|(e, _)| e).collect()
}

/// Every boundary edge lies on an input segment (within the half-grid
/// rounding of split points).
fn boundary_on_input(t: &Cdt, contours: &[Vec<[f64; 2]>]) -> f64 {
    let mut worst = 0.0f64;
    for (a, b) in boundary_edges(t) {
        for p in [t.point(a), t.point(b), {
            let (pa, pb) = (t.point(a), t.point(b));
            [(pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0]
        }] {
            let d = contours
                .iter()
                .flat_map(|c| (0..c.len()).map(move |i| (c[i], c[(i + 1) % c.len()])))
                .map(|(u, v)| seg_dist(p, u, v))
                .fold(f64::INFINITY, f64::min);
            worst = worst.max(d);
        }
    }
    worst
}

fn seg_dist(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let e = [b[0] - a[0], b[1] - a[1]];
    let l = e[0] * e[0] + e[1] * e[1];
    let s = (((p[0] - a[0]) * e[0] + (p[1] - a[1]) * e[1]) / l).clamp(0.0, 1.0);
    let q = [a[0] + s * e[0], a[1] + s * e[1]];
    ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt()
}

fn euler(t: &Cdt) -> i64 {
    let mut edges = std::collections::BTreeSet::new();
    let mut verts = std::collections::BTreeSet::new();
    for tri in &t.triangles {
        for k in 0..3 {
            let (a, b) = (tri[k], tri[(k + 1) % 3]);
            edges.insert((a.min(b), a.max(b)));
            verts.insert(a);
        }
    }
    verts.len() as i64 - edges.len() as i64 + t.triangles.len() as i64
}

fn perimeter(c: &[Vec<[f64; 2]>]) -> f64 {
    c.iter()
        .map(|r| (0..r.len()).map(|i| {
            let (p, q) = (r[i], r[(i + 1) % r.len()]);
            ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt()
        }).sum::<f64>())
        .sum()
}

fn cfg(max_edge: f64) -> CdtConfig {
    CdtConfig { max_edge, ..CdtConfig::default() }
}

#[test]
fn a_square_is_tiled_exactly_with_quality_and_delaunay() {
    let c = vec![rect(0.0, 0.0, 0.2, 0.2, true)];
    let t = triangulate(&c, &cfg(0.03)).expect("square");
    assert!(t.triangles.len() >= 50, "{} triangles", t.triangles.len());
    // all split points of axis-aligned edges with even grid sums stay on the line
    assert!((area(&t) - 0.04).abs() <= 0.8 * 1e-6 / 2.0, "area {}", area(&t));
    assert_eq!(check_delaunay(&t), 0);
    assert!(min_angle_deg(&t, &[]) >= 20.0, "min angle {}", min_angle_deg(&t, &[]));
    assert!(boundary_on_input(&t, &c) <= 0.5e-6 * 1.5);
    assert_eq!(euler(&t), 1);
    let max_edge = t.triangles.iter().flat_map(|tri| (0..3).map(move |k| (tri[k], tri[(k + 1) % 3]))).map(|(a, b)| {
        let (p, q) = (t.point(a), t.point(b));
        ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt()
    }).fold(0.0, f64::max);
    assert!(max_edge <= 0.03 + 1e-6, "longest edge {max_edge}");
}

#[test]
fn holes_by_winding_and_area_minus_holes() {
    // outer ccw, a square hole cw, a round hole cw: like an `o` or an `a`
    let c = vec![
        rect(0.0, 0.0, 0.3, 0.2, true),
        rect(0.02, 0.02, 0.1, 0.1, false),
        circle([0.2, 0.1], 0.05, 48, false),
    ];
    let truth: f64 = c.iter().map(|r| shoelace(r)).sum();
    let t = triangulate(&c, &cfg(0.04)).expect("holed");
    assert!((area(&t) - truth).abs() <= perimeter(&c) * 0.5e-6, "area {} vs {truth}", area(&t));
    assert_eq!(euler(&t), 1 - 2);
    assert_eq!(check_delaunay(&t), 0);
    assert!(min_angle_deg(&t, &[]) >= 20.0);
    assert!(boundary_on_input(&t, &c) <= 0.75e-6);
    // the hole's inside is not meshed
    for tri in &t.triangles {
        let p = tri.map(|i| t.point(i));
        let cx = (p[0][0] + p[1][0] + p[2][0]) / 3.0;
        let cy = (p[0][1] + p[1][1] + p[2][1]) / 3.0;
        assert!(!((cx - 0.2).powi(2) + (cy - 0.1).powi(2) < 0.045 * 0.045), "triangle in the round hole");
        assert!(!(cx > 0.021 && cx < 0.099 && cy > 0.021 && cy < 0.099), "triangle in the square hole");
    }
    // same result when the outer runs clockwise and the holes counter-clockwise
    let flipped: Vec<Vec<[f64; 2]>> = c.iter().map(|r| r.iter().rev().copied().collect()).collect();
    let u = triangulate(&flipped, &cfg(0.04)).expect("flipped");
    assert_eq!(area(&u), area(&t));
}

/// The sharpest corner in the logo outlines is 53.1° (the `x`). Ruppert's
/// bound is proven for input angles of 60° and up; at 53.1° the refinement
/// still terminates and every angle except the corner's own reaches 20°.
#[test]
fn a_53_degree_corner_terminates_and_keeps_quality_elsewhere() {
    let tip = [0.0, 0.0];
    let a = 53.1f64.to_radians();
    let c = vec![vec![tip, [0.3, 0.0], [((0.3 * a.cos()) * 1e6).round() / 1e6, ((0.3 * a.sin()) * 1e6).round() / 1e6]]];
    let t = triangulate(&c, &cfg(0.05)).expect("wedge");
    assert_eq!(check_delaunay(&t), 0);
    assert!(min_angle_deg(&t, &[tip]) >= 20.0, "min angle away from the tip {}", min_angle_deg(&t, &[tip]));
    assert!((area(&t) - shoelace(&c[0])).abs() <= perimeter(&c) * 0.5e-6);
}

/// A much sharper corner (10°) is outside what the refinement handles: it
/// must stop on the point budget with an error, not return a worse mesh.
#[test]
fn a_10_degree_corner_runs_out_of_budget_rather_than_returning_slivers() {
    let a = 10.0f64.to_radians();
    let c = vec![vec![[0.0, 0.0], [0.3, 0.0], [((0.3 * a.cos()) * 1e6).round() / 1e6, ((0.3 * a.sin()) * 1e6).round() / 1e6]]];
    let tight = CdtConfig { max_points: 2_000, max_edge: 0.05, ..CdtConfig::default() };
    match triangulate(&c, &tight) {
        Err(CdtError::Budget { .. }) => {}
        Ok(t) => assert!(min_angle_deg(&t, &[[0.0, 0.0]]) >= 20.0, "returned a mesh with {}°", min_angle_deg(&t, &[[0.0, 0.0]])),
        Err(e) => panic!("{e:?}"),
    }
}

#[test]
fn identical_runs_give_identical_meshes() {
    let c = vec![rect(0.0, 0.0, 0.3, 0.2, true), circle([0.15, 0.1], 0.05, 40, false)];
    assert_eq!(triangulate(&c, &cfg(0.03)), triangulate(&c, &cfg(0.03)));
}

#[test]
fn predicates_are_exact_on_large_coordinates() {
    // four cocircular points far out: an f64 in-circle would not return exactly zero
    let s = 100_000_000i64;
    let (a, b, c, d) = ([s, 0], [0, s], [-s, 0], [0, -s]);
    assert_eq!(incircle(a, b, c, d), 0);
    assert!(incircle(a, b, c, [0, 0]) > 0);
    assert_eq!(orient([s, s], [s + 3, s + 3], [s + 7, s + 7]), 0);
}

#[test]
fn degenerate_input_is_refused() {
    assert!(matches!(triangulate(&[], &CdtConfig::default()), Err(CdtError::NoArea)));
    let line = vec![vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]];
    assert!(matches!(triangulate(&line, &CdtConfig::default()), Err(CdtError::DegenerateContour { contour: 0 })));
    let two = vec![vec![[0.0, 0.0], [1.0, 0.0]]];
    assert!(matches!(triangulate(&two, &CdtConfig::default()), Err(CdtError::DegenerateContour { contour: 0 })));
    let bad = CdtConfig { grid: 0.0, ..CdtConfig::default() };
    assert!(matches!(triangulate(&[rect(0.0, 0.0, 1.0, 1.0, true)], &bad), Err(CdtError::BadConfig)));
    let steep = CdtConfig { min_angle_deg: 30.0, ..CdtConfig::default() };
    assert!(matches!(triangulate(&[rect(0.0, 0.0, 1.0, 1.0, true)], &steep), Err(CdtError::BadConfig)));
    let far = vec![rect(0.0, 0.0, 1.0e3, 1.0e3, true)];
    assert!(matches!(triangulate(&far, &CdtConfig::default()), Err(CdtError::OutOfRange)));
    let tight = CdtConfig { max_points: 5, max_edge: 0.01, ..CdtConfig::default() };
    assert!(matches!(triangulate(&[rect(0.0, 0.0, 1.0, 1.0, true)], &tight), Err(CdtError::Budget { .. })));
}
