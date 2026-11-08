//! Oracles for [`extruded_tet_mesh::mesh_rounded_slab`] against the law it meshes,
//! `length(max(w, 0)) + min(max(w.x, w.y), 0) - skin` with `w = (d + skin, |z| - (h - skin))`
//! and `d` the outline's exact signed distance, evaluated here independently in f64.
//!
//! # Volume in closed form
//!
//! A slice at `|z| ≤ h - skin` is the outline itself (area `A`); a slice in the round,
//! `|z| = h - skin + t`, is the outline eroded by `δ(t) = skin - sqrt(skin² - t²)`, whose area is
//! `A - L δ + κ δ²` with `κ = Σ_convex cot(θ/2) - Σ_reflex (θ - π)/2` (see
//! `tests/analytic_polygon_offset.rs`). With `∫₀^s δ dt = s²(1 - π/4)` and
//! `∫₀^s δ² dt = s³(5/3 - π/2)`:
//!
//! `V = 2 h A - 2 L s² (1 - π/4) + 2 κ s³ (5/3 - π/2)`.
//!
//! The mesh inscribes the round with `k` chords per quarter: per unit length of boundary the
//! cross-section misses `k (s²/2)(α - sin α)`, `α = π/(2k)`, on each side. The eroded outline is
//! chords too (its reflex arcs, and vertices dropped within the same sagitta `s (1 - cos(α/2))`),
//! which moves the boundary by at most twice that sagitta (the two add) along at most the perimeter `L`, over the full
//! height. The bound is the sum.
//!
//! # Bottom face
//!
//! The boundary triangles on `z = -h` are the eroded outline: their area is `A(skin)` within
//! `2 sagitta x L` and the triangulation grid.
#![cfg(feature = "std")]
#![allow(
    clippy::disallowed_methods,
    reason = "analytic reference in f64: the law, areas and angles are computed from the input, never by the code under test"
)]

use alice_physics_core::extruded_tet_mesh::{mesh_rounded_slab, RoundedSlab, ROUND_SEGMENTS};
use alice_physics_core::sdf_fem_mesh::SdfTetMesh;
use std::f64::consts::PI;

const S: f64 = 0.012;
const H: f64 = 0.08;

fn cfg() -> RoundedSlab {
    RoundedSlab {
        skin: S,
        half_depth: H,
        boundary_step: 0.012,
        max_edge: 0.03,
        layer: 0.015,
        grid: 1.0e-6,
    }
}

fn shoelace(c: &[[f64; 2]]) -> f64 {
    let n = c.len();
    0.5 * (0..n)
        .map(|i| c[i][0] * c[(i + 1) % n][1] - c[(i + 1) % n][0] * c[i][1])
        .sum::<f64>()
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

/// (A, L, κ, total reflex turn)
fn outline_terms(c: &[Vec<[f64; 2]>]) -> (f64, f64, f64, f64) {
    let (mut a, mut l, mut kappa, mut reflex) = (0.0, 0.0, 0.0, 0.0);
    for r in &oriented(c) {
        a += shoelace(r);
        let n = r.len();
        for i in 0..n {
            let (p, v, q) = (r[(i + n - 1) % n], r[i], r[(i + 1) % n]);
            let e = [v[0] - p[0], v[1] - p[1]];
            let f = [q[0] - v[0], q[1] - v[1]];
            l += (e[0] * e[0] + e[1] * e[1]).sqrt();
            let turn = (e[0] * f[1] - e[1] * f[0]).atan2(e[0] * f[0] + e[1] * f[1]);
            let theta = PI - turn;
            if turn > 0.0 {
                kappa += 1.0 / (theta / 2.0).tan();
            } else if turn < 0.0 {
                kappa -= (theta - PI) / 2.0;
                reflex += theta - PI;
            }
        }
    }
    (a, l, kappa, reflex)
}

fn seg_dist(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let e = [b[0] - a[0], b[1] - a[1]];
    let l = e[0] * e[0] + e[1] * e[1];
    let s = (((p[0] - a[0]) * e[0] + (p[1] - a[1]) * e[1]) / l).clamp(0.0, 1.0);
    ((p[0] - a[0] - s * e[0]).powi(2) + (p[1] - a[1] - s * e[1]).powi(2)).sqrt()
}

fn d_poly(c: &[Vec<[f64; 2]>], p: [f64; 2]) -> f64 {
    let mut inside = false;
    let mut dist = f64::INFINITY;
    for r in c {
        for i in 0..r.len() {
            let (a, b) = (r[i], r[(i + 1) % r.len()]);
            dist = dist.min(seg_dist(p, a, b));
            if (a[1] > p[1]) != (b[1] > p[1])
                && p[0] < a[0] + (p[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            {
                inside = !inside;
            }
        }
    }
    if inside {
        -dist
    } else {
        dist
    }
}

fn law(c: &[Vec<[f64; 2]>], p: [f32; 3]) -> f64 {
    let w = [
        d_poly(c, [f64::from(p[0]), f64::from(p[1])]) + S,
        f64::from(p[2]).abs() - (H - S),
    ];
    let o = [w[0].max(0.0), w[1].max(0.0)];
    (o[0] * o[0] + o[1] * o[1]).sqrt() + w[0].max(w[1]).min(0.0) - S
}

fn tet_vol6(m: &SdfTetMesh, t: [u32; 4]) -> f64 {
    let p = t.map(|i| m.vertices[i as usize].map(f64::from));
    let e = |i: usize| [0, 1, 2].map(|k| p[i][k] - p[0][k]);
    let (a, b, c) = (e(1), e(2), e(3));
    a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
}

fn min_dihedral_deg(m: &SdfTetMesh) -> f64 {
    let mut worst = 180.0f64;
    for t in &m.tets {
        let p = t.vertices.map(|i| m.vertices[i as usize].map(f64::from));
        let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        let cr = |a: [f64; 3], b: [f64; 3]| {
            [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]
        };
        let nrm = |a: [f64; 3]| (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
        for (i, j) in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] {
            let o: Vec<usize> = (0..4).filter(|&k| k != i && k != j).collect();
            let e = sub(p[j], p[i]);
            let n1 = cr(e, sub(p[o[0]], p[i]));
            let n2 = cr(e, sub(p[o[1]], p[i]));
            let c = (n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2]) / (nrm(n1) * nrm(n2));
            worst = worst.min(c.clamp(-1.0, 1.0).acos().to_degrees());
        }
    }
    worst
}

fn check(name: &str, c: &[Vec<[f64; 2]>]) -> SdfTetMesh {
    let cf = cfg();
    let m = mesh_rounded_slab(c, &cf).unwrap_or_else(|e| panic!("{name}: {e:?}"));
    // conforming, manifold boundary, positively wound
    let faces = m
        .boundary_faces()
        .unwrap_or_else(|e| panic!("{name}: {e:?}"));
    assert!(
        m.tets.iter().all(|t| tet_vol6(&m, t.vertices) > 0.0),
        "{name}: an element is inverted or flat"
    );
    // every vertex inside or on the law; every boundary vertex on it
    let mut on = 0.0f64;
    for f in &faces {
        for &v in f {
            on = on.max(law(c, m.vertices[v as usize]).abs());
        }
    }
    let inside = m
        .vertices
        .iter()
        .map(|&v| law(c, v))
        .fold(f64::MIN, f64::max);
    // the eroded outline is chords of its arcs (sagitta `s (1 - cos(α/2))`,
    // α = 45°), and the mesher drops vertices within that sagitta of the chord
    // of their neighbours: the two add, so a boundary point is off the law by at
    // most twice the sagitta, plus the grid
    let sagitta = 2.0 * S * (1.0 - (PI / (4.0 * ROUND_SEGMENTS as f64)).cos());
    assert!(
        on <= sagitta + 2e-6,
        "{name}: a boundary vertex is {on:.3e} off the law (sagitta {sagitta:.2e})"
    );
    assert!(
        inside <= sagitta + 2e-6,
        "{name}: a vertex is {inside:.3e} outside the law"
    );
    // volume
    let (a, l, kappa, reflex) = outline_terms(c);
    let truth = 2.0 * H * a - 2.0 * l * S * S * (1.0 - PI / 4.0)
        + 2.0 * kappa * S.powi(3) * (5.0 / 3.0 - PI / 2.0);
    let k = ROUND_SEGMENTS as f64;
    let alpha = PI / (2.0 * k);
    let round_chords = 2.0 * l * k * (S * S / 2.0) * (alpha - alpha.sin());
    // the eroded outline's chords (reflex arcs, and the vertices dropped within
    // the sagitta) move its boundary by at most the sagitta, over at most the
    // perimeter and the whole height
    let sag = 2.0 * S * (1.0 - (alpha / 2.0).cos());
    let arc_chords = sag * l * 2.0 * H;
    let _ = reflex;
    let grid = l * 2.0 * H * 1.0e-6;
    let vol: f64 = m.tets.iter().map(|t| tet_vol6(&m, t.vertices) / 6.0).sum();
    let bound = round_chords + arc_chords + grid;
    assert!(
        (vol - truth).abs() <= bound,
        "{name}: volume {vol:.6e} vs law {truth:.6e} (bound {bound:.2e})"
    );
    assert!(
        vol <= truth + arc_chords + grid,
        "{name}: inscribed mesh larger than the law"
    );
    // bottom face = eroded outline
    let bottom: f64 = faces
        .iter()
        .filter(|f| {
            f.iter()
                .all(|&v| (f64::from(m.vertices[v as usize][2]) + H).abs() < 1e-7)
        })
        .map(|f| {
            let p = f.map(|v| m.vertices[v as usize].map(f64::from));
            0.5 * ((p[1][0] - p[0][0]) * (p[2][1] - p[0][1])
                - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1]))
                .abs()
        })
        .sum();
    let eroded = a - l * S + kappa * S * S;
    let bb = sag * l + l * 1.0e-6;
    assert!(
        (bottom - eroded).abs() <= bb,
        "{name}: bottom {bottom:.6e} vs eroded outline {eroded:.6e} (bound {bb:.2e})"
    );
    let d = min_dihedral_deg(&m);
    eprintln!("{name}: {} tets, volume {vol:.6e} law {truth:.6e}, bottom {bottom:.6e} eroded {eroded:.6e}, min dihedral {d:.2}", m.tets.len());
    assert!(d >= 8.0, "{name}: min dihedral {d:.2}");
    // ratchet: the worst angle measured when this mesher landed, so a
    // regression shows before it reaches the floor
    let measured = match name {
        "square" => 19.11,
        "L" | "E" => 14.64,
        "x" => 10.32,
        "t" => 12.88,
        "o" => 20.65,
        "r" => 8.88,
        "i" => 18.45,
        "a" => 10.86,
        _ => 8.0,
    };
    assert!(
        d >= measured - 0.01,
        "{name}: min dihedral {d:.2} under its landed value {measured}"
    );
    m
}

fn logo() -> Vec<(char, Vec<Vec<[f64; 2]>>)> {
    include_str!("data/logo_outlines.txt")
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            let (ch, rest) = l.split_once(' ').expect("char");
            let c = rest
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
            (ch.chars().next().expect("c"), c)
        })
        .collect()
}

#[test]
fn a_square_slab_meets_the_law() {
    check(
        "square",
        &[vec![[0.0, 0.0], [0.2, 0.0], [0.2, 0.2], [0.0, 0.2]]],
    );
}

#[test]
fn an_l_slab_meets_the_law() {
    check(
        "L",
        &[vec![
            [0.0, 0.0],
            [0.2, 0.0],
            [0.2, 0.08],
            [0.08, 0.08],
            [0.08, 0.2],
            [0.0, 0.2],
        ]],
    );
}

#[test]
fn the_logo_letters_meet_the_law() {
    for (ch, c) in logo() {
        check(&ch.to_string(), &c);
    }
}

#[test]
fn identical_runs_give_identical_meshes() {
    let c = vec![vec![
        [0.0, 0.0],
        [0.2, 0.0],
        [0.2, 0.08],
        [0.08, 0.08],
        [0.08, 0.2],
        [0.0, 0.2],
    ]];
    assert_eq!(
        mesh_rounded_slab(&c, &cfg()).expect("a"),
        mesh_rounded_slab(&c, &cfg()).expect("b")
    );
}
