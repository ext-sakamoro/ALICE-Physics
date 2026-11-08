//! Oracles for [`extruded_tet_mesh::mesh_rounded_slab`] against the law it meshes,
//! `length(max(w, 0)) + min(max(w.x, w.y), 0) - skin` with `w = (d + skin, |z| - (h - skin))`
//! and `d` the outline's exact signed distance, evaluated here independently in f64.
//!
//! # Volume, measured independently
//!
//! A slice at `|z| ≤ h - skin` is the outline itself (its shoelace area); a slice in the round,
//! `|z| = h - skin + s sin φ`, is the outline eroded by `s (1 - cos φ)`, whose area the scanline
//! oracle (`support/eroded_area.rs`) measures for any outline, including letters whose bends
//! erode away. The round's volume is integrated over `φ` by Gauss-Legendre.
//!
//! The mesh inscribes the round with `k` chords per quarter: per unit length of boundary the
//! cross-section misses `k (s²/2)(α - sin α)`, `α = π/(2k)`, on each side. The eroded outline's
//! arcs are tangent polygons (outside by at most the sagitta `sag`) and vertices within half of
//! it are dropped, so every slice's boundary is within 1.5 sag of the law's, over the full
//! height and at most the longer of the outline's and the eroded outline's perimeters. The bound
//! is the sum, with the oracle's own error and the f32 storage of the coordinates. That slice
//! term is larger than the rounds' own volume, so the volume check does not see whether the
//! rounds are there; the vertex check does (a missing round leaves its corner `s (√2 - 1)` off
//! the law, far past 1.5 sag).
//!
//! # Bottom face
//!
//! The boundary triangles on `z = -h` are the outline eroded by the skin: their area is the
//! scanline area within `1.5 sag x L` and the storage terms.
#![cfg(feature = "std")]
#![allow(
    clippy::disallowed_methods,
    reason = "analytic reference: in f64, the law, areas and angles are computed from the input, never by the code under test"
)]

use alice_physics_core::extruded_tet_mesh::{mesh_rounded_slab, RoundedSlab, ROUND_SEGMENTS};
use alice_physics_core::sdf_fem_mesh::SdfTetMesh;

#[path = "support/eroded_area.rs"]
mod eroded_area;
use eroded_area::eroded_area;
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

/// `f32` storage of `z = ±0.08`: half an ulp at 0.08.
const Z_STORAGE: f64 = 0.5 * 7.450_580_596_923_828e-9;

/// `∫₀^s A(s - sqrt(s² - t²)) dt` by `n`-node Gauss-Legendre in `φ` (t = s sin φ),
/// with A from the scanline oracle; and the summed oracle error.
fn round_volume(c: &[Vec<[f64; 2]>], n: usize) -> (f64, f64) {
    let (x, w): (&[f64], &[f64]) = match n {
        4 => (
            &[
                -0.861_136_311_594_052_6,
                -0.339_981_043_584_856_3,
                0.339_981_043_584_856_3,
                0.861_136_311_594_052_6,
            ],
            &[
                0.347_854_845_137_453_9,
                0.652_145_154_862_546_1,
                0.652_145_154_862_546_1,
                0.347_854_845_137_453_9,
            ],
        ),
        _ => (
            &[
                -0.960_289_856_497_536_3,
                -0.796_666_477_413_626_7,
                -0.525_532_409_916_329,
                -0.183_434_642_495_649_8,
                0.183_434_642_495_649_8,
                0.525_532_409_916_329,
                0.796_666_477_413_626_7,
                0.960_289_856_497_536_3,
            ],
            &[
                0.101_228_536_290_376_3,
                0.222_381_034_453_374_5,
                0.313_706_645_877_887_3,
                0.362_683_783_378_362,
                0.362_683_783_378_362,
                0.313_706_645_877_887_3,
                0.222_381_034_453_374_5,
                0.101_228_536_290_376_3,
            ],
        ),
    };
    let half = PI / 4.0;
    let (mut v, mut e) = (0.0, 0.0);
    for (xi, wi) in x.iter().zip(w) {
        let phi = half * (xi + 1.0);
        let delta = S * (1.0 - phi.cos());
        let (a, err) = eroded_area(c, delta, 1000);
        let dt = S * phi.cos() * half * wi;
        v += a * dt;
        e += err * dt;
    }
    (v, e)
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
    // every vertex inside or on the law; every boundary vertex on it. The
    // eroded outline's arcs are tangent polygons (outside the circle by at
    // most `sag`, the round's half-chord sagitta `s (1 - cos 22.5°)`), and the
    // mesher drops vertices within half of it: 1.5 sag. Then the triangulation
    // grid (half a diagonal) and the f32 storage of the coordinates (half an
    // ulp of 2.0 per axis, √3 of it in all).
    let sag = S * (1.0 - (PI / (4.0 * ROUND_SEGMENTS as f64)).cos());
    let storage = 3.0f64.sqrt() * 0.5 * 2.0f64.powi(-22) + std::f64::consts::FRAC_1_SQRT_2 * 1e-6;
    let tol = 1.5 * sag + storage;
    let mut on = 0.0f64;
    for f in &faces {
        for &v in f {
            on = on.max(law(c, m.vertices[v as usize]).abs());
        }
    }
    let outside = m
        .vertices
        .iter()
        .map(|&v| law(c, v))
        .fold(f64::MIN, f64::max);
    assert!(
        on <= tol,
        "{name}: a boundary vertex is {on:.3e} off the law (bound {tol:.2e})"
    );
    assert!(
        outside <= tol,
        "{name}: a vertex is {outside:.3e} outside the law (bound {tol:.2e})"
    );

    // volume of the law, measured independently: slices at |z| <= h - s are
    // the outline (shoelace), slices in the round are the outline eroded by
    // δ = s (1 - cos φ) at |z| = h - s + s sin φ, measured by the scanline
    // oracle and integrated over φ by Gauss-Legendre (8 nodes; the 4-node
    // value bounds the quadrature error)
    let a0: f64 = oriented(c).iter().map(|r| shoelace(r)).sum();
    let (round8, e8) = round_volume(c, 8);
    let (round4, _) = round_volume(c, 4);
    let truth = 2.0 * (H - S) * a0 + 2.0 * round8;
    let oracle = 2.0 * (round8 - round4).abs() + 2.0 * e8;
    // the mesh: the round inscribed by two chords per quarter (per unit of
    // boundary the cross-section misses k (s²/2)(α - sin α) per side), and every
    // slice's boundary within 1.5 sag of the law's over the full height
    let l_in = perimeter(c);
    let eroded = alice_physics_core::polygon_offset::erode(
        c,
        S,
        &alice_physics_core::polygon_offset::OffsetConfig { sagitta: sag },
    )
    .expect("erode");
    let l_max = l_in.max(perimeter(&eroded.points()));
    let k = ROUND_SEGMENTS as f64;
    let alpha = PI / (2.0 * k);
    let round_chords = 2.0 * l_max * k * (S * S / 2.0) * (alpha - alpha.sin());
    let slices = 1.5 * sag * l_max * 2.0 * H;
    let vol: f64 = m.tets.iter().map(|t| tet_vol6(&m, t.vertices) / 6.0).sum();
    let bound = round_chords + slices + oracle + storage * l_max * 2.0 * H;
    assert!(
        (vol - truth).abs() <= bound,
        "{name}: volume {vol:.6e} vs law {truth:.6e} (bound {bound:.2e})"
    );

    // bottom face = the outline eroded by s (scanline truth)
    let bottom: f64 = faces
        .iter()
        .filter(|f| {
            f.iter()
                .all(|&v| (f64::from(m.vertices[v as usize][2]) + H).abs() <= Z_STORAGE)
        })
        .map(|f| {
            let p = f.map(|v| m.vertices[v as usize].map(f64::from));
            0.5 * ((p[1][0] - p[0][0]) * (p[2][1] - p[0][1])
                - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1]))
                .abs()
        })
        .sum();
    let (true_bottom, eb) = eroded_area(c, S, 1500);
    let bb = 1.5 * sag * l_max + storage * l_max + eb;
    assert!(
        (bottom - true_bottom).abs() <= bb,
        "{name}: bottom {bottom:.6e} vs eroded outline {true_bottom:.6e} (bound {bb:.2e})"
    );

    // the side wall stands on the outline: the mesh reaches the outline's
    // lowest point (a letter on -y touches the ground there) within the same
    // 1.5 sag
    let lowest = m
        .vertices
        .iter()
        .map(|v| f64::from(v[1]))
        .fold(f64::MAX, f64::min);
    let outline_lowest = c.iter().flatten().map(|p| p[1]).fold(f64::MAX, f64::min);
    assert!(
        (lowest - outline_lowest).abs() <= tol,
        "{name}: lowest y {lowest:.6} vs outline {outline_lowest:.6}"
    );
    eprintln!("{name}: {} tets, volume {vol:.6e} law {truth:.6e}, bottom {bottom:.6e} eroded {true_bottom:.6e}, lowest gap {:.2e}", m.tets.len(), lowest - outline_lowest);
    let d = min_dihedral_deg(&m);
    let lowest = m
        .vertices
        .iter()
        .map(|v| f64::from(v[1]))
        .fold(f64::MAX, f64::min);
    let outline_lowest = c.iter().flatten().map(|p| p[1]).fold(f64::MAX, f64::min);
    // the side wall is the outline itself, so the mesh reaches its extremes:
    // a letter standing on -y touches the ground where the outline does
    assert!(
        (lowest - outline_lowest).abs() <= 1.5 * S * (1.0 - (PI / 8.0).cos()) * 0.5 + 1e-6,
        "{name}: lowest y {lowest:.6} vs outline {outline_lowest:.6}"
    );
    eprintln!("{name}: min dihedral {d:.2}");
    // the floor is the gate of tests/mesh_quality.rs (8°), not derived here
    assert!(d >= 8.0, "{name}: min dihedral {d:.2}");
    // ratchet: the worst angle measured when this mesher landed, so a
    // regression shows before it reaches the floor
    let measured = match name {
        "square" => 19.11,
        "L" | "E" | "t" | "a" => 9.29,
        "x" => 11.03,
        "o" => 19.08,
        "r" => 9.47,
        "i" => 18.45,
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

fn letter(ch: char) {
    let (_, c) = logo().into_iter().find(|(x, _)| *x == ch).expect("letter");
    check(&ch.to_string(), &c);
}

#[test]
fn letter_e_slab_meets_the_law() {
    letter('E');
}
#[test]
fn letter_x_slab_meets_the_law() {
    letter('x');
}
#[test]
fn letter_t_slab_meets_the_law() {
    letter('t');
}
#[test]
fn letter_o_slab_meets_the_law() {
    letter('o');
}
/// `r` loses edges at the skin (a bend tighter than 12 mm): its rim there is
/// the eroded corner's miter sweep, held to the same bounds.
#[test]
fn letter_r_slab_meets_the_law() {
    letter('r');
}
#[test]
fn letter_i_slab_meets_the_law() {
    letter('i');
}
/// `a` loses edges at the skin, like `r`; its bowl's lowest point is checked
/// against the outline's.
#[test]
fn letter_a_slab_meets_the_law() {
    letter('a');
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

/// A bend of radius under the skin closes in the erosion, and the corner it
/// leaves is not a miter of the outline: refused, never a mesh with a vertex
/// far off the shape.
#[test]
fn a_bend_tighter_than_the_skin_is_refused_or_meshed_on_the_law() {
    // a 20-degree wedge 0.3 m long whose tip is cut by three 2 mm edges: the
    // erosion drops them and the two long sides meet at a 20-degree corner,
    // whose miter (5.8 skins) is past MAX_MITER
    let half = 10.0f64.to_radians();
    let tip = [0.3, 0.0];
    let side = |t: f64, sgn: f64| [tip[0] - t * half.cos(), sgn * t * half.sin()];
    let spike = vec![vec![
        side(0.3, -1.0),
        side(0.004, -1.0),
        [tip[0] - 0.001, -0.0005],
        [tip[0] - 0.001, 0.0005],
        side(0.004, 1.0),
        side(0.3, 1.0),
    ]];
    match mesh_rounded_slab(&spike, &cfg()) {
        Err(alice_physics_core::extruded_tet_mesh::ExtrudeError::SharpCorner { .. }) => {}
        Err(e) => panic!("{e:?}"),
        Ok(m) => {
            let worst = m
                .vertices
                .iter()
                .map(|&v| law(&spike, v))
                .fold(f64::MIN, f64::max);
            assert!(worst <= 1e-3, "a vertex {worst:.3e} outside the law");
        }
    }
}

#[test]
fn settings_out_of_range_are_refused() {
    use alice_physics_core::extruded_tet_mesh::ExtrudeError;
    let sq = vec![vec![[0.0, 0.0], [0.2, 0.0], [0.2, 0.2], [0.0, 0.2]]];
    let bad = |f: &dyn Fn(&mut RoundedSlab)| {
        let mut c = cfg();
        f(&mut c);
        mesh_rounded_slab(&sq, &c)
    };
    assert!(matches!(
        bad(&|c| c.half_depth = f64::INFINITY),
        Err(ExtrudeError::BadConfig)
    ));
    assert!(matches!(
        bad(&|c| c.skin = f64::NAN),
        Err(ExtrudeError::BadConfig)
    ));
    assert!(matches!(
        bad(&|c| c.half_depth = 0.01),
        Err(ExtrudeError::BadConfig)
    ));
    assert!(matches!(
        bad(&|c| c.layer = 1e-9),
        Err(ExtrudeError::TooLarge)
    ));
    // the step is at least a convex corner's miter reach next to one, so take
    // the inner floor of a U, between two reflex corners
    let u = vec![vec![
        [0.0, 0.0],
        [0.3, 0.0],
        [0.3, 0.3],
        [0.2, 0.3],
        [0.2, 0.1],
        [0.1, 0.1],
        [0.1, 0.3],
        [0.0, 0.3],
    ]];
    let mut tiny = cfg();
    tiny.boundary_step = 1e-9;
    assert!(matches!(
        mesh_rounded_slab(&u, &tiny),
        Err(ExtrudeError::TooLarge)
    ));
}
