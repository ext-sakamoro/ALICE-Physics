//! Oracles for solid mass properties: mass, centre of mass and the inertia tensor
//! about the centre of mass, for the shapes `mass_properties` covers and for convex
//! hulls built from point clouds.
//!
//! # Two independent expectations
//!
//! Each shape is checked against its textbook closed form (written below, not
//! obtained from the function under test) and against a brute-force **quadrature**
//! of the solid (a midpoint grid with an `inside` predicate, which knows nothing
//! of the formulas). A convex hull has no closed form for an arbitrary point set,
//! so there the quadrature is the oracle, and a cube and an octahedron are the
//! exact cases.
//!
//! Author: Moroya Sakamoto

#![allow(clippy::disallowed_methods)]
// Matrix oracles index rows and columns on purpose: `t[a][b]` reads as the maths does.
#![allow(clippy::needless_range_loop)]

use alice_physics::convex_mesh_builder::build_hull_mesh;
use alice_physics::mass_properties::{
    box_mass_properties, capsule_mass_properties, convex_hull_mass_properties,
    cylinder_mass_properties, sphere_mass_properties, translate_inertia, MassProperties,
};
use alice_physics::math::{Fix128, Mat3Fix, Vec3Fix};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn arr(v: Vec3Fix) -> [f64; 3] {
    [v.x.to_f64(), v.y.to_f64(), v.z.to_f64()]
}

/// The tensor as `[row][col]`.
fn tensor(m: Mat3Fix) -> [[f64; 3]; 3] {
    let c = [arr(m.col0), arr(m.col1), arr(m.col2)];
    [
        [c[0][0], c[1][0], c[2][0]],
        [c[0][1], c[1][1], c[2][1]],
        [c[0][2], c[1][2], c[2][2]],
    ]
}

fn rel(got: f64, want: f64) -> f64 {
    (got - want).abs() / want.abs().max(1e-12)
}

fn assert_close(got: f64, want: f64, tol: f64, what: &str) {
    assert!(
        rel(got, want) <= tol || (got - want).abs() <= 1e-9,
        "{what}: got {got:.12e}, expected {want:.12e} (relative error {:.3e} > {tol:.1e})",
        rel(got, want)
    );
}

const PI: f64 = std::f64::consts::PI;
const EXACT: f64 = 1e-11;
const GRID: f64 = 0.03;

/// `(volume, centre of mass, inertia diagonal about the COM)` at unit density of the
/// solid `inside` over `lo..hi`, by a midpoint grid of `n³` cells; the full tensor
/// is returned so products of inertia are checked as well.
fn quadrature(
    lo: [f64; 3],
    hi: [f64; 3],
    n: usize,
    inside: impl Fn(f64, f64, f64) -> bool,
) -> (f64, [f64; 3], [[f64; 3]; 3]) {
    let step = [
        (hi[0] - lo[0]) / n as f64,
        (hi[1] - lo[1]) / n as f64,
        (hi[2] - lo[2]) / n as f64,
    ];
    let cell = step[0] * step[1] * step[2];
    let mut m = 0.0;
    let mut c = [0.0; 3];
    let mut s = [[0.0; 3]; 3];
    for i in 0..n {
        let x = lo[0] + (i as f64 + 0.5) * step[0];
        for j in 0..n {
            let y = lo[1] + (j as f64 + 0.5) * step[1];
            for k in 0..n {
                let z = lo[2] + (k as f64 + 0.5) * step[2];
                if inside(x, y, z) {
                    let p = [x, y, z];
                    m += cell;
                    for a in 0..3 {
                        c[a] += p[a] * cell;
                        for b in 0..3 {
                            s[a][b] += p[a] * p[b] * cell;
                        }
                    }
                }
            }
        }
    }
    let com = [c[0] / m, c[1] / m, c[2] / m];
    let mut i = [[0.0; 3]; 3];
    let tr = (s[0][0] - m * com[0] * com[0])
        + (s[1][1] - m * com[1] * com[1])
        + (s[2][2] - m * com[2] * com[2]);
    for a in 0..3 {
        for b in 0..3 {
            let second = s[a][b] - m * com[a] * com[b];
            i[a][b] = if a == b { tr - second } else { -second };
        }
    }
    (m, com, i)
}

fn check_against_quadrature(
    p: &MassProperties,
    vol: f64,
    com: [f64; 3],
    i: [[f64; 3]; 3],
    tol: f64,
    what: &str,
) {
    assert_close(p.mass.to_f64(), vol, tol, &format!("{what}: mass"));
    let got_com = arr(p.center_of_mass);
    for a in 0..3 {
        assert!(
            (got_com[a] - com[a]).abs() < 0.03,
            "{what}: centre of mass axis {a} is {} against the solid's {}",
            got_com[a],
            com[a]
        );
    }
    let t = tensor(p.inertia_tensor);
    for a in 0..3 {
        for b in 0..3 {
            if a == b {
                assert_close(t[a][b], i[a][b], tol, &format!("{what}: I[{a}][{a}]"));
            } else {
                // The scale of an off-diagonal term is the diagonal's.
                assert!(
                    (t[a][b] - i[a][b]).abs() < tol * i[a][a].max(i[b][b]),
                    "{what}: I[{a}][{b}] is {} against {}",
                    t[a][b],
                    i[a][b]
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The analytic shapes
// ---------------------------------------------------------------------------

#[test]
fn a_sphere_has_the_textbook_mass_and_inertia() {
    let (r, rho) = (1.5, 2.0);
    let p = sphere_mass_properties(fx(r), fx(rho));
    let m = rho * 4.0 / 3.0 * PI * r * r * r;
    assert_close(p.mass.to_f64(), m, EXACT, "sphere mass");
    let t = tensor(p.inertia_tensor);
    for a in 0..3 {
        assert_close(t[a][a], 0.4 * m * r * r, EXACT, "sphere I");
    }
    assert_eq!(arr(p.center_of_mass), [0.0; 3]);
}

#[test]
fn a_box_has_the_textbook_mass_and_inertia() {
    let (h, rho) = ([1.0, 2.0, 3.0], 1.5);
    let p = box_mass_properties(v3(h[0], h[1], h[2]), fx(rho));
    let m = rho * 8.0 * h[0] * h[1] * h[2];
    assert_close(p.mass.to_f64(), m, EXACT, "box mass");
    let t = tensor(p.inertia_tensor);
    // Half-extents: I_xx = m (b² + c²) / 3.
    assert_close(
        t[0][0],
        m * (h[1] * h[1] + h[2] * h[2]) / 3.0,
        EXACT,
        "box Ixx",
    );
    assert_close(
        t[1][1],
        m * (h[0] * h[0] + h[2] * h[2]) / 3.0,
        EXACT,
        "box Iyy",
    );
    assert_close(
        t[2][2],
        m * (h[0] * h[0] + h[1] * h[1]) / 3.0,
        EXACT,
        "box Izz",
    );
}

#[test]
fn a_cylinder_has_the_textbook_mass_and_inertia() {
    let (r, hh, rho) = (1.2, 2.0, 0.8);
    let p = cylinder_mass_properties(fx(r), fx(hh), fx(rho));
    let m = rho * PI * r * r * 2.0 * hh;
    assert_close(p.mass.to_f64(), m, EXACT, "cylinder mass");
    let t = tensor(p.inertia_tensor);
    assert_close(t[1][1], 0.5 * m * r * r, EXACT, "cylinder Iyy");
    assert_close(
        t[0][0],
        m * (3.0 * r * r + 4.0 * hh * hh) / 12.0,
        EXACT,
        "cylinder Ixx",
    );
}

/// A capsule is a cylinder with two hemispherical caps: quadrature of the solid.
#[test]
fn a_capsule_matches_a_quadrature_of_its_solid() {
    let (r, hh) = (1.0, 1.5);
    let p = capsule_mass_properties(fx(r), fx(hh), fx(1.0));
    let (vol, com, i) = quadrature([-r, -hh - r, -r], [r, hh + r, r], 80, |x, y, z| {
        let dy = (y.abs() - hh).max(0.0);
        x * x + z * z + dy * dy <= r * r
    });
    check_against_quadrature(&p, vol, com, i, GRID, "capsule");
}

// ---------------------------------------------------------------------------
// Convex hulls
// ---------------------------------------------------------------------------

/// The 8 corners of a cube of side 2, in a deliberately scrambled order.
fn cube_corners(centre: [f64; 3]) -> Vec<Vec3Fix> {
    let mut v = Vec::new();
    for &(sx, sy, sz) in &[
        (1.0, -1.0, 1.0),
        (-1.0, -1.0, -1.0),
        (1.0, 1.0, -1.0),
        (-1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
        (-1.0, -1.0, 1.0),
        (1.0, -1.0, -1.0),
        (-1.0, 1.0, -1.0),
    ] {
        v.push(v3(centre[0] + sx, centre[1] + sy, centre[2] + sz));
    }
    v
}

/// A cube (side 2, density 3): mass 24, COM at its centre, `I = m (2² + 2²)/12`
/// about each axis, no products of inertia — wherever it sits.
#[test]
fn a_hull_of_a_cube_has_the_exact_mass_centre_and_inertia() {
    for centre in [[0.0, 0.0, 0.0], [5.0, -3.0, 2.0]] {
        let p = convex_hull_mass_properties(&cube_corners(centre), fx(3.0));
        assert_close(p.mass.to_f64(), 24.0, EXACT, "cube mass");
        let com = arr(p.center_of_mass);
        for a in 0..3 {
            assert!(
                (com[a] - centre[a]).abs() < 1e-9,
                "cube COM {com:?} for centre {centre:?}"
            );
        }
        let t = tensor(p.inertia_tensor);
        for a in 0..3 {
            for b in 0..3 {
                let want = if a == b { 24.0 * 8.0 / 12.0 } else { 0.0 };
                assert!(
                    (t[a][b] - want).abs() < 1e-8,
                    "cube I[{a}][{b}] = {} for centre {centre:?}, expected {want}",
                    t[a][b]
                );
            }
        }
    }
}

/// An octahedron with semi-axes (2, 1, 1.5): `|x|/a + |y|/b + |z|/c ≤ 1`.
#[test]
fn a_hull_of_an_octahedron_matches_a_quadrature_of_its_solid() {
    let (a, b, c) = (2.0, 1.0, 1.5);
    let pts = vec![
        v3(a, 0.0, 0.0),
        v3(-a, 0.0, 0.0),
        v3(0.0, b, 0.0),
        v3(0.0, -b, 0.0),
        v3(0.0, 0.0, c),
        v3(0.0, 0.0, -c),
    ];
    let p = convex_hull_mass_properties(&pts, fx(1.0));
    // Volume of the octahedron: 4/3 a b c.
    assert_close(
        p.mass.to_f64(),
        4.0 / 3.0 * a * b * c,
        EXACT,
        "octahedron mass",
    );
    let (vol, com, i) = quadrature([-a, -b, -c], [a, b, c], 90, |x, y, z| {
        x.abs() / a + y.abs() / b + z.abs() / c <= 1.0
    });
    check_against_quadrature(&p, vol, com, i, GRID, "octahedron");
}

/// A skew hull — a tilted, off-centre convex body with no symmetry — against the
/// quadrature of the same solid, whose `inside` predicate is the convex hull's own
/// half-space description derived here from the six vertices of a wedge.
#[test]
fn a_hull_of_an_asymmetric_wedge_matches_a_quadrature_of_its_solid() {
    // A right triangular prism with legs 3 and 2 and length 4, placed off-centre.
    let o = [1.0, -2.0, 0.5];
    let pts = vec![
        v3(o[0], o[1], o[2]),
        v3(o[0] + 3.0, o[1], o[2]),
        v3(o[0], o[1] + 2.0, o[2]),
        v3(o[0], o[1], o[2] + 4.0),
        v3(o[0] + 3.0, o[1], o[2] + 4.0),
        v3(o[0], o[1] + 2.0, o[2] + 4.0),
    ];
    let p = convex_hull_mass_properties(&pts, fx(2.0));
    // Triangle area 3, length 4: volume 12, mass 24.
    assert_close(p.mass.to_f64(), 24.0, EXACT, "prism mass");
    let (vol, com, i) = quadrature(
        [o[0], o[1], o[2]],
        [o[0] + 3.0, o[1] + 2.0, o[2] + 4.0],
        90,
        |x, y, z| {
            let (u, v) = (x - o[0], y - o[1]);
            u >= 0.0 && v >= 0.0 && u / 3.0 + v / 2.0 <= 1.0 && z >= o[2] && z <= o[2] + 4.0
        },
    );
    // The quadrature is at unit density; scale to density 2.
    let scaled = [
        [i[0][0] * 2.0, i[0][1] * 2.0, i[0][2] * 2.0],
        [i[1][0] * 2.0, i[1][1] * 2.0, i[1][2] * 2.0],
        [i[2][0] * 2.0, i[2][1] * 2.0, i[2][2] * 2.0],
    ];
    check_against_quadrature(&p, vol * 2.0, com, scaled, 0.04, "prism");
}

/// The inertia is about the **centre of mass**, as for every other shape: moving a
/// hull moves its centre of mass and leaves the tensor alone.
#[test]
fn the_hull_inertia_does_not_depend_on_where_the_hull_is() {
    let at_origin = convex_hull_mass_properties(&cube_corners([0.0; 3]), fx(1.0));
    let far = convex_hull_mass_properties(&cube_corners([100.0, 50.0, -75.0]), fx(1.0));
    let (a, b) = (tensor(at_origin.inertia_tensor), tensor(far.inertia_tensor));
    for r in 0..3 {
        for c in 0..3 {
            assert!(
                (a[r][c] - b[r][c]).abs() < 1e-6,
                "I[{r}][{c}]: {} vs {}",
                a[r][c],
                b[r][c]
            );
        }
    }
}

/// Fewer than four points, or points in one plane, have no volume: zero mass
/// properties, not a panic.
#[test]
fn degenerate_point_sets_have_zero_mass_properties() {
    let zero = |p: &MassProperties| p.mass.to_f64() == 0.0;
    assert!(zero(&convex_hull_mass_properties(&[], fx(1.0))));
    assert!(zero(&convex_hull_mass_properties(
        &[v3(0.0, 0.0, 0.0), v3(1.0, 0.0, 0.0)],
        fx(1.0)
    )));
    let flat = [
        v3(0.0, 0.0, 0.0),
        v3(1.0, 0.0, 0.0),
        v3(0.0, 1.0, 0.0),
        v3(1.0, 1.0, 0.0),
        v3(0.5, 0.5, 0.0),
    ];
    assert!(
        zero(&convex_hull_mass_properties(&flat, fx(1.0))),
        "a plane has no volume"
    );
    assert!(
        zero(&convex_hull_mass_properties(
            &cube_corners([0.0; 3]),
            fx(0.0)
        )),
        "no density, no mass"
    );
}

/// The hull mesh's triangles close the surface: every edge is shared by two
/// triangles (traversed in opposite directions), and the signed volumes sum to the
/// cube's.
#[test]
fn the_hull_mesh_of_a_cube_is_a_closed_outward_triangulation() {
    let mesh = build_hull_mesh(&cube_corners([0.0; 3])).expect("a cube is a solid");
    assert_eq!(mesh.vertices.len(), 8, "every cube corner is a hull vertex");
    let mut edges = std::collections::HashMap::new();
    for f in &mesh.faces {
        for k in 0..3 {
            *edges.entry((f[k], f[(k + 1) % 3])).or_insert(0usize) += 1;
        }
    }
    for (&(a, b), &n) in &edges {
        assert_eq!(n, 1, "directed edge ({a},{b}) used {n} times");
        assert_eq!(
            edges.get(&(b, a)),
            Some(&1),
            "edge ({a},{b}) has no opposite"
        );
    }
    let vol: f64 = mesh
        .faces
        .iter()
        .map(|f| {
            let (a, b, c) = (
                arr(mesh.vertices[f[0]]),
                arr(mesh.vertices[f[1]]),
                arr(mesh.vertices[f[2]]),
            );
            let cross = [
                b[1] * c[2] - b[2] * c[1],
                b[2] * c[0] - b[0] * c[2],
                b[0] * c[1] - b[1] * c[0],
            ];
            (a[0] * cross[0] + a[1] * cross[1] + a[2] * cross[2]) / 6.0
        })
        .sum();
    assert_close(vol, 8.0, EXACT, "signed volume of the cube mesh");
    assert!(build_hull_mesh(&[v3(0.0, 0.0, 0.0), v3(1.0, 0.0, 0.0)]).is_none());
}

// ---------------------------------------------------------------------------
// The parallel-axis theorem
// ---------------------------------------------------------------------------

/// `I' = I + m (d² E − d dᵀ)`: a box moved 3 along x gains `m·9` about y and z, and
/// a general offset gains the full outer-product term.
#[test]
fn translate_inertia_applies_the_parallel_axis_theorem() {
    let p = box_mass_properties(v3(1.0, 1.0, 1.0), fx(1.0));
    let m = p.mass.to_f64();
    let base = tensor(p.inertia_tensor);
    let moved = tensor(translate_inertia(&p, v3(3.0, 0.0, 0.0)));
    assert_close(
        moved[0][0],
        base[0][0],
        EXACT,
        "Ixx along the offset is unchanged",
    );
    assert_close(moved[1][1], base[1][1] + 9.0 * m, EXACT, "Iyy gains m d²");
    assert_close(moved[2][2], base[2][2] + 9.0 * m, EXACT, "Izz gains m d²");
    let d = [1.0, -2.0, 0.5];
    let moved = tensor(translate_inertia(&p, v3(d[0], d[1], d[2])));
    let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    for a in 0..3 {
        for b in 0..3 {
            let want = base[a][b] + m * (if a == b { d2 } else { 0.0 } - d[a] * d[b]);
            assert!(
                (moved[a][b] - want).abs() < 1e-9,
                "I'[{a}][{b}] = {} expected {want}",
                moved[a][b]
            );
        }
    }
}
