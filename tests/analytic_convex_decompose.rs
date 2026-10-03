//! Oracles for `decompose_sdf`: the convex pieces an SDF is split into.
//!
//! # Where the closed forms come from
//!
//! The decomposition samples the SDF on a voxel grid and keeps the centres of the
//! *inside* cells that touch an outside cell. Put the faces of a box on grid lines
//! and a boundary cell's centre sits exactly `c/2` inside the face (`c` = cell
//! size), so the hull of the surface centres is the box shrunk by `c/2` on every
//! side, and its volume is a product of `(side − c)`. That is a closed form
//! derived from the grid, not from the code under test.
//!
//! A sphere has no such form, but the hull of points inside a convex solid lies in
//! it (an upper bound), and every surface centre is within one cell of the surface
//! (a lower bound).
//!
//! Author: Moroya Sakamoto

#![allow(clippy::disallowed_methods)]

use alice_physics::convex_decompose::{decompose_sdf, DecomposeConfig, DecompositionResult};
use alice_physics::mass_properties::convex_hull_mass_properties;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::sdf_collider::{ClosureSdf, SdfField};

fn v3(x: f32, y: f32, z: f32) -> Vec3Fix {
    Vec3Fix::from_f32(x, y, z)
}

/// A box with faces at `lo`/`hi` (exact distance, negative inside).
fn box_sd(p: [f32; 3], lo: [f32; 3], hi: [f32; 3]) -> f32 {
    let mut outside = 0.0f32;
    let mut inside = f32::MIN;
    for k in 0..3 {
        let centre = 0.5 * (lo[k] + hi[k]);
        let half = 0.5 * (hi[k] - lo[k]);
        let d = (p[k] - centre).abs() - half;
        outside += d.max(0.0) * d.max(0.0);
        inside = inside.max(d);
    }
    outside.sqrt() + inside.min(0.0)
}

fn sdf(f: impl Fn(f32, f32, f32) -> f32 + Send + Sync + 'static) -> ClosureSdf {
    // Decomposition only samples distances; the normal is not used.
    ClosureSdf::new(f, |_, _, _| (0.0, 1.0, 0.0))
}

fn sphere(c: [f32; 3], r: f32) -> impl Fn(f32, f32, f32) -> f32 + Send + Sync + 'static {
    move |x, y, z| ((x - c[0]).powi(2) + (y - c[1]).powi(2) + (z - c[2]).powi(2)).sqrt() - r
}

/// The L of two boxes: `[0,2]×[0,1]×[0,1]` and `[0,1]×[1,2]×[0,1]`, volume 3.
fn l_shape() -> ClosureSdf {
    sdf(|x, y, z| {
        let p = [x, y, z];
        box_sd(p, [0.0, 0.0, 0.0], [2.0, 1.0, 1.0]).min(box_sd(p, [0.0, 1.0, 0.0], [1.0, 2.0, 1.0]))
    })
}

/// Grid `[-1, 3]³` with 32 cells: `c = 0.125`, and every face above is a grid line.
const CELL: f64 = 0.125;

fn grid() -> (Vec3Fix, Vec3Fix) {
    (v3(-1.0, -1.0, -1.0), v3(3.0, 3.0, 3.0))
}

fn config(max_hulls: usize) -> DecomposeConfig {
    DecomposeConfig {
        resolution: 32,
        max_hulls,
        ..DecomposeConfig::default()
    }
}

fn run(field: &dyn SdfField, cfg: &DecomposeConfig) -> DecompositionResult {
    let (lo, hi) = grid();
    decompose_sdf(field, lo, hi, cfg)
}

/// The volume of a hull's solid (mass at unit density), from the exact integral.
fn hull_volume(vertices: &[Vec3Fix]) -> f64 {
    convex_hull_mass_properties(vertices, Fix128::ONE)
        .mass
        .to_f64()
}

fn shrunk_box(sides: [f64; 3]) -> f64 {
    sides.iter().map(|s| s - CELL).product()
}

/// A box is convex: one hull, however many the budget allows, and its volume is
/// the product of `(side − c)`.
#[test]
fn a_convex_box_is_one_hull_of_the_shrunk_box() {
    let field = sdf(|x, y, z| box_sd([x, y, z], [0.0, 0.0, 0.0], [2.0, 1.0, 1.5]));
    let r = run(&field, &config(16));
    assert_eq!(r.hulls.len(), 1, "a convex solid is not split");
    let want = shrunk_box([2.0, 1.0, 1.5]);
    let got = hull_volume(&r.hulls[0].vertices);
    assert!(
        (got - want).abs() < 1e-4,
        "hull volume {got} but the shrunk box is {want}"
    );
}

/// The decomposition reports each hull's volume; it is the hull's, not its
/// bounding box's.
#[test]
fn the_reported_volume_is_the_volume_of_the_hull() {
    let field = sdf(sphere([1.0, 1.0, 1.0], 0.8));
    let r = run(&field, &config(1));
    assert_eq!(r.hulls.len(), 1);
    let got = r.volumes[0].to_f64();
    let want = hull_volume(&r.hulls[0].vertices);
    assert!(
        (got - want).abs() < 1e-4 * want,
        "reported {got}, the hull's own volume is {want} (its bounding box is far larger)"
    );
}

/// A sphere is convex: one hull, inside the sphere and within a cell of it.
#[test]
fn a_sphere_is_one_hull_inside_it_and_within_a_cell() {
    let radius = 0.8f64;
    let field = sdf(sphere([1.0, 1.0, 1.0], radius as f32));
    let r = run(&field, &config(16));
    assert_eq!(r.hulls.len(), 1, "a convex solid is not split");
    let volume = hull_volume(&r.hulls[0].vertices);
    let sphere_volume = 4.0 / 3.0 * std::f64::consts::PI * radius.powi(3);
    assert!(
        volume <= sphere_volume,
        "the hull of points inside a sphere is inside it: {volume} > {sphere_volume}"
    );
    let floor = ((radius - CELL) / radius).powi(3) * sphere_volume;
    assert!(
        volume >= floor,
        "every surface centre is within a cell of the surface: {volume} < {floor}"
    );
}

/// The L is not convex: two hulls, the vertical bar and the rest, whose volumes
/// are the shrunk boxes `(1−c)(2−c)(1−c)` and `(1−c)(1−c)(1−c)`.
#[test]
fn an_l_shape_splits_into_its_two_bars() {
    let r = run(&l_shape(), &config(16));
    assert_eq!(r.hulls.len(), 2, "one concave corner needs one cut");
    let mut got: Vec<f64> = r.hulls.iter().map(|h| hull_volume(&h.vertices)).collect();
    got.sort_by(f64::total_cmp);
    let want = [shrunk_box([1.0, 1.0, 1.0]), shrunk_box([1.0, 2.0, 1.0])];
    for k in 0..2 {
        assert!(
            (got[k] - want[k]).abs() < 1e-4,
            "hull {k}: volume {} but the shrunk bar is {}",
            got[k],
            want[k]
        );
    }
}

/// The budget is a ceiling: with one hull allowed the L stays one piece.
#[test]
fn the_hull_budget_is_a_ceiling() {
    let r = run(&l_shape(), &config(1));
    assert_eq!(r.hulls.len(), 1);
    assert_eq!(r.centers.len(), 1);
    assert_eq!(r.volumes.len(), 1);
}

/// Two separate spheres are two hulls, each around its own centre.
#[test]
fn two_separate_spheres_are_two_hulls() {
    let field = sdf(|x, y, z| {
        sphere([0.0, 1.0, 1.0], 0.5)(x, y, z).min(sphere([2.0, 1.0, 1.0], 0.5)(x, y, z))
    });
    let r = run(&field, &config(16));
    assert_eq!(r.hulls.len(), 2);
    let mut xs: Vec<f64> = r.centers.iter().map(|c| c.x.to_f64()).collect();
    xs.sort_by(f64::total_cmp);
    assert!(xs[0].abs() < CELL, "left centre {}", xs[0]);
    assert!((xs[1] - 2.0).abs() < CELL, "right centre {}", xs[1]);
}

/// Four spheres in a row (the middle gap at the middle of the bounding box, so
/// every cut falls in a gap): the budget is shared between the two sides of a cut,
/// so a budget of `k` gives `k` hulls until all four are apart.
#[test]
fn the_hull_budget_is_shared_between_the_sides_of_a_cut() {
    let field = sdf(|x, y, z| {
        [0.0, 0.7, 1.6, 2.3]
            .iter()
            .map(|&cx| sphere([cx, 1.0, 1.0], 0.25)(x, y, z))
            .fold(f32::MAX, f32::min)
    });
    for (budget, hulls) in [(1, 1), (2, 2), (3, 3), (4, 4), (16, 4)] {
        let r = run(&field, &config(budget));
        assert_eq!(r.hulls.len(), hulls, "a budget of {budget}");
    }
}

/// The concavity threshold is a fraction of the hull's volume (the L's notch is
/// about a sixth of its hull): a threshold above it keeps the L whole, one below
/// it cuts.
#[test]
fn the_concavity_threshold_is_a_fraction_of_the_hull_volume() {
    let kept = run(
        &l_shape(),
        &DecomposeConfig {
            concavity_threshold: 0.6,
            ..config(16)
        },
    );
    assert_eq!(kept.hulls.len(), 1, "a notch of ~15% is under 60%");
    let cut = run(
        &l_shape(),
        &DecomposeConfig {
            concavity_threshold: 0.05,
            ..config(16)
        },
    );
    assert_eq!(cut.hulls.len(), 2, "a notch of ~15% is over 5%");
}

/// A cap below four is raised to four: a hull needs four corners to be a solid.
#[test]
fn a_vertex_cap_below_four_still_gives_a_solid_hull() {
    let field = sdf(sphere([1.0, 1.0, 1.0], 0.8));
    let r = run(
        &field,
        &DecomposeConfig {
            max_vertices_per_hull: 1,
            ..config(1)
        },
    );
    assert_eq!(r.hulls.len(), 1);
    assert_eq!(r.hulls[0].vertices.len(), 4);
    assert!(hull_volume(&r.hulls[0].vertices) > 0.0);
}

/// A hull keeps at most `max_vertices_per_hull` vertices, also when the cluster
/// has fewer than twice as many points as the cap.
#[test]
fn a_hull_keeps_at_most_the_vertex_cap() {
    let field = sdf(sphere([1.0, 1.0, 1.0], 0.9));
    let free = run(
        &field,
        &DecomposeConfig {
            max_vertices_per_hull: 100_000,
            ..config(1)
        },
    );
    let n = free.hulls[0].vertices.len();
    assert!(n > 30, "the sphere has many surface cells ({n})");
    let cap = n * 2 / 3; // between n/2 and n: a floor(n / cap) step is 1
    let capped = run(
        &field,
        &DecomposeConfig {
            max_vertices_per_hull: cap,
            ..config(1)
        },
    );
    assert_eq!(capped.hulls.len(), 1);
    assert!(
        capped.hulls[0].vertices.len() <= cap,
        "{} vertices with a cap of {cap}",
        capped.hulls[0].vertices.len()
    );
}

/// `min_volume` drops hulls whose *volume* is below it.
#[test]
fn a_hull_below_the_minimum_volume_is_dropped() {
    let field = sdf(sphere([1.0, 1.0, 1.0], 0.3));
    let kept = run(&field, &config(1));
    assert_eq!(kept.hulls.len(), 1);
    let volume = hull_volume(&kept.hulls[0].vertices);
    let above = run(
        &field,
        &DecomposeConfig {
            min_volume: (volume * 1.2) as f32,
            ..config(1)
        },
    );
    assert!(
        above.hulls.is_empty(),
        "a hull of volume {volume} is below the minimum {}",
        volume * 1.2
    );
    let below = run(
        &field,
        &DecomposeConfig {
            min_volume: (volume * 0.8) as f32,
            ..config(1)
        },
    );
    assert_eq!(below.hulls.len(), 1);
}

/// Nothing to decompose: an SDF that is nowhere inside the bounds gives no hulls.
#[test]
fn an_sdf_outside_the_bounds_has_no_hulls() {
    let field = sdf(sphere([10.0, 10.0, 10.0], 0.5));
    let r = run(&field, &config(16));
    assert!(r.hulls.is_empty() && r.centers.is_empty() && r.volumes.is_empty());
}

// ---------------------------------------------------------------------------
// The compound of the pieces, and the body it makes
// ---------------------------------------------------------------------------

use alice_physics::compound::CompoundShape;
use alice_physics::shape::Shape;
use alice_physics::solver::{PhysicsWorld, SolverConfig};

/// The L's two hulls as a compound: mass and centre of mass are the sums over the
/// two shrunk bars (whose centres are at the middle of each bar).
#[test]
fn the_compound_of_an_l_has_the_mass_and_centre_of_its_two_bars() {
    let (lo, hi) = grid();
    let c = CELL;
    let compound = CompoundShape::from_sdf(&l_shape(), lo, hi, &config(16));
    assert_eq!(compound.len(), 2);
    let density = 3.0;
    let p = compound.mass_properties(Fix128::from_f64(density));

    // The vertical bar [c/2, 1−c/2]×[c/2, 2−c/2]×[c/2, 1−c/2] and the right bar
    // [1+c/2, 2−c/2]×[c/2, 1−c/2]×[c/2, 1−c/2] (the cut is at x = 1).
    let v_left = (1.0 - c) * (2.0 - c) * (1.0 - c);
    let v_right = (1.0 - c) * (1.0 - c) * (1.0 - c);
    let centre_left = [0.5, 1.0, 0.5];
    let centre_right = [1.5, 0.5, 0.5];
    let mass = density * (v_left + v_right);
    assert!(
        (p.mass.to_f64() - mass).abs() < 1e-3 * mass,
        "mass {} but the two bars weigh {mass}",
        p.mass.to_f64()
    );
    let want = [
        (v_left * centre_left[0] + v_right * centre_right[0]) / (v_left + v_right),
        (v_left * centre_left[1] + v_right * centre_right[1]) / (v_left + v_right),
        (v_left * centre_left[2] + v_right * centre_right[2]) / (v_left + v_right),
    ];
    let got = [
        p.center_of_mass.x.to_f64(),
        p.center_of_mass.y.to_f64(),
        p.center_of_mass.z.to_f64(),
    ];
    for k in 0..3 {
        assert!(
            (got[k] - want[k]).abs() < 1e-3,
            "centre of mass axis {k}: {} but the bars say {}",
            got[k],
            want[k]
        );
    }
}

/// The body made of the L collides as an L: a probe in the notch, which the single
/// convex hull of the L would fill, touches nothing, and a probe on a bar does.
#[test]
fn the_body_of_an_l_collides_as_an_l_not_as_its_hull() {
    let (lo, hi) = grid();
    let compound = CompoundShape::from_sdf(&l_shape(), lo, hi, &config(16));
    let density = Fix128::from_int(1000);
    let com = compound.mass_properties(density).center_of_mass;

    let mut world = PhysicsWorld::new(SolverConfig::default());
    // The body's centre of mass at the origin: a world point is the L's point minus
    // the centre of mass.
    let l = world
        .add_compound_body(&compound, density, Vec3Fix::ZERO)
        .expect("the L has volume");
    let probe = Shape::Ellipsoid {
        radii: Vec3Fix::new(
            Fix128::from_f64(0.15),
            Fix128::from_f64(0.15),
            Fix128::from_f64(0.15),
        ),
    };
    let at = |x: f64, y: f64, z: f64| {
        Vec3Fix::new(
            Fix128::from_f64(x) - com.x,
            Fix128::from_f64(y) - com.y,
            Fix128::from_f64(z) - com.z,
        )
    };
    // The notch (1.3, 1.3) is outside the L but inside its convex hull (the hull's
    // slanted edge is x + y = 3, and the nearest bar is 0.42 away); the
    // vertical bar's top (0.5, 1.8) and the horizontal bar's end (1.8, 0.5) are in it.
    for (what, point, touches) in [
        ("the notch", at(1.3, 1.3, 0.5), false),
        ("the top of the vertical bar", at(0.5, 1.8, 0.5), true),
        ("the end of the horizontal bar", at(1.8, 0.5, 0.5), true),
    ] {
        let p = world
            .add_shaped_body(&probe, Fix128::from_int(1000), point)
            .expect("a valid solid");
        assert_eq!(
            world.colliders_overlap(l, p),
            touches,
            "a probe at {what}: touching is {touches}"
        );
    }
}
