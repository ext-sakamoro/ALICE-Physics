//! Oracles for `sdf_manifold`: multi-point contact manifold of a sphere
//! pressed into an SDF.
//!
//! Flat ground (`f = y`): a sphere of radius 1 centred at `y = 0.3`
//! penetrates by exactly 0.7 at every tangent-plane sample, the contact
//! normal is +Y, `point_b` is the foot on `y = 0` and `point_a` the sphere
//! surface at `y = -0.7`. The default 5x5 grid spans a 1 m square
//! (offsets -0.5..0.5 step 0.25), so a manifold reduced to 4 points that
//! maximises the contact area is the square's four corners (area 1.0).

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::sdf_collider::{ClosureSdf, SdfCollider};
use alice_physics::sdf_manifold::{generate_sdf_manifold, ManifoldConfig};

fn ground() -> SdfCollider {
    SdfCollider::new_static(
        Box::new(ClosureSdf::new(|_x, y, _z| y, |_x, _y, _z| (0.0, 1.0, 0.0))),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    )
}

fn unit_sphere() -> SdfCollider {
    SdfCollider::new_static(
        Box::new(ClosureSdf::new(
            |x, y, z| (x * x + y * y + z * z).sqrt() - 1.0,
            |x, y, z| {
                let l = (x * x + y * y + z * z).sqrt();
                (x / l, y / l, z / l)
            },
        )),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    )
}

fn p(x: f32, y: f32, z: f32) -> Vec3Fix {
    Vec3Fix::from_f32(x, y, z)
}

fn r(x: f32) -> Fix128 {
    Fix128::from_f32(x)
}

/// Area of the convex hull of points in the XZ plane (monotone chain + shoelace).
fn hull_area_xz(points: &[(f32, f32)]) -> f32 {
    let mut pts = points.to_vec();
    pts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    pts.dedup();
    if pts.len() < 3 {
        return 0.0;
    }
    let cross = |o: (f32, f32), a: (f32, f32), b: (f32, f32)| {
        (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
    };
    let mut lower: Vec<(f32, f32)> = Vec::new();
    for &q in &pts {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], q) <= 1e-9 {
            lower.pop();
        }
        lower.push(q);
    }
    let mut upper: Vec<(f32, f32)> = Vec::new();
    for &q in pts.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], q) <= 1e-9 {
            upper.pop();
        }
        upper.push(q);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    let mut a = 0.0;
    for i in 0..lower.len() {
        let (x1, z1) = lower[i];
        let (x2, z2) = lower[(i + 1) % lower.len()];
        a += x1 * z2 - x2 * z1;
    }
    a.abs() * 0.5
}

#[test]
fn flat_ground_manifold_has_closed_form_contacts() {
    let m = generate_sdf_manifold(
        p(0.0, 0.3, 0.0),
        r(1.0),
        &ground(),
        &ManifoldConfig::default(),
    );
    assert_eq!(m.len(), 4);
    assert!(!m.is_empty());
    for c in &m.contacts {
        assert!((c.depth.to_f32() - 0.7).abs() < 1e-4);
        let (nx, ny, nz) = c.normal.to_f32();
        assert!(nx.abs() < 1e-5 && (ny - 1.0).abs() < 1e-5 && nz.abs() < 1e-5);
        let (_, by, _) = c.point_b.to_f32();
        let (_, ay, _) = c.point_a.to_f32();
        assert!(by.abs() < 1e-4, "point_b y = {by}");
        assert!((ay + 0.7).abs() < 1e-4, "point_a y = {ay}");
        // sample lies inside the +-0.5 square.
        let (bx, _, bz) = c.point_b.to_f32();
        assert!(bx.abs() <= 0.5 + 1e-4 && bz.abs() <= 0.5 + 1e-4);
    }
    assert!((m.avg_depth.to_f32() - 0.7).abs() < 1e-4);
    let (nx, ny, nz) = m.normal.to_f32();
    assert!(nx.abs() < 1e-5 && (ny - 1.0).abs() < 1e-5 && nz.abs() < 1e-5);
    assert!(m.deepest().is_some());
}

#[test]
fn reduced_manifold_spans_the_whole_contact_patch() {
    // The four points must be the corners of the 1 m x 1 m sample square: area 1.0.
    let m = generate_sdf_manifold(
        p(0.0, 0.3, 0.0),
        r(1.0),
        &ground(),
        &ManifoldConfig::default(),
    );
    let pts: Vec<(f32, f32)> = m
        .contacts
        .iter()
        .map(|c| {
            let (x, _, z) = c.point_b.to_f32();
            (x, z)
        })
        .collect();
    let area = hull_area_xz(&pts);
    assert!(
        (area - 1.0).abs() < 1e-3,
        "manifold area {area}, points {pts:?}"
    );
}

#[test]
fn curved_surface_deepest_contact_is_under_the_centre() {
    // Mover r = 0.5 at (0, 1.3, 0) over the unit sphere: centre sample distance 0.3,
    // penetration 0.2; tangent-plane samples further out are shallower.
    let cfg = ManifoldConfig {
        max_contacts: 25,
        ..ManifoldConfig::default()
    };
    let m = generate_sdf_manifold(p(0.0, 1.3, 0.0), r(0.5), &unit_sphere(), &cfg);
    assert_eq!(m.len(), 25);
    let d = m.deepest().unwrap();
    assert!(
        (d.depth.to_f32() - 0.2).abs() < 1e-3,
        "{}",
        d.depth.to_f32()
    );
    let (bx, by, bz) = d.point_b.to_f32();
    assert!(bx.abs() < 1e-3 && (by - 1.0).abs() < 2e-3 && bz.abs() < 1e-3);
    // Closed form per sample: depth = r - (sqrt(1.3^2 + rho^2) - 1), rho the tangent offset.
    let mut want: Vec<f32> = Vec::new();
    for i in -2..=2 {
        for j in -2..=2 {
            let rho2 = (0.25 * i as f32).powi(2) + (0.25 * j as f32).powi(2);
            want.push(0.5 - ((1.69 + rho2).sqrt() - 1.0));
        }
    }
    let mut got: Vec<f32> = m.contacts.iter().map(|c| c.depth.to_f32()).collect();
    got.sort_by(|a, b| a.partial_cmp(b).unwrap());
    want.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for (g, w) in got.iter().zip(&want) {
        assert!((g - w).abs() < 2e-3, "{g} vs {w}");
    }
    // Average normal points up (symmetric samples).
    let (nx, ny, nz) = m.normal.to_f32();
    assert!(nx.abs() < 1e-3 && ny > 0.99 && nz.abs() < 1e-3);
}

#[test]
fn no_contact_and_min_depth_give_an_empty_manifold() {
    let cfg = ManifoldConfig::default();
    let far = generate_sdf_manifold(p(0.0, 5.0, 0.0), r(1.0), &ground(), &cfg);
    assert!(far.is_empty());
    assert_eq!(far.len(), 0);
    assert_eq!(far.avg_depth, Fix128::ZERO);
    assert_eq!(far.normal, Vec3Fix::ZERO);
    assert!(far.deepest().is_none());
    // Exactly touching (gap 0): radius - dist = 0 -> no contact.
    let touch = generate_sdf_manifold(p(0.0, 1.0, 0.0), r(1.0), &ground(), &cfg);
    assert!(touch.is_empty());
    // min_depth above the 0.7 penetration drops every sample.
    let strict = ManifoldConfig {
        min_depth: 0.8,
        ..cfg
    };
    assert!(generate_sdf_manifold(p(0.0, 0.3, 0.0), r(1.0), &ground(), &strict).is_empty());
    // ... and just below keeps them.
    let loose = ManifoldConfig {
        min_depth: 0.69,
        ..cfg
    };
    assert_eq!(
        generate_sdf_manifold(p(0.0, 0.3, 0.0), r(1.0), &ground(), &loose).len(),
        4
    );
}

#[test]
fn max_contacts_is_an_upper_bound() {
    for max in [0usize, 1, 2, 3, 4] {
        let cfg = ManifoldConfig {
            max_contacts: max,
            ..ManifoldConfig::default()
        };
        let m = generate_sdf_manifold(p(0.0, 0.3, 0.0), r(1.0), &ground(), &cfg);
        assert_eq!(m.len(), max, "max_contacts {max}");
        assert_eq!(m.is_empty(), max == 0);
    }
    // More candidates than max, but a larger max keeps all of them.
    let cfg = ManifoldConfig {
        max_contacts: 25,
        ..ManifoldConfig::default()
    };
    assert_eq!(
        generate_sdf_manifold(p(0.0, 0.3, 0.0), r(1.0), &ground(), &cfg).len(),
        25
    );
    // The reduction never returns the same sample twice.
    let m = generate_sdf_manifold(
        p(0.0, 0.3, 0.0),
        r(1.0),
        &ground(),
        &ManifoldConfig::default(),
    );
    for a in 0..m.len() {
        for b in (a + 1)..m.len() {
            assert_ne!(m.contacts[a].point_b, m.contacts[b].point_b);
        }
    }
}

#[test]
fn sample_grid_is_centred_for_every_size() {
    // With max_contacts above the sample count every sample is kept: the point_b
    // coordinates are the grid. It must be symmetric about the contact centre
    // and reach +-sample_radius.
    for n in [1usize, 2, 3, 4, 5, 6] {
        let cfg = ManifoldConfig {
            samples_per_axis: n,
            max_contacts: 64,
            ..ManifoldConfig::default()
        };
        let m = generate_sdf_manifold(p(0.0, 0.3, 0.0), r(1.0), &ground(), &cfg);
        assert_eq!(m.len(), n * n, "n = {n}");
        let xs: Vec<f32> = m.contacts.iter().map(|c| c.point_b.to_f32().0).collect();
        let zs: Vec<f32> = m.contacts.iter().map(|c| c.point_b.to_f32().2).collect();
        let sum_x: f32 = xs.iter().sum();
        let sum_z: f32 = zs.iter().sum();
        assert!(
            sum_x.abs() < 1e-3 && sum_z.abs() < 1e-3,
            "n = {n}: centroid ({sum_x}, {sum_z}) not at the centre"
        );
        let ext = xs
            .iter()
            .chain(zs.iter())
            .fold(0.0_f32, |m, v| m.max(v.abs()));
        let want = if n == 1 { 0.0 } else { 0.5 };
        assert!(
            (ext - want).abs() < 1e-3,
            "n = {n}: extent {ext}, want {want}"
        );
    }
}

#[test]
fn deepest_prefers_the_largest_depth() {
    let cfg = ManifoldConfig {
        max_contacts: 25,
        ..ManifoldConfig::default()
    };
    let m = generate_sdf_manifold(p(0.0, 1.3, 0.0), r(0.5), &unit_sphere(), &cfg);
    let max = m
        .contacts
        .iter()
        .map(|c| c.depth)
        .fold(Fix128::ZERO, |a, b| if b > a { b } else { a });
    assert_eq!(m.deepest().unwrap().depth, max);
}

#[test]
fn touching_is_not_a_contact_even_with_a_negative_min_depth() {
    // Ground y = 0, radius 1 at y = 1: gap 0 -> radius - dist = 0 -> no contact,
    // although every sample has penetration 0 > min_depth = -1.
    let cfg = ManifoldConfig {
        min_depth: -1.0,
        ..ManifoldConfig::default()
    };
    assert!(generate_sdf_manifold(p(0.0, 1.0, 0.0), r(1.0), &ground(), &cfg).is_empty());
}

#[test]
fn min_depth_comparison_is_strict() {
    // Penetration exactly 0.5 (dyadic): kept only when min_depth < 0.5.
    let at = ManifoldConfig {
        min_depth: 0.5,
        ..ManifoldConfig::default()
    };
    assert!(generate_sdf_manifold(p(0.0, 0.5, 0.0), r(1.0), &ground(), &at).is_empty());
    let below = ManifoldConfig {
        min_depth: 0.499,
        ..ManifoldConfig::default()
    };
    assert_eq!(
        generate_sdf_manifold(p(0.0, 0.5, 0.0), r(1.0), &ground(), &below).len(),
        4
    );
}

#[test]
fn scaled_collider_uses_world_distances() {
    // Unit sphere scaled x2 (world radius 2): mover r = 0.5 at (0, 2.3, 0) touches 0.2 deep.
    let sdf = SdfCollider::new_static(
        Box::new(ClosureSdf::new(
            |x, y, z| (x * x + y * y + z * z).sqrt() - 1.0,
            |x, y, z| {
                let l = (x * x + y * y + z * z).sqrt();
                (x / l, y / l, z / l)
            },
        )),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    )
    .with_scale(Fix128::from_int(2));
    let cfg = ManifoldConfig {
        samples_per_axis: 1,
        ..ManifoldConfig::default()
    };
    let m = generate_sdf_manifold(p(0.0, 2.3, 0.0), r(0.5), &sdf, &cfg);
    assert_eq!(m.len(), 1);
    assert!(
        (m.contacts[0].depth.to_f32() - 0.2).abs() < 2e-3,
        "{}",
        m.contacts[0].depth.to_f32()
    );
    let (_, by, _) = m.contacts[0].point_b.to_f32();
    assert!((by - 2.0).abs() < 2e-3, "point_b y = {by}");
}

#[test]
fn reduced_curved_manifold_keeps_the_deepest_sample() {
    // Default max_contacts 4 on the curved field: the centre sample (depth 0.2) is the deepest
    // and the first pick of the reduction.
    let m = generate_sdf_manifold(
        p(0.0, 1.3, 0.0),
        r(0.5),
        &unit_sphere(),
        &ManifoldConfig::default(),
    );
    assert_eq!(m.len(), 4);
    assert!((m.deepest().unwrap().depth.to_f32() - 0.2).abs() < 1e-3);
    assert!(m.contacts.iter().any(|c| {
        let (x, _, z) = c.point_b.to_f32();
        x.abs() < 1e-3 && z.abs() < 1e-3
    }));
}

#[test]
fn zero_max_contacts_is_fully_empty() {
    let cfg = ManifoldConfig {
        max_contacts: 0,
        ..ManifoldConfig::default()
    };
    let m = generate_sdf_manifold(p(0.0, 0.3, 0.0), r(1.0), &ground(), &cfg);
    assert!(m.is_empty());
    assert_eq!(m.normal, Vec3Fix::ZERO);
    assert_eq!(m.avg_depth, Fix128::ZERO);
}

#[test]
fn shrunk_collider_decides_contact_in_world_distance() {
    // Unit sphere scaled x0.5 (world radius 0.5): centre distance of (0, 1.1, 0) is 0.6;
    // a mover of radius 0.7 penetrates by 0.1 (local-space distance would be 1.2: no contact).
    let sdf = SdfCollider::new_static(
        Box::new(ClosureSdf::new(
            |x, y, z| (x * x + y * y + z * z).sqrt() - 1.0,
            |x, y, z| {
                let l = (x * x + y * y + z * z).sqrt();
                (x / l, y / l, z / l)
            },
        )),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    )
    .with_scale(Fix128::from_f32(0.5));
    let cfg = ManifoldConfig {
        samples_per_axis: 1,
        ..ManifoldConfig::default()
    };
    let m = generate_sdf_manifold(p(0.0, 1.1, 0.0), r(0.7), &sdf, &cfg);
    assert_eq!(m.len(), 1);
    assert!((m.contacts[0].depth.to_f32() - 0.1).abs() < 2e-3);
}

#[test]
fn contact_normal_along_x_still_gets_a_full_tangent_frame() {
    // Wall x = 0, normal +X: the tangent frame must come from another axis, otherwise
    // every sample collapses onto the centre. Patch is 1 m x 1 m in the YZ plane.
    let wall = SdfCollider::new_static(
        Box::new(ClosureSdf::new(|x, _y, _z| x, |_x, _y, _z| (1.0, 0.0, 0.0))),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    );
    let m = generate_sdf_manifold(p(0.3, 0.0, 0.0), r(1.0), &wall, &ManifoldConfig::default());
    assert_eq!(m.len(), 4);
    let pts: Vec<(f32, f32)> = m
        .contacts
        .iter()
        .map(|c| {
            let (_, y, z) = c.point_b.to_f32();
            (y, z)
        })
        .collect();
    let area = hull_area_xz(&pts);
    assert!((area - 1.0).abs() < 1e-3, "area {area}, points {pts:?}");
    for c in &m.contacts {
        let (nx, ny, nz) = c.normal.to_f32();
        assert!((nx - 1.0).abs() < 1e-5 && ny.abs() < 1e-5 && nz.abs() < 1e-5);
    }
}
