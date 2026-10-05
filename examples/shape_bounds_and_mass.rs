//! Shape Bounds and Mass Example
//!
//! Production entry point for four geometric building blocks that no other
//! example reaches: `HeightField::aabb` (`src/heightfield.rs`),
//! `cylinder_mass_properties` (`src/mass_properties.rs`),
//! `PlaneCollider::new` (`src/plane_collider.rs`) and `build_convex_hull`
//! (`src/convex_mesh_builder.rs`).
//!
//! Every value this example prints is checked against a closed-form
//! expectation computed independently of the function under test:
//! - a height field of `w × d` samples spaced `s` from origin `o` spans
//!   `[o_x, o_x + (w − 1) s] × [min h, max h] × [o_z, o_z + (d − 1) s]`,
//!   and every bilinear sample lies inside that box
//! - a solid cylinder of radius `r`, height `h` and density `ρ` has
//!   `m = ρ π r² h`, `I_axis = m r² / 2` and `I_⊥ = m (3 r² + h²) / 12`; the
//!   transverse moment is also summed slice by slice
//!   (`Σ dm (r²/4 + y²)`, midpoint rule), which converges to the same value
//! - `PlaneCollider::new(n, d)` normalizes `n` and keeps `d`, so the plane is
//!   `n̂ · p = d`; the 3-4-5 normal makes `n̂ = (0, 3/5, 4/5)` exact enough to
//!   put `(7, 6, 8)` on the plane `d = 10`
//! - the convex hull of a cube's corners plus points strictly inside it is the
//!   eight corners
//!
//! Run with: `cargo run --example shape_bounds_and_mass`

use alice_physics::convex_mesh_builder::build_convex_hull;
use alice_physics::heightfield::HeightField;
use alice_physics::mass_properties::cylinder_mass_properties;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::plane_collider::PlaneCollider;

fn close(got: f64, want: f64, tol: f64, what: &str) {
    assert!(
        (got - want).abs() <= tol,
        "{what}: got {got:.12}, closed form {want:.12} (tolerance {tol:.1e})"
    );
}

fn height_field_bounds() {
    // 4 x 3 samples, spacing 1/2, origin (1, 0, -2).
    let heights: Vec<i64> = vec![3, -1, 4, 1, 5, 9, -2, 6, 5, 3, 5, 8];
    let (w, d) = (4u32, 3u32);
    let spacing = Fix128::from_ratio(1, 2);
    let origin = Vec3Fix::from_int(1, 0, -2);
    let field = HeightField::new(
        heights.iter().map(|&h| Fix128::from_int(h)).collect(),
        w,
        d,
        spacing,
        origin,
    );
    let aabb = field.aabb();
    let lo = *heights.iter().min().expect("non-empty");
    let hi = *heights.iter().max().expect("non-empty");
    let want_min = Vec3Fix::new(Fix128::ONE, Fix128::from_int(lo), Fix128::from_int(-2));
    let want_max = Vec3Fix::new(
        Fix128::ONE + spacing * Fix128::from_int(i64::from(w - 1)),
        Fix128::from_int(hi),
        Fix128::from_int(-2) + spacing * Fix128::from_int(i64::from(d - 1)),
    );
    assert_eq!(aabb.min, want_min, "height field AABB min");
    assert_eq!(aabb.max, want_max, "height field AABB max");

    // Every bilinear sample over the footprint lies inside the box.
    for i in 0..=12 {
        for k in 0..=8 {
            let x = Fix128::ONE + Fix128::from_ratio(i, 8);
            let z = Fix128::from_int(-2) + Fix128::from_ratio(k, 8);
            let y = field.sample_height(x, z);
            assert!(
                aabb.min.y <= y && y <= aabb.max.y,
                "sample at ({}, {}) = {} leaves the AABB",
                x.to_f64(),
                z.to_f64(),
                y.to_f64()
            );
        }
    }
    println!(
        "HeightField 4x3: AABB ({}, {}, {}) .. ({}, {}, {})",
        aabb.min.x.to_f64(),
        aabb.min.y.to_f64(),
        aabb.min.z.to_f64(),
        aabb.max.x.to_f64(),
        aabb.max.y.to_f64(),
        aabb.max.z.to_f64()
    );
}

fn cylinder_mass() {
    let (r, half_h, rho) = (0.5_f64, 1.5_f64, 2.0_f64);
    let props = cylinder_mass_properties(
        Fix128::from_ratio(1, 2),
        Fix128::from_ratio(3, 2),
        Fix128::from_int(2),
    );
    let h = 2.0 * half_h;
    let m = rho * std::f64::consts::PI * r * r * h;
    let i_axis = m * r * r / 2.0;
    let i_perp = m * (3.0 * r * r + h * h) / 12.0;
    close(props.mass.to_f64(), m, 1e-12, "cylinder mass ρπr²h");
    let t = props.inertia_tensor;
    close(t.col1.y.to_f64(), i_axis, 1e-12, "I_yy = m r² / 2");
    close(t.col0.x.to_f64(), i_perp, 1e-12, "I_xx = m (3r² + h²) / 12");
    close(t.col2.z.to_f64(), i_perp, 1e-12, "I_zz = I_xx");
    for off in [t.col0.y, t.col0.z, t.col1.x, t.col1.z, t.col2.x, t.col2.y] {
        assert_eq!(off, Fix128::ZERO, "principal axes: no products of inertia");
    }
    assert_eq!(props.center_of_mass, Vec3Fix::ZERO, "centred on the origin");

    // Slice the cylinder into thin disks: each contributes dm (r²/4 + y²)
    // about the x axis. Midpoint rule, error O(1/n²).
    let n = 4000;
    let dy = h / f64::from(n);
    let mut summed = 0.0;
    for k in 0..n {
        let y = -half_h + (f64::from(k) + 0.5) * dy;
        let dm = rho * std::f64::consts::PI * r * r * dy;
        summed += dm * (r * r / 4.0 + y * y);
    }
    close(
        t.col0.x.to_f64(),
        summed,
        1e-6,
        "I_xx against the slice sum",
    );
    println!("cylinder r 0.5 h 3 ρ 2: m = {m:.9}, I_yy = {i_axis:.9}, I_xx = {i_perp:.9}");
}

fn plane() {
    let plane = PlaneCollider::new(Vec3Fix::from_int(0, 3, 4), Fix128::from_int(10));
    close(plane.normal.x.to_f64(), 0.0, 1e-15, "n̂_x");
    close(plane.normal.y.to_f64(), 0.6, 1e-15, "n̂_y = 3/5");
    close(plane.normal.z.to_f64(), 0.8, 1e-15, "n̂_z = 4/5");
    assert_eq!(plane.offset, Fix128::from_int(10), "the offset is kept");
    let on = Vec3Fix::from_int(7, 6, 8);
    close(
        plane.distance_to_point(on).to_f64(),
        0.0,
        1e-15,
        "(7, 6, 8) is on n̂·p = 10",
    );
    close(
        plane.distance_to_point(Vec3Fix::ZERO).to_f64(),
        -10.0,
        1e-15,
        "the origin is 10 behind",
    );
    close(
        plane.distance_to_point(Vec3Fix::from_int(0, 6, 8)).to_f64(),
        0.0,
        1e-15,
        "moving along x stays on the plane",
    );
    close(
        plane
            .distance_to_point(Vec3Fix::from_int(0, 9, 12))
            .to_f64(),
        5.0,
        1e-15,
        "(0, 9, 12) is 15 along n̂, 5 in front",
    );

    let fallback = PlaneCollider::new(Vec3Fix::ZERO, Fix128::from_int(-3));
    assert_eq!(
        fallback.normal,
        Vec3Fix::UNIT_Y,
        "a zero normal falls back to +y"
    );
    close(
        fallback
            .distance_to_point(Vec3Fix::from_int(5, 1, 5))
            .to_f64(),
        4.0,
        0.0,
        "y = -3 plane, point at y = 1",
    );
    println!("PlaneCollider::new((0,3,4), 10): n̂ = (0, 0.6, 0.8), d = 10");
}

fn convex_hull() {
    let corners: Vec<Vec3Fix> = (0..8)
        .map(|i| Vec3Fix::from_int(i & 1, (i >> 1) & 1, (i >> 2) & 1))
        .map(|v| v * Fix128::from_int(2) - Vec3Fix::from_int(1, 1, 1))
        .collect();
    let mut points = Vec::new();
    // Interleave strictly interior points with the corners so the hull has to
    // discard them wherever they come in the insertion order.
    for (k, c) in corners.iter().enumerate() {
        let k = k as i64;
        points.push(Vec3Fix::new(
            Fix128::from_ratio(k - 3, 5),
            Fix128::from_ratio(2 - k, 7),
            Fix128::from_ratio(k % 3 - 1, 3),
        ));
        points.push(*c);
    }
    points.push(Vec3Fix::ZERO);
    let hull = build_convex_hull(&points);
    let mut got = hull.vertices.clone();
    let key = |v: &Vec3Fix| (v.x, v.y, v.z);
    got.sort_by_key(key);
    let mut want = corners.clone();
    want.sort_by_key(key);
    assert_eq!(
        got, want,
        "the hull of a cube plus interior points is its corners"
    );
    println!(
        "build_convex_hull: {} points in, {} hull vertices (the cube corners)",
        points.len(),
        hull.vertices.len()
    );
}

fn main() {
    height_field_bounds();
    cylinder_mass();
    plane();
    convex_hull();
    println!("all closed forms hold");
}
