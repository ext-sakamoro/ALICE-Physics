//! SDF contact manifold: a sphere resting on flat ground.
//!
//! Ground `f = y`, sphere of radius 1 centred at `y = 0.3`: every tangent-plane
//! sample penetrates by 1 - 0.3 = 0.7, `point_b` is the foot on the plane and
//! the 4-point reduction keeps the corners of the 1 m x 1 m sample patch, so
//! the manifold's footprint area is 1.0 m^2.
//!
//! ```bash
//! cargo run --release --example sdf_manifold_patch --features std
//! ```

use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::sdf_collider::{ClosureSdf, SdfCollider};
use alice_physics::sdf_manifold::{generate_sdf_manifold, ManifoldConfig};

fn main() {
    let ground = SdfCollider::new_static(
        Box::new(ClosureSdf::new(|_x, y, _z| y, |_x, _y, _z| (0.0, 1.0, 0.0))),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    );
    let m = generate_sdf_manifold(
        Vec3Fix::from_f32(0.0, 0.3, 0.0),
        Fix128::ONE,
        &ground,
        &ManifoldConfig::default(),
    );
    println!(
        "[sdf_manifold] {} contacts, average depth {:.4}",
        m.len(),
        m.avg_depth.to_f32()
    );
    assert_eq!(m.len(), 4);
    assert!((m.avg_depth.to_f32() - 0.7).abs() < 1e-4);

    // Footprint: shoelace area of the four feet, ordered around their centroid.
    let mut pts: Vec<(f32, f32)> = m
        .contacts
        .iter()
        .map(|c| {
            let (x, _, z) = c.point_b.to_f32();
            (x, z)
        })
        .collect();
    let (cx, cz) = (
        pts.iter().map(|p| p.0).sum::<f32>() / 4.0,
        pts.iter().map(|p| p.1).sum::<f32>() / 4.0,
    );
    pts.sort_by(|a, b| {
        (a.1 - cz)
            .atan2(a.0 - cx)
            .total_cmp(&(b.1 - cz).atan2(b.0 - cx))
    });
    let area = (0..4)
        .map(|i| pts[i].0 * pts[(i + 1) % 4].1 - pts[(i + 1) % 4].0 * pts[i].1)
        .sum::<f32>()
        .abs()
        * 0.5;
    println!("[sdf_manifold] footprint area {area:.4} m^2 (closed form 1.0)");
    assert!((area - 1.0).abs() < 1e-3);

    let deepest = m.deepest().expect("non-empty manifold");
    println!(
        "[sdf_manifold] deepest contact depth {:.4}",
        deepest.depth.to_f32()
    );
    assert!((deepest.depth.to_f32() - 0.7).abs() < 1e-4);
}
