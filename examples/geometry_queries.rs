//! Geometry queries on the collider primitives
//!
//! The corners of an oriented box, the apex and base of a cone, the box of a ball
//! in a non-Euclidean metric, and the children of a compound that meet a region.
//! Each line prints the query next to the value worked out by hand.
//!
//! ```bash
//! cargo run --example geometry_queries --features std
//! ```

use alice_physics::box_collider::OrientedBox;
use alice_physics::collider::AABB;
use alice_physics::compound::CompoundShape;
use alice_physics::cone::Cone;
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::metric::MetricWeights;

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(
        Fix128::from_f64(x),
        Fix128::from_f64(y),
        Fix128::from_f64(z),
    )
}

fn show(v: Vec3Fix) -> String {
    format!(
        "({:.3}, {:.3}, {:.3})",
        v.x.to_f64(),
        v.y.to_f64(),
        v.z.to_f64()
    )
}

fn main() {
    // A box with half-extents (1, 2, 3) at the origin: the corners are (±1, ±2, ±3).
    let upright = OrientedBox::axis_aligned(Vec3Fix::ZERO, v3(1.0, 2.0, 3.0));
    println!(
        "axis-aligned box, corner 0 = {}  (1, 2, 3)",
        show(upright.corner(0))
    );

    // Turned a quarter about z, x -> y and y -> -x: corner 0 (1, 2, 3) -> (-2, 1, 3).
    let quarter = QuatFix::from_axis_angle(v3(0.0, 0.0, 1.0), Fix128::HALF_PI);
    let turned = OrientedBox::new(Vec3Fix::ZERO, v3(1.0, 2.0, 3.0), quarter);
    println!(
        "turned box, corner 0 = {}  (-2, 1, 3)",
        show(turned.corner(0))
    );
    println!("turned box, all eight corners:");
    for (i, c) in turned.corners().iter().enumerate() {
        println!("  {i}: {}", show(*c));
    }

    // A cone 1.5 above and below its centre: apex and base centre.
    let cone = Cone::new(v3(0.0, 1.0, 0.0), Fix128::ONE, Fix128::from_f64(1.5));
    println!(
        "cone apex {}  (0, 2.5, 0), base centre {}  (0, -0.5, 0)",
        show(cone.apex()),
        show(cone.base_center())
    );

    // The box of a ball of radius 2 in three metrics: Euclidean 2, cube 2,
    // octahedral 2, and the mixed (1, 2, 3) weights 2 / 6.
    let mixed = MetricWeights::new(Fix128::ONE, Fix128::from_int(2), Fix128::from_int(3))
        .expect("positive weights");
    for (name, metric, half) in [
        ("Euclidean", MetricWeights::L2, 2.0),
        ("cube", MetricWeights::LINF, 2.0),
        ("octahedron", MetricWeights::L1, 2.0),
        ("mixed 1/2/3", mixed, 2.0 / 6.0),
    ] {
        let b = AABB::from_metric_ball(Vec3Fix::ZERO, Fix128::from_int(2), metric);
        println!(
            "ball of radius 2, {name:<11}: box half-width {:.4}  ({half:.4})",
            b.max.x.to_f64()
        );
    }

    // Three boxes in a row, and the children that meet a region around the right one.
    let unit = OrientedBox::axis_aligned(Vec3Fix::ZERO, v3(1.0, 1.0, 1.0));
    let mut row = CompoundShape::new();
    for x in [-4.0, 0.0, 4.0] {
        row.add_box(unit, v3(x, 0.0, 0.0), QuatFix::IDENTITY);
    }
    let region = AABB::new(v3(2.5, -1.0, -1.0), v3(6.0, 1.0, 1.0));
    println!(
        "children meeting x in [2.5, 6]: {:?}  ([2])",
        row.overlapping_children(&region, Vec3Fix::ZERO, QuatFix::IDENTITY)
    );
    let all = row.world_aabb(Vec3Fix::ZERO, QuatFix::IDENTITY);
    println!(
        "the row's box: x from {:.1} to {:.1}  (-5 to 5)",
        all.min.x.to_f64(),
        all.max.x.to_f64()
    );
}
