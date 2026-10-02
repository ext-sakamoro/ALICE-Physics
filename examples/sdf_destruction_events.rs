//! SDF boolean-destruction event pipeline: named damage-event constructors
//! (explosion / impact / projectile), the raw shape constructors they build
//! on (sphere / cube / cylinder), `apply_destruction`'s damage-count
//! accounting, `optimize`'s compaction, `reset`, and `with_smoothing`'s
//! effect on the carved surface.
//!
//! Every printed quantity here is pinned by a closed-form oracle in
//! `tests/analytic_sdf_destruction_wiring.rs`.
//!
//! ```bash
//! cargo run --example sdf_destruction_events --features std
//! ```

use alice_physics::collider::Contact;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::sdf_collider::{ClosureSdf, SdfField};
use alice_physics::sdf_destruction::{
    destruction_from_explosion, destruction_from_impact, destruction_from_projectile,
    DestructibleSdf, DestructionShape,
};
use std::panic::{self, AssertUnwindSafe};

/// Flat ground: `distance(x, y, z) = y`. A *sharp* crater of radius `r`
/// centered at `(cx, 0, cz)` evaluates to exactly `max(0, -(0 - r)) = r` at
/// its own center (ground value there is `0`), which is what every closed
/// form below checks against.
fn ground() -> ClosureSdf {
    ClosureSdf::new(|_x, y, _z| y, |_x, _y, _z| (0.0, 1.0, 0.0))
}

/// A bounded unit sphere centered at the origin, for the "100% removal"
/// degenerate case (a crater that engulfs the whole shape).
fn unit_sphere() -> ClosureSdf {
    ClosureSdf::new(
        |x, y, z| z.mul_add(z, x.mul_add(x, y * y)).sqrt() - 1.0,
        |x, y, z| {
            let len = z.mul_add(z, x.mul_add(x, y * y)).sqrt();
            if len < 1e-10 {
                (0.0, 1.0, 0.0)
            } else {
                (x / len, y / len, z / len)
            }
        },
    )
}

fn main() {
    // ------------------------------------------------------------------
    // 1. apply_destruction via the three named event constructors +
    //    the three raw shape constructors — each call increments both
    //    destruction_count() and total_destruction_count() by exactly 1.
    // ------------------------------------------------------------------
    let mut dsdf = DestructibleSdf::new(Box::new(ground()));
    println!(
        "[sdf_destruction] start: count={} total={}",
        dsdf.destruction_count(),
        dsdf.total_destruction_count()
    );
    assert_eq!(dsdf.destruction_count(), 0);
    assert_eq!(dsdf.total_destruction_count(), 0);

    // raw constructor: sphere
    dsdf.apply_destruction(DestructionShape::sphere(
        Vec3Fix::from_f32(5.0, 0.0, 0.0),
        1.0,
    ));
    println!(
        "[sdf_destruction] sphere() applied: count={} total={} d(5,0,0)={}",
        dsdf.destruction_count(),
        dsdf.total_destruction_count(),
        dsdf.distance(5.0, 0.0, 0.0)
    );
    assert_eq!(dsdf.destruction_count(), 1);
    assert_eq!(dsdf.total_destruction_count(), 1);
    assert!((dsdf.distance(5.0, 0.0, 0.0) - 1.0).abs() < 1e-6);

    // raw constructor: cube
    dsdf.apply_destruction(DestructionShape::cube(
        Vec3Fix::from_f32(10.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
    ));
    println!(
        "[sdf_destruction] cube() applied: count={} total={} d(10,0,0)={}",
        dsdf.destruction_count(),
        dsdf.total_destruction_count(),
        dsdf.distance(10.0, 0.0, 0.0)
    );
    assert_eq!(dsdf.destruction_count(), 2);
    assert_eq!(dsdf.total_destruction_count(), 2);
    assert!((dsdf.distance(10.0, 0.0, 0.0) - 1.0).abs() < 1e-6);

    // raw constructor: cylinder
    dsdf.apply_destruction(DestructionShape::cylinder(
        Vec3Fix::from_f32(15.0, 0.0, 0.0),
        1.0,
        1.0,
    ));
    println!(
        "[sdf_destruction] cylinder() applied: count={} total={} d(15,0,0)={}",
        dsdf.destruction_count(),
        dsdf.total_destruction_count(),
        dsdf.distance(15.0, 0.0, 0.0)
    );
    assert_eq!(dsdf.destruction_count(), 3);
    assert_eq!(dsdf.total_destruction_count(), 3);
    assert!((dsdf.distance(15.0, 0.0, 0.0) - 1.0).abs() < 1e-6);

    // named event constructor: explosion (sphere + with_smoothing chained inside)
    let explosion = destruction_from_explosion(Vec3Fix::from_f32(20.0, 0.0, 0.0), 1.0, 0.0);
    dsdf.apply_destruction(explosion);
    println!(
        "[sdf_destruction] destruction_from_explosion() applied: count={} total={} d(20,0,0)={}",
        dsdf.destruction_count(),
        dsdf.total_destruction_count(),
        dsdf.distance(20.0, 0.0, 0.0)
    );
    assert_eq!(dsdf.destruction_count(), 4);
    assert_eq!(dsdf.total_destruction_count(), 4);
    assert!((dsdf.distance(20.0, 0.0, 0.0) - 1.0).abs() < 1e-6);

    // named event constructor: impact (velocity -> clamped radius)
    let contact = Contact {
        depth: Fix128::from_f32(0.1),
        normal: Vec3Fix::UNIT_Y,
        point_a: Vec3Fix::from_f32(25.0, 0.0, 0.0),
        point_b: Vec3Fix::from_f32(25.0, 0.0, 0.0),
    };
    let impact = destruction_from_impact(&contact, Fix128::from_f32(20.0), 0.05, 0.2, 1.0);
    dsdf.apply_destruction(impact);
    println!(
        "[sdf_destruction] destruction_from_impact() applied: count={} total={} d(25,0,0)={}",
        dsdf.destruction_count(),
        dsdf.total_destruction_count(),
        dsdf.distance(25.0, 0.0, 0.0)
    );
    assert_eq!(dsdf.destruction_count(), 5);
    assert_eq!(dsdf.total_destruction_count(), 5);
    assert!((dsdf.distance(25.0, 0.0, 0.0) - 1.0).abs() < 1e-6);

    // named event constructor: projectile (cylinder bore along a direction)
    let projectile =
        destruction_from_projectile(Vec3Fix::from_f32(30.0, 0.0, 0.0), Vec3Fix::UNIT_Y, 1.0, 2.0);
    dsdf.apply_destruction(projectile);
    println!(
        "[sdf_destruction] destruction_from_projectile() applied: count={} total={} d(30,1,0)={}",
        dsdf.destruction_count(),
        dsdf.total_destruction_count(),
        dsdf.distance(30.0, 1.0, 0.0)
    );
    assert_eq!(dsdf.destruction_count(), 6);
    assert_eq!(dsdf.total_destruction_count(), 6);
    assert!((dsdf.distance(30.0, 1.0, 0.0) - 1.0).abs() < 1e-6);

    // ------------------------------------------------------------------
    // 2. with_smoothing: a smoothed crater reads back differently from a
    //    sharp one of identical center/radius (observable builder effect).
    // ------------------------------------------------------------------
    let mut sharp = DestructibleSdf::new(Box::new(ground()));
    sharp.apply_destruction(DestructionShape::sphere(Vec3Fix::ZERO, 1.0));
    let mut smooth = DestructibleSdf::new(Box::new(ground()));
    smooth.apply_destruction(DestructionShape::sphere(Vec3Fix::ZERO, 1.0).with_smoothing(3.0));
    let (d_sharp, d_smooth) = (
        sharp.distance(0.0, 0.0, 0.0),
        smooth.distance(0.0, 0.0, 0.0),
    );
    println!("[sdf_destruction] with_smoothing: sharp={d_sharp} smooth={d_smooth}");
    assert!(
        (d_sharp - 1.0).abs() < 1e-6,
        "sharp should read back exactly 1.0: {d_sharp}"
    );
    assert!(
        (d_smooth - d_sharp).abs() > 0.1,
        "smoothed crater must read back observably differently: sharp={d_sharp} smooth={d_smooth}"
    );

    // ------------------------------------------------------------------
    // 3. optimize(): no-op below the 32-crater cap, and a no-op on an
    //    undamaged volume.
    // ------------------------------------------------------------------
    dsdf.optimize();
    println!(
        "[sdf_destruction] optimize() below cap is a no-op: count={} total={}",
        dsdf.destruction_count(),
        dsdf.total_destruction_count()
    );
    assert_eq!(dsdf.destruction_count(), 6);
    assert_eq!(dsdf.total_destruction_count(), 6);

    let mut empty = DestructibleSdf::new(Box::new(ground()));
    let before = empty.distance(0.0, 0.0, 0.0);
    empty.optimize();
    let after = empty.distance(0.0, 0.0, 0.0);
    println!("[sdf_destruction] optimize() on an undamaged volume is a no-op: before={before} after={after}");
    assert_eq!(empty.destruction_count(), 0);
    assert!((before - after).abs() < 1e-12);

    // ------------------------------------------------------------------
    // 4. Degenerate inputs.
    // ------------------------------------------------------------------

    // (a) An event whose geometry does not touch anywhere we query is still
    //     recorded: apply_destruction never inspects the shape, it always
    //     pushes + counts.
    let far = destruction_from_explosion(Vec3Fix::from_f32(1.0e6, 0.0, 0.0), 0.5, 0.0);
    dsdf.apply_destruction(far);
    println!(
        "[sdf_destruction] far-away event is still recorded: count={} total={}",
        dsdf.destruction_count(),
        dsdf.total_destruction_count()
    );
    assert_eq!(dsdf.destruction_count(), 7);
    assert_eq!(dsdf.total_destruction_count(), 7);
    // and it does not perturb the ground anywhere near the other craters:
    assert!((dsdf.distance(5.0, 0.0, 0.0) - 1.0).abs() < 1e-6);

    // (b) A crater that removes 100% of a bounded shape.
    let mut bounded = DestructibleSdf::new(Box::new(unit_sphere()));
    let inside_before = bounded.distance(0.0, 0.0, 0.0);
    bounded.apply_destruction(destruction_from_explosion(Vec3Fix::ZERO, 100.0, 0.0));
    let inside_after = bounded.distance(0.0, 0.0, 0.0);
    println!(
        "[sdf_destruction] 100% removal: inside_before={inside_before} inside_after={inside_after}"
    );
    assert!(inside_before < 0.0, "origin starts inside the unit sphere");
    assert!(
        inside_after > 0.0,
        "origin must read back outside once a crater larger than the shape is applied"
    );

    // (c) Zero-radius / zero-size shapes do not panic, and the documented
    //     formula still applies: a zero-radius sphere only carves where the
    //     base value was already negative (`0 > dist`), so a ground point
    //     (dist == 0) is left untouched.
    let mut zero = DestructibleSdf::new(Box::new(ground()));
    zero.apply_destruction(DestructionShape::sphere(
        Vec3Fix::from_f32(50.0, 0.0, 0.0),
        0.0,
    ));
    let zero_cube = panic::catch_unwind(|| {
        DestructionShape::cube(Vec3Fix::from_f32(60.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    });
    println!(
        "[sdf_destruction] zero-radius sphere d(50,0,0)={} (unchanged ground=0), zero-size cube ctor ok={}",
        zero.distance(50.0, 0.0, 0.0),
        zero_cube.is_ok()
    );
    assert!((zero.distance(50.0, 0.0, 0.0) - 0.0).abs() < 1e-6);
    assert!(zero_cube.is_ok());

    // (d) Extreme coordinates do not panic. `1e30` is still well inside
    //     `f32::MAX` (~3.4e38), so the ground plane's `distance(x,y,z) = y`
    //     and every crater's `(dx*dx+dy*dy+dz*dz).sqrt()` stay finite; this
    //     `catch_unwind` documents that no panic occurs even this close to
    //     the overflow boundary.
    let extreme = panic::catch_unwind(AssertUnwindSafe(|| dsdf.distance(1.0e30, 1.0e30, 1.0e30)));
    println!("[sdf_destruction] extreme coordinates: {extreme:?}");
    assert!(
        extreme.is_ok(),
        "distance() must not panic on extreme coordinates"
    );
    assert!(
        extreme.unwrap().is_finite(),
        "1e30 is within f32 range, so the result must stay finite"
    );

    // ------------------------------------------------------------------
    // 5. reset(): drops all destructions, total_destruction_count()
    //    (the lifetime statistic) is unaffected, and the surface is
    //    restored to the original.
    // ------------------------------------------------------------------
    let total_before_reset = dsdf.total_destruction_count();
    dsdf.reset();
    println!(
        "[sdf_destruction] reset(): count={} total={} (total unchanged from {total_before_reset})",
        dsdf.destruction_count(),
        dsdf.total_destruction_count()
    );
    assert_eq!(dsdf.destruction_count(), 0);
    assert_eq!(dsdf.total_destruction_count(), total_before_reset);
    assert!(
        (dsdf.distance(5.0, 0.0, 0.0) - 0.0).abs() < 1e-6,
        "after reset the ground should read back as if undamaged"
    );

    // optimize() on the now-empty (reset) volume is also a no-op.
    dsdf.optimize();
    assert_eq!(dsdf.destruction_count(), 0);

    println!("[sdf_destruction] all assertions passed");
}
