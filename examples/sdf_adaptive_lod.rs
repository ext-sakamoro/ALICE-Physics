//! Adaptive SDF evaluation: cache hits, teleport invalidation.
//!
//! A unit sphere SDF at the origin gives `d(x, 0, 0) = x - 1`. Three bodies sit
//! at x = 15 (d = 14, skipped while still), x = 5 (d = 4, cached) and x = 2
//! (d = 1, high resolution). After a frame the counters show 2 of 3
//! evaluations saved; moving the sphere (a "teleport") without invalidating
//! returns stale distances, `invalidate` / `invalidate_all` fix that.
//!
//! ```bash
//! cargo run --release --example sdf_adaptive_lod --features std
//! ```

use alice_physics::math::{QuatFix, Vec3Fix};
use alice_physics::sdf_adaptive::{AdaptiveConfig, AdaptiveSdfEvaluator};
use alice_physics::sdf_collider::{ClosureSdf, SdfCollider};

fn sphere_at(x: i64) -> SdfCollider {
    let f = ClosureSdf::new(
        |x, y, z| (x * x + y * y + z * z).sqrt() - 1.0,
        |x, y, z| {
            let l = (x * x + y * y + z * z).sqrt();
            (x / l, y / l, z / l)
        },
    );
    SdfCollider::new_static(Box::new(f), Vec3Fix::from_int(x, 0, 0), QuatFix::IDENTITY)
}

fn main() {
    let xs = [15.0_f32, 5.0, 2.0];
    let pos = |i: usize| Vec3Fix::from_f32(xs[i], 0.0, 0.0);
    let mut ev = AdaptiveSdfEvaluator::new(3, AdaptiveConfig::default());
    let before = sphere_at(0);

    ev.begin_frame();
    for i in 0..3 {
        let (d, _) = ev.evaluate(i, pos(i), &before);
        println!("[sdf_adaptive] frame 1 body {i}: d = {d:.3} (closed form {})", xs[i] - 1.0);
        assert!((d - (xs[i] - 1.0)).abs() < 1e-4);
    }
    assert_eq!(ev.stats(), (0, 3));

    ev.begin_frame();
    for i in 0..3 {
        let _ = ev.evaluate(i, pos(i), &before);
    }
    let (saved, total) = ev.stats();
    println!("[sdf_adaptive] frame 2: saved {saved} of {total} evaluations");
    assert_eq!((saved, total), (2, 3));

    // Teleport: the sphere moves 1 m in +x. Body 1 is told, body 0 is not.
    let after = sphere_at(1);
    ev.begin_frame();
    ev.invalidate(1);
    let (d0, _) = ev.evaluate(0, pos(0), &after);
    let (d1, _) = ev.evaluate(1, pos(1), &after);
    println!("[sdf_adaptive] after teleport: body 0 (stale) d = {d0:.3}, body 1 (invalidated) d = {d1:.3}");
    assert!((d0 - 14.0).abs() < 1e-4, "stale cached value expected");
    assert!((d1 - 3.0).abs() < 1e-4, "fresh value expected (5 - 1 - 1)");

    ev.begin_frame();
    ev.invalidate_all();
    for i in 0..3 {
        let (d, _) = ev.evaluate(i, pos(i), &after);
        assert!((d - (xs[i] - 2.0)).abs() < 1e-4);
    }
    println!("[sdf_adaptive] invalidate_all: all 3 re-evaluated, stats {:?}", ev.stats());
    assert_eq!(ev.stats(), (0, 3));
}
