//! Production entry point for SDF-driven force fields: the four named
//! constructors (`attract` / `repel` / `contain` / `surface_flow`), the
//! `with_affected_bodies` scope filter, and the two driver functions
//! `compute_sdf_force` (single field, single body) and
//! `apply_sdf_force_fields` (all registered fields, one physics step —
//! `F = m*a`, `v += a*dt`).
//!
//! Every printed quantity here is pinned by a closed-form oracle in
//! `tests/analytic_sdf_force_wiring.rs`, reconstructed from `Fix128` /
//! `Vec3Fix` primitives (not by calling the functions under test).
//!
//! ```bash
//! cargo run --example sdf_force_fields --features std
//! ```

use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::sdf_collider::{ClosureSdf, SdfCollider};
use alice_physics::sdf_force::{apply_sdf_force_fields, compute_sdf_force};
use alice_physics::{RigidBody, SdfForceField, SdfForceType};
use std::panic::{self, AssertUnwindSafe};

/// Unit sphere centered at the origin: `distance(p) = |p| - 1`,
/// `normal(p) = p / |p|` (falls back to `+Y` at the degenerate center,
/// where the gradient direction is undefined).
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

/// Flat ground: `distance(x, y, z) = y`, constant normal `+Y`.
fn ground_plane() -> ClosureSdf {
    ClosureSdf::new(|_x, y, _z| y, |_x, _y, _z| (0.0, 1.0, 0.0))
}

/// `SdfCollider` is not `Clone` (its field is `Box<dyn SdfField>`), so every
/// place below that needs its own owned instance (slice arguments to
/// `apply_sdf_force_fields`) builds a fresh one from these two factories
/// instead of sharing one value across multiple owning sites.
fn sphere_collider() -> SdfCollider {
    SdfCollider::new_static(Box::new(unit_sphere()), Vec3Fix::ZERO, QuatFix::IDENTITY)
}

fn ground_collider() -> SdfCollider {
    SdfCollider::new_static(Box::new(ground_plane()), Vec3Fix::ZERO, QuatFix::IDENTITY)
}

fn main() {
    let sphere = sphere_collider();
    let ground = ground_collider();

    // ------------------------------------------------------------------
    // 1. attract(sdf_index, strength): max_force is fixed at strength*10.
    // ------------------------------------------------------------------
    let attract = SdfForceField::attract(0, Fix128::from_int(3));
    let body_out = RigidBody::new(Vec3Fix::from_int(3, 0, 0), Fix128::ONE); // dist = 2
    let f_attract = compute_sdf_force(&body_out, &sphere, &attract.force_type);
    println!("[sdf_force] attract(strength=3) at dist=2: force={f_attract}");
    assert_eq!(f_attract, Vec3Fix::from_int(-6, 0, 0));

    // clamp: strength=3 at dist=20 would be 60, but max_force = 3*10 = 30.
    let body_far = RigidBody::new(Vec3Fix::from_int(21, 0, 0), Fix128::ONE); // dist = 20
    let f_clamped = compute_sdf_force(&body_far, &sphere, &attract.force_type);
    println!("[sdf_force] attract(strength=3) at dist=20 (clamped): force={f_clamped}");
    assert_eq!(f_clamped, Vec3Fix::from_int(-30, 0, 0));

    // ------------------------------------------------------------------
    // 2. repel(sdf_index, strength, range): strength/range pass straight
    //    through (const fn, no derived defaults).
    // ------------------------------------------------------------------
    let repel = SdfForceField::repel(0, Fix128::from_int(8), Fix128::from_int(4));
    let body_mid = RigidBody::new(Vec3Fix::from_int(2, 0, 0), Fix128::ONE); // dist = 1
    let f_repel = compute_sdf_force(&body_mid, &sphere, &repel.force_type);
    let factor = Fix128::ONE - Fix128::ONE / Fix128::from_int(4); // 1 - dist/range
    let expect_repel = Fix128::from_int(8) * factor * factor;
    println!(
        "[sdf_force] repel(strength=8, range=4) at dist=1: force={f_repel} (expect {expect_repel})"
    );
    assert_eq!(
        f_repel,
        Vec3Fix::new(expect_repel, Fix128::ZERO, Fix128::ZERO)
    );

    // degenerate: far outside the field's range of effect -> zero.
    let body_outside_range = RigidBody::new(Vec3Fix::from_int(10, 0, 0), Fix128::ONE); // dist = 9
    let f_repel_out = compute_sdf_force(&body_outside_range, &sphere, &repel.force_type);
    println!("[sdf_force] repel() beyond range (dist=9 > range=4): force={f_repel_out}");
    assert_eq!(f_repel_out, Vec3Fix::ZERO);

    // ------------------------------------------------------------------
    // 3. contain(sdf_index, strength): damping is fixed at 1/10.
    // ------------------------------------------------------------------
    let contain = SdfForceField::contain(1, Fix128::from_int(10));
    let damping = Fix128::from_ratio(1, 10);
    let mut body_inside = RigidBody::new(Vec3Fix::from_int(0, -2, 0), Fix128::ONE);
    body_inside.velocity = Vec3Fix::from_int(8, 0, 0);
    let f_in = compute_sdf_force(&body_inside, &ground, &contain.force_type);
    let expect_in = body_inside.velocity * (-damping);
    println!("[sdf_force] contain(strength=10) inside (dist=-2): force={f_in} (expect {expect_in}, damping-only)");
    assert_eq!(f_in, expect_in);

    let mut body_outside = RigidBody::new(Vec3Fix::from_int(0, 3, 0), Fix128::ONE); // dist = 3
    body_outside.velocity = Vec3Fix::from_int(8, 0, 0);
    let f_out = compute_sdf_force(&body_outside, &ground, &contain.force_type);
    let push = -Vec3Fix::UNIT_Y * (Fix128::from_int(10) * Fix128::from_int(3));
    let damp = body_outside.velocity * (-damping);
    println!(
        "[sdf_force] contain(strength=10) outside (dist=3): force={f_out} (expect {})",
        push + damp
    );
    assert_eq!(f_out, push + damp);

    // ------------------------------------------------------------------
    // 4. surface_flow(sdf_index, direction, strength): influence_distance
    //    is fixed at 2.
    // ------------------------------------------------------------------
    let flow = SdfForceField::surface_flow(0, Vec3Fix::from_int(0, 0, 8), Fix128::from_int(5));
    let body_flow = RigidBody::new(Vec3Fix::from_int(3, 0, 0), Fix128::ONE); // dist = 2
    let f_flow = compute_sdf_force(&body_flow, &sphere, &flow.force_type);
    println!("[sdf_force] surface_flow(dir=(0,0,8), strength=5) at dist=2 (== influence_distance): force={f_flow}");
    // dist == influence_distance -> falloff = 1 - 2/2 = 0 -> zero, even though
    // the tangential direction itself is well-defined (flow (0,0,8) is
    // already tangent to the +X normal here).
    assert_eq!(f_flow, Vec3Fix::ZERO);

    let body_flow_mid = RigidBody::new(Vec3Fix::from_f32(2.0, 0.0, 0.0), Fix128::ONE); // dist = 1
    let f_flow_mid = compute_sdf_force(&body_flow_mid, &sphere, &flow.force_type);
    let falloff = Fix128::ONE - Fix128::ONE / Fix128::from_int(2);
    let expect_flow_mid = Vec3Fix::from_int(0, 0, 1) * (Fix128::from_int(5) * falloff);
    println!("[sdf_force] surface_flow() at dist=1: force={f_flow_mid} (expect {expect_flow_mid})");
    assert_eq!(f_flow_mid, expect_flow_mid);

    // degenerate: flow parallel to the normal has no tangential component.
    let radial_flow =
        SdfForceField::surface_flow(0, Vec3Fix::from_int(5, 0, 0), Fix128::from_int(5));
    let f_radial = compute_sdf_force(&body_flow_mid, &sphere, &radial_flow.force_type);
    println!("[sdf_force] surface_flow(dir=(5,0,0)) parallel to normal: force={f_radial} (expect zero, no tangent)");
    assert_eq!(f_radial, Vec3Fix::ZERO);

    // ------------------------------------------------------------------
    // 5. with_affected_bodies(): builder restricts which body indices a
    //    field acts on. `Some(vec![])` is a documented no-op (distinct
    //    from the default `None`, which affects every body).
    // ------------------------------------------------------------------
    let push_field = SdfForceField::new(
        0,
        SdfForceType::Contain {
            strength: Fix128::from_int(4),
            damping: Fix128::ZERO,
        },
    );
    let dt = Fix128::ONE;
    let make_body = || {
        let mut b = RigidBody::new_dynamic(Vec3Fix::from_int(0, 3, 0), Fix128::ONE); // dist = 3
        b.velocity = Vec3Fix::ZERO;
        b
    };

    // default scope (None) affects every body.
    let mut all_bodies = vec![make_body(), make_body()];
    apply_sdf_force_fields(
        std::slice::from_ref(&push_field),
        &[ground_collider()],
        &mut all_bodies,
        dt,
    );
    println!(
        "[sdf_force] with_affected_bodies: default scope -> body0.v={} body1.v={}",
        all_bodies[0].velocity, all_bodies[1].velocity
    );
    let expect_push = -Vec3Fix::UNIT_Y * (Fix128::from_int(4) * Fix128::from_int(3));
    assert_eq!(all_bodies[0].velocity, expect_push);
    assert_eq!(all_bodies[1].velocity, expect_push);

    // empty scope (Some(vec![])) affects no body: documented no-op.
    let empty_scope = push_field.clone().with_affected_bodies(vec![]);
    let mut none_bodies = vec![make_body(), make_body()];
    apply_sdf_force_fields(&[empty_scope], &[ground_collider()], &mut none_bodies, dt);
    println!(
        "[sdf_force] with_affected_bodies(vec![]): body0.v={} body1.v={} (expect both zero, no-op)",
        none_bodies[0].velocity, none_bodies[1].velocity
    );
    assert_eq!(none_bodies[0].velocity, Vec3Fix::ZERO);
    assert_eq!(none_bodies[1].velocity, Vec3Fix::ZERO);

    // scoped to body 1 only.
    let scoped = push_field.with_affected_bodies(vec![1]);
    let mut scoped_bodies = vec![make_body(), make_body()];
    apply_sdf_force_fields(&[scoped], &[ground_collider()], &mut scoped_bodies, dt);
    println!(
        "[sdf_force] with_affected_bodies(vec![1]): body0.v={} body1.v={}",
        scoped_bodies[0].velocity, scoped_bodies[1].velocity
    );
    assert_eq!(scoped_bodies[0].velocity, Vec3Fix::ZERO);
    assert_eq!(scoped_bodies[1].velocity, expect_push);

    // ------------------------------------------------------------------
    // 6. apply_sdf_force_fields(): the production driver — sums every
    //    field that affects a body, then integrates F = m*a, v += a*dt.
    //    Static bodies are skipped entirely.
    // ------------------------------------------------------------------
    let combo_attract = SdfForceField::attract(0, Fix128::from_int(3)); // max_force=30
    let combo_repel = SdfForceField::repel(0, Fix128::from_int(8), Fix128::from_int(4));
    let colliders = [sphere_collider()];
    let dt2 = Fix128::from_ratio(1, 4);

    let mut dyn_body = RigidBody::new_dynamic(Vec3Fix::from_int(3, 0, 0), Fix128::ONE); // dist = 2
    dyn_body.inv_mass = Fix128::from_int(2);
    dyn_body.velocity = Vec3Fix::from_int(5, 0, 0);
    let mut static_body = RigidBody::new_static(Vec3Fix::from_int(3, 0, 0));
    static_body.velocity = Vec3Fix::from_int(5, 0, 0); // should never move: is_static() skips it
    let mut bodies = vec![dyn_body, static_body];

    apply_sdf_force_fields(
        &[combo_attract.clone(), combo_repel.clone()],
        &colliders,
        &mut bodies,
        dt2,
    );
    println!(
        "[sdf_force] apply_sdf_force_fields(attract+repel) dyn.v={} static.v={}",
        bodies[0].velocity, bodies[1].velocity
    );
    // total_force = attract(-6,0,0) + repel(2,0,0) = (-4,0,0)
    // accel = total_force * inv_mass(2) = (-8,0,0); dv = accel * dt(1/4) = (-2,0,0)
    assert_eq!(bodies[0].velocity, Vec3Fix::from_int(3, 0, 0));
    assert_eq!(bodies[1].velocity, Vec3Fix::from_int(5, 0, 0)); // unchanged: static

    // degenerate: an empty fields slice and an empty bodies slice are both
    // no-ops (the production loop simply has nothing to iterate).
    let mut untouched = vec![RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE)];
    apply_sdf_force_fields(&[], &colliders, &mut untouched, dt2);
    println!(
        "[sdf_force] apply_sdf_force_fields(no fields): v={}",
        untouched[0].velocity
    );
    assert_eq!(untouched[0].velocity, Vec3Fix::ZERO);
    let mut no_bodies: Vec<RigidBody> = Vec::new();
    apply_sdf_force_fields(
        std::slice::from_ref(&combo_attract),
        &colliders,
        &mut no_bodies,
        dt2,
    );
    println!(
        "[sdf_force] apply_sdf_force_fields(no bodies): len={}",
        no_bodies.len()
    );
    assert_eq!(no_bodies.len(), 0);

    // ------------------------------------------------------------------
    // 7. Degenerate: a body exactly on the surface (dist == 0) -> attract
    //    force is exactly zero regardless of which branch is taken.
    // ------------------------------------------------------------------
    let body_on_surface = RigidBody::new(Vec3Fix::from_int(1, 0, 0), Fix128::ONE);
    let f_on_surface = compute_sdf_force(&body_on_surface, &sphere, &attract.force_type);
    println!(
        "[sdf_force] attract() exactly on surface (dist=0): force={f_on_surface} (expect zero)"
    );
    assert_eq!(f_on_surface, Vec3Fix::ZERO);

    // ------------------------------------------------------------------
    // 8a. Degenerate: a large-but-not-extreme distance still clamps
    //     correctly (strength(3) * dist(1e9) = 3e9, far below Fix128's
    //     representable range of ~9.22e18, so no overflow occurs).
    // ------------------------------------------------------------------
    let large_pos = Vec3Fix::new(Fix128::from_int(1_000_000_000), Fix128::ZERO, Fix128::ZERO);
    let body_large = RigidBody::new(large_pos, Fix128::ONE);
    let large_attract = panic::catch_unwind(AssertUnwindSafe(|| {
        compute_sdf_force(&body_large, &sphere, &attract.force_type)
    }));
    println!("[sdf_force] large position (dist~1e9), attract(): {large_attract:?} (expect clamped to max_force=30)");
    assert!(large_attract.is_ok());
    assert_eq!(large_attract.unwrap(), Vec3Fix::from_int(-30, 0, 0));

    // ------------------------------------------------------------------
    // 8b. Degenerate: extreme coordinates do not panic, and the clamp still
    //     holds. The computation stays finite because Fix128's own
    //     representable range (~9.22e18) keeps the sum-of-squares used by
    //     the sphere's distance formula well under f32::MAX (~3.4e38), so no
    //     NaN/Inf is ever produced.
    //
    //     At this magnitude (x ~ 2^62 ~ 4.6e18) `strength * dist` = 3 * 4.6e18
    //     ~ 1.4e19 exceeds Fix128's range. `SdfForceType::Attract` is
    //     `force = -normal * min(strength*|dist|, max_force)` with the product
    //     taken by `checked_mul` (`compute_sdf_force`): an overflowing product
    //     is certainly larger than max_force, so the clamp value is used
    //     instead of a wrapped one. The closed form is therefore min(1.4e19, 30) = 30 along
    //     -normal = -x.
    // ------------------------------------------------------------------
    let extreme_pos = Vec3Fix::new(
        Fix128::from_raw(i64::MAX / 2, u64::MAX),
        Fix128::ZERO,
        Fix128::ZERO,
    );
    let body_extreme = RigidBody::new(extreme_pos, Fix128::ONE);
    let extreme_attract = panic::catch_unwind(AssertUnwindSafe(|| {
        compute_sdf_force(&body_extreme, &sphere, &attract.force_type)
    }));
    let (lx, _, _) = extreme_pos.to_f32();
    let dist_f64 = f64::from(lx) - 1.0; // unit_sphere()'s eval_fn at (lx, 0, 0)
    assert!(
        3.0 * dist_f64 > 9.223_372_036_854_776e18, // 2^63, Fix128's integer range
        "the product must exceed Fix128's integer range for this case to test the overflow path"
    );
    let expect_extreme = Vec3Fix::from_int(-30, 0, 0);
    println!("[sdf_force] extreme position, attract(): {extreme_attract:?} (expect {expect_extreme}, clamped to max_force=30)");
    assert!(extreme_attract.is_ok());
    assert_eq!(extreme_attract.unwrap(), expect_extreme);

    let extreme_repel = panic::catch_unwind(AssertUnwindSafe(|| {
        compute_sdf_force(&body_extreme, &sphere, &repel.force_type)
    }));
    println!(
        "[sdf_force] extreme position, repel(): {extreme_repel:?} (expect zero, beyond range)"
    );
    assert!(extreme_repel.is_ok());
    assert_eq!(extreme_repel.unwrap(), Vec3Fix::ZERO);

    println!("[sdf_force] all production-entry checks passed");
}
