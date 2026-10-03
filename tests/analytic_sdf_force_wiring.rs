//! Closed-form oracle for `sdf_force`'s public surface: the four named
//! constructors (`attract` / `repel` / `contain` / `surface_flow`), the
//! `with_affected_bodies` scope filter, and the two driver functions
//! `compute_sdf_force` / `apply_sdf_force_fields`.
//!
//! # Scenes
//!
//! Two fixed SDFs, both with exact (non-transcendental, where it matters)
//! local evaluation on the axis-aligned points used below:
//!
//! - `unit_sphere()`: `distance(p) = |p| - 1`, `normal(p) = p / |p|`
//!   (`+Y` fallback at the degenerate center, where `|p| < 1e-10`). On the
//!   `+X` axis the normal is exactly `(1, 0, 0)` and `|p|` is an exact f32
//!   value for every integer radius used here (`sqrt` of a perfect square).
//! - `ground_plane()`: `distance(x, y, z) = y`, constant normal `(0, 1, 0)`.
//!
//! # Oracle construction
//!
//! Every expected value below is built from the module's own documented
//! formula (see `src/sdf_force.rs` doc comments), reconstructed with
//! `Fix128` / `Vec3Fix` arithmetic **operators** (`+`, `-`, `*`, `/`,
//! `.abs()`, `.min()`) directly in this file — never by calling
//! `compute_sdf_force` or `apply_sdf_force_fields` to produce the expected
//! side of an assertion. Because `Fix128` arithmetic is itself deterministic
//! (documented `wrapping_*`/truncating contracts in `src/math.rs`), building
//! the formula from the same primitives the implementation uses gives a
//! bit-exact oracle without any epsilon tolerance, while still catching a
//! mutation to `compute_sdf_force`'s own wiring of those primitives (wrong
//! sign, swapped branch, dropped clamp, wrong operand order, ...).
#![cfg(feature = "std")]

use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::sdf_collider::{ClosureSdf, SdfCollider};
use alice_physics::sdf_force::{apply_sdf_force_fields, compute_sdf_force};
use alice_physics::{RigidBody, SdfForceField, SdfForceType};
use std::panic::{self, AssertUnwindSafe};

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

fn ground_plane() -> ClosureSdf {
    ClosureSdf::new(|_x, y, _z| y, |_x, _y, _z| (0.0, 1.0, 0.0))
}

fn sphere_collider() -> SdfCollider {
    SdfCollider::new_static(Box::new(unit_sphere()), Vec3Fix::ZERO, QuatFix::IDENTITY)
}

fn ground_collider() -> SdfCollider {
    SdfCollider::new_static(Box::new(ground_plane()), Vec3Fix::ZERO, QuatFix::IDENTITY)
}

fn body_at(x: i64, y: i64, z: i64) -> RigidBody {
    RigidBody::new(Vec3Fix::from_int(x, y, z), Fix128::ONE)
}

// ============================================================================
// attract(): force_mag = min(strength * |dist|, max_force); direction is
// -normal when outside (dist > 0), +normal when inside/on (dist <= 0).
// ============================================================================

#[test]
fn attract_linear_falloff_clamp_and_inside_push_outward() {
    let sphere = sphere_collider();
    let field = SdfForceField::attract(0, Fix128::from_int(3)); // max_force = 30

    // outside, unclamped: dist = 2 -> force_mag = min(6, 30) = 6
    let f = compute_sdf_force(&body_at(3, 0, 0), &sphere, &field.force_type);
    assert_eq!(f, Vec3Fix::from_int(-6, 0, 0));

    // outside, clamped: dist = 20 -> force_mag = min(60, 30) = 30
    let f_clamped = compute_sdf_force(&body_at(21, 0, 0), &sphere, &field.force_type);
    assert_eq!(f_clamped, Vec3Fix::from_int(-30, 0, 0));

    // inside: dist = -0.5 -> force_mag = min(1.5, 30) = 1.5, pushed outward
    // (+normal, since dist > 0.0 is false)
    let body_half = RigidBody::new(Vec3Fix::from_f32(0.5, 0.0, 0.0), Fix128::ONE);
    let f_inside = compute_sdf_force(&body_half, &sphere, &field.force_type);
    let expect_inside = Vec3Fix::UNIT_X * (Fix128::from_int(3) * Fix128::from_ratio(1, 2));
    assert_eq!(f_inside, expect_inside);
}

/// Degenerate: exactly on the surface (dist == 0), force is exactly zero
/// regardless of which branch (`dist > 0.0` is false, taking the "inside"
/// branch) is taken, because `force_mag = min(strength * 0, max_force) = 0`.
#[test]
fn attract_on_surface_is_exactly_zero() {
    let sphere = sphere_collider();
    let field = SdfForceField::attract(0, Fix128::from_int(5));
    let f = compute_sdf_force(&body_at(1, 0, 0), &sphere, &field.force_type);
    assert_eq!(f, Vec3Fix::ZERO);
}

// ============================================================================
// repel(): force_mag = strength * (1 - |dist|/range)^2, zero beyond range;
// direction is +normal outside, -normal inside.
// ============================================================================

#[test]
fn repel_quadratic_falloff_and_zero_beyond_range() {
    let sphere = sphere_collider();
    let field = SdfForceField::repel(0, Fix128::from_int(8), Fix128::from_int(4));

    // dist = 1, range = 4: factor = 1 - 1/4 = 3/4, mag = 8 * 9/16 = 4.5
    let f = compute_sdf_force(&body_at(2, 0, 0), &sphere, &field.force_type);
    let factor = Fix128::ONE - Fix128::ONE / Fix128::from_int(4);
    let expect = Fix128::from_int(8) * factor * factor;
    assert_eq!(f, Vec3Fix::new(expect, Fix128::ZERO, Fix128::ZERO));

    // exactly at the range boundary: factor = 0 -> force = 0 (not an
    // early-return, the formula itself zeroes out).
    let f_boundary = compute_sdf_force(&body_at(5, 0, 0), &sphere, &field.force_type); // dist = 4
    assert_eq!(f_boundary, Vec3Fix::ZERO);

    // degenerate: far beyond the field's range of effect -> the early
    // `return Vec3Fix::ZERO` branch.
    let f_far = compute_sdf_force(&body_at(100, 0, 0), &sphere, &field.force_type); // dist = 99
    assert_eq!(f_far, Vec3Fix::ZERO);
}

/// Degenerate: inside the sphere at its own center, the SDF's gradient is
/// undefined (`|p| < 1e-10`) and `unit_sphere()` documents a `+Y` fallback.
/// `repel` at `dist = -1` (center) still produces a finite force along that
/// fallback direction instead of propagating a NaN/undefined direction.
#[test]
fn repel_at_degenerate_normal_uses_documented_fallback_direction() {
    let sphere = sphere_collider();
    let field = SdfForceField::repel(0, Fix128::from_int(8), Fix128::from_int(4));
    let f = compute_sdf_force(&body_at(0, 0, 0), &sphere, &field.force_type); // dist = -1
    let factor = Fix128::ONE - Fix128::ONE / Fix128::from_int(4); // |dist|=1, range=4
    let mag = Fix128::from_int(8) * factor * factor;
    // inside (dist <= 0) -> -normal * mag; fallback normal is +Y.
    let expect = -Vec3Fix::UNIT_Y * mag;
    assert_eq!(f, expect);
    assert!(
        f.x.is_zero() && f.z.is_zero(),
        "no component leaks onto X/Z: {f:?}"
    );
}

/// Degenerate: `range == 0` exercises `Fix128::Div`'s documented
/// division-by-zero contract (returns `ZERO`, see `src/math.rs`), not a
/// panic. On the surface (`dist_abs == range == 0`) the early-return guard
/// `dist_abs > range` is false, so the formula proceeds with `factor = 1 -
/// 0/0 = 1 - 0 = 1`, i.e. full strength; one unit off the surface the same
/// guard is true, giving zero.
#[test]
fn repel_range_zero_hits_documented_division_by_zero_contract() {
    let sphere = sphere_collider();
    let field = SdfForceField::repel(0, Fix128::from_int(10), Fix128::ZERO);

    let on_surface = panic::catch_unwind(AssertUnwindSafe(|| {
        compute_sdf_force(&body_at(1, 0, 0), &sphere, &field.force_type)
    }));
    assert!(on_surface.is_ok(), "range=0 must not panic on the surface");
    // dist == 0.0 -> "inside" branch (`dist > 0.0` is false) -> -normal * strength
    assert_eq!(on_surface.unwrap(), Vec3Fix::from_int(-10, 0, 0));

    let off_surface = panic::catch_unwind(AssertUnwindSafe(|| {
        compute_sdf_force(&body_at(2, 0, 0), &sphere, &field.force_type)
    }));
    assert!(
        off_surface.is_ok(),
        "range=0 must not panic off the surface"
    );
    assert_eq!(off_surface.unwrap(), Vec3Fix::ZERO);
}

// ============================================================================
// contain(): inside (dist <= 0) -> damping only; outside -> inward push
// (-normal * strength * dist) plus damping.
// ============================================================================

#[test]
fn contain_damping_inside_and_push_plus_damping_outside() {
    let ground = ground_collider();
    let field = SdfForceField::contain(0, Fix128::from_int(10)); // damping = 1/10
    let damping = Fix128::from_ratio(1, 10);

    let mut inside = RigidBody::new(Vec3Fix::from_int(0, -2, 0), Fix128::ONE);
    inside.velocity = Vec3Fix::from_int(8, 0, 0);
    let f_in = compute_sdf_force(&inside, &ground, &field.force_type);
    assert_eq!(f_in, inside.velocity * (-damping));

    let mut outside = RigidBody::new(Vec3Fix::from_int(0, 3, 0), Fix128::ONE);
    outside.velocity = Vec3Fix::from_int(8, 0, 0);
    let f_out = compute_sdf_force(&outside, &ground, &field.force_type);
    let push = -Vec3Fix::UNIT_Y * (Fix128::from_int(10) * Fix128::from_int(3));
    let damp = outside.velocity * (-damping);
    assert_eq!(f_out, push + damp);
}

/// Degenerate: exactly on the boundary (`dist == 0`) takes the `dist <= 0.0`
/// branch (damping-only), not the outside push branch — a boundary that is
/// easy to get backwards with a strict `<` mutation.
#[test]
fn contain_boundary_takes_the_inside_damping_only_branch() {
    let ground = ground_collider();
    let field = SdfForceField::contain(0, Fix128::from_int(10));
    let damping = Fix128::from_ratio(1, 10);
    let mut boundary = RigidBody::new(Vec3Fix::ZERO, Fix128::ONE);
    boundary.velocity = Vec3Fix::from_int(8, 0, 0);
    let f = compute_sdf_force(&boundary, &ground, &field.force_type);
    assert_eq!(f, boundary.velocity * (-damping));
}

// ============================================================================
// surface_flow(): tangential projection of flow_direction onto the surface
// tangent plane, scaled by linear falloff, zero beyond influence_distance
// or when the projection is zero.
// ============================================================================

#[test]
fn surface_flow_tangential_projection_with_linear_falloff() {
    let sphere = sphere_collider();
    // influence_distance is fixed at 2 by the surface_flow() constructor.
    let field = SdfForceField::surface_flow(0, Vec3Fix::from_int(0, 0, 8), Fix128::from_int(5));

    // dist = 1: falloff = 1 - 1/2 = 1/2 -> force = (0,0,1) * (5 * 1/2) = (0,0,2.5)
    let f = compute_sdf_force(&body_at(2, 0, 0), &sphere, &field.force_type);
    let falloff = Fix128::ONE - Fix128::ONE / Fix128::from_int(2);
    let expect = Vec3Fix::from_int(0, 0, 1) * (Fix128::from_int(5) * falloff);
    assert_eq!(f, expect);

    // on the surface: falloff = 1 -> force = (0,0,5)
    let f_on = compute_sdf_force(&body_at(1, 0, 0), &sphere, &field.force_type);
    assert_eq!(f_on, Vec3Fix::from_int(0, 0, 5));

    // exactly at influence_distance (dist == 2): falloff = 1 - 2/2 = 0.
    let f_edge = compute_sdf_force(&body_at(3, 0, 0), &sphere, &field.force_type);
    assert_eq!(f_edge, Vec3Fix::ZERO);

    // degenerate: beyond influence_distance -> the early `return ZERO`.
    let f_far = compute_sdf_force(&body_at(10, 0, 0), &sphere, &field.force_type); // dist = 9
    assert_eq!(f_far, Vec3Fix::ZERO);
}

/// Degenerate: `flow_direction` parallel to the surface normal has zero
/// tangential component (`tangent.length().is_zero()`), so the force is
/// zero even though the body is well within `influence_distance`.
#[test]
fn surface_flow_parallel_to_normal_has_no_tangential_component() {
    let sphere = sphere_collider();
    let field = SdfForceField::surface_flow(0, Vec3Fix::from_int(5, 0, 0), Fix128::from_int(5));
    let f = compute_sdf_force(&body_at(2, 0, 0), &sphere, &field.force_type); // dist = 1
    assert_eq!(f, Vec3Fix::ZERO);
}

/// Degenerate: `influence_distance == 0` hits the same documented
/// division-by-zero contract as `repel_range_zero_...` above: on the
/// surface the guard `dist_abs > 0` is false, so `falloff = 1 - 0/0 = 1`
/// (full strength); one unit off the surface it returns zero.
#[test]
fn surface_flow_influence_distance_zero_hits_documented_division_by_zero_contract() {
    let sphere = sphere_collider();
    // surface_flow() always fixes influence_distance = 2, so build the
    // SurfaceFlow variant directly to get influence_distance = 0 (the
    // constructor itself is exercised by other tests and the example).
    let zero_influence = SdfForceType::SurfaceFlow {
        flow_direction: Vec3Fix::from_int(0, 0, 6),
        strength: Fix128::from_int(6),
        influence_distance: Fix128::ZERO,
    };

    let on_surface = panic::catch_unwind(AssertUnwindSafe(|| {
        compute_sdf_force(&body_at(1, 0, 0), &sphere, &zero_influence)
    }));
    assert!(on_surface.is_ok());
    assert_eq!(on_surface.unwrap(), Vec3Fix::from_int(0, 0, 6));

    let off_surface = panic::catch_unwind(AssertUnwindSafe(|| {
        compute_sdf_force(&body_at(2, 0, 0), &sphere, &zero_influence)
    }));
    assert!(off_surface.is_ok());
    assert_eq!(off_surface.unwrap(), Vec3Fix::ZERO);
}

// ============================================================================
// with_affected_bodies(): `None` (default) affects every body index;
// `Some(vec![])` is a documented no-op distinct from `None`; `Some(list)`
// restricts to exactly those indices.
// ============================================================================

#[test]
fn with_affected_bodies_none_vs_empty_vs_subset() {
    let field = SdfForceField::new(
        0,
        SdfForceType::Contain {
            strength: Fix128::from_int(4),
            damping: Fix128::ZERO,
        },
    );
    let dt = Fix128::ONE;
    let expect_push = -Vec3Fix::UNIT_Y * (Fix128::from_int(4) * Fix128::from_int(3)); // dist = 3

    let make_body = || {
        let mut b = RigidBody::new_dynamic(Vec3Fix::from_int(0, 3, 0), Fix128::ONE);
        b.velocity = Vec3Fix::ZERO;
        b
    };

    // default (None): affects every body.
    let mut all = vec![make_body(), make_body()];
    apply_sdf_force_fields(
        std::slice::from_ref(&field),
        &[ground_collider()],
        &mut all,
        dt,
    );
    assert_eq!(all[0].velocity, expect_push);
    assert_eq!(all[1].velocity, expect_push);

    // Some(vec![]): documented no-op, distinct from None.
    let empty = field.clone().with_affected_bodies(vec![]);
    let mut none_affected = vec![make_body(), make_body()];
    apply_sdf_force_fields(&[empty], &[ground_collider()], &mut none_affected, dt);
    assert_eq!(none_affected[0].velocity, Vec3Fix::ZERO);
    assert_eq!(none_affected[1].velocity, Vec3Fix::ZERO);

    // Some(vec![1]): only body index 1.
    let scoped = field.with_affected_bodies(vec![1]);
    let mut subset = vec![make_body(), make_body()];
    apply_sdf_force_fields(&[scoped], &[ground_collider()], &mut subset, dt);
    assert_eq!(subset[0].velocity, Vec3Fix::ZERO);
    assert_eq!(subset[1].velocity, expect_push);
}

/// Degenerate: `with_affected_bodies` restricts by *index*, not by presence
/// — an index beyond the bodies slice's length is simply never matched
/// (`enumerate()` never reaches it), not an error.
#[test]
fn with_affected_bodies_index_beyond_body_count_is_simply_unreachable() {
    let field = SdfForceField::new(
        0,
        SdfForceType::Contain {
            strength: Fix128::from_int(4),
            damping: Fix128::ZERO,
        },
    )
    .with_affected_bodies(vec![5, 6, 7]);
    let mut bodies = vec![RigidBody::new_dynamic(
        Vec3Fix::from_int(0, 3, 0),
        Fix128::ONE,
    )];
    let before = bodies[0].velocity;
    apply_sdf_force_fields(&[field], &[ground_collider()], &mut bodies, Fix128::ONE);
    assert_eq!(
        bodies[0].velocity, before,
        "no in-range body is in the filter list"
    );
}

// ============================================================================
// apply_sdf_force_fields(): sums every affecting field per body, skips
// static bodies entirely, integrates F = m*a, v += a*dt.
// ============================================================================

#[test]
fn apply_sdf_force_fields_sums_fields_skips_static_and_integrates() {
    let colliders = [sphere_collider()];
    let attract = SdfForceField::attract(0, Fix128::from_int(3)); // max_force = 30
    let repel = SdfForceField::repel(0, Fix128::from_int(8), Fix128::from_int(4));
    let dt = Fix128::from_ratio(1, 4);

    let mut dynamic = RigidBody::new_dynamic(Vec3Fix::from_int(3, 0, 0), Fix128::ONE); // dist = 2
    dynamic.inv_mass = Fix128::from_int(2);
    dynamic.velocity = Vec3Fix::from_int(5, 0, 0);
    let mut stationary = RigidBody::new_static(Vec3Fix::from_int(3, 0, 0));
    stationary.velocity = Vec3Fix::from_int(5, 0, 0);
    let mut bodies = vec![dynamic, stationary];

    apply_sdf_force_fields(
        &[attract.clone(), repel.clone()],
        &colliders,
        &mut bodies,
        dt,
    );

    // attract: dist=2 -> force_mag=min(6,30)=6 -> (-6,0,0)
    // repel: dist=2, range=4 -> factor=1-2/4=1/2 -> mag=8*1/4=2 -> (2,0,0)
    // total = (-4,0,0); accel = total * inv_mass(2) = (-8,0,0); dv = accel*dt(1/4) = (-2,0,0)
    assert_eq!(bodies[0].velocity, Vec3Fix::from_int(3, 0, 0));
    // static body is skipped by `body.is_static()` and keeps its velocity.
    assert_eq!(bodies[1].velocity, Vec3Fix::from_int(5, 0, 0));
}

/// Degenerate: an out-of-range `sdf_index` is skipped (not a panic/index
/// error) — `field.sdf_index >= sdf_colliders.len()` guards every lookup.
#[test]
fn apply_sdf_force_fields_dangling_sdf_index_is_skipped_not_panicking() {
    let colliders = [sphere_collider()];
    let dangling = SdfForceField::attract(5, Fix128::from_int(3));
    let mut bodies = vec![RigidBody::new_dynamic(
        Vec3Fix::from_int(3, 0, 0),
        Fix128::ONE,
    )];
    let before = bodies[0].velocity;
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        apply_sdf_force_fields(&[dangling], &colliders, &mut bodies, Fix128::ONE);
    }));
    assert!(result.is_ok());
    assert_eq!(bodies[0].velocity, before);
}

/// Degenerate: an empty fields slice and an empty bodies slice are both
/// no-ops — the production loop simply has nothing to iterate over either
/// axis.
#[test]
fn apply_sdf_force_fields_empty_fields_and_empty_bodies_are_no_ops() {
    let colliders = [sphere_collider()];
    let attract = SdfForceField::attract(0, Fix128::from_int(3));

    let mut one_body = vec![RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE)];
    apply_sdf_force_fields(&[], &colliders, &mut one_body, Fix128::ONE);
    assert_eq!(one_body[0].velocity, Vec3Fix::ZERO);

    let mut no_bodies: Vec<RigidBody> = Vec::new();
    apply_sdf_force_fields(&[attract], &colliders, &mut no_bodies, Fix128::ONE);
    assert_eq!(no_bodies.len(), 0);
}

// ============================================================================
// Extreme magnitude: structural robustness + a real numeric finding.
// ============================================================================

/// Degenerate: a large-but-not-extreme distance clamps correctly —
/// `strength(3) * dist(~1e9) = 3e9` stays far below Fix128's representable
/// range (~9.22e18), so `min()` picks the true `max_force`.
#[test]
fn attract_large_distance_still_clamps_correctly() {
    let sphere = sphere_collider();
    let field = SdfForceField::attract(0, Fix128::from_int(3)); // max_force = 30
    let body = RigidBody::new(Vec3Fix::from_int(1_000_000_000, 0, 0), Fix128::ONE);
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        compute_sdf_force(&body, &sphere, &field.force_type)
    }));
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), Vec3Fix::from_int(-30, 0, 0));
}

/// Degenerate + finding: at `dist ~ 4.6e18` (half of Fix128's representable
/// magnitude), `strength(3) * dist_fix` itself overflows Fix128's range and
/// wraps (`Mul`'s documented mod-2^128 contract, `src/math.rs`), so
/// `min(wrapped, max_force)` is not guaranteed to pick `max_force`. This
/// does **not** panic (Fix128 has no NaN/trap representation — wrapping
/// arithmetic always returns *some* in-range value), but the clamp
/// invariant that holds at ordinary magnitudes (see
/// `attract_large_distance_still_clamps_correctly` above) does not hold
/// here. Pinned bit-exact via the same public `Fix128` primitives the
/// implementation uses, not asserted away with an epsilon.
///
/// Not fixed by this wiring pass (would require saturating/checked
/// arithmetic in `compute_sdf_force` or in `Fix128::Mul` itself — a design
/// change, out of scope here); reported to the parent as a real fact about
/// existing code.
#[test]
fn attract_overflow_defeats_clamp_at_extreme_magnitude_documented_not_fixed() {
    let sphere = sphere_collider();
    let field = SdfForceField::attract(0, Fix128::from_int(3)); // max_force = 30
    let extreme_pos = Vec3Fix::new(
        Fix128::from_raw(i64::MAX / 2, u64::MAX),
        Fix128::ZERO,
        Fix128::ZERO,
    );
    let body = RigidBody::new(extreme_pos, Fix128::ONE);

    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        compute_sdf_force(&body, &sphere, &field.force_type)
    }));
    assert!(
        result.is_ok(),
        "Fix128 wrapping arithmetic must never panic"
    );

    let (lx, _, _) = extreme_pos.to_f32();
    let dist_f32 = lx - 1.0; // same formula as unit_sphere()'s eval_fn at (lx, 0, 0)
    let dist_fix = Fix128::from_f32(dist_f32);
    let wrapped_force_mag = (Fix128::from_int(3) * dist_fix.abs()).min(Fix128::from_int(30));
    let expect = -Vec3Fix::UNIT_X * wrapped_force_mag;
    assert_eq!(result.unwrap(), expect);
    // The point of this test: the wrapped value is NOT the intended clamp.
    assert_ne!(wrapped_force_mag, Fix128::from_int(30), "if this ever equals max_force, the overflow finding is stale and this test should be revisited");
}

/// Degenerate: at the same extreme magnitude, `repel`'s early
/// `dist_abs > range` guard does not itself multiply anything, so it is
/// unaffected by the overflow above and correctly returns zero.
#[test]
fn repel_at_extreme_magnitude_still_returns_zero_beyond_range() {
    let sphere = sphere_collider();
    let field = SdfForceField::repel(0, Fix128::from_int(1), Fix128::from_int(1000));
    let extreme_pos = Vec3Fix::new(
        Fix128::from_raw(i64::MAX / 2, u64::MAX),
        Fix128::ZERO,
        Fix128::ZERO,
    );
    let body = RigidBody::new(extreme_pos, Fix128::ONE);
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        compute_sdf_force(&body, &sphere, &field.force_type)
    }));
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), Vec3Fix::ZERO);
}
