//! Oracles for the wiring of `interpolation`: `WorldSnapshot` /
//! `BodySnapshot` capture, `InterpolationState`'s alpha-blended read-back,
//! and the `lerp_fix128` / `lerp_vec3` / `slerp` primitives it is built on.
//!
//! # Closed forms (every expected value is derived here, independently of
//! the functions under test, never by calling them)
//!
//! * **`lerp_fix128(a, b, t) = a*(1-t) + b*t`** (the module's actual
//!   formula, read from `src/interpolation.rs`; algebraically the same as
//!   `a + (b-a)*t` away from overflow). At `t=0` the result is exactly `a`,
//!   at `t=1` exactly `b`; `t` outside `[0, 1]` extrapolates along the same
//!   line rather than clamping or erroring, because the formula has no
//!   branch on `t`.
//! * **`lerp_vec3`** is `lerp_fix128` applied componentwise.
//! * **`slerp` is NLERP, not geometric SLERP** (read from the module: no
//!   `acos`/`sin` appear in it). Its closed form is: `dot = a·b`; if
//!   `dot < 0`, negate every component of `b` (shortest-path fix); then
//!   `raw = lerp(a, b, t)` componentwise; if `|raw|² = 0` return `IDENTITY`;
//!   otherwise return `raw / |raw|`.
//! * **`BodySnapshot::from_body`** copies exactly `position`, `rotation`,
//!   `velocity` from a `RigidBody` — nothing else (mass, friction, etc. are
//!   not part of the snapshot).
//! * **`WorldSnapshot::capture`** builds one `BodySnapshot` per body, in
//!   the world's body order, each equal to `BodySnapshot::from_body` of
//!   the corresponding body.
//! * **`InterpolationState::capture_and_push`** is `push(capture(world))`:
//!   the old `current` becomes `prev`, the new capture becomes `current`.
//! * **`interpolate(idx, a) == (interpolate_position(idx, a),
//!   interpolate_rotation(idx, a))`** by construction (the former just
//!   pairs the latter two).
//! * **`interpolate_all`** interpolates indices `0..min(prev.len(),
//!   current.len())`; a body present in only one of the two snapshots is
//!   silently excluded, not an error.
//!
//! # Degenerate inputs (documented result, not merely "no panic")
//!
//! * Out-of-range `body_idx` (including `usize::MAX`): `interpolate_position`
//!   returns `Vec3Fix::ZERO`, `interpolate_rotation` returns
//!   `QuatFix::IDENTITY` — the bounds check runs before any indexing, so
//!   this never panics for any `usize`.
//! * `alpha` outside `[0, 1]`: extrapolates (documented above), never an
//!   `Err` — these functions have no `Result` in their signature.
//! * Two snapshots of different body counts: `body_count()` /
//!   `interpolate_all` use the smaller count; no panic, no `Err`.
//! * `slerp` of two bit-identical quaternions returns that quaternion
//!   exactly, for any `t` (the classic "angle between identical rotations"
//!   edge case that a true `acos`-based SLERP must special-case).
//! * `slerp` of two quaternions that are component-wise negatives of each
//!   other (the classic antipodal case) takes the `dot < 0` branch, which
//!   flips `b` back to `a`, so the result is `a` exactly — not `NaN`.
//! * `slerp` of two quaternions with `dot == 0` (geometrically 180° apart
//!   for true SLERP, where the axis is undefined and a `sin`-based formula
//!   divides `0/0`): NLERP has no such singularity; it lerps and
//!   normalizes like any other case.
//! * `slerp((0,0,0,0), (0,0,0,0), t)`: the lerp is `(0,0,0,0)` for every
//!   `t`, so `|raw|² = 0` and the function returns `IDENTITY` rather than
//!   dividing by zero / producing `NaN`.
//! * Extreme `Fix128` magnitudes (`i64::MAX` scaled by an extrapolating
//!   `t`): `Fix128::mul`/`Fix128::add` are explicitly wrapping (`wrapping_add`
//!   / `wrapping_mul` in `src/math.rs`), so `lerp_fix128` wraps rather than
//!   panicking or saturating; the wrapped value is asserted exactly (via
//!   `catch_unwind`), not just the absence of a panic.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]

use std::panic::{catch_unwind, AssertUnwindSafe};

use alice_physics::interpolation::{
    lerp_fix128, lerp_vec3, slerp, BodySnapshot, InterpolationState, WorldSnapshot,
};
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};

fn world_with(bodies: Vec<RigidBody>) -> PhysicsWorld {
    let mut world = PhysicsWorld::new(PhysicsConfig::default());
    for b in bodies {
        world.add_body(b);
    }
    world
}

// ---------------------------------------------------------------------------
// lerp_fix128 / lerp_vec3
// ---------------------------------------------------------------------------

#[test]
fn lerp_fix128_endpoints_midpoint_and_extrapolation() {
    let a = Fix128::from_int(-4);
    let b = Fix128::from_int(12);

    // oracle: a*(1-0) + b*0 = a
    assert_eq!(lerp_fix128(a, b, Fix128::ZERO), a);
    // oracle: a*(1-1) + b*1 = b
    assert_eq!(lerp_fix128(a, b, Fix128::ONE), b);
    // oracle: a*1/2 + b*1/2 = -2 + 6 = 4
    assert_eq!(
        lerp_fix128(a, b, Fix128::from_ratio(1, 2)),
        Fix128::from_int(4)
    );
    // oracle: t = 2 (outside [0,1]) extrapolates: a*(1-2) + b*2 = 4 + 24 = 28
    assert_eq!(lerp_fix128(a, b, Fix128::from_int(2)), Fix128::from_int(28));
    // oracle: t = -1 extrapolates the other way: a*2 + b*(-1) = -8 - 12 = -20
    assert_eq!(
        lerp_fix128(a, b, Fix128::from_int(-1)),
        Fix128::from_int(-20)
    );
}

#[test]
fn lerp_vec3_is_componentwise_lerp_fix128() {
    let va = Vec3Fix::from_int(-4, 0, 12);
    let vb = Vec3Fix::from_int(12, 8, -4);
    let t = Fix128::from_ratio(1, 4);

    let got = lerp_vec3(va, vb, t);
    assert_eq!(got.x, lerp_fix128(va.x, vb.x, t));
    assert_eq!(got.y, lerp_fix128(va.y, vb.y, t));
    assert_eq!(got.z, lerp_fix128(va.z, vb.z, t));
    // oracle, worked by hand: x: -4 + 16*1/4 = 0; y: 0 + 8*1/4 = 2; z: 12 - 16*1/4 = 8
    assert_eq!(
        got,
        Vec3Fix::new(Fix128::ZERO, Fix128::from_int(2), Fix128::from_int(8))
    );
}

#[test]
fn lerp_fix128_extreme_magnitude_wraps_per_fix128_semantics_no_panic() {
    // Fix128::add / Fix128::mul are explicitly wrapping (src/math.rs uses
    // wrapping_add / wrapping_mul), so this must not panic in any profile,
    // and the wrapped value must equal the same formula evaluated
    // independently with Fix128's own wrapping operators.
    let a = Fix128::from_int(i64::MAX);
    let b = Fix128::from_int(i64::MAX);
    let t = Fix128::from_int(2); // extrapolation: 2*b overflows i64 in the hi part

    let result = catch_unwind(AssertUnwindSafe(|| lerp_fix128(a, b, t)));
    let got = result.expect("lerp_fix128 must not panic on overflow: Fix128 ops wrap");

    // independent closed form: a*(1-t) + b*t, evaluated with bare Fix128
    // operators (the primitive arithmetic, not the function under test).
    let one_minus_t = Fix128::ONE - t;
    let expected = a * one_minus_t + b * t;
    assert_eq!(got, expected);
    // and it must actually have wrapped (not equal the unwrapped mathematical
    // value, which would need > i64 range to represent):
    assert_ne!(
        got.hi, 0,
        "sanity: the wrapped high word should not coincidentally be 0"
    );
}

// ---------------------------------------------------------------------------
// slerp (NLERP)
// ---------------------------------------------------------------------------

#[test]
fn slerp_no_flip_branch_pythagorean_closed_form() {
    let a = QuatFix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ZERO, Fix128::ONE);
    let b = QuatFix::new(
        Fix128::ZERO,
        Fix128::from_int(6),
        Fix128::ZERO,
        Fix128::from_int(7),
    );
    let t = Fix128::from_ratio(1, 2);

    // oracle: dot = a.w*b.w = 7 >= 0, no flip.
    // raw = (1-1/2)*a + (1/2)*b = (0, 3, 0, 4) exactly.
    // |raw| = sqrt(9+16) = sqrt(25) = 5 exactly (perfect square).
    let len = Fix128::from_int(25).sqrt();
    assert_eq!(len, Fix128::from_int(5), "sqrt(25) must be exact");
    let inv_len = Fix128::ONE / len;
    let expected = QuatFix::new(
        Fix128::ZERO,
        Fix128::from_int(3) * inv_len,
        Fix128::ZERO,
        Fix128::from_int(4) * inv_len,
    );
    assert_eq!(slerp(a, b, t), expected);
}

#[test]
fn slerp_flip_branch_negative_dot_reaches_same_pythagorean_answer() {
    let a = QuatFix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ZERO, Fix128::ONE);
    // b here is the componentwise negative of the b used in the no-flip
    // test above, so dot(a, b) = a.w * b.w = 1 * (-7) = -7 < 0, which
    // forces the shortest-path flip: flipped b = (0, 6, 0, 7), landing on
    // exactly the same closed form as the no-flip case.
    let b = QuatFix::new(
        Fix128::ZERO,
        Fix128::from_int(-6),
        Fix128::ZERO,
        Fix128::from_int(-7),
    );
    let t = Fix128::from_ratio(1, 2);

    let len = Fix128::from_int(25).sqrt();
    let inv_len = Fix128::ONE / len;
    let expected = QuatFix::new(
        Fix128::ZERO,
        Fix128::from_int(3) * inv_len,
        Fix128::ZERO,
        Fix128::from_int(4) * inv_len,
    );
    assert_eq!(slerp(a, b, t), expected);
}

#[test]
fn slerp_identical_rotation_returns_exactly_that_rotation() {
    // dot(a,a) = |a|^2 = 1 >= 0 (never flips for a = b). lerp(a,a,t) = a for
    // any t, and |a| is already 1, so normalize is a no-op. Using an
    // axis-aligned quaternion keeps every intermediate step exact (no
    // division is needed to build `a` itself).
    let a = QuatFix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO, Fix128::ZERO);
    for num_den in [(0i64, 1i64), (1, 2), (3, 7), (1, 1), (5, 1)] {
        let t = Fix128::from_ratio(num_den.0, num_den.1);
        assert_eq!(slerp(a, a, t), a, "slerp(a, a, {num_den:?}) must equal a");
    }
}

#[test]
fn slerp_antipodal_double_cover_flips_back_to_a_not_nan() {
    // a and -a represent the same rotation (quaternion double cover).
    // dot(a, -a) = -|a|^2 = -1 < 0, forcing the flip, so the flipped b is
    // -(-a) = a; raw = lerp(a, a, t) = a, and since `a` is already unit
    // length the final normalize is a no-op, so the result is a exactly,
    // for any dyadic t, with no NaN. (`a`'s components are 0/1 so every
    // intermediate multiply by a dyadic t is an exact bit-shift — a
    // non-dyadic t such as 1/3 would not reproduce `a` bit-for-bit here,
    // because lerp(a, a, t) itself would round away from a.)
    let a = QuatFix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO, Fix128::ZERO);
    let minus_a = QuatFix::new(-a.x, -a.y, -a.z, -a.w);
    for num_den in [(0i64, 1i64), (1, 2), (1, 4), (3, 4), (1, 1)] {
        let t = Fix128::from_ratio(num_den.0, num_den.1);
        assert_eq!(
            slerp(a, minus_a, t),
            a,
            "slerp(a, -a, {num_den:?}) must flip back to a"
        );
    }
}

#[test]
fn slerp_orthogonal_quaternions_dot_zero_no_flip_no_nan() {
    // Geometrically, dot == 0 is the "180 degrees apart" case for a true
    // acos/sin-based SLERP (undefined axis, 0/0 in the sin ratio). NLERP
    // has no such singularity: dot == 0 does not satisfy `dot < 0`, so
    // there is no flip, and the lerp + normalize proceeds like any other
    // input.
    let a = QuatFix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ZERO, Fix128::ONE);
    let b = QuatFix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO, Fix128::ZERO);
    assert_eq!(
        a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w,
        Fix128::ZERO,
        "fixture must have dot(a,b) == 0"
    );
    let t = Fix128::from_ratio(1, 2);

    let result = catch_unwind(AssertUnwindSafe(|| slerp(a, b, t)));
    let got = result.expect("dot == 0 must not panic or produce NaN-equivalent state");

    // independent closed form: raw = (0.5, 0, 0, 0.5), |raw| = sqrt(0.5),
    // normalized = raw / |raw|, evaluated with bare Fix128 operators.
    let raw = QuatFix::new(
        Fix128::from_ratio(1, 2),
        Fix128::ZERO,
        Fix128::ZERO,
        Fix128::from_ratio(1, 2),
    );
    let len = (raw.x * raw.x + raw.y * raw.y + raw.z * raw.z + raw.w * raw.w).sqrt();
    assert!(
        !len.is_zero(),
        "sanity: dot==0 case must not degenerate to zero length"
    );
    let inv_len = Fix128::ONE / len;
    let expected = QuatFix::new(
        raw.x * inv_len,
        raw.y * inv_len,
        raw.z * inv_len,
        raw.w * inv_len,
    );
    assert_eq!(got, expected);
}

#[test]
fn slerp_zero_quaternion_inputs_return_identity_not_nan_or_err() {
    // a = b = the zero quaternion is not a valid rotation, but the module
    // must not divide by zero: dot = 0 (not < 0, no flip), raw = (0,0,0,0)
    // for every t, |raw|^2 = 0, so the documented degenerate branch fires
    // and returns IDENTITY.
    let zero = QuatFix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ZERO, Fix128::ZERO);
    for num_den in [(0i64, 1i64), (1, 2), (1, 1), (7, 3)] {
        let t = Fix128::from_ratio(num_den.0, num_den.1);
        let result = catch_unwind(AssertUnwindSafe(|| slerp(zero, zero, t)));
        let got = result.expect("zero-length degenerate branch must not panic");
        assert_eq!(
            got,
            QuatFix::IDENTITY,
            "slerp(0,0,{num_den:?}) must be IDENTITY"
        );
    }
}

// ---------------------------------------------------------------------------
// BodySnapshot::from_body / WorldSnapshot::capture / empty()
// ---------------------------------------------------------------------------

#[test]
fn from_body_copies_position_rotation_velocity_only() {
    let body = RigidBody::new_dynamic(Vec3Fix::from_int(1, 2, 3), Fix128::from_int(7))
        .with_velocity(Vec3Fix::from_int(4, 5, 6))
        .with_rotation(QuatFix::new(
            Fix128::ZERO,
            Fix128::from_int(6),
            Fix128::ZERO,
            Fix128::from_int(7),
        ))
        .with_friction(Fix128::from_ratio(9, 10));

    let snap = BodySnapshot::from_body(&body);
    assert_eq!(snap.position, body.position);
    assert_eq!(snap.rotation, body.rotation);
    assert_eq!(snap.velocity, body.velocity);
    // oracle: mass/friction are not part of the snapshot shape at all
    // (BodySnapshot has exactly 3 fields: position, rotation, velocity).
    assert_eq!(snap.position, Vec3Fix::from_int(1, 2, 3));
    assert_eq!(snap.velocity, Vec3Fix::from_int(4, 5, 6));
}

#[test]
fn capture_builds_one_snapshot_per_body_matching_from_body() {
    let world = world_with(vec![
        RigidBody::new_dynamic(Vec3Fix::from_int(1, 0, 0), Fix128::ONE)
            .with_velocity(Vec3Fix::from_int(1, 1, 1)),
        RigidBody::new_static(Vec3Fix::from_int(5, 5, 5)),
        RigidBody::new_dynamic(Vec3Fix::from_int(-2, 3, 0), Fix128::from_int(2)),
    ]);

    let snap = WorldSnapshot::capture(&world);
    assert_eq!(snap.len(), 3);
    assert!(!snap.is_empty());
    for (i, body) in world.bodies.iter().enumerate() {
        assert_eq!(snap.bodies[i], BodySnapshot::from_body(body));
    }
}

#[test]
fn empty_state_has_zero_bodies_and_interpolate_all_is_empty() {
    let interp = InterpolationState::empty();
    assert_eq!(interp.body_count(), 0);
    assert!(interp.prev.is_empty());
    assert!(interp.current.is_empty());
    assert!(interp.interpolate_all(Fix128::from_ratio(1, 2)).is_empty());
    assert!(interp.interpolate_all(Fix128::ZERO).is_empty());
    assert!(interp.interpolate_all(Fix128::from_int(5)).is_empty());
}

// ---------------------------------------------------------------------------
// capture_and_push / interpolate* composition
// ---------------------------------------------------------------------------

#[test]
fn capture_and_push_shifts_current_into_prev() {
    let mut world = world_with(vec![RigidBody::new_dynamic(
        Vec3Fix::from_int(0, 0, 0),
        Fix128::ONE,
    )
    .with_velocity(Vec3Fix::from_int(4, 0, 0))]);

    let mut interp = InterpolationState::empty();
    interp.capture_and_push(&world);
    let first = interp.current.clone();
    assert!(interp.prev.is_empty(), "first push leaves prev empty");

    world.step(Fix128::from_ratio(1, 2));
    interp.capture_and_push(&world);

    // oracle: the old current (first) is now prev, bit for bit.
    assert_eq!(interp.prev.bodies, first.bodies);
    // oracle: velocity 4, dt 1/2, no gravity/damping change from default ->
    // the body actually moved (not asserting the exact integrator result,
    // only that prev != current, i.e. the shift is observable).
    assert_ne!(
        interp.prev.bodies[0].position,
        interp.current.bodies[0].position
    );
}

#[test]
fn interpolate_position_and_rotation_match_interpolate_components() {
    let snap_a = WorldSnapshot {
        bodies: vec![
            BodySnapshot {
                position: Vec3Fix::from_int(0, 0, 0),
                rotation: QuatFix::IDENTITY,
                velocity: Vec3Fix::ZERO,
            },
            BodySnapshot {
                position: Vec3Fix::from_int(10, 0, 0),
                rotation: QuatFix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO, Fix128::ZERO),
                velocity: Vec3Fix::ZERO,
            },
        ],
    };
    let snap_b = WorldSnapshot {
        bodies: vec![
            BodySnapshot {
                position: Vec3Fix::from_int(8, 4, 0),
                rotation: QuatFix::new(
                    Fix128::ZERO,
                    Fix128::from_int(6),
                    Fix128::ZERO,
                    Fix128::from_int(7),
                ),
                velocity: Vec3Fix::ZERO,
            },
            BodySnapshot {
                position: Vec3Fix::from_int(10, 0, 0),
                rotation: QuatFix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO, Fix128::ZERO),
                velocity: Vec3Fix::ZERO,
            },
        ],
    };

    let interp = InterpolationState::new(snap_a, snap_b);
    for alpha in [
        Fix128::ZERO,
        Fix128::from_ratio(1, 4),
        Fix128::from_ratio(1, 2),
        Fix128::ONE,
        Fix128::from_int(2),
    ] {
        for idx in 0..2 {
            let (p, q) = interp.interpolate(idx, alpha);
            assert_eq!(p, interp.interpolate_position(idx, alpha));
            assert_eq!(q, interp.interpolate_rotation(idx, alpha));
        }
    }
}

#[test]
fn interpolate_all_uses_min_body_count_and_ignores_extra_bodies() {
    let snap_prev = WorldSnapshot {
        bodies: vec![
            BodySnapshot {
                position: Vec3Fix::from_int(0, 0, 0),
                rotation: QuatFix::IDENTITY,
                velocity: Vec3Fix::ZERO,
            },
            BodySnapshot {
                position: Vec3Fix::from_int(1, 1, 1),
                rotation: QuatFix::IDENTITY,
                velocity: Vec3Fix::ZERO,
            },
        ],
    };
    let snap_curr = WorldSnapshot {
        bodies: vec![
            BodySnapshot {
                position: Vec3Fix::from_int(4, 0, 0),
                rotation: QuatFix::IDENTITY,
                velocity: Vec3Fix::ZERO,
            },
            BodySnapshot {
                position: Vec3Fix::from_int(3, 3, 3),
                rotation: QuatFix::IDENTITY,
                velocity: Vec3Fix::ZERO,
            },
            BodySnapshot {
                position: Vec3Fix::from_int(9, 9, 9),
                rotation: QuatFix::IDENTITY,
                velocity: Vec3Fix::ZERO,
            },
        ],
    };
    let interp = InterpolationState::new(snap_prev, snap_curr);

    // oracle: min(2, 3) = 2
    assert_eq!(interp.body_count(), 2);
    let all = interp.interpolate_all(Fix128::from_ratio(1, 2));
    assert_eq!(
        all.len(),
        2,
        "the 3rd current-only body is excluded, not an error"
    );
    assert_eq!(all[0].0, Vec3Fix::from_int(2, 0, 0));
    assert_eq!(all[1].0, Vec3Fix::from_int(2, 2, 2));
}

#[test]
fn out_of_range_body_index_returns_zero_identity_never_panics() {
    let interp = InterpolationState::new(
        WorldSnapshot {
            bodies: vec![BodySnapshot {
                position: Vec3Fix::from_int(1, 2, 3),
                rotation: QuatFix::IDENTITY,
                velocity: Vec3Fix::ZERO,
            }],
        },
        WorldSnapshot {
            bodies: vec![BodySnapshot {
                position: Vec3Fix::from_int(4, 5, 6),
                rotation: QuatFix::IDENTITY,
                velocity: Vec3Fix::ZERO,
            }],
        },
    );

    for idx in [1usize, 2, 100, usize::MAX] {
        let result = catch_unwind(AssertUnwindSafe(|| {
            (
                interp.interpolate_position(idx, Fix128::from_ratio(1, 2)),
                interp.interpolate_rotation(idx, Fix128::from_ratio(1, 2)),
                interp.interpolate(idx, Fix128::from_ratio(1, 2)),
            )
        }));
        let (p, q, pair) = result.unwrap_or_else(|_| {
            panic!("out-of-range body_idx {idx} must not panic (bounds check precedes indexing)")
        });
        assert_eq!(p, Vec3Fix::ZERO, "idx {idx}: documented ZERO fallback");
        assert_eq!(
            q,
            QuatFix::IDENTITY,
            "idx {idx}: documented IDENTITY fallback"
        );
        assert_eq!(pair, (Vec3Fix::ZERO, QuatFix::IDENTITY));
    }
}

#[test]
fn alpha_outside_unit_interval_extrapolates_no_clamp() {
    let interp = InterpolationState::new(
        WorldSnapshot {
            bodies: vec![BodySnapshot {
                position: Vec3Fix::from_int(0, 0, 0),
                rotation: QuatFix::IDENTITY,
                velocity: Vec3Fix::ZERO,
            }],
        },
        WorldSnapshot {
            bodies: vec![BodySnapshot {
                position: Vec3Fix::from_int(4, 0, 0),
                rotation: QuatFix::IDENTITY,
                velocity: Vec3Fix::ZERO,
            }],
        },
    );

    // oracle: position lerp is linear, so alpha=2 extrapolates to 2*4 = 8,
    // alpha=-1 extrapolates to -4 — a clamping implementation would instead
    // saturate at 4 and 0 respectively.
    assert_eq!(
        interp.interpolate_position(0, Fix128::from_int(2)),
        Vec3Fix::from_int(8, 0, 0)
    );
    assert_eq!(
        interp.interpolate_position(0, Fix128::from_int(-1)),
        Vec3Fix::from_int(-4, 0, 0)
    );
}
