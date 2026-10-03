//! Oracles for the wiring of `ccd`'s time-of-impact surface: `needs_ccd`,
//! `sphere_capsule_toi`, `capsule_plane_toi`, `aabb_plane_toi`,
//! `swept_aabb`, `speculative_contact`, `conservative_advancement`. All 7
//! had zero production callers before this crate's
//! `examples/continuous_collision_detection.rs`
//! (`scripts/wiring-baseline.txt` `unwired src/ccd.rs::*`).
//!
//! `adaptive_toi_substeps` (the 8th item in that baseline list) is
//! deliberately **not** covered here: it is a documented skeleton
//! ("Skeleton -- the TOI-derived scaling policy is scheduled for the
//! follow-up commit") that does not yet call its own documented
//! dependency (`solver_tgs::adaptive_substeps_for_ccd`). A separate task
//! is implementing it for real and will add its own oracle once that
//! lands.
//!
//! # Fix128 is binary fixed point (I64F64) -- a load-bearing constraint
//!
//! Every closed form below is chosen so that every intermediate division
//! reduces to a power-of-two denominator. `Fix128::from_ratio`/`Div` work
//! in base 2, so `16/40 = 2/5` is NOT exactly representable (0.4 is a
//! repeating binary fraction) even though both operands are small
//! integers, while `16/32 = 1/2` is exact. Oracles that need bit-exact
//! `assert_eq!` therefore pick magnitudes with this in mind; oracles that
//! only need an inequality or an `Option` variant do not, since rounding
//! of a few ULP cannot flip a comparison with a wide margin.
//!
//! # Closed forms (every expected value is derived here, none from the
//! crate)
//!
//! * **`needs_ccd`**: `velocity.length() * dt > radius.half() &&
//!   velocity.length() > config.velocity_threshold` (both conditions
//!   ANDed, both strict `>`).
//! * **`sphere_capsule_toi`**: projects `sphere_center` onto the capsule
//!   segment `cap_a..cap_b` **at t=0** (`t_param = (sphere_center -
//!   cap_a)·ab / |ab|^2`, clamped to `[0,1]`, `cap_a` when `ab` is zero
//!   length), then calls `sphere_sphere_toi` against that *fixed* closest
//!   point treated as a static sphere of `cap_radius`. This is an
//!   approximation (the closest point is not re-projected as the sphere
//!   moves), not an exact swept capsule TOI -- this file's oracle matches
//!   that documented approximation, not a "true" capsule TOI.
//! * **`capsule_plane_toi`**: picks whichever endpoint (`cap_a` or
//!   `cap_b`) is nearer the plane **at t=0** (`dist_a < dist_b` picks
//!   `a`, otherwise -- including ties -- `b`), then calls
//!   `sphere_plane_toi` on that one endpoint. Also an approximation: the
//!   *other* endpoint's motion is ignored even though it may cross the
//!   plane first for some velocities.
//! * **`aabb_plane_toi`**: picks the AABB's support vertex opposite the
//!   plane normal per-axis (`plane_normal.<axis> < 0` picks `max.<axis>`,
//!   otherwise `min.<axis>`) -- unlike `sphere_plane_toi`, there is no
//!   "side" flip, so the returned normal is always exactly the input
//!   `plane_normal`.
//! * **`swept_aabb`**: per-axis slab test (`slab_test`), combining
//!   per-axis `(t_enter, t_exit)` via `t_enter = max(...)`, `t_exit =
//!   min(...)`; a zero-velocity axis is "static" and either rejects (no
//!   overlap) or contributes an unrestricted `(-huge, +huge)` interval.
//! * **`speculative_contact`**: `gap = |pos_b - pos_a| - (r_a + r_b)`;
//!   `gap <= 0` is an immediate contact with `depth = -gap`; otherwise
//!   `closing_speed = -(vel_b - vel_a)·normal`, and a breach is predicted
//!   (`depth = -(gap - closing_speed*dt)`) only when `closing_speed > 0`
//!   AND `gap - closing_speed*dt < 0` (strict).
//! * **`conservative_advancement`**: iterates `pos = start + displacement
//!   times t`; stops with a hit once `distance_fn(pos).0 - radius <=
//!   config.tolerance`; otherwise advances `t` by `gap / |displacement|`;
//!   returns `None` if `t` exceeds `1` or `displacement` is zero-length.
//!   Structurally, if `config.max_iterations == 0` the loop body never
//!   runs at all.
//!
//! # Degenerate inputs (documented result, not merely "no panic")
//!
//! * **`needs_ccd`**: zero velocity / zero radius / `dt == 0` all produce
//!   the boolean the formula dictates; extreme-magnitude velocity must
//!   not panic (`Fix128` arithmetic is documented wrapping, mod 2^128).
//! * **`sphere_capsule_toi`**: a collapsed capsule (`cap_a == cap_b`)
//!   reduces exactly to a sphere; zero radii on both shapes still find
//!   the coincidence instant, and the resulting zero-length `rel_pos`
//!   normalizes to `Vec3Fix::ZERO` rather than panicking (`normalize`'s
//!   documented zero-length contract); the segment-parameter clamp can
//!   produce a geometrically real miss.
//! * **`capsule_plane_toi`**: zero velocity while clear of the plane
//!   misses (via the same early-return as `sphere_plane_toi`); zero
//!   radius still hits; already-penetrating reports `t=0`.
//! * **`aabb_plane_toi`**: zero velocity while clear misses; a zero-size
//!   (degenerate point) AABB reduces to point-vs-plane; already-
//!   penetrating reports `t=0`.
//! * **`swept_aabb`**: zero velocity with overlapping boxes is
//!   `Some(ZERO)`; zero velocity with disjoint boxes is `None`.
//! * **`speculative_contact`**: coincident centers (`dist.is_zero()`)
//!   always return `None`; separating pairs (`closing_speed <= 0`) never
//!   produce a contact regardless of `dt`; `dt == 0` on a closing-but-
//!   clear pair never predicts a breach; the zero-combined-radius
//!   boundary (`predicted_gap == 0` exactly) is a strict miss, not a hit.
//! * **`conservative_advancement`**: zero agent radius still converges;
//!   a displacement too short to close the gap within `t<=1` misses;
//!   `max_iterations == 0` misses even when already touching (the loop
//!   body never executes).

#![cfg(feature = "std")]

use alice_physics::ccd::{
    aabb_plane_toi, capsule_plane_toi, conservative_advancement, needs_ccd, speculative_contact,
    sphere_capsule_toi, swept_aabb, CcdConfig,
};
use alice_physics::collider::AABB;
use alice_physics::math::{Fix128, Vec3Fix};

fn fi(n: i64) -> Fix128 {
    Fix128::from_int(n)
}

fn r(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn v3i(x: i64, y: i64, z: i64) -> Vec3Fix {
    Vec3Fix::from_int(x, y, z)
}

// ---------------------------------------------------------------------
// needs_ccd
// ---------------------------------------------------------------------

#[test]
fn needs_ccd_requires_both_displacement_and_speed_strictly_over_threshold() {
    let cfg = CcdConfig::default(); // velocity_threshold = 5
                                    // speed 10 (6,8,0), dt 1/2 -> displacement 5. radius 4 -> half 2.
                                    // 5 > 2 and 10 > 5 -> true.
    assert!(needs_ccd(v3i(6, 8, 0), fi(4), r(1, 2), &cfg));
    // Same speed, dt 1/10 -> displacement 1 <= 2 -> false even though the
    // speed test alone would pass.
    assert!(!needs_ccd(v3i(6, 8, 0), fi(4), r(1, 10), &cfg));
    // speed exactly 5 (3,4,0): strict `>` against the threshold fails
    // regardless of displacement (radius 0 -> half 0, displacement 5*1=5>0).
    assert!(!needs_ccd(v3i(3, 4, 0), Fix128::ZERO, fi(1), &cfg));
    // speed one unit over the threshold with the same zero radius -> true.
    assert!(needs_ccd(v3i(0, 0, 6), Fix128::ZERO, fi(1), &cfg));
    // Displacement EXACTLY at radius/2 (speed 8, radius 1 -> half 1/2,
    // dt 1/16 -> displacement 8/16=1/2): the contract is strict `>`, so a
    // displacement equal to the boundary must NOT need CCD even though
    // speed (8) is over the threshold.
    assert!(!needs_ccd(v3i(8, 0, 0), fi(1), r(1, 16), &cfg));
    // One step past that boundary (displacement 8/8=1 > 1/2) needs CCD.
    assert!(needs_ccd(v3i(8, 0, 0), fi(1), r(1, 8), &cfg));
}

#[test]
fn needs_ccd_degenerate_inputs() {
    let cfg = CcdConfig::default();
    // Zero velocity: speed 0 is never > any nonnegative threshold.
    assert!(!needs_ccd(Vec3Fix::ZERO, fi(10), fi(10), &cfg));
    // Zero radius: half is 0, so any nonzero displacement from a
    // fast body clears it.
    assert!(needs_ccd(v3i(6, 8, 0), Fix128::ZERO, r(1, 1000), &cfg));
    // dt == 0: displacement is 0 regardless of speed.
    assert!(!needs_ccd(v3i(100, 0, 0), Fix128::ZERO, Fix128::ZERO, &cfg));
    // Extreme magnitude must not panic: Fix128 Mul/Add are documented
    // wrapping (mod 2^128), so an overflowing speed*dt must still return
    // a plain bool, not unwind.
    let extreme = Vec3Fix::new(
        Fix128::from_raw(i64::MAX, u64::MAX),
        Fix128::ZERO,
        Fix128::ZERO,
    );
    let result = std::panic::catch_unwind(|| needs_ccd(extreme, fi(1), fi(1), &cfg));
    assert!(result.is_ok(), "extreme-magnitude velocity must not panic");
}

// ---------------------------------------------------------------------
// sphere_capsule_toi
// ---------------------------------------------------------------------

#[test]
fn sphere_capsule_toi_direct_approach_matches_closed_form() {
    // sphere (0,3,0) r1 vel(16,0,0); capsule (10,-7,0)-(10,13,0) r1.
    // ab=(0,20,0), ab_len_sq=400. (sphere-cap_a)=(-10,10,0), dot ab=200,
    // t_param=200/400=1/2 (unclamped) -> closest=(10,3,0) (same y as the
    // sphere, so this reduces to a 1-D TOI along x).
    // sphere_sphere_toi(center=(0,3,0) r1 vel(16,0,0), closest=(10,3,0) r1
    // static): rel_pos=(10,0,0), rel_vel=(-16,0,0), combined_r=2.
    // a=256, b=2*(10*-16)=-320, c=100-4=96.
    // disc=320^2-4*256*96=102400-98304=4096, sqrt=64.
    // t=(320-64)/512=256/512=1/2. pos_a=(8,3,0), pos_b=(10,3,0),
    // normal=(1,0,0), point=(9,3,0).
    let toi = sphere_capsule_toi(
        v3i(0, 3, 0),
        fi(1),
        v3i(16, 0, 0),
        v3i(10, -7, 0),
        v3i(10, 13, 0),
        fi(1),
    )
    .expect("direct approach must hit");
    assert_eq!(toi.t, r(1, 2));
    assert_eq!(toi.normal, v3i(1, 0, 0));
    assert_eq!(toi.point, v3i(9, 3, 0));
}

#[test]
fn sphere_capsule_toi_clamps_to_segment_endpoint() {
    // Raw segment parameter is negative (sphere is far below cap_a along
    // the axis), so the closest point clamps to cap_a. With that clamp,
    // the vertical offset (8) exceeds the combined radius (2) and the
    // horizontal-only velocity never closes it:
    // discriminant = 160^2 - 4*64*160 = 25600-40960 = -15360 < 0 -> miss
    // (exact integers throughout; no sqrt needed since the result is a
    // miss).
    assert!(sphere_capsule_toi(
        v3i(0, -13, 0),
        fi(1),
        v3i(8, 0, 0),
        v3i(10, -5, 0),
        v3i(10, 5, 0),
        fi(1)
    )
    .is_none());

    // Control: without the clamp mattering (sphere level with the clamped
    // endpoint), the same geometry hits head-on. sphere (0,-5,0) r1
    // vel(8,0,0); same capsule; t_param = ((-10,0,0)·(0,10,0))/100 = 0 ->
    // closest=cap_a=(10,-5,0) exactly (boundary of the clamp, not
    // negative). rel_pos=(10,0,0), combined_r=2, a=64,b=2*(10*-8)=-160,
    // c=100-4=96. disc=160^2-4*64*96=25600-24576=1024,sqrt=32.
    // t=(160-32)/128=128/128=1.
    let boundary = sphere_capsule_toi(
        v3i(0, -5, 0),
        fi(1),
        v3i(8, 0, 0),
        v3i(10, -5, 0),
        v3i(10, 5, 0),
        fi(1),
    )
    .expect("head-on at the clamp boundary must hit at t=1");
    assert_eq!(boundary.t, Fix128::ONE);
}

#[test]
fn sphere_capsule_toi_clamps_to_upper_segment_endpoint() {
    // The upper clamp (`t > ONE -> ONE`) is a separate branch from the
    // lower one tested above; this exercises it specifically. cap_a=
    // (0,0,0), cap_b=(10,0,0) (ab=(10,0,0), ab_len_sq=100). sphere_center
    // =(25,0,0) is well past cap_b along the segment's own axis:
    // (sphere_center-cap_a)·ab = 25*10=250, t_param=250/100=5/2>1 ->
    // clamps to cap_b=(10,0,0) exactly (NOT the unclamped projection,
    // which would be further out).
    // sphere_sphere_toi(center=(25,0,0) r1 vel(-16,0,0), closest=
    // (10,0,0) r1 static): rel_pos=(-15,0,0), rel_vel=(16,0,0),
    // combined_r=2. a=256, b=2*(-15*16)=-480, c=225-4=221.
    // disc=480^2-4*256*221=230400-226304=4096, sqrt=64.
    // t=(480-64)/512=416/512=13/16 (512 is a power of two; 416/512
    // reduces to 13/16, exact binary).
    // pos_a=(25,0,0)+(-16,0,0)*(13/16)=(12,0,0), pos_b=(10,0,0),
    // normal=(-1,0,0), point=(12,0,0)+(-1,0,0)=(11,0,0).
    let toi = sphere_capsule_toi(
        v3i(25, 0, 0),
        fi(1),
        v3i(-16, 0, 0),
        v3i(0, 0, 0),
        v3i(10, 0, 0),
        fi(1),
    )
    .expect("approach from past cap_b, clamped to cap_b, must still hit");
    assert_eq!(toi.t, r(13, 16));
    assert_eq!(toi.normal, v3i(-1, 0, 0));
    assert_eq!(toi.point, v3i(11, 0, 0));
}

#[test]
fn sphere_capsule_toi_degenerate_inputs() {
    // Collapsed capsule (cap_a == cap_b) reduces to a sphere. closest =
    // cap_a = (5,0,0) unconditionally (ab_len_sq == 0 branch).
    // sphere_sphere_toi(center=(-5,0,0) r1 vel(16,0,0), (5,0,0) r1
    // static): rel_pos=(10,0,0), rel_vel=(-16,0,0), combined_r=2.
    // a=256,b=-320,c=96. disc=4096,sqrt=64. t=256/512=1/2.
    // pos_a=(3,0,0), pos_b=(5,0,0), normal=(1,0,0), point=(4,0,0).
    let collapsed = sphere_capsule_toi(
        v3i(-5, 0, 0),
        fi(1),
        v3i(16, 0, 0),
        v3i(5, 0, 0),
        v3i(5, 0, 0),
        fi(1),
    )
    .expect("collapsed capsule must behave exactly like a sphere");
    assert_eq!(collapsed.t, r(1, 2));
    assert_eq!(collapsed.point, v3i(4, 0, 0));

    // Zero radii on both shapes: combined_r=0 means contact only happens
    // when centers exactly coincide, so rel_pos is zero at that instant
    // and `normal = rel_pos.normalize()` documented-returns Vec3Fix::ZERO
    // rather than panicking; point reduces to pos_a exactly.
    // closest for sphere(0,0,0) vel(16,0,0) vs capsule (10,-5,0)-(10,5,0):
    // ab=(0,10,0), (sphere-cap_a)=(-10,5,0), dot=50, t=1/2 -> closest=(10,0,0).
    // sphere_sphere_toi(r0 vel(16,0,0), r0 static): a=256,b=-320,c=100.
    // disc=102400-102400=0,sqrt=0. t=320/512=5/8. pos_a=(10,0,0)=pos_b.
    let zero_radius = sphere_capsule_toi(
        Vec3Fix::ZERO,
        Fix128::ZERO,
        v3i(16, 0, 0),
        v3i(10, -5, 0),
        v3i(10, 5, 0),
        Fix128::ZERO,
    )
    .expect("zero-radius pair must still find the coincidence instant");
    assert_eq!(zero_radius.t, r(5, 8));
    assert_eq!(zero_radius.point, v3i(10, 0, 0));
    assert_eq!(
        zero_radius.normal,
        Vec3Fix::ZERO,
        "zero-length rel_pos must normalize to ZERO, not panic"
    );

    // Extreme magnitude must not panic.
    let extreme = std::panic::catch_unwind(|| {
        sphere_capsule_toi(
            Vec3Fix::new(
                Fix128::from_raw(i64::MIN / 2, 0),
                Fix128::ZERO,
                Fix128::ZERO,
            ),
            fi(1),
            v3i(8, 0, 0),
            Vec3Fix::new(
                Fix128::from_raw(i64::MAX / 2, 0),
                Fix128::ZERO,
                Fix128::ZERO,
            ),
            v3i(10, 5, 0),
            fi(1),
        )
    });
    assert!(
        extreme.is_ok(),
        "extreme-magnitude endpoints must not panic"
    );
}

// ---------------------------------------------------------------------
// capsule_plane_toi
// ---------------------------------------------------------------------

#[test]
fn capsule_plane_toi_picks_nearer_endpoint_and_tie_breaks_to_b() {
    // plane y=3 (normal (0,1,0), offset 3). cap_a=(0,20,0) dist 17,
    // cap_b=(0,10,0) dist 7 -> b is nearer -> center=(0,10,0).
    // sphere_plane_toi(center=(0,10,0) r2 vel(0,-16,0), plane y=3):
    // dist=7, side=+1. t=(2-7)/(-16)=5/16.
    // point=(0,10,0)+(0,-5,0)-(0,2,0)=(0,3,0).
    let near_b = capsule_plane_toi(
        v3i(0, 20, 0),
        v3i(0, 10, 0),
        fi(2),
        v3i(0, -16, 0),
        v3i(0, 1, 0),
        fi(3),
    )
    .expect("endpoint b (nearer) must drive the TOI");
    assert_eq!(near_b.t, r(5, 16));
    assert_eq!(near_b.point, v3i(0, 3, 0));
    assert_eq!(near_b.normal, v3i(0, 1, 0));

    // Swap which endpoint is nearer: cap_a=(0,10,0) now nearer than
    // cap_b=(0,20,0) -> a drives it, same t and point by symmetry.
    let near_a = capsule_plane_toi(
        v3i(0, 10, 0),
        v3i(0, 20, 0),
        fi(2),
        v3i(0, -16, 0),
        v3i(0, 1, 0),
        fi(3),
    )
    .expect("endpoint a (nearer) must drive the TOI");
    assert_eq!(near_a.t, r(5, 16));
    assert_eq!(near_a.point, v3i(0, 3, 0));

    // Tie-break: cap_a=(0,13,-5), cap_b=(0,13,5) are equidistant from
    // plane y=0 -> `dist_a < dist_b` is false -> b (the `else` branch).
    // sphere_plane_toi(center=(0,13,5) r1 vel(0,-16,0), plane y=0):
    // dist=13, t=(1-13)/(-16)=12/16=3/4.
    // point=(0,13,5)+(0,-12,0)-(0,1,0)=(0,0,5).
    let tie = capsule_plane_toi(
        v3i(0, 13, -5),
        v3i(0, 13, 5),
        fi(1),
        v3i(0, -16, 0),
        v3i(0, 1, 0),
        Fix128::ZERO,
    )
    .expect("tied endpoints must pick b");
    assert_eq!(tie.t, r(3, 4));
    assert_eq!(tie.point, v3i(0, 0, 5));
}

#[test]
fn capsule_plane_toi_degenerate_inputs() {
    // Zero velocity while clear of the plane: the early-return
    // `vel_toward >= 0 && dist > radius` fires with vel_toward == 0.
    assert!(capsule_plane_toi(
        v3i(0, 20, 0),
        v3i(0, 15, 0),
        fi(1),
        Vec3Fix::ZERO,
        v3i(0, 1, 0),
        Fix128::ZERO
    )
    .is_none());

    // Zero radius reduces correctly to point-vs-plane. cap_a=(0,10,0)
    // (dist10), cap_b=(0,4,0) (dist4) -> b nearer. sphere_plane_toi(r0,
    // center=(0,4,0), vel(0,-8,0), plane y=0): dist=4, t=(0-4)/(-8)=1/2.
    // point=(0,4,0)+(0,-4,0)=(0,0,0).
    let zero_r = capsule_plane_toi(
        v3i(0, 10, 0),
        v3i(0, 4, 0),
        Fix128::ZERO,
        v3i(0, -8, 0),
        v3i(0, 1, 0),
        Fix128::ZERO,
    )
    .expect("zero-radius capsule must still hit");
    assert_eq!(zero_r.t, r(1, 2));
    assert_eq!(zero_r.point, Vec3Fix::ZERO);

    // Already penetrating at t=0: cap_b=(0,1,0) dist 1 <= radius 2.
    let pen = capsule_plane_toi(
        v3i(0, 10, 0),
        v3i(0, 1, 0),
        fi(2),
        v3i(0, -8, 0),
        v3i(0, 1, 0),
        Fix128::ZERO,
    )
    .expect("already-penetrating capsule must report t=0");
    assert_eq!(pen.t, Fix128::ZERO);
    assert_eq!(pen.point, Vec3Fix::ZERO);

    // Moving away from the plane -> None.
    assert!(capsule_plane_toi(
        v3i(0, 10, 0),
        v3i(0, 15, 0),
        fi(1),
        v3i(0, 8, 0),
        v3i(0, 1, 0),
        Fix128::ZERO
    )
    .is_none());

    // Extreme magnitude must not panic.
    let extreme = std::panic::catch_unwind(|| {
        capsule_plane_toi(
            Vec3Fix::new(fi(0), Fix128::from_raw(i64::MAX / 2, 0), fi(0)),
            Vec3Fix::new(fi(0), Fix128::from_raw(i64::MAX, u64::MAX), fi(0)),
            fi(1),
            v3i(0, -8, 0),
            v3i(0, 1, 0),
            Fix128::ZERO,
        )
    });
    assert!(extreme.is_ok(), "extreme-magnitude capsule must not panic");
}

// ---------------------------------------------------------------------
// aabb_plane_toi
// ---------------------------------------------------------------------

#[test]
fn aabb_plane_toi_support_vertex_follows_normal_sign() {
    // AABB [-2,-2,-2]..[6,6,6], plane y=-10 (normal (0,1,0) offset -10).
    // normal.y=1 is not < 0 -> support.y = min.y = -2 (same for x,z, whose
    // normal components are 0, also not < 0 -> min).
    // support=(-2,-2,-2). dist=-2-(-10)=8. vel=(0,-32,0), vel_toward=-32.
    // t=-8/-32=1/4. point=(-2,-2,-2)+(0,-8,0)=(-2,-10,-2).
    let aabb1 = AABB::new(v3i(-2, -2, -2), v3i(6, 6, 6));
    let hit1 = aabb_plane_toi(&aabb1, v3i(0, -32, 0), v3i(0, 1, 0), fi(-10)).expect("must hit");
    assert_eq!(hit1.t, r(1, 4));
    assert_eq!(hit1.point, v3i(-2, -10, -2));
    assert_eq!(hit1.normal, v3i(0, 1, 0));

    // Negative-normal-component branch: plane x=10 with normal (-1,0,0),
    // offset -10. normal.x=-1 < 0 -> support.x = max.x = 5 (y,z use min).
    // AABB [-3,-3,-3]..[5,5,5] -> support=(5,-3,-3).
    // dist=5*(-1)-(-10)=5. vel=(8,0,0), vel_toward=8*(-1)=-8.
    // t=-5/-8=5/8. point=(5,-3,-3)+(5,0,0)=(10,-3,-3).
    let aabb2 = AABB::new(v3i(-3, -3, -3), v3i(5, 5, 5));
    let hit2 = aabb_plane_toi(&aabb2, v3i(8, 0, 0), v3i(-1, 0, 0), fi(-10)).expect("must hit");
    assert_eq!(hit2.t, r(5, 8));
    assert_eq!(hit2.point, v3i(10, -3, -3));

    // Zero velocity while clear of the plane -> None.
    assert!(aabb_plane_toi(&aabb1, Vec3Fix::ZERO, v3i(0, 1, 0), fi(-10)).is_none());
}

#[test]
fn aabb_plane_toi_degenerate_inputs() {
    // Zero-size AABB (a degenerate point) reduces to point-vs-plane:
    // support = the point itself regardless of normal sign (min==max).
    // dist=10-0=10, vel_toward=-20, t=10/20=1/2.
    // point=(5,10,5)+(0,-10,0)=(5,0,5).
    let point_box = AABB::new(v3i(5, 10, 5), v3i(5, 10, 5));
    let pb =
        aabb_plane_toi(&point_box, v3i(0, -20, 0), v3i(0, 1, 0), Fix128::ZERO).expect("must hit");
    assert_eq!(pb.t, r(1, 2));
    assert_eq!(pb.point, v3i(5, 0, 5));

    // Already penetrating: support.y = -5 <= plane y=0 -> t=0.
    let pen_box = AABB::new(v3i(-1, -5, -1), v3i(1, 5, 1));
    let pen = aabb_plane_toi(&pen_box, v3i(0, 8, 0), v3i(0, 1, 0), Fix128::ZERO)
        .expect("must report t=0");
    assert_eq!(pen.t, Fix128::ZERO);
    assert_eq!(pen.point, v3i(-1, -5, -1));

    // Moving away.
    let clear_box = AABB::new(v3i(1, 1, 1), v3i(3, 3, 3));
    assert!(aabb_plane_toi(&clear_box, v3i(0, 8, 0), v3i(0, 1, 0), Fix128::ZERO).is_none());

    // Extreme magnitude must not panic.
    let extreme_box = AABB::new(
        v3i(-1, -1, -1),
        Vec3Fix::new(Fix128::from_raw(i64::MAX, u64::MAX), fi(1), fi(1)),
    );
    let extreme = std::panic::catch_unwind(|| {
        aabb_plane_toi(&extreme_box, v3i(1, 0, 0), v3i(-1, 0, 0), fi(-10))
    });
    assert!(extreme.is_ok(), "extreme-magnitude AABB must not panic");
}

// ---------------------------------------------------------------------
// swept_aabb
// ---------------------------------------------------------------------

#[test]
fn swept_aabb_combines_per_axis_slab_times() {
    // moving [-2,2]^3, target [10,14]x[10,14]x[-2,2], velocity (16,16,0).
    // x: t0=(10-2)/16=1/2, t1=(14+2)/16=1. y: identical numbers.
    // z: static axis, ranges [-2,2] vs [-2,2] overlap -> unrestricted
    // interval. Combined: t_enter=max(1/2,1/2,-huge)=1/2,
    // t_exit=min(1,1,huge)=1 -> Some(1/2).
    let moving = AABB::new(v3i(-2, -2, -2), v3i(2, 2, 2));
    let target = AABB::new(v3i(10, 10, -2), v3i(14, 14, 2));
    assert_eq!(swept_aabb(&moving, v3i(16, 16, 0), &target), Some(r(1, 2)));

    // Same boxes, velocity too slow to arrive within the timestep: at
    // velocity (4,4,0), t_enter = max((8)/4, (8)/4) = 2 > 1 -> None.
    assert_eq!(swept_aabb(&moving, v3i(4, 4, 0), &target), None);

    // Reversed velocity never reaches the target -> None.
    assert_eq!(swept_aabb(&moving, v3i(-16, -16, 0), &target), None);
}

#[test]
fn swept_aabb_every_active_axis_must_contribute_its_own_entry_time() {
    // A scene where the X axis's own entry time is strictly the largest
    // (dominant) of the two active axes -- the combination logic must
    // fold EVERY axis's `te` into `t_enter` via `max`, not just whichever
    // axis happens to run last. moving [-1,1]^3, velocity (8,8,0).
    // x: target [9,11] -> te=(9-1)/8=1, tx=(11+1)/8=3/2.
    // y: target [2,4] -> te=(2-1)/8=1/8, tx=(4+1)/8=5/8.
    // z: target [-1,1] (same as moving) -> static overlap, unrestricted.
    // Correct combine: t_enter=max(1,1/8)=1, t_exit=min(3/2,5/8)=5/8.
    // t_enter(1) > t_exit(5/8) -> the box exits on the y-slab before it
    // has finished entering on the x-slab -> a genuine miss (None).
    // (If the x axis's `te` were dropped from the `max`, t_enter would
    // wrongly stay at y's 1/8 and this would wrongly report Some(1/8).)
    let moving = AABB::new(v3i(-1, -1, -1), v3i(1, 1, 1));
    let target = AABB::new(v3i(9, 2, -1), v3i(11, 4, 1));
    assert_eq!(swept_aabb(&moving, v3i(8, 8, 0), &target), None);
}

#[test]
fn swept_aabb_degenerate_inputs() {
    // Zero velocity, overlapping -> Some(ZERO).
    let overlap_a = AABB::new(v3i(0, 0, 0), v3i(4, 4, 4));
    let overlap_b = AABB::new(v3i(2, 2, 2), v3i(6, 6, 6));
    assert_eq!(
        swept_aabb(&overlap_a, Vec3Fix::ZERO, &overlap_b),
        Some(Fix128::ZERO)
    );

    // Zero velocity, disjoint -> None.
    let far_a = AABB::new(v3i(0, 0, 0), v3i(1, 1, 1));
    let far_b = AABB::new(v3i(5, 5, 5), v3i(6, 6, 6));
    assert_eq!(swept_aabb(&far_a, Vec3Fix::ZERO, &far_b), None);

    // Extreme-magnitude target must not panic.
    let moving = AABB::new(v3i(-2, -2, -2), v3i(2, 2, 2));
    let extreme_target = AABB::new(
        Vec3Fix::new(Fix128::from_raw(i64::MAX / 2, 0), fi(0), fi(0)),
        Vec3Fix::new(Fix128::from_raw(i64::MAX, 0), fi(1), fi(1)),
    );
    let extreme = std::panic::catch_unwind(|| swept_aabb(&moving, v3i(1, 0, 0), &extreme_target));
    assert!(
        extreme.is_ok(),
        "extreme-magnitude target AABB must not panic"
    );
}

// ---------------------------------------------------------------------
// speculative_contact
// ---------------------------------------------------------------------

#[test]
fn speculative_contact_breach_and_overlap_match_closed_form() {
    // Breach: A(0,0,0) vel(10,0,0) r2, B(20,0,0) static r3, dt=2.
    // rel_pos=(20,0,0), dist=20, combined_r=5, gap=15. normal=(1,0,0).
    // rel_vel=(-10,0,0), closing_speed=10. predicted=15-20=-5<0.
    // depth=5, point_a=(2,0,0), point_b=(17,0,0).
    let breach = speculative_contact(
        Vec3Fix::ZERO,
        v3i(10, 0, 0),
        fi(2),
        v3i(20, 0, 0),
        Vec3Fix::ZERO,
        fi(3),
        fi(2),
    )
    .expect("closing pair must breach within dt");
    assert_eq!(breach.depth, fi(5));
    assert_eq!(breach.point_a, v3i(2, 0, 0));
    assert_eq!(breach.point_b, v3i(17, 0, 0));
    assert_eq!(breach.normal, v3i(1, 0, 0));

    // Already overlapping: A(0,0,0) r4, B(5,0,0) r4. dist=5, combined=8,
    // gap=-3 -> depth=3 regardless of velocity/dt.
    // point_a=(4,0,0), point_b=(1,0,0).
    let overlap = speculative_contact(
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
        fi(4),
        v3i(5, 0, 0),
        Vec3Fix::ZERO,
        fi(4),
        fi(1),
    )
    .expect("overlapping pair must report a contact regardless of velocity");
    assert_eq!(overlap.depth, fi(3));
    assert_eq!(overlap.point_a, v3i(4, 0, 0));
    assert_eq!(overlap.point_b, v3i(1, 0, 0));

    // Not breaching: same breach setup but dt halved so the predicted gap
    // stays positive: predicted = 15 - 10*1 = 5, not < 0 -> None.
    assert!(speculative_contact(
        Vec3Fix::ZERO,
        v3i(10, 0, 0),
        fi(2),
        v3i(20, 0, 0),
        Vec3Fix::ZERO,
        fi(3),
        fi(1)
    )
    .is_none());
}

#[test]
fn speculative_contact_degenerate_inputs() {
    // Coincident centers: dist.is_zero() -> None unconditionally.
    assert!(speculative_contact(
        v3i(5, 5, 5),
        Vec3Fix::ZERO,
        fi(1),
        v3i(5, 5, 5),
        Vec3Fix::ZERO,
        fi(1),
        fi(1)
    )
    .is_none());

    // Separating pair: closing_speed <= 0 -> None even with a huge dt.
    assert!(speculative_contact(
        Vec3Fix::ZERO,
        v3i(-10, 0, 0),
        fi(1),
        v3i(20, 0, 0),
        Vec3Fix::ZERO,
        fi(1),
        fi(100)
    )
    .is_none());

    // dt == 0 with a positive gap: predicted_gap == gap > 0 -> None.
    assert!(speculative_contact(
        Vec3Fix::ZERO,
        v3i(5, 0, 0),
        fi(1),
        v3i(10, 0, 0),
        Vec3Fix::ZERO,
        fi(1),
        Fix128::ZERO
    )
    .is_none());

    // Zero combined radius, exact boundary: dist=8 (sqrt(64)=8 exact),
    // gap=8, closing=8, dt=1 -> predicted=0. The strict `< ZERO` check
    // means an exact-zero predicted gap is still a miss.
    assert!(speculative_contact(
        Vec3Fix::ZERO,
        v3i(8, 0, 0),
        Fix128::ZERO,
        v3i(8, 0, 0),
        Vec3Fix::ZERO,
        Fix128::ZERO,
        fi(1)
    )
    .is_none());

    // Extreme magnitude must not panic.
    let extreme = std::panic::catch_unwind(|| {
        speculative_contact(
            Vec3Fix::new(Fix128::from_raw(i64::MIN / 2, 0), fi(0), fi(0)),
            v3i(1, 0, 0),
            fi(1),
            Vec3Fix::new(Fix128::from_raw(i64::MAX / 2, 0), fi(0), fi(0)),
            Vec3Fix::ZERO,
            fi(1),
            fi(1),
        )
    });
    assert!(
        extreme.is_ok(),
        "extreme-magnitude positions must not panic"
    );
}

// ---------------------------------------------------------------------
// conservative_advancement
// ---------------------------------------------------------------------

/// Sphere obstacle centered off-origin so the generic closure path (not
/// just surrounding arithmetic) is actually exercised, distinct from this
/// crate's own plane-based `#[cfg(test)]` oracles for this function.
fn sphere_obstacle(center: Vec3Fix, radius: Fix128) -> impl Fn(Vec3Fix) -> (Fix128, Vec3Fix) {
    move |p: Vec3Fix| {
        let to_center = p - center;
        (to_center.length() - radius, to_center.normalize())
    }
}

#[test]
fn conservative_advancement_converges_on_a_colinear_sphere_obstacle() {
    let obstacle = sphere_obstacle(v3i(20, 0, 0), fi(3));
    let cfg = CcdConfig::default();

    // start (0,0,0), displacement (32,0,0), agent radius 1. Motion is
    // exactly colinear with the vector to the obstacle's center. 32 is a
    // power of two so `gap / speed` reduces exactly:
    //   iter0: dist=|{-20,0,0}|-3=17, gap=16>tolerance, speed=32,
    //          dt=16/32=1/2, t=1/2.
    //   iter1: pos=(16,0,0), dist=|{-4,0,0}|-3=1, gap=0<=tolerance ->
    //          Some(t=1/2, point=(16,0,0)+(1,0,0)=(17,0,0), normal=(-1,0,0)).
    let hit = conservative_advancement(Vec3Fix::ZERO, v3i(32, 0, 0), fi(1), &obstacle, &cfg)
        .expect("colinear approach must converge to a hit");
    assert_eq!(hit.t, r(1, 2));
    assert_eq!(hit.point, v3i(17, 0, 0));
    assert_eq!(hit.normal, v3i(-1, 0, 0));

    // t landing EXACTLY on 1 after the first advance must still be
    // treated as in-range (`t > ONE` is false at t == ONE) and continue
    // to the next iteration, not stop early. displacement (16,0,0):
    //   iter0: dist=17, gap=16, speed=16, dt=16/16=1, t=1 (not > 1, loop
    //          continues).
    //   iter1: pos=(16,0,0), dist=1, gap=0<=tolerance ->
    //          Some(t=1, point=(16,0,0)+(1,0,0)=(17,0,0)).
    let t_equals_one =
        conservative_advancement(Vec3Fix::ZERO, v3i(16, 0, 0), fi(1), &obstacle, &cfg)
            .expect("t landing exactly on 1 must still be followed by the confirming iteration");
    assert_eq!(t_equals_one.t, Fix128::ONE);
    assert_eq!(t_equals_one.point, v3i(17, 0, 0));

    // gap EXACTLY equal to config.tolerance must count as a hit (`<=`,
    // not `<`). Use a position-independent distance function so that if
    // the `<=` were weakened to `<`, the loop would never converge
    // (gap stays unchanged every iteration) and would instead exhaust
    // max_iterations and return None.
    let tol = r(1, 1000); // bit-identical to CcdConfig::default().tolerance
    let const_normal = v3i(1, 0, 0);
    let const_radius = fi(2);
    let const_dist_fn = move |_p: Vec3Fix| (tol + const_radius, const_normal);
    let at_tolerance = conservative_advancement(
        Vec3Fix::ZERO,
        v3i(5, 0, 0),
        const_radius,
        const_dist_fn,
        &cfg,
    )
    .expect("gap exactly at the tolerance boundary must be an immediate hit");
    assert_eq!(at_tolerance.t, Fix128::ZERO);
}

#[test]
fn conservative_advancement_degenerate_inputs() {
    let obstacle = sphere_obstacle(v3i(20, 0, 0), fi(3));
    let cfg = CcdConfig::default();

    // Zero agent radius: same colinear reasoning without the radius
    // offset. iter0: dist=17, gap=17, dt=17/32 (exact, 32 is a power of
    // two). iter1: pos=(17,0,0), dist=0, gap=0<=tolerance ->
    // Some(t=17/32, point=(17,0,0)).
    let zero_radius =
        conservative_advancement(Vec3Fix::ZERO, v3i(32, 0, 0), Fix128::ZERO, &obstacle, &cfg)
            .expect("zero agent radius must still converge");
    assert_eq!(zero_radius.t, r(17, 32));
    assert_eq!(zero_radius.point, v3i(17, 0, 0));

    // Displacement too short to ever close the gap within t<=1:
    // iter0: dist=17, gap=16, speed=10, dt=16/10=1.6>1 -> None.
    assert!(
        conservative_advancement(Vec3Fix::ZERO, v3i(10, 0, 0), fi(1), &obstacle, &cfg).is_none()
    );

    // Zero displacement, starting clear of the surface: the loop reaches
    // `speed.is_zero()` and returns None (gap 16 > tolerance first).
    assert!(
        conservative_advancement(Vec3Fix::ZERO, Vec3Fix::ZERO, fi(1), &obstacle, &cfg).is_none()
    );

    // max_iterations == 0: the loop body never runs at all, so the result
    // is None even starting exactly on the surface (where gap would be
    // <= tolerance on the very first -- unrun -- iteration).
    let zero_iters = CcdConfig {
        max_iterations: 0,
        ..cfg
    };
    assert!(
        conservative_advancement(
            v3i(17, 0, 0),
            v3i(1, 0, 0),
            Fix128::ZERO,
            &obstacle,
            &zero_iters
        )
        .is_none(),
        "max_iterations == 0 must return None even when already touching"
    );

    // Extreme-magnitude displacement and distance_fn output must not panic.
    let extreme_displacement = Vec3Fix::new(Fix128::from_raw(i64::MAX, u64::MAX), fi(0), fi(0));
    let extreme_distance_fn = |_p: Vec3Fix| (Fix128::from_raw(i64::MAX, u64::MAX), v3i(1, 0, 0));
    let extreme = std::panic::catch_unwind(|| {
        conservative_advancement(
            Vec3Fix::ZERO,
            extreme_displacement,
            fi(1),
            extreme_distance_fn,
            &cfg,
        )
    });
    assert!(
        extreme.is_ok(),
        "extreme-magnitude displacement/distance must not panic"
    );
}
