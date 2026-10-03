//! Continuous collision detection (CCD): time-of-impact queries for the
//! primitive pairs the solver needs to prevent tunneling.
//!
//! Wiring: `needs_ccd`, `sphere_capsule_toi`, `capsule_plane_toi`,
//! `aabb_plane_toi`, `swept_aabb`, `speculative_contact`,
//! `conservative_advancement` had zero production callers
//! (`scripts/wiring-baseline.txt` `unwired src/ccd.rs::*`; 7 of the 8 items
//! listed there -- `sphere_sphere_toi`/`sphere_plane_toi` were already
//! wired elsewhere and are reused here only as the closed-form arithmetic
//! this file's oracles are built on top of -- every expected value below
//! is still hand-derived from the primitive types, not from calling the
//! function under test). This is their production entry point.
//!
//! `adaptive_toi_substeps` is **deliberately excluded**: it is a
//! documented skeleton ("Skeleton -- the TOI-derived scaling policy is
//! scheduled for the follow-up commit") that does not yet call
//! `solver_tgs::adaptive_substeps_for_ccd` as its own doc says it should.
//! A separate task is implementing it for real and will wire it (with its
//! own oracle) once that lands; wiring it here first would pin the current
//! stub behaviour as if it were correct.
//!
//! ```bash
//! cargo run --example continuous_collision_detection --features std
//! ```

use alice_physics::ccd::{
    aabb_plane_toi, capsule_plane_toi, conservative_advancement, needs_ccd, speculative_contact,
    sphere_capsule_toi, swept_aabb, CcdConfig,
};
use alice_physics::collider::AABB;
use alice_physics::math::{Fix128, Vec3Fix};
use std::panic::catch_unwind;

fn fi(n: i64) -> Fix128 {
    Fix128::from_int(n)
}

fn r(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn v3i(x: i64, y: i64, z: i64) -> Vec3Fix {
    Vec3Fix::from_int(x, y, z)
}

fn main() {
    // ==================================================================
    // needs_ccd: closed form `speed*dt > radius/2 && speed > threshold`.
    // ==================================================================
    let cfg = CcdConfig::default(); // velocity_threshold = 5

    // speed 10 (6,8,0 is a scaled 3-4-5 triangle), dt 1/2 -> displacement 5
    // > radius/2 (4/2=2), and 10 > 5 -> needs CCD.
    let fast = v3i(6, 8, 0);
    assert!(
        needs_ccd(fast, fi(4), r(1, 2), &cfg),
        "displacement 5 > 2 and speed 10 > 5 must need CCD"
    );
    // Same speed, tiny dt: displacement 1 <= 2 -> CCD not needed even though
    // the speed test alone would pass.
    assert!(
        !needs_ccd(fast, fi(4), r(1, 10), &cfg),
        "displacement 1 <= radius/2 2 must not need CCD"
    );
    // Speed exactly AT the threshold (5): strict `>` must fail regardless
    // of displacement (radius 0 -> half 0, so displacement 5*1=5 > 0).
    assert!(
        !needs_ccd(v3i(3, 4, 0), Fix128::ZERO, fi(1), &cfg),
        "speed == threshold (5) must not need CCD (strict >)"
    );
    // Degenerate: zero velocity.
    assert!(!needs_ccd(Vec3Fix::ZERO, fi(4), fi(1), &cfg));
    // Degenerate: zero radius with a fast body and tiny positive dt -- any
    // nonzero displacement exceeds radius/2 == 0.
    assert!(needs_ccd(fast, Fix128::ZERO, r(1, 1000), &cfg));
    // Degenerate: dt == 0 -> displacement 0, never needs CCD regardless of
    // speed.
    assert!(!needs_ccd(fast, Fix128::ZERO, Fix128::ZERO, &cfg));
    // Degenerate: extreme magnitude must not panic (Fix128 add/mul are
    // documented wrapping, mod 2^128 -- src/math.rs `impl Mul for Fix128`).
    let extreme_vel = Vec3Fix::new(
        Fix128::from_raw(i64::MAX, u64::MAX),
        Fix128::ZERO,
        Fix128::ZERO,
    );
    let extreme_result = catch_unwind(|| needs_ccd(extreme_vel, fi(1), fi(1), &cfg));
    assert!(
        extreme_result.is_ok(),
        "needs_ccd must not panic on extreme-magnitude velocity (wrapping arithmetic)"
    );
    println!("[ccd] needs_ccd: boundary + degenerate cases match closed form, extreme input did not panic");

    // ==================================================================
    // sphere_capsule_toi: closest point on a static capsule segment, then
    // sphere_sphere_toi against that fixed point.
    // ==================================================================
    // Direct approach: sphere (0,3,0) r1 vel (16,0,0); capsule
    // (10,-7,0)-(10,13,0) r1. ab=(0,20,0), ab_len_sq=400.
    // t = ((-10,10,0)-(10,-7,0))... (sphere-cap_a)=(-10,10,0), dot ab=200,
    // t=200/400=0.5 (unclamped) -> closest=(10,3,0) (same y as the sphere,
    // so this reduces to a clean 1-D TOI along x).
    // sphere_sphere_toi(center=(0,3,0) r1 vel(16,0,0), closest=(10,3,0) r1
    // static): rel_pos=(10,0,0), rel_vel=(-16,0,0), combined_r=2.
    // a=256, b=2*(10*-16)=-320, c=100-4=96.
    // disc = 320^2 - 4*256*96 = 102400 - 98304 = 4096, sqrt=64.
    // t = (320-64)/512 = 256/512 = 1/2 (exact).
    // pos_a=(0,3,0)+(16,0,0)*0.5=(8,3,0), pos_b=(10,3,0), normal=(1,0,0),
    // point=pos_a+normal*1=(9,3,0).
    let sc = sphere_capsule_toi(
        v3i(0, 3, 0),
        fi(1),
        v3i(16, 0, 0),
        v3i(10, -7, 0),
        v3i(10, 13, 0),
        fi(1),
    )
    .expect("direct approach to an off-axis capsule must hit");
    assert_eq!(sc.t, r(1, 2));
    assert_eq!(sc.normal, v3i(1, 0, 0));
    assert_eq!(sc.point, v3i(9, 3, 0));
    println!(
        "[ccd] sphere_capsule_toi direct approach: t={:?} point={:?}",
        sc.t, sc.point
    );

    // Clamp case: the raw segment parameter is negative (sphere is far
    // below cap_a along the axis), so the closest point clamps to cap_a
    // itself. With that clamp, the vertical offset (8) exceeds the
    // combined radius (2) and the horizontal-only velocity never closes
    // it: discriminant = 160^2 - 4*64*160 = 25600-40960 = -15360 < 0 -> miss.
    // (Exact integers throughout; no sqrt needed because the result is a
    // miss.)
    assert!(
        sphere_capsule_toi(
            v3i(0, -13, 0),
            fi(1),
            v3i(8, 0, 0),
            v3i(10, -5, 0),
            v3i(10, 5, 0),
            fi(1)
        )
        .is_none(),
        "clamped closest point (cap_a) with a vertical gap exceeding combined radius must miss"
    );

    // Degenerate: collapsed capsule (cap_a == cap_b) reduces to a sphere.
    // closest = cap_a = (5,0,0) unconditionally (ab_len_sq == 0 branch).
    // sphere_sphere_toi(center=(-5,0,0) r1 vel(16,0,0), (5,0,0) r1 static):
    // rel_pos=(10,0,0), rel_vel=(-16,0,0), combined_r=2.
    // a=256, b=2*(10*-16)=-320, c=100-4=96.
    // disc = 320^2-4*256*96 = 102400-98304=4096, sqrt=64.
    // t=(320-64)/512=256/512=1/2 (a power-of-two denominator -- Fix128 is
    // I64F64 binary fixed point, so a fraction like 4/5 below would not be
    // exactly representable; every closed form in this file is chosen to
    // reduce to a power-of-two denominator for this reason).
    // pos_a=(-5,0,0)+(8,0,0)=(3,0,0), pos_b=(5,0,0), normal=(1,0,0),
    // point=(4,0,0).
    let degenerate_cap = sphere_capsule_toi(
        v3i(-5, 0, 0),
        fi(1),
        v3i(16, 0, 0),
        v3i(5, 0, 0),
        v3i(5, 0, 0),
        fi(1),
    )
    .expect("collapsed capsule must behave exactly like a sphere");
    assert_eq!(degenerate_cap.t, r(1, 2));
    assert_eq!(degenerate_cap.point, v3i(4, 0, 0));

    // Degenerate: zero radii on both shapes. combined_r=0 means contact
    // only happens when centers exactly coincide; at that instant
    // `rel_pos` is zero, so `normal = rel_pos.normalize()` documented-
    // returns Vec3Fix::ZERO (src/math.rs `Vec3Fix::normalize`) rather than
    // panicking, and `point = pos_a + ZERO*0 = pos_a`.
    // closest for sphere(0,0,0) vel(16,0,0) vs capsule (10,-5,0)-(10,5,0):
    // ab=(0,10,0), (sphere-cap_a)=(-10,5,0), dot=50, t=0.5 -> closest=(10,0,0).
    // sphere_sphere_toi(center=0 r0 vel(16,0,0), (10,0,0) r0 static):
    // a=256, b=2*(10*-16)=-320, c=100-0=100.
    // disc = 320^2-4*256*100 = 102400-102400=0, sqrt=0.
    // t=320/512=0.625, pos_a=(0,0,0)+(10,0,0)=(10,0,0)=pos_b -> coincident.
    let zero_radius = sphere_capsule_toi(
        Vec3Fix::ZERO,
        Fix128::ZERO,
        v3i(16, 0, 0),
        v3i(10, -5, 0),
        v3i(10, 5, 0),
        Fix128::ZERO,
    )
    .expect("zero-radius sphere vs zero-radius capsule must still find the coincidence instant");
    assert_eq!(zero_radius.t, r(5, 8));
    assert_eq!(zero_radius.point, v3i(10, 0, 0));
    assert_eq!(
        zero_radius.normal,
        Vec3Fix::ZERO,
        "zero-length rel_pos must normalize to ZERO, not panic (documented contract)"
    );
    println!("[ccd] sphere_capsule_toi: clamp miss / collapsed capsule / zero-radius coincidence all match closed form");

    let extreme_cap = catch_unwind(|| {
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
        extreme_cap.is_ok(),
        "extreme-magnitude capsule endpoints must not panic"
    );

    // ==================================================================
    // capsule_plane_toi: picks whichever endpoint is nearer the plane at
    // t=0, then defers to sphere_plane_toi on that endpoint.
    // ==================================================================
    // plane y=3 (normal (0,1,0), offset 3). cap_a=(0,20,0) dist 17,
    // cap_b=(0,10,0) dist 7 -> b is nearer -> center=(0,10,0).
    // sphere_plane_toi(center=(0,10,0) r2 vel(0,-16,0), plane y=3):
    // dist=7, vel_toward=-16, side=+1 (dist positive).
    // t = (2-7)/(-16) = 5/16. point = (0,10,0)+(0,-5,0)-(0,2,0) = (0,3,0).
    let near_b = capsule_plane_toi(
        v3i(0, 20, 0),
        v3i(0, 10, 0),
        fi(2),
        v3i(0, -16, 0),
        v3i(0, 1, 0),
        fi(3),
    )
    .expect("endpoint b (nearer the plane) must drive the TOI");
    assert_eq!(near_b.t, r(5, 16));
    assert_eq!(near_b.point, v3i(0, 3, 0));
    assert_eq!(near_b.normal, v3i(0, 1, 0));

    // Tie-break: cap_a=(0,13,-5), cap_b=(0,13,5) are equidistant from
    // plane y=0 -> `dist_a < dist_b` is false -> b is used.
    // sphere_plane_toi(center=(0,13,5) r1 vel(0,-16,0), plane y=0):
    // dist=13, t=(1-13)/(-16)=12/16=3/4,
    // point=(0,13,5)+(0,-12,0)-(0,1,0)=(0,0,5).
    let tie = capsule_plane_toi(
        v3i(0, 13, -5),
        v3i(0, 13, 5),
        fi(1),
        v3i(0, -16, 0),
        v3i(0, 1, 0),
        Fix128::ZERO,
    )
    .expect("equidistant endpoints must pick b (the `else` branch of `<`)");
    assert_eq!(tie.t, r(3, 4));
    assert_eq!(tie.point, v3i(0, 0, 5));
    println!("[ccd] capsule_plane_toi: nearer-endpoint selection and tie-break match closed form");

    // Degenerate: zero velocity, not already touching -> the
    // `vel_toward >= 0 && dist > radius` early-return fires with
    // vel_toward == 0.
    assert!(
        capsule_plane_toi(
            v3i(0, 20, 0),
            v3i(0, 15, 0),
            fi(1),
            Vec3Fix::ZERO,
            v3i(0, 1, 0),
            Fix128::ZERO
        )
        .is_none(),
        "zero velocity while clear of the plane must miss"
    );
    // Degenerate: zero radius still reduces correctly to point-vs-plane.
    let zero_r_cap = capsule_plane_toi(
        v3i(0, 10, 0),
        v3i(0, 4, 0),
        Fix128::ZERO,
        v3i(0, -8, 0),
        v3i(0, 1, 0),
        Fix128::ZERO,
    )
    .expect("zero-radius capsule vs plane must still hit");
    assert_eq!(zero_r_cap.t, r(1, 2));
    assert_eq!(zero_r_cap.point, Vec3Fix::ZERO);
    // Degenerate: already overlapping at t=0 (dist 1 <= radius 2).
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
    // Degenerate: moving away from the plane -> None.
    assert!(
        capsule_plane_toi(
            v3i(0, 10, 0),
            v3i(0, 15, 0),
            fi(1),
            v3i(0, 8, 0),
            v3i(0, 1, 0),
            Fix128::ZERO
        )
        .is_none(),
        "capsule moving away from the plane must miss"
    );
    println!("[ccd] capsule_plane_toi: zero velocity / zero radius / already-penetrating / moving-away all match closed form");

    // ==================================================================
    // aabb_plane_toi: support vertex chosen per normal sign, then a
    // point-vs-plane TOI (no "side" flip -- unlike sphere_plane_toi, the
    // returned normal is always exactly `plane_normal`).
    // ==================================================================
    // AABB [-2,-2,-2]..[6,6,6], plane y=-10 (normal (0,1,0) offset -10).
    // normal.y=1 is not < 0 -> support.y = min.y = -2 (same for x, z since
    // their normal components are 0, also not < 0 -> min).
    // support=(-2,-2,-2). dist = -2 - (-10) = 8. vel=(0,-32,0),
    // vel_toward=-32. t = -8/-32 = 1/4. point=(-2,-2,-2)+(0,-8,0)=(-2,-10,-2).
    let aabb1 = AABB::new(v3i(-2, -2, -2), v3i(6, 6, 6));
    let ab1 = aabb_plane_toi(&aabb1, v3i(0, -32, 0), v3i(0, 1, 0), fi(-10)).expect("must hit");
    assert_eq!(ab1.t, r(1, 4));
    assert_eq!(ab1.point, v3i(-2, -10, -2));
    assert_eq!(ab1.normal, v3i(0, 1, 0));

    // Negative-normal-component branch: plane x=10 with normal (-1,0,0),
    // offset -10. normal.x=-1 < 0 -> support.x = max.x = 5 (y,z use min
    // since their normal components are 0, not < 0).
    // AABB [-3,-3,-3]..[5,5,5] -> support=(5,-3,-3).
    // dist = 5*(-1) - (-10) = 5. vel=(8,0,0), vel_toward = 8*(-1) = -8.
    // t = -5/-8 = 5/8. point = (5,-3,-3)+(5,0,0) = (10,-3,-3).
    let aabb2 = AABB::new(v3i(-3, -3, -3), v3i(5, 5, 5));
    let ab2 = aabb_plane_toi(&aabb2, v3i(8, 0, 0), v3i(-1, 0, 0), fi(-10)).expect("must hit");
    assert_eq!(ab2.t, r(5, 8));
    assert_eq!(ab2.point, v3i(10, -3, -3));
    println!(
        "[ccd] aabb_plane_toi: normal-sign support selection matches closed form on both branches"
    );

    // Degenerate: zero velocity, clear of the plane -> None.
    assert!(
        aabb_plane_toi(&aabb1, Vec3Fix::ZERO, v3i(0, 1, 0), fi(-10)).is_none(),
        "zero velocity while clear of the plane must miss"
    );
    // Degenerate: zero-size AABB (a point) reduces to point-vs-plane.
    let point_box = AABB::new(v3i(5, 10, 5), v3i(5, 10, 5));
    let pb =
        aabb_plane_toi(&point_box, v3i(0, -20, 0), v3i(0, 1, 0), Fix128::ZERO).expect("must hit");
    assert_eq!(pb.t, r(1, 2));
    assert_eq!(pb.point, v3i(5, 0, 5));
    // Degenerate: already penetrating (support.y = -5 <= plane y=0).
    let pen_box = AABB::new(v3i(-1, -5, -1), v3i(1, 5, 1));
    let pen_ab = aabb_plane_toi(&pen_box, v3i(0, 8, 0), v3i(0, 1, 0), Fix128::ZERO)
        .expect("must report t=0");
    assert_eq!(pen_ab.t, Fix128::ZERO);
    assert_eq!(pen_ab.point, v3i(-1, -5, -1));
    // Degenerate: moving away.
    let clear_box = AABB::new(v3i(1, 1, 1), v3i(3, 3, 3));
    assert!(
        aabb_plane_toi(&clear_box, v3i(0, 8, 0), v3i(0, 1, 0), Fix128::ZERO).is_none(),
        "AABB moving away from the plane must miss"
    );
    println!("[ccd] aabb_plane_toi: zero-size box / already-penetrating / moving-away all match closed form");

    let extreme_box = AABB::new(
        v3i(-1, -1, -1),
        Vec3Fix::new(Fix128::from_raw(i64::MAX, u64::MAX), fi(1), fi(1)),
    );
    let extreme_ab =
        catch_unwind(|| aabb_plane_toi(&extreme_box, v3i(1, 0, 0), v3i(-1, 0, 0), fi(-10)));
    assert!(extreme_ab.is_ok(), "extreme-magnitude AABB must not panic");

    // ==================================================================
    // swept_aabb: per-axis slab test, combined via the entry/exit times.
    // ==================================================================
    // moving [-2,2]^3, target [10,14]x[10,14]x[-2,2], velocity (16,16,0).
    // x: t0=(10-2)/16=1/2, t1=(14+2)/16=1. y: identical (same numbers).
    // z: static axis, ranges [-2,2] vs [-2,2] overlap -> unrestricted
    // interval. Combined: t_enter=max(1/2,1/2,-huge)=1/2,
    // t_exit=min(1,1,huge)=1 -> Some(1/2).
    let moving = AABB::new(v3i(-2, -2, -2), v3i(2, 2, 2));
    let target = AABB::new(v3i(10, 10, -2), v3i(14, 14, 2));
    let swept = swept_aabb(&moving, v3i(16, 16, 0), &target)
        .expect("oblique approach with one static axis must hit");
    assert_eq!(swept, r(1, 2));
    println!("[ccd] swept_aabb oblique (2 active axes + 1 static-overlap axis): t={swept:?}");

    // Degenerate: zero velocity, overlapping -> Some(ZERO).
    let overlap_a = AABB::new(v3i(0, 0, 0), v3i(4, 4, 4));
    let overlap_b = AABB::new(v3i(2, 2, 2), v3i(6, 6, 6));
    assert_eq!(
        swept_aabb(&overlap_a, Vec3Fix::ZERO, &overlap_b),
        Some(Fix128::ZERO)
    );
    // Degenerate: zero velocity, not overlapping -> None.
    let far_a = AABB::new(v3i(0, 0, 0), v3i(1, 1, 1));
    let far_b = AABB::new(v3i(5, 5, 5), v3i(6, 6, 6));
    assert_eq!(swept_aabb(&far_a, Vec3Fix::ZERO, &far_b), None);
    println!("[ccd] swept_aabb: zero-velocity overlap -> Some(0), zero-velocity clear -> None");

    let extreme_target = AABB::new(
        Vec3Fix::new(Fix128::from_raw(i64::MAX / 2, 0), fi(0), fi(0)),
        Vec3Fix::new(Fix128::from_raw(i64::MAX, 0), fi(1), fi(1)),
    );
    let extreme_swept = catch_unwind(|| swept_aabb(&moving, v3i(1, 0, 0), &extreme_target));
    assert!(
        extreme_swept.is_ok(),
        "extreme-magnitude target AABB must not panic"
    );

    // ==================================================================
    // speculative_contact: either an immediate overlap contact (depth =
    // -gap) or a predicted breach within dt, otherwise None.
    // ==================================================================
    // Breach: A (0,0,0) vel(10,0,0) r2, B (20,0,0) static r3, dt=2.
    // rel_pos=(20,0,0), dist=20, combined_r=5, gap=15. normal=(1,0,0).
    // rel_vel=(-10,0,0), closing_speed = -(-10) = 10. predicted = 15-20=-5<0.
    // depth=5, point_a=(0,0,0)+(2,0,0)=(2,0,0), point_b=(20,0,0)-(3,0,0)=(17,0,0).
    let breach = speculative_contact(
        Vec3Fix::ZERO,
        v3i(10, 0, 0),
        fi(2),
        v3i(20, 0, 0),
        Vec3Fix::ZERO,
        fi(3),
        fi(2),
    )
    .expect("closing pair must breach the gap within dt");
    assert_eq!(breach.depth, fi(5));
    assert_eq!(breach.point_a, v3i(2, 0, 0));
    assert_eq!(breach.point_b, v3i(17, 0, 0));

    // Already overlapping: A (0,0,0) r4, B (5,0,0) r4. dist=5, combined=8,
    // gap=-3 -> depth=3 regardless of velocity/dt.
    // point_a=(0,0,0)+(4,0,0)=(4,0,0), point_b=(5,0,0)-(4,0,0)=(1,0,0).
    let overlap_contact = speculative_contact(
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
        fi(4),
        v3i(5, 0, 0),
        Vec3Fix::ZERO,
        fi(4),
        fi(1),
    )
    .expect("overlapping pair must report a contact independent of velocity");
    assert_eq!(overlap_contact.depth, fi(3));
    assert_eq!(overlap_contact.point_a, v3i(4, 0, 0));
    assert_eq!(overlap_contact.point_b, v3i(1, 0, 0));
    println!(
        "[ccd] speculative_contact: breach depth={:?}, overlap depth={:?}",
        breach.depth, overlap_contact.depth
    );

    // Degenerate: coincident centers (dist.is_zero()) -> None unconditionally.
    assert!(
        speculative_contact(
            v3i(5, 5, 5),
            Vec3Fix::ZERO,
            fi(1),
            v3i(5, 5, 5),
            Vec3Fix::ZERO,
            fi(1),
            fi(1)
        )
        .is_none(),
        "coincident centers must return None (normal is undefined)"
    );
    // Degenerate: moving apart (closing_speed <= 0) -> None even with a
    // huge dt.
    assert!(
        speculative_contact(
            Vec3Fix::ZERO,
            v3i(-10, 0, 0),
            fi(1),
            v3i(20, 0, 0),
            Vec3Fix::ZERO,
            fi(1),
            fi(100)
        )
        .is_none(),
        "separating pair must never produce a speculative contact"
    );
    // Degenerate: dt == 0 with a positive gap -> predicted_gap == gap > 0 -> None.
    assert!(
        speculative_contact(
            Vec3Fix::ZERO,
            v3i(5, 0, 0),
            fi(1),
            v3i(10, 0, 0),
            Vec3Fix::ZERO,
            fi(1),
            Fix128::ZERO
        )
        .is_none(),
        "dt == 0 must not predict a breach even on a closing course"
    );
    // Degenerate: zero combined radius, exact boundary (predicted_gap == 0
    // exactly) -> the strict `< ZERO` check means this is still a miss.
    // dist=8 (sqrt(64)=8 exact), gap=8, closing=8, dt=1 -> predicted=0.
    assert!(
        speculative_contact(
            Vec3Fix::ZERO,
            v3i(8, 0, 0),
            Fix128::ZERO,
            v3i(8, 0, 0),
            Vec3Fix::ZERO,
            Fix128::ZERO,
            fi(1)
        )
        .is_none(),
        "predicted_gap exactly 0 (zero-radius boundary) must not count as a breach"
    );
    println!("[ccd] speculative_contact: coincident / separating / dt=0 / zero-radius-boundary all None as derived");

    let extreme_contact = catch_unwind(|| {
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
        extreme_contact.is_ok(),
        "extreme-magnitude positions must not panic"
    );

    // NOTE: `adaptive_toi_substeps` is intentionally NOT exercised here --
    // see the module doc comment at the top of this file. It is a
    // documented skeleton being implemented for real by a separate task.

    // ==================================================================
    // conservative_advancement: iterative safe-advance against a generic
    // distance function. Using an off-axis sphere obstacle (distinct from
    // this crate's own plane-based unit tests) so the closure path itself
    // is exercised, not just the surrounding arithmetic.
    // ==================================================================
    let sphere_center = v3i(20, 0, 0);
    let sphere_radius = fi(3);
    let obstacle = move |p: Vec3Fix| {
        let to_center = p - sphere_center;
        let dist_to_surface = to_center.length() - sphere_radius;
        (dist_to_surface, to_center.normalize())
    };

    // start (0,0,0), displacement (32,0,0), agent radius 1. Motion is
    // exactly colinear with the vector to the obstacle's center, so the
    // distance function is exactly linear in t along this path (no
    // curvature term). Displacement magnitude 32 (a power of two) is
    // chosen deliberately: Fix128 is I64F64 binary fixed point, so
    // `gap / speed` is only exact when it reduces to a power-of-two
    // denominator (e.g. 40 would give 16/40 = 2/5, not exactly
    // representable in binary). The loop converges in 2 passes:
    //   iter0: dist=|{-20,0,0}|-3=17, gap=17-1=16>tolerance,
    //          speed=32, dt=16/32=1/2, t=1/2.
    //   iter1: pos=(16,0,0), dist=|{-4,0,0}|-3=1, gap=1-1=0<=tolerance ->
    //          Some(t=1/2, point=pos-normal*dist=(16,0,0)+(1,0,0)=(17,0,0),
    //          normal=(-1,0,0)).
    let cfg_ca = CcdConfig::default();
    let hit = conservative_advancement(Vec3Fix::ZERO, v3i(32, 0, 0), fi(1), obstacle, &cfg_ca)
        .expect("colinear approach to the sphere obstacle must converge to a hit");
    assert_eq!(hit.t, r(1, 2));
    assert_eq!(hit.point, v3i(17, 0, 0));
    assert_eq!(hit.normal, v3i(-1, 0, 0));
    println!(
        "[ccd] conservative_advancement sphere-obstacle: t={:?} point={:?}",
        hit.t, hit.point
    );

    // Degenerate: zero agent radius, same obstacle. Same colinear
    // reasoning but without the agent-radius offset:
    //   iter0: dist=17, gap=17-0=17, dt=17/32, t=17/32 (32 is a power of
    //          two, so any integer/32 is exactly representable).
    //   iter1: pos=(17,0,0), dist=|{-3,0,0}|-3=0, gap=0<=tolerance ->
    //          Some(t=17/32, point=(17,0,0)-(-1,0,0)*0=(17,0,0)).
    let zero_radius_hit = conservative_advancement(
        Vec3Fix::ZERO,
        v3i(32, 0, 0),
        Fix128::ZERO,
        obstacle,
        &cfg_ca,
    )
    .expect("zero agent radius must still converge");
    assert_eq!(zero_radius_hit.t, r(17, 32));
    assert_eq!(zero_radius_hit.point, v3i(17, 0, 0));

    // Degenerate: displacement too short to ever reach the surface within
    // this timestep. iter0: dist=17, gap=16, speed=10, dt=16/10=1.6>1 -> None.
    assert!(
        conservative_advancement(Vec3Fix::ZERO, v3i(10, 0, 0), fi(1), obstacle, &cfg_ca).is_none(),
        "displacement too short to close the gap within t<=1 must miss"
    );

    // Degenerate: max_iterations == 0 -> the loop body never runs, so the
    // result is None even starting exactly on the surface (gap would be
    // <= tolerance on the very first -- unrun -- iteration).
    let zero_iters = CcdConfig {
        max_iterations: 0,
        ..cfg_ca
    };
    assert!(
        conservative_advancement(v3i(17, 0, 0), v3i(1, 0, 0), fi(0), obstacle, &zero_iters)
            .is_none(),
        "max_iterations == 0 must return None even when already touching"
    );
    println!("[ccd] conservative_advancement: zero radius / too-short displacement / max_iterations=0 all match closed form");

    let extreme_displacement = Vec3Fix::new(Fix128::from_raw(i64::MAX, u64::MAX), fi(0), fi(0));
    let extreme_distance_fn = |_p: Vec3Fix| (Fix128::from_raw(i64::MAX, u64::MAX), v3i(1, 0, 0));
    let extreme_ca = catch_unwind(|| {
        conservative_advancement(
            Vec3Fix::ZERO,
            extreme_displacement,
            fi(1),
            extreme_distance_fn,
            &cfg_ca,
        )
    });
    assert!(
        extreme_ca.is_ok(),
        "extreme-magnitude displacement/distance must not panic"
    );

    println!("[ccd] all oracle checks passed");
}
