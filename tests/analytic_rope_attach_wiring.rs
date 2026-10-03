//! Oracles for `rope_attach::{RopeAttachment::compliance, solve_rope_attachments}`
//! (`examples/rope_body_attachment.rs`).
//!
//! Hand-derived XPBD point constraint (rope particle weight 1, body weight `wb`,
//! `alpha = compliance / dt^2`, gap `d = rope - anchor_world`, `|d| = L`):
//! * `lambda = L / (1 + wb + alpha)`, rope moves by `-d/L * lambda * 1`
//!   (so a rigid static anchor lands the particle exactly on the anchor);
//! * rope velocity changes by `-(rope move) / dt` (here: `+d/L*lambda/dt`
//!   subtracted, i.e. velocity -= correction / dt where correction = rope shift);
//! * break test is `lambda / dt^2 > break_force` (strict).
//! * anchor world point = `body.position + rotate(body.rotation, local_anchor)`.
#![allow(clippy::disallowed_methods)]

use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::rope_attach::{
    solve_rope_attachments, solve_rope_attachments_two_way, RopeAttachment,
};
use alice_physics::solver::RigidBody;

fn close(a: Vec3Fix, x: f64, y: f64, z: f64) -> bool {
    let e = 1e-12;
    (a.x.to_f64() - x).abs() < e && (a.y.to_f64() - y).abs() < e && (a.z.to_f64() - z).abs() < e
}
fn half() -> Fix128 {
    Fix128::from_ratio(1, 2)
}
fn rot_z180() -> QuatFix {
    QuatFix {
        x: Fix128::ZERO,
        y: Fix128::ZERO,
        z: Fix128::ONE,
        w: Fix128::ZERO,
    }
}

#[test]
fn builder_sets_compliance_and_keeps_other_fields() {
    let a = RopeAttachment::with_break_force(2, 3, Vec3Fix::UNIT_X, Fix128::from_int(7))
        .compliance(Fix128::from_ratio(1, 5));
    assert_eq!((a.rope_particle, a.body_index), (2, 3));
    assert_eq!(a.compliance, Fix128::from_ratio(1, 5));
    assert_eq!(a.break_force, Some(Fix128::from_int(7)));
    assert_eq!(a.local_anchor, Vec3Fix::UNIT_X);
    // last call wins
    let b = RopeAttachment::new(0, 0, Vec3Fix::ZERO)
        .compliance(Fix128::ONE)
        .compliance(Fix128::from_int(2));
    assert_eq!(b.compliance, Fix128::from_int(2));
}

#[test]
fn rigid_static_anchor_lands_particle_exactly_with_rotated_anchor() {
    let mut body = RigidBody::new_static(Vec3Fix::from_int(10, 5, 0));
    body.rotation = rot_z180(); // (x, y, z) -> (-x, -y, z)
    let att = [RopeAttachment::new(1, 0, Vec3Fix::from_int(1, 1, 0))];
    let mut p = vec![Vec3Fix::from_int(0, 0, 0), Vec3Fix::from_int(0, 0, 0)];
    let mut v = vec![Vec3Fix::ZERO; 2];
    let dt = Fix128::from_ratio(1, 60);
    let broken = solve_rope_attachments(&att, &mut p, &mut v, &[body], dt);
    assert!(broken.is_empty());
    assert!(close(p[1], 9.0, 4.0, 0.0), "{:?}", p[1]);
    assert_eq!(p[0], Vec3Fix::ZERO, "other particles untouched");
    // velocity -= correction/dt, correction = p_before - p_after = (-9,-4,0)
    // => v = (+9, +4, 0) * 60
    assert!(close(v[1], 540.0, 240.0, 0.0), "{:?}", v[1]);
    assert_eq!(v[0], Vec3Fix::ZERO);
}

#[test]
fn compliance_closes_one_over_one_plus_alpha() {
    let dt = half();
    let body = RigidBody::new_static(Vec3Fix::ZERO);
    for (c_num, c_den, frac) in [(1i64, 4i64, 0.5f64), (3, 4, 0.25), (1, 8, 2.0 / 3.0)] {
        // alpha = c / dt^2 = 4c -> frac = 1/(1+alpha)
        let att = [
            RopeAttachment::new(0, 0, Vec3Fix::ZERO).compliance(Fix128::from_ratio(c_num, c_den))
        ];
        let mut p = vec![Vec3Fix::from_int(3, 4, 0)];
        let mut v = vec![Vec3Fix::ZERO];
        solve_rope_attachments(&att, &mut p, &mut v, &[body], dt);
        let remaining = 1.0 - frac;
        assert!(
            close(p[0], 3.0 * remaining, 4.0 * remaining, 0.0),
            "c={c_num}/{c_den} {:?}",
            p[0]
        );
        // v = -(move)/dt, move = delta*frac -> v = -delta*frac/dt... sign: velocity -= correction/dt, correction = delta*frac
        assert!(
            close(v[0], -3.0 * frac * 2.0, -4.0 * frac * 2.0, 0.0),
            "{:?}",
            v[0]
        );
    }
}

#[test]
fn dynamic_body_takes_its_inverse_mass_share_of_the_gap() {
    // wb = 1 (mass 1): rope moves w_rope / (w_rope + wb) = 1/2 of the gap
    let body = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE);
    let att = [RopeAttachment::new(0, 0, Vec3Fix::ZERO)];
    let mut p = vec![Vec3Fix::from_int(3, 4, 0)];
    let mut v = vec![Vec3Fix::ZERO];
    solve_rope_attachments(&att, &mut p, &mut v, &[body], Fix128::from_ratio(1, 60));
    assert!(close(p[0], 1.5, 2.0, 0.0), "{:?}", p[0]);
    // mass 4 -> wb = 1/4 -> rope moves 1/(1.25) = 0.8
    let body = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::from_int(4));
    let mut p = vec![Vec3Fix::from_int(3, 4, 0)];
    solve_rope_attachments(&att, &mut p, &mut v, &[body], Fix128::from_ratio(1, 60));
    assert!(close(p[0], 3.0 * 0.2, 4.0 * 0.2, 0.0), "{:?}", p[0]);
}

#[test]
fn break_threshold_is_strict_and_leaves_the_particle_alone() {
    // dt = 1/2: dt^2 = 1/4; gap 5, static anchor, alpha 0: force = 5 / (1/4) = 20 exactly
    let body = RigidBody::new_static(Vec3Fix::ZERO);
    let dt = half();
    let run = |bf: Fix128| {
        let att = [
            RopeAttachment::new(0, 0, Vec3Fix::ZERO),
            RopeAttachment::with_break_force(1, 0, Vec3Fix::ZERO, bf),
        ];
        let mut p = vec![Vec3Fix::from_int(0, 0, 0), Vec3Fix::from_int(3, 4, 0)];
        let mut v = vec![Vec3Fix::ZERO; 2];
        let broken = solve_rope_attachments(&att, &mut p, &mut v, &[body], dt);
        (broken, p, v)
    };
    let (b, p, _) = run(Fix128::from_int(20));
    assert!(b.is_empty(), "force == threshold does not break");
    assert!(close(p[1], 0.0, 0.0, 0.0));
    let (b, p, v) = run(Fix128::from_ratio(1999, 100));
    assert_eq!(b, vec![Some(1)], "reports the attachment index");
    assert_eq!(
        p[1],
        Vec3Fix::from_int(3, 4, 0),
        "broken: particle untouched"
    );
    assert_eq!(v[1], Vec3Fix::ZERO);
}

#[test]
fn degenerate_inputs_do_nothing() {
    let body = RigidBody::new_static(Vec3Fix::ZERO);
    let att = [
        RopeAttachment::new(5, 0, Vec3Fix::ZERO), // particle out of range
        RopeAttachment::new(0, 9, Vec3Fix::ZERO), // body out of range
        RopeAttachment::new(1, 0, Vec3Fix::ZERO), // already coincident
    ];
    let start = vec![Vec3Fix::from_int(3, 4, 0), Vec3Fix::ZERO];
    let mut p = start.clone();
    let mut v = vec![Vec3Fix::ZERO; 2];
    let broken = solve_rope_attachments(&att, &mut p, &mut v, &[body], Fix128::from_ratio(1, 60));
    assert!(broken.is_empty());
    assert_eq!(p, start);
    assert_eq!(v, vec![Vec3Fix::ZERO; 2]);
    // dt = 0: no effect even for a valid attachment
    let att = [RopeAttachment::new(0, 0, Vec3Fix::ZERO)];
    let mut p = start.clone();
    assert!(solve_rope_attachments(&att, &mut p, &mut v, &[body], Fix128::ZERO).is_empty());
    assert_eq!(p, start);
    // empty attachment list
    assert!(solve_rope_attachments(&[], &mut p, &mut v, &[body], half()).is_empty());
}

#[test]
fn zero_weight_sum_is_skipped() {
    // static body, rigid attachment still has w_rope = 1 so w_sum > 0; instead use a
    // negative compliance that cancels the rope weight exactly: alpha = -1
    let body = RigidBody::new_static(Vec3Fix::ZERO);
    let att = [RopeAttachment::new(0, 0, Vec3Fix::ZERO).compliance(Fix128::from_ratio(-1, 4))];
    let start = vec![Vec3Fix::from_int(3, 4, 0)];
    let mut p = start.clone();
    let mut v = vec![Vec3Fix::ZERO];
    let broken = solve_rope_attachments(&att, &mut p, &mut v, &[body], half());
    assert!(broken.is_empty());
    assert_eq!(p, start, "w_sum == 0 is skipped, no division by zero");
}

#[test]
fn index_equal_to_length_is_out_of_range() {
    let body = RigidBody::new_static(Vec3Fix::ZERO);
    let start = vec![Vec3Fix::from_int(3, 4, 0), Vec3Fix::from_int(1, 0, 0)];
    // particle == len (2), body == len (1)
    let att = [
        RopeAttachment::new(2, 0, Vec3Fix::ZERO),
        RopeAttachment::new(0, 1, Vec3Fix::ZERO),
    ];
    let mut p = start.clone();
    let mut v = vec![Vec3Fix::ZERO; 2];
    let broken = solve_rope_attachments(&att, &mut p, &mut v, &[body], half());
    assert!(broken.is_empty());
    assert_eq!(p, start);
}

#[test]
fn skipped_attachments_never_break_even_with_a_negative_threshold() {
    // coincident particle (gap 0) and a zero weight sum are skipped before the break
    // test; with threshold -1 any evaluated force (>= 0) would otherwise break
    let body = RigidBody::new_static(Vec3Fix::ZERO);
    let neg = Fix128::NEG_ONE;
    let att = [RopeAttachment::with_break_force(0, 0, Vec3Fix::ZERO, neg)];
    let mut p = vec![Vec3Fix::ZERO];
    let mut v = vec![Vec3Fix::ZERO];
    assert!(solve_rope_attachments(&att, &mut p, &mut v, &[body], half()).is_empty());
    let att = [RopeAttachment::with_break_force(0, 0, Vec3Fix::ZERO, neg)
        .compliance(Fix128::from_ratio(-1, 4))];
    let mut p = vec![Vec3Fix::from_int(3, 4, 0)];
    assert!(solve_rope_attachments(&att, &mut p, &mut v, &[body], half()).is_empty());
    assert_eq!(p[0], Vec3Fix::from_int(3, 4, 0));
}

// ---- two-way variant -------------------------------------------------------
//
// gap C = |rope - anchor|, n = unit (rope - anchor), wr = 1, wb = inv_mass,
// alpha = c/dt^2:  dx_rope = -n C wr/(wr+wb+alpha),  dx_body = +n C wb/(wr+wb+alpha)

fn two_way(
    body: RigidBody,
    rope: Vec3Fix,
    compliance: Fix128,
    dt: Fix128,
) -> (Vec3Fix, Vec3Fix, Vec<Option<usize>>) {
    let att = [RopeAttachment::new(0, 0, Vec3Fix::ZERO).compliance(compliance)];
    let mut p = vec![rope];
    let mut v = vec![Vec3Fix::ZERO];
    let mut bodies = [body];
    let b = solve_rope_attachments_two_way(&att, &mut p, &mut v, &mut bodies, dt);
    (p[0], bodies[0].position, b)
}

#[test]
fn two_way_splits_the_gap_by_inverse_mass_and_conserves_momentum() {
    // gap (3,4,0), L = 5, n = (0.6, 0.8, 0)
    for (mass, wb) in [(1i64, 1.0f64), (2, 0.5), (4, 0.25), (10, 0.1)] {
        let body = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::from_int(mass));
        let (pr, pb, b) = two_way(body, Vec3Fix::from_int(3, 4, 0), Fix128::ZERO, half());
        assert!(b.is_empty());
        let s = 1.0 + wb;
        assert!(
            close(pr, 3.0 - 3.0 / s, 4.0 - 4.0 / s, 0.0),
            "mass {mass} rope {pr:?}"
        );
        assert!(
            close(pb, 3.0 * wb / s, 4.0 * wb / s, 0.0),
            "mass {mass} body {pb:?}"
        );
        // m_rope dx_rope + m_body dx_body = 0, with m_rope = 1, m_body = mass
        let mx = (pr.x.to_f64() - 3.0) + mass as f64 * pb.x.to_f64();
        let my = (pr.y.to_f64() - 4.0) + mass as f64 * pb.y.to_f64();
        assert!(mx.abs() < 1e-12 && my.abs() < 1e-12, "momentum {mx} {my}");
        // rigid: the particle and the (zero-offset) anchor coincide afterwards
        assert!((pr - pb).length().to_f64() < 1e-12);
    }
}

#[test]
fn two_way_with_static_body_equals_one_way() {
    let body = RigidBody::new_static(Vec3Fix::from_int(1, 2, 3));
    let att =
        [RopeAttachment::new(0, 0, Vec3Fix::from_int(0, 1, 0))
            .compliance(Fix128::from_ratio(1, 8))];
    let dt = Fix128::from_ratio(1, 4);
    let start = Vec3Fix::from_int(7, -3, 2);
    let (mut p1, mut v1) = (vec![start], vec![Vec3Fix::ZERO]);
    let (mut p2, mut v2) = (vec![start], vec![Vec3Fix::ZERO]);
    let mut bodies = [body];
    let b1 = solve_rope_attachments(&att, &mut p1, &mut v1, &[body], dt);
    let b2 = solve_rope_attachments_two_way(&att, &mut p2, &mut v2, &mut bodies, dt);
    assert_eq!((p1, v1, b1), (p2, v2, b2));
    assert_eq!(
        bodies[0].position, body.position,
        "static body is not moved"
    );
}

#[test]
fn two_way_compliance_shrinks_both_corrections() {
    // alpha = c/dt^2 = 4c; mass 1 -> wb 1; share denominators 2 + alpha
    let dt = half();
    let mut last = f64::MAX;
    for (c_num, c_den) in [(0i64, 1i64), (1, 4), (3, 4), (3, 1), (100, 1)] {
        let body = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE);
        let (pr, pb, _) = two_way(
            body,
            Vec3Fix::from_int(3, 4, 0),
            Fix128::from_ratio(c_num, c_den),
            dt,
        );
        let alpha = 4.0 * c_num as f64 / c_den as f64;
        let moved_rope = 5.0 - pr.length().to_f64();
        let moved_body = pb.length().to_f64();
        let d = 2.0 + alpha;
        assert!((moved_rope - 5.0 / d).abs() < 1e-12, "alpha {alpha}");
        assert!((moved_body - 5.0 / d).abs() < 1e-12, "alpha {alpha}");
        assert!(moved_rope < last);
        last = moved_rope;
    }
    assert!(last < 0.02, "alpha -> infinity: corrections -> 0");
}

#[test]
fn two_way_break_skip_and_degenerate_inputs_leave_body_alone() {
    let dt = half();
    // break: force = (5/2) / (1/4) = 10 with a unit-mass body; threshold 9 breaks
    let body = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE);
    let att = [RopeAttachment::with_break_force(
        0,
        0,
        Vec3Fix::ZERO,
        Fix128::from_int(9),
    )];
    let mut p = vec![Vec3Fix::from_int(3, 4, 0)];
    let mut v = vec![Vec3Fix::ZERO];
    let mut bodies = [body];
    assert_eq!(
        solve_rope_attachments_two_way(&att, &mut p, &mut v, &mut bodies, dt),
        vec![Some(0)]
    );
    assert_eq!(p[0], Vec3Fix::from_int(3, 4, 0));
    assert_eq!(bodies[0].position, Vec3Fix::ZERO);
    // threshold 10 (== force): not broken, body moves
    let att = [RopeAttachment::with_break_force(
        0,
        0,
        Vec3Fix::ZERO,
        Fix128::from_int(10),
    )];
    assert!(solve_rope_attachments_two_way(&att, &mut p, &mut v, &mut bodies, dt).is_empty());
    assert!(bodies[0].position.length().to_f64() > 2.4);
    // dt = 0, out-of-range indices, coincident
    let mut bodies = [body];
    let att = [
        RopeAttachment::new(0, 0, Vec3Fix::ZERO),
        RopeAttachment::new(3, 0, Vec3Fix::ZERO),
        RopeAttachment::new(0, 5, Vec3Fix::ZERO),
    ];
    let mut p = vec![Vec3Fix::from_int(3, 4, 0)];
    assert!(
        solve_rope_attachments_two_way(&att, &mut p, &mut v, &mut bodies, Fix128::ZERO).is_empty()
    );
    assert_eq!(
        (p[0], bodies[0].position),
        (Vec3Fix::from_int(3, 4, 0), Vec3Fix::ZERO)
    );
    let mut p = vec![Vec3Fix::ZERO];
    solve_rope_attachments_two_way(&att[..1], &mut p, &mut v, &mut bodies, dt);
    assert_eq!((p[0], bodies[0].position), (Vec3Fix::ZERO, Vec3Fix::ZERO));
    let mut p = vec![Vec3Fix::from_int(3, 4, 0)];
    solve_rope_attachments_two_way(&att[1..], &mut p, &mut v, &mut bodies, dt);
    assert_eq!(
        (p[0], bodies[0].position),
        (Vec3Fix::from_int(3, 4, 0), Vec3Fix::ZERO)
    );
}

#[test]
fn two_way_later_attachment_sees_the_corrected_body() {
    // two particles on the same unit-mass body at the same local anchor (origin)
    let att = [
        RopeAttachment::new(0, 0, Vec3Fix::ZERO),
        RopeAttachment::new(1, 0, Vec3Fix::ZERO),
    ];
    let mut p = vec![Vec3Fix::from_int(4, 0, 0), Vec3Fix::from_int(-4, 0, 0)];
    let mut v = vec![Vec3Fix::ZERO; 2];
    let mut bodies = [RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE)];
    solve_rope_attachments_two_way(&att, &mut p, &mut v, &mut bodies, half());
    // 1st: gap 4, body +2 (x = 2), rope0 = 2. 2nd: rope1 at -4, anchor at 2, gap 6 along -x:
    // body moves 3 toward rope1 (x = -1), rope1 = -4 + 3 = -1
    assert!(close(p[0], 2.0, 0.0, 0.0), "{:?}", p[0]);
    assert!(close(p[1], -1.0, 0.0, 0.0), "{:?}", p[1]);
    assert!(
        close(bodies[0].position, -1.0, 0.0, 0.0),
        "{:?}",
        bodies[0].position
    );
}

#[test]
fn two_way_index_equal_to_length_is_out_of_range_and_break_reports_attachment_index() {
    let dt = half();
    let mut bodies = [RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE)];
    let mut v = vec![Vec3Fix::ZERO; 2];
    let start = vec![Vec3Fix::from_int(3, 4, 0), Vec3Fix::from_int(1, 0, 0)];
    let mut p = start.clone();
    let att = [
        RopeAttachment::new(2, 0, Vec3Fix::ZERO), // particle == len
        RopeAttachment::new(0, 1, Vec3Fix::ZERO), // body == len
    ];
    assert!(solve_rope_attachments_two_way(&att, &mut p, &mut v, &mut bodies, dt).is_empty());
    assert_eq!((p.clone(), bodies[0].position), (start, Vec3Fix::ZERO));
    // the breaking attachment is #2 while its body index is 0
    let att = [
        RopeAttachment::new(1, 0, Vec3Fix::from_int(1, 0, 0)), // coincident -> skipped
        RopeAttachment::new(1, 0, Vec3Fix::from_int(0, 0, 0)),
        RopeAttachment::with_break_force(0, 0, Vec3Fix::ZERO, Fix128::ZERO),
    ];
    let broken = solve_rope_attachments_two_way(&att, &mut p, &mut v, &mut bodies, dt);
    assert_eq!(broken, vec![Some(2)]);
}
