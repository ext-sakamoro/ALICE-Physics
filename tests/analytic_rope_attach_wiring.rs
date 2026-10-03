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
use alice_physics::rope_attach::{solve_rope_attachments, RopeAttachment};
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
