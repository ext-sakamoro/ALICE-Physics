//! Audit oracles for `alice_physics::cloth_fluid` (S2-2 audit).
//!
//! Expected values are written from the module's documented formulas and from
//! first principles (drag as linear relaxation, Euler boundary closed form),
//! not copied from the implementation.

use alice_physics::cloth_fluid::{
    apply_cloth_boundary_to_fluid, apply_cloth_boundary_to_fluid_with_residual,
    apply_fluid_forces_to_cloth, apply_fluid_forces_to_cloth_with_residual, ClothFluidCoupling,
};
use alice_physics::math::{Fix128, Vec3Fix};

fn fx(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn v(x: Fix128, y: Fix128, z: Fix128) -> Vec3Fix {
    Vec3Fix::new(x, y, z)
}

fn drag_only(c: Fix128) -> ClothFluidCoupling {
    ClothFluidCoupling {
        drag_coefficient: c,
        buoyancy_factor: Fix128::ZERO,
        surface_tension: Fix128::ZERO,
    }
}

fn close(a: Fix128, b: Fix128, tol: f64) -> bool {
    (a.to_f64() - b.to_f64()).abs() <= tol
}

/// The interaction radius is 0.5 and strict: a fluid particle at 0.49 counts,
/// at 0.51 and at exactly 0.5 does not. Counted through the drag factor
/// 1 / (1 + C_d rho N dt) (only the in-range particles contribute to N).
#[test]
fn neighbour_radius_is_half_a_metre_and_strict() {
    let c = drag_only(Fix128::ONE);
    let dt = fx(1, 16);
    let cases: [(Fix128, f64); 3] = [(fx(49, 100), 1.0), (fx(51, 100), 0.0), (fx(1, 2), 0.0)];
    for (dist, n) in cases {
        let mut vel = [Vec3Fix::from_int(8, 0, 0)];
        apply_fluid_forces_to_cloth(
            &c,
            &[Vec3Fix::ZERO],
            &mut vel,
            &[v(dist, Fix128::ZERO, Fix128::ZERO)],
            &[Vec3Fix::ZERO],
            Fix128::ONE,
            dt,
        );
        let want = 8.0 / (1.0 + n / 16.0);
        assert!(
            close(vel[0].x, Fix128::from_f64(want), 1e-12),
            "dist {} want {want} got {}",
            dist.to_f64(),
            vel[0].x.to_f64()
        );
    }
}

/// Drag acts on the relative velocity to the *mean* neighbour velocity:
/// relative velocity u' = u / (1 + C_d rho N dt) (one implicit Euler step), N = 2 here.
#[test]
fn drag_uses_the_mean_neighbour_velocity_and_scales_with_count() {
    let c = drag_only(fx(1, 2));
    let rho = fx(3, 1);
    let dt = fx(1, 8);
    let fluid_pos = [Vec3Fix::ZERO, v(fx(1, 8), Fix128::ZERO, Fix128::ZERO)];
    let fluid_vel = [Vec3Fix::from_int(6, 0, 0), Vec3Fix::from_int(0, 2, 0)];
    let mut vel = [Vec3Fix::from_int(1, 1, 1)];
    apply_fluid_forces_to_cloth(
        &c,
        &[Vec3Fix::ZERO],
        &mut vel,
        &fluid_pos,
        &fluid_vel,
        rho,
        dt,
    );
    // mean fluid velocity (3,1,0); relative (-2,0,1); x = C_d rho N dt = 0.5*3*2/8 = 3/8,
    // applied fraction x / (1 + x) = 3/11
    let f = 3.0 / 11.0;
    let want = [1.0 + 2.0 * f, 1.0, 1.0 - 1.0 * f];
    assert!(close(vel[0].x, Fix128::from_f64(want[0]), 1e-12));
    assert!(close(vel[0].y, Fix128::from_f64(want[1]), 1e-12));
    assert!(close(vel[0].z, Fix128::from_f64(want[2]), 1e-12));
}

/// Each cloth particle sees only its own neighbourhood.
#[test]
fn cloth_particles_are_coupled_independently() {
    let c = drag_only(Fix128::ONE);
    let cloth = [Vec3Fix::ZERO, Vec3Fix::from_int(10, 0, 0)];
    let fluid = [Vec3Fix::ZERO];
    let mut vel = [Vec3Fix::from_int(4, 0, 0), Vec3Fix::from_int(4, 0, 0)];
    apply_fluid_forces_to_cloth(
        &c,
        &cloth,
        &mut vel,
        &fluid,
        &[Vec3Fix::ZERO],
        Fix128::ONE,
        fx(1, 4),
    );
    // x = 1 * 1 * 1 * 1/4: u' = 4 / 1.25 = 3.2
    assert!(close(vel[0].x, Fix128::from_f64(3.2), 1e-12));
    assert_eq!(vel[1].x, Fix128::from_int(4));
}

/// Zero density removes the drag term but not buoyancy.
#[test]
fn zero_fluid_density_removes_drag_only() {
    let c = ClothFluidCoupling {
        drag_coefficient: Fix128::ONE,
        buoyancy_factor: fx(1, 2),
        surface_tension: Fix128::ZERO,
    };
    let mut vel = [Vec3Fix::from_int(5, 0, 0)];
    apply_fluid_forces_to_cloth(
        &c,
        &[Vec3Fix::ZERO],
        &mut vel,
        &[Vec3Fix::ZERO],
        &[Vec3Fix::ZERO],
        Fix128::ZERO,
        fx(1, 2),
    );
    assert_eq!(vel[0].x, Fix128::from_int(5));
    assert_eq!(vel[0].y, fx(1, 4));
}

/// Permuting the particle order must not change any result bit (determinism).
#[test]
fn fluid_to_cloth_is_independent_of_fluid_order() {
    let c = ClothFluidCoupling::default();
    let fp = [
        v(fx(1, 10), fx(1, 7), Fix128::ZERO),
        v(fx(-1, 5), fx(1, 9), fx(1, 11)),
        v(fx(1, 3), Fix128::ZERO, fx(-1, 13)),
    ];
    let fv = [
        Vec3Fix::from_int(1, 2, 3),
        Vec3Fix::from_int(-2, 1, 0),
        v(fx(1, 3), fx(2, 7), fx(-5, 11)),
    ];
    let mut a = [Vec3Fix::from_int(1, 0, 2)];
    apply_fluid_forces_to_cloth(&c, &[Vec3Fix::ZERO], &mut a, &fp, &fv, fx(7, 3), fx(1, 60));
    let mut b = [Vec3Fix::from_int(1, 0, 2)];
    apply_fluid_forces_to_cloth(
        &c,
        &[Vec3Fix::ZERO],
        &mut b,
        &[fp[2], fp[0], fp[1]],
        &[fv[2], fv[0], fv[1]],
        fx(7, 3),
        fx(1, 60),
    );
    assert_eq!(a, b);
}

/// Boundary closed form: push = strength (R - d)/d along the normal, R = 1/4.
/// Fluid at (0.075, 0.1, 0): d = 0.125, (R-d)/d = 1, in front of a +y cloth normal.
#[test]
fn boundary_push_matches_strength_times_overlap_over_distance() {
    let s = fx(3, 1);
    let mut fv = [Vec3Fix::ZERO];
    apply_cloth_boundary_to_fluid(
        &[Vec3Fix::ZERO],
        &[Vec3Fix::UNIT_Y],
        &[v(fx(3, 40), fx(1, 10), Fix128::ZERO)],
        &mut fv,
        s,
    );
    assert!(
        close(fv[0].y, fx(3, 1), 1e-9),
        "front y = {}",
        fv[0].y.to_f64()
    );
    assert_eq!(fv[0].x, Fix128::ZERO);
    // back side: same magnitude, opposite sign
    let mut fb = [Vec3Fix::ZERO];
    apply_cloth_boundary_to_fluid(
        &[Vec3Fix::ZERO],
        &[Vec3Fix::UNIT_Y],
        &[v(fx(3, 40), fx(-1, 10), Fix128::ZERO)],
        &mut fb,
        s,
    );
    assert!(
        close(fb[0].y, fx(-3, 1), 1e-9),
        "back y = {}",
        fb[0].y.to_f64()
    );
    // d = R/2 -> (R-d)/d = 1; d = R/4 -> 3; check the 1/d growth with an on-axis fluid
    let mut f3 = [Vec3Fix::ZERO];
    apply_cloth_boundary_to_fluid(
        &[Vec3Fix::ZERO],
        &[Vec3Fix::UNIT_Y],
        &[v(Fix128::ZERO, fx(1, 16), Fix128::ZERO)],
        &mut f3,
        Fix128::ONE,
    );
    assert!(
        close(f3[0].y, fx(3, 1), 1e-9),
        "d = R/4: {}",
        f3[0].y.to_f64()
    );
}

/// Out of range (d >= R) is untouched, and the push vanishes continuously at
/// d = R (strict boundary, exactly at R no push).
#[test]
fn boundary_has_a_strict_quarter_metre_range() {
    for y in [fx(1, 4), fx(26, 100), fx(1, 2)] {
        let mut fv = [Vec3Fix::from_int(1, 2, 3)];
        apply_cloth_boundary_to_fluid(
            &[Vec3Fix::ZERO],
            &[Vec3Fix::UNIT_Y],
            &[v(Fix128::ZERO, y, Fix128::ZERO)],
            &mut fv,
            Fix128::from_int(5),
        );
        assert_eq!(fv[0], Vec3Fix::from_int(1, 2, 3));
    }
    // just inside: small and positive
    let mut fv = [Vec3Fix::ZERO];
    apply_cloth_boundary_to_fluid(
        &[Vec3Fix::ZERO],
        &[Vec3Fix::UNIT_Y],
        &[v(Fix128::ZERO, fx(249, 1000), Fix128::ZERO)],
        &mut fv,
        Fix128::ONE,
    );
    assert!(fv[0].y > Fix128::ZERO && fv[0].y < fx(1, 100));
}

/// The velocity pushes of different cloth particles accumulate; the order of
/// the particles does not change the bits.
#[test]
fn boundary_accumulates_over_cloth_particles_and_is_order_independent() {
    let cp = [
        v(Fix128::ZERO, fx(-1, 8), Fix128::ZERO),
        v(fx(1, 16), fx(-1, 10), fx(1, 20)),
    ];
    let cn = [Vec3Fix::UNIT_Y, v(fx(1, 3), fx(2, 3), fx(2, 3))];
    let fp = [Vec3Fix::ZERO];
    let mut a = [Vec3Fix::ZERO];
    apply_cloth_boundary_to_fluid(&cp, &cn, &fp, &mut a, fx(5, 3));
    let mut b = [Vec3Fix::ZERO];
    apply_cloth_boundary_to_fluid(&[cp[1], cp[0]], &[cn[1], cn[0]], &fp, &mut b, fx(5, 3));
    assert_eq!(a, b);
    assert!(
        a[0].y > fx(5, 3) * fx(1, 1),
        "two pushes must exceed a single unit push"
    );
}

/// Doc of `apply_cloth_boundary_to_fluid_with_residual`: the return value is
/// `||dv||_inf over the fluid particles`, i.e. the size of the actual velocity
/// change. Two cloth particles on opposite sides of a fluid particle give
/// opposite pushes whose net change is zero, so the report must be zero too,
/// not the single-pair magnitude.
#[test]
fn boundary_residual_is_the_norm_of_the_net_velocity_change() {
    let cp = [
        v(Fix128::ZERO, fx(-1, 8), Fix128::ZERO),
        v(Fix128::ZERO, fx(1, 8), Fix128::ZERO),
    ];
    let cn = [Vec3Fix::UNIT_Y, Vec3Fix::UNIT_Y];
    let fp = [Vec3Fix::ZERO];
    // A moving fluid particle: the report is the change, not the velocity
    let initial = Vec3Fix::from_int(1, 5, 0);
    let mut fv = [initial];
    let reported =
        apply_cloth_boundary_to_fluid_with_residual(&cp, &cn, &fp, &mut fv, Fix128::from_int(4));
    let dv = fv[0] - initial;
    let net = dv.x.abs().max(dv.y.abs()).max(dv.z.abs());
    assert_eq!(
        reported,
        net,
        "reported {} vs net |dv|_inf {}",
        reported.to_f64(),
        net.to_f64()
    );
}

/// Same claim, accumulating case: two same-side pushes of 4 add to 8 in the
/// velocity, so the report must be 8, not the single push of 4.
#[test]
fn boundary_residual_covers_accumulated_pushes() {
    let cp = [
        v(Fix128::ZERO, fx(-1, 8), Fix128::ZERO),
        v(Fix128::ZERO, fx(-1, 8), Fix128::ZERO),
    ];
    let cn = [Vec3Fix::UNIT_Y, Vec3Fix::UNIT_Y];
    let fp = [Vec3Fix::ZERO];
    let mut fv = [Vec3Fix::ZERO];
    let reported =
        apply_cloth_boundary_to_fluid_with_residual(&cp, &cn, &fp, &mut fv, Fix128::from_int(4));
    assert_eq!(reported, fv[0].y.abs());
}

/// Same claim, norm: `||dv||_inf` is the largest component of the net change,
/// not its L1 or L2 length. A push along `(3/5, 4/5, 0)` with magnitude 4 gives
/// `dv = (12/5, 16/5, 0)`: inf-norm 16/5, L2 4, L1 28/5.
#[test]
fn boundary_residual_is_the_largest_component_of_the_net_change() {
    let mut fv = [Vec3Fix::ZERO];
    let reported = apply_cloth_boundary_to_fluid_with_residual(
        &[Vec3Fix::ZERO],
        &[v(fx(3, 5), fx(4, 5), Fix128::ZERO)],
        &[v(Fix128::ZERO, fx(1, 8), Fix128::ZERO)],
        &mut fv,
        Fix128::from_int(4),
    );
    let net = fv[0].x.abs().max(fv[0].y.abs()).max(fv[0].z.abs());
    assert_eq!(reported, net);
    assert_eq!(net, fv[0].y.abs());
    assert!(fv[0].x > Fix128::ZERO && fv[0].x < fv[0].y);
}

/// Doc: surface tension "pulls cloth toward local fluid center". A fluid cloud
/// sitting at +x of a cloth particle, with nothing moving, must give the cloth
/// a +x velocity. The implementation adds `mean fluid velocity * surface_tension`,
/// which is zero for fluid at rest.
#[test]
// AUD-A-S2W2-005
fn surface_tension_pulls_the_cloth_toward_the_fluid_centre() {
    let c = ClothFluidCoupling {
        drag_coefficient: Fix128::ZERO,
        buoyancy_factor: Fix128::ZERO,
        surface_tension: Fix128::ONE,
    };
    let mut vel = [Vec3Fix::ZERO];
    apply_fluid_forces_to_cloth(
        &c,
        &[Vec3Fix::ZERO],
        &mut vel,
        &[v(fx(3, 10), Fix128::ZERO, Fix128::ZERO)],
        &[Vec3Fix::ZERO],
        Fix128::ONE,
        fx(1, 60),
    );
    assert!(vel[0].x > Fix128::ZERO, "vx = {}", vel[0].x.to_f64());
}

/// Explicit drag dv = -C_d rho N v dt overshoots when C_d rho N dt > 1 and flips
/// the cloth velocity past the fluid velocity (|v| grows), e.g. default drag 0.5
/// in water (rho 1000) at dt = 1/60 and one neighbour: factor 1 - 8.33 = -7.33.
#[test]
fn drag_never_overshoots_the_fluid_velocity() {
    let c = drag_only(fx(1, 2));
    let mut vel = [Vec3Fix::from_int(10, 0, 0)];
    apply_fluid_forces_to_cloth(
        &c,
        &[Vec3Fix::ZERO],
        &mut vel,
        &[Vec3Fix::ZERO],
        &[Vec3Fix::ZERO],
        Fix128::from_int(1000),
        fx(1, 60),
    );
    assert!(
        vel[0].x >= Fix128::ZERO && vel[0].x <= Fix128::from_int(10),
        "vx = {}",
        vel[0].x.to_f64()
    );
}

/// The residual variant reports the inf-norm of the net force exactly for a
/// drag-only scene (|F| = C_d rho N |v_rel|), independent of dt.
#[test]
fn reported_force_is_the_drag_magnitude() {
    let c = drag_only(fx(1, 2));
    let mut vel = [v(fx(-6, 1), fx(2, 1), Fix128::ZERO)];
    let f = apply_fluid_forces_to_cloth_with_residual(
        &c,
        &[Vec3Fix::ZERO],
        &mut vel,
        &[Vec3Fix::ZERO],
        &[Vec3Fix::ZERO],
        fx(3, 1),
        fx(1, 32),
    );
    assert_eq!(f, Fix128::from_int(9));
}

/// The push follows each cloth particle's own normal, not a fixed +y: with a +x
/// normal and the fluid in front (+x) at distance R/2 the whole push is along x.
#[test]
fn boundary_push_follows_the_supplied_normal() {
    let mut fv = [Vec3Fix::ZERO];
    apply_cloth_boundary_to_fluid(
        &[Vec3Fix::ZERO],
        &[Vec3Fix::from_int(1, 0, 0)],
        &[v(fx(1, 8), Fix128::ZERO, Fix128::ZERO)],
        &mut fv,
        Fix128::from_int(2),
    );
    assert_eq!(fv[0], v(Fix128::from_int(2), Fix128::ZERO, Fix128::ZERO));
    // a missing normal (shorter slice) falls back to +y, silently
    let mut fy = [Vec3Fix::ZERO];
    apply_cloth_boundary_to_fluid(
        &[Vec3Fix::ZERO],
        &[],
        &[v(Fix128::ZERO, fx(1, 8), Fix128::ZERO)],
        &mut fy,
        Fix128::from_int(2),
    );
    assert_eq!(fy[0], v(Fix128::ZERO, Fix128::from_int(2), Fix128::ZERO));
}

/// The boundary residual is a magnitude: a push along a negative axis still reports
/// a positive number (|dv| component), here 4 for a -y normal.
#[test]
fn boundary_residual_is_a_non_negative_magnitude() {
    let mut fv = [Vec3Fix::ZERO];
    let r = apply_cloth_boundary_to_fluid_with_residual(
        &[Vec3Fix::ZERO],
        &[v(Fix128::ZERO, fx(-1, 1), Fix128::ZERO)],
        &[v(Fix128::ZERO, fx(-1, 8), Fix128::ZERO)],
        &mut fv,
        Fix128::from_int(4),
    );
    assert_eq!(r, Fix128::from_int(4));
    assert_eq!(fv[0].y, Fix128::from_int(-4));
}

/// Doc of apply_fluid_forces_to_cloth_with_residual: the reported value is the force
/// before multiplication by dt, so it must not depend on dt. At dt == 0 the guard
/// returns 0 instead, so the reading jumps from F to 0 at exactly dt = 0.
#[test]
// AUD-A-S2W2-017
fn reported_force_does_not_vanish_at_zero_step() {
    let c = drag_only(fx(1, 2));
    let at = |dt: Fix128| {
        let mut vel = [Vec3Fix::from_int(10, 0, 0)];
        apply_fluid_forces_to_cloth_with_residual(
            &c,
            &[Vec3Fix::ZERO],
            &mut vel,
            &[Vec3Fix::ZERO],
            &[Vec3Fix::ZERO],
            Fix128::from_int(2),
            dt,
        )
    };
    assert_eq!(at(fx(1, 60)), Fix128::from_int(10));
    assert_eq!(at(Fix128::ZERO), at(fx(1, 60)));
}

/// AUD-A-S2W2-005 (centroid): with two fluid neighbours at (0.3, 0, 0) and
/// (0.1, 0.2, 0) the pull is toward their centroid (0.2, 0.1, 0): with k = 1,
/// no drag or buoyancy and dt = 1/60 the cloth velocity becomes (0.2, 0.1, 0)/60.
#[test]
fn surface_tension_pulls_toward_the_centroid_of_the_neighbours() {
    let c = ClothFluidCoupling {
        drag_coefficient: Fix128::ZERO,
        buoyancy_factor: Fix128::ZERO,
        surface_tension: Fix128::ONE,
    };
    let mut vel = [Vec3Fix::ZERO];
    let f = apply_fluid_forces_to_cloth_with_residual(
        &c,
        &[Vec3Fix::ZERO],
        &mut vel,
        &[
            v(fx(3, 10), Fix128::ZERO, Fix128::ZERO),
            v(fx(1, 10), fx(2, 10), Fix128::ZERO),
        ],
        &[Vec3Fix::ZERO, Vec3Fix::ZERO],
        Fix128::ONE,
        fx(1, 60),
    );
    assert!(
        (vel[0].x.to_f64() - 0.2 / 60.0).abs() < 1e-15,
        "{}",
        vel[0].x.to_f64()
    );
    assert!(
        (vel[0].y.to_f64() - 0.1 / 60.0).abs() < 1e-15,
        "{}",
        vel[0].y.to_f64()
    );
    assert!((f.to_f64() - 0.2).abs() < 1e-15, "reported {}", f.to_f64());
}

/// The pull is the offset from the cloth particle, not the centroid's position:
/// cloth at (1, 1, 0), fluid at (1.3, 1, 0), k = 1, dt = 1/60 gives vx = 0.3/60.
#[test]
fn surface_tension_is_the_offset_from_the_cloth_particle() {
    let c = ClothFluidCoupling {
        drag_coefficient: Fix128::ZERO,
        buoyancy_factor: Fix128::ZERO,
        surface_tension: Fix128::ONE,
    };
    let mut vel = [Vec3Fix::ZERO];
    apply_fluid_forces_to_cloth(
        &c,
        &[v(Fix128::ONE, Fix128::ONE, Fix128::ZERO)],
        &mut vel,
        &[v(fx(13, 10), Fix128::ONE, Fix128::ZERO)],
        &[Vec3Fix::ZERO],
        Fix128::ONE,
        fx(1, 60),
    );
    assert!(
        (vel[0].x.to_f64() - 0.3 / 60.0).abs() < 1e-15,
        "{}",
        vel[0].x.to_f64()
    );
    assert!(vel[0].y.to_f64().abs() < 1e-15, "{}", vel[0].y.to_f64());
}
