//! One step of each cloth-fluid coupling direction, checked by hand
//!
//! Reaches `ClothFluidCoupling`, `apply_fluid_forces_to_cloth`,
//! `apply_fluid_forces_to_cloth_with_residual`,
//! `apply_cloth_boundary_to_fluid` and
//! `apply_cloth_boundary_to_fluid_with_residual`.
//!
//! Closed forms, from the documented model:
//! - fluid -> cloth, for a cloth particle with `N` fluid neighbours within
//!   `1/2` whose mean velocity is `v_f` and centroid `x_f`: with
//!   `u = v_c - v_f` and `c = C_d * rho * N`, the drag is one implicit step
//!   `u -> u / (1 + c dt)`, buoyancy adds `b N dt` along `+y` and surface
//!   tension pulls toward the fluid, adding `s (x_f - x_c) dt`. The reported
//!   force is `|| b N y_hat + s (x_f - x_c) - c u ||_inf`, evaluated before
//!   `dt`. With the default coupling (`C_d = 1/2`, `b = 1/10`, `s = 1/100`),
//!   `rho = 1`, `N = 1`, `dt = 1/2`, `v_c = (2, 0, 0)`, `v_f = (0, 0, 1)` and
//!   the neighbour `1/4` along `+x`: `c dt = 1/4`, so the drag removes
//!   `u / 5`, tension adds `(1/800, 0, 0)`, and
//!   `v_c' = (8/5 + 1/800, 1/20, 1/5)`; the reported force is
//!   `max(|1/400 - 1|, |1/10|, |1/2|) = 399/400`
//! - cloth -> fluid: a fluid particle at distance `d < 1/4` from a cloth
//!   particle is pushed along the cloth normal (sign by side) by
//!   `strength * (1/4 - d) / d`. With strength `1/2`: `d = 1/8` gives `1/2`,
//!   `d = 1/16` gives `3/2`. The reported value is the largest component
//!
//! Run with: `cargo run --example cloth_fluid_interface`

use alice_physics::cloth_fluid::{
    apply_cloth_boundary_to_fluid_with_residual, apply_fluid_forces_to_cloth_with_residual,
};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::{
    apply_cloth_boundary_to_fluid, apply_fluid_forces_to_cloth, ClothFluidCoupling,
};

fn r(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn v(x: Fix128, y: Fix128, z: Fix128) -> Vec3Fix {
    Vec3Fix::new(x, y, z)
}

fn close(got: Vec3Fix, want: (f64, f64, f64), what: &str) {
    let g = (got.x.to_f64(), got.y.to_f64(), got.z.to_f64());
    for (a, b) in [(g.0, want.0), (g.1, want.1), (g.2, want.2)] {
        assert!(
            (a - b).abs() < 1e-12,
            "{what}: got {g:?}, closed form {want:?}"
        );
    }
}

fn main() {
    let zero = Fix128::ZERO;
    let coupling = ClothFluidCoupling::default();
    assert_eq!(coupling.drag_coefficient, r(1, 2));
    assert_eq!(coupling.buoyancy_factor, r(1, 10));
    assert_eq!(coupling.surface_tension, r(1, 100));

    // ---- fluid -> cloth ---------------------------------------------------
    // Cloth particle 0 has one fluid neighbour 1/4 away; particle 1 is 10 away
    // from any fluid and must be left alone.
    let cloth_pos = [Vec3Fix::ZERO, v(Fix128::from_int(10), zero, zero)];
    let cloth_vel0 = [
        v(Fix128::from_int(2), zero, zero),
        v(zero, Fix128::ONE, zero),
    ];
    let fluid_pos = [v(r(1, 4), zero, zero)];
    let fluid_vel = [v(zero, zero, Fix128::ONE)];
    let rho = Fix128::ONE;
    let dt = r(1, 2);

    let mut cloth_vel = cloth_vel0;
    let force = apply_fluid_forces_to_cloth_with_residual(
        &coupling,
        &cloth_pos,
        &mut cloth_vel,
        &fluid_pos,
        &fluid_vel,
        rho,
        dt,
    );
    println!(
        "[cloth_fluid] cloth velocity {:?} -> ({}, {}, {}), interface force {}",
        (2, 0, 0),
        cloth_vel[0].x.to_f64(),
        cloth_vel[0].y.to_f64(),
        cloth_vel[0].z.to_f64(),
        force.to_f64()
    );
    close(
        cloth_vel[0],
        (1.6 + 1.0 / 800.0, 0.05, 0.2),
        "submerged cloth particle",
    );
    assert_eq!(cloth_vel[1], cloth_vel0[1], "no neighbour, no force");
    assert!(
        (force.to_f64() - 399.0 / 400.0).abs() < 1e-12,
        "interface force"
    );

    let mut via_wrapper = cloth_vel0;
    apply_fluid_forces_to_cloth(
        &coupling,
        &cloth_pos,
        &mut via_wrapper,
        &fluid_pos,
        &fluid_vel,
        rho,
        dt,
    );
    assert_eq!(
        via_wrapper, cloth_vel,
        "the wrapper applies the same update"
    );

    // ---- cloth -> fluid ---------------------------------------------------
    let sheet = [Vec3Fix::ZERO];
    let up = [Vec3Fix::UNIT_Y];
    let strength = r(1, 2);
    let fluid = [
        v(zero, r(1, 8), zero),  // front side, d = 1/8
        v(zero, r(-1, 8), zero), // back side, d = 1/8
        v(r(3, 10), zero, zero), // outside the 1/4 radius
    ];
    let mut fluid_v = [Vec3Fix::ZERO; 3];
    let corr =
        apply_cloth_boundary_to_fluid_with_residual(&sheet, &up, &fluid, &mut fluid_v, strength);
    close(fluid_v[0], (0.0, 0.5, 0.0), "front side pushed along +n");
    close(fluid_v[1], (0.0, -0.5, 0.0), "back side pushed along -n");
    assert_eq!(fluid_v[2], Vec3Fix::ZERO, "outside the radius");
    assert!((corr.to_f64() - 0.5).abs() < 1e-12, "largest correction");

    let close_in = [v(zero, r(1, 16), zero)];
    let mut close_v = [Vec3Fix::ZERO];
    apply_cloth_boundary_to_fluid(&sheet, &up, &close_in, &mut close_v, strength);
    close(
        close_v[0],
        (0.0, 1.5, 0.0),
        "d = 1/16: (1/4 - 1/16) * 16 / 2",
    );

    println!("[cloth_fluid] both coupling directions match the hand-worked step");
}
