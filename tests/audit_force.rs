//! Audit oracles for `alice_physics::force`.
//!
//! Expected values come from the closed forms of each field (inverse-square,
//! linear drag, `(1 - d/R)^n`, `s / r^3 * cos`, `axis x radial`), computed in
//! `f64` from the stated parameters; the particle system's own copy of the field
//! laws is compared against `compute_force` as an independent second reading.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]
#![allow(clippy::needless_range_loop)]

use alice_physics::force::{apply_force_fields, compute_force, ForceField, ForceFieldInstance};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::particle::{Particle, ParticleSystem};
use alice_physics::solver::RigidBody;
use std::sync::mpsc;
use std::time::Duration;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}
fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}
fn arr(v: Vec3Fix) -> [f64; 3] {
    [v.x.to_f64(), v.y.to_f64(), v.z.to_f64()]
}
fn body_at(p: [f64; 3]) -> RigidBody {
    RigidBody::new_dynamic(v3(p[0], p[1], p[2]), Fix128::ONE)
}
fn body_moving(p: [f64; 3], v: [f64; 3]) -> RigidBody {
    let mut b = body_at(p);
    b.velocity = v3(v[0], v[1], v[2]);
    b
}
fn close(a: [f64; 3], b: [f64; 3], tol: f64) -> bool {
    (0..3).all(|i| (a[i] - b[i]).abs() <= tol * (1.0 + b[i].abs()))
}
fn norm(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

// ------------------------------------------------------------------ Directional

#[test]
fn directional_force_is_direction_times_strength() {
    let f = ForceField::Directional {
        direction: v3(0.0, 0.6, -0.8),
        strength: fx(12.5),
    };
    let got = arr(compute_force(&f, &body_at([3.0, -1.0, 7.0])));
    assert!(close(got, [0.0, 7.5, -10.0], 1e-12), "{got:?}");
}

#[test]
#[ignore = "known defect: AUD-A-S4W3-014: Directional documents `direction` as normalized but does not normalize or check it (Vortex axis and Magnetic moment are normalized internally): direction (0,3,0) with strength 10 gives a force of magnitude 30"]
fn directional_force_magnitude_is_the_strength_for_a_non_unit_direction() {
    let f = ForceField::Directional {
        direction: v3(0.0, 3.0, 0.0),
        strength: fx(10.0),
    };
    let got = arr(compute_force(&f, &body_at([0.0; 3])));
    assert!((norm(got) - 10.0).abs() < 1e-9, "magnitude {}", norm(got));
}

// ------------------------------------------------------------------ Point

#[test]
fn point_field_is_inverse_square_toward_or_away_from_the_center() {
    let c = [1.0, 2.0, 3.0];
    for &(p, s) in &[([4.0, 6.0, 15.0], 500.0), ([-2.0, 2.0, 3.0], 90.0)] {
        let delta = [c[0] - p[0], c[1] - p[1], c[2] - p[2]];
        let r = norm(delta);
        let want = [
            delta[0] / r * s / (r * r),
            delta[1] / r * s / (r * r),
            delta[2] / r * s / (r * r),
        ];
        let attract = ForceField::Point {
            center: v3(c[0], c[1], c[2]),
            strength: fx(s),
            repulsive: false,
            max_force: fx(1e9),
        };
        let repel = ForceField::Point {
            center: v3(c[0], c[1], c[2]),
            strength: fx(s),
            repulsive: true,
            max_force: fx(1e9),
        };
        let a = arr(compute_force(&attract, &body_at(p)));
        let b = arr(compute_force(&repel, &body_at(p)));
        assert!(close(a, want, 1e-9), "{a:?} vs {want:?}");
        assert!(close(b, [-want[0], -want[1], -want[2]], 1e-9), "{b:?}");
    }
}

#[test]
fn point_field_is_clamped_at_max_force_near_the_center() {
    let f = ForceField::Point {
        center: Vec3Fix::ZERO,
        strength: fx(100.0),
        repulsive: false,
        max_force: fx(50.0),
    };
    // r = 1: 100 > 50 -> clamped; r = 2: 25 < 50 -> unclamped
    let near = arr(compute_force(&f, &body_at([1.0, 0.0, 0.0])));
    assert!(close(near, [-50.0, 0.0, 0.0], 1e-12), "{near:?}");
    let far = arr(compute_force(&f, &body_at([2.0, 0.0, 0.0])));
    assert!(close(far, [-25.0, 0.0, 0.0], 1e-12), "{far:?}");
}

#[test]
fn point_field_near_the_center_never_flips_sign_or_drops_below_the_cap() {
    // strength / r^2 grows without bound as r -> 0; the cap must hold at every distance.
    let f = ForceField::Point {
        center: Vec3Fix::ZERO,
        strength: fx(100.0),
        repulsive: false,
        max_force: fx(1000.0),
    };
    for &d in &[
        1e-10, 5e-10, 1e-9, 3e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.3,
    ] {
        let got = arr(compute_force(&f, &body_at([d, 0.0, 0.0])));
        assert!(
            (got[0] + 1000.0).abs() < 1e-6 && got[1].abs() < 1e-9,
            "distance {d}: force {got:?}, expected (-1000, 0, 0)"
        );
    }
}

// ------------------------------------------------------------------ Drag

#[test]
fn drag_is_minus_coefficient_times_velocity() {
    let f = ForceField::Drag {
        coefficient: fx(0.75),
    };
    for &v in &[[10.0, 0.0, 0.0], [1.0, -2.0, 3.0], [0.001, 0.002, -0.004]] {
        let got = arr(compute_force(&f, &body_moving([0.0; 3], v)));
        let want = [-0.75 * v[0], -0.75 * v[1], -0.75 * v[2]];
        assert!(close(got, want, 1e-9), "{got:?} vs {want:?}");
    }
    assert_eq!(arr(compute_force(&f, &body_at([0.0; 3]))), [0.0; 3]);
}

// ------------------------------------------------------------------ Buoyancy

#[test]
fn buoyancy_is_density_times_depth_upward_plus_linear_drag_below_the_surface() {
    let f = ForceField::Buoyancy {
        surface_y: fx(5.0),
        density: fx(10.0),
        drag: fx(0.5),
    };
    let got = arr(compute_force(
        &f,
        &body_moving([1.0, 2.0, 3.0], [4.0, -2.0, 1.0]),
    ));
    // depth 3: up = 30; drag = -0.5 v
    assert!(close(got, [-2.0, 30.0 + 1.0, -0.5], 1e-12), "{got:?}");
    // exactly at the surface and above: nothing, including no drag
    for y in [5.0, 6.0, 100.0] {
        let g = arr(compute_force(
            &f,
            &body_moving([0.0, y, 0.0], [4.0, -2.0, 1.0]),
        ));
        assert_eq!(g, [0.0; 3], "y = {y}");
    }
}

// ------------------------------------------------------------------ Vortex

#[test]
fn vortex_is_tangential_right_handed_with_constant_strength_inside_the_radius() {
    let f = ForceField::Vortex {
        center: Vec3Fix::ZERO,
        axis: Vec3Fix::UNIT_Y,
        strength: fx(10.0),
        falloff_radius: fx(100.0),
    };
    // body at +X: axis x radial = Y x X = -Z
    let got = arr(compute_force(&f, &body_at([5.0, 0.0, 0.0])));
    assert!(close(got, [0.0, 0.0, -10.0], 1e-9), "{got:?}");
    // body at +Z: Y x Z = +X
    let got = arr(compute_force(&f, &body_at([0.0, 0.0, 3.0])));
    assert!(close(got, [10.0, 0.0, 0.0], 1e-9), "{got:?}");
}

#[test]
fn vortex_ignores_the_height_along_the_axis_and_accepts_a_non_unit_axis() {
    let f = ForceField::Vortex {
        center: v3(1.0, 1.0, 1.0),
        axis: v3(0.0, 4.0, 0.0),
        strength: fx(7.0),
        falloff_radius: fx(10.0),
    };
    let flat = arr(compute_force(&f, &body_at([4.0, 1.0, 1.0])));
    let high = arr(compute_force(&f, &body_at([4.0, 8.0, 1.0])));
    assert!(close(flat, [0.0, 0.0, -7.0], 1e-9), "{flat:?}");
    assert!(close(high, flat, 1e-9), "{high:?} vs {flat:?}");
}

#[test]
fn vortex_decays_as_radius_over_distance_beyond_the_falloff_radius() {
    let f = ForceField::Vortex {
        center: Vec3Fix::ZERO,
        axis: Vec3Fix::UNIT_Y,
        strength: fx(10.0),
        falloff_radius: fx(2.0),
    };
    let inside = arr(compute_force(&f, &body_at([1.5, 0.0, 0.0])));
    let edge = arr(compute_force(&f, &body_at([2.0, 0.0, 0.0])));
    let out = arr(compute_force(&f, &body_at([8.0, 0.0, 0.0])));
    assert!((norm(inside) - 10.0).abs() < 1e-9);
    assert!((norm(edge) - 10.0).abs() < 1e-9, "continuous at the radius");
    assert!(
        (norm(out) - 10.0 * 2.0 / 8.0).abs() < 1e-9,
        "|F| = s R / d, got {}",
        norm(out)
    );
}

#[test]
fn vortex_on_the_axis_or_with_zero_falloff_radius_is_zero() {
    let f = ForceField::Vortex {
        center: Vec3Fix::ZERO,
        axis: Vec3Fix::UNIT_Y,
        strength: fx(10.0),
        falloff_radius: fx(2.0),
    };
    assert_eq!(arr(compute_force(&f, &body_at([0.0, 3.0, 0.0]))), [0.0; 3]);
    let z = ForceField::Vortex {
        center: Vec3Fix::ZERO,
        axis: Vec3Fix::UNIT_Y,
        strength: fx(10.0),
        falloff_radius: Fix128::ZERO,
    };
    assert_eq!(arr(compute_force(&z, &body_at([1.0, 0.0, 0.0]))), [0.0; 3]);
}

// ------------------------------------------------------------------ Explosion

fn explosion(power: f64) -> ForceField {
    ForceField::Explosion {
        center: v3(1.0, 0.0, 0.0),
        strength: fx(100.0),
        radius: fx(10.0),
        falloff_power: fx(power),
    }
}

#[test]
fn explosion_integer_powers_follow_the_documented_falloff() {
    // body at distance 4 along +x from the center: (1 - 4/10) = 0.6
    for &(n, want) in &[
        (1.0, 100.0 * 0.6),
        (2.0, 100.0 * 0.36),
        (3.0, 100.0 * 0.216),
        (0.0, 100.0),
    ] {
        let got = arr(compute_force(&explosion(n), &body_at([5.0, 0.0, 0.0])));
        assert!(
            close(got, [want, 0.0, 0.0], 1e-9),
            "power {n}: {got:?}, want {want}"
        );
    }
}

#[test]
fn explosion_points_away_from_the_center_and_vanishes_at_and_beyond_the_radius() {
    let got = arr(compute_force(&explosion(1.0), &body_at([1.0, -3.0, 4.0])));
    // distance 5, direction (0, -0.6, 0.8), magnitude 100 * 0.5
    assert!(close(got, [0.0, -30.0, 40.0], 1e-9), "{got:?}");
    assert_eq!(
        arr(compute_force(&explosion(1.0), &body_at([11.0, 0.0, 0.0]))),
        [0.0; 3],
        "at the radius"
    );
    assert_eq!(
        arr(compute_force(&explosion(2.0), &body_at([50.0, 9.0, 9.0]))),
        [0.0; 3]
    );
}

#[test]
#[ignore = "known defect: AUD-A-S4W3-015: Explosion documents the falloff as (1 - dist/radius)^falloff_power but truncates the exponent to its integer part (hi): power 0.5 is treated as 0 (constant 100 at distance 4), power 1.5 as 1"]
fn explosion_fractional_power_follows_the_documented_formula() {
    for &n in &[0.5, 1.5, 2.5] {
        let want = 100.0 * 0.6f64.powf(n);
        let got = arr(compute_force(&explosion(n), &body_at([5.0, 0.0, 0.0])));
        assert!(
            (got[0] - want).abs() < 1e-6,
            "power {n}: {} vs {want}",
            got[0]
        );
    }
}

#[test]
#[ignore = "known defect: AUD-A-S4W3-016: Explosion evaluates (1 - d/R)^n by an n-iteration loop on the truncated exponent with no bound: falloff_power = 2e9 does not return within 3 s (a 4e9 value would loop 4e9 times)"]
fn explosion_with_a_huge_power_returns_promptly() {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let r = arr(compute_force(&explosion(2.0e9), &body_at([5.0, 0.0, 0.0])));
        let _ = tx.send(r);
    });
    let r = rx.recv_timeout(Duration::from_secs(3));
    assert!(r.is_ok(), "compute_force did not return within 3 s");
}

// ------------------------------------------------------------------ Magnetic

#[test]
fn magnetic_force_is_strength_over_r_cubed_times_the_axial_cosine_along_the_moment() {
    let f = ForceField::Magnetic {
        position: v3(0.0, 0.0, 0.0),
        moment: v3(3.0, 0.0, 0.0),
        strength: fx(1000.0),
    };
    // on axis at r = 2: |F| = 1000 / 8 = 125, along +x (cos = 1)
    let on = arr(compute_force(&f, &body_at([2.0, 0.0, 0.0])));
    assert!(close(on, [125.0, 0.0, 0.0], 1e-9), "{on:?}");
    // off axis at (2, 1, 0): r = sqrt 5, cos = 2/sqrt 5
    let r = 5.0f64.sqrt();
    let off = arr(compute_force(&f, &body_at([2.0, 1.0, 0.0])));
    let want = 1000.0 / (r * r * r) * (2.0 / r);
    assert!(close(off, [want, 0.0, 0.0], 1e-9), "{off:?} vs {want}");
    // the opposite side of the dipole: reversed along the axis
    let back = arr(compute_force(&f, &body_at([-2.0, 0.0, 0.0])));
    assert!(close(back, [-125.0, 0.0, 0.0], 1e-9), "{back:?}");
    // equatorial plane: zero
    assert!(norm(arr(compute_force(&f, &body_at([0.0, 4.0, 3.0])))) < 1e-9);
}

#[test]
fn magnetic_force_falls_off_exactly_as_the_inverse_cube() {
    let f = ForceField::Magnetic {
        position: Vec3Fix::ZERO,
        moment: Vec3Fix::UNIT_X,
        strength: fx(10000.0),
    };
    let near = arr(compute_force(&f, &body_at([2.0, 0.0, 0.0])))[0];
    let far = arr(compute_force(&f, &body_at([4.0, 0.0, 0.0])))[0];
    assert!((near / far - 8.0).abs() < 1e-9, "ratio {}", near / far);
}

#[test]
#[ignore = "known defect: AUD-A-S4W3-017: the Magnetic code comment says a body on the dipole axis 'is attracted', but with delta = body - dipole the force along the axis points away from the dipole (repulsive for positive strength): body at +2 on a +x moment gets +125, not a pull toward the origin"]
fn magnetic_force_on_the_axis_pulls_toward_the_dipole_as_the_code_comment_says() {
    let f = ForceField::Magnetic {
        position: Vec3Fix::ZERO,
        moment: Vec3Fix::UNIT_X,
        strength: fx(1000.0),
    };
    let got = arr(compute_force(&f, &body_at([2.0, 0.0, 0.0])));
    assert!(got[0] < 0.0, "force {got:?} points away from the dipole");
}

#[test]
fn magnetic_force_close_to_the_dipole_keeps_its_sign_and_grows_down_to_1e_minus_5() {
    let f = ForceField::Magnetic {
        position: Vec3Fix::ZERO,
        moment: Vec3Fix::UNIT_X,
        strength: fx(1.0),
    };
    let mut prev = 0.0;
    for &d in &[1.0, 0.1, 0.01, 1e-3, 1e-4, 1e-5] {
        let got = arr(compute_force(&f, &body_at([d, 0.0, 0.0])))[0];
        let want = 1.0 / (d * d * d);
        assert!(got > 0.0, "distance {d}: sign flipped, got {got}");
        assert!(
            ((got - want) / want).abs() < 1e-3,
            "distance {d}: {got} vs {want}"
        );
        assert!(got > prev, "distance {d}: force did not grow");
        prev = got;
    }
}

#[test]
fn magnetic_force_does_not_collapse_to_zero_very_close_to_the_dipole() {
    let f = ForceField::Magnetic {
        position: Vec3Fix::ZERO,
        moment: Vec3Fix::UNIT_X,
        strength: fx(1.0),
    };
    let mut prev = 0.0;
    for &d in &[1e-6, 5.6e-7, 3.2e-7, 1e-7, 1e-8] {
        let got = arr(compute_force(&f, &body_at([d, 0.0, 0.0])))[0];
        assert!(
            got >= prev && got > 0.0,
            "distance {d}: force {got} after {prev}"
        );
        prev = got;
    }
}

// ------------------------------------------------------------------ apply_force_fields

#[test]
fn apply_adds_force_times_inverse_mass_times_dt_to_the_velocity() {
    let fields = [
        ForceFieldInstance::new(ForceField::Directional {
            direction: Vec3Fix::UNIT_X,
            strength: fx(10.0),
        }),
        ForceFieldInstance::new(ForceField::Directional {
            direction: Vec3Fix::UNIT_Y,
            strength: fx(-4.0),
        }),
    ];
    let mut bodies = [RigidBody::new_dynamic(Vec3Fix::ZERO, fx(4.0))];
    bodies[0].velocity = v3(1.0, 1.0, 1.0);
    apply_force_fields(&fields, &mut bodies, fx(0.5));
    // F = (10, -4, 0); a = F / 4; dv = a * 0.5
    assert!(close(
        arr(bodies[0].velocity),
        [1.0 + 1.25, 1.0 - 0.5, 1.0],
        1e-12
    ));
    // a second step accumulates linearly
    apply_force_fields(&fields, &mut bodies, fx(0.5));
    assert!(close(
        arr(bodies[0].velocity),
        [1.0 + 2.5, 1.0 - 1.0, 1.0],
        1e-12
    ));
}

#[test]
fn static_bodies_are_not_accelerated() {
    let fields = [ForceFieldInstance::new(ForceField::Directional {
        direction: Vec3Fix::UNIT_X,
        strength: fx(10.0),
    })];
    let mut bodies = [
        RigidBody::new_static(Vec3Fix::ZERO),
        RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE),
    ];
    apply_force_fields(&fields, &mut bodies, fx(1.0));
    assert_eq!(arr(bodies[0].velocity), [0.0; 3]);
    assert!(close(arr(bodies[1].velocity), [10.0, 0.0, 0.0], 1e-12));
}

#[test]
fn affected_bodies_and_enabled_filter_the_fields() {
    let mk = || RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE);
    let wind = ForceField::Directional {
        direction: Vec3Fix::UNIT_X,
        strength: fx(8.0),
    };
    // only body 1
    let only1 = [ForceFieldInstance::new(wind).with_affected_bodies(vec![1])];
    let mut b = [mk(), mk(), mk()];
    apply_force_fields(&only1, &mut b, fx(1.0));
    assert_eq!(arr(b[0].velocity), [0.0; 3]);
    assert!(close(arr(b[1].velocity), [8.0, 0.0, 0.0], 1e-12));
    assert_eq!(arr(b[2].velocity), [0.0; 3]);
    // an empty list affects nobody
    let none = [ForceFieldInstance::new(wind).with_affected_bodies(Vec::new())];
    let mut b = [mk(), mk()];
    apply_force_fields(&none, &mut b, fx(1.0));
    assert!(b.iter().all(|x| arr(x.velocity) == [0.0; 3]));
    // disabled
    let mut off = ForceFieldInstance::new(wind);
    off.enabled = false;
    let mut b = [mk()];
    apply_force_fields(&[off], &mut b, fx(1.0));
    assert_eq!(arr(b[0].velocity), [0.0; 3]);
    // no filter = everybody
    let all = [ForceFieldInstance::new(wind)];
    let mut b = [mk(), mk()];
    apply_force_fields(&all, &mut b, fx(1.0));
    assert!(b
        .iter()
        .all(|x| close(arr(x.velocity), [8.0, 0.0, 0.0], 1e-12)));
}

#[test]
fn field_force_is_evaluated_at_the_velocity_before_the_step() {
    // Drag acting on its own: dv = -c v dt / m, using the incoming velocity.
    let fields = [ForceFieldInstance::new(ForceField::Drag {
        coefficient: fx(2.0),
    })];
    let mut b = [RigidBody::new_dynamic(Vec3Fix::ZERO, fx(2.0))];
    b[0].velocity = v3(4.0, 0.0, 0.0);
    apply_force_fields(&fields, &mut b, fx(0.25));
    assert!(close(arr(b[0].velocity), [4.0 - 1.0, 0.0, 0.0], 1e-12));
}

// ------------------------------------------------------------------ particle system copy of the laws

/// The velocity change of a unit-mass particle over the particle system's fixed
/// dt = 1/60, times 60 = the force the particle system applies.
fn particle_force(field: &ForceField, pos: [f64; 3], vel: [f64; 3]) -> [f64; 3] {
    let mut ps = ParticleSystem::new(4, Vec3Fix::ZERO);
    ps.particles.push(Particle::new(
        v3(pos[0], pos[1], pos[2]),
        v3(vel[0], vel[1], vel[2]),
        fx(10.0),
        Fix128::ONE,
    ));
    ps.apply_force_field(field);
    let dv = arr(ps.particles[0].velocity);
    [
        (dv[0] - vel[0]) * 60.0,
        (dv[1] - vel[1]) * 60.0,
        (dv[2] - vel[2]) * 60.0,
    ]
}

#[test]
fn particle_system_agrees_with_compute_force_for_directional_point_drag_and_buoyancy() {
    let cases: Vec<(ForceField, [f64; 3], [f64; 3])> = vec![
        (
            ForceField::Directional {
                direction: Vec3Fix::UNIT_Y,
                strength: fx(3.0),
            },
            [1.0, 2.0, 3.0],
            [0.0; 3],
        ),
        (
            ForceField::Point {
                center: Vec3Fix::ZERO,
                strength: fx(90.0),
                repulsive: false,
                max_force: fx(1000.0),
            },
            [3.0, 4.0, 0.0],
            [0.0; 3],
        ),
        (
            ForceField::Point {
                center: Vec3Fix::ZERO,
                strength: fx(90.0),
                repulsive: true,
                max_force: fx(5.0),
            },
            [1.0, 1.0, 1.0],
            [0.0; 3],
        ),
        (
            ForceField::Drag {
                coefficient: fx(0.5),
            },
            [0.0; 3],
            [2.0, -1.0, 4.0],
        ),
        (
            ForceField::Buoyancy {
                surface_y: fx(5.0),
                density: fx(10.0),
                drag: fx(0.5),
            },
            [0.0, 2.0, 0.0],
            [1.0, 0.0, 0.0],
        ),
    ];
    for (f, p, v) in cases {
        let a = arr(compute_force(&f, &body_moving(p, v)));
        let b = particle_force(&f, p, v);
        assert!(
            close(a, b, 1e-6),
            "{f:?}: compute_force {a:?} vs particle {b:?}"
        );
    }
}

#[test]
#[ignore = "known defect: AUD-B-S4W3-001: particle.rs keeps a private second implementation of the ForceField laws that disagrees with force::compute_force: Magnetic (no axial cosine: equatorial force 15.6 vs 0, off-axis 89.4 vs 80), Explosion with power 0 (treated as 1: 60 vs 100), Vortex off the axis plane (falloff uses the full distance: 2.48 vs 10)"]
fn particle_system_agrees_with_compute_force_for_magnetic_explosion_and_vortex() {
    let cases: Vec<(&str, ForceField, [f64; 3])> = vec![
        (
            "magnetic equatorial",
            ForceField::Magnetic {
                position: Vec3Fix::ZERO,
                moment: Vec3Fix::UNIT_X,
                strength: fx(1000.0),
            },
            [0.0, 4.0, 0.0],
        ),
        (
            "magnetic off axis",
            ForceField::Magnetic {
                position: Vec3Fix::ZERO,
                moment: Vec3Fix::UNIT_X,
                strength: fx(1000.0),
            },
            [2.0, 1.0, 0.0],
        ),
        ("explosion power 0", explosion(0.0), [5.0, 0.0, 0.0]),
        (
            "vortex off the plane",
            ForceField::Vortex {
                center: Vec3Fix::ZERO,
                axis: Vec3Fix::UNIT_Y,
                strength: fx(10.0),
                falloff_radius: fx(2.0),
            },
            [1.0, 8.0, 0.0],
        ),
    ];
    let mut bad = Vec::new();
    for (name, f, p) in cases {
        let a = arr(compute_force(&f, &body_at(p)));
        let b = particle_force(&f, p, [0.0; 3]);
        if !close(a, b, 1e-6) {
            bad.push(format!("{name}: compute_force {a:?} vs particle {b:?}"));
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("; "));
}

#[test]
fn explosion_force_is_zero_exactly_at_the_radius_for_every_power() {
    // doc: "zero force beyond radius"; pins the boundary as exclusive for power 0 too
    for &n in &[0.0, 1.0, 2.0] {
        let got = arr(compute_force(&explosion(n), &body_at([11.0, 0.0, 0.0])));
        assert_eq!(got, [0.0; 3], "power {n}");
    }
}
