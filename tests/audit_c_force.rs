//! Audit oracles for `force`: signed vortex, magnetic and explosion forces and the
//! signed velocity change of `apply_force_fields` (the inline tests compare
//! magnitudes only).
//!
//! Expected values: vortex `F = strength * falloff * (a_hat x r_hat)` (right-handed
//! about the axis, `falloff = 1` inside the radius and `R / dist` beyond); magnetic
//! `F = strength / r^3 * (r_hat . m_hat) * m_hat` (the formula stated in the
//! `ForceField::Magnetic` code comment); explosion `F = strength (1 - d/R)^n` away
//! from the centre; `dv = F / m * dt`.

use alice_physics::force::{apply_force_fields, compute_force, ForceField, ForceFieldInstance};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::RigidBody;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn arr(v: Vec3Fix) -> [f64; 3] {
    [v.x.to_f64(), v.y.to_f64(), v.z.to_f64()]
}

fn close(a: [f64; 3], b: [f64; 3], tol: f64, what: &str) {
    for k in 0..3 {
        assert!(
            (a[k] - b[k]).abs() <= tol,
            "{what}: axis {k} got {a:?} want {b:?}"
        );
    }
}

fn at(p: [f64; 3]) -> RigidBody {
    RigidBody::new(v3(p[0], p[1], p[2]), Fix128::ONE)
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn vortex(axis: [f64; 3]) -> ForceField {
    ForceField::Vortex {
        center: v3(1.0, 0.0, -1.0),
        axis: v3(axis[0], axis[1], axis[2]),
        strength: fx(5.0),
        falloff_radius: fx(10.0),
    }
}

/// Signed tangential force for bodies around the axis, both axis orientations,
/// inside and beyond the falloff radius.
#[test]
fn vortex_force_is_signed_right_handed_about_the_axis() {
    let axes: [[f64; 3]; 3] = [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 2.0]];
    for axis in axes {
        let len = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        let a = [axis[0] / len, axis[1] / len, axis[2] / len];
        let offsets: [[f64; 3]; 6] = [
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0],
            [-2.0, 3.0, 0.0],
            [0.0, -4.0, -3.0],
            [20.0, 0.0, 0.0],
            [0.0, 1.0, -16.0],
        ];
        for offset in offsets {
            let along = offset[0] * a[0] + offset[1] * a[1] + offset[2] * a[2];
            let radial = [
                offset[0] - along * a[0],
                offset[1] - along * a[1],
                offset[2] - along * a[2],
            ];
            let dist =
                (radial[0] * radial[0] + radial[1] * radial[1] + radial[2] * radial[2]).sqrt();
            let want = if dist == 0.0 {
                [0.0; 3]
            } else {
                let r_hat = [radial[0] / dist, radial[1] / dist, radial[2] / dist];
                let falloff = if dist < 10.0 { 1.0 } else { 10.0 / dist };
                let t = cross(a, r_hat);
                [
                    5.0 * falloff * t[0],
                    5.0 * falloff * t[1],
                    5.0 * falloff * t[2],
                ]
            };
            let body = at([1.0 + offset[0], offset[1], -1.0 + offset[2]]);
            let got = arr(compute_force(&vortex(axis), &body));
            close(
                got,
                want,
                1e-12,
                &format!("axis {axis:?} offset {offset:?}"),
            );
        }
    }
    // the inline scenario: axis +y, body on +x: the force points to -z
    let f = compute_force(&vortex([0.0, 1.0, 0.0]), &at([3.0, 0.0, -1.0]));
    assert!(f.z.to_f64() < -4.9, "{:?}", arr(f));
}

fn magnet() -> ForceField {
    ForceField::Magnetic {
        position: Vec3Fix::ZERO,
        moment: v3(2.0, 0.0, 0.0),
        strength: fx(1000.0),
    }
}

/// On the dipole axis the force is `+strength / r^3` along the moment on the
/// `+moment` side and `-strength / r^3` on the other side; off axis it scales with
/// the axial cosine and vanishes in the equatorial plane.
#[test]
fn magnetic_force_is_signed_by_the_axial_cosine() {
    for (p, want) in [
        ([2.0, 0.0, 0.0], [125.0, 0.0, 0.0]),
        ([-2.0, 0.0, 0.0], [-125.0, 0.0, 0.0]),
        ([4.0, 0.0, 0.0], [15.625, 0.0, 0.0]),
        ([-4.0, 0.0, 0.0], [-15.625, 0.0, 0.0]),
        ([3.0, 4.0, 0.0], [1000.0 / 125.0 * 0.6, 0.0, 0.0]),
        ([-3.0, 0.0, 4.0], [-1000.0 / 125.0 * 0.6, 0.0, 0.0]),
        ([0.0, 2.0, 0.0], [0.0, 0.0, 0.0]),
    ] {
        let got = arr(compute_force(&magnet(), &at(p)));
        close(got, want, 1e-12 * 125.0, &format!("body at {p:?}"));
    }
}

/// Inverse cube with sign: on the `-moment` side, halving the distance multiplies
/// the (negative) force by exactly 8.
#[test]
fn magnetic_inverse_cube_keeps_the_sign_on_both_sides() {
    for side in [1.0, -1.0] {
        let near = compute_force(&magnet(), &at([side, 0.0, 0.0])).x.to_f64();
        let far = compute_force(&magnet(), &at([2.0 * side, 0.0, 0.0]))
            .x
            .to_f64();
        assert!((near - side * 1000.0).abs() < 1e-9, "near {near}");
        assert!((far - side * 125.0).abs() < 1e-9, "far {far}");
        assert!((near / far - 8.0).abs() < 1e-12, "ratio {}", near / far);
    }
}

/// `apply_force_fields`: `dv = F / m * dt` with the sign of F, for a magnet on both
/// sides of the dipole and for a vortex (mass 2, dt 1/8).
#[test]
fn apply_force_fields_changes_velocity_by_signed_force_over_mass_times_dt() {
    let mut bodies = vec![
        RigidBody::new(Vec3Fix::from_int(2, 0, 0), Fix128::from_int(2)),
        RigidBody::new(Vec3Fix::from_int(-2, 0, 0), Fix128::from_int(2)),
    ];
    let fields = [ForceFieldInstance::new(magnet())];
    apply_force_fields(&fields, &mut bodies, Fix128::from_ratio(1, 8));
    close(
        arr(bodies[0].velocity),
        [7.8125, 0.0, 0.0],
        1e-12,
        "+x side",
    );
    close(
        arr(bodies[1].velocity),
        [-7.8125, 0.0, 0.0],
        1e-12,
        "-x side",
    );

    let mut bodies = vec![RigidBody::new(v3(3.0, 0.0, -1.0), Fix128::from_int(2))];
    let fields = [ForceFieldInstance::new(vortex([0.0, 1.0, 0.0]))];
    apply_force_fields(&fields, &mut bodies, Fix128::from_ratio(1, 8));
    // F = 5 * (y x x) = (0, 0, -5); dv = -5 / 2 / 8
    close(
        arr(bodies[0].velocity),
        [0.0, 0.0, -0.3125],
        1e-12,
        "vortex",
    );
}

/// Explosion is signed away from the centre: `strength (1 - d/R)^n` along
/// `(p - c) / d` (radius 8, n = 2, strength 64), and a body on the far side gets the
/// opposite velocity change.
#[test]
fn explosion_force_points_away_from_the_centre_with_signed_velocity_change() {
    let field = ForceField::Explosion {
        center: v3(1.0, 1.0, 1.0),
        strength: fx(64.0),
        radius: fx(8.0),
        falloff_power: fx(2.0),
    };
    let cases: [([f64; 3], [f64; 3]); 2] = [
        ([5.0, 1.0, 1.0], [1.0, 0.0, 0.0]),
        ([1.0, -1.0, 1.0], [0.0, -1.0, 0.0]),
    ];
    for (p, d) in cases {
        let dist: f64 = ((p[0] - 1.0) * (p[0] - 1.0) + (p[1] - 1.0) * (p[1] - 1.0)).sqrt();
        let mag = 64.0 * (1.0 - dist / 8.0) * (1.0 - dist / 8.0);
        let got = arr(compute_force(&field, &at(p)));
        close(
            got,
            [mag * d[0], mag * d[1], mag * d[2]],
            1e-12,
            &format!("{p:?}"),
        );
    }
    let mut bodies = vec![
        RigidBody::new(v3(5.0, 1.0, 1.0), Fix128::ONE),
        RigidBody::new(v3(-3.0, 1.0, 1.0), Fix128::ONE),
    ];
    apply_force_fields(&[ForceFieldInstance::new(field)], &mut bodies, fx(0.5));
    // |F| = 64 * (1/2)^2 = 16, dv = 8 along +x and -x
    close(arr(bodies[0].velocity), [8.0, 0.0, 0.0], 1e-12, "+x body");
    close(arr(bodies[1].velocity), [-8.0, 0.0, 0.0], 1e-12, "-x body");
}
