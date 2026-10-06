//! Audit oracles for `fsi_advanced`.
//!
//! Expected values come from textbook closed forms evaluated in `f64` on
//! dyadic inputs (exactly representable in `Fix128`), or from identities that
//! do not call the code under test twice with the same formula:
//!
//! ```text
//! F_d = -1/2 rho Cd A |v_rel| v_rel        (v_rel = v_solid - v_fluid)
//! F_b = rho V |g|  along +Y                (doc: "|g|")
//! tau = sum r_i x F_i,   r_i = p_i - ref
//! tau(ref2) = tau(ref1) - (ref2 - ref1) x F_net        (translation identity)
//! ```
//!
//! Author: Moroya Sakamoto

#![allow(clippy::disallowed_methods)]

use alice_physics::fsi_advanced::{
    aggregate_forces, buoyancy_force, drag_force, react_back_pressure, SolidSample,
};
use alice_physics::math::{Fix128, Vec3Fix};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}
fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}
fn f3(v: Vec3Fix) -> [f64; 3] {
    [v.x.to_f64(), v.y.to_f64(), v.z.to_f64()]
}
fn close(a: f64, b: f64, rel: f64) -> bool {
    (a - b).abs() <= rel * a.abs().max(b.abs()).max(1e-300)
}
fn smp(p: Vec3Fix, v: Vec3Fix, a: f64, vol: f64) -> SolidSample {
    SolidSample {
        position: p,
        velocity: v,
        area_m2: fx(a),
        volume_m3: fx(vol),
    }
}
fn drag_ref(vs: [f64; 3], vf: [f64; 3], rho: f64, cd: f64, a: f64) -> [f64; 3] {
    let r = [vs[0] - vf[0], vs[1] - vf[1], vs[2] - vf[2]];
    let m = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
    let k = -0.5 * rho * cd * a * m;
    [k * r[0], k * r[1], k * r[2]]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Drag against the textbook formula over a sweep of speeds spanning 6 decades
/// and fully 3D directions (relative 1e-9).
#[test]
fn drag_matches_closed_form_over_speed_sweep() {
    let rho = 1.25;
    let cd = 0.75;
    let a = 2.5;
    let dirs = [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.5, -0.25, 0.75],
        [-3.0, 4.0, 12.0],
    ];
    let vf = [0.25, -0.5, 0.125];
    for scale in [1e-3, 1e-2, 0.5, 1.0, 8.0, 64.0, 1000.0] {
        for d in dirs {
            let vs = [
                vf[0] + scale * d[0],
                vf[1] + scale * d[1],
                vf[2] + scale * d[2],
            ];
            let s = smp(Vec3Fix::default(), v3(vs[0], vs[1], vs[2]), a, 0.0);
            let got = f3(drag_force(&s, v3(vf[0], vf[1], vf[2]), fx(rho), fx(cd)));
            let want = drag_ref(vs, vf, rho, cd, a);
            for i in 0..3 {
                // Fix128 absolute resolution 2^-64 bounds small components.
                assert!(
                    close(got[i], want[i], 1e-9) || (got[i] - want[i]).abs() < 1e-15,
                    "scale {scale} dir {d:?} comp {i}: got {} want {}",
                    got[i],
                    want[i]
                );
            }
        }
    }
}

/// Drag is anti-parallel to v_rel: F x v_rel = 0 and F . v_rel < 0.
#[test]
fn drag_is_antiparallel_to_relative_velocity() {
    let vs = [3.0, -2.0, 6.0];
    let vf = [1.0, 1.0, -1.0];
    let s = smp(Vec3Fix::default(), v3(vs[0], vs[1], vs[2]), 1.0, 0.0);
    let f = f3(drag_force(&s, v3(vf[0], vf[1], vf[2]), fx(1000.0), fx(1.0)));
    let r = [vs[0] - vf[0], vs[1] - vf[1], vs[2] - vf[2]];
    let c = cross(f, r);
    let fm = (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt();
    let rm = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
    assert!(c.iter().all(|x| x.abs() < 1e-6 * fm * rm));
    assert!(f[0] * r[0] + f[1] * r[1] + f[2] * r[2] < 0.0);
}

/// Drag depends on v_solid - v_fluid only (Galilean invariance), exactly.
#[test]
fn drag_is_galilean_invariant() {
    let s1 = smp(Vec3Fix::default(), v3(3.0, 1.0, -2.0), 1.5, 0.0);
    let s2 = smp(
        Vec3Fix::default(),
        v3(3.0 + 7.0, 1.0 - 5.0, -2.0 + 9.0),
        1.5,
        0.0,
    );
    let a = drag_force(&s1, v3(1.0, 0.5, 0.25), fx(900.0), fx(0.5));
    let b = drag_force(
        &s2,
        v3(1.0 + 7.0, 0.5 - 5.0, 0.25 + 9.0),
        fx(900.0),
        fx(0.5),
    );
    assert_eq!(a, b);
}

/// Drag is linear in rho, Cd and A, and exactly odd in v_rel (F(-v) = -F(v)).
#[test]
fn drag_linear_in_coefficients_and_odd_in_velocity() {
    let base = |rho: f64, cd: f64, a: f64, vx: f64| {
        f3(drag_force(
            &smp(Vec3Fix::default(), v3(vx, 0.5, 0.0), a, 0.0),
            Vec3Fix::default(),
            fx(rho),
            fx(cd),
        ))
    };
    let f0 = base(2.0, 0.5, 1.5, 3.0);
    for (f1, name) in [
        (base(4.0, 0.5, 1.5, 3.0), "rho"),
        (base(2.0, 1.0, 1.5, 3.0), "Cd"),
        (base(2.0, 0.5, 3.0, 3.0), "A"),
    ] {
        for i in 0..3 {
            assert!(close(f1[i], 2.0 * f0[i], 1e-12), "doubling {name} comp {i}");
        }
    }
    let m = f3(drag_force(
        &smp(Vec3Fix::default(), v3(-3.0, -0.5, 0.0), 1.5, 0.0),
        Vec3Fix::default(),
        fx(2.0),
        fx(0.5),
    ));
    for i in 0..3 {
        assert!(close(m[i], -f0[i], 1e-12), "odd comp {i}");
    }
}

/// Quadratic law at tight tolerance: F(2v)/F(v) = 4 (the in-crate test allows 1e-2).
#[test]
fn drag_quadratic_ratio_is_four_tightly() {
    let f = |vx: f64| {
        drag_force(
            &smp(Vec3Fix::default(), v3(vx, 0.0, 0.0), 1.0, 0.0),
            Vec3Fix::default(),
            fx(1000.0),
            fx(1.0),
        )
        .x
        .to_f64()
    };
    for v in [0.5, 1.0, 3.0, 17.0] {
        assert!(close(f(2.0 * v) / f(v), 4.0, 1e-10), "v = {v}");
    }
    // Absolute value: -1/2 * 1000 * 1 * 1 * 3 * 3 = -4500
    assert_eq!(f(3.0), -4500.0);
}

/// Archimedes: F_b = rho V g along +Y, x = z = 0 exactly; linear in each factor.
#[test]
fn buoyancy_archimedes_closed_form() {
    for (rho, vol, g) in [(1000.0, 0.25, 9.75), (1.25, 8.0, 10.0), (998.0, 0.001, 9.5)] {
        let f = f3(buoyancy_force(
            &smp(Vec3Fix::default(), Vec3Fix::default(), 0.0, vol),
            fx(rho),
            fx(g),
        ));
        assert_eq!(f[0], 0.0);
        assert_eq!(f[2], 0.0);
        assert!(
            close(f[1], rho * vol * g, 1e-9),
            "{rho} {vol} {g}: {}",
            f[1]
        );
    }
}

/// AUD-A-S4W1-002 (known defect): the doc states `F_b = rho V |g|` along +Y,
/// but `gravity_m_per_s2` is multiplied in unsigned. A gravity vector's
/// Y component is conventionally negative (-9.81), and that flips the
/// buoyancy downward: F_b(g = -10) = -rho V 10 instead of +rho V 10.
#[test]
// AUD-A-S4W1-002
fn buoyancy_uses_gravity_magnitude() {
    let s = smp(Vec3Fix::default(), Vec3Fix::default(), 0.0, 1.0);
    let up = buoyancy_force(&s, fx(1000.0), fx(10.0)).y.to_f64();
    let dn = buoyancy_force(&s, fx(1000.0), fx(-10.0)).y.to_f64();
    assert_eq!(up, 10000.0);
    assert_eq!(
        dn, 10000.0,
        "doc: rho*V*|g| must be upward for a negative g"
    );
}

fn scene() -> Vec<SolidSample> {
    vec![
        smp(v3(1.0, 0.5, -0.25), v3(2.0, -1.0, 0.5), 1.5, 0.25),
        smp(v3(-0.5, 2.0, 1.0), v3(-1.0, 0.25, 3.0), 0.5, 0.125),
        smp(v3(0.25, -1.5, 2.0), v3(0.0, 0.0, 0.0), 2.0, 0.5),
    ]
}

fn fluid(p: Vec3Fix) -> Vec3Fix {
    // Linear rotating field v_f = (y, -x, 0.25 z): depends on the position the sampler is given.
    Vec3Fix::new(p.y, Fix128::ZERO - p.x, p.z.half().half())
}

/// aggregate = sum over samples of drag + buoyancy, torque = sum r x F, with
/// the sampler evaluated at each sample's own position (independent f64 sum).
#[test]
fn aggregate_matches_independent_sum_and_torque() {
    let rho = 1.5;
    let cd = 0.5;
    let g = 9.5;
    let refp = v3(0.5, -0.25, 0.125);
    let (f, t) = aggregate_forces(&scene(), fluid, fx(rho), fx(cd), fx(g), refp);
    let mut fs = [0.0; 3];
    let mut ts = [0.0; 3];
    for s in scene() {
        let p = f3(s.position);
        let vf = [p[1], -p[0], 0.25 * p[2]];
        let mut fi = drag_ref(f3(s.velocity), vf, rho, cd, s.area_m2.to_f64());
        fi[1] += rho * s.volume_m3.to_f64() * g;
        let r = [p[0] - 0.5, p[1] + 0.25, p[2] - 0.125];
        let tau = cross(r, fi);
        for i in 0..3 {
            fs[i] += fi[i];
            ts[i] += tau[i];
        }
    }
    let (f, t) = (f3(f), f3(t));
    for i in 0..3 {
        assert!(
            close(f[i], fs[i], 1e-9),
            "F comp {i}: {} vs {}",
            f[i],
            fs[i]
        );
        assert!(
            close(t[i], ts[i], 1e-9),
            "tau comp {i}: {} vs {}",
            t[i],
            ts[i]
        );
    }
}

/// Translation identity: tau about ref2 = tau about ref1 - (ref2-ref1) x F_net.
#[test]
fn aggregate_torque_translation_identity() {
    let args = |r: Vec3Fix| aggregate_forces(&scene(), fluid, fx(1.5), fx(0.5), fx(9.5), r);
    let r1 = v3(0.0, 0.0, 0.0);
    let r2 = v3(1.5, -2.0, 0.5);
    let (f1, t1) = args(r1);
    let (f2, t2) = args(r2);
    assert_eq!(f1, f2, "force does not depend on the reference point");
    let c = cross([1.5, -2.0, 0.5], f3(f1));
    let (t1, t2) = (f3(t1), f3(t2));
    for i in 0..3 {
        assert!(close(t2[i], t1[i] - c[i], 1e-9), "comp {i}");
    }
}

/// Samples are summed independently: permuting them leaves the result unchanged
/// (up to the fixed-point rounding of the order of summation: exact for Fix128 add).
#[test]
fn aggregate_is_permutation_invariant() {
    let mut s = scene();
    let a = aggregate_forces(&s, fluid, fx(1.5), fx(0.5), fx(9.5), Vec3Fix::default());
    s.reverse();
    let b = aggregate_forces(&s, fluid, fx(1.5), fx(0.5), fx(9.5), Vec3Fix::default());
    assert_eq!(a, b);
}

/// The sampler is called exactly once per sample with that sample's position.
#[test]
fn aggregate_calls_sampler_once_per_sample_at_its_position() {
    let calls = std::cell::RefCell::new(Vec::new());
    let sc = scene();
    let _ = aggregate_forces(
        &sc,
        |p| {
            calls.borrow_mut().push(p);
            Vec3Fix::default()
        },
        fx(1.0),
        fx(1.0),
        fx(10.0),
        Vec3Fix::default(),
    );
    let got = calls.into_inner();
    let want: Vec<Vec3Fix> = sc.iter().map(|s| s.position).collect();
    assert_eq!(got, want);
}

/// react_back_pressure deposits exactly -F at each sample position, in order.
#[test]
fn react_back_pressure_pairs_and_negates() {
    let sc = scene();
    let forces = [v3(1.0, 2.0, 3.0), v3(-4.0, 0.5, 0.0), v3(0.0, 0.0, -7.0)];
    let mut got = Vec::new();
    react_back_pressure(&sc, &forces, |p, f| got.push((p, f)));
    assert_eq!(got.len(), 3);
    for i in 0..3 {
        assert_eq!(got[i].0, sc[i].position);
        let fr = f3(forces[i]);
        assert_eq!(f3(got[i].1), [-fr[0], -fr[1], -fr[2]]);
    }
}

/// Newton III: the deposited impulse sums to minus the solid force.
#[test]
fn react_back_pressure_total_is_minus_solid_total() {
    let sc = scene();
    let forces = [v3(1.0, 2.0, 3.0), v3(-4.0, 0.5, 0.0), v3(0.0, 0.0, -7.0)];
    let mut sum = [0.0; 3];
    react_back_pressure(&sc, &forces, |_, f| {
        let a = f3(f);
        for i in 0..3 {
            sum[i] += a[i];
        }
    });
    assert_eq!(sum, [3.0, -2.5, 4.0]);
}

/// AUD-A-S4W1-003 (known defect, precondition unchecked): when the force slice
/// is shorter than the sample slice, `zip` silently drops the surplus samples:
/// no panic, no error, and fewer reactions than samples are deposited, so
/// Newton III (sum of deposits = -sum of solid forces over ALL samples) is
/// quietly broken. Expected: either every sample is served or the call fails
/// loudly.
#[test]
fn react_back_pressure_does_not_silently_truncate() {
    let sc = scene();
    let forces = [v3(1.0, 0.0, 0.0), v3(2.0, 0.0, 0.0)];
    let mut n = 0;
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        react_back_pressure(&sc, &forces, |_, _| n += 1);
    }));
    assert!(
        r.is_err() || n == sc.len(),
        "silently deposited {n} of {} samples",
        sc.len()
    );
}

/// Empty / zero inputs: no deposits, zero force; zero area gives zero drag.
#[test]
fn degenerate_inputs_are_zero_without_panic() {
    let mut n = 0;
    react_back_pressure(&[], &[], |_, _| n += 1);
    assert_eq!(n, 0);
    let s = smp(Vec3Fix::default(), v3(5.0, 0.0, 0.0), 0.0, 0.0);
    assert_eq!(
        drag_force(&s, Vec3Fix::default(), fx(1000.0), fx(1.0)),
        Vec3Fix::default()
    );
    assert_eq!(buoyancy_force(&s, fx(1000.0), fx(10.0)), Vec3Fix::default());
}
