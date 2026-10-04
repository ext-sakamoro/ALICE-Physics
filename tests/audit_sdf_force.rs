//! Audit oracles for `sdf_force`.
//!
//! The existing oracle only evaluates on-axis points of an origin-centred,
//! unrotated, unscaled sphere and rebuilds the module's own formulas from the
//! same `Fix128` operators. These oracles instead use a generic frame: a
//! half-space SDF whose local normal `n_l` and offset `h` are arbitrary, placed
//! at position `c` with rotation `R` (axis-angle) and uniform scale `s`. The
//! closed form is written in `f64` from geometry:
//!
//! ```text
//! n_w = R n_l,   d(p) = n_w . (p - c) - s h          (world signed distance)
//! Attract   F = -sign(d) n_w min(k |d|, F_max)       (toward the surface)
//! Repel     F =  sign(d) n_w k (1 - |d|/range)^2     (|d| <= range)
//! Contain   F = -c v   (d <= 0),   F = -k d n_w - c v   (d > 0)
//! Flow      F = k (1 - |d|/L) t/|t|,   t = f - (f.n_w) n_w
//! Vortex    F = k (1 - |d|/L) (a x n_w)/|a x n_w|
//! apply     dv = F invm dt
//! ```
//!
//! The SDF pipeline is f32 (`ClosureSdf`), so comparisons use a relative
//! 2e-4.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::sdf_collider::{ClosureSdf, SdfCollider};
use alice_physics::sdf_force::{apply_sdf_force_fields, compute_sdf_force};
use alice_physics::{RigidBody, SdfForceField, SdfForceType};

type V = [f64; 3];

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}
fn v3(a: V) -> Vec3Fix {
    Vec3Fix::new(fx(a[0]), fx(a[1]), fx(a[2]))
}
fn arr(v: Vec3Fix) -> V {
    [v.x.to_f64(), v.y.to_f64(), v.z.to_f64()]
}
fn dot(a: V, b: V) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V, b: V) -> V {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn scale(a: V, k: f64) -> V {
    [a[0] * k, a[1] * k, a[2] * k]
}
fn sub(a: V, b: V) -> V {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn norm(a: V) -> f64 {
    dot(a, a).sqrt()
}
fn unit(a: V) -> V {
    scale(a, 1.0 / norm(a))
}
/// Rodrigues rotation of `v` about unit `axis` by `ang` (f64 reference).
fn rot(axis: V, ang: f64, v: V) -> V {
    let (s, c) = ang.sin_cos();
    let kxv = cross(axis, v);
    let kdv = dot(axis, v);
    [
        v[0] * c + kxv[0] * s + axis[0] * kdv * (1.0 - c),
        v[1] * c + kxv[1] * s + axis[1] * kdv * (1.0 - c),
        v[2] * c + kxv[2] * s + axis[2] * kdv * (1.0 - c),
    ]
}
fn close(a: V, b: V) -> bool {
    let tol = 2e-4 * (1.0 + norm(a).max(norm(b)));
    norm(sub(a, b)) <= tol
}

/// A fixed generic frame (half-space SDF).
struct Frame {
    n_l: V,
    h: f64,
    c: V,
    axis: V,
    ang: f64,
    s: f64,
}

impl Frame {
    fn new() -> Self {
        Self {
            n_l: unit([0.6, 0.0, 0.8]),
            h: 0.25,
            c: [1.0, -2.0, 0.5],
            axis: unit([1.0, 2.0, 2.0]),
            ang: 0.7,
            s: 1.5,
        }
    }
    fn n_w(&self) -> V {
        rot(self.axis, self.ang, self.n_l)
    }
    fn dist(&self, p: V) -> f64 {
        dot(self.n_w(), sub(p, self.c)) - self.s * self.h
    }
    fn collider(&self) -> SdfCollider {
        let n = self.n_l;
        let h = self.h as f32;
        let nf = (n[0] as f32, n[1] as f32, n[2] as f32);
        let field = ClosureSdf::new(
            move |x, y, z| nf.0 * x + nf.1 * y + nf.2 * z - h,
            move |_, _, _| nf,
        );
        let q = QuatFix::from_axis_angle(v3(self.axis), fx(self.ang));
        SdfCollider::new_static(Box::new(field), v3(self.c), q).with_scale(fx(self.s))
    }
    /// A point at signed world distance `d` along the normal, offset sideways.
    fn point_at(&self, d: f64, side: f64) -> V {
        let n = self.n_w();
        let t = unit(cross(n, [0.0, 0.0, 1.0]));
        // point on the plane: c + n * s h  (n.(p-c) = s h) then move d along n
        let on = [
            self.c[0] + n[0] * self.s * self.h,
            self.c[1] + n[1] * self.s * self.h,
            self.c[2] + n[2] * self.s * self.h,
        ];
        [
            on[0] + n[0] * d + t[0] * side,
            on[1] + n[1] * d + t[1] * side,
            on[2] + n[2] * d + t[2] * side,
        ]
    }
}

fn force(f: &Frame, p: V, vel: V, t: &SdfForceType) -> V {
    let mut b = RigidBody::new(v3(p), Fix128::ONE);
    b.velocity = v3(vel);
    arr(compute_sdf_force(&b, &f.collider(), t))
}

#[test]
fn frame_distance_is_what_the_oracle_assumes() {
    // Sanity of the reference itself (independent of the code under test):
    let f = Frame::new();
    for d in [-3.0, -0.5, 0.0, 0.75, 4.0] {
        assert!((f.dist(f.point_at(d, 1.3)) - d).abs() < 1e-9);
    }
}

#[test]
fn attract_matches_closed_form_in_a_generic_frame() {
    let f = Frame::new();
    let n = f.n_w();
    for (d, k, fmax) in [
        (2.0, 3.0, 100.0),
        (-10.0, 3.0, 100.0),
        (6.0, 3.0, 12.0),
        (-0.25, 8.0, 50.0),
        (0.0, 3.0, 9.0),
    ] {
        let p = f.point_at(d, 0.8);
        let dd = f.dist(p);
        let mag = (k * dd.abs()).min(fmax);
        let want = if dd > 0.0 {
            scale(n, -mag)
        } else {
            scale(n, mag)
        };
        let got = force(
            &f,
            p,
            [0.0; 3],
            &SdfForceType::Attract {
                strength: fx(k),
                max_force: fx(fmax),
            },
        );
        assert!(close(got, want), "d={d}: got {got:?} want {want:?}");
    }
}

#[test]
fn repel_matches_closed_form_in_a_generic_frame() {
    let f = Frame::new();
    let n = f.n_w();
    let (k, range) = (5.0, 4.0);
    for d in [0.5, 1.0, 2.0, 3.5, -0.5, -1.0, -2.0, -3.5] {
        let p = f.point_at(d, -0.4);
        let dd = f.dist(p);
        let fac = 1.0 - dd.abs() / range;
        let mag = k * fac * fac;
        let want = if dd > 0.0 {
            scale(n, mag)
        } else {
            scale(n, -mag)
        };
        let got = force(
            &f,
            p,
            [0.0; 3],
            &SdfForceType::Repel {
                strength: fx(k),
                range: fx(range),
            },
        );
        assert!(close(got, want), "d={d}: got {got:?} want {want:?}");
    }
    // Beyond range: exactly zero.
    let far = f.point_at(4.5, 0.0);
    assert_eq!(
        force(
            &f,
            far,
            [0.0; 3],
            &SdfForceType::Repel {
                strength: fx(k),
                range: fx(range)
            }
        ),
        [0.0; 3]
    );
}

/// Mirror symmetry: F(+d) = -F(-d) for Repel (equal |d| on either side).
#[test]
fn repel_is_odd_under_reflection_through_the_surface() {
    let f = Frame::new();
    let t = SdfForceType::Repel {
        strength: fx(5.0),
        range: fx(4.0),
    };
    for d in [0.5, 1.5, 3.0] {
        let a = force(&f, f.point_at(d, 0.3), [0.0; 3], &t);
        let b = force(&f, f.point_at(-d, 0.3), [0.0; 3], &t);
        assert!(close(a, scale(b, -1.0)), "d = {d}");
    }
}

#[test]
fn contain_matches_closed_form_with_damping() {
    let f = Frame::new();
    let n = f.n_w();
    let (k, c) = (7.0, 0.5);
    let vel = [1.0, -2.0, 0.5];
    for d in [-2.0, -0.25, 0.25, 1.5, 3.0] {
        let p = f.point_at(d, 0.6);
        let dd = f.dist(p);
        let want = if dd <= 0.0 {
            scale(vel, -c)
        } else {
            let push = scale(n, -k * dd);
            let damp = scale(vel, -c);
            [push[0] + damp[0], push[1] + damp[1], push[2] + damp[2]]
        };
        let got = force(
            &f,
            p,
            vel,
            &SdfForceType::Contain {
                strength: fx(k),
                damping: fx(c),
            },
        );
        assert!(close(got, want), "d={d}: got {got:?} want {want:?}");
    }
}

/// AUD-A-S4W1-009 (doc defect): the `Contain` variant is documented as
/// "zero force inside, strong push inward when outside", but inside the
/// volume the force is the damping term `-c v` (non-zero for a moving body).
/// The module's own test names this "inside: damping only". The doc and the
/// code disagree.
#[test]
#[ignore = "known defect: AUD-A-S4W1-009: SdfForceType::Contain doc says zero force inside; code returns -damping*velocity (-0.5 for v=1)"]
fn contain_is_force_free_inside_as_documented() {
    let f = Frame::new();
    let got = force(
        &f,
        f.point_at(-2.0, 0.0),
        [1.0, 0.0, 0.0],
        &SdfForceType::Contain {
            strength: fx(7.0),
            damping: fx(0.5),
        },
    );
    assert_eq!(got, [0.0; 3], "inside force = {got:?}");
}

#[test]
fn surface_flow_matches_closed_form() {
    let f = Frame::new();
    let n = f.n_w();
    let flow = [0.3, -1.0, 2.0];
    let (k, infl) = (6.0, 3.0);
    for d in [-2.0, -0.5, 0.0, 0.75, 2.5] {
        let p = f.point_at(d, 0.2);
        let dd = f.dist(p);
        let t = sub(flow, scale(n, dot(flow, n)));
        let want = scale(unit(t), k * (1.0 - dd.abs() / infl));
        let got = force(
            &f,
            p,
            [0.0; 3],
            &SdfForceType::SurfaceFlow {
                flow_direction: v3(flow),
                strength: fx(k),
                influence_distance: fx(infl),
            },
        );
        assert!(close(got, want), "d={d}: got {got:?} want {want:?}");
        assert!(
            dot(got, n).abs() < 1e-3 * k,
            "flow force must be tangent to the surface"
        );
    }
    // beyond the influence distance: zero
    let far = f.point_at(3.5, 0.0);
    assert_eq!(
        force(
            &f,
            far,
            [0.0; 3],
            &SdfForceType::SurfaceFlow {
                flow_direction: v3(flow),
                strength: fx(k),
                influence_distance: fx(infl)
            }
        ),
        [0.0; 3]
    );
}

#[test]
fn vortex_matches_closed_form_and_is_axis_scale_invariant() {
    let f = Frame::new();
    let n = f.n_w();
    let axis = [0.0, 1.0, 0.0];
    let (k, infl) = (4.0, 5.0);
    for d in [-1.0, 0.0, 1.5, 4.0] {
        let p = f.point_at(d, 0.5);
        let dd = f.dist(p);
        let want = scale(unit(cross(axis, n)), k * (1.0 - dd.abs() / infl));
        let t1 = SdfForceType::SdfVortex {
            axis: v3(axis),
            strength: fx(k),
            influence_distance: fx(infl),
        };
        let t2 = SdfForceType::SdfVortex {
            axis: v3(scale(axis, 4.0)),
            strength: fx(k),
            influence_distance: fx(infl),
        };
        let g1 = force(&f, p, [0.0; 3], &t1);
        let g2 = force(&f, p, [0.0; 3], &t2);
        assert!(close(g1, want), "d={d}: got {g1:?} want {want:?}");
        assert!(close(g1, g2), "axis magnitude must not matter");
        assert!(dot(g1, n).abs() < 1e-3 * k && dot(g1, axis).abs() < 1e-3 * k);
    }
}

/// AUD-A-S4W1-010 (fixed; the text below describes the defect): when the swirl axis is parallel to the
/// surface normal the swirl direction `axis x n` is undefined and its length
/// is 0 in exact arithmetic (|axis x n| = sin(theta)). The code normalizes
/// the cross product, and `normalize` only returns zero for an exactly zero
/// vector; the f32-derived normal is off by ~1e-8, so the cross product is
/// tiny but non-zero and the result is a FULL-strength force (strength x
/// falloff = 3.2 here) in a rounding-noise direction. The swirl magnitude
/// does not go to 0 as the axis approaches the normal.
#[test]
fn vortex_vanishes_when_axis_is_parallel_to_normal() {
    let f = Frame::new();
    let n = f.n_w();
    let (k, infl) = (4.0, 5.0);
    let t = SdfForceType::SdfVortex {
        axis: v3(n),
        strength: fx(k),
        influence_distance: fx(infl),
    };
    let g = force(&f, f.point_at(1.0, 0.0), [0.0; 3], &t);
    assert!(norm(g) < 1e-2, "{g:?}");
}

/// AUD-A-S4W1-011 (fixed, same mechanism as AUD-A-S4W1-010): a flow
/// direction parallel to the surface normal has no tangential component, so
/// the force must vanish. `tangent.length().is_zero()` only catches an exactly
/// zero projection; with the f32 normal the projection is ~1e-8 and is then
/// normalized to a unit vector, giving |F| = strength x falloff in a noise
/// direction.
#[test]
fn surface_flow_vanishes_when_flow_is_parallel_to_normal() {
    let f = Frame::new();
    let n = f.n_w();
    let t = SdfForceType::SurfaceFlow {
        flow_direction: v3(n),
        strength: fx(6.0),
        influence_distance: fx(3.0),
    };
    let g = force(&f, f.point_at(0.5, 0.0), [0.0; 3], &t);
    assert!(norm(g) < 1e-2, "{g:?}");
}

/// Attract is monotone in the distance and saturates exactly at `max_force`.
#[test]
fn attract_is_monotone_and_saturates() {
    let f = Frame::new();
    let t = SdfForceType::Attract {
        strength: fx(2.0),
        max_force: fx(9.0),
    };
    let mut prev = 0.0;
    for i in 0..=24 {
        let d = 0.25 * i as f64;
        let m = norm(force(&f, f.point_at(d, 0.0), [0.0; 3], &t));
        assert!(m + 1e-3 >= prev, "not monotone at d={d}");
        assert!(m <= 9.0 + 1e-3, "exceeds max_force at d={d}: {m}");
        prev = m;
    }
    assert!(close([prev, 0.0, 0.0], [9.0, 0.0, 0.0]));
}

/// Convenience constructors: the constants they bake in (undocumented).
#[test]
fn constructor_constants_are_pinned() {
    let a = SdfForceField::attract(2, fx(3.0));
    assert_eq!(a.sdf_index, 2);
    assert!(a.enabled && a.affected_bodies.is_none());
    match a.force_type {
        SdfForceType::Attract {
            strength,
            max_force,
        } => {
            assert_eq!(strength, fx(3.0));
            assert_eq!(max_force, fx(30.0), "max_force = 10 x strength");
        }
        other => panic!("{other:?}"),
    }
    match SdfForceField::contain(0, fx(7.0)).force_type {
        SdfForceType::Contain { strength, damping } => {
            assert_eq!(strength, fx(7.0));
            assert_eq!(damping, Fix128::from_ratio(1, 10), "damping = 1/10");
        }
        other => panic!("{other:?}"),
    }
    match SdfForceField::surface_flow(0, Vec3Fix::UNIT_Z, fx(5.0)).force_type {
        SdfForceType::SurfaceFlow {
            flow_direction,
            strength,
            influence_distance,
        } => {
            assert_eq!(flow_direction, Vec3Fix::UNIT_Z);
            assert_eq!(strength, fx(5.0));
            assert_eq!(influence_distance, fx(2.0), "influence = 2");
        }
        other => panic!("{other:?}"),
    }
    match SdfForceField::repel(1, fx(8.0), fx(4.0)).force_type {
        SdfForceType::Repel { strength, range } => {
            assert_eq!((strength, range), (fx(8.0), fx(4.0)))
        }
        other => panic!("{other:?}"),
    }
}

fn plane_up() -> SdfCollider {
    // world plane y = 0, normal +Y
    SdfCollider::new_static(
        Box::new(ClosureSdf::new(|_, y, _| y, |_, _, _| (0.0, 1.0, 0.0))),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    )
}

/// apply: dv = F * inv_mass * dt exactly (dyadic numbers), sampled per body at
/// that body's position; static and disabled / out-of-filter bodies unchanged.
#[test]
fn apply_integrates_force_times_inverse_mass_times_dt() {
    let cols = [plane_up()];
    let field = SdfForceField::attract(0, fx(3.0)); // F = -3 d (toward the plane)
    let mut near = RigidBody::new_dynamic(Vec3Fix::new(fx(7.0), fx(2.0), fx(-1.0)), fx(4.0));
    near.velocity = v3([1.0, 0.0, 0.0]);
    let far = RigidBody::new_dynamic(Vec3Fix::new(fx(-3.0), fx(4.0), fx(0.0)), fx(2.0));
    let statik = RigidBody::new_static(Vec3Fix::new(fx(0.0), fx(2.0), fx(0.0)));
    let mut bodies = vec![near, far, statik];
    apply_sdf_force_fields(&[field], &cols, &mut bodies, fx(0.5));
    // near: d = 2 -> F_y = -6, invm = 1/4, dt = 1/2 -> dv_y = -0.75
    assert_eq!(arr(bodies[0].velocity), [1.0, -0.75, 0.0]);
    // far: d = 4 -> F_y = -12, invm = 1/2 -> dv_y = -3
    assert_eq!(arr(bodies[1].velocity), [0.0, -3.0, 0.0]);
    assert_eq!(arr(bodies[2].velocity), [0.0; 3]);
    // Positions are untouched by a force pass.
    assert_eq!(arr(bodies[0].position), [7.0, 2.0, -1.0]);
}

/// Disabled field, a body outside `affected_bodies`, and a dangling collider
/// index add nothing; two fields add up.
#[test]
fn apply_respects_enabled_filter_and_sums_fields() {
    let cols = [plane_up()];
    let mk = |y: f64| RigidBody::new_dynamic(Vec3Fix::new(fx(0.0), fx(y), fx(0.0)), fx(1.0));
    let f1 = SdfForceField::attract(0, fx(3.0));
    let mut f2 = SdfForceField::attract(0, fx(1.0));
    f2.enabled = false;
    let f3 = SdfForceField::attract(0, fx(2.0)).with_affected_bodies(vec![1]);
    let dangling = SdfForceField::attract(5, fx(100.0));
    let mut bodies = vec![mk(2.0), mk(2.0)];
    apply_sdf_force_fields(&[f1, f2, f3, dangling], &cols, &mut bodies, fx(1.0));
    // body 0: only f1: -3*2 = -6 ; body 1: f1 + f3 = -6 - 4 = -10
    assert_eq!(arr(bodies[0].velocity)[1], -6.0);
    assert_eq!(arr(bodies[1].velocity)[1], -10.0);
}

/// The collider's transform is honoured: translating the collider by `c` shifts
/// the field. A body at p with collider at c sees the same force as a body at
/// p - c with the collider at the origin.
#[test]
fn force_is_translation_covariant() {
    let t = SdfForceType::Repel {
        strength: fx(5.0),
        range: fx(4.0),
    };
    let a = {
        let b = RigidBody::new(Vec3Fix::new(fx(3.0), fx(2.0), fx(1.0)), Fix128::ONE);
        arr(compute_sdf_force(&b, &plane_up(), &t))
    };
    let shifted = SdfCollider::new_static(
        Box::new(ClosureSdf::new(|_, y, _| y, |_, _, _| (0.0, 1.0, 0.0))),
        Vec3Fix::new(fx(10.0), fx(-5.0), fx(2.0)),
        QuatFix::IDENTITY,
    );
    let b = RigidBody::new(Vec3Fix::new(fx(13.0), fx(-3.0), fx(3.0)), Fix128::ONE);
    let got = arr(compute_sdf_force(&b, &shifted, &t));
    assert!(close(a, got), "{a:?} vs {got:?}");
}

/// Beyond the influence distance the vortex is exactly zero; a collider index
/// equal to the collider count is dangling, not an out-of-bounds read.
#[test]
fn vortex_zero_beyond_influence_and_index_equal_to_len_is_skipped() {
    let f = Frame::new();
    let t = SdfForceType::SdfVortex {
        axis: v3([0.0, 1.0, 0.0]),
        strength: fx(4.0),
        influence_distance: fx(5.0),
    };
    assert_eq!(force(&f, f.point_at(6.0, 0.5), [0.0; 3], &t), [0.0; 3]);
    let cols = [plane_up()];
    let mut bodies = vec![RigidBody::new_dynamic(v3([0.0, 2.0, 0.0]), fx(1.0))];
    apply_sdf_force_fields(
        &[SdfForceField::attract(1, fx(3.0))],
        &cols,
        &mut bodies,
        fx(1.0),
    );
    assert_eq!(arr(bodies[0].velocity), [0.0; 3]);
}
