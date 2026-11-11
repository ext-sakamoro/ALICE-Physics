//! Oracle: a ray query finds a body whose collision box leaves the `Fix128` range.
//!
//! # What is measured
//!
//! `PhysicsWorld::raycast` and `PhysicsWorld::cast_ray` build a BVH of the bodies'
//! boxes for the query. A box built with `AABB::from_center_half` from a position
//! and a radius whose sum leaves the range wraps to the other side and comes out
//! inverted, so the BVH never offers that body as a candidate and the ray misses
//! it although the sphere's near surface is well inside the range. The broad-phase
//! of `step` builds its boxes with the saturating form since 001001f1; the query
//! paths are expected to see the same body.
//!
//! The closed form: a body at `x = 2^62 + 2^40` with radius `2^62` has its near
//! surface at `x = 2^40`; a ray from the origin along +x with reach `2^41` hits it
//! at `t = 2^40`. The mirrored body (`x = -(2^62 + 2^40)`, ray along -x) is the
//! same closed form with the box's minimum wrapping instead of its maximum. An
//! in-range twin (`x = 3`, radius `1`, hit at `t = 2`) keeps the same query honest.
//!
//! Author: Moroya Sakamoto

use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::shape_raycast::RayFilter;
use alice_physics_core::solver::{PhysicsWorld, RigidBody, SolverConfig};

fn pow2(n: u32) -> Fix128 {
    Fix128::from_raw(1i64 << n, 0)
}

fn x_at(x: Fix128) -> Vec3Fix {
    Vec3Fix::new(x, Fix128::ZERO, Fix128::ZERO)
}

fn world_with_sphere_at(center: Vec3Fix, radius: Fix128) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(SolverConfig {
        gravity: Vec3Fix::ZERO,
        ..SolverConfig::default()
    });
    w.add_body_with_radius(RigidBody::new(center, Fix128::ONE), radius);
    w
}

fn world_with_sphere(x: Fix128, radius: Fix128) -> PhysicsWorld {
    world_with_sphere_at(x_at(x), radius)
}

/// (body x, ray direction sign, expected t) for the two wrapping sides and the twin.
fn cases() -> Vec<(&'static str, Fix128, Fix128, Fix128, Fix128)> {
    let far = pow2(62) + pow2(40);
    vec![
        (
            "in range",
            Fix128::from_int(3),
            Fix128::ONE,
            Fix128::ONE,
            Fix128::from_int(2),
        ),
        ("box max wraps", far, pow2(62), Fix128::ONE, pow2(40)),
        ("box min wraps", -far, pow2(62), -Fix128::ONE, pow2(40)),
    ]
}

#[test]
fn raycast_hits_a_body_whose_box_leaves_the_range() {
    for (name, x, r, sign, want_t) in cases() {
        let w = world_with_sphere(x, r);
        let reach = if want_t > pow2(41) {
            want_t * Fix128::from_int(2)
        } else {
            pow2(41)
        };
        let hit = w.raycast(Vec3Fix::ZERO, x_at(sign), reach);
        assert_eq!(hit, Some((0, want_t)), "{name}: raycast");
    }
}

#[test]
fn cast_ray_hits_a_body_whose_box_leaves_the_range() {
    for (name, x, r, sign, want_t) in cases() {
        let w = world_with_sphere(x, r);
        let hit = w.cast_ray(Vec3Fix::ZERO, x_at(sign), pow2(41), &RayFilter::default());
        let hit = hit.unwrap_or_else(|| panic!("{name}: cast_ray missed"));
        assert_eq!((hit.body, hit.t), (Some(0), want_t), "{name}: cast_ray");
    }
}

/// `k` of the scaled path: the bits of the largest integer magnitude beyond 29
/// (a negative value by its one's complement, as the implementation counts it);
/// the implementation takes it over `2 · radius + 2` after advancing the ray.
fn scale_bits_of(values: &[i64]) -> u32 {
    let top = values
        .iter()
        .map(|&v| 64 - (if v < 0 { !v } else { v }).leading_zeros())
        .max()
        .unwrap_or(0);
    top.saturating_sub(29)
}

/// FNV-1a over the raw bits of every distance, for the cross-platform check.
fn fold(digest: &mut u64, t: Fix128) {
    for b in t.hi.to_le_bytes().iter().chain(t.lo.to_le_bytes().iter()) {
        *digest ^= u64::from(*b);
        *digest = digest.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

/// Recorded on the first run (ubuntu / macOS / windows / wasm must agree).
const FAR_DIGEST: u64 = 0xe0016b5b9b5bff9d;

/// Exact closed forms beyond the direct limit: the centre is `L` along an axis and
/// `h·m` beside it, the radius `r·m` with `r² = h² + s²` (a Pythagorean triple),
/// so the ray from the origin along the axis enters the sphere at exactly
/// `t = L − s·m`. Every input is an integer, so the reference needs no rounding.
/// The scaled path must find every one of them within `2^(k−60)` and give the
/// same bits on every platform.
#[test]
fn raycast_beyond_the_direct_limit_is_within_the_stated_bound_of_exact_closed_forms() {
    let triples = [
        (3i64, 4i64, 5i64),
        (5, 12, 13),
        (8, 15, 17),
        (7, 24, 25),
        (20, 21, 29),
        (12, 35, 37),
    ];
    let mut digest = 0xcbf2_9ce4_8422_2325u64;
    let mut checked = 0;
    let mut worst = Fix128::ZERO;
    for &(h, s, r) in &triples {
        for p in [40u32, 48, 52, 56, 57] {
            let m = 1i64 << p;
            for l in [1i64 << 61, 1i64 << 62, 3i64 << 60] {
                if s * m >= l {
                    // the origin would be inside the sphere (its far side beyond `l`)
                    continue;
                }
                for axis in 0..3 {
                    for sign in [1i64, -1] {
                        let mut c = [0i64; 3];
                        c[axis] = sign * l;
                        c[(axis + 1) % 3] = h * m;
                        let center = Vec3Fix::from_int(c[0], c[1], c[2]);
                        let mut d = [Fix128::ZERO; 3];
                        d[axis] = Fix128::from_int(sign);
                        let w = world_with_sphere_at(center, Fix128::from_int(r * m));
                        let want = Fix128::from_int(l - s * m);
                        // the scale is chosen from the body's own extent (the ray is advanced first)
                        let k = scale_bits_of(&[2 * r * m + 2]);
                        assert!(
                            k > 0,
                            "the sphere must be huge, beyond the direct limit by itself"
                        );
                        let bound = Fix128::from_raw(0, 1u64 << (k + 4)); // 2^(k−60)
                        let hit = w.raycast(
                            Vec3Fix::ZERO,
                            Vec3Fix::new(d[0], d[1], d[2]),
                            Fix128::from_int(l),
                        );
                        let (body, got) = hit.unwrap_or_else(|| panic!("silent None: triple {h},{s},{r} p={p} l={l} axis={axis} sign={sign}"));
                        assert_eq!(body, 0);
                        let err = (got - want).abs();
                        assert!(err <= bound, "triple {h},{s},{r} p={p} l={l} axis={axis} sign={sign}: |t − exact| = {err:?} > 2^({k}−60)");
                        if err > worst {
                            worst = err;
                        }
                        fold(&mut digest, got);
                        checked += 1;
                    }
                }
            }
        }
    }
    assert!(checked >= 400, "compared {checked}");
    assert_eq!(
        digest, FAR_DIGEST,
        "far-query bits moved (checked {checked}, worst error {worst:?})"
    );
}

/// Recorded on the first run (every platform must agree).
const FAR_GEOMETRY_DIGEST: u64 = 0x7c654fa22812aa85;

/// `cast_ray` beyond the direct limit, against the real geometry: each body sits
/// `l` along an axis (identity rotation), and the ray from the origin along that
/// axis enters it at an exact integer distance: a box or a wedge-free solid at
/// `l − extent`, a sphere from a Pythagorean triple as above, a compound's sphere
/// child at `l − offset − radius`. The analytic shapes must be within `2^(k−60)`;
/// the torus, which sphere-traces to `2⁻³²` at the scale, within `2^(k−32)`.
#[test]
fn cast_ray_beyond_the_direct_limit_is_within_the_stated_bound_of_exact_closed_forms() {
    use alice_physics_core::collider::Sphere;
    use alice_physics_core::compound::CompoundShape;
    use alice_physics_core::math::QuatFix;
    use alice_physics_core::shape::Shape;

    let filter = RayFilter::default();
    let mut digest = 0xcbf2_9ce4_8422_2325u64;
    let mut checked = 0;
    let mut worst = Fix128::ZERO;
    let mut check =
        |w: &PhysicsWorld, dir: Vec3Fix, l: i64, want: i64, k: u32, slack: u32, what: &str| {
            let hit = w
                .cast_ray(Vec3Fix::ZERO, dir, Fix128::from_int(l), &filter)
                .unwrap_or_else(|| panic!("silent None: {what}"));
            assert_eq!(hit.body, Some(0), "{what}");
            let err = (hit.t - Fix128::from_int(want)).abs();
            // 2^(k + slack − 64)
            let bound = if k + slack < 64 {
                Fix128::from_raw(0, 1u64 << (k + slack))
            } else {
                Fix128::from_raw(1i64 << (k + slack - 64), 0)
            };
            assert!(
                err <= bound,
                "{what}: |t − exact| = {err:?} > 2^({k}−{})",
                64 - slack
            );
            if err > worst {
                worst = err;
            }
            fold(&mut digest, hit.t);
            checked += 1;
        };
    for p in [40u32, 50, 56] {
        let m = 1i64 << p;
        for l in [1i64 << 61, 1i64 << 62, 3i64 << 60] {
            for axis in 0..3 {
                for sign in [1i64, -1] {
                    let mut c = [0i64; 3];
                    c[axis] = sign * l;
                    let center = Vec3Fix::from_int(c[0], c[1], c[2]);
                    let mut d = [Fix128::ZERO; 3];
                    d[axis] = Fix128::from_int(sign);
                    let dir = Vec3Fix::new(d[0], d[1], d[2]);
                    // the scale is chosen from the geometry's reach (sum of its lengths), and
                    // for an ellipsoid from its largest radius brought below 2
                    let k_box = scale_bits_of(&[2 * 10 * m + 2]);
                    let k_ellipsoid = k_box.max(64 - (4 * m).leading_zeros() - 1);
                    let k_torus = scale_bits_of(&[2 * 5 * m + 2]);
                    let k_sphere = scale_bits_of(&[2 * 5 * m + 2]);
                    let tag = format!("p={p} l={l} axis={axis} sign={sign}");
                    // a box 3m × 2m × 5m (half extents), entered on the face facing the origin
                    let mut he = [2 * m, 3 * m, 5 * m];
                    he.rotate_right(axis);
                    let mut w = world_with_sphere_at(center, Fix128::from_int(8 * m));
                    assert!(w.set_body_shape(
                        0,
                        &Shape::Box {
                            half_extents: Vec3Fix::from_int(he[0], he[1], he[2])
                        }
                    ));
                    check(&w, dir, l, l - 2 * m, k_box, 4, &format!("box {tag}"));
                    // an ellipsoid with radii (4m, 2m, 3m) along the axis
                    let mut ra = [4 * m, 2 * m, 3 * m];
                    ra.rotate_right(axis);
                    let mut w = world_with_sphere_at(center, Fix128::from_int(8 * m));
                    assert!(w.set_body_shape(
                        0,
                        &Shape::Ellipsoid {
                            radii: Vec3Fix::from_int(ra[0], ra[1], ra[2])
                        }
                    ));
                    check(
                        &w,
                        dir,
                        l,
                        l - 4 * m,
                        k_ellipsoid,
                        4,
                        &format!("ellipsoid {tag}"),
                    );
                    // a torus (its ring in the xz plane): along x or z the outer rim is at
                    // major + minor = 5m; along y the ray passes through the hole
                    if axis != 1 {
                        let mut w = world_with_sphere_at(center, Fix128::from_int(8 * m));
                        assert!(w.set_body_shape(
                            0,
                            &Shape::Torus {
                                major_radius: Fix128::from_int(4 * m),
                                minor_radius: Fix128::from_int(m)
                            }
                        ));
                        check(&w, dir, l, l - 5 * m, k_torus, 32, &format!("torus {tag}"));
                    }
                    // a compound of ordinary size far away: one sphere child of radius 1
                    // (a huge child has no finite mass and is refused); the body sits at
                    // the compound's centre of mass, so the child is centred on the body
                    let mut off = [0i64; 3];
                    off[axis] = -sign * 2;
                    let mut compound = CompoundShape::new();
                    compound.add_sphere(
                        Sphere {
                            center: Vec3Fix::from_int(off[0], off[1], off[2]),
                            radius: Fix128::ONE,
                        },
                        Vec3Fix::ZERO,
                        QuatFix::IDENTITY,
                    );
                    let mut w = PhysicsWorld::new(SolverConfig {
                        gravity: Vec3Fix::ZERO,
                        ..SolverConfig::default()
                    });
                    let i = w
                        .add_compound_body(&compound, Fix128::ONE, center)
                        .expect("valid compound");
                    assert_eq!(i, 0);
                    check(&w, dir, l, l - 1, 0, 4, &format!("compound {tag}"));
                    // the bounding sphere alone, from the triple (3, 4, 5) beside the axis
                    let mut c2 = c;
                    c2[(axis + 1) % 3] = 3 * m;
                    let w = world_with_sphere_at(
                        Vec3Fix::from_int(c2[0], c2[1], c2[2]),
                        Fix128::from_int(5 * m),
                    );
                    check(&w, dir, l, l - 4 * m, k_sphere, 4, &format!("sphere {tag}"));
                }
            }
        }
    }
    assert!(checked >= 200, "compared {checked}");
    assert_eq!(
        digest, FAR_GEOMETRY_DIGEST,
        "far geometry bits moved (checked {checked}, worst error {worst:?})"
    );
}

/// Recorded on the first run (every platform must agree).
const FAR_STATIC_DIGEST: u64 = 0x643f87519685a6e5;

/// Static colliders beyond the direct limit: a plane whose offset is far, a plane
/// hit from a far origin, and a flat height field and a triangle mesh of ordinary
/// size placed far along an axis, hit from straight above. Every distance has an
/// exact integer closed form; a plane must be within `2^(k−60)` (unit incidence
/// here), a field or mesh of ordinary size within `2⁻⁶⁰` whatever its distance.
#[test]
fn cast_ray_beyond_the_direct_limit_meets_static_colliders_within_the_stated_bound() {
    use alice_physics_core::heightfield::HeightField;
    use alice_physics_core::plane_collider::PlaneCollider;
    use alice_physics_core::static_collider::StaticCollider;
    use alice_physics_core::trimesh::TriMesh;

    let filter = RayFilter::default();
    let mut digest = 0xcbf2_9ce4_8422_2325u64;
    let mut checked = 0;
    let mut check = |w: &PhysicsWorld,
                     origin: Vec3Fix,
                     dir: Vec3Fix,
                     reach: i64,
                     want: i64,
                     k: u32,
                     what: &str| {
        let hit = w
            .cast_ray(origin, dir, Fix128::from_int(reach), &filter)
            .unwrap_or_else(|| panic!("silent None: {what}"));
        assert_eq!(hit.body, None, "{what}");
        let err = (hit.t - Fix128::from_int(want)).abs();
        let bound = Fix128::from_raw(0, 1u64 << (k + 4)); // 2^(k−60)
        assert!(err <= bound, "{what}: |t − exact| = {err:?} > 2^({k}−60)");
        fold(&mut digest, hit.t);
        checked += 1;
    };
    let new_world = || {
        PhysicsWorld::new(SolverConfig {
            gravity: Vec3Fix::ZERO,
            ..SolverConfig::default()
        })
    };
    for l in [1i64 << 40, 1i64 << 61, 1i64 << 62, 3i64 << 60] {
        for axis in 0..3 {
            for sign in [1i64, -1] {
                let mut n = [0i64; 3];
                n[axis] = sign;
                let dir = Vec3Fix::from_int(n[0], n[1], n[2]);
                let tag = format!("l={l} axis={axis} sign={sign}");
                // a plane far along the axis, facing the origin: t = l exactly
                let mut w = new_world();
                w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
                    dir,
                    Fix128::from_int(l),
                )));
                let k = scale_bits_of(&[l, l + 1]);
                check(
                    &w,
                    Vec3Fix::ZERO,
                    dir,
                    l + 1,
                    l,
                    k,
                    &format!("far plane {tag}"),
                );
                // the ground plane hit from a far origin 2^20 above it
                let mut p = [0i64; 3];
                p[axis] = sign * l;
                p[(axis + 1) % 3] = 1 << 20;
                let mut down = [0i64; 3];
                down[(axis + 1) % 3] = -1;
                let (origin, down) = (
                    Vec3Fix::from_int(p[0], p[1], p[2]),
                    Vec3Fix::from_int(down[0], down[1], down[2]),
                );
                let mut nn = [0i64; 3];
                nn[(axis + 1) % 3] = 1;
                let mut w = new_world();
                w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
                    Vec3Fix::from_int(nn[0], nn[1], nn[2]),
                    Fix128::ZERO,
                )));
                let k = scale_bits_of(&[l, 1 << 21]);
                check(
                    &w,
                    origin,
                    down,
                    1 << 21,
                    1 << 20,
                    k,
                    &format!("ground plane from far {tag}"),
                );
                // a flat height field (y = 0) of 5 × 5 points, spacing 1, around the far
                // point in the xz plane, hit from 2^20 above: only when "above" is y
                if (axis + 1) % 3 == 1 {
                    let mut fo = [0i64; 3];
                    fo[axis] = sign * l - 2;
                    fo[2] = -2;
                    let field = HeightField::new(
                        vec![Fix128::ZERO; 25],
                        5,
                        5,
                        Fix128::ONE,
                        Vec3Fix::from_int(fo[0], fo[1], fo[2]),
                    );
                    let mut w = new_world();
                    w.add_static_collider(StaticCollider::HeightField(field));
                    check(
                        &w,
                        origin,
                        down,
                        1 << 21,
                        1 << 20,
                        0,
                        &format!("far field {tag}"),
                    );
                    // two triangles making the square [l−2, l+2] × [−2, 2] at y = 0
                    let v = |x: i64, z: i64| Vec3Fix::from_int(x, 0, z);
                    let (x0, x1) = (sign * l - 2, sign * l + 2);
                    let mesh = TriMesh::from_indexed(
                        &[v(x0, -2), v(x1, -2), v(x1, 2), v(x0, 2)],
                        &[0, 1, 2, 0, 2, 3],
                    );
                    let mut w = new_world();
                    w.add_static_collider(StaticCollider::TriMesh(mesh));
                    check(
                        &w,
                        origin,
                        down,
                        1 << 21,
                        1 << 20,
                        0,
                        &format!("far mesh {tag}"),
                    );
                }
            }
        }
    }
    // huge statics, scaled by a copy per query: a flat field of 3 × 3 points at
    // spacing 2^40 and a mesh square of side 2^41, both at y = 0 around a far x,
    // hit from 2^20 above (t = 2^20); their scale k comes from their reach
    for l in [1i64 << 40, 1i64 << 61] {
        let origin = Vec3Fix::from_int(l, 1 << 20, 0);
        let down = Vec3Fix::from_int(0, -1, 0);
        let m = 1i64 << 40;
        // the box half reaches m along x and m along z: 2 · (m + m) + 2
        let k = scale_bits_of(&[2 * (m + m) + 2]);
        let field = HeightField::new(
            vec![Fix128::ZERO; 9],
            3,
            3,
            Fix128::from_int(m),
            Vec3Fix::from_int(l - m, 0, -m),
        );
        let mut w = new_world();
        w.add_static_collider(StaticCollider::HeightField(field));
        check(
            &w,
            origin,
            down,
            1 << 21,
            1 << 20,
            k,
            &format!("huge field l={l}"),
        );
        let v = |x: i64, z: i64| Vec3Fix::from_int(x, 0, z);
        let mesh = TriMesh::from_indexed(
            &[v(l - m, -m), v(l + m, -m), v(l + m, m), v(l - m, m)],
            &[0, 1, 2, 0, 2, 3],
        );
        let mut w = new_world();
        w.add_static_collider(StaticCollider::TriMesh(mesh));
        check(
            &w,
            origin,
            down,
            1 << 21,
            1 << 20,
            k,
            &format!("huge mesh l={l}"),
        );
    }
    assert!(checked >= 40, "compared {checked}");
    assert_eq!(
        digest, FAR_STATIC_DIGEST,
        "far static bits moved (checked {checked})"
    );
}

/// A unit-sphere SDF far along +x, hit from the origin: the march works in the
/// field's frame (`origin − position` in `Fix128`, then `f32`), so the far
/// surface is found to the march tolerance (`1e-3`) plus the `f32` resolution
/// of the local coordinates near the surface.
#[cfg(feature = "std")]
#[test]
fn cast_ray_beyond_the_direct_limit_meets_a_far_sdf_collider() {
    use alice_physics_core::math::QuatFix;
    use alice_physics_core::sdf_collider::{ClosureSdf, SdfCollider};

    for l in [1i64 << 40, 1i64 << 61] {
        let mut w = PhysicsWorld::new(SolverConfig {
            gravity: Vec3Fix::ZERO,
            ..SolverConfig::default()
        });
        w.add_sdf_collider(SdfCollider::new_static(
            Box::new(ClosureSdf::new(
                |x, y, z| (x * x + y * y + z * z).sqrt() - 1.0,
                |x, y, z| {
                    let n = (x * x + y * y + z * z).sqrt();
                    (x / n, y / n, z / n)
                },
            )),
            Vec3Fix::from_int(l, 0, 0),
            QuatFix::IDENTITY,
        ));
        let hit = w
            .cast_ray(
                Vec3Fix::ZERO,
                Vec3Fix::from_int(1, 0, 0),
                Fix128::from_int(l),
                &RayFilter::default(),
            )
            .unwrap_or_else(|| panic!("silent None: sdf l={l}"));
        let err = (hit.t - Fix128::from_int(l - 1)).abs().to_f64();
        assert!(err <= 2e-3, "sdf l={l}: |t − (l − 1)| = {err:e}");
    }
}
