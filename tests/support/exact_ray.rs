//! An exact ray / triangle reference for the raycast oracles, independent of the crate's
//! predicate: `num-bigint` arithmetic on the raw `Fix128` integers, and Cramer's rule on the
//! 3 × 3 system u·e1 + v·e2 − t·d = o − v0 (the crate tests orientation numerators instead).
//!
//! The ray meets the closed triangle at t in [0, max_t] exactly when the system's determinant is
//! not 0 and u ≥ 0, v ≥ 0, u + v ≤ 1, 0 ≤ t ≤ max_t; the hit's t is floor(t · 2^64) / 2^64. A ray
//! parallel to the triangle's plane does not hit it, coplanar or not: parallel is a sine of the
//! angle with the plane below 2^-48 (determinant² · 2^96 < |e1 × e2|² · |d|², 0 included).

use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::raycast::Ray;
use alice_physics_core::trimesh::Triangle;
use num_bigint::BigInt;
use num_traits::{Signed, Zero};

type V = [BigInt; 3];

fn raw(x: Fix128) -> BigInt {
    (BigInt::from(x.hi) << 64) + BigInt::from(x.lo)
}

fn vec(v: Vec3Fix) -> V {
    [raw(v.x), raw(v.y), raw(v.z)]
}

fn sub(a: &V, b: &V) -> V {
    [&a[0] - &b[0], &a[1] - &b[1], &a[2] - &b[2]]
}

fn cross(a: &V, b: &V) -> V {
    [
        &a[1] * &b[2] - &a[2] * &b[1],
        &a[2] * &b[0] - &a[0] * &b[2],
        &a[0] * &b[1] - &a[1] * &b[0],
    ]
}

fn dot(a: &V, b: &V) -> BigInt {
    &a[0] * &b[0] + &a[1] * &b[1] + &a[2] * &b[2]
}

/// The exact hit's t (floor to the `Fix128` grid), or None
pub fn exact_t(ray: &Ray, tri: &Triangle, max_t: Fix128) -> Option<Fix128> {
    let (o, d) = (vec(ray.origin), vec(ray.direction));
    let v0 = vec(tri.v0);
    let a = sub(&vec(tri.v1), &v0);
    let b = sub(&vec(tri.v2), &v0);
    let c = [-&d[0], -&d[1], -&d[2]];
    let r = sub(&o, &v0);
    let det = dot(&a, &cross(&b, &c));
    // parallel when the sine of the angle with the plane is below 2^-48:
    // det² · 2^96 < |a × b|² · |d|² (det 0 included)
    let n = cross(&a, &b);
    if det.is_zero() || (&det * &det) << 96 < dot(&n, &n) * dot(&d, &d) {
        return None;
    }
    // numerators of u, v, t, with the determinant's sign folded in
    let sign = BigInt::from(if det.is_negative() { -1 } else { 1 });
    let u = dot(&r, &cross(&b, &c)) * &sign;
    let v = dot(&a, &cross(&r, &c)) * &sign;
    let t = dot(&a, &cross(&b, &r)) * &sign;
    let det = det.abs();
    if u.is_negative() || v.is_negative() || &u + &v > det || t.is_negative() {
        return None;
    }
    // t ≤ max_t ⇔ t_num · 2^64 ≤ raw(max_t) · det
    let scaled = &t << 64;
    if scaled > raw(max_t) * &det {
        return None;
    }
    let q: BigInt = scaled / &det; // floor: both are non-negative
    let lo = &q & BigInt::from(u64::MAX);
    let hi: BigInt = &q >> 64;
    Some(Fix128::from_raw(
        i64::try_from(hi).expect("t fits Fix128"),
        u64::try_from(lo).expect("low word"),
    ))
}

/// The nearest exact hit over every triangle: (t, index), the smallest index on an exact tie
pub fn exact_nearest(tris: &[Triangle], ray: &Ray, max_t: Fix128) -> Option<(Fix128, usize)> {
    let mut best: Option<(Fix128, usize)> = None;
    for (i, tri) in tris.iter().enumerate() {
        if let Some(t) = exact_t(ray, tri, max_t) {
            if best.is_none_or(|(bt, _)| t < bt) {
                best = Some((t, i));
            }
        }
    }
    best
}
