//! Argument checking shared by the C ABI, WebAssembly and Python bindings.
//!
//! The bindings take plain `f64` / `u32` arguments from a host that cannot
//! see Rust's types. Every check lives here once, so the three bindings agree
//! on what they accept: a non-finite or non-positive size, an out-of-range
//! body id or index, a joint between a body and itself, or a malformed mesh
//! is refused (`None` / `false`) instead of reaching a `PhysicsWorld` method
//! that would panic (`add_joint`) or quietly store an unusable value.

use crate::heightfield::HeightField;
use crate::joint::Joint;
use crate::math::{Fix128, QuatFix, Vec3Fix};
use crate::plane_collider::PlaneCollider;
use crate::shape::Shape;
use crate::shape_raycast::{RayFilter, RayTarget};
use crate::solver::PhysicsWorld;
use crate::static_collider::StaticCollider;
use crate::trimesh::TriMesh;
use crate::world_shape_query::WorldShapeHit;

/// Shape kind codes used by every binding (`AlicePhysicsShape::kind` in C).
pub(crate) const SHAPE_BOX: u32 = 0;
pub(crate) const SHAPE_CYLINDER: u32 = 1;
pub(crate) const SHAPE_CONE: u32 = 2;
pub(crate) const SHAPE_ELLIPSOID: u32 = 3;
pub(crate) const SHAPE_WEDGE: u32 = 4;
pub(crate) const SHAPE_TORUS: u32 = 5;

/// A finite value.
pub(crate) fn finite(x: f64) -> Option<Fix128> {
    x.is_finite().then(|| Fix128::from_f64(x))
}

/// A finite value greater than zero.
pub(crate) fn positive(x: f64) -> Option<Fix128> {
    (x.is_finite() && x > 0.0).then(|| Fix128::from_f64(x))
}

/// A finite value that is not negative.
pub(crate) fn non_negative(x: f64) -> Option<Fix128> {
    (x.is_finite() && x >= 0.0).then(|| Fix128::from_f64(x))
}

/// A finite vector.
pub(crate) fn vec3(x: f64, y: f64, z: f64) -> Option<Vec3Fix> {
    Some(Vec3Fix::new(finite(x)?, finite(y)?, finite(z)?))
}

/// Two finite vectors from six values `[ax, ay, az, bx, by, bz]`.
#[cfg(feature = "wasm")]
pub(crate) fn two_vec3(v: &[f64]) -> Option<(Vec3Fix, Vec3Fix)> {
    let [ax, ay, az, bx, by, bz] = <[f64; 6]>::try_from(v).ok()?;
    Some((vec3(ax, ay, az)?, vec3(bx, by, bz)?))
}

/// A finite vector from exactly three values `[x, y, z]`.
#[cfg(feature = "wasm")]
pub(crate) fn vec3_slice(v: &[f64]) -> Option<Vec3Fix> {
    let [x, y, z] = <[f64; 3]>::try_from(v).ok()?;
    vec3(x, y, z)
}

/// A finite, non-zero quaternion, normalised.
pub(crate) fn unit_quat(x: f64, y: f64, z: f64, w: f64) -> Option<QuatFix> {
    let q = QuatFix::new(finite(x)?, finite(y)?, finite(z)?, finite(w)?);
    if q.length().is_zero() {
        return None;
    }
    Some(q.normalize())
}

/// The shape `kind` with sizes `a`, `b`, `c` (see the `SHAPE_*` codes):
/// box half extents `a, b, c`; cylinder / cone radius `a`, half height `b`;
/// ellipsoid radii `a, b, c`; wedge width `a`, height `b`, depth `c`; torus
/// major radius `a`, minor radius `b`. Unused sizes are ignored. Every used
/// size must be finite and positive, and a torus needs `minor < major`.
pub(crate) fn shape(kind: u32, a: f64, b: f64, c: f64) -> Option<Shape> {
    Some(match kind {
        SHAPE_BOX => Shape::Box {
            half_extents: Vec3Fix::new(positive(a)?, positive(b)?, positive(c)?),
        },
        SHAPE_CYLINDER => Shape::Cylinder {
            radius: positive(a)?,
            half_height: positive(b)?,
        },
        SHAPE_CONE => Shape::Cone {
            radius: positive(a)?,
            half_height: positive(b)?,
        },
        SHAPE_ELLIPSOID => Shape::Ellipsoid {
            radii: Vec3Fix::new(positive(a)?, positive(b)?, positive(c)?),
        },
        SHAPE_WEDGE => Shape::Wedge {
            width: positive(a)?,
            height: positive(b)?,
            depth: positive(c)?,
        },
        SHAPE_TORUS => {
            let (major, minor) = (positive(a)?, positive(b)?);
            if minor >= major {
                return None;
            }
            Shape::Torus {
                major_radius: major,
                minor_radius: minor,
            }
        }
        _ => return None,
    })
}

/// Set a body's collision sphere radius. `false` for an unknown body or a
/// radius that is not finite and positive.
pub(crate) fn set_collision_radius(w: &mut PhysicsWorld, body: usize, radius: f64) -> bool {
    match positive(radius) {
        Some(r) if body < w.bodies.len() => {
            w.set_body_collision_radius(body, r);
            true
        }
        _ => false,
    }
}

/// Drop a body's own collision radius (it falls back to the world default).
pub(crate) fn clear_collision_radius(w: &mut PhysicsWorld, body: usize) -> bool {
    if body >= w.bodies.len() {
        return false;
    }
    w.clear_body_collision_radius(body);
    true
}

/// Add a dynamic body with `shape`, its mass and inertia from `density`.
pub(crate) fn add_shaped_body(
    w: &mut PhysicsWorld,
    shape: Shape,
    density: f64,
    position: Vec3Fix,
) -> Option<usize> {
    w.add_shaped_body(&shape, positive(density)?, position).ok()
}

/// Give an existing body a collision shape (its mass is unchanged).
pub(crate) fn set_body_shape(w: &mut PhysicsWorld, body: usize, shape: Shape) -> bool {
    body < w.bodies.len() && w.set_body_shape(body, &shape)
}

/// Add the plane `normal · p = offset`; the normal is normalised and must
/// not be zero.
pub(crate) fn add_static_plane(
    w: &mut PhysicsWorld,
    normal: Vec3Fix,
    offset: f64,
) -> Option<usize> {
    let n = normal.try_normalize()?;
    Some(
        w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
            n,
            finite(offset)?,
        ))),
    )
}

/// Add a height field of `width × depth` heights (row-major, `x` fastest)
/// spaced `spacing` apart from the min corner `origin`.
pub(crate) fn add_static_heightfield(
    w: &mut PhysicsWorld,
    heights: &[f64],
    width: u32,
    depth: u32,
    spacing: f64,
    origin: Vec3Fix,
) -> Option<usize> {
    if width < 2 || depth < 2 {
        return None;
    }
    let count = (width as usize).checked_mul(depth as usize)?;
    if heights.len() != count {
        return None;
    }
    let spacing = positive(spacing)?;
    let heights: Option<Vec<Fix128>> = heights.iter().map(|&h| finite(h)).collect();
    Some(
        w.add_static_collider(StaticCollider::HeightField(HeightField::new(
            heights?, width, depth, spacing, origin,
        ))),
    )
}

/// Add a triangle mesh: `vertices` holds `x, y, z` triples, `indices` holds
/// three vertex indices per triangle.
pub(crate) fn add_static_trimesh(
    w: &mut PhysicsWorld,
    vertices: &[f64],
    indices: &[u32],
) -> Option<usize> {
    if vertices.is_empty()
        || vertices.len() % 3 != 0
        || indices.is_empty()
        || indices.len() % 3 != 0
    {
        return None;
    }
    let verts: Option<Vec<Vec3Fix>> = vertices
        .chunks_exact(3)
        .map(|v| vec3(v[0], v[1], v[2]))
        .collect();
    let verts = verts?;
    if indices.iter().any(|&i| i as usize >= verts.len()) {
        return None;
    }
    Some(
        w.add_static_collider(StaticCollider::TriMesh(TriMesh::from_indexed(
            &verts, indices,
        ))),
    )
}

/// Remove static collider `index` (later colliders shift down by one).
pub(crate) fn remove_static_collider(w: &mut PhysicsWorld, index: usize) -> bool {
    w.remove_static_collider(index).is_some()
}

/// Add `joint` after checking its bodies: both must exist and differ (a
/// joint between a body and itself injects velocity, AUD-A-S34-010).
pub(crate) fn add_joint(w: &mut PhysicsWorld, joint: Joint) -> Option<usize> {
    let (a, b) = joint.bodies();
    if a >= w.bodies.len() || b >= w.bodies.len() || a == b {
        return None;
    }
    Some(w.add_joint(joint))
}

/// Remove joint `index`. The last joint moves into `index` (`swap_remove`).
pub(crate) fn remove_joint(w: &mut PhysicsWorld, index: usize) -> bool {
    w.remove_joint(index).is_some()
}

// ============================================================================
// World queries and body observation
// ============================================================================

/// Target kind codes of a query result, used by every binding: a body, a
/// static collider or an SDF collider (each with its index).
pub(crate) const TARGET_BODY: u32 = 0;
pub(crate) const TARGET_STATIC: u32 = 1;
pub(crate) const TARGET_SDF: u32 = 2;

/// A query hit as the bindings return it: every value of the Rust hit
/// through [`Fix128::to_f64`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct QueryHit {
    pub t: f64,
    pub point: [f64; 3],
    pub normal: [f64; 3],
    /// `(TARGET_*, index)`.
    pub target: (u32, usize),
    /// The body the hit belongs to (`None` for a static collider or an SDF
    /// collider attached to no body).
    pub body: Option<usize>,
}

/// A body observation as the bindings return it (see
/// [`PhysicsWorld::observe_body`]), every value through [`Fix128::to_f64`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Observation {
    pub body_index: usize,
    pub position: [f64; 3],
    pub velocity: [f64; 3],
    /// `[x, y, z, w]`.
    pub rotation: [f64; 4],
    pub angular_velocity: [f64; 3],
    pub sleeping: bool,
    pub in_contact: bool,
}

fn f64x3(v: Vec3Fix) -> [f64; 3] {
    [v.x.to_f64(), v.y.to_f64(), v.z.to_f64()]
}

/// `(TARGET_*, index)` of a query target.
pub(crate) fn target_code(t: RayTarget) -> (u32, usize) {
    match t {
        RayTarget::Body(i) => (TARGET_BODY, i),
        RayTarget::StaticCollider(i) => (TARGET_STATIC, i),
        RayTarget::Sdf(i) => (TARGET_SDF, i),
    }
}

fn shape_hit(h: WorldShapeHit) -> QueryHit {
    QueryHit {
        t: h.t.to_f64(),
        point: f64x3(h.point),
        normal: f64x3(h.normal),
        target: target_code(h.target),
        body: h.body,
    }
}

/// The filter of a binding query: the default [`RayFilter`], ignoring
/// `exclude_body` when given. `None` when `exclude_body` is not a body of `w`.
pub(crate) fn query_filter(w: &PhysicsWorld, exclude_body: Option<usize>) -> Option<RayFilter> {
    match exclude_body {
        None => Some(RayFilter::default()),
        Some(b) if b < w.bodies.len() => Some(RayFilter::default().excluding_body(b)),
        Some(_) => None,
    }
}

/// [`PhysicsWorld::cast_ray`] in f64.
pub(crate) fn cast_ray(
    w: &PhysicsWorld,
    origin: Vec3Fix,
    direction: Vec3Fix,
    max_t: Fix128,
    filter: &RayFilter,
) -> Option<QueryHit> {
    w.cast_ray(origin, direction, max_t, filter)
        .map(|h| QueryHit {
            t: h.t.to_f64(),
            point: f64x3(h.point),
            normal: f64x3(h.normal),
            target: target_code(h.target),
            body: h.body,
        })
}

/// [`PhysicsWorld::cast_sphere`] in f64.
pub(crate) fn cast_sphere(
    w: &PhysicsWorld,
    center: Vec3Fix,
    radius: Fix128,
    direction: Vec3Fix,
    max_t: Fix128,
    filter: &RayFilter,
) -> Option<QueryHit> {
    w.cast_sphere(center, radius, direction, max_t, filter)
        .map(shape_hit)
}

/// [`PhysicsWorld::cast_capsule`] in f64.
pub(crate) fn cast_capsule(
    w: &PhysicsWorld,
    a: Vec3Fix,
    b: Vec3Fix,
    radius: Fix128,
    direction: Vec3Fix,
    max_t: Fix128,
    filter: &RayFilter,
) -> Option<QueryHit> {
    w.cast_capsule(a, b, radius, direction, max_t, filter)
        .map(shape_hit)
}

/// [`PhysicsWorld::overlap_sphere`] as `(TARGET_*, index)` pairs, sorted.
pub(crate) fn overlap_sphere(
    w: &PhysicsWorld,
    center: Vec3Fix,
    radius: Fix128,
    filter: &RayFilter,
) -> Vec<(u32, usize)> {
    w.overlap_sphere(center, radius, filter)
        .into_iter()
        .map(target_code)
        .collect()
}

/// [`PhysicsWorld::observe_body`] in f64; `None` for an unknown body.
pub(crate) fn observe_body(w: &PhysicsWorld, body: usize) -> Option<Observation> {
    let o = w.observe_body(body)?;
    Some(Observation {
        body_index: o.body_index,
        position: f64x3(o.position),
        velocity: f64x3(o.velocity),
        rotation: [
            o.rotation.x.to_f64(),
            o.rotation.y.to_f64(),
            o.rotation.z.to_f64(),
            o.rotation.w.to_f64(),
        ],
        angular_velocity: f64x3(o.angular_velocity),
        sleeping: o.sleeping,
        in_contact: o.in_contact,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::{PhysicsConfig, RigidBody};

    fn vf(x: f64, y: f64, z: f64) -> Vec3Fix {
        vec3(x, y, z).unwrap()
    }

    fn arr(v: Vec3Fix) -> [f64; 3] {
        [v.x.to_f64(), v.y.to_f64(), v.z.to_f64()]
    }

    /// A static body of collision radius 1 at (0, 0, 10), a dynamic body of
    /// radius 1 at (0, 0, 20) and the static plane y = -2.
    fn scene() -> PhysicsWorld {
        let mut w = PhysicsWorld::new(PhysicsConfig::default());
        w.add_body(RigidBody::new_static(vf(0.0, 0.0, 10.0)));
        w.add_body(RigidBody::new_dynamic(vf(0.0, 0.0, 20.0), Fix128::ONE));
        assert!(set_collision_radius(&mut w, 0, 1.0));
        assert!(set_collision_radius(&mut w, 1, 1.0));
        add_static_plane(&mut w, Vec3Fix::UNIT_Y, -2.0).unwrap();
        w
    }

    /// oracle: the f64 query results every binding returns are the Rust
    /// API's results through `Fix128::to_f64` (bit-equal), and they are the
    /// closed form on this scene (t = 9 / 8.5 / 8.5 / 2).
    #[test]
    fn query_results_are_the_rust_results_in_f64() {
        let w = scene();
        let f = RayFilter::default();
        let fx = Fix128::from_f64;

        let h = cast_ray(&w, Vec3Fix::ZERO, Vec3Fix::UNIT_Z, fx(100.0), &f).unwrap();
        let r = w
            .cast_ray(Vec3Fix::ZERO, Vec3Fix::UNIT_Z, fx(100.0), &f)
            .unwrap();
        assert_eq!(h.t.to_bits(), r.t.to_f64().to_bits());
        assert_eq!((h.point, h.normal), (arr(r.point), arr(r.normal)));
        assert_eq!((h.target, h.body), ((TARGET_BODY, 0), Some(0)));
        assert!((h.t - 9.0).abs() < 1e-12);

        // off-axis: lateral d^2 = 0.3125, t = 10 - sqrt(1 - d^2)
        let o = vf(0.25, 0.5, 0.0);
        let h = cast_ray(&w, o, Vec3Fix::UNIT_Z, fx(100.0), &f).unwrap();
        let r = w.cast_ray(o, Vec3Fix::UNIT_Z, fx(100.0), &f).unwrap();
        assert_eq!((h.point, h.normal), (arr(r.point), arr(r.normal)));
        let tc = 10.0 - 0.6875f64.sqrt();
        assert!((h.t - tc).abs() < 1e-12);
        for (got, want) in
            h.point
                .iter()
                .chain(&h.normal)
                .zip([0.25, 0.5, tc, 0.25, 0.5, tc - 10.0])
        {
            assert!((got - want).abs() < 1e-12, "{h:?}");
        }

        let h = cast_ray(&w, Vec3Fix::ZERO, vf(0.0, -1.0, 0.0), fx(10.0), &f).unwrap();
        assert_eq!((h.target, h.body), ((TARGET_STATIC, 0), None));
        assert!((h.t - 2.0).abs() < 1e-12);
        assert!((h.normal[1] - 1.0).abs() < 1e-12);

        let h = cast_sphere(&w, Vec3Fix::ZERO, fx(0.5), Vec3Fix::UNIT_Z, fx(100.0), &f).unwrap();
        let r = w
            .cast_sphere(Vec3Fix::ZERO, fx(0.5), Vec3Fix::UNIT_Z, fx(100.0), &f)
            .unwrap();
        assert_eq!(h.t.to_bits(), r.t.to_f64().to_bits());
        assert_eq!((h.point, h.normal), (arr(r.point), arr(r.normal)));
        assert!((h.t - 8.5).abs() < 1e-12);

        let (a, b) = (vf(-1.0, 0.0, 0.0), vf(1.0, 0.0, 0.0));
        let h = cast_capsule(&w, a, b, fx(0.5), Vec3Fix::UNIT_Z, fx(100.0), &f).unwrap();
        let r = w
            .cast_capsule(a, b, fx(0.5), Vec3Fix::UNIT_Z, fx(100.0), &f)
            .unwrap();
        assert_eq!(h.t.to_bits(), r.t.to_f64().to_bits());
        assert_eq!((h.point, h.normal), (arr(r.point), arr(r.normal)));
        assert!((h.t - 8.5).abs() < 1e-12);

        assert_eq!(
            overlap_sphere(&w, vf(0.0, -1.2, 10.0), Fix128::ONE, &f),
            vec![(TARGET_BODY, 0), (TARGET_STATIC, 0)]
        );
        assert!(cast_ray(&w, Vec3Fix::ZERO, Vec3Fix::UNIT_X, fx(100.0), &f).is_none());
    }

    /// oracle: an exclude index that is not a body is refused; one that is
    /// gives the default filter ignoring it.
    #[test]
    fn query_filter_checks_the_excluded_body() {
        let w = scene();
        assert_eq!(query_filter(&w, None), Some(RayFilter::default()));
        assert_eq!(
            query_filter(&w, Some(1)),
            Some(RayFilter::default().excluding_body(1))
        );
        assert_eq!(query_filter(&w, Some(2)), None);
    }

    /// oracle: the f64 observation is the Rust observation through `to_f64`.
    #[test]
    fn observation_is_the_rust_observation_in_f64() {
        let mut w = scene();
        w.bodies[1].velocity = vf(1.0, 2.0, 3.0);
        w.step(Fix128::from_f64(1.0 / 60.0));
        let o = observe_body(&w, 1).unwrap();
        let r = w.observe_body(1).unwrap();
        assert_eq!(o.body_index, 1);
        assert_eq!(o.position, arr(r.position));
        assert_eq!(o.velocity, arr(r.velocity));
        assert_eq!(
            o.rotation,
            [
                r.rotation.x.to_f64(),
                r.rotation.y.to_f64(),
                r.rotation.z.to_f64(),
                r.rotation.w.to_f64()
            ]
        );
        assert_eq!(o.angular_velocity, arr(r.angular_velocity));
        assert_eq!((o.sleeping, o.in_contact), (r.sleeping, r.in_contact));
        assert!(observe_body(&w, 2).is_none());
    }
}
