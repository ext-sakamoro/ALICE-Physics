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
use crate::solver::PhysicsWorld;
use crate::static_collider::StaticCollider;
use crate::trimesh::TriMesh;

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
