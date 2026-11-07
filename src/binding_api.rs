//! Argument checking shared by the C ABI, WebAssembly and Python bindings.
//!
//! The bindings take plain `f64` / `u32` arguments from a host that cannot
//! see Rust's types. Every check lives here once, so the three bindings agree
//! on what they accept: a non-finite or non-positive size, an out-of-range
//! body id or index, a joint between a body and itself, or a malformed mesh
//! is refused (`None` / `false`) instead of reaching a `PhysicsWorld` method
//! that would panic (`add_joint`) or quietly store an unusable value.

use crate::electromagnetic::MagneticResponse;
use crate::force::{ForceField, ForceFieldInstance};
use crate::heightfield::HeightField;
use crate::joint::Joint;
use crate::material::{MaterialId, PhysicsMaterial};
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

/// The smallest and largest size of the largest component that [`unit_quat`] takes as it is
const UNIT_QUAT_SMALLEST: f64 = 1.0 / 65_536.0;
const UNIT_QUAT_LARGEST: f64 = 1_073_741_824.0;

/// A finite, non-zero quaternion, normalised.
pub(crate) fn unit_quat(x: f64, y: f64, z: f64, w: f64) -> Option<QuatFix> {
    // the largest size, by comparison only, so no value is chosen by a max of equal zeros; a NaN
    // component leaves a NaN or a finite size that its own division then refuses
    let mut largest = x.abs();
    for c in [y.abs(), z.abs(), w.abs()] {
        if c > largest {
            largest = c;
        }
    }
    // Inside these sizes the squared length of four components is a `Fix128` with digits to spare.
    // Outside, it overflows or rounds to zero, and a rotation does not depend on the scale.
    // The zero quaternion and a component that is not finite give NaN here, which `finite` refuses.
    let divisor = if (UNIT_QUAT_SMALLEST..=UNIT_QUAT_LARGEST).contains(&largest) {
        1.0
    } else {
        largest
    };
    let q = QuatFix::new(
        finite(x / divisor)?,
        finite(y / divisor)?,
        finite(z / divisor)?,
        finite(w / divisor)?,
    );
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

/// Set a body's orientation to the quaternion `(x, y, z, w)`, normalised, through
/// [`RigidBody::set_rotation`] (which also sets the previous orientation the parking
/// check and continuous collision read, so the change is not a motion). `false` for an
/// unknown body, a non-finite component or the zero quaternion.
///
/// [`RigidBody::set_rotation`]: crate::solver::RigidBody::set_rotation
pub(crate) fn set_rotation(
    w: &mut PhysicsWorld,
    body: usize,
    x: f64,
    y: f64,
    z: f64,
    qw: f64,
) -> bool {
    match (unit_quat(x, y, z, qw), w.bodies.get_mut(body)) {
        (Some(q), Some(b)) => {
            b.set_rotation(q);
            // a sleeping body would stay as it was set, with nothing to move it
            if b.is_dynamic() {
                w.wake_body(body);
            }
            true
        }
        _ => false,
    }
}

/// Set a body's angular velocity (world frame, radians per second). `false`
/// for an unknown body or a non-finite component.
pub(crate) fn set_angular_velocity(
    w: &mut PhysicsWorld,
    body: usize,
    x: f64,
    y: f64,
    z: f64,
) -> bool {
    match (vec3(x, y, z), w.bodies.get_mut(body)) {
        (Some(v), Some(b)) => {
            b.angular_velocity = v;
            // a sleeping body would not spin; a zero spin leaves it, so a caller that stops a body
            // every frame still lets it sleep (the rule of `PhysicsWorld::set_body_velocity`)
            if b.is_dynamic() && v != Vec3Fix::ZERO {
                w.wake_body(body);
            }
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

/// A material id from a host integer (`MaterialId` is 16 bits); `None` above `u16::MAX`.
pub(crate) fn material_id(material: u32) -> Option<MaterialId> {
    MaterialId::try_from(material).ok()
}

/// Set the material of a body. `false` for an unknown body or a material id that does not
/// fit `MaterialId`.
pub(crate) fn set_body_material(w: &mut PhysicsWorld, body: usize, material: u32) -> bool {
    match material_id(material) {
        Some(m) if body < w.bodies.len() => {
            w.set_body_material(body, m);
            true
        }
        _ => false,
    }
}

/// Register a material with `friction` and `restitution` (both finite and not negative; the
/// average combine rule of the default material) and return its id. A body given this id with
/// `set_body_material` meets others with these values instead of its own; `None` for a bad value
/// or when every id is in use.
pub(crate) fn register_material(
    w: &mut PhysicsWorld,
    friction: f64,
    restitution: f64,
) -> Option<usize> {
    let m = PhysicsMaterial::new(0, non_negative(friction)?, non_negative(restitution)?);
    w.material_table.try_register(m).ok().map(usize::from)
}

/// The magnetic part of a magnet: the material it pulls and that material's magnetic
/// properties (see [`ForceField::InducedDipole`]).
pub(crate) struct MagnetMaterial {
    /// The material whose bodies are pulled (host integer, checked against `MaterialId`)
    pub material: u32,
    /// Effective susceptibility (dimensionless), finite and not negative
    pub chi_eff: f64,
    /// Saturation magnetisation (A/m), finite and not negative
    pub saturation: f64,
    /// Density of the material (kg/m³), finite and positive
    pub density: f64,
}

/// A blow from a point: a finite centre and strength, a positive radius.
#[cfg(feature = "wasm")]
pub(crate) fn radial_impulse(
    w: &mut PhysicsWorld,
    center: Vec3Fix,
    radius: f64,
    strength: f64,
) -> Option<usize> {
    Some(w.apply_radial_impulse(center, positive(radius)?, finite(strength)?))
}

/// Add a permanent magnet ([`ForceField::InducedDipole`]) at `position` with dipole `moment`
/// and its own `radius`; returns the field index. `None` for a radius or density that is not
/// finite and positive, a susceptibility or saturation that is negative or not finite, a
/// material id that does not fit `MaterialId`, or a material another magnet already names with
/// different magnetic properties (the world keeps its fields unchanged).
pub(crate) fn add_magnet(
    w: &mut PhysicsWorld,
    position: Vec3Fix,
    moment: Vec3Fix,
    radius: f64,
    m: &MagnetMaterial,
) -> Option<usize> {
    // positive after the conversion too (1e-300 would become a fixed-point zero)
    let radius = positive(radius).filter(|r| *r > Fix128::ZERO)?;
    let density = positive(m.density).filter(|d| *d > Fix128::ZERO)?;
    let field = ForceField::InducedDipole {
        position,
        moment,
        radius,
        material: material_id(m.material)?,
        response: MagneticResponse {
            chi_eff: non_negative(m.chi_eff)?,
            saturation_magnetization: non_negative(m.saturation)?,
        },
        density,
    };
    w.try_add_force_field(ForceFieldInstance::new(field)).ok()
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

/// The contacts of the last step, flat for a host that sees only numbers:
/// `[bodyA, bodyB, kind, speed, px, py, pz, nx, ny, nz]` per contact. `kind` is 0 begin,
/// 1 persist, 2 end, 3 a static collider (`bodyB` is then the collider's index, negated
/// and less one, so it cannot be taken for a body). `speed` is the speed of approach.
#[cfg(feature = "wasm")]
#[must_use]
pub(crate) fn contact_events(world: &PhysicsWorld) -> Vec<f64> {
    use crate::event::ContactEventType;

    let mut out = Vec::new();
    for e in world.contact_events() {
        let kind = match e.event_type {
            ContactEventType::Begin => 0.0,
            ContactEventType::Persist => 1.0,
            ContactEventType::End => 2.0,
        };
        // `relative_velocity` is `(v_a - v_b) . normal` with the normal from b to a, so
        // approaching is negative.
        let speed = (-e.relative_velocity).max(Fix128::ZERO).to_f64();
        out.extend_from_slice(&[e.body_a as f64, e.body_b as f64, kind, speed]);
        out.extend_from_slice(&[e.point.x.to_f64(), e.point.y.to_f64(), e.point.z.to_f64()]);
        out.extend_from_slice(&[
            e.normal.x.to_f64(),
            e.normal.y.to_f64(),
            e.normal.z.to_f64(),
        ]);
    }
    for h in world.static_hits() {
        out.extend_from_slice(&[
            h.body as f64,
            -(h.collider as f64) - 1.0,
            3.0,
            h.speed.to_f64(),
        ]);
        out.extend_from_slice(&[h.point.x.to_f64(), h.point.y.to_f64(), h.point.z.to_f64()]);
        out.extend_from_slice(&[
            h.normal.x.to_f64(),
            h.normal.y.to_f64(),
            h.normal.z.to_f64(),
        ]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::{PhysicsConfig, RigidBody};

    fn steel() -> MagnetMaterial {
        MagnetMaterial {
            material: 1,
            chi_eff: 3.0,
            saturation: 1.7e6,
            density: 7850.0,
        }
    }

    /// Two 1 cm steel-mass bodies 6 cm from a magnet at the origin (moment along +y), one
    /// of material 1 and one of the default material, no gravity.
    fn magnet_scene() -> (PhysicsWorld, usize, usize) {
        let c = PhysicsConfig {
            gravity: Vec3Fix::ZERO,
            damping_rate: Fix128::ZERO,
            ..PhysicsConfig::default()
        };
        let mut w = PhysicsWorld::new(c);
        let mass = Fix128::from_ratio(4_110, 1_000_000);
        let pulled = w.add_body(RigidBody::new_dynamic(vf(0.0, 0.06, 0.0), mass));
        let other = w.add_body(RigidBody::new_dynamic(vf(0.0, -0.06, 0.0), mass));
        assert!(set_collision_radius(&mut w, pulled, 0.005));
        assert!(set_collision_radius(&mut w, other, 0.005));
        assert!(set_body_material(&mut w, pulled, 1));
        (w, pulled, other)
    }

    /// oracle: the binding's magnet is the Rust API's `ForceField::InducedDipole` with the
    /// f64 arguments through `Fix128::from_f64` (positions bit-equal after 60 steps); it pulls
    /// the body of its material toward it along the moment's axis (the body at +y moves down,
    /// staying on the axis) and leaves a body of another material where it was.
    #[test]
    fn magnet_is_the_rust_field_and_pulls_only_its_material() {
        let (mut a, pulled, other) = magnet_scene();
        let m = steel();
        let idx = add_magnet(&mut a, vf(0.0, 0.0, 0.0), vf(0.0, 0.5, 0.0), 0.01, &m).unwrap();
        assert_eq!(idx, 0);
        let (mut b, _, _) = magnet_scene();
        b.add_force_field(ForceFieldInstance::new(ForceField::InducedDipole {
            position: Vec3Fix::ZERO,
            moment: vf(0.0, 0.5, 0.0),
            radius: Fix128::from_f64(0.01),
            material: 1,
            response: MagneticResponse {
                chi_eff: Fix128::from_f64(3.0),
                saturation_magnetization: Fix128::from_f64(1.7e6),
            },
            density: Fix128::from_f64(7850.0),
        }));
        let dt = Fix128::from_ratio(1, 120);
        for _ in 0..60 {
            a.step(dt);
            b.step(dt);
        }
        for i in [pulled, other] {
            assert_eq!(a.bodies[i].position, b.bodies[i].position, "body {i}");
        }
        assert!(
            a.bodies[pulled].position.y < vf(0.0, 0.06, 0.0).y
                && a.bodies[pulled].position.x == Fix128::ZERO,
            "{:?}",
            arr(a.bodies[pulled].position)
        );
        assert_eq!(a.bodies[other].position, vf(0.0, -0.06, 0.0));
    }

    /// A ball rolling into a resting ball with no gravity, once with its own friction and
    /// restitution and once tagged with a registered material carrying the same two values;
    /// returns the two worlds after 120 steps
    fn bounce_pair(friction: f64, restitution: f64) -> (PhysicsWorld, PhysicsWorld) {
        let make = |tag: bool| {
            let c = PhysicsConfig {
                gravity: Vec3Fix::ZERO,
                ..PhysicsConfig::default()
            };
            let mut w = PhysicsWorld::new(c);
            // shaped balls, as the browser game makes them (`addShapedBody` with an ellipsoid)
            let ball = |w: &mut PhysicsWorld, x: f64| {
                add_shaped_body(
                    w,
                    shape(SHAPE_ELLIPSOID, 0.2, 0.2, 0.2).unwrap(),
                    7800.0,
                    vf(x, 0.0, 0.0),
                )
                .unwrap()
            };
            let a = ball(&mut w, 0.0);
            assert_eq!(ball(&mut w, 1.0), 1, "the resting ball is body 1");
            w.bodies[a].friction = Fix128::from_f64(friction);
            w.bodies[a].restitution = Fix128::from_f64(restitution);
            w.bodies[a].velocity = vf(1.5, 0.1, 0.0);
            if tag {
                let m = register_material(&mut w, friction, restitution).unwrap();
                assert_eq!(m, 1, "the first id after the default material");
                assert!(set_body_material(&mut w, a, u32::try_from(m).unwrap()));
            }
            let dt = Fix128::from_ratio(1, 120);
            for _ in 0..120 {
                w.step(dt);
            }
            w
        };
        (make(false), make(true))
    }

    /// oracle: a body tagged with a registered material of its own friction and restitution meets
    /// another body exactly as it did untagged (bit-equal positions and velocities after 120 steps
    /// with a hit), so tagging a body for a magnet keeps its surface; other values change the run.
    #[test]
    fn registered_material_with_the_body_s_own_values_changes_nothing() {
        let (a, b) = bounce_pair(0.3, 0.15);
        for i in 0..2 {
            assert_eq!(a.bodies[i].position, b.bodies[i].position, "body {i}");
            assert_eq!(a.bodies[i].velocity, b.bodies[i].velocity, "body {i}");
        }
        // the hit happened: the resting ball moves
        assert!(a.bodies[1].velocity.x > Fix128::ZERO);
        let (c, _) = bounce_pair(0.9, 0.8);
        assert_ne!(a.bodies[1].velocity, c.bodies[1].velocity);
        let mut w = PhysicsWorld::new(PhysicsConfig::default());
        for (f, r) in [
            (-0.1, 0.5),
            (0.5, -0.1),
            (f64::NAN, 0.5),
            (0.5, f64::INFINITY),
        ] {
            assert_eq!(register_material(&mut w, f, r), None, "{f} {r}");
        }
        assert_eq!(w.material_table.len(), 1);
        assert_eq!(register_material(&mut w, 0.0, 0.0), Some(1));
    }

    /// A ball that comes to rest on a mesh floor (asleep), then a magnet beside it at 0.8 m.
    /// `tagged_asleep`: the ball has the magnet's material while it sleeps and the magnet comes
    /// after (adding the magnet must wake it); otherwise the magnet comes first and the ball
    /// gets the material while asleep (the material change must wake it).
    fn sleeping_ball_then_magnet(tagged_asleep: bool) -> (PhysicsWorld, usize) {
        let mut w = PhysicsWorld::new(PhysicsConfig::default());
        add_static_trimesh(
            &mut w,
            &[
                -10.0, 0.0, -10.0, 10.0, 0.0, -10.0, 10.0, 0.0, 10.0, -10.0, 0.0, 10.0,
            ],
            &[0, 2, 1, 0, 3, 2],
        )
        .unwrap();
        let b = add_shaped_body(
            &mut w,
            shape(SHAPE_ELLIPSOID, 0.2, 0.2, 0.2).unwrap(),
            7800.0,
            vf(0.0, 0.2, 0.0),
        )
        .unwrap();
        if tagged_asleep {
            assert!(set_body_material(&mut w, b, 1));
        }
        let dt = Fix128::from_ratio(1, 120);
        for _ in 0..900 {
            w.step(dt);
        }
        assert!(
            w.is_sleeping(b),
            "the ball should be asleep before the magnet comes"
        );
        let m = MagnetMaterial {
            material: 1,
            chi_eff: 3.0,
            saturation: 1.7e6,
            density: 7800.0,
        };
        add_magnet(&mut w, vf(0.8, 0.2, 0.0), vf(2000.0, 0.0, 0.0), 0.1, &m).unwrap();
        if !tagged_asleep {
            assert!(
                w.is_sleeping(b),
                "a magnet of another material leaves the ball asleep"
            );
            assert!(set_body_material(&mut w, b, 1));
        }
        for _ in 0..120 {
            w.step(dt);
        }
        (w, b)
    }

    /// A sleeping ball feels a magnet: adding the magnet wakes the bodies of its material, and
    /// giving a sleeping body that material wakes it; either way it moves toward the magnet.
    #[test]
    fn a_magnet_moves_a_ball_that_was_asleep() {
        for tagged_asleep in [true, false] {
            let (w, b) = sleeping_ball_then_magnet(tagged_asleep);
            assert!(
                w.bodies[b].position.x > Fix128::ZERO,
                "tagged while asleep: {tagged_asleep}, x {:?}",
                arr(w.bodies[b].position)
            );
        }
    }

    /// Setting the material a sleeping body already has leaves it asleep.
    #[test]
    fn setting_the_same_material_does_not_wake() {
        let mut w = PhysicsWorld::new(PhysicsConfig::default());
        add_static_trimesh(
            &mut w,
            &[
                -10.0, 0.0, -10.0, 10.0, 0.0, -10.0, 10.0, 0.0, 10.0, -10.0, 0.0, 10.0,
            ],
            &[0, 2, 1, 0, 3, 2],
        )
        .unwrap();
        let b = add_shaped_body(
            &mut w,
            shape(SHAPE_ELLIPSOID, 0.2, 0.2, 0.2).unwrap(),
            7800.0,
            vf(0.0, 0.2, 0.0),
        )
        .unwrap();
        assert!(set_body_material(&mut w, b, 1));
        for _ in 0..900 {
            w.step(Fix128::from_ratio(1, 120));
        }
        assert!(w.is_sleeping(b));
        assert!(set_body_material(&mut w, b, 1));
        assert!(w.is_sleeping(b), "the same material woke the body");
        assert!(set_body_material(&mut w, b, 2));
        assert!(!w.is_sleeping(b), "a new material leaves the body asleep");
    }

    /// Every bad value is refused and leaves the world's fields as they were.
    #[test]
    fn bad_magnets_and_materials_are_refused() {
        let (mut w, pulled, _) = magnet_scene();
        let at = Vec3Fix::ZERO;
        let mo = vf(0.0, 0.5, 0.0);
        let with = |f: fn(&mut MagnetMaterial)| {
            let mut m = steel();
            f(&mut m);
            m
        };
        let bad = [
            (0.0, steel()),
            (-0.01, steel()),
            (f64::NAN, steel()),
            (0.01, with(|m| m.material = 65_536)),
            (0.01, with(|m| m.chi_eff = -1.0)),
            (0.01, with(|m| m.saturation = f64::INFINITY)),
            (0.01, with(|m| m.density = 0.0)),
            (f64::INFINITY, steel()),
            (1e-300, steel()),
            (0.01, with(|m| m.density = f64::NAN)),
            (0.01, with(|m| m.density = -7850.0)),
            (0.01, with(|m| m.density = 1e-300)),
            (0.01, with(|m| m.chi_eff = f64::NAN)),
            (0.01, with(|m| m.saturation = -1.0)),
        ];
        for (r, m) in &bad {
            assert_eq!(add_magnet(&mut w, at, mo, *r, m), None);
        }
        assert!(w.force_fields.is_empty());
        assert_eq!(add_magnet(&mut w, at, mo, 0.01, &steel()), Some(0));
        // a second magnet naming material 1 with other properties is refused
        for other in [
            with(|m| m.chi_eff = 2.0),
            with(|m| m.saturation = 1.6e6),
            with(|m| m.density = 7800.0),
        ] {
            assert_eq!(add_magnet(&mut w, at, mo, 0.01, &other), None);
        }
        assert_eq!(w.force_fields.len(), 1);
        // zero susceptibility and saturation are allowed (a material the field does not move)
        let inert = MagnetMaterial {
            material: 2,
            chi_eff: 0.0,
            saturation: 0.0,
            density: 1000.0,
        };
        assert_eq!(add_magnet(&mut w, at, mo, 0.01, &inert), Some(1));
        assert!(w.force_fields.pop().is_some());
        assert_eq!(add_magnet(&mut w, at, mo, 0.01, &steel()), Some(1));
        assert!(!set_body_material(&mut w, 99, 1));
        assert!(!set_body_material(&mut w, pulled, 65_536));
        assert_eq!(w.body_materials[pulled], 1);
        assert!(set_body_material(&mut w, pulled, 65_535));
        assert_eq!(w.body_materials[pulled], 65_535);
    }

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

    fn free_world() -> PhysicsWorld {
        // no gravity and no damping (a rate of 0 per second), so the closed forms below hold
        let mut w = PhysicsWorld::new(PhysicsConfig {
            gravity: Vec3Fix::ZERO,
            damping_rate: Fix128::ZERO,
            ..PhysicsConfig::default()
        });
        w.add_body(RigidBody::new_dynamic(vf(0.0, 0.0, 0.0), Fix128::ONE));
        w
    }

    // oracle: a quaternion is normalised (0, 0, 2, 0) -> (0, 0, 1, 0); the zero
    // quaternion, a non-finite component and an unknown body are refused
    #[test]
    fn a_quaternion_of_any_size_gives_the_same_rotation() {
        // oracle: a rotation does not depend on the scale, so s (0, 0, 0, 1) and s (1, 0, 0, 0)
        // give those unit quaternions and s (1, 1, 0, 0) gives (1, 1, 0, 0) / sqrt 2, at sizes
        // whose squared length overflows `Fix128` or rounds to zero in it
        let close = |got: f64, want: f64| (got - want).abs() < 1e-15;
        let h = core::f64::consts::FRAC_1_SQRT_2;
        for s in [
            1e18,
            3.1e9,
            1e10,
            1e-10,
            1e-12,
            1e300,
            1e-300,
            5e-324,
            f64::MAX,
        ] {
            for (input, want) in [
                ((s, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
                ((-s, 0.0, 0.0, 0.0), (-1.0, 0.0, 0.0, 0.0)),
                ((0.0, 0.0, 0.0, s), (0.0, 0.0, 0.0, 1.0)),
                ((s, s, 0.0, 0.0), (h, h, 0.0, 0.0)),
            ] {
                let q = unit_quat(input.0, input.1, input.2, input.3).expect("not zero");
                assert!(
                    close(q.x.to_f64(), want.0)
                        && close(q.y.to_f64(), want.1)
                        && close(q.z.to_f64(), want.2)
                        && close(q.w.to_f64(), want.3),
                    "{input:?}: {q:?}"
                );
            }
        }
        // sizes that stay as they are give the bits of normalising the value itself
        for s in [UNIT_QUAT_SMALLEST, 1.0, 0.25, UNIT_QUAT_LARGEST / 2.0] {
            let direct = QuatFix::new(
                Fix128::from_f64(s),
                Fix128::from_f64(2.0 * s),
                Fix128::from_f64(0.5 * s),
                Fix128::from_f64(0.25 * s),
            )
            .normalize();
            assert_eq!(
                unit_quat(s, 2.0 * s, 0.5 * s, 0.25 * s),
                Some(direct),
                "{s}"
            );
        }
        for zero in [0.0, -0.0] {
            assert_eq!(unit_quat(zero, zero, zero, zero), None);
        }
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(unit_quat(bad, 0.0, 0.0, 1.0), None);
            assert_eq!(unit_quat(0.0, bad, 0.0, 1.0), None);
            assert_eq!(unit_quat(0.0, 0.0, bad, 1.0), None);
            assert_eq!(unit_quat(0.0, 0.0, 0.0, bad), None);
        }
    }

    /// The unit quaternion by the closed form, scaled first so that no square overflows in f64
    fn closed_form(q: [f64; 4]) -> [f64; 4] {
        let largest = q
            .iter()
            .fold(0.0_f64, |m, c| crate::det_math::max64(m, c.abs()));
        let s = q.map(|c| c / largest);
        let length = (s[0] * s[0] + s[1] * s[1] + s[2] * s[2] + s[3] * s[3]).sqrt();
        s.map(|c| c / length)
    }

    /// How far a component may be from the closed form: rounding in f64 and, inside the sizes
    /// that are not scaled, the unit u = 2^-64 of `Fix128`. Four products in the squared length are
    /// off by at most 4u, so the length is off by at most 2u / length, relative 2u / length^2, and
    /// `length >= largest`. The inverse of the length is off by u, and multiplies `largest`. The
    /// square root and the division add a unit each, so the bound is doubled.
    fn close_enough(q: [f64; 4]) -> f64 {
        let largest = q
            .iter()
            .fold(0.0_f64, |m, c| crate::det_math::max64(m, c.abs()));
        let unit = 5.421_010_862_427_522e-20; // 2^-64
        let kept_as_it_is = if (UNIT_QUAT_SMALLEST..=UNIT_QUAT_LARGEST).contains(&largest) {
            4.0 * unit / (largest * largest) + 4.0 * unit * largest
        } else {
            0.0
        };
        1e-14 + kept_as_it_is
    }

    fn assert_is_the_closed_form(q: [f64; 4]) {
        let got = unit_quat(q[0], q[1], q[2], q[3]).unwrap_or_else(|| panic!("refused {q:?}"));
        let got = [
            got.x.to_f64(),
            got.y.to_f64(),
            got.z.to_f64(),
            got.w.to_f64(),
        ];
        let want = closed_form(q);
        let tolerance = close_enough(q);
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() <= tolerance, "{q:?}: {got:?}, not {want:?}");
        }
    }

    // oracle: q / |q|. These inputs came back wrong with success, or were refused, when the squared
    // length wrapped in `Fix128`: components past about 3e9
    #[test]
    fn quaternions_whose_squares_wrap_give_the_closed_form() {
        for q in [
            [8.3e15, -1.5e16, 3.1e15, -2.2e17],
            [-1.9e17, -1.4e17, -1.7e17, 1.07e18],
            [2.3e17, -4.1e14, 1.6e17, -7e15],
            [3.1e9, 0.0, 0.0, 0.0],
            [4.3e9, 0.0, 0.0, 0.0],
            [1e10, 1e10, 0.0, 0.0],
            [1e10, 0.0, 0.0, 0.0],
            [1e18, 0.0, 0.0, 0.0],
            [1e-10, 0.0, 0.0, 0.0],
            [1e-12, 1e-12, 0.0, 1e-12],
        ] {
            assert_is_the_closed_form(q);
        }
    }

    // the same oracle over the sizes the first report scanned, 1e-14 to 3e18, from a fixed sequence
    #[test]
    fn every_size_gives_the_closed_form() {
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut unit = move || f64::from(u32::try_from(next() >> 32).unwrap()) / 4_294_967_296.0;
        let mut compared = 0;
        for _ in 0..100_000 {
            let exponent = unit() * 32.5 - 14.0;
            let size = crate::det_math::exp64(exponent * core::f64::consts::LN_10);
            let q = [
                (unit() - 0.5) * size,
                (unit() - 0.5) * size,
                (unit() - 0.5) * size,
                (unit() - 0.5) * size,
            ];
            assert_is_the_closed_form(q);
            compared += 1;
        }
        assert_eq!(compared, 100_000);
    }

    // the sizes on each side of where scaling starts both give the closed form, and the sizes kept
    // as they are give the bits of normalising the value itself
    #[test]
    fn the_edges_of_the_sizes_not_scaled() {
        for largest in [
            UNIT_QUAT_SMALLEST * 0.99,
            UNIT_QUAT_SMALLEST,
            UNIT_QUAT_LARGEST,
            UNIT_QUAT_LARGEST * 1.01,
        ] {
            assert_is_the_closed_form([largest, -largest, largest, largest]);
        }
        // written out, not from the constants, so that moving a limit shows: 2^30 and 2^-16 give
        // the bits of normalising the value itself, which scaling would not
        for largest in [1_073_741_824.0, 1.0 / 65_536.0] {
            let q = [largest, -0.3 * largest, 0.7 * largest, 0.1 * largest];
            let direct = QuatFix::new(
                Fix128::from_f64(q[0]),
                Fix128::from_f64(q[1]),
                Fix128::from_f64(q[2]),
                Fix128::from_f64(q[3]),
            )
            .normalize();
            let got = unit_quat(q[0], q[1], q[2], q[3]);
            assert_eq!(got, Some(direct), "{largest}");
            let scaled = QuatFix::new(
                Fix128::from_f64(1.0),
                Fix128::from_f64(-0.3),
                Fix128::from_f64(0.7),
                Fix128::from_f64(0.1),
            )
            .normalize();
            assert_ne!(
                got,
                Some(scaled),
                "{largest}: the row does not tell the two apart"
            );
        }
    }

    #[test]
    fn set_rotation_normalises_and_refuses_what_is_not_a_rotation() {
        let mut w = free_world();
        assert!(set_rotation(&mut w, 0, 0.0, 0.0, 2.0, 0.0));
        let q = w.bodies[0].rotation;
        assert!(
            (q.z.to_f64() - 1.0).abs() < 1e-15 && q.w.to_f64().abs() < 1e-15,
            "{q:?}"
        );
        assert!(!set_rotation(&mut w, 0, 0.0, 0.0, 0.0, 0.0));
        assert!(!set_rotation(&mut w, 0, f64::NAN, 0.0, 0.0, 1.0));
        assert!(!set_rotation(&mut w, 1, 0.0, 0.0, 0.0, 1.0));
        assert!(!set_angular_velocity(&mut w, 1, 0.0, 0.0, 1.0));
        assert!(!set_angular_velocity(&mut w, 0, f64::INFINITY, 0.0, 0.0));
        assert_eq!(w.bodies[0].rotation, q, "a refused call changed the body");
    }

    // oracle: setting the orientation is not a turn; with no angular velocity the
    // body keeps the orientation it was given over a step (prev_rotation is set too)
    #[test]
    fn a_set_rotation_does_not_read_as_an_angular_velocity() {
        let mut w = free_world();
        let h = core::f64::consts::FRAC_1_SQRT_2;
        assert!(set_rotation(&mut w, 0, 0.0, h, 0.0, h)); // a quarter turn about y
        let q = w.bodies[0].rotation;
        // the engine's own setter contract: the previous orientation is the new one too
        assert_eq!(w.bodies[0].prev_rotation, q);
        w.step(Fix128::from_ratio(1, 60));
        let b = &w.bodies[0];
        assert_eq!(b.angular_velocity, Vec3Fix::ZERO);
        assert!(
            (b.rotation.y.to_f64() - q.y.to_f64()).abs() < 1e-12,
            "{:?} vs {q:?}",
            b.rotation
        );
        assert!((b.rotation.w.to_f64() - q.w.to_f64()).abs() < 1e-12);
    }

    // oracle (closed form): a free body spinning at pi rad/s about z for 1 s turns
    // by pi, q = (0, 0, sin(pi/2), cos(pi/2)) = (0, 0, 1, 0), and keeps its spin
    #[test]
    fn an_angular_velocity_turns_a_free_body_by_omega_t() {
        let mut w = free_world();
        assert!(set_angular_velocity(
            &mut w,
            0,
            0.0,
            0.0,
            core::f64::consts::PI
        ));
        for _ in 0..60 {
            w.step(Fix128::from_ratio(1, 60));
        }
        let b = &w.bodies[0];
        let q = b.rotation;
        // a half turn: |qz| = sin(pi/2) = 1 and qw = cos(pi/2) = 0, compared in the
        // quaternion (no platform transcendental function in a test either)
        assert!(
            (q.z.to_f64().abs() - 1.0).abs() < 1e-6 && q.w.to_f64().abs() < 1e-3,
            "{q:?}"
        );
        assert!(q.x.to_f64().abs() < 1e-9 && q.y.to_f64().abs() < 1e-9);
        assert!((b.angular_velocity.z.to_f64() - core::f64::consts::PI).abs() < 1e-6);
    }
}
