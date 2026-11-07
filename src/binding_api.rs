//! Argument checking shared by the C ABI, WebAssembly and Python bindings.
//!
//! The bindings take plain `f64` / `u32` arguments from a host that cannot
//! see Rust's types. Every check lives here once, so the three bindings agree
//! on what they accept: a non-finite or non-positive size, an out-of-range
//! body id or index, a joint between a body and itself, or a malformed mesh
//! is refused (`None` / `false`) instead of reaching a `PhysicsWorld` method
//! that would panic (`add_joint`) or quietly store an unusable value.

#[cfg(feature = "std")]
use crate::buoyancy_zone::ZoneShape;
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
#[cfg(feature = "std")]
use crate::wind_zone::{WindZone, WindZoneParticipant};
use crate::world_shape_query::WorldShapeHit;

/// Shape kind codes used by every binding (`AlicePhysicsShape::kind` in C).
pub(crate) const SHAPE_BOX: u32 = 0;
pub(crate) const SHAPE_CYLINDER: u32 = 1;
pub(crate) const SHAPE_CONE: u32 = 2;
pub(crate) const SHAPE_ELLIPSOID: u32 = 3;
pub(crate) const SHAPE_WEDGE: u32 = 4;
pub(crate) const SHAPE_TORUS: u32 = 5;

/// `x` as a [`Fix128`], or `None` when it is not one: not a number, or outside
/// `-2^63 <= x < 2^63`, the range of the integer part. Past that range
/// `Fix128::from_f64` has no right answer to give, so a binding refuses the value
/// instead of working with another one.
pub(crate) fn finite(x: f64) -> Option<Fix128> {
    const LIMIT: f64 = 9_223_372_036_854_775_808.0;
    (-LIMIT..LIMIT).contains(&x).then(|| Fix128::from_f64(x))
}

/// [`finite`], and greater than zero.
pub(crate) fn positive(x: f64) -> Option<Fix128> {
    finite(x).filter(|_| x > 0.0)
}

/// [`finite`], and not negative.
pub(crate) fn non_negative(x: f64) -> Option<Fix128> {
    finite(x).filter(|_| x >= 0.0)
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

/// The numbers of a box-shaped wind zone, in the units of [`WindZone`].
#[cfg(feature = "std")]
pub(crate) struct WindZoneArgs {
    pub(crate) min: Vec3Fix,
    pub(crate) max: Vec3Fix,
    pub(crate) direction: Vec3Fix,
    pub(crate) base_speed: f64,
    pub(crate) gust_amplitude: f64,
    pub(crate) gust_hz: f64,
    pub(crate) air_density: f64,
    pub(crate) drag_coefficient: f64,
    pub(crate) area: f64,
}

/// Register a wind zone as a resident of the world and return its participant
/// index. The box must not be inside out and the direction is normalised and
/// must not be zero; the speeds and the frequency must not be negative, and
/// the density, drag coefficient and area must be greater than zero.
#[cfg(feature = "std")]
pub(crate) fn add_wind_zone(w: &mut PhysicsWorld, a: &WindZoneArgs) -> Option<usize> {
    if a.min.x > a.max.x || a.min.y > a.max.y || a.min.z > a.max.z {
        return None;
    }
    let zone = WindZone {
        shape: ZoneShape::Aabb {
            min: a.min,
            max: a.max,
        },
        // positive after the conversion too (1e-300 would become a fixed-point zero)
        air_density_kg_m3: positive(a.air_density).filter(|v| *v > Fix128::ZERO)?,
        direction: a.direction.try_normalize()?,
        base_speed_m_s: non_negative(a.base_speed)?,
        turbulence_amplitude: non_negative(a.gust_amplitude)?,
        gust_frequency_hz: non_negative(a.gust_hz)?,
        drag_coefficient: positive(a.drag_coefficient).filter(|v| *v > Fix128::ZERO)?,
        reference_area_m2: positive(a.area).filter(|v| *v > Fix128::ZERO)?,
    };
    w.add_participant(Box::new(WindZoneParticipant::new(zone)))
        .ok()
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

    /// oracle: the range is the integer part's, both ends exact; inside it the value
    /// is `from_f64`'s, bit for bit, so a world built from in-range values is unchanged.
    #[test]
    fn a_value_is_taken_inside_the_range_of_fix128_and_refused_outside() {
        const TWO_63: f64 = 9_223_372_036_854_775_808.0;
        let below = f64::from_bits(TWO_63.to_bits() - 1);
        let beyond = f64::from_bits((-TWO_63).to_bits() + 1);
        for x in [TWO_63, 1e19, 1e30, f64::MAX, f64::INFINITY, f64::NAN] {
            assert_eq!(finite(x), None, "{x}");
            assert_eq!(positive(x), None, "{x}");
            assert_eq!(non_negative(x), None, "{x}");
        }
        for x in [beyond, -1e19, -1e30, f64::MIN, f64::NEG_INFINITY] {
            assert_eq!(finite(x), None, "{x}");
        }
        for x in [below, -below, -TWO_63, 0.0, 0.1, -0.1, 1e-300, 1234.5] {
            assert_eq!(finite(x), Some(Fix128::from_f64(x)), "{x}");
        }
        assert_eq!(positive(below), Some(Fix128::from_f64(below)));
        assert_eq!(non_negative(below), Some(Fix128::from_f64(below)));
        assert_eq!(positive(0.0), None);
        assert_eq!(positive(-1.0), None);
        assert_eq!(non_negative(0.0), Some(Fix128::ZERO));
        assert_eq!(non_negative(-1.0), None);
        assert_eq!(vec3(0.0, 1e30, 0.0), None);
    }

    /// oracle: every entry that takes a float refuses one out of range and leaves
    /// the world as it was.
    #[test]
    fn every_float_entry_refuses_a_value_out_of_range() {
        let huge = 1e30;
        let mut w = PhysicsWorld::new(PhysicsConfig::default());
        let body = w.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        let before = w.snapshot_world();
        assert!(shape(SHAPE_BOX, huge, 1.0, 1.0).is_none());
        assert!(!set_collision_radius(&mut w, body, huge));
        assert_eq!(register_material(&mut w, huge, 0.0), None);
        assert_eq!(register_material(&mut w, 0.0, huge), None);
        let mut m = steel();
        m.density = huge;
        assert_eq!(
            add_magnet(&mut w, Vec3Fix::ZERO, Vec3Fix::ZERO, 1.0, &m),
            None
        );
        let m = steel();
        assert_eq!(
            add_magnet(&mut w, Vec3Fix::ZERO, Vec3Fix::ZERO, huge, &m),
            None
        );
        let unit_box = shape(SHAPE_BOX, 1.0, 1.0, 1.0).expect("shape");
        assert_eq!(add_shaped_body(&mut w, unit_box, huge, Vec3Fix::ZERO), None);
        let up = Vec3Fix::new(Fix128::ZERO, Fix128::ONE, Fix128::ZERO);
        assert_eq!(add_static_plane(&mut w, up, huge), None);
        let flat = [0.0; 4];
        assert_eq!(
            add_static_heightfield(&mut w, &flat, 2, 2, huge, Vec3Fix::ZERO),
            None
        );
        let tall = [0.0, 0.0, 0.0, huge];
        assert_eq!(
            add_static_heightfield(&mut w, &tall, 2, 2, 1.0, Vec3Fix::ZERO),
            None
        );
        let far = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, huge, 0.0];
        assert_eq!(add_static_trimesh(&mut w, &far, &[0, 1, 2]), None);
        assert_eq!(w.snapshot_world(), before);
    }

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
}
