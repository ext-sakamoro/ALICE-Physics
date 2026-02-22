//! SDF (Signed Distance Field) Collision Detection
//!
//! Bypass GJK/EPA entirely — sample the distance field directly.
//!
//! # How It Works
//!
//! Traditional collision: GJK → EPA → penetration depth + normal
//! SDF collision: distance(point) → penetration depth, gradient(point) → normal
//!
//! One distance query replaces the entire GJK+EPA pipeline.
//! Works with concave shapes, fractals, CSG — anything expressible as an SDF.
//!
//! # Integration with ALICE-SDF
//!
//! ```ignore
//! use alice_sdf::CompiledSdf;
//! use alice_physics::sdf_collider::{SdfCollider, SdfField};
//!
//! // CompiledSdf implements SdfField via the physics_bridge feature
//! let sdf = CompiledSdf::compile(&node);
//! let collider = SdfCollider::new(Box::new(bridge), position, rotation);
//! world.add_sdf_collider(collider);
//! ```
//!
//! Author: Moroya Sakamoto

use crate::collider::Contact;
use crate::math::{Fix128, QuatFix, Vec3Fix};

#[cfg(not(feature = "std"))]
use alloc::boxed::Box;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ============================================================================
// SDF Field Trait
// ============================================================================

/// Trait for evaluating a signed distance field.
///
/// All methods use f32 because SDF evaluation is inherently floating-point.
/// The Fix128 ↔ f32 conversion happens at the collider boundary.
pub trait SdfField: Send + Sync {
    /// Returns the signed distance from point to the nearest surface.
    ///
    /// - Positive: outside the shape
    /// - Zero: on the surface
    /// - Negative: inside the shape
    fn distance(&self, x: f32, y: f32, z: f32) -> f32;

    /// Returns the surface normal (normalized gradient) at the given point.
    ///
    /// Points away from the surface (outward direction).
    fn normal(&self, x: f32, y: f32, z: f32) -> (f32, f32, f32);

    /// Combined distance + normal query (default: two separate calls).
    ///
    /// Override for implementations that can compute both efficiently
    /// (e.g., ALICE-SDF's `eval_distance_and_gradient_simd`).
    fn distance_and_normal(&self, x: f32, y: f32, z: f32) -> (f32, (f32, f32, f32)) {
        (self.distance(x, y, z), self.normal(x, y, z))
    }
}

/// Type alias for a normal-returning closure used in [`ClosureSdf`]
type NormalFn = Box<dyn Fn(f32, f32, f32) -> (f32, f32, f32) + Send + Sync>;

/// Closure-based SDF field implementation.
///
/// Allows connecting any SDF evaluation function without trait implementation.
///
/// ```
/// use alice_physics::sdf_collider::ClosureSdf;
///
/// let sdf = ClosureSdf::new(
///     |x, y, z| ((x*x + y*y + z*z).sqrt() - 1.0), // sphere r=1
///     |x, y, z| {
///         let len = (x*x + y*y + z*z).sqrt();
///         (x / len, y / len, z / len)
///     },
/// );
/// ```
pub struct ClosureSdf {
    eval_fn: Box<dyn Fn(f32, f32, f32) -> f32 + Send + Sync>,
    normal_fn: NormalFn,
}

impl ClosureSdf {
    /// Create from evaluation and normal closures
    pub fn new(
        eval_fn: impl Fn(f32, f32, f32) -> f32 + Send + Sync + 'static,
        normal_fn: impl Fn(f32, f32, f32) -> (f32, f32, f32) + Send + Sync + 'static,
    ) -> Self {
        Self {
            eval_fn: Box::new(eval_fn),
            normal_fn: Box::new(normal_fn),
        }
    }
}

impl SdfField for ClosureSdf {
    #[inline]
    fn distance(&self, x: f32, y: f32, z: f32) -> f32 {
        (self.eval_fn)(x, y, z)
    }

    #[inline]
    fn normal(&self, x: f32, y: f32, z: f32) -> (f32, f32, f32) {
        (self.normal_fn)(x, y, z)
    }
}

// ============================================================================
// SDF Collider (world-space placement)
// ============================================================================

/// An SDF field placed in the physics world with a transform.
///
/// The SDF is evaluated in local space; the collider transforms
/// world-space query points into local space before evaluation.
///
/// Cached fields (`inv_rotation`, `scale_f32`, `inv_scale_f32`) avoid
/// recomputing invariants on every collision query.
pub struct SdfCollider {
    /// The underlying distance field
    pub field: Box<dyn SdfField>,
    /// World-space position of the SDF origin
    pub position: Vec3Fix,
    /// World-space orientation of the SDF
    pub rotation: QuatFix,
    /// Uniform scale factor
    pub scale: Fix128,
    /// Body index this SDF is attached to (usize::MAX = static world geometry)
    pub body_index: usize,
    // -- Cached invariants (derived from rotation/scale) --
    /// Inverse rotation (cached)
    inv_rotation: QuatFix,
    /// Scale as f32 (cached)
    pub(crate) scale_f32: f32,
    /// Inverse scale as f32 (cached, guards against zero)
    inv_scale_f32: f32,
}

/// Sentinel value for static (world-fixed) SDF colliders
pub const SDF_STATIC: usize = usize::MAX;

impl SdfCollider {
    /// Create a new SDF collider attached to the world (static).
    pub fn new_static(field: Box<dyn SdfField>, position: Vec3Fix, rotation: QuatFix) -> Self {
        Self {
            field,
            position,
            inv_rotation: rotation.conjugate(),
            rotation,
            scale: Fix128::ONE,
            body_index: SDF_STATIC,
            scale_f32: 1.0,
            inv_scale_f32: 1.0,
        }
    }

    /// Create a new SDF collider attached to a rigid body.
    pub fn new_dynamic(field: Box<dyn SdfField>, body_index: usize) -> Self {
        Self {
            field,
            position: Vec3Fix::ZERO,
            rotation: QuatFix::IDENTITY,
            inv_rotation: QuatFix::IDENTITY,
            scale: Fix128::ONE,
            body_index,
            scale_f32: 1.0,
            inv_scale_f32: 1.0,
        }
    }

    /// Set uniform scale
    pub fn with_scale(mut self, scale: Fix128) -> Self {
        self.scale = scale;
        let s = scale.to_f32();
        self.scale_f32 = s;
        self.inv_scale_f32 = if s.abs() < 1e-10 { 1.0 } else { 1.0 / s };
        self
    }

    /// Recompute cached fields after changing position/rotation externally.
    pub fn update_cache(&mut self) {
        self.inv_rotation = self.rotation.conjugate();
        let s = self.scale.to_f32();
        self.scale_f32 = s;
        self.inv_scale_f32 = if s.abs() < 1e-10 { 1.0 } else { 1.0 / s };
    }

    /// Transform world-space point to SDF local space (f32).
    ///
    /// Uses cached `inv_rotation` and `inv_scale_f32` to avoid recomputation.
    #[cfg(feature = "std")]
    #[inline]
    pub(crate) fn world_to_local(&self, world_point: Vec3Fix) -> (f32, f32, f32) {
        let relative = world_point - self.position;
        let local = self.inv_rotation.rotate_vec(relative);
        let (lx, ly, lz) = local.to_f32();
        let inv_s = self.inv_scale_f32;
        (lx * inv_s, ly * inv_s, lz * inv_s)
    }

    /// Transform local-space normal to world space (Fix128).
    #[cfg(feature = "std")]
    #[inline]
    pub(crate) fn local_normal_to_world(&self, nx: f32, ny: f32, nz: f32) -> Vec3Fix {
        let local_n = Vec3Fix::from_f32(nx, ny, nz);
        self.rotation.rotate_vec(local_n).normalize()
    }
}

// ============================================================================
// Collision Detection Functions
// ============================================================================

/// Collide a single point against an SDF.
///
/// Returns a contact if the point is inside the SDF (distance < 0).
/// - `depth` = penetration depth (positive)
/// - `normal` = direction to push the point out
/// - `point_a` = the query point itself
/// - `point_b` = nearest surface point
///
/// Optimization: evaluates distance first (1 eval), then normal only on hit (4 evals).
#[cfg(feature = "std")]
pub fn collide_point_sdf(point: Vec3Fix, sdf: &SdfCollider) -> Option<Contact> {
    let (lx, ly, lz) = sdf.world_to_local(point);

    // Early-out: distance only (1 eval instead of 5)
    let dist = sdf.field.distance(lx, ly, lz);
    let scale_f32 = sdf.scale_f32;
    let world_dist = dist * scale_f32;

    if world_dist >= 0.0 {
        return None; // Outside or on surface
    }

    // Only compute normal for penetrating points (4 additional evals)
    let (nx, ny, nz) = sdf.field.normal(lx, ly, lz);

    let depth = Fix128::from_f32(-world_dist);
    let normal = sdf.local_normal_to_world(nx, ny, nz);
    let surface_point = point + normal * depth;

    Some(Contact {
        depth,
        normal,
        point_a: point,
        point_b: surface_point,
    })
}

/// Collide a sphere against an SDF.
///
/// Evaluates distance at sphere center, then adjusts by radius.
///
/// Optimization: evaluates distance first (1 eval), then normal only on hit (4 evals).
/// Most bodies are NOT colliding, so this saves 4 evals per non-colliding body.
#[cfg(feature = "std")]
pub fn collide_sphere_sdf(center: Vec3Fix, radius: Fix128, sdf: &SdfCollider) -> Option<Contact> {
    let (lx, ly, lz) = sdf.world_to_local(center);

    // Early-out: distance only (1 eval)
    let dist = sdf.field.distance(lx, ly, lz);
    let scale_f32 = sdf.scale_f32;
    let world_dist = dist * scale_f32;
    let radius_f32 = radius.to_f32();

    // Penetration = radius - distance_to_surface
    let penetration = radius_f32 - world_dist;

    if penetration <= 0.0 {
        return None; // Sphere doesn't reach the surface
    }

    // Only compute normal for penetrating spheres (4 additional evals)
    let (nx, ny, nz) = sdf.field.normal(lx, ly, lz);

    let depth = Fix128::from_f32(penetration);
    let normal = sdf.local_normal_to_world(nx, ny, nz);

    // Contact point on sphere surface (toward SDF)
    let point_a = center - normal * radius;
    // Contact point on SDF surface
    let point_b = center - normal * Fix128::from_f32(world_dist);

    Some(Contact {
        depth,
        normal,
        point_a,
        point_b,
    })
}

/// Collide a capsule against an SDF.
///
/// Samples 3 points along the capsule axis (endpoints + midpoint),
/// returns the deepest penetrating contact.
#[cfg(feature = "std")]
pub fn collide_capsule_sdf(
    a: Vec3Fix,
    b: Vec3Fix,
    radius: Fix128,
    sdf: &SdfCollider,
) -> Option<Contact> {
    let mid = Vec3Fix::new((a.x + b.x).half(), (a.y + b.y).half(), (a.z + b.z).half());

    let samples = [a, mid, b];
    let mut best: Option<Contact> = None;

    for &point in &samples {
        if let Some(contact) = collide_sphere_sdf(point, radius, sdf) {
            match &best {
                Some(prev) if prev.depth >= contact.depth => {}
                _ => best = Some(contact),
            }
        }
    }

    best
}

/// Collide an AABB against an SDF.
///
/// Samples 8 corner vertices + center (9 points total),
/// returns the deepest penetrating contact.
#[cfg(feature = "std")]
pub fn collide_aabb_sdf(min: Vec3Fix, max: Vec3Fix, sdf: &SdfCollider) -> Option<Contact> {
    let center = Vec3Fix::new(
        (min.x + max.x).half(),
        (min.y + max.y).half(),
        (min.z + max.z).half(),
    );

    // 8 corners + center
    let corners = [
        Vec3Fix::new(min.x, min.y, min.z),
        Vec3Fix::new(max.x, min.y, min.z),
        Vec3Fix::new(min.x, max.y, min.z),
        Vec3Fix::new(max.x, max.y, min.z),
        Vec3Fix::new(min.x, min.y, max.z),
        Vec3Fix::new(max.x, min.y, max.z),
        Vec3Fix::new(min.x, max.y, max.z),
        Vec3Fix::new(max.x, max.y, max.z),
        center,
    ];

    let mut best: Option<Contact> = None;

    for &point in &corners {
        if let Some(contact) = collide_point_sdf(point, sdf) {
            match &best {
                Some(prev) if prev.depth >= contact.depth => {}
                _ => best = Some(contact),
            }
        }
    }

    best
}

/// Detect all SDF collisions for a set of bodies.
///
/// For each body, tests against all SDF colliders and returns contacts.
/// Bodies with `inv_mass == 0` (static) are skipped.
#[cfg(feature = "std")]
pub fn detect_sdf_contacts(
    bodies: &[crate::solver::RigidBody],
    sdf_colliders: &[SdfCollider],
    collision_radius: Fix128,
) -> Vec<(usize, Contact)> {
    let mut contacts = Vec::new();

    for (body_idx, body) in bodies.iter().enumerate() {
        if body.is_static() {
            continue;
        }

        for sdf in sdf_colliders {
            // Skip self-collision
            if sdf.body_index == body_idx {
                continue;
            }

            // Treat each body as a sphere with the given collision radius
            if let Some(contact) = collide_sphere_sdf(body.position, collision_radius, sdf) {
                contacts.push((body_idx, contact));
            }
        }
    }

    contacts
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple unit sphere SDF for testing
    fn unit_sphere() -> ClosureSdf {
        ClosureSdf::new(
            |x, y, z| (x * x + y * y + z * z).sqrt() - 1.0,
            |x, y, z| {
                let len = (x * x + y * y + z * z).sqrt();
                if len < 1e-10 {
                    (0.0, 1.0, 0.0) // Degenerate: point normal upward
                } else {
                    (x / len, y / len, z / len)
                }
            },
        )
    }

    /// Infinite ground plane at y=0 (negative below)
    fn ground_plane() -> ClosureSdf {
        ClosureSdf::new(|_x, y, _z| y, |_x, _y, _z| (0.0, 1.0, 0.0))
    }

    #[test]
    fn test_point_outside_sphere() {
        let sdf =
            SdfCollider::new_static(Box::new(unit_sphere()), Vec3Fix::ZERO, QuatFix::IDENTITY);

        // Point at (2, 0, 0) — outside sphere of radius 1
        let point = Vec3Fix::from_f32(2.0, 0.0, 0.0);
        let result = collide_point_sdf(point, &sdf);
        assert!(result.is_none(), "Point outside sphere should not collide");
    }

    #[test]
    fn test_point_inside_sphere() {
        let sdf =
            SdfCollider::new_static(Box::new(unit_sphere()), Vec3Fix::ZERO, QuatFix::IDENTITY);

        // Point at (0.5, 0, 0) — inside sphere of radius 1
        let point = Vec3Fix::from_f32(0.5, 0.0, 0.0);
        let result = collide_point_sdf(point, &sdf);
        assert!(result.is_some(), "Point inside sphere should collide");

        let contact = result.unwrap();
        // Penetration depth should be ~0.5
        let depth = contact.depth.to_f32();
        assert!(
            (depth - 0.5).abs() < 0.05,
            "Depth should be ~0.5, got {}",
            depth
        );

        // Normal should point in +X
        let (nx, _, _) = contact.normal.to_f32();
        assert!(nx > 0.9, "Normal should point in +X, got {}", nx);
    }

    #[test]
    fn test_sphere_vs_ground() {
        let sdf =
            SdfCollider::new_static(Box::new(ground_plane()), Vec3Fix::ZERO, QuatFix::IDENTITY);

        // Sphere at y=0.5 with radius 1.0 — penetrates ground by 0.5
        let center = Vec3Fix::from_f32(0.0, 0.5, 0.0);
        let radius = Fix128::ONE;
        let result = collide_sphere_sdf(center, radius, &sdf);
        assert!(result.is_some(), "Sphere should penetrate ground");

        let contact = result.unwrap();
        let depth = contact.depth.to_f32();
        assert!(
            (depth - 0.5).abs() < 0.05,
            "Penetration should be ~0.5, got {}",
            depth
        );
    }

    #[test]
    fn test_sphere_above_ground() {
        let sdf =
            SdfCollider::new_static(Box::new(ground_plane()), Vec3Fix::ZERO, QuatFix::IDENTITY);

        // Sphere at y=2.0 with radius 1.0 — floating above ground
        let center = Vec3Fix::from_f32(0.0, 2.0, 0.0);
        let radius = Fix128::ONE;
        let result = collide_sphere_sdf(center, radius, &sdf);
        assert!(result.is_none(), "Sphere above ground should not collide");
    }

    #[test]
    fn test_capsule_vs_ground() {
        let sdf =
            SdfCollider::new_static(Box::new(ground_plane()), Vec3Fix::ZERO, QuatFix::IDENTITY);

        // Horizontal capsule at y=0.3, radius=0.5 — penetrates
        let a = Vec3Fix::from_f32(-1.0, 0.3, 0.0);
        let b = Vec3Fix::from_f32(1.0, 0.3, 0.0);
        let radius = Fix128::from_f32(0.5);
        let result = collide_capsule_sdf(a, b, radius, &sdf);
        assert!(result.is_some(), "Capsule should penetrate ground");

        let contact = result.unwrap();
        let depth = contact.depth.to_f32();
        assert!(
            (depth - 0.2).abs() < 0.05,
            "Penetration should be ~0.2, got {}",
            depth
        );
    }

    #[test]
    fn test_translated_sdf() {
        // Sphere SDF centered at (5, 0, 0)
        let sdf = SdfCollider::new_static(
            Box::new(unit_sphere()),
            Vec3Fix::from_f32(5.0, 0.0, 0.0),
            QuatFix::IDENTITY,
        );

        // Point at (5.5, 0, 0) — inside the translated sphere
        let point = Vec3Fix::from_f32(5.5, 0.0, 0.0);
        let result = collide_point_sdf(point, &sdf);
        assert!(
            result.is_some(),
            "Point inside translated sphere should collide"
        );

        // Point at (0, 0, 0) — far outside
        let origin = Vec3Fix::ZERO;
        let result2 = collide_point_sdf(origin, &sdf);
        assert!(
            result2.is_none(),
            "Origin should be outside translated sphere"
        );
    }

    #[test]
    fn test_scaled_sdf() {
        // Sphere scaled by 3x — effective radius 3
        let sdf =
            SdfCollider::new_static(Box::new(unit_sphere()), Vec3Fix::ZERO, QuatFix::IDENTITY)
                .with_scale(Fix128::from_int(3));

        // Point at (2, 0, 0) — inside scaled sphere (radius 3)
        let point = Vec3Fix::from_f32(2.0, 0.0, 0.0);
        let result = collide_point_sdf(point, &sdf);
        assert!(
            result.is_some(),
            "Point inside scaled sphere should collide"
        );

        // Point at (4, 0, 0) — outside scaled sphere
        let point_outside = Vec3Fix::from_f32(4.0, 0.0, 0.0);
        let result2 = collide_point_sdf(point_outside, &sdf);
        assert!(
            result2.is_none(),
            "Point outside scaled sphere should not collide"
        );
    }

    #[test]
    fn test_aabb_vs_ground() {
        let sdf =
            SdfCollider::new_static(Box::new(ground_plane()), Vec3Fix::ZERO, QuatFix::IDENTITY);

        // AABB with bottom at y=-0.5 — penetrates ground
        let min = Vec3Fix::from_f32(-1.0, -0.5, -1.0);
        let max = Vec3Fix::from_f32(1.0, 1.0, 1.0);
        let result = collide_aabb_sdf(min, max, &sdf);
        assert!(result.is_some(), "AABB below ground should collide");

        let contact = result.unwrap();
        let depth = contact.depth.to_f32();
        assert!(
            (depth - 0.5).abs() < 0.05,
            "Penetration should be ~0.5, got {}",
            depth
        );
    }
}
