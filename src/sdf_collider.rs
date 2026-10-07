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
//! // ignored: requires the external `alice_sdf` crate and the `physics_bridge`
//! // feature, which are not available in this crate's doc-test environment.
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

#[cfg(feature = "std")]
use crate::collider::Contact;
use crate::math::{Fix128, QuatFix, Vec3Fix};

#[cfg(not(feature = "std"))]
use alloc::boxed::Box;

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

/// A signed distance field queried by reference, with no thread-safety or
/// lifetime requirement.
///
/// [`SdfField`] requires `Send + Sync` (a collider owns its field as
/// `Box<dyn SdfField>` and the world may be shared across threads). Entry
/// points that only borrow a field for the length of one call, such as
/// [`sphere_trace_sdf_field`](crate::sdf_ccd::sphere_trace_sdf_field), take
/// this trait instead, so a field that borrows local data or holds an `Rc`
/// can be queried too. Every [`SdfField`] implements it (the methods forward
/// to [`SdfField::distance`] and [`SdfField::normal`]); a pair of borrowed
/// closures can be wrapped in [`ClosureSdfQuery`].
///
/// The methods are named apart from [`SdfField`]'s so that a type with both
/// traits in scope still resolves `field.distance(..)` unambiguously.
pub trait SdfQuery {
    /// Signed distance from the point to the nearest surface (positive
    /// outside), as [`SdfField::distance`].
    fn query_distance(&self, x: f32, y: f32, z: f32) -> f32;

    /// Outward surface normal at the point, as [`SdfField::normal`].
    fn query_normal(&self, x: f32, y: f32, z: f32) -> (f32, f32, f32);
}

impl<T: SdfField + ?Sized> SdfQuery for T {
    #[inline]
    fn query_distance(&self, x: f32, y: f32, z: f32) -> f32 {
        self.distance(x, y, z)
    }

    #[inline]
    fn query_normal(&self, x: f32, y: f32, z: f32) -> (f32, f32, f32) {
        self.normal(x, y, z)
    }
}

/// A distance closure and a normal closure as an [`SdfQuery`], unboxed.
///
/// Unlike [`ClosureSdf`] the closures need not be `'static`, `Send` or
/// `Sync`: they may borrow local data for the length of a query.
///
/// ```
/// use alice_physics::sdf_collider::{ClosureSdfQuery, SdfQuery};
///
/// let radius = 2.0_f32; // a local, borrowed by the closure
/// let sdf = ClosureSdfQuery::new(
///     |x: f32, y: f32, z: f32| (x * x + y * y + z * z).sqrt() - radius,
///     |x: f32, y: f32, z: f32| {
///         let l = (x * x + y * y + z * z).sqrt();
///         (x / l, y / l, z / l)
///     },
/// );
/// assert_eq!(sdf.query_distance(3.0, 0.0, 0.0), 1.0);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct ClosureSdfQuery<D, N> {
    distance_fn: D,
    normal_fn: N,
}

impl<D, N> ClosureSdfQuery<D, N>
where
    D: Fn(f32, f32, f32) -> f32,
    N: Fn(f32, f32, f32) -> (f32, f32, f32),
{
    /// Wrap a distance closure and a normal closure.
    #[must_use]
    pub fn new(distance_fn: D, normal_fn: N) -> Self {
        Self {
            distance_fn,
            normal_fn,
        }
    }
}

impl<D, N> SdfQuery for ClosureSdfQuery<D, N>
where
    D: Fn(f32, f32, f32) -> f32,
    N: Fn(f32, f32, f32) -> (f32, f32, f32),
{
    #[inline]
    fn query_distance(&self, x: f32, y: f32, z: f32) -> f32 {
        (self.distance_fn)(x, y, z)
    }

    #[inline]
    fn query_normal(&self, x: f32, y: f32, z: f32) -> (f32, f32, f32) {
        (self.normal_fn)(x, y, z)
    }
}

/// A distance closure alone as an [`SdfQuery`]; the normal is the crate's
/// central-difference gradient of that closure.
///
/// For a field with no analytic gradient (a height map, a blend of
/// primitives, anything sampled): the normal is the one
/// [`ModifiedSdf`](crate::sim_modifier::ModifiedSdf) and
/// [`DestructibleSdf`](crate::sdf_destruction::DestructibleSdf) compute for
/// their own fields, bit for bit. At `(x, y, z)` with step
/// `e = max(FD_NORMAL_BASE_EPS, 1e-4 * max(|x|, |y|, |z|))` it is
/// `(f(x+e)-f(x-e), f(y+e)-f(y-e), f(z+e)-f(z-e))` normalised (six
/// evaluations; the step scales with the coordinates so it keeps resolving
/// far from the origin). As with [`ClosureSdfQuery`] the closure need not be
/// `'static`, `Send` or `Sync`.
///
/// Degenerate input: where the six samples give a gradient of length below
/// `1e-10` (the centre of a sphere, a constant field) the normal is
/// `(0, 1, 0)`. A NaN or infinite distance among the samples (or a
/// non-finite query point) gives a NaN normal; the distance is returned as
/// the closure gives it.
///
/// Requires the `std` feature (the normal needs `f32::sqrt`).
///
/// ```
/// use alice_physics::sdf_collider::{DistanceSdfQuery, SdfQuery};
///
/// let radius = 2.0_f32; // a local, borrowed by the closure
/// let sdf = DistanceSdfQuery::new(|x: f32, y: f32, z: f32| {
///     (x * x + y * y + z * z).sqrt() - radius
/// });
/// assert_eq!(sdf.query_distance(3.0, 0.0, 0.0), 1.0);
/// let (nx, ny, nz) = sdf.query_normal(3.0, 0.0, 0.0);
/// assert!((nx - 1.0).abs() < 1e-3 && ny.abs() < 1e-3 && nz.abs() < 1e-3);
/// ```
#[cfg(feature = "std")]
#[derive(Clone, Copy, Debug)]
pub struct DistanceSdfQuery<D> {
    distance_fn: D,
}

#[cfg(feature = "std")]
impl<D> DistanceSdfQuery<D>
where
    D: Fn(f32, f32, f32) -> f32,
{
    /// Wrap a distance closure.
    #[must_use]
    pub fn new(distance_fn: D) -> Self {
        Self { distance_fn }
    }
}

#[cfg(feature = "std")]
impl<D> SdfQuery for DistanceSdfQuery<D>
where
    D: Fn(f32, f32, f32) -> f32,
{
    #[inline]
    fn query_distance(&self, x: f32, y: f32, z: f32) -> f32 {
        (self.distance_fn)(x, y, z)
    }

    #[inline]
    fn query_normal(&self, x: f32, y: f32, z: f32) -> (f32, f32, f32) {
        fd_normal(&self.distance_fn, FD_NORMAL_BASE_EPS, x, y, z)
    }
}

/// Union of two fields: `min(a, b)`, with the normal of whichever is nearer.
///
/// On a tie the first operand wins. The union of two exact distance fields
/// is exact outside both shapes and a lower bound inside; the union of two
/// `L`-Lipschitz fields is `L`-Lipschitz.
///
/// The common use is a surface made of several layers, such as ground and
/// a water level that can stand above it: a character colliding with the
/// union stands on whichever is higher.
#[derive(Debug, Clone)]
pub struct SdfUnion<A, B> {
    a: A,
    b: B,
}

impl<A: SdfField, B: SdfField> SdfUnion<A, B> {
    /// The union of `a` and `b`.
    #[must_use]
    pub fn new(a: A, b: B) -> Self {
        Self { a, b }
    }
}

impl<A: SdfField, B: SdfField> SdfField for SdfUnion<A, B> {
    #[inline]
    fn distance(&self, x: f32, y: f32, z: f32) -> f32 {
        self.a.distance(x, y, z).min(self.b.distance(x, y, z))
    }

    fn normal(&self, x: f32, y: f32, z: f32) -> (f32, f32, f32) {
        if self.a.distance(x, y, z) <= self.b.distance(x, y, z) {
            self.a.normal(x, y, z)
        } else {
            self.b.normal(x, y, z)
        }
    }
}

/// Central-difference step for an f32 SDF normal at `(x, y, z)`.
///
/// `base_eps` is used as is near the origin. An absolute step stops resolving
/// far from it: once `base_eps` falls below the f32 spacing of the
/// coordinates, `x + eps` rounds back to `x` and the difference is zero or
/// noise (AUD-A-S5W3-008, AUD-A-S5W3-013). The step therefore grows in
/// proportion to the largest coordinate magnitude, `1e-4 * max(|x|, |y|, |z|)`
/// (at least about 840 f32 ulps of that coordinate), whenever that exceeds
/// `base_eps`.
/// The same step is used on all three axes so the gradient direction is not
/// skewed.
#[inline]
#[must_use]
pub(crate) fn fd_normal_step(base_eps: f32, x: f32, y: f32, z: f32) -> f32 {
    let scale = x.abs().max(y.abs()).max(z.abs());
    base_eps.max(1.0e-4 * scale)
}

/// Base step of the central-difference SDF normal, used near the origin
/// (see [`DistanceSdfQuery`]; farther out the step grows with the
/// coordinate magnitude).
pub const FD_NORMAL_BASE_EPS: f32 = 1.0e-3;

/// Central-difference normal of the distance `f` at `(x, y, z)`: step
/// [`fd_normal_step`]`(base_eps, ..)`, the three axis differences
/// normalised, `(0, 1, 0)` when their length is below `1e-10`, NaN when a
/// sample is not finite. The one formula behind [`DistanceSdfQuery`] and
/// the normals of `ModifiedSdf`, `SingleModifiedSdf` and `DestructibleSdf`.
#[cfg(feature = "std")]
#[inline]
#[must_use]
pub(crate) fn fd_normal<F: Fn(f32, f32, f32) -> f32>(
    f: F,
    base_eps: f32,
    x: f32,
    y: f32,
    z: f32,
) -> (f32, f32, f32) {
    let e = fd_normal_step(base_eps, x, y, z);
    let dx = f(x + e, y, z) - f(x - e, y, z);
    let dy = f(x, y + e, z) - f(x, y - e, z);
    let dz = f(x, y, z + e) - f(x, y, z - e);

    let len = dz.mul_add(dz, dx.mul_add(dx, dy * dy)).sqrt();
    if len < 1e-10 {
        (0.0, 1.0, 0.0)
    } else {
        (dx / len, dy / len, dz / len)
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
    /// Body index this SDF is attached to (`usize::MAX` = static world geometry)
    pub body_index: usize,
    // -- Cached invariants (derived from rotation/scale) --
    /// Inverse rotation (cached)
    pub(crate) inv_rotation: QuatFix,
    /// Scale as f32 (cached)
    pub(crate) scale_f32: f32,
    /// Inverse scale as f32 (cached, guards against zero)
    pub(crate) inv_scale_f32: f32,
}

/// The placement of a distance field in the world: position, orientation
/// and uniform scale, with the inverse rotation and the scale as `f32`
/// precomputed.
///
/// A field is evaluated in its local space: a world point `p` is queried at
/// `R⁻¹ (p - position) / scale`, the returned distance is multiplied by
/// `scale`, and a local normal is rotated by `R` back to world space. This is
/// the transform an [`SdfCollider`] applies; [`SdfCollider::frame`] returns
/// the collider's own frame (from its cached values, so a collider and its
/// frame agree bit for bit even when the cache was not refreshed with
/// [`SdfCollider::update_cache`]).
///
/// A scale whose magnitude is below `1e-10` is inverted as `1`, as in
/// [`SdfCollider::with_scale`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfFrame {
    position: Vec3Fix,
    rotation: QuatFix,
    inv_rotation: QuatFix,
    scale_f32: f32,
    inv_scale_f32: f32,
}

impl SdfFrame {
    /// The world frame: no translation, no rotation, scale 1.
    pub const IDENTITY: Self = Self {
        position: Vec3Fix::ZERO,
        rotation: QuatFix::IDENTITY,
        inv_rotation: QuatFix::IDENTITY,
        scale_f32: 1.0,
        inv_scale_f32: 1.0,
    };

    /// A field placed at `position`, rotated by `rotation`, scaled by `scale`.
    #[must_use]
    pub fn new(position: Vec3Fix, rotation: QuatFix, scale: Fix128) -> Self {
        let s = scale.to_f32();
        Self {
            position,
            rotation,
            inv_rotation: rotation.conjugate(),
            scale_f32: s,
            inv_scale_f32: if s.abs() < 1e-10 { 1.0 } else { 1.0 / s },
        }
    }

    /// The uniform scale as `f32` (local distances are multiplied by it).
    #[must_use]
    pub fn scale_f32(&self) -> f32 {
        self.scale_f32
    }

    /// Transform a world-space point to the field's local space (f32).
    #[cfg(feature = "std")]
    #[inline]
    pub(crate) fn world_to_local(&self, world_point: Vec3Fix) -> (f32, f32, f32) {
        let relative = world_point - self.position;
        let local = self.inv_rotation.rotate_vec(relative);
        let (lx, ly, lz) = local.to_f32();
        let inv_s = self.inv_scale_f32;
        (lx * inv_s, ly * inv_s, lz * inv_s)
    }

    /// Transform a local-space normal to world space (Fix128).
    #[cfg(feature = "std")]
    #[inline]
    pub(crate) fn local_normal_to_world(&self, nx: f32, ny: f32, nz: f32) -> Vec3Fix {
        let local_n = Vec3Fix::from_f32(nx, ny, nz);
        self.rotation.rotate_vec(local_n).normalize()
    }
}

/// Sentinel value for static (world-fixed) SDF colliders
pub const SDF_STATIC: usize = usize::MAX;

impl SdfCollider {
    /// Create a new SDF collider attached to the world (static).
    #[must_use]
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
    ///
    /// The field's local origin is the body's position and its local axes are
    /// the body's axes. The collider starts at the world origin with no
    /// rotation and does not follow the body by itself: the body's pose is
    /// copied in by [`Self::sync_to_body`] (or [`sync_dynamic_sdf_colliders`]
    /// for a whole set), which has to run after the body moves and before the
    /// collider is queried.
    #[must_use]
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
    #[must_use]
    pub fn with_scale(mut self, scale: Fix128) -> Self {
        self.scale = scale;
        let s = scale.to_f32();
        self.scale_f32 = s;
        self.inv_scale_f32 = if s.abs() < 1e-10 { 1.0 } else { 1.0 / s };
        self
    }

    /// Recompute cached fields after changing position/rotation externally.
    ///
    /// The queries read the cached inverse rotation and scale, so a collider
    /// whose `rotation` or `scale` field was written directly keeps answering
    /// for the old orientation and scale until this is called.
    /// [`Self::set_pose`] and [`Self::sync_to_body`] call it themselves.
    pub fn update_cache(&mut self) {
        self.inv_rotation = self.rotation.conjugate();
        let s = self.scale.to_f32();
        self.scale_f32 = s;
        self.inv_scale_f32 = if s.abs() < 1e-10 { 1.0 } else { 1.0 / s };
    }

    /// Place the collider at `position` with orientation `rotation` (a unit
    /// quaternion) and refresh the cached inverse rotation, so every query
    /// that follows evaluates the field in the new placement. The scale is
    /// kept.
    pub fn set_pose(&mut self, position: Vec3Fix, rotation: QuatFix) {
        self.position = position;
        self.rotation = rotation;
        self.update_cache();
    }

    /// Copy the pose of the body this collider is attached to:
    /// `set_pose(bodies[body_index].position, bodies[body_index].rotation)`.
    ///
    /// Returns `true` when the pose was copied. A static collider
    /// ([`SDF_STATIC`]) or a `body_index` outside `bodies` is left untouched
    /// and gives `false`.
    pub fn sync_to_body(&mut self, bodies: &[crate::solver::RigidBody]) -> bool {
        if self.body_index == SDF_STATIC {
            return false;
        }
        match bodies.get(self.body_index) {
            Some(body) => {
                self.set_pose(body.position, body.rotation);
                true
            }
            None => false,
        }
    }

    /// Transform world-space point to SDF local space (f32).
    ///
    /// Uses cached `inv_rotation` and `inv_scale_f32` to avoid recomputation.
    #[cfg(feature = "std")]
    #[inline]
    pub(crate) fn world_to_local(&self, world_point: Vec3Fix) -> (f32, f32, f32) {
        self.frame().world_to_local(world_point)
    }

    /// Transform local-space normal to world space (Fix128).
    #[cfg(feature = "std")]
    #[inline]
    pub(crate) fn local_normal_to_world(&self, nx: f32, ny: f32, nz: f32) -> Vec3Fix {
        self.frame().local_normal_to_world(nx, ny, nz)
    }

    /// The collider's placement as an [`SdfFrame`], taken from its cached
    /// inverse rotation and scale (the values every query uses), so a
    /// collider and its frame transform points identically.
    #[must_use]
    pub fn frame(&self) -> SdfFrame {
        SdfFrame {
            position: self.position,
            rotation: self.rotation,
            inv_rotation: self.inv_rotation,
            scale_f32: self.scale_f32,
            inv_scale_f32: self.inv_scale_f32,
        }
    }
}

/// Move every dynamic collider of `colliders` to the current pose of the
/// body it is attached to ([`SdfCollider::sync_to_body`]); static colliders
/// are not touched. Returns how many colliders were moved.
///
/// The queries below take the collider as it is stored, so a set that holds
/// colliders made by [`SdfCollider::new_dynamic`] has to be synced after the
/// bodies move and before it is queried.
pub fn sync_dynamic_sdf_colliders(
    colliders: &mut [SdfCollider],
    bodies: &[crate::solver::RigidBody],
) -> usize {
    let mut moved = 0;
    for collider in colliders {
        if collider.sync_to_body(bodies) {
            moved += 1;
        }
    }
    moved
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
#[must_use]
pub fn collide_point_sdf(point: Vec3Fix, sdf: &SdfCollider) -> Option<Contact> {
    collide_point_sdf_field(point, &*sdf.field, &sdf.frame())
}

/// [`collide_point_sdf`] against a borrowed field placed by `frame`.
///
/// `field` need not be `'static`, `Send` or `Sync` (see [`SdfQuery`]); the
/// result is bit for bit that of [`collide_point_sdf`] for a collider with
/// the same field and [`SdfCollider::frame`]. A point on the surface or
/// outside (world distance `>= 0`) gives `None`.
#[cfg(feature = "std")]
#[must_use]
pub fn collide_point_sdf_field<F: SdfQuery + ?Sized>(
    point: Vec3Fix,
    field: &F,
    frame: &SdfFrame,
) -> Option<Contact> {
    let (lx, ly, lz) = frame.world_to_local(point);

    // Early-out: distance only (1 eval instead of 5)
    let dist = field.query_distance(lx, ly, lz);
    let world_dist = dist * frame.scale_f32();

    if world_dist >= 0.0 {
        return None; // Outside or on surface
    }

    // Only compute normal for penetrating points (4 additional evals)
    let (nx, ny, nz) = field.query_normal(lx, ly, lz);

    let depth = Fix128::from_f32(-world_dist);
    let normal = frame.local_normal_to_world(nx, ny, nz);
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
#[must_use]
pub fn collide_sphere_sdf(center: Vec3Fix, radius: Fix128, sdf: &SdfCollider) -> Option<Contact> {
    collide_sphere_sdf_field(center, radius, &*sdf.field, &sdf.frame())
}

/// [`collide_sphere_sdf`] against a borrowed field placed by `frame`.
///
/// `field` need not be `'static`, `Send` or `Sync` (see [`SdfQuery`]); the
/// result is bit for bit that of [`collide_sphere_sdf`] for a collider with
/// the same field and [`SdfCollider::frame`]. A sphere that does not reach
/// the surface (`radius - world distance <= 0`) gives `None`; with zero
/// `radius` this is the point test with the touching case excluded.
#[cfg(feature = "std")]
#[must_use]
pub fn collide_sphere_sdf_field<F: SdfQuery + ?Sized>(
    center: Vec3Fix,
    radius: Fix128,
    field: &F,
    frame: &SdfFrame,
) -> Option<Contact> {
    let (lx, ly, lz) = frame.world_to_local(center);

    // Early-out: distance only (1 eval)
    let dist = field.query_distance(lx, ly, lz);
    let world_dist = dist * frame.scale_f32();
    let radius_f32 = radius.to_f32();

    // Penetration = radius - distance_to_surface
    let penetration = radius_f32 - world_dist;

    if penetration <= 0.0 {
        return None; // Sphere doesn't reach the surface
    }

    // Only compute normal for penetrating spheres (4 additional evals)
    let (nx, ny, nz) = field.query_normal(lx, ly, lz);

    let depth = Fix128::from_f32(penetration);
    let normal = frame.local_normal_to_world(nx, ny, nz);

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
#[must_use]
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

/// The 27 points of a box that decide its contact with a field: its 8 corners, 12
/// edge midpoints, 6 face centres and its centre (`centre + R · (i·hx, j·hy, k·hz)`
/// for `i, j, k ∈ {−1, 0, 1}`). A flat field is deepest at a corner; a curved one
/// (a ball, a hill) can be deepest at a face centre or an edge midpoint.
#[cfg(feature = "std")]
#[must_use]
pub(crate) fn box_sample_points(
    center: Vec3Fix,
    half_extents: Vec3Fix,
    rotation: QuatFix,
) -> [Vec3Fix; 27] {
    let mut points = [center; 27];
    let steps = [-Fix128::ONE, Fix128::ZERO, Fix128::ONE];
    let mut n = 0;
    for &i in &steps {
        for &j in &steps {
            for &k in &steps {
                let local =
                    Vec3Fix::new(half_extents.x * i, half_extents.y * j, half_extents.z * k);
                points[n] = center + rotation.unit_rotation().rotate_vec(local);
                n += 1;
            }
        }
    }
    points
}

/// The deepest contact of a set of points with an SDF.
#[cfg(feature = "std")]
#[must_use]
pub(crate) fn collide_points_sdf(points: &[Vec3Fix], sdf: &SdfCollider) -> Option<Contact> {
    let mut best: Option<Contact> = None;
    for &point in points {
        if let Some(contact) = collide_point_sdf(point, sdf) {
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
/// Samples 27 points of the box (corners, edge midpoints, face centres, centre;
/// `box_sample_points`) and returns the deepest penetrating contact. Exact for a
/// flat field; for a curved one the deepest point of the box can lie between the
/// samples.
#[cfg(feature = "std")]
#[must_use]
pub fn collide_aabb_sdf(min: Vec3Fix, max: Vec3Fix, sdf: &SdfCollider) -> Option<Contact> {
    let center = Vec3Fix::new(
        (min.x + max.x).half(),
        (min.y + max.y).half(),
        (min.z + max.z).half(),
    );
    let half = Vec3Fix::new(
        (max.x - min.x).half(),
        (max.y - min.y).half(),
        (max.z - min.z).half(),
    );
    collide_points_sdf(&box_sample_points(center, half, QuatFix::IDENTITY), sdf)
}

/// Detect all SDF collisions for a set of bodies.
///
/// For each body, tests against all SDF colliders and returns contacts.
/// Bodies with `inv_mass == 0` (static) are skipped, and so is the collider
/// attached to the body itself. Each collider is queried at its stored pose;
/// call [`sync_dynamic_sdf_colliders`] first when the set holds dynamic
/// colliders.
#[cfg(feature = "std")]
#[must_use]
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

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    /// Simple unit sphere SDF for testing
    fn unit_sphere() -> ClosureSdf {
        ClosureSdf::new(
            |x, y, z| z.mul_add(z, x.mul_add(x, y * y)).sqrt() - 1.0,
            |x, y, z| {
                let len = z.mul_add(z, x.mul_add(x, y * y)).sqrt();
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
            "Depth should be ~0.5, got {depth}"
        );

        // Normal should point in +X
        let (nx, _, _) = contact.normal.to_f32();
        assert!(nx > 0.9, "Normal should point in +X, got {nx}");
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
            "Penetration should be ~0.5, got {depth}"
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
            "Penetration should be ~0.2, got {depth}"
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
            "Penetration should be ~0.5, got {depth}"
        );
    }

    #[test]
    fn detect_sdf_contacts_reports_penetrating_dynamic_bodies_with_closed_form_depth() {
        use crate::solver::RigidBody;
        let ground =
            SdfCollider::new_static(Box::new(ground_plane()), Vec3Fix::ZERO, QuatFix::IDENTITY);
        // 単位球 SDF (原点) を body 5 に attach (self-collision skip の確認用)
        let attached = SdfCollider::new_dynamic(Box::new(unit_sphere()), 5);
        let colliders = [ground, attached];
        let radius = Fix128::from_ratio(1, 2);

        let mut bodies = vec![
            // 0: dynamic、(5, 0.25, 0) → 地面貫入 0.5 - 0.25 = 0.25、球からは遠い
            RigidBody::new_dynamic(Vec3Fix::from_f32(5.0, 0.25, 0.0), Fix128::ONE),
            // 1: dynamic、(0, 2, 0) → 地面 非接触、球 距離 1 ≥ 0.5 → 非接触
            RigidBody::new_dynamic(Vec3Fix::from_f32(0.0, 2.0, 0.0), Fix128::ONE),
            // 2: static、y = -1 (完全貫入) でも skip
            RigidBody::new_static(Vec3Fix::from_f32(0.0, -1.0, 0.0)),
            // 3: dynamic、(0, 0, 5) 地面上 → 貫入 0.5、球からは距離 4 → 非接触
            RigidBody::new_dynamic(Vec3Fix::from_f32(0.0, 0.0, 5.0), Fix128::ONE),
            // 4: dynamic、(1.2, 3, 0) → 地面 非接触、球 距離 sqrt(10.44)-1 ≈ 2.23 → 非接触
            RigidBody::new_dynamic(Vec3Fix::from_f32(1.2, 3.0, 0.0), Fix128::ONE),
            // 5: dynamic、(1.2, 0, 0) → 地面貫入 0.5、自身に attach された球 SDF は skip
            RigidBody::new_dynamic(Vec3Fix::from_f32(1.2, 0.0, 0.0), Fix128::ONE),
        ];

        let contacts = detect_sdf_contacts(&bodies, &colliders, radius);
        let idxs: Vec<usize> = contacts.iter().map(|(i, _)| *i).collect();
        // body 順 → collider 順で列挙される
        assert_eq!(idxs, vec![0, 3, 5]);
        let expected_depth = [0.25_f32, 0.5, 0.5];
        for (k, want) in expected_depth.iter().enumerate() {
            let d = contacts[k].1.depth.to_f32();
            assert!(
                (d - want).abs() < 1e-5,
                "contact {k} depth {d}, want {want}"
            );
            let (nx, ny, nz) = contacts[k].1.normal.to_f32();
            assert!(nx.abs() < 1e-5 && (ny - 1.0).abs() < 1e-5 && nz.abs() < 1e-5);
            // point_b は SDF 表面 (y = 0)、point_a は球の最下点 (y = pos - 0.5)
            let (_, pby, _) = contacts[k].1.point_b.to_f32();
            let (_, pay, _) = contacts[k].1.point_a.to_f32();
            let (_, y, _) = bodies[idxs[k]].position.to_f32();
            assert!(pby.abs() < 1e-5, "contact {k} point_b.y {pby}");
            assert!(
                (pay - (y - 0.5)).abs() < 1e-5,
                "contact {k} point_a.y {pay}"
            );
        }

        // 同じ位置 (1.2, 0, 0) の別 body (index 6) は球 SDF との接触も報告される:
        // 距離 0.2 < 0.5 → 貫入 0.3、法線 +X (self-collision skip は index で判定)
        bodies.push(RigidBody::new_dynamic(
            Vec3Fix::from_f32(1.2, 0.0, 0.0),
            Fix128::ONE,
        ));
        let c2 = detect_sdf_contacts(&bodies, &colliders, radius);
        let idxs2: Vec<usize> = c2.iter().map(|(i, _)| *i).collect();
        assert_eq!(idxs2, vec![0, 3, 5, 6, 6]);
        let sphere_contact = &c2[4].1;
        let d = sphere_contact.depth.to_f32();
        assert!((d - 0.3).abs() < 1e-5, "sphere depth {d}");
        let (nx, ny, nz) = sphere_contact.normal.to_f32();
        assert!((nx - 1.0).abs() < 1e-5 && ny.abs() < 1e-5 && nz.abs() < 1e-5);

        // 半径 0 なら地面上 (y=0) の body は接触しない (penetration <= 0 → None)
        let none = detect_sdf_contacts(&bodies[3..4], &colliders[..1], Fix128::ZERO);
        assert!(none.is_empty());
        // collider なし → 空
        assert!(detect_sdf_contacts(&bodies, &[], radius).is_empty());
    }
}
