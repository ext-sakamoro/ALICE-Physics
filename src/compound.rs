//! Compound Shape
//!
//! Combines multiple collision shapes into a single rigid body.
//! Each child shape has a local offset (position + rotation) relative to the body.
//!
//! # Usage
//!
//! ```
//! use alice_physics::compound::CompoundShape;
//! use alice_physics::{Vec3Fix, Fix128};
//! use alice_physics::collider::Sphere;
//! use alice_physics::math::QuatFix;
//!
//! let mut compound = CompoundShape::new();
//! compound.add_sphere(Sphere::new(Vec3Fix::ZERO, Fix128::ONE), Vec3Fix::ZERO, QuatFix::IDENTITY);
//! compound.add_sphere(Sphere::new(Vec3Fix::ZERO, Fix128::ONE), Vec3Fix::from_int(3, 0, 0), QuatFix::IDENTITY);
//! ```

use crate::box_collider::OrientedBox;
use crate::collider::{Capsule, ConvexHull, Sphere, Support, AABB};
use crate::mass_properties::{
    box_mass_properties, capsule_mass_properties, convex_hull_mass_properties,
    sphere_mass_properties, translate_inertia, MassProperties,
};
use crate::math::{Fix128, Mat3Fix, QuatFix, Vec3Fix};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Enumeration of supported child shape types
#[derive(Clone, Debug)]
pub enum ShapeRef {
    /// Sphere child
    Sphere(Sphere),
    /// Capsule child
    Capsule(Capsule),
    /// Oriented box child
    Box(OrientedBox),
    /// Convex hull child
    ConvexHull(ConvexHull),
}

impl ShapeRef {
    /// Compute AABB of this shape
    #[must_use]
    pub fn aabb(&self) -> AABB {
        match self {
            Self::Sphere(s) => {
                let r = Vec3Fix::new(s.radius, s.radius, s.radius);
                AABB::new(s.center - r, s.center + r)
            }
            Self::Capsule(c) => {
                let r = Vec3Fix::new(c.radius, c.radius, c.radius);
                let min = Vec3Fix::new(
                    if c.a.x < c.b.x { c.a.x } else { c.b.x },
                    if c.a.y < c.b.y { c.a.y } else { c.b.y },
                    if c.a.z < c.b.z { c.a.z } else { c.b.z },
                );
                let max = Vec3Fix::new(
                    if c.a.x > c.b.x { c.a.x } else { c.b.x },
                    if c.a.y > c.b.y { c.a.y } else { c.b.y },
                    if c.a.z > c.b.z { c.a.z } else { c.b.z },
                );
                AABB::new(min - r, max + r)
            }
            Self::Box(b) => b.aabb(),
            Self::ConvexHull(hull) => {
                let first = hull.vertices[0];
                let mut min = first;
                let mut max = first;
                for &v in &hull.vertices[1..] {
                    if v.x < min.x {
                        min.x = v.x;
                    }
                    if v.y < min.y {
                        min.y = v.y;
                    }
                    if v.z < min.z {
                        min.z = v.z;
                    }
                    if v.x > max.x {
                        max.x = v.x;
                    }
                    if v.y > max.y {
                        max.y = v.y;
                    }
                    if v.z > max.z {
                        max.z = v.z;
                    }
                }
                AABB::new(min, max)
            }
        }
    }
}

/// A child shape within a compound, with local transform
#[derive(Clone, Debug)]
pub struct CompoundChild {
    /// The shape
    pub shape: ShapeRef,
    /// Local position offset from compound center
    pub local_position: Vec3Fix,
    /// Local rotation offset
    pub local_rotation: QuatFix,
}

/// Compound shape: multiple child shapes combined into one body
#[derive(Clone, Debug)]
pub struct CompoundShape {
    /// Child shapes with their local transforms
    pub children: Vec<CompoundChild>,
    /// Cached world-space AABB (recomputed on transform)
    pub(crate) cached_aabb: AABB,
    /// Whether the cached AABB needs recomputation
    pub(crate) dirty: bool,
}

impl CompoundShape {
    /// Create an empty compound shape
    #[must_use]
    pub const fn new() -> Self {
        Self {
            children: Vec::new(),
            cached_aabb: AABB::new(Vec3Fix::ZERO, Vec3Fix::ZERO),
            dirty: true,
        }
    }

    /// Add a sphere child
    pub fn add_sphere(&mut self, sphere: Sphere, position: Vec3Fix, rotation: QuatFix) {
        self.children.push(CompoundChild {
            shape: ShapeRef::Sphere(sphere),
            local_position: position,
            local_rotation: rotation,
        });
        self.dirty = true;
    }

    /// Add a capsule child
    pub fn add_capsule(&mut self, capsule: Capsule, position: Vec3Fix, rotation: QuatFix) {
        self.children.push(CompoundChild {
            shape: ShapeRef::Capsule(capsule),
            local_position: position,
            local_rotation: rotation,
        });
        self.dirty = true;
    }

    /// Add a convex hull child
    pub fn add_convex_hull(&mut self, hull: ConvexHull, position: Vec3Fix, rotation: QuatFix) {
        self.children.push(CompoundChild {
            shape: ShapeRef::ConvexHull(hull),
            local_position: position,
            local_rotation: rotation,
        });
        self.dirty = true;
    }

    /// A compound of the convex pieces of a decomposed SDF: one hull child per
    /// piece, at the origin, so the compound is in the SDF's own coordinates.
    ///
    /// Together the hulls are the *concave* solid the SDF describes, up to the
    /// grid's resolution (a hull is smaller than the solid by up to a cell per
    /// side), where a single convex hull would fill every dent.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn from_decomposition(result: &crate::convex_decompose::DecompositionResult) -> Self {
        let mut compound = Self::new();
        for hull in &result.hulls {
            compound.add_convex_hull(hull.clone(), Vec3Fix::ZERO, QuatFix::IDENTITY);
        }
        compound
    }

    /// Decompose `sdf` inside `[min, max]` ([`crate::convex_decompose::decompose_sdf`])
    /// and build the compound of its convex pieces
    /// ([`from_decomposition`](Self::from_decomposition)).
    #[cfg(feature = "std")]
    #[must_use]
    pub fn from_sdf(
        sdf: &dyn crate::sdf_collider::SdfField,
        min: Vec3Fix,
        max: Vec3Fix,
        config: &crate::convex_decompose::DecomposeConfig,
    ) -> Self {
        Self::from_decomposition(&crate::convex_decompose::decompose_sdf(
            sdf, min, max, config,
        ))
    }

    /// Add a box child
    pub fn add_box(&mut self, obox: OrientedBox, position: Vec3Fix, rotation: QuatFix) {
        self.children.push(CompoundChild {
            shape: ShapeRef::Box(obox),
            local_position: position,
            local_rotation: rotation,
        });
        self.dirty = true;
    }

    /// Number of children
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.children.len()
    }

    /// Check if empty
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.children.is_empty()
    }

    /// Compute the AABB enclosing all children (in local space)
    pub fn compute_aabb(&mut self) -> AABB {
        if self.children.is_empty() {
            // an empty compound has a valid (degenerate) box: cache it like any other
            self.cached_aabb = AABB::new(Vec3Fix::ZERO, Vec3Fix::ZERO);
            self.dirty = false;
            return self.cached_aabb;
        }

        let first = self.child_world_aabb(0, Vec3Fix::ZERO, QuatFix::IDENTITY);
        let mut result = first;

        for i in 1..self.children.len() {
            let child_aabb = self.child_world_aabb(i, Vec3Fix::ZERO, QuatFix::IDENTITY);
            result = result.union(&child_aabb);
        }

        self.cached_aabb = result;
        self.dirty = false;
        result
    }

    /// Get world-space AABB for a specific child, given body transform
    #[must_use]
    pub fn child_world_aabb(&self, child_idx: usize, body_pos: Vec3Fix, body_rot: QuatFix) -> AABB {
        let child = &self.children[child_idx];
        let body_rot = body_rot.unit_rotation();
        let world_pos = body_pos + body_rot.rotate_vec(child.local_position);

        match &child.shape {
            ShapeRef::Sphere(s) => {
                let r = Vec3Fix::new(s.radius, s.radius, s.radius);
                // The sphere's own offset is turned by the child's rotation (composed
                // with the body's), like every other child type's, and as
                // `support_world` places it.
                let child_rot = body_rot.mul(child.local_rotation);
                let center = world_pos + child_rot.rotate_vec(s.center);
                AABB::new(center - r, center + r)
            }
            ShapeRef::Capsule(c) => {
                let child_rot = body_rot.mul(child.local_rotation);
                let a = world_pos + child_rot.rotate_vec(c.a);
                let b = world_pos + child_rot.rotate_vec(c.b);
                let r = Vec3Fix::new(c.radius, c.radius, c.radius);
                let min = Vec3Fix::new(
                    if a.x < b.x { a.x } else { b.x },
                    if a.y < b.y { a.y } else { b.y },
                    if a.z < b.z { a.z } else { b.z },
                );
                let max = Vec3Fix::new(
                    if a.x > b.x { a.x } else { b.x },
                    if a.y > b.y { a.y } else { b.y },
                    if a.z > b.z { a.z } else { b.z },
                );
                AABB::new(min - r, max + r)
            }
            ShapeRef::Box(ob) => {
                let child_rot = body_rot.mul(child.local_rotation);
                let transformed = OrientedBox::new(
                    world_pos + child_rot.rotate_vec(ob.center),
                    ob.half_extents,
                    child_rot.mul(ob.rotation),
                );
                transformed.aabb()
            }
            ShapeRef::ConvexHull(hull) => {
                let child_rot = body_rot.mul(child.local_rotation);
                let first = world_pos + child_rot.rotate_vec(hull.vertices[0]);
                let mut min = first;
                let mut max = first;
                for &v in &hull.vertices[1..] {
                    let wv = world_pos + child_rot.rotate_vec(v);
                    if wv.x < min.x {
                        min.x = wv.x;
                    }
                    if wv.y < min.y {
                        min.y = wv.y;
                    }
                    if wv.z < min.z {
                        min.z = wv.z;
                    }
                    if wv.x > max.x {
                        max.x = wv.x;
                    }
                    if wv.y > max.y {
                        max.y = wv.y;
                    }
                    if wv.z > max.z {
                        max.z = wv.z;
                    }
                }
                AABB::new(min, max)
            }
        }
    }

    /// Get world-space AABB for entire compound given body transform
    #[must_use]
    pub fn world_aabb(&self, body_pos: Vec3Fix, body_rot: QuatFix) -> AABB {
        if self.children.is_empty() {
            return AABB::new(body_pos, body_pos);
        }

        let mut result = self.child_world_aabb(0, body_pos, body_rot);
        for i in 1..self.children.len() {
            let child_aabb = self.child_world_aabb(i, body_pos, body_rot);
            result = result.union(&child_aabb);
        }
        result
    }

    /// AABB-based closest pair pruning: 対象AABBと重なる子のみを返す。
    /// ブロードフェーズで不要なナローフェーズ判定を除外する。
    #[must_use]
    pub fn overlapping_children(
        &self,
        target_aabb: &AABB,
        body_pos: Vec3Fix,
        body_rot: QuatFix,
    ) -> Vec<usize> {
        let mut result = Vec::new();
        for i in 0..self.children.len() {
            let child_aabb = self.child_world_aabb(i, body_pos, body_rot);
            if child_aabb.intersects(target_aabb) {
                result.push(i);
            }
        }
        result
    }

    /// Support function for the compound shape (selects the child with maximum support)
    #[must_use]
    pub fn support_world(
        &self,
        direction: Vec3Fix,
        body_pos: Vec3Fix,
        body_rot: QuatFix,
    ) -> Vec3Fix {
        if self.children.is_empty() {
            return body_pos;
        }
        // The best so far; `None` until the first child answers, so the choice
        // does not depend on how large the dot products are (a fixed sentinel floor
        // returned the origin for a compound farther than it from the origin).
        let mut best: Option<(Fix128, Vec3Fix)> = None;

        for child in &self.children {
            let s = child.support_world(direction, body_pos, body_rot);
            let d = s.dot(direction);
            if best.is_none_or(|(best_dot, _)| d > best_dot) {
                best = Some((d, s));
            }
        }

        best.map_or(body_pos, |(_, point)| point)
    }

    /// Mass, centre of mass and the inertia about it, for the compound made of
    /// `density`, in the compound's own frame.
    ///
    /// Each child contributes its shape's mass properties (a capsule's along its
    /// own axis, a box's in its own frame, a hull's by the exact integral over its
    /// mesh), carried to where the child is, and the tensors are summed about the
    /// common centre of mass with the parallel-axis theorem
    /// ([`crate::mass_properties::translate_inertia`]). Overlapping children are
    /// counted twice where they overlap: a compound is a union of solids, not a
    /// boolean one.
    ///
    /// [`MassProperties::ZERO`] for an empty compound, a non-positive density, or
    /// children with no volume.
    #[must_use]
    pub fn mass_properties(&self, density: Fix128) -> MassProperties {
        if density <= Fix128::ZERO {
            return MassProperties::ZERO;
        }
        // (mass, centre of mass, inertia about it), all in the compound's frame.
        let parts: Vec<MassProperties> = self
            .children
            .iter()
            .map(|child| child.mass_properties(density))
            .collect();
        let mass = parts.iter().fold(Fix128::ZERO, |acc, p| acc + p.mass);
        if mass <= Fix128::ZERO {
            return MassProperties::ZERO;
        }
        let first_moment = parts
            .iter()
            .fold(Vec3Fix::ZERO, |acc, p| acc + p.center_of_mass * p.mass);
        let com = first_moment / mass;
        let mut inertia = Mat3Fix::ZERO;
        for p in &parts {
            let about_com = translate_inertia(p, com - p.center_of_mass);
            inertia = Mat3Fix::from_cols(
                inertia.col0 + about_com.col0,
                inertia.col1 + about_com.col1,
                inertia.col2 + about_com.col2,
            );
        }
        MassProperties {
            mass,
            center_of_mass: com,
            inertia_tensor: inertia,
        }
    }
}

impl CompoundChild {
    /// The farthest point of this child along `direction`, for a body at
    /// `body_pos` turned by `body_rot`.
    #[must_use]
    pub fn support_world(
        &self,
        direction: Vec3Fix,
        body_pos: Vec3Fix,
        body_rot: QuatFix,
    ) -> Vec3Fix {
        let body_rot = body_rot.unit_rotation();
        let child_rot = body_rot.mul(self.local_rotation);
        let child_pos = body_pos + body_rot.rotate_vec(self.local_position);
        match &self.shape {
            ShapeRef::Sphere(sphere) => {
                let shifted = Sphere::new(
                    child_pos + child_rot.rotate_vec(sphere.center),
                    sphere.radius,
                );
                shifted.support(direction)
            }
            ShapeRef::Capsule(cap) => {
                let a = child_pos + child_rot.rotate_vec(cap.a);
                let b = child_pos + child_rot.rotate_vec(cap.b);
                let shifted = Capsule::new(a, b, cap.radius);
                shifted.support(direction)
            }
            ShapeRef::Box(ob) => {
                let shifted = OrientedBox::new(
                    child_pos + child_rot.rotate_vec(ob.center),
                    ob.half_extents,
                    child_rot.mul(ob.rotation),
                );
                shifted.support(direction)
            }
            ShapeRef::ConvexHull(hull) => {
                // Transform each vertex to world space and find support
                let mut best_v = child_pos + child_rot.rotate_vec(hull.vertices[0]);
                let mut best_d = best_v.dot(direction);
                for &v in &hull.vertices[1..] {
                    let wv = child_pos + child_rot.rotate_vec(v);
                    let dd = wv.dot(direction);
                    if dd > best_d {
                        best_v = wv;
                        best_d = dd;
                    }
                }
                best_v
            }
        }
    }

    /// Mass, centre of mass and the inertia about it of this child, in the
    /// compound's frame (the child's own offset and rotation applied).
    fn mass_properties(&self, density: Fix128) -> MassProperties {
        let rot = self.local_rotation;
        // The child's solid in its own frame: (props, frame rotation of the tensor).
        let (own, frame) = match &self.shape {
            ShapeRef::Sphere(s) => {
                let mut p = sphere_mass_properties(s.radius, density);
                p.center_of_mass = s.center;
                (p, QuatFix::IDENTITY)
            }
            ShapeRef::Box(ob) => {
                let mut p = box_mass_properties(ob.half_extents, density);
                p.center_of_mass = ob.center;
                (p, ob.rotation)
            }
            ShapeRef::Capsule(cap) => {
                let axis = cap.b - cap.a;
                let length = axis.length();
                let middle = (cap.a + cap.b) * Fix128::from_ratio(1, 2);
                if length.is_zero() {
                    let mut p = sphere_mass_properties(cap.radius, density);
                    p.center_of_mass = middle;
                    (p, QuatFix::IDENTITY)
                } else {
                    // The capsule solid is Y-aligned in `capsule_mass_properties`:
                    // `I = I⊥ (E − u uᵀ) + I∥ u uᵀ` for the axis `u`, which needs no
                    // rotation.
                    let p = capsule_mass_properties(cap.radius, length.half(), density);
                    let u = axis / length;
                    let d = p.inertia_tensor;
                    let (i_perp, i_axis) = (d.col0.x, d.col1.y);
                    let outer = Mat3Fix::from_cols(u * u.x, u * u.y, u * u.z);
                    let e = Mat3Fix::IDENTITY;
                    let diff = i_axis - i_perp;
                    let tensor = Mat3Fix::from_cols(
                        e.col0 * i_perp + outer.col0 * diff,
                        e.col1 * i_perp + outer.col1 * diff,
                        e.col2 * i_perp + outer.col2 * diff,
                    );
                    (
                        MassProperties {
                            mass: p.mass,
                            center_of_mass: middle,
                            inertia_tensor: tensor,
                        },
                        QuatFix::IDENTITY,
                    )
                }
            }
            ShapeRef::ConvexHull(hull) => (
                convex_hull_mass_properties(&hull.vertices, density),
                QuatFix::IDENTITY,
            ),
        };
        // Into the compound's frame: the tensor turned by (child rotation ∘ frame),
        // the centre of mass offset by the child's own turning and position.
        let turn = rot.mul(frame);
        let r = rotation_matrix(turn);
        let inertia = r.mul_mat(own.inertia_tensor).mul_mat(r.transpose());
        MassProperties {
            mass: own.mass,
            center_of_mass: self.local_position + rot.rotate_vec(own.center_of_mass),
            inertia_tensor: inertia,
        }
    }
}

/// The rotation matrix of a unit quaternion: the images of the basis vectors as
/// columns.
fn rotation_matrix(q: QuatFix) -> Mat3Fix {
    Mat3Fix::from_cols(
        q.rotate_vec(Vec3Fix::UNIT_X),
        q.rotate_vec(Vec3Fix::UNIT_Y),
        q.rotate_vec(Vec3Fix::UNIT_Z),
    )
}

impl Default for CompoundShape {
    fn default() -> Self {
        Self::new()
    }
}

/// Wrapper for using `CompoundShape` with GJK (needs body transform context)
pub struct TransformedCompound<'a> {
    /// Reference to the compound shape
    pub compound: &'a CompoundShape,
    /// Body position
    pub position: Vec3Fix,
    /// Body rotation
    pub rotation: QuatFix,
}

impl Support for TransformedCompound<'_> {
    fn support(&self, direction: Vec3Fix) -> Vec3Fix {
        self.compound
            .support_world(direction, self.position, self.rotation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    #[test]
    fn test_compound_basic() {
        let mut compound = CompoundShape::new();
        assert!(compound.is_empty());

        compound.add_sphere(
            Sphere::new(Vec3Fix::ZERO, Fix128::ONE),
            Vec3Fix::ZERO,
            QuatFix::IDENTITY,
        );
        compound.add_sphere(
            Sphere::new(Vec3Fix::ZERO, Fix128::ONE),
            Vec3Fix::from_int(5, 0, 0),
            QuatFix::IDENTITY,
        );

        assert_eq!(compound.len(), 2);
    }

    #[test]
    fn test_compound_aabb() {
        let mut compound = CompoundShape::new();
        compound.add_sphere(
            Sphere::new(Vec3Fix::ZERO, Fix128::ONE),
            Vec3Fix::ZERO,
            QuatFix::IDENTITY,
        );
        compound.add_sphere(
            Sphere::new(Vec3Fix::ZERO, Fix128::ONE),
            Vec3Fix::from_int(10, 0, 0),
            QuatFix::IDENTITY,
        );

        let aabb = compound.world_aabb(Vec3Fix::ZERO, QuatFix::IDENTITY);
        // Should span from -1 to 11 on X
        assert!(aabb.min.x <= -Fix128::ONE);
        assert!(aabb.max.x >= Fix128::from_int(11));
    }

    #[test]
    fn test_compound_support() {
        let mut compound = CompoundShape::new();
        compound.add_sphere(
            Sphere::new(Vec3Fix::ZERO, Fix128::ONE),
            Vec3Fix::ZERO,
            QuatFix::IDENTITY,
        );
        compound.add_sphere(
            Sphere::new(Vec3Fix::ZERO, Fix128::ONE),
            Vec3Fix::from_int(10, 0, 0),
            QuatFix::IDENTITY,
        );

        // Support in +X should come from the far sphere (center=10, radius=1 → 11)
        let s = compound.support_world(Vec3Fix::UNIT_X, Vec3Fix::ZERO, QuatFix::IDENTITY);
        assert!(
            s.x >= Fix128::from_int(10),
            "Support should be from far sphere"
        );
    }

    #[test]
    fn test_compound_with_box() {
        let mut compound = CompoundShape::new();
        compound.add_box(
            OrientedBox::axis_aligned(Vec3Fix::ZERO, Vec3Fix::from_int(1, 1, 1)),
            Vec3Fix::ZERO,
            QuatFix::IDENTITY,
        );
        compound.add_capsule(
            Capsule::new(
                Vec3Fix::from_int(0, -2, 0),
                Vec3Fix::from_int(0, 2, 0),
                Fix128::from_ratio(5, 10),
            ),
            Vec3Fix::from_int(3, 0, 0),
            QuatFix::IDENTITY,
        );

        assert_eq!(compound.len(), 2);

        let aabb = compound.world_aabb(Vec3Fix::ZERO, QuatFix::IDENTITY);
        assert!(aabb.min.y <= -Fix128::from_int(2));
    }

    #[test]
    fn test_transformed_compound_support() {
        let mut compound = CompoundShape::new();
        compound.add_sphere(
            Sphere::new(Vec3Fix::ZERO, Fix128::ONE),
            Vec3Fix::from_int(5, 0, 0),
            QuatFix::IDENTITY,
        );

        let tc = TransformedCompound {
            compound: &compound,
            position: Vec3Fix::from_int(10, 0, 0),
            rotation: QuatFix::IDENTITY,
        };

        let s = tc.support(Vec3Fix::UNIT_X);
        // Body at 10, child offset 5, sphere radius 1 → 16
        assert!(s.x >= Fix128::from_int(15));
    }

    #[test]
    fn test_compound_convex_hull() {
        let mut compound = CompoundShape::new();

        // 三角形の凸包
        let hull = ConvexHull::new(vec![
            Vec3Fix::from_int(0, 0, 0),
            Vec3Fix::from_int(2, 0, 0),
            Vec3Fix::from_int(1, 2, 0),
        ]);
        compound.add_convex_hull(hull, Vec3Fix::ZERO, QuatFix::IDENTITY);

        assert_eq!(compound.len(), 1);

        let aabb = compound.world_aabb(Vec3Fix::ZERO, QuatFix::IDENTITY);
        assert!(aabb.min.x <= Fix128::ZERO);
        assert!(aabb.max.x >= Fix128::from_int(2));
        assert!(aabb.max.y >= Fix128::from_int(2));
    }

    #[test]
    fn test_compound_convex_hull_support() {
        let mut compound = CompoundShape::new();

        let hull = ConvexHull::new(vec![
            Vec3Fix::from_int(0, 0, 0),
            Vec3Fix::from_int(5, 0, 0),
            Vec3Fix::from_int(0, 5, 0),
        ]);
        compound.add_convex_hull(hull, Vec3Fix::from_int(10, 0, 0), QuatFix::IDENTITY);

        // +X方向のサポート点は (10+5, 0, 0) = (15, 0, 0)
        let s = compound.support_world(Vec3Fix::UNIT_X, Vec3Fix::ZERO, QuatFix::IDENTITY);
        assert!(
            s.x >= Fix128::from_int(14),
            "Support in +X should be at x=15, got {}",
            s.x.to_f32()
        );
    }

    #[test]
    fn test_overlapping_children() {
        let mut compound = CompoundShape::new();
        // 左側の球 (x=-10)
        compound.add_sphere(
            Sphere::new(Vec3Fix::ZERO, Fix128::ONE),
            Vec3Fix::from_int(-10, 0, 0),
            QuatFix::IDENTITY,
        );
        // 右側の球 (x=10)
        compound.add_sphere(
            Sphere::new(Vec3Fix::ZERO, Fix128::ONE),
            Vec3Fix::from_int(10, 0, 0),
            QuatFix::IDENTITY,
        );

        // x=9..12のAABBは右側の球のみと重なる
        let query = AABB::new(Vec3Fix::from_int(9, -1, -1), Vec3Fix::from_int(12, 1, 1));
        let result = compound.overlapping_children(&query, Vec3Fix::ZERO, QuatFix::IDENTITY);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 1); // 右側の球（インデックス1）

        // 両方にまたがるAABB
        let query_all = AABB::new(Vec3Fix::from_int(-11, -1, -1), Vec3Fix::from_int(11, 1, 1));
        let result_all =
            compound.overlapping_children(&query_all, Vec3Fix::ZERO, QuatFix::IDENTITY);
        assert_eq!(result_all.len(), 2);
    }

    #[test]
    fn compute_aabb_is_exact_union_of_children_in_local_space_and_clears_dirty() {
        let mut compound = CompoundShape::new();
        // 空: 退化 AABB (原点) を返し、他の case と同じく cache して dirty を下ろす (1.2.0)
        assert!(compound.dirty);
        assert_eq!(
            compound.compute_aabb(),
            AABB::new(Vec3Fix::ZERO, Vec3Fix::ZERO)
        );
        assert!(!compound.dirty);
        assert_eq!(
            compound.cached_aabb,
            AABB::new(Vec3Fix::ZERO, Vec3Fix::ZERO)
        );

        // 球 r=1 at (10, 0, 0): [9, 11] × [-1, 1] × [-1, 1]
        compound.add_sphere(
            Sphere::new(Vec3Fix::ZERO, Fix128::ONE),
            Vec3Fix::from_int(10, 0, 0),
            QuatFix::IDENTITY,
        );
        assert!(compound.dirty, "add_* は dirty を立てる");
        let one = compound.compute_aabb();
        assert_eq!(one.min, Vec3Fix::from_int(9, -1, -1));
        assert_eq!(one.max, Vec3Fix::from_int(11, 1, 1));
        assert_eq!(compound.cached_aabb, one);
        assert!(!compound.dirty);

        // capsule (0,-3,0)-(0,3,0) r=2 at (-4, 0, 5): x [-6, -2]、y [-5, 5]、z [3, 7]
        compound.add_capsule(
            Capsule::new(
                Vec3Fix::from_int(0, -3, 0),
                Vec3Fix::from_int(0, 3, 0),
                Fix128::from_int(2),
            ),
            Vec3Fix::from_int(-4, 0, 5),
            QuatFix::IDENTITY,
        );
        // 球 center offset (0, 0, -8) r=1 at 原点: z [-9, -7]
        compound.add_sphere(
            Sphere::new(Vec3Fix::from_int(0, 0, -8), Fix128::ONE),
            Vec3Fix::ZERO,
            QuatFix::IDENTITY,
        );
        let all = compound.compute_aabb();
        // union: x [-6, 11]、y [-5, 5]、z [-9, 7]
        assert_eq!(all.min, Vec3Fix::from_int(-6, -5, -9));
        assert_eq!(all.max, Vec3Fix::from_int(11, 5, 7));
        assert_eq!(compound.cached_aabb, all);
        assert!(!compound.dirty);
        // world_aabb(identity) と同一、各 child AABB を全て包む
        assert_eq!(all, compound.world_aabb(Vec3Fix::ZERO, QuatFix::IDENTITY));
        for i in 0..compound.len() {
            let c = compound.child_world_aabb(i, Vec3Fix::ZERO, QuatFix::IDENTITY);
            assert_eq!(all.union(&c), all, "child {i} not enclosed");
        }
        // 冪等
        assert_eq!(compound.compute_aabb(), all);
        // body transform には依存しない (local space)、world_aabb は平行移動で追従する
        let shifted = compound.world_aabb(Vec3Fix::from_int(100, 0, 0), QuatFix::IDENTITY);
        assert_eq!(shifted.min, Vec3Fix::from_int(94, -5, -9));
        assert_eq!(shifted.max, Vec3Fix::from_int(111, 5, 7));
        assert_eq!(compound.compute_aabb(), all);
    }
}
