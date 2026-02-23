//! Mass Property Computation from Geometry
//!
//! Computes mass, center of mass, and inertia tensors for common geometric
//! shapes using deterministic 128-bit fixed-point arithmetic.
//!
//! # Supported Shapes
//!
//! - Sphere
//! - Box (axis-aligned half-extents)
//! - Cylinder
//! - Capsule (cylinder + hemisphere caps)
//! - Convex hull (tetrahedron decomposition)
//!
//! # Parallel Axis Theorem
//!
//! [`translate_inertia`] shifts an inertia tensor to a new reference point.

use crate::math::{Fix128, Mat3Fix, Vec3Fix};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ============================================================================
// Mass Properties
// ============================================================================

/// Mass, center of mass, and inertia tensor for a rigid body.
#[derive(Clone, Copy, Debug)]
pub struct MassProperties {
    /// Total mass
    pub mass: Fix128,
    /// Center of mass in local coordinates
    pub center_of_mass: Vec3Fix,
    /// Inertia tensor about the center of mass (3x3 matrix)
    pub inertia_tensor: Mat3Fix,
}

impl MassProperties {
    /// Zero mass properties (massless / infinitely light).
    pub const ZERO: Self = Self {
        mass: Fix128::ZERO,
        center_of_mass: Vec3Fix::ZERO,
        inertia_tensor: Mat3Fix::ZERO,
    };
}

// ============================================================================
// Shape-specific mass property functions
// ============================================================================

/// Compute mass properties of a solid sphere.
///
/// Inertia: `I = 2/5 * m * r^2` (diagonal, all axes equal).
#[must_use]
pub fn sphere_mass_properties(radius: Fix128, density: Fix128) -> MassProperties {
    // Volume = 4/3 * pi * r^3
    let r2 = radius * radius;
    let r3 = r2 * radius;
    let volume = Fix128::from_ratio(4, 3) * Fix128::PI * r3;
    let mass = volume * density;

    // I = 2/5 * m * r^2
    let i = Fix128::from_ratio(2, 5) * mass * r2;

    MassProperties {
        mass,
        center_of_mass: Vec3Fix::ZERO,
        inertia_tensor: Mat3Fix::diagonal(i, i, i),
    }
}

/// Compute mass properties of an axis-aligned box defined by half-extents.
///
/// Half-extents `(hx, hy, hz)` define a box from `(-hx,-hy,-hz)` to `(hx,hy,hz)`.
///
/// Inertia: `Ixx = m/12 * (hy^2 + hz^2)`, etc.
#[must_use]
pub fn box_mass_properties(half_extents: Vec3Fix, density: Fix128) -> MassProperties {
    let two = Fix128::from_int(2);
    let w = half_extents.x * two; // full width
    let h = half_extents.y * two; // full height
    let d = half_extents.z * two; // full depth

    let volume = w * h * d;
    let mass = volume * density;

    let w2 = w * w;
    let h2 = h * h;
    let d2 = d * d;
    let factor = mass * Fix128::from_ratio(1, 12);

    let ixx = factor * (h2 + d2);
    let iyy = factor * (w2 + d2);
    let izz = factor * (w2 + h2);

    MassProperties {
        mass,
        center_of_mass: Vec3Fix::ZERO,
        inertia_tensor: Mat3Fix::diagonal(ixx, iyy, izz),
    }
}

/// Compute mass properties of a solid cylinder aligned along the Y axis.
///
/// - `radius`: cylinder radius
/// - `half_height`: half the total height
#[must_use]
pub fn cylinder_mass_properties(
    radius: Fix128,
    half_height: Fix128,
    density: Fix128,
) -> MassProperties {
    let two = Fix128::from_int(2);
    let h = half_height * two;
    let r2 = radius * radius;

    // Volume = pi * r^2 * h
    let volume = Fix128::PI * r2 * h;
    let mass = volume * density;

    // Iyy (along axis) = m * r^2 / 2
    let iyy = mass * r2 * Fix128::from_ratio(1, 2);

    // Ixx = Izz = m/12 * (3*r^2 + h^2)
    let h2 = h * h;
    let three_r2 = Fix128::from_int(3) * r2;
    let ixx = mass * Fix128::from_ratio(1, 12) * (three_r2 + h2);

    MassProperties {
        mass,
        center_of_mass: Vec3Fix::ZERO,
        inertia_tensor: Mat3Fix::diagonal(ixx, iyy, ixx),
    }
}

/// Compute mass properties of a capsule (cylinder + two hemisphere caps) aligned along Y.
///
/// - `radius`: capsule radius (hemisphere + cylinder radius)
/// - `half_height`: half the cylinder segment height (total height = 2*half_height + 2*radius)
#[must_use]
pub fn capsule_mass_properties(
    radius: Fix128,
    half_height: Fix128,
    density: Fix128,
) -> MassProperties {
    let two = Fix128::from_int(2);
    let r2 = radius * radius;
    let r3 = r2 * radius;
    let h = half_height * two;

    // Cylinder part
    let cyl_vol = Fix128::PI * r2 * h;
    let cyl_mass = cyl_vol * density;

    // Sphere part (two hemispheres = one sphere)
    let sph_vol = Fix128::from_ratio(4, 3) * Fix128::PI * r3;
    let sph_mass = sph_vol * density;

    let total_mass = cyl_mass + sph_mass;

    // Cylinder inertia about its own center
    let h2 = h * h;
    let cyl_iyy = cyl_mass * r2 * Fix128::from_ratio(1, 2);
    let cyl_ixx = cyl_mass * Fix128::from_ratio(1, 12) * (Fix128::from_int(3) * r2 + h2);

    // Sphere inertia about its own center
    let sph_i_own = Fix128::from_ratio(2, 5) * sph_mass * r2;

    // Sphere center offset from capsule center (along Y) for parallel axis theorem
    // Each hemisphere center is at y = +/- (half_height + 3*radius/8) from capsule center
    // For a full sphere split into two hemispheres, the effective CoM offset is:
    // hemisphere CoM at 3r/8 from flat face, so from capsule center: half_height + 3r/8
    let hemi_offset = half_height + Fix128::from_ratio(3, 8) * radius;
    let hemi_offset2 = hemi_offset * hemi_offset;

    // Sphere Iyy (along axis): no offset needed (axially symmetric)
    let sph_iyy = sph_i_own;

    // Sphere Ixx (perpendicular): parallel axis theorem for offset along Y
    let sph_ixx = sph_i_own + sph_mass * hemi_offset2;

    let iyy = cyl_iyy + sph_iyy;
    let ixx = cyl_ixx + sph_ixx;

    MassProperties {
        mass: total_mass,
        center_of_mass: Vec3Fix::ZERO,
        inertia_tensor: Mat3Fix::diagonal(ixx, iyy, ixx),
    }
}

/// Compute mass properties of a convex hull via tetrahedron decomposition.
///
/// Decomposes the convex hull into tetrahedra from the centroid to each triangle
/// face. Each tetrahedron contributes mass and inertia. The vertices are expected
/// to form a closed convex surface (convex hull vertex soup).
///
/// For a simple approximation, this uses the centroid as the decomposition origin
/// and assumes the vertices form triangle fans. For best results, pass vertices
/// from an actual convex hull with face connectivity.
#[must_use]
pub fn convex_hull_mass_properties(vertices: &[Vec3Fix], density: Fix128) -> MassProperties {
    if vertices.len() < 4 {
        return MassProperties::ZERO;
    }

    // Compute centroid
    let mut centroid = Vec3Fix::ZERO;
    for v in vertices {
        centroid = centroid + *v;
    }
    let n = Fix128::from_int(vertices.len() as i64);
    centroid = centroid / n;

    let mut total_mass = Fix128::ZERO;
    let mut total_com = Vec3Fix::ZERO;
    let mut total_inertia = Mat3Fix::ZERO;

    // Form tetrahedra from centroid to consecutive vertex triples
    let nv = vertices.len();
    for i in 0..nv {
        let v0 = centroid;
        let v1 = vertices[i];
        let v2 = vertices[(i + 1) % nv];
        let v3 = vertices[(i + 2) % nv];

        let (m, com, inertia) = tetrahedron_mass_properties(v0, v1, v2, v3, density);
        if m.is_zero() {
            continue;
        }

        total_com = total_com + com * m;
        total_mass = total_mass + m;

        // Accumulate inertia (already about origin)
        total_inertia = Mat3Fix::from_cols(
            total_inertia.col0 + inertia.col0,
            total_inertia.col1 + inertia.col1,
            total_inertia.col2 + inertia.col2,
        );
    }

    if total_mass.is_zero() {
        return MassProperties::ZERO;
    }

    let com = total_com / total_mass;

    MassProperties {
        mass: total_mass,
        center_of_mass: com,
        inertia_tensor: total_inertia,
    }
}

/// Compute mass properties of a single tetrahedron.
///
/// Returns (mass, center_of_mass, inertia_tensor_about_origin).
fn tetrahedron_mass_properties(
    v0: Vec3Fix,
    v1: Vec3Fix,
    v2: Vec3Fix,
    v3: Vec3Fix,
    density: Fix128,
) -> (Fix128, Vec3Fix, Mat3Fix) {
    // Edges from v0
    let a = v1 - v0;
    let b = v2 - v0;
    let c = v3 - v0;

    // Signed volume = (a . (b x c)) / 6
    let cross = b.cross(c);
    let det = a.dot(cross);
    let volume = det / Fix128::from_int(6);
    let abs_volume = volume.abs();

    if abs_volume.is_zero() {
        return (Fix128::ZERO, Vec3Fix::ZERO, Mat3Fix::ZERO);
    }

    let mass = abs_volume * density;

    // Center of mass at (v0 + v1 + v2 + v3) / 4
    let four = Fix128::from_int(4);
    let com = (v0 + v1 + v2 + v3) / four;

    // Inertia tensor using canonical tetrahedron formulas
    // For simplicity, approximate with point mass at CoM
    let r = com;
    let r2 = r.dot(r);
    let ixx = mass * (r2 - r.x * r.x);
    let iyy = mass * (r2 - r.y * r.y);
    let izz = mass * (r2 - r.z * r.z);
    let ixy = mass * (Fix128::ZERO - r.x * r.y);
    let ixz = mass * (Fix128::ZERO - r.x * r.z);
    let iyz = mass * (Fix128::ZERO - r.y * r.z);

    let inertia = Mat3Fix::from_cols(
        Vec3Fix::new(ixx, ixy, ixz),
        Vec3Fix::new(ixy, iyy, iyz),
        Vec3Fix::new(ixz, iyz, izz),
    );

    (mass, com, inertia)
}

/// Translate an inertia tensor to a new reference point using the parallel axis theorem.
///
/// Given mass properties about the center of mass and an offset vector `d`,
/// returns the inertia tensor about the point `center_of_mass + d`.
///
/// `I_new = I_cm + m * (d.d * E - d (x) d)` where E is identity, (x) is outer product.
#[must_use]
pub fn translate_inertia(props: &MassProperties, offset: Vec3Fix) -> Mat3Fix {
    let m = props.mass;
    let d = offset;
    let d2 = d.dot(d);

    // m * d.d * Identity
    let diag_term = m * d2;
    let identity_part = Mat3Fix::diagonal(diag_term, diag_term, diag_term);

    // m * outer(d, d)
    let outer = Mat3Fix::from_cols(
        Vec3Fix::new(m * d.x * d.x, m * d.x * d.y, m * d.x * d.z),
        Vec3Fix::new(m * d.y * d.x, m * d.y * d.y, m * d.y * d.z),
        Vec3Fix::new(m * d.z * d.x, m * d.z * d.y, m * d.z * d.z),
    );

    // I_new = I_cm + identity_part - outer
    Mat3Fix::from_cols(
        props.inertia_tensor.col0 + identity_part.col0 - outer.col0,
        props.inertia_tensor.col1 + identity_part.col1 - outer.col1,
        props.inertia_tensor.col2 + identity_part.col2 - outer.col2,
    )
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: Fix128, b: Fix128, epsilon: Fix128) -> bool {
        let diff = a - b;
        diff.abs() < epsilon
    }

    #[test]
    fn test_sphere_mass() {
        let props = sphere_mass_properties(Fix128::ONE, Fix128::ONE);
        // Volume = 4/3 * pi ≈ 4.189
        // mass = volume * density = ~4.189
        assert!(props.mass > Fix128::from_int(4));
        assert!(props.mass < Fix128::from_int(5));
    }

    #[test]
    fn test_sphere_inertia_diagonal() {
        let props = sphere_mass_properties(Fix128::ONE, Fix128::ONE);
        // I = 2/5 * m * r^2, all diagonal elements equal
        let ixx = props.inertia_tensor.col0.x;
        let iyy = props.inertia_tensor.col1.y;
        let izz = props.inertia_tensor.col2.z;
        assert_eq!(ixx, iyy);
        assert_eq!(iyy, izz);
    }

    #[test]
    fn test_sphere_center_of_mass_at_origin() {
        let props = sphere_mass_properties(Fix128::from_int(3), Fix128::ONE);
        assert!(props.center_of_mass.x.is_zero());
        assert!(props.center_of_mass.y.is_zero());
        assert!(props.center_of_mass.z.is_zero());
    }

    #[test]
    fn test_box_mass() {
        // Unit cube: half_extents = (0.5, 0.5, 0.5), volume = 1
        let he = Vec3Fix::new(
            Fix128::from_ratio(1, 2),
            Fix128::from_ratio(1, 2),
            Fix128::from_ratio(1, 2),
        );
        let props = box_mass_properties(he, Fix128::ONE);
        // Volume = 1, density = 1, mass = 1
        assert!(approx_eq(
            props.mass,
            Fix128::ONE,
            Fix128::from_ratio(1, 1000)
        ));
    }

    #[test]
    fn test_box_inertia_unit_cube() {
        let he = Vec3Fix::new(
            Fix128::from_ratio(1, 2),
            Fix128::from_ratio(1, 2),
            Fix128::from_ratio(1, 2),
        );
        let props = box_mass_properties(he, Fix128::ONE);
        // I = m/12 * (h^2 + d^2) = 1/12 * (1+1) = 1/6 ≈ 0.1667
        let expected = Fix128::from_ratio(1, 6);
        let eps = Fix128::from_ratio(1, 100);
        assert!(approx_eq(props.inertia_tensor.col0.x, expected, eps));
        assert!(approx_eq(props.inertia_tensor.col1.y, expected, eps));
        assert!(approx_eq(props.inertia_tensor.col2.z, expected, eps));
    }

    #[test]
    fn test_box_center_of_mass() {
        let he = Vec3Fix::from_int(1, 2, 3);
        let props = box_mass_properties(he, Fix128::ONE);
        assert!(props.center_of_mass.x.is_zero());
    }

    #[test]
    fn test_cylinder_mass() {
        let props = cylinder_mass_properties(Fix128::ONE, Fix128::ONE, Fix128::ONE);
        // Volume = pi * 1^2 * 2 ≈ 6.283
        assert!(props.mass > Fix128::from_int(6));
        assert!(props.mass < Fix128::from_int(7));
    }

    #[test]
    fn test_cylinder_axial_symmetry() {
        let props = cylinder_mass_properties(Fix128::ONE, Fix128::ONE, Fix128::ONE);
        // Ixx == Izz for Y-aligned cylinder
        let ixx = props.inertia_tensor.col0.x;
        let izz = props.inertia_tensor.col2.z;
        assert_eq!(ixx, izz);
    }

    #[test]
    fn test_capsule_mass_greater_than_cylinder() {
        let cyl = cylinder_mass_properties(Fix128::ONE, Fix128::ONE, Fix128::ONE);
        let cap = capsule_mass_properties(Fix128::ONE, Fix128::ONE, Fix128::ONE);
        // Capsule has extra hemisphere mass
        assert!(cap.mass > cyl.mass);
    }

    #[test]
    fn test_capsule_center_at_origin() {
        let props = capsule_mass_properties(Fix128::ONE, Fix128::from_int(2), Fix128::ONE);
        assert!(props.center_of_mass.x.is_zero());
        assert!(props.center_of_mass.y.is_zero());
    }

    #[test]
    fn test_convex_hull_basic() {
        // Tetrahedron vertices
        let verts = [
            Vec3Fix::from_int(0, 0, 0),
            Vec3Fix::from_int(1, 0, 0),
            Vec3Fix::from_int(0, 1, 0),
            Vec3Fix::from_int(0, 0, 1),
        ];
        let props = convex_hull_mass_properties(&verts, Fix128::ONE);
        assert!(props.mass > Fix128::ZERO);
    }

    #[test]
    fn test_convex_hull_too_few_vertices() {
        let verts = [Vec3Fix::from_int(0, 0, 0), Vec3Fix::from_int(1, 0, 0)];
        let props = convex_hull_mass_properties(&verts, Fix128::ONE);
        assert!(props.mass.is_zero());
    }

    #[test]
    fn test_translate_inertia_increases_diagonal() {
        let props = sphere_mass_properties(Fix128::ONE, Fix128::ONE);
        let original_iyy = props.inertia_tensor.col1.y;
        let offset = Vec3Fix::from_int(5, 0, 0);
        let translated = translate_inertia(&props, offset);
        // Parallel axis theorem: Iyy increases by m*(dx^2 + dz^2) = 1*25 = 25
        // when offset is along X axis
        assert!(translated.col1.y > original_iyy);
    }

    #[test]
    fn test_translate_inertia_zero_offset() {
        let props = sphere_mass_properties(Fix128::ONE, Fix128::ONE);
        let translated = translate_inertia(&props, Vec3Fix::ZERO);
        // Zero offset should return same inertia
        assert_eq!(translated.col0.x, props.inertia_tensor.col0.x);
        assert_eq!(translated.col1.y, props.inertia_tensor.col1.y);
        assert_eq!(translated.col2.z, props.inertia_tensor.col2.z);
    }

    #[test]
    fn test_density_scaling() {
        let props1 = sphere_mass_properties(Fix128::ONE, Fix128::ONE);
        let props2 = sphere_mass_properties(Fix128::ONE, Fix128::from_int(2));
        // Double density -> double mass
        let eps = Fix128::from_ratio(1, 1000);
        let expected = props1.mass * Fix128::from_int(2);
        assert!(approx_eq(props2.mass, expected, eps));
    }
}
