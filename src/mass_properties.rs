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

// ============================================================================
// Mass Properties
// ============================================================================

/// Mass, center of mass, and inertia tensor for a rigid body.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

    // Sphere Ixx (perpendicular): each hemisphere about its *own centroid*
    // is 83/320 m_h r² (not the full sphere's 2/5 m r² about the sphere
    // centre — that value already contains the hemisphere-centroid offset
    // of 3r/8), then the parallel-axis shift to the capsule centre:
    //   2 · [83/320 m_h r² + m_h (h/2 + 3r/8)²]
    //   = m_sph · (2/5 r² + h²/4 + 3 h r / 8)
    // Before 1.2.0 the shift was added on top of 2/5 m r², over-predicting
    // I⊥ by 9/64 m_sph r² (a capsule with h = 0 did not reduce to a sphere;
    // `tests/engineering_oracles_solid.rs`).
    let hemi_own = Fix128::from_ratio(83, 320) * sph_mass * r2;
    let sph_ixx = hemi_own + sph_mass * hemi_offset2;

    let iyy = cyl_iyy + sph_iyy;
    let ixx = cyl_ixx + sph_ixx;

    MassProperties {
        mass: total_mass,
        center_of_mass: Vec3Fix::ZERO,
        inertia_tensor: Mat3Fix::diagonal(ixx, iyy, ixx),
    }
}

fn as_array(v: Vec3Fix) -> [Fix128; 3] {
    [v.x, v.y, v.z]
}

/// Compute mass properties of the convex hull of a point set.
///
/// The hull is built as a closed triangle mesh ([`crate::convex_mesh_builder::build_hull_mesh`]),
/// and mass, centre of mass and inertia are the exact integrals over the solid it
/// encloses: the solid is decomposed into the tetrahedra between the origin and each
/// boundary triangle (signed, so any origin gives the same total), and for a
/// tetrahedron with vertices `p₀…p₃` of volume `V`,
///
/// ```text
/// ∫ x xᵀ dV = V/20 · ( Σ pₖ pₖᵀ + (Σ pₖ)(Σ pₖ)ᵀ ),    I_O = tr(C)·E − C
/// ```
///
/// The result is **about the centre of mass**, like every other shape here: the
/// tensor does not depend on where the hull is, and `center_of_mass` says where it
/// is. [`MassProperties::ZERO`] when there is nothing solid — fewer than four
/// points, points in one plane, or a non-positive density.
#[must_use]
pub fn convex_hull_mass_properties(vertices: &[Vec3Fix], density: Fix128) -> MassProperties {
    if density <= Fix128::ZERO {
        return MassProperties::ZERO;
    }
    let Some(mesh) = crate::convex_mesh_builder::build_hull_mesh(vertices) else {
        return MassProperties::ZERO;
    };

    let mut volume = Fix128::ZERO;
    let mut first_moment = Vec3Fix::ZERO;
    // ∫ x xᵀ dV, row-major.
    let mut c = [[Fix128::ZERO; 3]; 3];
    let twenty = Fix128::from_int(20);
    let four = Fix128::from_int(4);
    let six = Fix128::from_int(6);
    for f in &mesh.faces {
        let (a, b, d) = (
            mesh.vertices[f[0]],
            mesh.vertices[f[1]],
            mesh.vertices[f[2]],
        );
        // Tetrahedron (origin, a, b, d): signed volume det/6.
        let v = a.dot(b.cross(d)) / six;
        volume = volume + v;
        let sum = a + b + d;
        first_moment = first_moment + sum * (v / four);
        let pts = [as_array(a), as_array(b), as_array(d)];
        let total = as_array(sum);
        for r in 0..3 {
            for col in 0..3 {
                let mut acc = total[r] * total[col];
                for p in &pts {
                    acc = acc + p[r] * p[col];
                }
                c[r][col] = c[r][col] + acc * (v / twenty);
            }
        }
    }
    if volume <= Fix128::ZERO {
        return MassProperties::ZERO;
    }

    let mass = volume * density;
    let com = first_moment / volume;
    // Inertia about the origin from the second-moment matrix, then moved to the COM.
    let trace = (c[0][0] + c[1][1] + c[2][2]) * density;
    let cd = |r: usize, col: usize| c[r][col] * density;
    let origin = |r: usize, col: usize| {
        if r == col {
            trace - cd(r, col)
        } else {
            Fix128::ZERO - cd(r, col)
        }
    };
    let d2 = com.dot(com);
    let d = as_array(com);
    let about_com = |r: usize, col: usize| {
        let shift = if r == col {
            d2 - d[r] * d[col]
        } else {
            Fix128::ZERO - d[r] * d[col]
        };
        origin(r, col) - mass * shift
    };
    let inertia = Mat3Fix::from_cols(
        Vec3Fix::new(about_com(0, 0), about_com(1, 0), about_com(2, 0)),
        Vec3Fix::new(about_com(0, 1), about_com(1, 1), about_com(2, 1)),
        Vec3Fix::new(about_com(0, 2), about_com(1, 2), about_com(2, 2)),
    );

    MassProperties {
        mass,
        center_of_mass: com,
        inertia_tensor: inertia,
    }
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

#[cfg(all(test, feature = "std"))]
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

/// The principal moments and principal axes of a symmetric 3×3 tensor (an inertia
/// tensor): `tensor = R · diag(moments) · Rᵀ`, with `R` the returned matrix whose
/// **columns** are the axes — orthonormal and right-handed (`det R = +1`).
///
/// A cyclic Jacobi iteration with a fixed number of sweeps, so the answer is the
/// same on every platform. The tensor is scaled to unit size first, so the rotation
/// angles never overflow whatever the magnitudes, and an off-diagonal term below
/// the resolution of the scaled arithmetic is left alone (an already diagonal
/// tensor, or a sphere's repeated moments, gives the identity frame). The moments
/// come back in the order of the axes; they are not sorted.
///
/// Only the lower triangle's symmetric part is read: `(I + Iᵀ)/2`.
#[must_use]
pub fn principal_axes(tensor: Mat3Fix) -> (Vec3Fix, Mat3Fix) {
    const SWEEPS: usize = 24;
    let t = [
        [tensor.col0.x, tensor.col1.x, tensor.col2.x],
        [tensor.col0.y, tensor.col1.y, tensor.col2.y],
        [tensor.col0.z, tensor.col1.z, tensor.col2.z],
    ];
    let half = Fix128::from_ratio(1, 2);
    let mut a = [[Fix128::ZERO; 3]; 3];
    let mut scale = Fix128::ZERO;
    for r in 0..3 {
        for c in 0..3 {
            a[r][c] = (t[r][c] + t[c][r]) * half;
            if a[r][c].abs() > scale {
                scale = a[r][c].abs();
            }
        }
    }
    if scale.is_zero() {
        return (Vec3Fix::ZERO, Mat3Fix::IDENTITY);
    }
    for row in &mut a {
        for v in row.iter_mut() {
            *v = *v / scale;
        }
    }
    let mut v = [
        [Fix128::ONE, Fix128::ZERO, Fix128::ZERO],
        [Fix128::ZERO, Fix128::ONE, Fix128::ZERO],
        [Fix128::ZERO, Fix128::ZERO, Fix128::ONE],
    ];
    // Below this an off-diagonal term is rounding, not structure.
    let tiny = Fix128::from_raw(0, 1 << 20);
    for _ in 0..SWEEPS {
        let mut rotated = false;
        for &(p, q) in &[(0usize, 1usize), (0, 2), (1, 2)] {
            let apq = a[p][q];
            if apq.abs() <= tiny {
                continue;
            }
            rotated = true;
            let tau = (a[q][q] - a[p][p]) / (apq + apq);
            let sign = if tau >= Fix128::ZERO {
                Fix128::ONE
            } else {
                -Fix128::ONE
            };
            let t_rot = sign / (tau.abs() + (Fix128::ONE + tau * tau).sqrt());
            let c = Fix128::ONE / (Fix128::ONE + t_rot * t_rot).sqrt();
            let s = t_rot * c;
            // A ← Jᵀ A J on rows/columns p and q.
            for row in &mut a {
                let (akp, akq) = (row[p], row[q]);
                row[p] = c * akp - s * akq;
                row[q] = s * akp + c * akq;
            }
            let (row_p, row_q) = (a[p], a[q]);
            for k in 0..3 {
                a[p][k] = c * row_p[k] - s * row_q[k];
                a[q][k] = s * row_p[k] + c * row_q[k];
            }
            a[p][q] = Fix128::ZERO;
            a[q][p] = Fix128::ZERO;
            for row in &mut v {
                let (vp, vq) = (row[p], row[q]);
                row[p] = c * vp - s * vq;
                row[q] = s * vp + c * vq;
            }
        }
        if !rotated {
            break;
        }
    }
    let moments = Vec3Fix::new(a[0][0] * scale, a[1][1] * scale, a[2][2] * scale);
    let mut axes = Mat3Fix::from_cols(
        Vec3Fix::new(v[0][0], v[1][0], v[2][0]),
        Vec3Fix::new(v[0][1], v[1][1], v[2][1]),
        Vec3Fix::new(v[0][2], v[1][2], v[2][2]),
    );
    if axes.determinant() < Fix128::ZERO {
        axes = Mat3Fix::from_cols(axes.col0, axes.col1, -axes.col2);
    }
    (moments, axes)
}
