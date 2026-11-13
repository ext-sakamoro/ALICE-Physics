//! Ellipsoid Collider
//!
//! Ellipsoid collider with arbitrary orientation, supporting GJK/EPA collision detection.
//!
//! # Features
//!
//! - **Ellipsoid**: Three independent semi-axis radii (rx, ry, rz)
//! - **GJK Support**: Implements `Support` trait for GJK/EPA pipeline
//! - **Bounding Sphere**: Conservative bounding sphere from max semi-axis
//!
//! # Geometry
//!
//! The ellipsoid is defined by a center, three semi-axis radii stored as a
//! `Vec3Fix`, and an orientation quaternion. In local space the surface
//! satisfies `(x/rx)^2 + (y/ry)^2 + (z/rz)^2 = 1`.
//!
//! Author: Moroya Sakamoto

use crate::collider::{hypot3, Support, AABB};
use crate::math::{Fix128, QuatFix, Vec3Fix};

/// Ellipsoid collider
///
/// An ellipsoid defined by center position, three semi-axis radii, and an
/// orientation quaternion. The radii are stored as `Vec3Fix` where
/// `radii.x`, `radii.y`, `radii.z` are the semi-axes along local X, Y, Z.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ellipsoid {
    /// Center position in world space
    pub center: Vec3Fix,
    /// Semi-axis radii (rx, ry, rz) in local space
    pub radii: Vec3Fix,
    /// Orientation quaternion
    pub rotation: QuatFix,
}

impl Ellipsoid {
    /// Create a new axis-aligned ellipsoid
    #[inline]
    #[must_use]
    pub const fn new(center: Vec3Fix, radii: Vec3Fix) -> Self {
        Self {
            center,
            radii,
            rotation: QuatFix::IDENTITY,
        }
    }

    /// Create a new oriented ellipsoid
    #[inline]
    #[must_use]
    pub const fn with_rotation(center: Vec3Fix, radii: Vec3Fix, rotation: QuatFix) -> Self {
        Self {
            center,
            radii,
            rotation,
        }
    }

    /// Bounding sphere radius (maximum semi-axis)
    #[must_use]
    pub fn bounding_sphere_radius(&self) -> Fix128 {
        let mut max = self.radii.x;
        if self.radii.y > max {
            max = self.radii.y;
        }
        if self.radii.z > max {
            max = self.radii.z;
        }
        max
    }

    /// The smallest world-axis box enclosing this ellipsoid.
    ///
    /// With the turned semi-axes `e_j = R·(r_j ê_j)`, the half-extent on world
    /// axis `i` is `√(Σ_j e_{j,i}²) = √(Σ_j R_ij² r_j²)`: the support of the
    /// ellipsoid along `ê_i`.
    #[must_use]
    pub fn aabb(&self) -> AABB {
        let r = self.rotation.unit_rotation();
        let ex = r.rotate_vec(Vec3Fix::new(self.radii.x, Fix128::ZERO, Fix128::ZERO));
        let ey = r.rotate_vec(Vec3Fix::new(Fix128::ZERO, self.radii.y, Fix128::ZERO));
        let ez = r.rotate_vec(Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, self.radii.z));
        let extent = Vec3Fix::new(
            hypot3(ex.x, ey.x, ez.x),
            hypot3(ex.y, ey.y, ez.y),
            hypot3(ex.z, ey.z, ez.z),
        );
        AABB::new(self.center - extent, self.center + extent)
    }

    /// Volume of the ellipsoid: (4/3) * pi * rx * ry * rz
    #[inline]
    #[must_use]
    pub fn volume(&self) -> Fix128 {
        let pi = Fix128::PI;
        let four_thirds = Fix128::from_ratio(4, 3);
        four_thirds * pi * self.radii.x * self.radii.y * self.radii.z
    }

    /// Compute inertia tensor (diagonal) for given mass
    ///
    /// For a solid ellipsoid:
    /// - Ixx = m/5 * (ry^2 + rz^2)
    /// - Iyy = m/5 * (rx^2 + rz^2)
    /// - Izz = m/5 * (rx^2 + ry^2)
    #[must_use]
    pub fn inertia_diagonal(&self, mass: Fix128) -> Vec3Fix {
        let five = Fix128::from_int(5);
        let rx2 = self.radii.x * self.radii.x;
        let ry2 = self.radii.y * self.radii.y;
        let rz2 = self.radii.z * self.radii.z;

        Vec3Fix::new(
            mass * (ry2 + rz2) / five,
            mass * (rx2 + rz2) / five,
            mass * (rx2 + ry2) / five,
        )
    }
}

/// Halvings of the bracket of the nearest-point root (Eberly's `GetRoot`): a
/// fixed count, not a tolerance, so the result is the same bits everywhere.
/// 128 halvings shrink any bracket of the `Fix128` range below one raw unit.
const NEAREST_ROOT_HALVINGS: usize = 128;

/// The root of `F(s) = Σ (rᵢ zᵢ / (s + rᵢ))² − 1` on the bracket
/// `(z_last − 1, g < 0 ? 0 : |r ⊙ z| − 1)` by `NEAREST_ROOT_HALVINGS` fixed
/// halvings (Eberly, "Distance from a Point to an Ellipse, an Ellipsoid, or a
/// Hyperellipsoid", 2013, §2.6, normalised coordinates `r = (e/e_last)²`,
/// `z = y/e`).
fn nearest_root(n: usize, r: [Fix128; 3], z: [Fix128; 3], g: Fix128) -> Fix128 {
    let mut rz = [Fix128::ZERO; 3];
    let mut norm2 = Fix128::ZERO;
    for i in 0..n {
        rz[i] = r[i] * z[i];
        norm2 = norm2 + rz[i] * rz[i];
    }
    let mut s0 = z[n - 1] - Fix128::ONE;
    let mut s1 = if g < Fix128::ZERO {
        Fix128::ZERO
    } else {
        norm2.sqrt() - Fix128::ONE
    };
    let two = Fix128::from_int(2);
    let mut s = (s0 + s1) / two;
    for _ in 0..NEAREST_ROOT_HALVINGS {
        s = (s0 + s1) / two;
        let mut f = -Fix128::ONE;
        for i in 0..n {
            let q = rz[i] / (s + r[i]);
            f = f + q * q;
        }
        if f > Fix128::ZERO {
            s0 = s;
        } else {
            s1 = s;
        }
    }
    s
}

/// The nearest point of the ellipsoid with semi-axes `e` (decreasing) to `y`
/// (first orthant), on the first `n` axes: Eberly's `DistancePointEllipsoid`,
/// recursive in the dimension when the last coordinate is 0.
fn nearest_sorted(n: usize, e: [Fix128; 3], y: [Fix128; 3]) -> [Fix128; 3] {
    let mut x = [Fix128::ZERO; 3];
    if n == 1 {
        x[0] = e[0];
        return x;
    }
    let last = n - 1;
    if y[last] > Fix128::ZERO {
        let mut z = [Fix128::ZERO; 3];
        let mut g = -Fix128::ONE;
        for i in 0..n {
            z[i] = y[i] / e[i];
            g = g + z[i] * z[i];
        }
        if g.is_zero() {
            x[..n].copy_from_slice(&y[..n]);
            return x;
        }
        let mut r = [Fix128::ZERO; 3];
        for i in 0..n {
            let q = e[i] / e[last];
            r[i] = q * q;
        }
        let s = nearest_root(n, r, z, g);
        for i in 0..n {
            x[i] = r[i] * y[i] / (s + r[i]);
        }
        return x;
    }
    // y[last] = 0: the nearest point may lie off the y[last] = 0 plane
    let mut numer = [Fix128::ZERO; 3];
    let mut denom = [Fix128::ZERO; 3];
    let mut inside = true;
    for i in 0..last {
        numer[i] = e[i] * y[i];
        denom[i] = e[i] * e[i] - e[last] * e[last];
        if numer[i] >= denom[i] {
            inside = false;
        }
    }
    if inside {
        let mut discr = Fix128::ONE;
        let mut xde = [Fix128::ZERO; 3];
        for i in 0..last {
            xde[i] = numer[i] / denom[i];
            discr = discr - xde[i] * xde[i];
        }
        if discr > Fix128::ZERO {
            for i in 0..last {
                x[i] = e[i] * xde[i];
            }
            x[last] = e[last] * discr.sqrt();
            return x;
        }
    }
    let sub = nearest_sorted(last, e, y);
    x[..last].copy_from_slice(&sub[..last]);
    x
}

impl Ellipsoid {
    /// The point of the ellipsoid's surface nearest to `p` (world frame), and
    /// whether `p` is inside. Exact (Eberly's algorithm, fixed-count bisection):
    /// a point on a symmetry axis gets the axis point, so the normal there is
    /// the axis itself, not an iterative estimate.
    #[must_use]
    pub fn nearest_point(&self, p: Vec3Fix) -> (Vec3Fix, bool) {
        let q = self.rotation.unit_rotation();
        let local = q.conjugate().rotate_vec(p - self.center);
        let mut y = [local.x.abs(), local.y.abs(), local.z.abs()];
        let mut e = [self.radii.x, self.radii.y, self.radii.z];
        let mut axis = [0usize, 1, 2];
        // Sort the semi-axes decreasing, carrying the point and the axis index.
        for (a, b) in [(0, 1), (1, 2), (0, 1)] {
            if e[a] < e[b] {
                e.swap(a, b);
                y.swap(a, b);
                axis.swap(a, b);
            }
        }
        let x_sorted = nearest_sorted(3, e, y);
        let mut out = [Fix128::ZERO; 3];
        let signs = [local.x, local.y, local.z];
        for k in 0..3 {
            let i = axis[k];
            out[i] = if signs[i].is_negative() {
                -x_sorted[k]
            } else {
                x_sorted[k]
            };
        }
        let level = (local.x / self.radii.x) * (local.x / self.radii.x)
            + (local.y / self.radii.y) * (local.y / self.radii.y)
            + (local.z / self.radii.z) * (local.z / self.radii.z);
        let nearest = self.center + q.rotate_vec(Vec3Fix::new(out[0], out[1], out[2]));
        (nearest, level < Fix128::ONE)
    }

    /// The outward unit normal of the surface at its point `x` (world frame):
    /// the gradient of the quadric, `(xᵢ / eᵢ²)` in the local frame.
    #[must_use]
    pub fn normal_at(&self, x: Vec3Fix) -> Vec3Fix {
        let q = self.rotation.unit_rotation();
        let l = q.conjugate().rotate_vec(x - self.center);
        let grad = Vec3Fix::new(
            l.x / (self.radii.x * self.radii.x),
            l.y / (self.radii.y * self.radii.y),
            l.z / (self.radii.z * self.radii.z),
        );
        q.rotate_vec(grad.rescaled_direction().normalize())
    }
}

impl Support for Ellipsoid {
    fn support(&self, direction: Vec3Fix) -> Vec3Fix {
        // Only the direction matters: bring its scale into range first so the
        // squares below neither underflow nor overflow
        let direction = direction.rescaled_direction();
        // Transform direction to local space
        let local_dir = self
            .rotation
            .unit_rotation()
            .conjugate()
            .rotate_vec(direction);

        // Scale direction by radii (componentwise)
        let scaled = Vec3Fix::new(
            local_dir.x * self.radii.x,
            local_dir.y * self.radii.y,
            local_dir.z * self.radii.z,
        );

        let len = scaled.length();
        let local_support = if len.is_zero() {
            // Degenerate: return point along X semi-axis
            Vec3Fix::new(self.radii.x, Fix128::ZERO, Fix128::ZERO)
        } else {
            // Normalize the scaled direction and scale by radii again
            Vec3Fix::new(
                self.radii.x * scaled.x / len,
                self.radii.y * scaled.y / len,
                self.radii.z * scaled.z / len,
            )
        };

        // Transform back to world space
        self.center + self.rotation.unit_rotation().rotate_vec(local_support)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Points on a symmetry axis get the axis point exactly (oracle: the
    /// nearest point of an ellipsoid to a point on an axis outside it is the
    /// axis's end), and the normal there is the axis, for semi-axes given in
    /// either order (the algorithm sorts them).
    #[test]
    fn a_point_on_an_axis_gets_the_axis_end() {
        for radii in [Vec3Fix::from_int(3, 2, 1), Vec3Fix::from_int(1, 2, 3)] {
            let e = Ellipsoid::new(Vec3Fix::ZERO, radii);
            for (p, want) in [
                (
                    Vec3Fix::from_int(0, 5, 0),
                    Vec3Fix::new(Fix128::ZERO, radii.y, Fix128::ZERO),
                ),
                (
                    Vec3Fix::from_int(0, -5, 0),
                    Vec3Fix::new(Fix128::ZERO, -radii.y, Fix128::ZERO),
                ),
                (
                    Vec3Fix::from_int(7, 0, 0),
                    Vec3Fix::new(radii.x, Fix128::ZERO, Fix128::ZERO),
                ),
                (
                    Vec3Fix::from_int(0, 0, -7),
                    Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, -radii.z),
                ),
            ] {
                let (x, inside) = e.nearest_point(p);
                assert_eq!(x, want, "{radii:?} {p:?}");
                assert!(!inside);
                let n = e.normal_at(x);
                let axis = (p - x).normalize();
                assert!(
                    (n - axis).length() < Fix128::from_ratio(1, 1 << 40),
                    "{n:?} {axis:?}"
                );
            }
        }
    }

    /// A point off the axes, outside and inside: the nearest point lies on the
    /// surface (`Σ xᵢ² / eᵢ² = 1`) and the segment to the point runs along the
    /// surface normal there (the condition that defines the nearest point), to
    /// the bisection's resolution; a turned ellipsoid gives the same in its frame.
    #[test]
    fn an_off_axis_point_meets_the_surface_along_its_normal() {
        let tol = Fix128::from_ratio(1, 1 << 30);
        let turn = QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, Fix128::HALF_PI);
        for (rotation, p, outside) in [
            (QuatFix::IDENTITY, Vec3Fix::from_int(2, 2, 2), true),
            (
                QuatFix::IDENTITY,
                Vec3Fix::new(
                    Fix128::from_ratio(1, 2),
                    Fix128::from_ratio(1, 2),
                    Fix128::from_ratio(1, 4),
                ),
                false,
            ),
            (turn, Vec3Fix::from_int(3, 1, 1), true),
            // in the plane of the shortest semi-axis (z = 0), inside: the nearest
            // point leaves that plane (Eberly's y_last = 0 branch)
            (
                QuatFix::IDENTITY,
                Vec3Fix::new(
                    Fix128::from_ratio(1, 2),
                    Fix128::from_ratio(1, 4),
                    Fix128::ZERO,
                ),
                false,
            ),
        ] {
            let e = Ellipsoid {
                center: Vec3Fix::ZERO,
                radii: Vec3Fix::from_int(3, 2, 1),
                rotation,
            };
            let (x, inside) = e.nearest_point(p);
            assert_eq!(inside, !outside, "{p:?}");
            let l = rotation.conjugate().rotate_vec(x);
            let level = (l.x / e.radii.x) * (l.x / e.radii.x)
                + (l.y / e.radii.y) * (l.y / e.radii.y)
                + (l.z / e.radii.z) * (l.z / e.radii.z);
            assert!(
                (level - Fix128::ONE).abs() < tol,
                "level {level:?} at {x:?}"
            );
            let n = e.normal_at(x);
            let along = (p - x).normalize();
            let along = if outside { along } else { -along };
            assert!(n.cross(along).length() < tol, "{n:?} vs {along:?}");
            assert!(n.dot(along) > Fix128::ZERO);
        }
    }

    #[test]
    fn test_ellipsoid_new() {
        let e = Ellipsoid::new(Vec3Fix::ZERO, Vec3Fix::from_int(1, 2, 3));
        assert_eq!(e.radii.x.hi, 1);
        assert_eq!(e.radii.y.hi, 2);
        assert_eq!(e.radii.z.hi, 3);
    }

    #[test]
    fn test_ellipsoid_bounding_sphere_radius() {
        let e = Ellipsoid::new(Vec3Fix::ZERO, Vec3Fix::from_int(1, 5, 3));
        assert_eq!(e.bounding_sphere_radius().hi, 5);
    }

    #[test]
    fn test_ellipsoid_support_x() {
        let e = Ellipsoid::new(Vec3Fix::ZERO, Vec3Fix::from_int(3, 1, 1));
        let s = e.support(Vec3Fix::UNIT_X);
        // Support along +X for axis-aligned ellipsoid should be (rx, 0, 0)
        assert_eq!(s.x.hi, 3);
        assert!(s.y.abs().hi == 0);
        assert!(s.z.abs().hi == 0);
    }

    #[test]
    fn test_ellipsoid_support_y() {
        let e = Ellipsoid::new(Vec3Fix::ZERO, Vec3Fix::from_int(1, 4, 1));
        let s = e.support(Vec3Fix::UNIT_Y);
        assert_eq!(s.y.hi, 4);
    }

    #[test]
    fn test_ellipsoid_support_neg_z() {
        let e = Ellipsoid::new(Vec3Fix::ZERO, Vec3Fix::from_int(1, 1, 7));
        let s = e.support(-Vec3Fix::UNIT_Z);
        assert_eq!(s.z.hi, -7);
    }

    #[test]
    fn test_ellipsoid_sphere_degenerates() {
        // An ellipsoid with equal radii is a sphere
        let e = Ellipsoid::new(Vec3Fix::ZERO, Vec3Fix::from_int(2, 2, 2));
        let s = e.support(Vec3Fix::UNIT_X);
        assert_eq!(s.x.hi, 2);

        let s2 = e.support(Vec3Fix::UNIT_Y);
        assert_eq!(s2.y.hi, 2);
    }

    #[test]
    fn test_ellipsoid_volume() {
        let e = Ellipsoid::new(Vec3Fix::ZERO, Vec3Fix::from_int(1, 1, 1));
        // V = (4/3)*pi*1*1*1 = 4*pi/3 ≈ 4.189
        let v = e.volume();
        assert!(
            v.hi >= 4 && v.hi <= 5,
            "Volume should be ~4.189, got {}",
            v.hi
        );
    }

    #[test]
    fn test_ellipsoid_aabb() {
        let e = Ellipsoid::new(Vec3Fix::from_int(5, 0, 0), Vec3Fix::from_int(2, 3, 1));
        let aabb = e.aabb();
        assert_eq!(aabb.min.x.hi, 3); // 5 - 2
        assert_eq!(aabb.max.x.hi, 7); // 5 + 2
        assert_eq!(aabb.min.y.hi, -3); // 0 - 3
        assert_eq!(aabb.max.y.hi, 3); // 0 + 3
    }

    #[test]
    fn test_ellipsoid_gjk_collision() {
        use crate::collider::{gjk, Sphere};
        let e = Ellipsoid::new(Vec3Fix::ZERO, Vec3Fix::from_int(2, 2, 2));
        let sphere = Sphere::new(Vec3Fix::from_int(1, 0, 0), Fix128::ONE);
        let result = gjk(&e, &sphere);
        assert!(
            result.colliding,
            "Overlapping ellipsoid and sphere should collide"
        );
    }

    #[test]
    fn test_ellipsoid_gjk_no_collision() {
        use crate::collider::{gjk, Sphere};
        let e = Ellipsoid::new(Vec3Fix::ZERO, Vec3Fix::from_int(1, 1, 1));
        let sphere = Sphere::new(Vec3Fix::from_int(10, 0, 0), Fix128::ONE);
        let result = gjk(&e, &sphere);
        assert!(
            !result.colliding,
            "Separated ellipsoid and sphere should not collide"
        );
    }

    #[test]
    fn test_ellipsoid_offset_center() {
        let e = Ellipsoid::new(Vec3Fix::from_int(10, 20, 30), Vec3Fix::from_int(1, 1, 1));
        let s = e.support(Vec3Fix::UNIT_X);
        assert_eq!(s.x.hi, 11);
        assert_eq!(s.y.hi, 20);
        assert_eq!(s.z.hi, 30);
    }
}
