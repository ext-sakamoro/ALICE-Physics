//! Building blocks for a world whose ground is a sphere: a radial height
//! field, and gravity that points at the centre.
//!
//! A planet's ground is "a sphere of radius `R` raised by `h` in each
//! direction". [`SphericalHeightField`] turns that description into an
//! [`SdfField`], so the same ground can be handed to
//! [`crate::sdf_character::SdfCharacter`] and to any [`crate::sdf_collider::SdfCollider`].
//! The height function is the caller's: this module knows nothing about
//! what raises or lowers the ground, only how a height per direction becomes
//! a distance.
//!
//! Several such surfaces compose with [`crate::sdf_collider::SdfUnion`]:
//! a character standing on "the higher of the land and the water" is a
//! character colliding with the union of the two height fields.
//!
//! [`central_gravity`] is the constant-magnitude field `-g · r̂`. It differs
//! from [`crate::force::ForceField::Point`], which is the inverse-square law
//! on fixed-point rigid bodies: near a planet's surface the gravity a walker
//! feels is constant, and the walker here is the `f32`
//! [`crate::sdf_character::SdfCharacter`].
//!
//! # Distance
//!
//! `f(p) = (|p − c| − R − h(dir)) / sqrt(1 + s²)`, `dir = (p − c) / |p − c|`.
//!
//! The numerator is exact along the radial line: its zero set is the
//! surface and [`SphericalHeightField::radial_height`] returns it unscaled.
//! Off the radial line it over-states the distance where the ground is
//! sloped — `|∇(|p − c| − R − h)| = sqrt(1 + |∇h|²) > 1`. With
//! [`SphericalHeightField::with_max_slope`] set to a bound `s` on
//! `|dh/dθ| / R` (rise per unit of arc on the reference sphere), the scaled
//! field is 1-Lipschitz everywhere outside the reference sphere
//! (`|p − c| ≥ R`), which is what a sphere tracer or a push-out needs. Inside
//! the reference sphere the bound loosens to `sqrt(1 + (s·R/|p − c|)²) / sqrt(1 + s²)`.
//! The default `s = 0` keeps the unscaled radial value.
//!
//! The scaling is conservative, not free: a push-out that establishes a
//! clearance `ρ` in the scaled field leaves `ρ·sqrt(1 + s²)` along the
//! radial line, so a character standing on a scaled field stands that much
//! higher. A field used only for standing on gentle ground can keep `s = 0`.
//!
//! # Normal
//!
//! The gradient of the radial value, `r̂ − ∇h`, with `∇h` taken by central
//! differences of `h` on the unit sphere over the angle
//! [`SphericalHeightField::with_normal_step`] (default `1e-3` rad).
//!
//! # Determinism
//!
//! `f32` throughout; every square root and trigonometric call goes through
//! [`crate::det_math`], so the field evaluates to the same bits on every
//! target.

use crate::det_math;
use crate::sdf_collider::SdfField;

/// Height of the ground above a reference sphere, per direction.
///
/// `dir` is a unit vector from the sphere's centre. Implemented for every
/// `Fn([f32; 3]) -> f32 + Send + Sync`, so a closure is a height function.
pub trait SurfaceHeight: Send + Sync {
    /// Height in metres above the reference sphere in direction `dir`
    /// (negative below it).
    fn height(&self, dir: [f32; 3]) -> f32;
}

impl<F: Fn([f32; 3]) -> f32 + Send + Sync> SurfaceHeight for F {
    #[inline]
    fn height(&self, dir: [f32; 3]) -> f32 {
        self(dir)
    }
}

/// A sphere of radius `radius` centred at `center`, raised by `height(dir)`.
///
/// See the [module documentation](self) for the distance law and its
/// Lipschitz bound.
pub struct SphericalHeightField<H: SurfaceHeight> {
    center: [f32; 3],
    radius: f32,
    height: H,
    inv_scale: f32,
    normal_step: f32,
}

impl<H: SurfaceHeight> core::fmt::Debug for SphericalHeightField<H> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SphericalHeightField")
            .field("center", &self.center)
            .field("radius", &self.radius)
            .field("inv_scale", &self.inv_scale)
            .field("normal_step", &self.normal_step)
            .finish_non_exhaustive()
    }
}

impl<H: SurfaceHeight> SphericalHeightField<H> {
    /// Ground of the reference sphere `(center, radius)` raised by `height`.
    ///
    /// # Panics
    ///
    /// If `radius` is not a finite positive number, or `center` is not
    /// finite.
    #[must_use]
    pub fn new(center: [f32; 3], radius: f32, height: H) -> Self {
        assert!(
            radius.is_finite() && radius > 0.0,
            "SphericalHeightField: radius must be finite and positive, got {radius}"
        );
        assert!(
            center.iter().all(|v| v.is_finite()),
            "SphericalHeightField: center must be finite, got {center:?}"
        );
        Self {
            center,
            radius,
            height,
            inv_scale: 1.0,
            normal_step: 1.0e-3,
        }
    }

    /// Declare `s`, a bound on `|dh/dθ| / R`, and divide the distance by
    /// `sqrt(1 + s²)` so the field is 1-Lipschitz outside the reference
    /// sphere. `0` (the default) keeps the unscaled radial value.
    ///
    /// # Panics
    ///
    /// If `max_slope` is negative or not finite.
    #[must_use]
    pub fn with_max_slope(mut self, max_slope: f32) -> Self {
        assert!(
            max_slope.is_finite() && max_slope >= 0.0,
            "SphericalHeightField: max_slope must be finite and >= 0, got {max_slope}"
        );
        self.inv_scale = 1.0 / det_math::sqrt(1.0 + max_slope * max_slope);
        self
    }

    /// Angle (radians) of the central differences that estimate the
    /// height gradient for [`SdfField::normal`]. Default `1e-3`.
    ///
    /// # Panics
    ///
    /// If `step` is not a finite positive number.
    #[must_use]
    pub fn with_normal_step(mut self, step: f32) -> Self {
        assert!(
            step.is_finite() && step > 0.0,
            "SphericalHeightField: normal_step must be finite and positive, got {step}"
        );
        self.normal_step = step;
        self
    }

    /// Centre of the reference sphere.
    #[must_use]
    pub fn center(&self) -> [f32; 3] {
        self.center
    }

    /// Radius of the reference sphere.
    #[must_use]
    pub fn radius(&self) -> f32 {
        self.radius
    }

    /// Distance from the centre to the ground in direction `dir` (unit
    /// vector): `R + h(dir)`.
    #[must_use]
    pub fn surface_radius(&self, dir: [f32; 3]) -> f32 {
        self.radius + self.height.height(dir)
    }

    /// Signed height of `p` above the ground along the radial line through
    /// it, `|p − c| − R − h(dir)`, without the slope scaling.
    #[must_use]
    pub fn radial_height(&self, p: [f32; 3]) -> f32 {
        let (r, dir) = self.polar(p);
        r - self.radius - self.height.height(dir)
    }

    /// `(|p − c|, (p − c) / |p − c|)`; the centre itself maps to `+Y`.
    fn polar(&self, p: [f32; 3]) -> (f32, [f32; 3]) {
        let rel = [
            p[0] - self.center[0],
            p[1] - self.center[1],
            p[2] - self.center[2],
        ];
        let r = det_math::sqrt(rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]);
        if r > 0.0 && r.is_finite() {
            (r, [rel[0] / r, rel[1] / r, rel[2] / r])
        } else {
            (0.0, [0.0, 1.0, 0.0])
        }
    }
}

impl<H: SurfaceHeight> SdfField for SphericalHeightField<H> {
    #[inline]
    fn distance(&self, x: f32, y: f32, z: f32) -> f32 {
        self.radial_height([x, y, z]) * self.inv_scale
    }

    fn normal(&self, x: f32, y: f32, z: f32) -> (f32, f32, f32) {
        let (r, up) = self.polar([x, y, z]);
        let [t1, t2] = tangent_basis(up);
        let e = self.normal_step;
        let slope = |t: [f32; 3]| {
            let plus = self.height.height(normalize(add(up, scale(t, e))));
            let minus = self.height.height(normalize(add(up, scale(t, -e))));
            (plus - minus) / (2.0 * e)
        };
        // ∇h at radius r: the angular derivative divided by r. At the
        // centre the radial value has no gradient direction other than up.
        let g = if r > 0.0 {
            let d1 = slope(t1) / r;
            let d2 = slope(t2) / r;
            [
                up[0] - t1[0] * d1 - t2[0] * d2,
                up[1] - t1[1] * d1 - t2[1] * d2,
                up[2] - t1[2] * d1 - t2[2] * d2,
            ]
        } else {
            up
        };
        let n = normalize(g);
        (n[0], n[1], n[2])
    }
}

/// Constant-magnitude gravity toward `center`: `−g · (p − c) / |p − c|`.
///
/// Zero at the centre itself (no direction) and for a non-finite `p`.
#[must_use]
pub fn central_gravity(center: [f32; 3], g: f32, position: [f32; 3]) -> [f32; 3] {
    let rel = [
        position[0] - center[0],
        position[1] - center[1],
        position[2] - center[2],
    ];
    let r = det_math::sqrt(rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]);
    if r > 0.0 && r.is_finite() {
        let k = -g / r;
        [rel[0] * k, rel[1] * k, rel[2] * k]
    } else {
        [0.0; 3]
    }
}

#[inline]
fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
fn scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Unit vector, `+Y` for a zero or non-finite input.
fn normalize(a: [f32; 3]) -> [f32; 3] {
    let l = det_math::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
    if l > 0.0 && l.is_finite() {
        [a[0] / l, a[1] / l, a[2] / l]
    } else {
        [0.0, 1.0, 0.0]
    }
}

/// Two unit vectors orthogonal to the unit vector `n` and to each other.
fn tangent_basis(n: [f32; 3]) -> [[f32; 3]; 2] {
    // cross with the world axis least aligned with n
    let a = if n[0].abs() <= n[1].abs() && n[0].abs() <= n[2].abs() {
        [1.0, 0.0, 0.0]
    } else if n[1].abs() <= n[2].abs() {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    let t1 = normalize(cross(n, a));
    let t2 = cross(n, t1);
    [t1, t2]
}

#[inline]
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tangent_basis_is_orthonormal() {
        for n in [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            normalize([0.3, -0.4, 0.5]),
        ] {
            let [t1, t2] = tangent_basis(n);
            let d = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
            assert!(d(n, t1).abs() < 1e-6 && d(n, t2).abs() < 1e-6 && d(t1, t2).abs() < 1e-6);
            assert!((d(t1, t1) - 1.0).abs() < 1e-6 && (d(t2, t2) - 1.0).abs() < 1e-6);
        }
    }
}
