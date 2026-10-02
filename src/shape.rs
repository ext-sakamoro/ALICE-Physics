//! Solid shapes with mass properties, and the bodies built from them.
//!
//! [`Shape`] names a solid by its dimensions and answers the three questions a
//! rigid body needs of it: how much volume it has, where its centre of mass is,
//! and what its inertia is **about that centre**. [`PhysicsWorld::add_shaped_body`](crate::solver::PhysicsWorld::add_shaped_body)
//! turns a shape and a density into a body with the right mass, the right
//! principal inertia and a collision radius that encloses the solid.
//!
//! # Why this exists
//!
//! [`RigidBody::new`](crate::solver::RigidBody::new) gives every body the inertia of a unit-radius sphere of its
//! mass, whatever it is meant to be: a long box and a ball spin identically under
//! the same torque. The closed forms for the common solids have lived in the
//! shape modules ([`crate::box_collider`], [`crate::cylinder`], [`crate::cone`],
//! [`crate::ellipsoid`], [`crate::wedge`], [`crate::torus`]) as methods nothing
//! called; a shape is how they reach a body.
//!
//! # Frames
//!
//! A shaped body's position is its **centre of mass**, and its inertia is the
//! tensor about that point in the body's local frame, whose axes are the shape's
//! (`Y` is the axis of a cylinder, cone or torus; a wedge's apex is at `+Y`).
//! A shape whose centroid is not at its geometric centre — the cone and the
//! wedge — reports the offset in [`Shape::center_of_mass_offset`] so a caller
//! that places geometry relative to the body can account for it.
//!
//! # Collision
//!
//! The solver detects collisions between spheres. A shaped body is registered
//! with the **bounding sphere about its centre of mass** as its collision radius:
//! it encloses the whole solid and is reached by it. A pair of boxes is therefore
//! detected as a pair of spheres; a narrow-phase on the shape itself is a separate
//! piece of work.
//!
//! Author: Moroya Sakamoto

use crate::box_collider::OrientedBox;
use crate::collider::Support;
use crate::cone::Cone;
use crate::cylinder::Cylinder;
use crate::ellipsoid::Ellipsoid;
use crate::math::{Fix128, QuatFix, Vec3Fix};
use crate::torus::Torus;
use crate::wedge::Wedge;

/// A solid, by its dimensions.
///
/// Dimensions are lengths and must be positive; [`Shape::validate`] says whether
/// they are. The pose is not part of a shape: it is the body's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    /// A box of the given half-extents along `X`, `Y`, `Z`.
    Box {
        /// Half the side length along each local axis.
        half_extents: Vec3Fix,
    },
    /// A solid cylinder along `Y`.
    Cylinder {
        /// Radius of the circular section.
        radius: Fix128,
        /// Half the length along `Y`.
        half_height: Fix128,
    },
    /// A solid cone along `Y`, apex at `+half_height`, base at `−half_height`.
    Cone {
        /// Radius of the base.
        radius: Fix128,
        /// Half the height, from the geometric centre to the apex or the base.
        half_height: Fix128,
    },
    /// A solid ellipsoid with the given semi-axes.
    Ellipsoid {
        /// Semi-axes along `X`, `Y`, `Z`.
        radii: Vec3Fix,
    },
    /// A triangular prism: an isosceles triangle in the `XY` plane (base `width`
    /// at `−height / 2`, apex at `+height / 2`) extruded `depth` along `Z`.
    Wedge {
        /// Base of the triangle, along `X`.
        width: Fix128,
        /// Height of the triangle, along `Y`.
        height: Fix128,
        /// Extrusion length, along `Z`.
        depth: Fix128,
    },
    /// A solid torus around `Y`.
    Torus {
        /// Distance from the axis to the centre of the tube.
        major_radius: Fix128,
        /// Radius of the tube.
        minor_radius: Fix128,
    },
}

/// Why a shape could not become a body.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShapeError {
    /// The density is zero or negative: the body would have no mass.
    NonPositiveDensity,
    /// A dimension is zero or negative, or a torus's tube is at least as wide as
    /// its ring: the solid is flat or does not exist.
    DegenerateShape,
    /// The mass or the inertia is too large for `Fix128`, whose arithmetic wraps:
    /// a body built from the wrapped numbers would be silently wrong.
    MassNotRepresentable,
}

/// Largest mass-times-length-squared the solve accepts, as an `f64`: a quarter of
/// the `Fix128` integer range, so the inertia and its reciprocal stay well inside
/// it.
const MAGNITUDE_LIMIT: f64 = (1u64 << 61) as f64;

impl Shape {
    /// Whether every dimension is positive (and a torus has a hole).
    pub fn validate(&self) -> Result<(), ShapeError> {
        let positive = |x: Fix128| x > Fix128::ZERO;
        let ok = match *self {
            Self::Box { half_extents } => {
                positive(half_extents.x) && positive(half_extents.y) && positive(half_extents.z)
            }
            Self::Cylinder {
                radius,
                half_height,
            }
            | Self::Cone {
                radius,
                half_height,
            } => positive(radius) && positive(half_height),
            Self::Ellipsoid { radii } => {
                positive(radii.x) && positive(radii.y) && positive(radii.z)
            }
            Self::Wedge {
                width,
                height,
                depth,
            } => positive(width) && positive(height) && positive(depth),
            Self::Torus {
                major_radius,
                minor_radius,
            } => positive(major_radius) && positive(minor_radius) && minor_radius < major_radius,
        };
        if ok {
            Ok(())
        } else {
            Err(ShapeError::DegenerateShape)
        }
    }

    /// Volume.
    ///
    /// Meaningful for a shape that passes [`Self::validate`] and whose size is
    /// representable; for an enormous one `Fix128` wraps, which is why
    /// [`PhysicsWorld::add_shaped_body`](crate::solver::PhysicsWorld::add_shaped_body)(crate::solver::PhysicsWorld::add_shaped_body)
    /// checks the magnitude first.
    #[must_use]
    pub fn volume(&self) -> Fix128 {
        match *self {
            Self::Box { half_extents } => {
                OrientedBox::new(Vec3Fix::ZERO, half_extents, QuatFix::IDENTITY).volume()
            }
            Self::Cylinder {
                radius,
                half_height,
            } => Cylinder::new(Vec3Fix::ZERO, half_height, radius).volume(),
            Self::Cone {
                radius,
                half_height,
            } => Cone::new(Vec3Fix::ZERO, radius, half_height).volume(),
            Self::Ellipsoid { radii } => Ellipsoid::new(Vec3Fix::ZERO, radii).volume(),
            Self::Wedge {
                width,
                height,
                depth,
            } => Wedge::new(Vec3Fix::ZERO, width, height, depth).volume(),
            Self::Torus {
                major_radius,
                minor_radius,
            } => Torus::new(Vec3Fix::ZERO, major_radius, minor_radius).volume(),
        }
    }

    /// Offset of the centre of mass from the geometric centre, in the shape's
    /// local frame: zero for the shapes symmetric about their centre, and along
    /// `Y` for the cone (`−half_height / 2`) and the wedge (`−height / 6`).
    #[must_use]
    pub fn center_of_mass_offset(&self) -> Vec3Fix {
        match *self {
            Self::Cone {
                radius,
                half_height,
            } => Cone::new(Vec3Fix::ZERO, radius, half_height).center_of_mass_offset(),
            Self::Wedge {
                width,
                height,
                depth,
            } => Wedge::new(Vec3Fix::ZERO, width, height, depth).center_of_mass_offset(),
            Self::Box { .. }
            | Self::Cylinder { .. }
            | Self::Ellipsoid { .. }
            | Self::Torus { .. } => Vec3Fix::ZERO,
        }
    }

    /// Diagonal of the inertia tensor about the centre of mass, for a given mass,
    /// in the shape's local frame.
    #[must_use]
    pub fn inertia_diagonal(&self, mass: Fix128) -> Vec3Fix {
        match *self {
            Self::Box { half_extents } => {
                OrientedBox::new(Vec3Fix::ZERO, half_extents, QuatFix::IDENTITY)
                    .inertia_diagonal(mass)
            }
            Self::Cylinder {
                radius,
                half_height,
            } => Cylinder::new(Vec3Fix::ZERO, half_height, radius).inertia_diagonal(mass),
            Self::Cone {
                radius,
                half_height,
            } => Cone::new(Vec3Fix::ZERO, radius, half_height).inertia_diagonal(mass),
            Self::Ellipsoid { radii } => {
                Ellipsoid::new(Vec3Fix::ZERO, radii).inertia_diagonal(mass)
            }
            Self::Wedge {
                width,
                height,
                depth,
            } => Wedge::new(Vec3Fix::ZERO, width, height, depth).inertia_diagonal(mass),
            Self::Torus {
                major_radius,
                minor_radius,
            } => Torus::new(Vec3Fix::ZERO, major_radius, minor_radius).inertia_diagonal(mass),
        }
    }

    /// Radius of the smallest sphere about the **centre of mass** that encloses
    /// the solid.
    #[must_use]
    pub fn bounding_radius(&self) -> Fix128 {
        match *self {
            Self::Box { half_extents } => half_extents.length(),
            Self::Cylinder {
                radius,
                half_height,
            } => (radius * radius + half_height * half_height).sqrt(),
            Self::Cone {
                radius,
                half_height,
            } => {
                // The centroid is `half_height / 2` below the geometric centre: the
                // apex is `3·half_height / 2` from it, a point of the base rim
                // `√(r² + (half_height / 2)²)`.
                let quarter = half_height.half();
                let apex = half_height + quarter;
                let rim = (radius * radius + quarter * quarter).sqrt();
                if apex > rim {
                    apex
                } else {
                    rim
                }
            }
            Self::Ellipsoid { radii } => {
                Ellipsoid::new(Vec3Fix::ZERO, radii).bounding_sphere_radius()
            }
            Self::Wedge {
                width,
                height,
                depth,
            } => {
                // The centroid is `height / 6` below the geometric centre. The
                // farthest points are a base corner (`h/3` below the centroid) and
                // an apex vertex (`2h/3` above it), at the extreme depth.
                let hd = depth.half();
                let hw = width.half();
                let third = height / Fix128::from_int(3);
                let two_thirds = third + third;
                let corner = (hw * hw + third * third + hd * hd).sqrt();
                let apex = (two_thirds * two_thirds + hd * hd).sqrt();
                if corner > apex {
                    corner
                } else {
                    apex
                }
            }
            Self::Torus {
                major_radius,
                minor_radius,
            } => major_radius + minor_radius,
        }
    }

    /// The largest length among the dimensions, as an `f64`: what the magnitude
    /// check scales the mass by.
    fn longest_f64(&self) -> f64 {
        let f = Fix128::to_f64;
        match *self {
            Self::Box { half_extents } => f(half_extents.x)
                .max(f(half_extents.y))
                .max(f(half_extents.z)),
            Self::Cylinder {
                radius,
                half_height,
            }
            | Self::Cone {
                radius,
                half_height,
            } => f(radius).max(f(half_height)),
            Self::Ellipsoid { radii } => f(radii.x).max(f(radii.y)).max(f(radii.z)),
            Self::Wedge {
                width,
                height,
                depth,
            } => f(width).max(f(height)).max(f(depth)),
            Self::Torus {
                major_radius,
                minor_radius,
            } => f(major_radius) + f(minor_radius),
        }
    }

    /// The mass of this shape at `density`, with the principal inertia about its
    /// centre of mass; refuses what cannot be a body.
    ///
    /// # Errors
    ///
    /// [`ShapeError::NonPositiveDensity`], [`ShapeError::DegenerateShape`] (from
    /// [`Self::validate`]), and [`ShapeError::MassNotRepresentable`] when the mass
    /// times the square of the longest dimension — the scale of the inertia — does
    /// not fit comfortably in `Fix128`, or the arithmetic came out non-positive
    /// anyway.
    pub fn mass_and_inertia(&self, density: Fix128) -> Result<(Fix128, Vec3Fix), ShapeError> {
        if density <= Fix128::ZERO {
            return Err(ShapeError::NonPositiveDensity);
        }
        self.validate()?;
        let longest = self.longest_f64();
        // Bound the *f64* estimate of the largest quantity the solve forms before
        // forming it: after the fact a wrapped `Fix128` is indistinguishable from a
        // small valid one.
        let estimate = density.to_f64() * longest * longest * longest * 8.0 * longest * longest;
        if !(estimate.is_finite() && estimate < MAGNITUDE_LIMIT) {
            return Err(ShapeError::MassNotRepresentable);
        }
        let mass = density * self.volume();
        let inertia = self.inertia_diagonal(mass);
        let positive = |x: Fix128| x > Fix128::ZERO;
        if positive(mass) && positive(inertia.x) && positive(inertia.y) && positive(inertia.z) {
            Ok((mass, inertia))
        } else {
            Err(ShapeError::MassNotRepresentable)
        }
    }
}

/// A [`Shape`] placed in the world: its centre of mass at `position`, turned by
/// `rotation`. It is the convex solid the narrow-phase works on, through
/// [`Support`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PosedShape {
    /// The solid.
    pub shape: Shape,
    /// Where its centre of mass is.
    pub position: Vec3Fix,
    /// Its orientation.
    pub rotation: QuatFix,
}

impl PosedShape {
    /// Where the solid's geometric centre is: the centre of mass moved back by the
    /// shape's centre-of-mass offset, turned like the solid.
    fn geometric_center(&self) -> Vec3Fix {
        self.position - self.rotation.rotate_vec(self.shape.center_of_mass_offset())
    }
}

impl Support for PosedShape {
    fn support(&self, direction: Vec3Fix) -> Vec3Fix {
        let center = self.geometric_center();
        let rotation = self.rotation;
        match self.shape {
            Shape::Box { half_extents } => {
                OrientedBox::new(center, half_extents, rotation).support(direction)
            }
            Shape::Cylinder {
                radius,
                half_height,
            } => {
                let mut c = Cylinder::new(center, half_height, radius);
                c.rotation = rotation;
                c.support(direction)
            }
            Shape::Cone {
                radius,
                half_height,
            } => {
                let mut c = Cone::new(center, radius, half_height);
                c.rotation = rotation;
                c.support(direction)
            }
            Shape::Ellipsoid { radii } => {
                let mut e = Ellipsoid::new(center, radii);
                e.rotation = rotation;
                e.support(direction)
            }
            Shape::Wedge {
                width,
                height,
                depth,
            } => {
                let mut w = Wedge::new(center, width, height, depth);
                w.rotation = rotation;
                w.support(direction)
            }
            Shape::Torus {
                major_radius,
                minor_radius,
            } => {
                let mut t = Torus::new(center, major_radius, minor_radius);
                t.rotation = rotation;
                t.support(direction)
            }
        }
    }
}
