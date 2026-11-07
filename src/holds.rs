//! Hands on a world's bodies: a blow that spreads from a point, and holding a body at a place while everything else
//! keeps its weight.
//!
//! A hold is a critically damped spring to a place, in the world's own joints, so it is solved inside the substeps,
//! is in `snapshot_world` and `state_digest`, and is undone by removing it. A body held to `x_t` moves as
//! `x(t) = x_t + (x0 + (v0 + ω x0) t) e^(−ωt)` with `x0` measured from `x_t`, whatever its mass, and comes to rest at
//! the place under gravity: the spring's other end is lifted by the sag `m g / k`.

use crate::filter::CollisionFilter;
use crate::joint::{Joint, SpringJoint};
use crate::math::{Fix128, Vec3Fix};
use crate::solver::{BodyType, PhysicsWorld, RigidBody};

/// Why a hold was refused
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HoldError {
    /// There is no such body
    NoBody(usize),
    /// A static or kinematic body cannot be pulled by a spring
    NotDynamic(usize),
    /// The body is not held
    NotHeld(usize),
    /// `omega` is not greater than zero
    BadOmega,
    /// The body is already held
    AlreadyHeld(usize),
}

impl PhysicsWorld {
    /// A blow from `center`: each dynamic body within `radius` gets the impulse `strength (1 − d/radius)` along the
    /// way from the centre to it, a negative strength pulling it in. A body at the very centre has no such way and
    /// is left alone. Bodies it reaches are woken. Returns how many it reached; none for a radius that is not
    /// positive.
    pub fn apply_radial_impulse(&mut self, _center: Vec3Fix, _radius: Fix128, _strength: Fix128) -> usize {
        todo!("radial impulse")
    }
}

/// The hands on one world. They keep only the index of a static body the springs start from, so after
/// `restore_world` the same value goes on working, and everything else about a hold is in the world.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Holds {
    anchor: usize,
}

impl Holds {
    /// New hands: adds one static body that collides with nothing
    #[must_use]
    pub fn new(_world: &mut PhysicsWorld) -> Self {
        todo!("holds")
    }

    /// Hold `body` at `target`, reaching it at the rate `omega` (radians per second)
    pub fn hold(
        &self,
        _world: &mut PhysicsWorld,
        _body: usize,
        _target: Vec3Fix,
        _omega: Fix128,
    ) -> Result<(), HoldError> {
        todo!("hold")
    }

    /// Move the place a held body is going to
    pub fn move_to(&self, _world: &mut PhysicsWorld, _body: usize, _target: Vec3Fix) -> Result<(), HoldError> {
        todo!("move")
    }

    /// Let go: the body keeps its velocity and its weight
    pub fn release(&self, _world: &mut PhysicsWorld, _body: usize) -> Result<(), HoldError> {
        todo!("release")
    }

    /// Hold each of `bodies` at the place at the same position in `targets`, one after the other: all, or on an
    /// error none. The pieces of something that forms
    pub fn form(
        &self,
        _world: &mut PhysicsWorld,
        _bodies: &[usize],
        _targets: &[Vec3Fix],
        _omega: Fix128,
    ) -> Result<(), HoldError> {
        todo!("form")
    }

    /// Let go of everything these hands hold. Returns how many
    pub fn release_all(&self, _world: &mut PhysicsWorld) -> usize {
        todo!("release all")
    }

    /// Whether `body` is held
    #[must_use]
    pub fn holds(&self, _world: &PhysicsWorld, _body: usize) -> bool {
        todo!("holds")
    }
}

#[allow(dead_code)]
fn _kept(_: Joint, _: SpringJoint, _: CollisionFilter, _: BodyType, _: RigidBody) {}
