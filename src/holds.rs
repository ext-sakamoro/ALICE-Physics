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
    pub fn apply_radial_impulse(
        &mut self,
        center: Vec3Fix,
        radius: Fix128,
        strength: Fix128,
    ) -> usize {
        if radius <= Fix128::ZERO {
            return 0;
        }
        let mut reached = 0;
        for i in 0..self.bodies.len() {
            if self.bodies[i].body_type != BodyType::Dynamic {
                continue;
            }
            let way = self.bodies[i].position - center;
            let d = way.length();
            if d >= radius || d <= Fix128::ZERO {
                continue;
            }
            let along = strength * (Fix128::ONE - d / radius) / d;
            self.bodies[i].apply_impulse(way * along);
            self.wake_body(i);
            reached += 1;
        }
        reached
    }
}

/// The hands on one world. They keep only the index of a static body the springs start from, so after
/// `restore_world` the same value goes on working, and everything else about a hold is in the world.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Holds {
    anchor: usize,
}

impl Holds {
    /// New hands: adds one static body at the origin that collides with nothing
    #[must_use]
    pub fn new(world: &mut PhysicsWorld) -> Self {
        let anchor = world.add_body(RigidBody::new_static(Vec3Fix::ZERO));
        world.set_body_filter(anchor, CollisionFilter::NONE);
        Self { anchor }
    }

    /// The joint that holds `body`, if any
    fn joint_of(&self, world: &PhysicsWorld, body: usize) -> Option<usize> {
        world.joints.iter().position(
            |j| matches!(j, Joint::Spring(s) if s.body_a == self.anchor && s.body_b == body),
        )
    }

    fn holdable(&self, world: &PhysicsWorld, body: usize) -> Result<(), HoldError> {
        let b = world.bodies.get(body).ok_or(HoldError::NoBody(body))?;
        if b.body_type != BodyType::Dynamic || body == self.anchor {
            return Err(HoldError::NotDynamic(body));
        }
        if self.joint_of(world, body).is_some() {
            return Err(HoldError::AlreadyHeld(body));
        }
        Ok(())
    }

    /// Where the spring has to start for `body` to rest at `target` under its weight, `sag_per_omega2` being the
    /// inverse of ω²: the body hangs below the spring's end by `g / ω²`
    fn lifted(
        world: &PhysicsWorld,
        body: usize,
        target: Vec3Fix,
        inverse_omega2: Fix128,
    ) -> Vec3Fix {
        let g = world.config.gravity * world.bodies[body].gravity_scale;
        target - g * inverse_omega2
    }

    /// Hold `body` at `target`, reaching it at the rate `omega` (radians per second)
    pub fn hold(
        &self,
        world: &mut PhysicsWorld,
        body: usize,
        target: Vec3Fix,
        omega: Fix128,
    ) -> Result<(), HoldError> {
        self.holdable(world, body)?;
        if omega <= Fix128::ZERO {
            return Err(HoldError::BadOmega);
        }
        self.add_hold(world, body, target, omega);
        Ok(())
    }

    fn add_hold(&self, world: &mut PhysicsWorld, body: usize, target: Vec3Fix, omega: Fix128) {
        let mass = Fix128::ONE / world.bodies[body].inv_mass;
        let start = Self::lifted(world, body, target, Fix128::ONE / (omega * omega));
        world.add_joint(Joint::Spring(SpringJoint::new(
            self.anchor,
            body,
            start,
            Vec3Fix::ZERO,
            Fix128::ZERO,
            mass * omega * omega,
            Fix128::from_int(2) * mass * omega,
        )));
        world.wake_body(body);
    }

    /// Move the place a held body is going to
    pub fn move_to(
        &self,
        world: &mut PhysicsWorld,
        body: usize,
        target: Vec3Fix,
    ) -> Result<(), HoldError> {
        let idx = self.joint_of(world, body).ok_or(HoldError::NotHeld(body))?;
        let mass = Fix128::ONE / world.bodies[body].inv_mass;
        let stiffness = match &world.joints[idx] {
            Joint::Spring(s) => s.stiffness,
            _ => return Err(HoldError::NotHeld(body)),
        };
        let start = Self::lifted(world, body, target, mass / stiffness);
        if let Joint::Spring(s) = &mut world.joints[idx] {
            s.local_anchor_a = start;
        }
        world.wake_body(body);
        Ok(())
    }

    /// Let go: the body keeps its velocity and its weight
    pub fn release(&self, world: &mut PhysicsWorld, body: usize) -> Result<(), HoldError> {
        let idx = self.joint_of(world, body).ok_or(HoldError::NotHeld(body))?;
        world.remove_joint(idx);
        world.wake_body(body);
        Ok(())
    }

    /// Hold each of `bodies` at the place at the same position in `targets`: all, or on an error none. The pieces of
    /// something that forms
    pub fn form(
        &self,
        world: &mut PhysicsWorld,
        bodies: &[usize],
        targets: &[Vec3Fix],
        omega: Fix128,
    ) -> Result<(), HoldError> {
        if omega <= Fix128::ZERO {
            return Err(HoldError::BadOmega);
        }
        for (k, &body) in bodies.iter().enumerate() {
            self.holdable(world, body)?;
            if targets.get(k).is_none() {
                return Err(HoldError::NoBody(body));
            }
            if bodies[..k].contains(&body) {
                return Err(HoldError::AlreadyHeld(body));
            }
        }
        for (&body, &target) in bodies.iter().zip(targets) {
            self.add_hold(world, body, target, omega);
        }
        Ok(())
    }

    /// Let go of everything these hands hold. Returns how many
    pub fn release_all(&self, world: &mut PhysicsWorld) -> usize {
        let mut count = 0;
        // from the end, which keeps the order of what stays: it is part of the state
        for idx in (0..world.joints.len()).rev() {
            if let Joint::Spring(s) = &world.joints[idx] {
                if s.body_a == self.anchor {
                    let body = s.body_b;
                    world.remove_joint(idx);
                    world.wake_body(body);
                    count += 1;
                }
            }
        }
        count
    }

    /// Whether `body` is held
    #[must_use]
    pub fn holds(&self, world: &PhysicsWorld, body: usize) -> bool {
        self.joint_of(world, body).is_some()
    }
}
