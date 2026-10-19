//! `alice-physics`: the API of [`alice-physics-core`](https://docs.rs/alice-physics-core) under
//! the `alice_physics` paths.
//!
//! The engine lives in `alice-physics-core`; this crate re-exports every public item of it at
//! the same path (`alice_physics::PhysicsWorld`, `alice_physics::thermal`,
//! `alice_physics::prelude::*`, ...) and forwards every feature under the same name, so code
//! written against `alice_physics` builds unchanged. It defines nothing of its own: the
//! semantics identifier is the engine's
//! (`alice_physics::PHYSICS_SEMANTICS_ID == alice_physics_core::PHYSICS_SEMANTICS_ID`).
//!
//! The C ABI libraries, the Python extension and the WebAssembly package are built from this
//! crate, so their file and module names are `alice_physics`.
//!
//! # Example
//!
//! ```rust
//! use alice_physics::{PhysicsWorld, PhysicsConfig, RigidBody, Fix128, Vec3Fix};
//!
//! // Create physics world
//! let config = PhysicsConfig::default();
//! let mut world = PhysicsWorld::new(config);
//!
//! // Add a dynamic body
//! let body = RigidBody::new_dynamic(
//!     Vec3Fix::from_int(0, 10, 0),  // position
//!     Fix128::ONE,                   // mass = 1
//! );
//! let body_id = world.add_body(body);
//!
//! // Step simulation (60 frames at 1/60 second)
//! let dt = Fix128::from_ratio(1, 60);
//! for _ in 0..60 {
//!     world.step(dt);
//! }
//!
//! // Body should have fallen under gravity
//! let pos = world.bodies[body_id].position;
//! assert!(pos.y < Fix128::from_int(10), "Body fell under gravity");
//! ```
#![no_std]

pub use alice_physics_core::*;
