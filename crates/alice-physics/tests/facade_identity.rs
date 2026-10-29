//! The facade is the engine, not a copy of it: the semantics identifier is the engine's
//! value and the pinned one, and the types reached through `alice_physics` (directly and
//! through the prelude glob) are the engine's types.
//!
//! oracle: the engine's pinned identifier (tests/physics_semantics_id.rs `PINNED_ID_HEX`),
//! copied here so the test builds from the published package; scripts/facade_check.py fails
//! when the two differ.

use core::any::TypeId;

const PINNED_ID_HEX: &str = "e90000da9e7073b2c82c1851e78a9c08d1640298bb3dc4e11ea91fb0b8ad01d6";

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn the_semantics_identifier_is_the_engines_and_the_pinned_one() {
    assert_eq!(
        alice_physics::PHYSICS_SEMANTICS_ID,
        alice_physics_core::PHYSICS_SEMANTICS_ID
    );
    assert_eq!(hex(&alice_physics::PHYSICS_SEMANTICS_ID), PINNED_ID_HEX);
}

#[test]
fn the_types_are_the_engines_types() {
    assert_eq!(
        TypeId::of::<alice_physics::PhysicsWorld>(),
        TypeId::of::<alice_physics_core::PhysicsWorld>()
    );
    assert_eq!(
        TypeId::of::<alice_physics::Fix128>(),
        TypeId::of::<alice_physics_core::Fix128>()
    );
    assert_eq!(
        TypeId::of::<alice_physics::solver::BodyAttachments>(),
        TypeId::of::<alice_physics_core::solver::BodyAttachments>()
    );
}

mod through_the_prelude {
    use super::TypeId;
    use alice_physics::prelude::*;

    #[test]
    fn the_prelude_glob_gives_the_engines_types() {
        assert_eq!(
            TypeId::of::<PhysicsWorld>(),
            TypeId::of::<alice_physics_core::prelude::PhysicsWorld>()
        );
        assert_eq!(
            TypeId::of::<LinearBvh>(),
            TypeId::of::<alice_physics_core::bvh::LinearBvh>()
        );
        assert_eq!(
            TypeId::of::<OrientedBox>(),
            TypeId::of::<alice_physics_core::box_collider::OrientedBox>()
        );
    }
}
