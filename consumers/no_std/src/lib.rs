//! Uses alice-physics without `std`, as a no_std dependent would.
#![no_std]

/// The size of the fixed-point scalar (makes the dependency reachable).
#[must_use]
pub const fn fix128_size() -> usize {
    core::mem::size_of::<alice_physics::Fix128>()
}
