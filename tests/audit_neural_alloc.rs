//! Allocation audit for `neural`: `DeterministicNetwork::forward` is documented
//! as "zero heap allocations" once built ("All buffers are pre-allocated at
//! construction"). A counting global allocator (per-thread counter) checks it.
//!
//! Author: Moroya Sakamoto

#![cfg(all(feature = "std", feature = "neural"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use alice_ml::TernaryWeight;
use alice_physics::math::Fix128;
use alice_physics::neural::{Activation, DeterministicNetwork, FixedTernaryWeight};

struct Counting;

thread_local! {
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
}

// SAFETY: forwards every call to the system allocator unchanged and only bumps a
// thread-local counter (no allocation, no lock) on the way.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.with(|c| c.set(c.get() + 1));
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.with(|c| c.set(c.get() + 1));
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn allocs() -> usize {
    ALLOCS.with(Cell::get)
}

#[test]
fn forward_performs_no_heap_allocation_after_construction() {
    let l1 = FixedTernaryWeight::from_ternary_weight_with_scale(
        TernaryWeight::from_ternary(&[1, 0, -1, 0, 1, 1, -1, 0, 1, 1, 0, -1], 4, 3),
        Fix128::from_ratio(1, 2),
    );
    let l2 = FixedTernaryWeight::from_ternary_weight_with_scale(
        TernaryWeight::from_ternary(&[1, -1, 0, 1, 0, 1, -1, 1], 2, 4),
        Fix128::ONE,
    );
    let mut net = DeterministicNetwork::new(
        vec![l1, l2],
        vec![Activation::TanhApprox, Activation::HardTanh],
    );
    let input = [
        Fix128::from_int(1),
        Fix128::from_int(-2),
        Fix128::from_int(3),
    ];
    let _ = net.forward(&input); // warm up
    let before = allocs();
    for _ in 0..100 {
        let out = net.forward(&input);
        assert_eq!(out.len(), 2);
    }
    assert_eq!(allocs() - before, 0, "forward allocated");
}

#[test]
fn the_counter_sees_allocations_so_the_zero_above_is_not_vacuous() {
    let before = allocs();
    let v: Vec<u64> = Vec::with_capacity(64);
    assert!(allocs() > before);
    drop(v);
}
