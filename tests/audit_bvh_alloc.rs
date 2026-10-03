//! Allocation audit for `bvh`: the stackless traversal is documented as
//! "zero heap allocation" (`query_callback`: "even more allocation-free").
//! `query` returns a `Vec`, so only its result buffer may allocate.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use alice_physics::bvh::{BvhPrimitive, LinearBvh};
use alice_physics::collider::AABB;
use alice_physics::math::Vec3Fix;

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

fn lattice(n: i64) -> Vec<AABB> {
    let mut v = Vec::new();
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let o = Vec3Fix::from_int(i * 3, j * 3, k * 3);
                v.push(AABB::new(o, o + Vec3Fix::from_int(2, 2, 2)));
            }
        }
    }
    v
}

#[test]
fn query_callback_traversal_performs_no_heap_allocation() {
    let boxes = lattice(6);
    let bvh = LinearBvh::build(
        boxes
            .iter()
            .enumerate()
            .map(|(i, b)| BvhPrimitive {
                aabb: *b,
                index: i as u32,
                morton: 0,
            })
            .collect(),
    );
    let q = AABB::new(Vec3Fix::from_int(4, 4, 4), Vec3Fix::from_int(10, 10, 10));
    let mut hits = 0usize;
    bvh.query_callback(&q, |_| hits += 1);
    assert!(hits > 0);
    let before = allocs();
    let mut total = 0usize;
    for _ in 0..200 {
        bvh.query_callback(&q, |_| total += 1);
    }
    assert_eq!(allocs() - before, 0, "query_callback allocated");
    assert_eq!(total, 200 * hits);
}

#[test]
fn query_allocates_only_its_result_vector() {
    let boxes = lattice(5);
    let bvh = LinearBvh::build(
        boxes
            .iter()
            .enumerate()
            .map(|(i, b)| BvhPrimitive {
                aabb: *b,
                index: i as u32,
                morton: 0,
            })
            .collect(),
    );
    let q = AABB::new(Vec3Fix::from_int(1, 1, 1), Vec3Fix::from_int(2, 2, 2));
    let _ = bvh.query(&q);
    let before = allocs();
    let r = bvh.query(&q);
    let used = allocs() - before;
    assert!(!r.is_empty());
    // a growing Vec reallocates O(log n) times; a per-node allocation would scale with the tree
    assert!(
        used <= 8,
        "query made {used} allocations for {} results",
        r.len()
    );
}

#[test]
fn the_counter_sees_allocations_so_the_zero_above_is_not_vacuous() {
    let before = allocs();
    let v: Vec<u64> = Vec::with_capacity(64);
    assert!(allocs() > before);
    drop(v);
}
