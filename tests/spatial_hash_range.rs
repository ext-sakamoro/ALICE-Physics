//! `SpatialGrid` cell lookup at the ends of the `Fix128` range.
//!
//! The documented contract (`tests/audit_spatial.rs`) is
//! `cell = clamp(floor(x / cell_size) + dim/2, 0, dim - 1)` per axis.
//! The scaled coordinate `x * (1 / cell_size)` and the `+ dim/2` offset used to
//! be computed with plain `Fix128` / `i64` arithmetic, so a coordinate whose
//! cell index lies near `i64::MIN` / `i64::MAX` (huge position or tiny cell
//! size) panicked in debug (`i64` add overflow) and wrapped to a wrong cell in
//! release. These tests pin the contract for every input:
//!
//! 1. in-range inputs give bit-identical cells to the previous formula
//!    (copied below as `old_hash`),
//! 2. inputs whose scaled coordinate leaves the representable range clamp to
//!    the border cell on the side of the true sign, in both profiles,
//! 3. neither `hash` nor `query_neighbors_into` panics for any of them.
//!
//! `Fix128` is a fixed-point type: there is no NaN / infinity to test.
//! Run with both `cargo test --test spatial_hash_range` and
//! `cargo test --release --test spatial_hash_range`.

use std::panic::{catch_unwind, AssertUnwindSafe};

use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::spatial::SpatialGrid;

/// The cell formula before the range fix (valid only while `x * inv` and
/// `hi + half` do not overflow, which the callers below guarantee).
fn old_axis(x: Fix128, inv: Fix128, dim: usize) -> usize {
    let gd = dim as i64;
    let half = gd / 2;
    ((x * inv).hi + half).clamp(0, gd - 1) as usize
}

fn old_hash(p: Vec3Fix, cell: Fix128, dim: usize) -> usize {
    if dim == 0 {
        return 0;
    }
    let inv = if cell.is_zero() {
        Fix128::ONE
    } else {
        Fix128::ONE / cell
    };
    old_axis(p.x, inv, dim) + old_axis(p.y, inv, dim) * dim + old_axis(p.z, inv, dim) * dim * dim
}

fn lcg(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *s
}

/// Hash, turning a panic into `Err` so both profiles can be asserted alike.
fn try_hash(g: &SpatialGrid, p: Vec3Fix) -> Result<usize, ()> {
    catch_unwind(AssertUnwindSafe(|| g.hash(p))).map_err(|_| ())
}

fn try_query(g: &SpatialGrid, p: Vec3Fix) -> Result<Vec<usize>, ()> {
    catch_unwind(AssertUnwindSafe(|| {
        let mut out = vec![usize::MAX];
        g.query_neighbors_into(p, Fix128::ONE, &mut out);
        out
    }))
    .map_err(|_| ())
}

/// (1) In-range inputs: bit-identical to the old formula over 10k points per
/// configuration, cell sizes from 2^-30 to 2^20, positions up to 2^40.
#[test]
fn in_range_cells_are_unchanged() {
    // (cell size, largest k for positions in [-2^k, 2^k)) chosen so that
    // `|x / cell| < 2^62`: the old formula does not overflow there
    let cells = [
        (Fix128::from_raw(0, 1u64 << 34), 30), // 2^-30
        (Fix128::from_ratio(1, 5), 58),
        (Fix128::from_ratio(1, 3), 58),
        (Fix128::ONE, 60),
        (Fix128::from_int(7), 60),
        (Fix128::from_int(1 << 20), 60),
        (Fix128::from_raw(-1, 1u64 << 63), 59), // -0.5 (negative cell size)
        (Fix128::ZERO, 60),                     // treated as 1
    ];
    let dims = [0usize, 1, 2, 3, 8, 32, 33];
    let mut s = 0x5eed_u64;
    let mut checked = 0usize;
    for &(cell, kmax) in &cells {
        for &dim in &dims {
            let g = SpatialGrid::new(cell, dim);
            for _ in 0..10_000 {
                let mut c = || {
                    // hi in [-2^k, 2^k) with k drawn from 0..=kmax, random lo
                    let k = (lcg(&mut s) >> 32) % (kmax + 1);
                    let hi = (lcg(&mut s) as i64) >> (63 - k);
                    Fix128::from_raw(hi, lcg(&mut s))
                };
                let p = Vec3Fix::new(c(), c(), c());
                assert_eq!(
                    g.hash(p),
                    old_hash(p, cell, dim),
                    "cell {cell:?} dim {dim} pos {p:?}"
                );
                checked += 1;
            }
        }
    }
    assert_eq!(checked, cells.len() * dims.len() * 10_000);
}

/// Cell sizes of 1 or 2 raw units (2^-64, 2^-63): `1 / cell` is not
/// representable in `Fix128`, so the cell must be `floor(x_raw / cell_raw)`.
/// The expectation is computed here with `i128` floor division.
#[test]
fn smallest_cell_sizes_use_the_exact_quotient() {
    let dim = 16usize;
    let half = (dim / 2) as i128;
    let mut failures = Vec::new();
    for c_raw in [1i64, 2, -1, -2] {
        // cell = c_raw * 2^-64
        let cell = Fix128::from_raw(if c_raw < 0 { -1 } else { 0 }, c_raw as u64);
        let g = SpatialGrid::new(cell, dim);
        for x_raw in -40i64..=40 {
            let x = Fix128::from_raw(if x_raw < 0 { -1 } else { 0 }, x_raw as u64);
            let (xr, cr) = (x_raw as i128, c_raw as i128);
            let mut q = xr / cr;
            if xr % cr != 0 && ((xr < 0) != (cr < 0)) {
                q -= 1;
            }
            let want_x = (q + half).clamp(0, dim as i128 - 1) as usize;
            let mid = dim / 2;
            let want = want_x + mid * dim + mid * dim * dim;
            let got = try_hash(&g, Vec3Fix::new(x, Fix128::ZERO, Fix128::ZERO));
            if got != Ok(want) {
                failures.push(format!(
                    "c_raw {c_raw} x_raw {x_raw}: got {got:?}, want Ok({want})"
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// Extreme coordinates on one axis: (raw position, cell size, expected side).
/// `true` = the scaled coordinate is huge positive (last cell), `false` =
/// huge negative (cell 0).
fn extreme_cases() -> Vec<(Fix128, Fix128, bool)> {
    let tiny = Fix128::from_raw(0, 1); // 2^-64
    let half_plus = Fix128::from_raw(0, (1u64 << 63) + 1); // just above 0.5
    let one_plus = Fix128::from_raw(1, u64::MAX); // just below 2
    let neg_half = Fix128::from_raw(-1, 1u64 << 63); // -0.5
    let max = Fix128::from_raw(i64::MAX, u64::MAX);
    let max_hi = Fix128::from_raw(i64::MAX, 0);
    let max_m1 = Fix128::from_raw(i64::MAX - 1, 0);
    let min = Fix128::from_raw(i64::MIN, 0);
    let min_p1 = Fix128::from_raw(i64::MIN + 1, 0);
    let big = Fix128::from_int(1 << 62);
    let mut v = Vec::new();
    for &(p, pos) in &[
        (max, true),
        (max_hi, true),
        (max_m1, true),
        (big, true),
        (min, false),
        (min_p1, false),
        (Fix128::from_int(-(1 << 62)), false),
    ] {
        // cell 1: `hi + dim/2` overflows i64 near i64::MAX
        v.push((p, Fix128::ONE, pos));
        // cell 1/4: `x * 4` leaves the Fix128 range
        v.push((p, Fix128::from_ratio(1, 4), pos));
        // cell just above 0.5 (1/cell = 1.99..): carries through the middle
        // 128-bit partial sum of the product
        v.push((p, half_plus, pos));
        v.push((p, one_plus, pos));
        // tiny cell: 1/cell = 2^64 is itself out of range, product even more so
        v.push((p, tiny, pos));
        // negative cell size mirrors the side
        v.push((p, neg_half, !pos));
    }
    // moderate position, tiny cell: the scaled coordinate is ~2^84
    v.push((Fix128::from_int(1 << 20), tiny, true));
    v.push((Fix128::from_int(-(1 << 20)), tiny, false));
    v
}

/// (2)+(3) Extreme inputs clamp to the border cell on the side of the true
/// sign, identically in debug and release, without a panic.
#[test]
fn extreme_inputs_clamp_to_the_border_cell() {
    let mut failures = Vec::new();
    for &dim in &[1usize, 2, 4, 32] {
        for (x, cell, positive) in extreme_cases() {
            let g = SpatialGrid::new(cell, dim);
            let side = if positive { dim - 1 } else { 0 };
            // extreme on x only, origin on y / z (origin is cell dim/2)
            let mid = dim / 2;
            let want_x = side + mid * dim + mid * dim * dim;
            let got = try_hash(&g, Vec3Fix::new(x, Fix128::ZERO, Fix128::ZERO));
            if got != Ok(want_x) {
                failures.push(format!(
                    "x: dim {dim} cell {cell:?} pos {x:?}: got {got:?}, want Ok({want_x})"
                ));
            }
            // same value on all three axes
            let want_all = side + side * dim + side * dim * dim;
            let got = try_hash(&g, Vec3Fix::new(x, x, x));
            if got != Ok(want_all) {
                failures.push(format!(
                    "xyz: dim {dim} cell {cell:?} pos {x:?}: got {got:?}, want Ok({want_all})"
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// `query_neighbors_into` uses the same cell coordinate: an extreme query
/// position finds the particle stored in the border cell on its side.
#[test]
fn extreme_query_finds_the_border_cell() {
    let dim = 8usize;
    let mut failures = Vec::new();
    for (x, cell, positive) in extreme_cases() {
        // particle 0 at the extreme position, particle 1 at +-10 cells (in
        // range, clamps to the border cell on the expected side), particle 2
        // at the origin cell (not adjacent to either border for dim 8)
        let far = Vec3Fix::new(x, Fix128::ZERO, Fix128::ZERO);
        let ten = Fix128::from_int(if positive { 10 } else { -10 });
        let border = Vec3Fix::new(cell * ten, Fix128::ZERO, Fix128::ZERO);
        let built = catch_unwind(AssertUnwindSafe(|| {
            let mut g = SpatialGrid::new(cell, dim);
            g.clear();
            g.insert(0, far);
            g.insert(1, border);
            g.insert(2, Vec3Fix::ZERO);
            g.build();
            g
        }));
        let got = match built {
            Ok(g) => try_query(&g, far),
            Err(_) => Err(()),
        };
        match got {
            Ok(n) if n.contains(&0) && n.contains(&1) && !n.contains(&2) => {}
            other => failures.push(format!(
                "cell {cell:?} pos {x:?} positive {positive}: got {other:?}"
            )),
        }
    }
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
