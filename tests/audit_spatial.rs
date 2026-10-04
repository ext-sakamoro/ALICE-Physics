//! Audit oracles for `spatial::SpatialGrid` (module had no tests/ oracle).
//!
//! Expected values come from the documented behaviour (3x3x3 neighbourhood of
//! uniform cells, `floor(x / cell) + dim/2` clamped to `[0, dim-1]`, stable
//! CSR order) re-derived here with `f64` floor / brute force, never by calling
//! the grid under test to produce the expectation.

use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::spatial::SpatialGrid;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

/// Cell coordinate as the module documents it: floor(x / cell) + dim/2, clamped.
fn cell_of(x: f64, cell: f64, dim: i64) -> i64 {
    ((x / cell).floor() as i64 + dim / 2).clamp(0, dim - 1)
}

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

fn build(cell: f64, dim: usize, pts: &[Vec3Fix]) -> SpatialGrid {
    let mut g = SpatialGrid::new(fx(cell), dim);
    g.clear();
    for (i, p) in pts.iter().enumerate() {
        g.insert(i, *p);
    }
    g.build();
    g
}

fn query(g: &SpatialGrid, p: Vec3Fix) -> Vec<usize> {
    let mut out = vec![999_999];
    g.query_neighbors_into(p, Fix128::ONE, &mut out);
    out
}

/// hash(pos) = ix + iy*dim + iz*dim^2 with floor semantics, including negatives.
#[test]
fn hash_is_the_floor_cell_index_with_centered_origin() {
    let dim = 8usize;
    let cell = 0.5;
    let g = SpatialGrid::new(fx(cell), dim);
    let mut s = 7u64;
    for _ in 0..300 {
        let x = (lcg(&mut s) - 0.5) * 6.0;
        let y = (lcg(&mut s) - 0.5) * 6.0;
        let z = (lcg(&mut s) - 0.5) * 6.0;
        let d = dim as i64;
        let want = cell_of(x, cell, d) + cell_of(y, cell, d) * d + cell_of(z, cell, d) * d * d;
        assert_eq!(g.hash(v3(x, y, z)) as i64, want, "pos ({x}, {y}, {z})");
    }
    // Origin sits at the corner of cell dim/2 (floor(0) = 0 -> half).
    assert_eq!(g.hash(v3(0.0, 0.0, 0.0)), 4 + 4 * 8 + 4 * 64);
    // -0.1 floors to cell -1 -> half - 1 (a truncating implementation gives half).
    assert_eq!(g.hash(v3(-0.1, 0.0, 0.0)), 3 + 4 * 8 + 4 * 64);
}

/// Positions outside the grid clamp to the border cell, never an out-of-range index.
#[test]
fn hash_clamps_outside_positions_to_the_border_cells() {
    let g = SpatialGrid::new(fx(1.0), 4);
    assert_eq!(g.hash(v3(1000.0, 1000.0, 1000.0)), 3 + 3 * 4 + 3 * 16);
    assert_eq!(g.hash(v3(-1000.0, -1000.0, -1000.0)), 0);
    assert!(g.hash(v3(1e9, -1e9, 5.0)) < 64);
}

/// Every particle within one cell size of the query (in range of the grid) is
/// returned: this is what makes the 3x3x3 scan a valid neighbour search when
/// the cell size equals the interaction radius.
#[test]
fn query_returns_every_particle_within_one_cell_size() {
    let cell = 0.25;
    let dim = 24usize;
    let mut s = 99u64;
    let pts: Vec<Vec3Fix> = (0..400)
        .map(|_| {
            v3(
                (lcg(&mut s) - 0.5) * 4.0,
                (lcg(&mut s) - 0.5) * 4.0,
                (lcg(&mut s) - 0.5) * 4.0,
            )
        })
        .collect();
    let g = build(cell, dim, &pts);
    for qi in 0..40 {
        let q = pts[qi * 7];
        let got = query(&g, q);
        let (qx, qy, qz) = (q.x.to_f64(), q.y.to_f64(), q.z.to_f64());
        for (i, p) in pts.iter().enumerate() {
            let (dx, dy, dz) = (p.x.to_f64() - qx, p.y.to_f64() - qy, p.z.to_f64() - qz);
            if dx * dx + dy * dy + dz * dz <= cell * cell {
                assert!(
                    got.contains(&i),
                    "particle {i} within radius missing for query {qi}"
                );
            }
        }
    }
}

/// The result is exactly the particles whose cell is within +-1 of the query cell
/// (brute force over cell coordinates), each once, and the buffer is cleared first.
#[test]
fn query_is_exactly_the_3x3x3_cell_neighbourhood() {
    let cell = 0.5;
    let dim = 10usize;
    let d = dim as i64;
    let mut s = 5u64;
    let pts: Vec<Vec3Fix> = (0..300)
        .map(|_| {
            v3(
                (lcg(&mut s) - 0.5) * 6.0,
                (lcg(&mut s) - 0.5) * 6.0,
                (lcg(&mut s) - 0.5) * 6.0,
            )
        })
        .collect();
    let g = build(cell, dim, &pts);
    for qi in 0..30 {
        let q = pts[qi * 9];
        let (qx, qy, qz) = (q.x.to_f64(), q.y.to_f64(), q.z.to_f64());
        let (cx, cy, cz) = (
            cell_of(qx, cell, d),
            cell_of(qy, cell, d),
            cell_of(qz, cell, d),
        );
        let mut want: Vec<usize> = (0..pts.len())
            .filter(|&i| {
                let (px, py, pz) = (pts[i].x.to_f64(), pts[i].y.to_f64(), pts[i].z.to_f64());
                (cell_of(px, cell, d) - cx).abs() <= 1
                    && (cell_of(py, cell, d) - cy).abs() <= 1
                    && (cell_of(pz, cell, d) - cz).abs() <= 1
            })
            .collect();
        let mut got = query(&g, q);
        got.sort_unstable();
        want.sort_unstable();
        assert_eq!(got, want, "query {qi}");
    }
}

/// Deterministic order: cells in dz-major, dy, dx-minor order; within a cell the
/// insertion order (stable counting sort).
#[test]
fn query_order_is_cell_scan_order_then_insertion_order() {
    // cell 1, dim 4: origin cell = (2,2,2).
    let pts = [
        v3(0.5, 0.5, 0.5),  // 0: cell (2,2,2)
        v3(1.5, 0.5, 0.5),  // 1: (3,2,2)  dx=+1
        v3(-0.5, 0.5, 0.5), // 2: (1,2,2)  dx=-1
        v3(0.5, 1.5, 0.5),  // 3: (2,3,2)  dy=+1
        v3(0.5, 0.5, -0.5), // 4: (2,2,1)  dz=-1
        v3(0.6, 0.6, 0.6),  // 5: (2,2,2) again, inserted later
    ];
    let g = build(1.0, 4, &pts);
    let got = query(&g, v3(0.5, 0.5, 0.5));
    // scan: dz=-1 first (4), then dz=0: dy=0: dx=-1 (2), dx=0 (0,5), dx=+1 (1), then dy=+1 (3)
    assert_eq!(got, vec![4, 2, 0, 5, 1, 3]);
}

/// `clear` empties a built grid, and the grid is reusable for a fresh pass.
#[test]
fn clear_then_rebuild_reflects_only_the_new_particles() {
    let mut g = SpatialGrid::new(fx(1.0), 8);
    g.insert(0, v3(0.5, 0.5, 0.5));
    g.insert(1, v3(0.6, 0.5, 0.5));
    g.build();
    assert_eq!(query(&g, v3(0.5, 0.5, 0.5)), vec![0, 1]);
    g.clear();
    assert!(query(&g, v3(0.5, 0.5, 0.5)).is_empty());
    g.insert(7, v3(0.5, 0.5, 0.5));
    g.build();
    assert_eq!(query(&g, v3(0.5, 0.5, 0.5)), vec![7]);
}

/// Insertions are invisible until `build` (documented two-pass flow).
#[test]
fn inserted_particles_are_not_visible_before_build() {
    let mut g = SpatialGrid::new(fx(1.0), 8);
    g.insert(3, v3(0.5, 0.5, 0.5));
    assert!(query(&g, v3(0.5, 0.5, 0.5)).is_empty());
}

/// `query_neighbors_into` clears the output buffer first (documented).
#[test]
fn query_clears_the_output_buffer_first() {
    let g = build(1.0, 4, &[v3(0.5, 0.5, 0.5)]);
    let mut out = vec![11, 22, 33];
    g.query_neighbors_into(v3(0.5, 0.5, 0.5), Fix128::ONE, &mut out);
    assert_eq!(out, vec![0]);
    // far from anything: still cleared
    let mut out = vec![11];
    g.query_neighbors_into(v3(-1.5, -1.5, -1.5), Fix128::ONE, &mut out);
    assert!(out.is_empty());
}

/// A query at the border does not wrap or index out of range; the neighbourhood is
/// truncated at the grid edge.
#[test]
fn border_queries_truncate_instead_of_wrapping() {
    // dim 4, cell 1: cells x in 0..=3 map to world x in [-2, 2).
    let pts = [v3(-1.9, -1.9, -1.9), v3(1.9, 1.9, 1.9), v3(-1.9, 1.9, -1.9)];
    let g = build(1.0, 4, &pts);
    assert_eq!(query(&g, pts[0]), vec![0]);
    assert_eq!(query(&g, pts[1]), vec![1]);
    assert_eq!(query(&g, pts[2]), vec![2]);
}

/// The doc says `cell_size` is a side length; a zero cell size falls back to 1
/// (module behaviour) rather than dividing by zero.
#[test]
fn zero_cell_size_falls_back_to_unit_cells() {
    let g = SpatialGrid::new(Fix128::ZERO, 4);
    let unit = SpatialGrid::new(Fix128::ONE, 4);
    for p in [v3(0.3, 1.7, -1.2), v3(-2.5, 0.0, 1.0)] {
        assert_eq!(g.hash(p), unit.hash(p));
    }
}

/// A grid created with `grid_dim = 0` should not panic when used (it has no
/// cells; every particle is simply dropped).  KNOWN DEFECT: `hash` calls
/// `clamp(0, gd - 1)` with `gd - 1 = -1`, which panics (min > max).
#[test]
fn zero_dimension_grid_does_not_panic() {
    let r = std::panic::catch_unwind(|| {
        let mut g = SpatialGrid::new(Fix128::ONE, 0);
        g.insert(0, Vec3Fix::ZERO);
        g.build();
        let mut out = Vec::new();
        g.query_neighbors_into(Vec3Fix::ZERO, Fix128::ONE, &mut out);
        out.len()
    });
    assert_eq!(r.ok(), Some(0), "grid_dim = 0 panicked");
}

/// Odd `grid_dim`: the centre cell index is `floor(dim/2)` (dim 5 -> 2), so the
/// origin falls at cell 2 on each axis and world x in [-2, 3) maps to cells 0..=4.
#[test]
fn odd_grid_dimension_centres_on_floor_half() {
    let g = SpatialGrid::new(fx(1.0), 5);
    assert_eq!(g.hash(v3(0.0, 0.0, 0.0)), 2 + 2 * 5 + 2 * 25);
    assert_eq!(g.hash(v3(-2.0, -2.0, -2.0)), 0);
    assert_eq!(g.hash(v3(2.5, 2.5, 2.5)), 4 + 4 * 5 + 4 * 25);
    assert_eq!(
        g.hash(v3(-2.5, 0.0, 0.0)),
        2 * 5 + 2 * 25,
        "floor(-2.5) = -3 clamps to cell 0"
    );
}

/// After `clear` + a smaller rebuild, the total number of indices reachable from the
/// whole grid equals the number inserted in the new pass (no stale pairs).
#[test]
fn clear_leaves_no_stale_indices_across_cells() {
    let mut g = SpatialGrid::new(fx(1.0), 8);
    for i in 0..6 {
        g.insert(i * 70 + 30, v3(i as f64 - 3.0 + 0.5, 0.5, 0.5));
    }
    g.build();
    g.clear();
    g.insert(42, v3(-3.5, -3.5, -3.5));
    g.build();
    let mut all: Vec<usize> = Vec::new();
    for x in -3..4 {
        for y in -3..4 {
            for z in -3..4 {
                for n in query(
                    &g,
                    v3(
                        x as f64 * 3.0 + 0.5,
                        y as f64 * 3.0 + 0.5,
                        z as f64 * 3.0 + 0.5,
                    ),
                ) {
                    all.push(n);
                }
            }
        }
    }
    assert!(
        !all.is_empty() && all.iter().all(|&i| i == 42),
        "stale index leaked: {all:?}"
    );
    assert_eq!(query(&g, v3(-3.5, -3.5, -3.5)), vec![42]);
}
