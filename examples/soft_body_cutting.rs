//! Production entry point for `alice_physics::soft_body_cut`: `CutPlane`,
//! `CutResult`, `cut_cloth`, and `cut_deformable`.
//!
//! Wiring: `scripts/wiring-baseline.txt` lists all four as `unwired` -- the
//! module's own `#[cfg(all(test, feature = "std"))]` block exercises them,
//! but tests do not count as production callers for the wiring guard, and
//! nothing in `src/` / `examples/` / `benches/` called any of the four
//! before this file existed. This example is that caller.
//!
//! `tests/analytic_soft_body_cut_wiring.rs` holds the closed-form oracles
//! for degenerate (on-plane vertex, plane-misses-mesh) and conservation
//! invariants that this file's diagnostic prints do not repeat.
//!
//! ```bash
//! cargo run --example soft_body_cutting --features std
//! ```
//!
//! Author: Moroya Sakamoto

use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::soft_body_cut::{cut_cloth, cut_deformable, CutPlane, CutResult};

fn main() {
    // ------------------------------------------------------------------
    // 1. cut_cloth -- a 2x2 cloth patch (4 particles) with the 2 grid
    //    edges plus 1 diagonal, cut by the horizontal plane y = 0.
    //
    //    p0 = (0, 2, 0)   p1 = (2, 2, 0)      (both above the plane)
    //    p2 = (0,-2, 0)   p3 = (2,-2, 0)      (both below the plane)
    //
    //    edges: (0,1) top    -- both above  -> no cut
    //           (2,3) bottom -- both below  -> no cut
    //           (0,2) left   -- crosses
    //           (1,3) right  -- crosses
    //           (0,3) diag   -- crosses
    //
    //    Every crossing edge here has one endpoint at y=+2 and the other
    //    at y=-2, so the interpolation parameter is the same exact
    //    dyadic fraction for all three: t = d0 / (d0 - d1) = 2 / 4 = 0.5,
    //    which Fix128 (binary fixed-point) represents with zero rounding
    //    error. The intersection point is therefore the exact componentwise
    //    midpoint of the two endpoints -- computed below by hand, not by
    //    calling cut_cloth a second time.
    // ------------------------------------------------------------------
    let cloth_particles = vec![
        Vec3Fix::from_int(0, 2, 0),  // p0
        Vec3Fix::from_int(2, 2, 0),  // p1
        Vec3Fix::from_int(0, -2, 0), // p2
        Vec3Fix::from_int(2, -2, 0), // p3
    ];
    let cloth_edges = vec![(0, 1), (2, 3), (0, 2), (1, 3), (0, 3)];
    let horizontal_plane = CutPlane {
        point: Vec3Fix::ZERO,
        normal: Vec3Fix::UNIT_Y,
    };

    let cloth_result: CutResult = cut_cloth(&cloth_particles, &cloth_edges, &horizontal_plane);

    // Hand-derived expected values.
    let want_side_a = vec![0usize, 1usize]; // p0, p1 (y >= 0)
    let want_side_b = vec![2usize, 3usize]; // p2, p3 (y < 0)
                                            // midpoint(p0,p2)=(0,0,0); midpoint(p1,p3)=(2,0,0); midpoint(p0,p3)=(1,0,0)
    let want_new_particles = vec![
        Vec3Fix::from_int(0, 0, 0),
        Vec3Fix::from_int(2, 0, 0),
        Vec3Fix::from_int(1, 0, 0),
    ];
    let want_removed = vec![(0usize, 2usize), (1usize, 3usize), (0usize, 3usize)];

    assert_eq!(
        cloth_result.side_a_particles, want_side_a,
        "[soft_body_cut] MISMATCH cut_cloth side_a_particles: got {:?}, want {:?}",
        cloth_result.side_a_particles, want_side_a
    );
    assert_eq!(
        cloth_result.side_b_particles, want_side_b,
        "[soft_body_cut] MISMATCH cut_cloth side_b_particles: got {:?}, want {:?}",
        cloth_result.side_b_particles, want_side_b
    );
    assert_eq!(
        cloth_result.new_particles, want_new_particles,
        "[soft_body_cut] MISMATCH cut_cloth new_particles: got {:?}, want {:?}",
        cloth_result.new_particles, want_new_particles
    );
    assert_eq!(
        cloth_result.removed_constraints, want_removed,
        "[soft_body_cut] MISMATCH cut_cloth removed_constraints: got {:?}, want {:?}",
        cloth_result.removed_constraints, want_removed
    );
    println!(
        "[soft_body_cut] ok cut_cloth(2x2 quad + diagonal, plane y=0): side_a={:?} side_b={:?} \
         new_particles={} removed_constraints={:?}",
        cloth_result.side_a_particles,
        cloth_result.side_b_particles,
        cloth_result.new_particles.len(),
        cloth_result.removed_constraints
    );

    // Particle-count conservation: every original cloth particle must land
    // in exactly one of side_a / side_b (the two sides partition the
    // original particle set; new_particles are additional vertices, not a
    // re-count of the originals).
    assert_eq!(
        cloth_result.side_a_particles.len() + cloth_result.side_b_particles.len(),
        cloth_particles.len(),
        "[soft_body_cut] MISMATCH cut_cloth particle-count conservation: \
         side_a.len() + side_b.len() must equal the original particle count"
    );
    println!(
        "[soft_body_cut] ok cut_cloth particle-count conservation: {} + {} == {}",
        cloth_result.side_a_particles.len(),
        cloth_result.side_b_particles.len(),
        cloth_particles.len()
    );

    // ------------------------------------------------------------------
    // 2. cut_deformable -- a single tetrahedron (4 particles, 6 edges)
    //    with one vertex above the plane y = 0 and the other three below.
    //
    //    t0 = (10, 5, 0)   (apex, above)
    //    t1 = ( 9,-3, 0)   (below)
    //    t2 = (11,-3, 0)   (below)
    //    t3 = (10,-3, 2)   (below)
    //
    //    All three edges from the apex (t0) to the base (t1,t2,t3) cross
    //    the plane; the three base-to-base edges (t1,t2), (t1,t3), (t2,t3)
    //    do not (all strictly below). For every crossing edge here,
    //    d0 = 5, d_other = -3, so denom = 5-(-3) = 8 and
    //    t = d0/denom = 5/8 = 0.625 exactly (a 3-bit binary fraction, so
    //    again zero rounding error in Fix128). The three intersection
    //    points below are hand-interpolated at that exact t, independent
    //    of cut_deformable's own implementation:
    //
    //    (0,1): x = 10 + (9 -10)*0.625  = 10 - 0.625  =  9.375
    //           y =  5 + (-3- 5)*0.625  =  5 - 5       =  0
    //           z =  0 + (0 - 0)*0.625  =  0
    //    (0,2): x = 10 + (11-10)*0.625  = 10 + 0.625   = 10.625
    //           y, z as above           =  0, 0
    //    (0,3): x = 10 + (10-10)*0.625  = 10
    //           y as above              =  0
    //           z =  0 + (2 - 0)*0.625  =  1.25
    // ------------------------------------------------------------------
    let tet_particles = vec![
        Vec3Fix::from_int(10, 5, 0),  // t0 apex, above
        Vec3Fix::from_int(9, -3, 0),  // t1 base, below
        Vec3Fix::from_int(11, -3, 0), // t2 base, below
        Vec3Fix::from_int(10, -3, 2), // t3 base, below
    ];
    let tet_edges = vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

    let tet_result: CutResult = cut_deformable(&tet_particles, &tet_edges, &horizontal_plane);

    let want_tet_side_a = vec![0usize];
    let want_tet_side_b = vec![1usize, 2usize, 3usize];
    let want_tet_new_particles = vec![
        Vec3Fix::new(Fix128::from_ratio(75, 8), Fix128::ZERO, Fix128::ZERO), // 9.375
        Vec3Fix::new(Fix128::from_ratio(85, 8), Fix128::ZERO, Fix128::ZERO), // 10.625
        Vec3Fix::new(Fix128::from_int(10), Fix128::ZERO, Fix128::from_ratio(5, 4)), // (10,0,1.25)
    ];
    let want_tet_removed = vec![(0usize, 1usize), (0usize, 2usize), (0usize, 3usize)];

    assert_eq!(
        tet_result.side_a_particles, want_tet_side_a,
        "[soft_body_cut] MISMATCH cut_deformable side_a_particles: got {:?}, want {:?}",
        tet_result.side_a_particles, want_tet_side_a
    );
    assert_eq!(
        tet_result.side_b_particles, want_tet_side_b,
        "[soft_body_cut] MISMATCH cut_deformable side_b_particles: got {:?}, want {:?}",
        tet_result.side_b_particles, want_tet_side_b
    );
    assert_eq!(
        tet_result.new_particles, want_tet_new_particles,
        "[soft_body_cut] MISMATCH cut_deformable new_particles: got {:?}, want {:?}",
        tet_result.new_particles, want_tet_new_particles
    );
    assert_eq!(
        tet_result.removed_constraints, want_tet_removed,
        "[soft_body_cut] MISMATCH cut_deformable removed_constraints: got {:?}, want {:?}",
        tet_result.removed_constraints, want_tet_removed
    );
    println!(
        "[soft_body_cut] ok cut_deformable(tetrahedron, plane y=0): side_a={:?} side_b={:?} \
         new_particles=[{:.3},{:.3},{:.3}] (x-coords) removed_constraints={:?}",
        tet_result.side_a_particles,
        tet_result.side_b_particles,
        tet_result.new_particles[0].x.to_f64(),
        tet_result.new_particles[1].x.to_f64(),
        tet_result.new_particles[2].x.to_f64(),
        tet_result.removed_constraints
    );

    // Particle-count conservation, same invariant as above.
    assert_eq!(
        tet_result.side_a_particles.len() + tet_result.side_b_particles.len(),
        tet_particles.len(),
        "[soft_body_cut] MISMATCH cut_deformable particle-count conservation: \
         side_a.len() + side_b.len() must equal the original particle count"
    );
    // removed_constraints and new_particles grow 1:1 -- every crossing edge
    // produces exactly one intersection particle, and the denominator
    // (d_i - d_j) can never be zero for a crossing edge: crossing requires
    // one endpoint's signed distance >= 0 and the other's < 0, so the two
    // signed distances can never be equal.
    assert_eq!(
        tet_result.removed_constraints.len(),
        tet_result.new_particles.len(),
        "[soft_body_cut] MISMATCH cut_deformable removed_constraints/new_particles 1:1 invariant"
    );
    println!(
        "[soft_body_cut] ok cut_deformable particle-count conservation: {} + {} == {}; \
         removed_constraints.len() == new_particles.len() == {}",
        tet_result.side_a_particles.len(),
        tet_result.side_b_particles.len(),
        tet_particles.len(),
        tet_result.removed_constraints.len()
    );

    println!("[soft_body_cut] all checks passed");
}
