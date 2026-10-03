//! Error-driven adaptive refinement on the quadratic (P2) and cubic (P3)
//! tetrahedra.
//!
//! Solves a smooth problem — `u = (y²/64, 0, 0)` prescribed on the surface of a
//! 2 mm cube, the interior free — and prints, per round, the total error
//! indicator `Σ η_e²` the driver acted on.
//!
//! ⚠️ **The two drivers this calls are the reason this example exists.**
//! `solve_adaptive_quadratic` and `solve_adaptive_cubic` are library entry
//! points that the crate never calls itself, so without a caller here
//! `scripts/wiring_guard.py` reports them as unreferenced from production code.
//! An example is *run*, so a change to the signature or the boundary-condition
//! contract fails to build instead of being waved through by an exemption.
//!
//! ⚠️ The closure receives the **high-order mesh** and prescribes by node
//! position: the edge (and face) nodes on the clamped surface have to be
//! constrained too, and only that mesh knows where they are.
//!
//! ```text
//! cargo run --example adaptive_refinement_high_order --features std
//! ```
//!
//! Author: Moroya Sakamoto

#[cfg(not(feature = "std"))]
fn main() {
    eprintln!("this example needs the `std` feature: cargo run --example adaptive_refinement_high_order --features std");
}

#[cfg(feature = "std")]
fn main() {
    use alice_physics::cubic_elastic_fem::solve_adaptive_cubic;
    use alice_physics::linear_elastic_fem::{
        AdaptiveConfig, BoundaryConditions, ElasticMaterial, SolverConfig,
    };
    use alice_physics::math::Fix128;
    use alice_physics::quadratic_elastic_fem::solve_adaptive_quadratic;
    use alice_physics::sdf_fem_mesh::{SdfTetMesh, Tetrahedron};

    const N: usize = 2;
    let node = |i: usize, j: usize, k: usize| {
        u32::try_from(i + j * (N + 1) + k * (N + 1) * (N + 1)).expect("fits")
    };

    let mut base = SdfTetMesh::default();
    for k in 0..=N {
        for j in 0..=N {
            for i in 0..=N {
                base.vertices.push([i as f32, j as f32, k as f32]);
            }
        }
    }
    const PATHS: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    for k in 0..N {
        for j in 0..N {
            for i in 0..N {
                for path in PATHS {
                    let mut step = [0usize; 3];
                    let mut corners = [0u32; 4];
                    corners[0] = node(i, j, k);
                    for (m, axis) in path.into_iter().enumerate() {
                        step[axis] = 1;
                        corners[m + 1] = node(i + step[0], j + step[1], k + step[2]);
                    }
                    base.tets.push(Tetrahedron { vertices: corners });
                }
            }
        }
    }

    let material = ElasticMaterial::new(Fix128::from_int(1024), Fix128::from_f64(0.25))
        .expect("E > 0 and ν in (-1, 0.5)");
    let extent = N as f64;
    let surface_bc = |pos: &[[f64; 3]]| {
        let mut bc = BoundaryConditions::new();
        for (n, p) in pos.iter().enumerate() {
            if p.iter()
                .any(|&c| c.abs() < 1e-6 || (c - extent).abs() < 1e-6)
            {
                bc.prescribe_all(
                    u32::try_from(n).expect("fits"),
                    [
                        Fix128::from_f64(p[1] * p[1] / 64.0),
                        Fix128::ZERO,
                        Fix128::ZERO,
                    ],
                );
            }
        }
        bc
    };

    let config = AdaptiveConfig::try_new(SolverConfig::default(), Fix128::ONE, 3, 32)
        .expect("valid adaptive config");

    let quadratic = solve_adaptive_quadratic(
        &base,
        &material,
        |m| {
            let pos: Vec<[f64; 3]> = (0..m.node_count())
                .map(|n| {
                    let p = m
                        .node_position(u32::try_from(n).expect("fits"))
                        .expect("in range");
                    [p[0].to_f64(), p[1].to_f64(), p[2].to_f64()]
                })
                .collect();
            surface_bc(&pos)
        },
        &config,
    )
    .expect("quadratic adaptive solve succeeds");
    println!(
        "P2: {} rounds, {} -> {} tets, {} nodes, indicator per round {:?}",
        quadratic.rounds,
        base.tets.len(),
        quadratic.mesh.tets.len(),
        quadratic.high_order.node_count(),
        quadratic
            .total_indicator_history
            .iter()
            .map(|v| v.to_f64())
            .collect::<Vec<_>>()
    );

    let cubic = solve_adaptive_cubic(
        &base,
        &material,
        |m| {
            let pos: Vec<[f64; 3]> = (0..m.node_count())
                .map(|n| {
                    let p = m
                        .node_position(u32::try_from(n).expect("fits"))
                        .expect("in range");
                    [p[0].to_f64(), p[1].to_f64(), p[2].to_f64()]
                })
                .collect();
            surface_bc(&pos)
        },
        &config,
    )
    .expect("cubic adaptive solve succeeds");
    println!(
        "P3: {} rounds, {} -> {} tets, {} nodes, indicator per round {:?}",
        cubic.rounds,
        base.tets.len(),
        cubic.mesh.tets.len(),
        cubic.high_order.node_count(),
        cubic
            .total_indicator_history
            .iter()
            .map(|v| v.to_f64())
            .collect::<Vec<_>>()
    );
}
