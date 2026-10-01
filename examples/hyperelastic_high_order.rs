//! Finite-strain (hyperelastic) solves on the quadratic and cubic tetrahedra.
//!
//! Stretches a 4 mm cube to 140 % of its length under a Neo-Hookean law and
//! prints the lateral contraction and the axial Cauchy stress each element
//! reports, once with ten-node (P2) elements and once with twenty-node (P3)
//! ones.
//!
//! ⚠️ **The two entry points this drives are the reason this example exists.**
//! `solve_quadratic_hyperelastic` and `solve_cubic_hyperelastic` are library
//! entry points: the crate never calls them itself, so without a caller here
//! they are unreferenced from production code and `scripts/wiring_guard.py`
//! reports them. An example is a better answer than a marker exempting them,
//! because an example is *run* — a change that breaks the signature or the
//! configuration contract fails to build rather than being waved through by a
//! comment.
//!
//! ```text
//! cargo run --example hyperelastic_high_order --features std
//! ```
//!
//! Author: Moroya Sakamoto

#[cfg(not(feature = "std"))]
fn main() {
    eprintln!("this example needs the `std` feature: cargo run --example hyperelastic_high_order --features std");
}

#[cfg(feature = "std")]
fn main() {
    use alice_physics::cubic_elastic_fem::{solve_cubic_hyperelastic, CubicMesh};
    use alice_physics::hyperelastic::HyperelasticModel;
    use alice_physics::linear_elastic_fem::{
        Axis, BoundaryConditions, CorotationalConfig, ElasticMaterial, SolverConfig,
    };
    use alice_physics::math::Fix128;
    use alice_physics::quadratic_elastic_fem::{solve_quadratic_hyperelastic, QuadraticMesh};
    use alice_physics::sdf_fem_mesh::{SdfTetMesh, Tetrahedron};

    const SIDE: f64 = 4.0;
    const STRETCH: f64 = 1.4;

    let fx = Fix128::from_f64;

    // One cube, six tetrahedra (the Kuhn decomposition).
    let mut tets = SdfTetMesh::default();
    for k in 0..2 {
        for j in 0..2 {
            for i in 0..2 {
                tets.vertices.push([
                    i as f32 * SIDE as f32,
                    j as f32 * SIDE as f32,
                    k as f32 * SIDE as f32,
                ]);
            }
        }
    }
    for path in [
        [0usize, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        let mut step = [0usize; 3];
        let mut corners = [0u32; 4];
        for (m, axis) in path.into_iter().enumerate() {
            step[axis] = 1;
            corners[m + 1] = (step[0] + step[1] * 2 + step[2] * 4) as u32;
        }
        tets.tets.push(Tetrahedron { vertices: corners });
    }

    let material = ElasticMaterial::new(fx(10.0), fx(0.45)).expect("E and nu are in range");
    let model = HyperelasticModel::NeoHookean {
        mu_mpa: fx(10.0 / (2.0 * 1.45)),
    };
    let linear = SolverConfig::try_new(500_000, Fix128::from_raw(0, 1 << 34))
        .expect("valid")
        .with_stagnation(2_000, Fix128::from_raw(0, 1 << 54))
        .expect("valid");
    let config = CorotationalConfig::try_new(linear, 80, fx(1.0e-7), 4, 64)
        .expect("valid")
        .with_hyperelastic(model);

    // Uniaxial tension: the two `x` faces driven apart, the `y = 0` and `z = 0`
    // planes held in their own normal direction so the body cannot drift, and
    // the remaining faces left traction free so the material decides how much
    // the section contracts.
    let on = |v: f64, plane: f64| (v - plane).abs() < 5.0e-7;

    let p2 = QuadraticMesh::from_tet_mesh(&tets).expect("well formed");
    let mut bc2 = BoundaryConditions::new();
    for n in 0..p2.node_count() {
        let idx = u32::try_from(n).expect("fits");
        let p = p2.node_position(idx).expect("node in range");
        let (x, y, z) = (p[0].to_f64(), p[1].to_f64(), p[2].to_f64());
        if on(x, 0.0) {
            bc2.prescribe(idx, Axis::X, Fix128::ZERO);
        } else if on(x, SIDE) {
            bc2.prescribe(idx, Axis::X, fx((STRETCH - 1.0) * SIDE));
        }
        if on(y, 0.0) {
            bc2.prescribe(idx, Axis::Y, Fix128::ZERO);
        }
        if on(z, 0.0) {
            bc2.prescribe(idx, Axis::Z, Fix128::ZERO);
        }
    }
    let s2 = solve_quadratic_hyperelastic(&p2, &material, &bc2, &config).expect("converged");

    let p3 = CubicMesh::from_tet_mesh(&tets).expect("well formed");
    let mut bc3 = BoundaryConditions::new();
    for n in 0..p3.node_count() {
        let idx = u32::try_from(n).expect("fits");
        let p = p3.node_position(idx).expect("node in range");
        let (x, y, z) = (p[0].to_f64(), p[1].to_f64(), p[2].to_f64());
        if on(x, 0.0) {
            bc3.prescribe(idx, Axis::X, Fix128::ZERO);
        } else if on(x, SIDE) {
            bc3.prescribe(idx, Axis::X, fx((STRETCH - 1.0) * SIDE));
        }
        if on(y, 0.0) {
            bc3.prescribe(idx, Axis::Y, Fix128::ZERO);
        }
        if on(z, 0.0) {
            bc3.prescribe(idx, Axis::Z, Fix128::ZERO);
        }
    }
    let s3 = solve_cubic_hyperelastic(&p3, &material, &bc3, &config).expect("converged");

    // The lateral stretch is read off the first node on the far `y` face: the
    // deformation is uniform, so any node on that face carries it.
    let mut t2 = f64::NAN;
    for (n, d) in s2.field.displacements.iter().enumerate() {
        let q = p2
            .node_position(u32::try_from(n).expect("fits"))
            .expect("in range");
        if on(q[1].to_f64(), SIDE) {
            t2 = 1.0 + d[1].to_f64() / SIDE;
            break;
        }
    }
    let mut t3 = f64::NAN;
    for (n, d) in s3.field.displacements.iter().enumerate() {
        let q = p3
            .node_position(u32::try_from(n).expect("fits"))
            .expect("in range");
        if on(q[1].to_f64(), SIDE) {
            t3 = 1.0 + d[1].to_f64() / SIDE;
            break;
        }
    }

    println!("Neo-Hookean uniaxial tension of a {SIDE} mm cube to {STRETCH}x");
    println!(
        "  P2  {:>4} nodes  lateral stretch {t2:.6}  sigma_xx {:.6} MPa  ({} Newton steps)",
        p2.node_count(),
        s2.field.element_stress[0].xx.to_f64(),
        s2.newton_iterations
    );
    println!(
        "  P3  {:>4} nodes  lateral stretch {t3:.6}  sigma_xx {:.6} MPa  ({} Newton steps)",
        p3.node_count(),
        s3.field.element_stress[0].xx.to_f64(),
        s3.newton_iterations
    );
    println!(
        "  max von Mises: P2 {:.6} MPa, P3 {:.6} MPa",
        s2.field.max_von_mises_mpa().to_f64(),
        s3.field.max_von_mises_mpa().to_f64()
    );
}
