//! Support reactions of a stretched bar
//!
//! Solves a displacement-driven uniaxial tension problem and reads the force
//! each support carries, which is the one quantity a displacement-driven solve
//! does not reveal on its own: scaling the whole internal force leaves the
//! displacement and the stress exactly where they were, and only the reaction
//! moves with it.
//!
//! ```bash
//! cargo run --example support_reactions --features std
//! ```

use alice_physics::cubic_elastic_fem::{self, CubicMesh};
use alice_physics::linear_elastic_fem::{
    corotational_reactions, reactions, solve, solve_corotational, BoundaryConditions,
    CorotationalConfig, ElasticMaterial, SolverConfig,
};
use alice_physics::math::Fix128;
use alice_physics::quadratic_elastic_fem::{self, QuadraticMesh};
use alice_physics::sdf_fem_mesh::{SdfTetMesh, Tetrahedron};

/// Node index within an `(n+1)³` lattice.
fn node(n: usize, i: usize, j: usize, k: usize) -> u32 {
    u32::try_from(i + j * (n + 1) + k * (n + 1) * (n + 1)).expect("lattice fits u32")
}

/// A cube of side `n·h`, split into Kuhn 6-tet cells.
fn kuhn_cube(n: usize, h: f32) -> SdfTetMesh {
    let mut mesh = SdfTetMesh::default();
    for k in 0..=n {
        for j in 0..=n {
            for i in 0..=n {
                mesh.vertices
                    .push([i as f32 * h, j as f32 * h, k as f32 * h]);
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
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                for path in PATHS {
                    let mut step = [0usize; 3];
                    let mut corners = [0u32; 4];
                    corners[0] = node(n, i, j, k);
                    for (slot, axis) in path.into_iter().enumerate() {
                        step[axis] = 1;
                        corners[slot + 1] = node(n, i + step[0], j + step[1], k + step[2]);
                    }
                    mesh.tets.push(Tetrahedron { vertices: corners });
                }
            }
        }
    }
    mesh
}

fn main() {
    // A 10 mm PLA cube: E = 3500 MPa, ν = 0.35.
    let (n, h) = (2usize, 5.0_f32);
    let mesh = kuhn_cube(n, h);
    let material = ElasticMaterial::new(Fix128::from_int(3500), Fix128::from_ratio(35, 100))
        .expect("E > 0 and ν in (-1, 0.5)");

    // Prescribe the exact uniaxial field u = ε·(x, −νy, −νz) on every boundary
    // node, leaving the single interior node for the solver to find. The field
    // is linear, so P1 tetrahedra reproduce it exactly and every element ends
    // up carrying σ_xx = E·ε with nothing else.
    let strain = Fix128::from_ratio(1, 500); // ε = 0.002, so σ = 7 MPa
    let nu = Fix128::from_ratio(35, 100);
    let interior = node(n, 1, 1, 1);
    let mut boundary = BoundaryConditions::new();
    for (v, p) in mesh.vertices.iter().enumerate() {
        let v = u32::try_from(v).expect("fits");
        if v == interior {
            continue;
        }
        let at = |c: f32| Fix128::from_f32(c);
        boundary.prescribe_all(
            v,
            [
                strain * at(p[0]),
                -(nu * strain * at(p[1])),
                -(nu * strain * at(p[2])),
            ],
        );
    }

    let field = solve(&mesh, &material, &boundary, &SolverConfig::default())
        .expect("the patch test is well posed");
    let support = reactions(&mesh, &material, &boundary, None, &field)
        .expect("the solution has one entry per vertex");

    // Sum the reaction over the x = 0 face. For a uniform σ the closed form is
    // −A₀·σ e_x with A₀ the *reference* area, so this should print −700 N
    // (10 mm × 10 mm × 7 MPa), with the far face carrying the opposite.
    let mut near = Fix128::ZERO;
    let mut far = Fix128::ZERO;
    for (v, p) in mesh.vertices.iter().enumerate() {
        if p[0] <= 0.0 {
            near = near + support[v][0];
        } else if p[0] >= h * n as f32 {
            far = far + support[v][0];
        }
    }

    println!("uniaxial strain      ε   = {}", strain.to_f64());
    println!(
        "axial stress         σ   = {} MPa",
        3500.0 * strain.to_f64()
    );
    println!("x = 0 face reaction  ΣRx = {:.3} N", near.to_f64());
    println!("far  face reaction   ΣRx = {:.3} N", far.to_f64());
    println!(
        "closed form              = ∓{:.3} N  (E·ε·A₀)",
        3500.0 * strain.to_f64() * 100.0
    );

    // A free degree of freedom carries no reaction: that row is the
    // equilibrium equation the solve satisfied, so it is reported as zero.
    println!(
        "interior node reaction   = {:?} N (free, so exactly zero)",
        support[interior as usize].map(Fix128::to_f64)
    );

    // The same problem on the ten-node and twenty-node elements. The affine
    // field is exact on every element, so all three read the same −700 N; the
    // boundary set has to name the edge (and face) nodes on the box surface as
    // well, which `node_position` makes a one-line test.
    let far = h * n as f32;
    let on_surface = |p: [Fix128; 3]| p.iter().any(|c| c.to_f32() <= 0.0 || c.to_f32() >= far);
    let prescribe_surface = |count: usize, at: &dyn Fn(u32) -> Option<[Fix128; 3]>| {
        let mut bc = BoundaryConditions::new();
        for node in 0..u32::try_from(count).expect("fits") {
            let p = at(node).expect("in range");
            if on_surface(p) {
                bc.prescribe_all(
                    node,
                    [strain * p[0], -(nu * strain * p[1]), -(nu * strain * p[2])],
                );
            }
        }
        bc
    };

    let p2 = QuadraticMesh::from_tet_mesh(&mesh).expect("no degenerate cell");
    let p2_bc = prescribe_surface(p2.node_count(), &|i| p2.node_position(i));
    let p2_field =
        quadratic_elastic_fem::solve_quadratic(&p2, &material, &p2_bc, &SolverConfig::default())
            .expect("well posed");
    let p2_support = quadratic_elastic_fem::reactions(&p2, &material, &p2_bc, None, &p2_field)
        .expect("one entry per node");
    let p2_near: f64 = (0..p2.node_count())
        .filter(|&i| p2.node_position(i as u32).expect("in range")[0].to_f32() <= 0.0)
        .map(|i| p2_support[i][0].to_f64())
        .sum();
    println!("P2 x = 0 face reaction  ΣRx = {p2_near:.3} N");

    let p3 = CubicMesh::from_tet_mesh(&mesh).expect("no degenerate cell");
    let p3_bc = prescribe_surface(p3.node_count(), &|i| p3.node_position(i));
    let p3_field = cubic_elastic_fem::solve_cubic(&p3, &material, &p3_bc, &SolverConfig::default())
        .expect("well posed");
    let p3_support = cubic_elastic_fem::reactions(&p3, &material, &p3_bc, None, &p3_field)
        .expect("one entry per node");
    let p3_near: f64 = (0..p3.node_count())
        .filter(|&i| p3.node_position(i as u32).expect("in range")[0].to_f32() <= 0.0)
        .map(|i| p3_support[i][0].to_f64())
        .sum();
    println!("P3 x = 0 face reaction  ΣRx = {p3_near:.3} N");

    // The same read for the co-rotational solver, on the case that formulation
    // exists for: a rigid rotation of the boundary. `R` is extracted per
    // element and the strain measured in that frame, so `RᵀF = I` and every
    // support comes back unloaded — where the small-strain solve above would
    // read the rotation as strain and invent one.
    let (cos, sin) = (Fix128::from_ratio(4, 5), Fix128::from_ratio(3, 5));
    let mut turned = BoundaryConditions::new();
    for (v, p) in mesh.vertices.iter().enumerate() {
        let v = u32::try_from(v).expect("fits");
        if v == interior {
            continue;
        }
        let (x, y) = (Fix128::from_f32(p[0]), Fix128::from_f32(p[1]));
        turned.prescribe_all(
            v,
            [cos * x - sin * y - x, sin * x + cos * y - y, Fix128::ZERO],
        );
    }
    let config = CorotationalConfig::try_new(
        SolverConfig::default(),
        32,
        Fix128::from_ratio(1, 1_000_000),
        1,
        64,
    )
    .expect("every count is positive and the tolerance is in (0, 1)");
    let rotated = solve_corotational(&mesh, &material, &turned, &config)
        .expect("a boundary rotation of 36.87° is within the co-rotational range");
    let turned_support = corotational_reactions(&mesh, &material, &turned, &config, &rotated)
        .expect("the solution has one entry per vertex");
    let worst = turned_support
        .iter()
        .flatten()
        .fold(0.0_f64, |m, v| m.max(v.to_f64().abs()));
    println!("rigid 36.87° turn: worst |R| = {worst:.3e} N (a rotation stores no energy)");
}
