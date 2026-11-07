//! A column standing on its base, released under its own weight, advanced by the
//! co-rotational Newmark solver: the sag oscillates about the static value with the period of
//! the column's first axial mode. Then the same column with no base fix standing on the ground
//! plane (every bottom vertex held, frictionless), and rotated as a whole by 36.87°: the
//! co-rotational element sees no strain in a rotation, so nothing moves.
//!
//! Units are the finite element module's: mm, N, MPa, tonne (density in t/mm³).
//!
//! ```text
//! cargo run --example dynamic_fem_corotational
//! ```

use alice_physics_core::damping_rayleigh::RayleighCoefficients;
use alice_physics_core::dynamic_fem::{CorotationalTransientSolver, DynamicsConfig, MassLumping};
use alice_physics_core::linear_elastic_fem::{
    Axis, BoundaryConditions, ElasticMaterial, SolverConfig,
};
use alice_physics_core::math::Fix128;
use alice_physics_core::sdf_fem_mesh::{SdfTetMesh, Tetrahedron};

/// a `w × h × w` mm column of `cell` mm cubes, each split into six Kuhn tetrahedra
fn column(cell: f32, w: f32, h: f32) -> SdfTetMesh {
    let nx = (w / cell).round() as usize;
    let ny = (h / cell).round() as usize;
    let at = |i: usize, j: usize, k: usize| ((i * (ny + 1) + j) * (nx + 1) + k) as u32;
    let mut vertices = Vec::new();
    for i in 0..=nx {
        for j in 0..=ny {
            for k in 0..=nx {
                vertices.push([
                    i as f32 * cell - w / 2.0,
                    j as f32 * cell,
                    k as f32 * cell - w / 2.0,
                ]);
            }
        }
    }
    let paths = [
        [(1, 0, 0), (1, 1, 0)],
        [(1, 0, 0), (1, 0, 1)],
        [(0, 1, 0), (1, 1, 0)],
        [(0, 1, 0), (0, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, 1), (0, 1, 1)],
    ];
    let mut tets = Vec::new();
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nx {
                for [(a0, a1, a2), (b0, b1, b2)] in paths {
                    tets.push(Tetrahedron {
                        vertices: [
                            at(i, j, k),
                            at(i + a0, j + a1, k + a2),
                            at(i + b0, j + b1, k + b2),
                            at(i + 1, j + 1, k + 1),
                        ],
                    });
                }
            }
        }
    }
    SdfTetMesh { vertices, tets }
}

fn main() {
    // 1.2725 MPa (G = 0.427 MPa, ν = 0.49: an elastomer), 1.1e-9 t/mm³
    let material = ElasticMaterial::new(Fix128::from_f64(1.2725), Fix128::from_f64(0.49))
        .expect("E > 0, ν in range");
    let mesh = column(12.5, 25.0, 200.0);
    let n = mesh.vertices.len();
    // lumped masses in tonnes: a quarter of each tetrahedron's mass per vertex
    let mut mass = vec![0.0f64; n];
    for t in &mesh.tets {
        let p: Vec<[f64; 3]> = t
            .vertices
            .iter()
            .map(|&i| mesh.vertices[i as usize].map(f64::from))
            .collect();
        let e = |a: usize, b: usize| [p[b][0] - p[a][0], p[b][1] - p[a][1], p[b][2] - p[a][2]];
        let (ab, ac, ad) = (e(0, 1), e(0, 2), e(0, 3));
        let cross = [
            ac[1] * ad[2] - ac[2] * ad[1],
            ac[2] * ad[0] - ac[0] * ad[2],
            ac[0] * ad[1] - ac[1] * ad[0],
        ];
        let volume = (ab[0] * cross[0] + ab[1] * cross[1] + ab[2] * cross[2]).abs() / 6.0;
        for &i in &t.vertices {
            mass[i as usize] += 1.1e-9 * volume / 4.0;
        }
    }
    let base: Vec<usize> = (0..n).filter(|&i| mesh.vertices[i][1] == 0.0).collect();
    let top: Vec<usize> = (0..n).filter(|&i| mesh.vertices[i][1] == 200.0).collect();

    // under gravity (9810 mm/s²)
    let mut bc = BoundaryConditions::new();
    for &i in &base {
        bc.fix(i as u32);
    }
    for (i, m) in mass.iter().enumerate() {
        bc.add_load(i as u32, Axis::Y, Fix128::from_f64(-m * 9810.0));
    }
    let dt = 1.0 / 2000.0;
    let dynamics = DynamicsConfig::try_new(
        Fix128::from_f64(1.1e-9),
        Fix128::from_f64(dt),
        MassLumping::Lumped,
    )
    .expect("dt > 0");
    let config =
        SolverConfig::try_new(20_000, Fix128::from_f64(1e-7)).expect("budget and tolerance");
    let mut solver = CorotationalTransientSolver::new(&mesh, &material, &bc, &dynamics, &config)
        .expect("the column solves");
    // Rayleigh damping, 5 % of critical on the first axial mode (about 282 rad/s): β = 2ζ/ω
    solver.set_damping(Some(RayleighCoefficients {
        alpha: Fix128::ZERO,
        beta: Fix128::from_f64(2.0 * 0.05 / 282.0),
    }));
    println!(
        "column 25 × 200 × 25 mm, {} vertices, {} tetrahedra, dt {dt} s, 5 % damping on the first mode",
        n,
        mesh.tets.len()
    );
    for step in 1..=160 {
        let iterations = solver.step().expect("the step converges");
        if step % 20 == 0 {
            let u = solver.displacements();
            let sag: f64 =
                top.iter().map(|&i| -u[3 * i + 1].to_f64()).sum::<f64>() / top.len() as f64;
            println!(
                "t = {:.4} s: top sag {sag:.5} mm ({iterations} conjugate gradient iterations)",
                step as f64 * dt
            );
        }
    }

    // the same column with no base fix, standing on the ground plane y = 0 under gravity
    let mut bc = BoundaryConditions::new();
    for (i, m) in mass.iter().enumerate() {
        bc.add_load(i as u32, Axis::Y, Fix128::from_f64(-m * 9810.0));
    }
    let mut solver = CorotationalTransientSolver::new(&mesh, &material, &bc, &dynamics, &config)
        .expect("the column solves");
    solver.set_damping(Some(RayleighCoefficients {
        alpha: Fix128::ZERO,
        beta: Fix128::from_f64(2.0 * 0.2 / 282.0),
    }));
    solver.set_ground(Some(Fix128::ZERO));
    for _ in 0..400 {
        solver.step().expect("the step converges");
    }
    let held = solver.on_ground().iter().filter(|&&h| h).count();
    let u = solver.displacements();
    let sag: f64 = top
        .iter()
        .map(|&i| 200.0 - (f64::from(mesh.vertices[i][1]) + u[3 * i + 1].to_f64()))
        .sum::<f64>()
        / top.len() as f64;
    println!("standing on the ground, no base fix: {held} of {} bottom vertices held, top sag {sag:.5} mm", base.len());

    // rotated as a whole by 36.87° about the base centre, no load: nothing moves
    // the 3-4-5 rotation (36.87°): sine 0.6 and cosine 0.8 exactly, no transcendental call
    let (s, c) = (0.6f64, 0.8f64);
    let rotated: Vec<[f64; 3]> = mesh
        .vertices
        .iter()
        .map(|v| {
            let (x, y) = (f64::from(v[0]), f64::from(v[1]));
            [
                c * x - s * y - f64::from(v[0]),
                s * x + c * y - f64::from(v[1]),
                0.0,
            ]
        })
        .collect();
    let mut bc = BoundaryConditions::new();
    for &i in &base {
        bc.prescribe_all(i as u32, rotated[i].map(Fix128::from_f64));
    }
    let mut solver = CorotationalTransientSolver::new(&mesh, &material, &bc, &dynamics, &config)
        .expect("the rotated column solves");
    solver
        .set_displacements(
            &rotated
                .iter()
                .flatten()
                .map(|x| Fix128::from_f64(*x))
                .collect::<Vec<_>>(),
        )
        .expect("the rotated state has a polar factor");
    for _ in 0..40 {
        solver.step().expect("the step converges");
    }
    let u = solver.displacements();
    let worst = (0..n)
        .map(|i| {
            (0..3)
                .map(|k| (u[3 * i + k].to_f64() - rotated[i][k]).abs())
                .fold(0.0, f64::max)
        })
        .fold(0.0, f64::max);
    println!("rotated by 36.87° and released: the largest move after 40 steps is {worst:.3e} mm");
}
