//! Linear Elastic FEM State Accessors Example
//!
//! Production entry point for the read side of `src/linear_elastic_fem.rs`
//! that no other example touches: `ThermalExpansion::field`,
//! `ElastoplasticState::displacements` / `newton_iterations` and
//! `CorotationalConfig::with_consistent_tangent`.
//!
//! Every value this example prints is checked against a closed-form
//! expectation computed independently of the function under test:
//! - a free body under a uniform temperature rise `ΔT` expands without stress,
//!   `u = α ΔT x`, where `ΔT` is the rise the expansion reports through
//!   `field()` (itself checked against `T − T_ref`)
//! - a bar pulled to a uniform strain `ε` is homogeneous: `u_x = ε x` exactly;
//!   below yield the lateral strain is `−ν ε`, above it radial return with
//!   linear hardening gives `ε_p = (ε − σ_y/E) / (1 + H/E)`,
//!   `σ = σ_y + H ε_p` and a lateral strain of `−ν σ/E − ε_p/2`
//! - the committed Newton count is the sum of the committed increments' counts
//! - a homogeneous deformation `x ↦ F X` prescribed on the boundary of a
//!   cube is an equilibrium of every hyperelastic law, so the free centre node
//!   lands on `(F − I) X_c` with the consistent tangent switched on
//!
//! Run with: `cargo run --example linear_elastic_fem_state_accessors`

use alice_physics::coupled_field::{CoupledField, TemperatureRise};
use alice_physics::hyperelastic::HyperelasticModel;
use alice_physics::linear_elastic_fem::{
    solve_corotational, solve_with_eigenstrain, Axis, BoundaryConditions, CorotationalConfig,
    ElasticMaterial, ElastoplasticConfig, ElastoplasticIncrementRequest, ElastoplasticProblem,
    FemError, SolverConfig, ThermalExpansion,
};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::sdf_fem_mesh::{SdfTetMesh, Tetrahedron};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn close(got: f64, want: f64, tol: f64, what: &str) {
    assert!(
        (got - want).abs() <= tol,
        "{what}: got {got:.12e}, closed form {want:.12e} (tolerance {tol:.1e})"
    );
}

/// Kuhn 6-tet subdivision of an `nx × ny × nz` lattice of cubes of side `h`.
fn kuhn_box(nx: usize, ny: usize, nz: usize, h: f64) -> SdfTetMesh {
    let node = |i: usize, j: usize, k: usize| -> u32 {
        u32::try_from(i + j * (nx + 1) + k * (nx + 1) * (ny + 1)).expect("lattice fits u32")
    };
    let mut mesh = SdfTetMesh::default();
    for k in 0..=nz {
        for j in 0..=ny {
            for i in 0..=nx {
                mesh.vertices.push([
                    (i as f64 * h) as f32,
                    (j as f64 * h) as f32,
                    (k as f64 * h) as f32,
                ]);
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
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                for path in PATHS {
                    let mut step = [0usize; 3];
                    let mut corners = [0u32; 4];
                    corners[0] = node(i, j, k);
                    for (slot, axis) in path.into_iter().enumerate() {
                        step[axis] = 1;
                        corners[slot + 1] = node(i + step[0], j + step[1], k + step[2]);
                    }
                    mesh.tets.push(Tetrahedron { vertices: corners });
                }
            }
        }
    }
    mesh
}

/// Remove the six rigid modes of a body whose corner `0` sits at the origin:
/// `0` fixed in all three axes, `y_node` (on the `y` axis) fixed in `x` and `z`,
/// `z_node` (on the `z` axis) fixed in `x`.
fn pin_rigid_modes(bc: &mut BoundaryConditions, y_node: u32, z_node: u32) {
    bc.fix(0);
    bc.prescribe(y_node, Axis::X, Fix128::ZERO);
    bc.prescribe(y_node, Axis::Z, Fix128::ZERO);
    bc.prescribe(z_node, Axis::X, Fix128::ZERO);
}

// ---------------------------------------------------------------------------
// 1. ThermalExpansion::field: free thermal expansion
// ---------------------------------------------------------------------------

fn free_thermal_expansion() {
    const SIDE: f64 = 4.0;
    const T_REF: i64 = 25;
    const T_ABS: i64 = 65;
    let alpha = Fix128::from_ratio(1, 1024);

    let mesh = kuhn_box(2, 2, 2, SIDE / 2.0);
    let side = Fix128::from_int(SIDE as i64);
    let absolute = CoupledField::try_new_filled(
        3,
        3,
        3,
        (Fix128::ZERO, Fix128::ZERO, Fix128::ZERO),
        (side, side, side),
        Fix128::from_int(T_ABS),
    )
    .expect("a 3^3 grid over the cube is valid");
    let rise = TemperatureRise::from_absolute(&absolute, Fix128::from_int(T_REF));
    let thermal = ThermalExpansion::from_rise(&rise, alpha);

    // The field the expansion reads is the rise, cell by cell.
    let field = thermal.field();
    assert_eq!(field.cell_count(), 27, "field() hands back the 3^3 grid");
    let delta_t = (T_ABS - T_REF) as f64;
    for &v in field.as_slice() {
        close(v.to_f64(), delta_t, 0.0, "ΔT in every cell");
    }

    // Free body: the rigid modes are the only constraints, so the eigenstrain
    // is accommodated without stress and the field is the affine expansion.
    let mut bc = BoundaryConditions::new();
    // (0, 2, 0) is node 6 and (0, 0, 2) is node 18 on the 3x3x3 lattice.
    pin_rigid_modes(&mut bc, 6, 18);
    let material = ElasticMaterial::new(Fix128::from_int(3500), Fix128::from_ratio(35, 100))
        .expect("E > 0 and ν in (-1, 0.5)");
    let out = solve_with_eigenstrain(
        &mesh,
        &material,
        &bc,
        &SolverConfig::default(),
        Some(thermal),
    )
    .expect("a free cube under a uniform eigenstrain is well posed");

    let strain = alpha.to_f64() * delta_t;
    for (p, u) in mesh.vertices.iter().zip(out.displacements.iter()) {
        let sampled = field
            .sample(Vec3Fix::new(
                Fix128::from_f64(f64::from(p[0])),
                Fix128::from_f64(f64::from(p[1])),
                Fix128::from_f64(f64::from(p[2])),
            ))
            .to_f64();
        for a in 0..3 {
            close(
                u[a].to_f64(),
                alpha.to_f64() * sampled * f64::from(p[a]),
                1e-9,
                "free thermal expansion u = α ΔT x",
            );
        }
    }
    // Suppressing the same expansion would cost E ε_th / (1 - 2ν) = 455 MPa;
    // the free body carries six orders less (CG round-off).
    let suppressed = 3500.0 * strain / (1.0 - 2.0 * 0.35);
    for s in &out.element_stress {
        for c in [s.xx, s.yy, s.zz, s.xy, s.yz, s.zx] {
            close(
                c.to_f64(),
                0.0,
                suppressed * 1e-6,
                "free expansion is stress free",
            );
        }
    }
    println!(
        "thermal: ΔT = {delta_t} K, α = 1/1024, ε_th = {strain:.9}, u(4,4,4) = {:.9} mm",
        out.displacements[26][0].to_f64()
    );
}

// ---------------------------------------------------------------------------
// 2. ElastoplasticState: displacements and Newton count across increments
// ---------------------------------------------------------------------------

fn elastoplastic_state() {
    const E: f64 = 1024.0;
    const NU: f64 = 0.25;
    const SIGMA_Y: f64 = 2.0;
    const H: f64 = 1024.0;
    const STRAIN: f64 = 0.005;
    const LENGTH: f64 = 4.0;

    // 2 x 1 x 1 cubes of side 2: [0,4] x [0,2] x [0,2], 12 nodes.
    let mesh = kuhn_box(2, 1, 1, 2.0);
    let node = |i: usize, j: usize, k: usize| -> u32 {
        u32::try_from(i + j * 3 + k * 6).expect("lattice fits u32")
    };
    let mut bc = BoundaryConditions::new();
    for k in 0..=1 {
        for j in 0..=1 {
            bc.prescribe(node(0, j, k), Axis::X, Fix128::ZERO);
            bc.prescribe(node(2, j, k), Axis::X, fx(STRAIN * LENGTH));
        }
    }
    bc.prescribe(node(0, 0, 0), Axis::Y, Fix128::ZERO);
    bc.prescribe(node(0, 0, 0), Axis::Z, Fix128::ZERO);
    bc.prescribe(node(0, 1, 0), Axis::Z, Fix128::ZERO);

    let material = ElasticMaterial::new(fx(E), fx(NU)).expect("E > 0 and ν in (-1, 0.5)");
    let config = ElastoplasticConfig::try_new(
        SolverConfig::default(),
        60,
        Fix128::from_raw(0, 1 << 24),
        fx(SIGMA_Y),
        fx(H),
    )
    .expect("a valid elastoplastic config");
    let problem = ElastoplasticProblem::try_new(&mesh, &material, &bc, &config)
        .expect("the bar is well posed");

    let mut state = problem.virgin_state();
    assert_eq!(
        state.newton_iterations(),
        0,
        "a virgin state has solved nothing"
    );
    let virgin = state.displacements();
    assert_eq!(
        virgin.len(),
        mesh.vertices.len(),
        "one displacement per node"
    );
    assert!(
        virgin.iter().flatten().all(|c| *c == Fix128::ZERO),
        "a virgin state sits at u = 0"
    );

    let check_field = |u: &[[Fix128; 3]], eps: f64, lateral: f64, what: &str| {
        for (p, d) in mesh.vertices.iter().zip(u.iter()) {
            close(
                d[0].to_f64(),
                eps * f64::from(p[0]),
                1e-9,
                &format!("{what}: u_x = ε x"),
            );
            close(
                d[1].to_f64(),
                lateral * f64::from(p[1]),
                1e-9,
                &format!("{what}: u_y = ε_lat y"),
            );
            close(
                d[2].to_f64(),
                lateral * f64::from(p[2]),
                1e-9,
                &format!("{what}: u_z = ε_lat z"),
            );
        }
    };

    // Increment 1: a quarter of the stretch, below the yield strain σ_y/E.
    let eps1 = STRAIN / 4.0;
    assert!(eps1 < SIGMA_Y / E, "increment 1 is meant to stay elastic");
    let inc1 = problem
        .step(
            &state,
            &ElastoplasticIncrementRequest::new(Fix128::from_ratio(1, 4)),
        )
        .expect("increment 1 solves");
    let n1 = inc1.newton_iterations();
    assert!(
        n1 >= 1,
        "an increment that moves the boundary solves at least once"
    );
    inc1.commit(&mut state);
    assert_eq!(
        state.newton_iterations(),
        n1,
        "committed count after one increment"
    );
    check_field(&state.displacements(), eps1, -NU * eps1, "elastic");

    // Increment 2: the full stretch, past yield.
    let inc2 = problem
        .step(&state, &ElastoplasticIncrementRequest::new(Fix128::ONE))
        .expect("increment 2 solves");
    let n2 = inc2.newton_iterations();
    inc2.commit(&mut state);
    assert_eq!(
        state.newton_iterations(),
        n1 + n2,
        "the committed count is the sum of the committed increments"
    );

    let eps_p = (STRAIN - SIGMA_Y / E) / (1.0 + H / E);
    let sigma = SIGMA_Y + H * eps_p;
    let lateral = -NU * sigma / E - eps_p / 2.0;
    check_field(&state.displacements(), STRAIN, lateral, "plastic");
    for &e in &state.equivalent_plastic_strain() {
        close(e.to_f64(), eps_p, 1e-9, "equivalent plastic strain");
    }
    println!(
        "elastoplastic: Newton {n1} + {n2} = {}, ε_p = {eps_p:.9}, σ = {sigma:.6} MPa, \
         ε_lat = {lateral:.9}",
        state.newton_iterations()
    );
}

// ---------------------------------------------------------------------------
// 3. CorotationalConfig::with_consistent_tangent: homogeneous patch
// ---------------------------------------------------------------------------

fn consistent_tangent_patch() {
    const SIDE: f64 = 4.0;
    // A large stretch with shear: F = [[1.25, 0.1, 0], [0, 0.9, 0], [0, 0, 0.95]].
    let f = [[1.25, 0.1, 0.0], [0.0, 0.9, 0.0], [0.0, 0.0, 0.95]];
    let mesh = kuhn_box(2, 2, 2, SIDE / 2.0);
    let mut bc = BoundaryConditions::new();
    let mut centre = None;
    for (v, p) in mesh.vertices.iter().enumerate() {
        let id = u32::try_from(v).expect("fits");
        let on_face = p.iter().any(|c| *c <= 0.0 || f64::from(*c) >= SIDE);
        if !on_face {
            centre = Some((id, *p));
            continue;
        }
        for (a, axis) in [Axis::X, Axis::Y, Axis::Z].into_iter().enumerate() {
            let x = [f64::from(p[0]), f64::from(p[1]), f64::from(p[2])];
            let disp = f[a][0] * x[0] + f[a][1] * x[1] + f[a][2] * x[2] - x[a];
            bc.prescribe(id, axis, fx(disp));
        }
    }
    let (centre, xc) = centre.expect("the 2^3 cube has one interior node");

    let material = ElasticMaterial::new(Fix128::from_int(3500), Fix128::from_ratio(35, 100))
        .expect("E > 0 and ν in (-1, 0.5)");
    let mu = material.lame().1;
    let base = CorotationalConfig::try_new(
        SolverConfig::try_new(200_000, Fix128::from_raw(0, 1 << 34)).expect("valid linear config"),
        64,
        Fix128::from_raw(0, 1 << 34),
        1,
        32,
    )
    .expect("valid co-rotational config")
    .with_hyperelastic(HyperelasticModel::NeoHookean { mu_mpa: mu });
    assert!(
        !base.consistent_tangent(),
        "the default is the modified iteration"
    );
    let config = base.with_consistent_tangent();
    assert!(config.consistent_tangent(), "the builder switches it on");

    // Both iterations must land on the closed form; the consistent tangent is
    // the one that gets there in fewer Newton steps (3 against 11 measured).
    let x = [f64::from(xc[0]), f64::from(xc[1]), f64::from(xc[2])];
    let mut steps = [0u32; 2];
    for (slot, cfg) in [base, config].into_iter().enumerate() {
        let out =
            solve_corotational(&mesh, &material, &bc, &cfg).expect("a homogeneous patch converges");
        let u = out.field.displacements[centre as usize];
        for a in 0..3 {
            let want = f[a][0] * x[0] + f[a][1] * x[1] + f[a][2] * x[2] - x[a];
            close(u[a].to_f64(), want, 1e-7, "centre node on (F - I) X_c");
        }
        steps[slot] = out.newton_iterations;
    }
    assert!(
        steps[1] < steps[0],
        "the consistent tangent took {} Newton steps, the modified iteration {}",
        steps[1],
        steps[0]
    );

    // Without a law there is nothing to differentiate, so the request is
    // refused instead of being ignored.
    let lawless = CorotationalConfig::try_new(
        SolverConfig::default(),
        64,
        Fix128::from_raw(0, 1 << 34),
        1,
        32,
    )
    .expect("valid co-rotational config")
    .with_consistent_tangent();
    assert!(
        matches!(
            solve_corotational(&mesh, &material, &bc, &lawless),
            Err(FemError::InvalidConfig(_))
        ),
        "a consistent tangent without a hyperelastic law must be refused"
    );
    println!(
        "consistent tangent: centre on (F - I) X_c, {} Newton steps against {} modified",
        steps[1], steps[0]
    );
}

fn main() {
    free_thermal_expansion();
    elastoplastic_state();
    consistent_tangent_patch();
    println!("all closed forms hold");
}
