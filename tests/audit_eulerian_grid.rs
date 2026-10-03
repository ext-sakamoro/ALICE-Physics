//! Audit oracles for `eulerian_grid` (claims of the public items).
//!
//! Every expected value below is derived from the doc of the item under test
//! plus the MAC geometry (face positions in half-cell units) or a hand
//! computation, never from the implementation's own helper. Dyadic data
//! (`dx = 1/2` or `1/4`, small integer coefficients) keeps `Fix128` exact so
//! every assertion is `assert_eq!`.
//!
//! A test marked `#[ignore = "known defect: AUD-..."]` is a red oracle kept
//! for the repair pass; run it with `--ignored` to see the failure.

#![cfg(feature = "std")]

use alice_physics::eulerian_grid::{g2p_velocity, FaceBc, MacGrid};
use alice_physics::math::{Fix128, Vec3Fix};

fn q(num: i64, den: i64) -> Fix128 {
    Fix128::from_ratio(num, den)
}

fn v3(x: Fix128, y: Fix128, z: Fix128) -> Vec3Fix {
    Vec3Fix::new(x, y, z)
}

/// Deterministic generator for the random boundary patterns.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn random_bc(rng: &mut Lcg) -> FaceBc {
    match rng.below(8) {
        0 => FaceBc::Fluid,
        1 => FaceBc::Wall {
            velocity: Vec3Fix::ZERO,
        },
        2 => FaceBc::Wall {
            velocity: v3(q(1, 2), q(-1, 4), q(1, 8)),
        },
        3 => FaceBc::Wall {
            velocity: v3(q(-3, 4), q(1, 2), q(5, 8)),
        },
        4 => FaceBc::SlipWall,
        5 => FaceBc::Inflow {
            normal_velocity: q(3, 8),
        },
        6 => FaceBc::Outflow,
        _ => FaceBc::Wall {
            velocity: v3(q(1, 4), q(1, 4), q(-1, 2)),
        },
    }
}

/// Number of faces of component `comp` along `axis`.
fn face_extent(g: &MacGrid, comp: usize, axis: usize) -> usize {
    let n = [g.nx, g.ny, g.nz][axis];
    if axis == comp {
        n + 1
    } else {
        n
    }
}

fn bc_at(g: &MacGrid, comp: usize, i: [usize; 3]) -> FaceBc {
    match comp {
        0 => g.u_bc(i[0], i[1], i[2]),
        1 => g.v_bc(i[0], i[1], i[2]),
        _ => g.w_bc(i[0], i[1], i[2]),
    }
}

fn set_bc_at(g: &mut MacGrid, comp: usize, i: [usize; 3], bc: FaceBc) {
    match comp {
        0 => g.set_u_bc(i[0], i[1], i[2], bc),
        1 => g.set_v_bc(i[0], i[1], i[2], bc),
        _ => g.set_w_bc(i[0], i[1], i[2], bc),
    }
}

fn val_at(g: &MacGrid, comp: usize, i: [usize; 3]) -> Fix128 {
    match comp {
        0 => g.u(i[0], i[1], i[2]),
        1 => g.v(i[0], i[1], i[2]),
        _ => g.w(i[0], i[1], i[2]),
    }
}

fn set_val_at(g: &mut MacGrid, comp: usize, i: [usize; 3], x: Fix128) {
    let ix = match comp {
        0 => g.idx_u_for_test(i),
        1 => g.idx_v_for_test(i),
        _ => g.idx_w_for_test(i),
    };
    match comp {
        0 => g.u[ix] = x,
        1 => g.v[ix] = x,
        _ => g.w[ix] = x,
    }
}

/// Row-major face index, spelled out from the documented shapes
/// `(nx+1) x ny x nz`, `nx x (ny+1) x nz`, `nx x ny x (nz+1)`.
trait FaceIndex {
    fn idx_u_for_test(&self, i: [usize; 3]) -> usize;
    fn idx_v_for_test(&self, i: [usize; 3]) -> usize;
    fn idx_w_for_test(&self, i: [usize; 3]) -> usize;
}

impl FaceIndex for MacGrid {
    fn idx_u_for_test(&self, i: [usize; 3]) -> usize {
        i[0] + (self.nx + 1) * (i[1] + self.ny * i[2])
    }
    fn idx_v_for_test(&self, i: [usize; 3]) -> usize {
        i[0] + self.nx * (i[1] + (self.ny + 1) * i[2])
    }
    fn idx_w_for_test(&self, i: [usize; 3]) -> usize {
        i[0] + self.nx * (i[1] + self.ny * i[2])
    }
}

fn all_faces(g: &MacGrid, comp: usize) -> Vec<[usize; 3]> {
    let e = [
        face_extent(g, comp, 0),
        face_extent(g, comp, 1),
        face_extent(g, comp, 2),
    ];
    let mut out = Vec::new();
    for k in 0..e[2] {
        for j in 0..e[1] {
            for i in 0..e[0] {
                out.push([i, j, k]);
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// FaceBc predicates
// ---------------------------------------------------------------------------

/// Oracle A-1: truth table of the four `FaceBc` predicates, read off the
/// doc of each variant (walls: Wall and SlipWall; inflow: Inflow only;
/// blocks_pressure: both walls and Inflow; no_slip_velocity: Wall only).
#[test]
fn face_bc_predicates_follow_the_documented_truth_table() {
    let vel = v3(q(1, 2), q(-1, 4), q(1, 8));
    let rows: [(FaceBc, bool, bool, bool, Option<Vec3Fix>); 6] = [
        (FaceBc::Fluid, false, false, false, None),
        (
            FaceBc::Wall {
                velocity: Vec3Fix::ZERO,
            },
            true,
            false,
            true,
            Some(Vec3Fix::ZERO),
        ),
        (FaceBc::Wall { velocity: vel }, true, false, true, Some(vel)),
        (FaceBc::SlipWall, true, false, true, None),
        (
            FaceBc::Inflow {
                normal_velocity: q(3, 8),
            },
            false,
            true,
            true,
            None,
        ),
        (FaceBc::Outflow, false, false, false, None),
    ];
    for (bc, wall, inflow, blocks, slip) in rows {
        assert_eq!(bc.is_wall(), wall, "is_wall of {bc:?}");
        assert_eq!(bc.is_inflow(), inflow, "is_inflow of {bc:?}");
        assert_eq!(bc.blocks_pressure(), blocks, "blocks_pressure of {bc:?}");
        assert_eq!(bc.no_slip_velocity(), slip, "no_slip_velocity of {bc:?}");
    }
    assert_eq!(FaceBc::default(), FaceBc::Fluid);
}

// ---------------------------------------------------------------------------
// MacGrid construction and out-of-range rules
// ---------------------------------------------------------------------------

/// Oracle A-2: `MacGrid::new` shapes, zero initialisation, all faces fluid.
#[test]
fn new_grid_has_the_documented_shapes_and_is_fluid_and_zero() {
    let g = MacGrid::new(4, 3, 2, q(1, 2));
    assert_eq!(g.u.len(), 5 * 3 * 2);
    assert_eq!(g.v.len(), 4 * 4 * 2);
    assert_eq!(g.w.len(), 4 * 3 * 3);
    assert_eq!(g.pressure.len(), 4 * 3 * 2);
    assert_eq!(g.u_solid.len(), g.u.len());
    assert_eq!(g.v_solid.len(), g.v.len());
    assert_eq!(g.w_solid.len(), g.w.len());
    for comp in 0..3 {
        for f in all_faces(&g, comp) {
            assert_eq!(val_at(&g, comp, f), Fix128::ZERO);
            assert_eq!(bc_at(&g, comp, f), FaceBc::Fluid);
        }
    }
    assert!(g.u.iter().chain(&g.v).chain(&g.w).all(|x| x.is_zero()));
    assert!(g.pressure.iter().all(|x| x.is_zero()));
    assert!(!g
        .u_solid
        .iter()
        .chain(&g.v_solid)
        .chain(&g.w_solid)
        .any(|&s| s));
}

/// Oracle A-3: "Out of range returns 0 / false / Fluid; out-of-range
/// setters are ignored". Every in-range element is non-zero and walled so an
/// out-of-range read that aliases into the next row (`idx = i + stride*...`)
/// is visible, and the last valid index on every axis is checked as in range.
#[test]
fn out_of_range_reads_are_neutral_and_out_of_range_writes_are_ignored() {
    let (nx, ny, nz) = (3usize, 2usize, 2usize);
    let mut g = MacGrid::new(nx, ny, nz, q(1, 2));
    let wall = FaceBc::Wall {
        velocity: v3(q(1, 2), q(1, 2), q(1, 2)),
    };
    for comp in 0..3 {
        for f in all_faces(&g, comp) {
            set_bc_at(&mut g, comp, f, wall);
            set_val_at(&mut g, comp, f, Fix128::ONE);
        }
    }
    for p in g.pressure.iter_mut() {
        *p = Fix128::ONE;
    }
    let snapshot = g.clone();
    // The last valid index on each axis is in range.
    assert_eq!(g.u(nx, ny - 1, nz - 1), Fix128::ONE);
    assert_eq!(g.v(nx - 1, ny, nz - 1), Fix128::ONE);
    assert_eq!(g.w(nx - 1, ny - 1, nz), Fix128::ONE);
    assert_eq!(g.pressure(nx - 1, ny - 1, nz - 1), Fix128::ONE);
    assert!(g.is_u_solid(nx, ny - 1, nz - 1));
    assert!(g.is_v_solid(nx - 1, ny, nz - 1));
    assert!(g.is_w_solid(nx - 1, ny - 1, nz));
    assert_eq!(g.u_bc(nx, ny - 1, nz - 1), wall);
    assert_eq!(g.v_bc(nx - 1, ny, nz - 1), wall);
    assert_eq!(g.w_bc(nx - 1, ny - 1, nz), wall);
    // One past the end, on each axis in turn.
    let big = 1000;
    for (i, j, k) in [
        (nx + 1, 0, 0),
        (0, ny, 0),
        (0, 0, nz),
        (big, big, big),
        (nx + 1, ny - 1, nz - 1),
    ] {
        assert_eq!(g.u(i, j, k), Fix128::ZERO, "u({i},{j},{k})");
        assert!(!g.is_u_solid(i, j, k), "is_u_solid({i},{j},{k})");
        assert_eq!(g.u_bc(i, j, k), FaceBc::Fluid, "u_bc({i},{j},{k})");
    }
    for (i, j, k) in [
        (nx, 0, 0),
        (0, ny + 1, 0),
        (0, 0, nz),
        (big, big, big),
        (nx - 1, ny + 1, nz - 1),
    ] {
        assert_eq!(g.v(i, j, k), Fix128::ZERO, "v({i},{j},{k})");
        assert!(!g.is_v_solid(i, j, k), "is_v_solid({i},{j},{k})");
        assert_eq!(g.v_bc(i, j, k), FaceBc::Fluid, "v_bc({i},{j},{k})");
    }
    for (i, j, k) in [
        (nx, 0, 0),
        (0, ny, 0),
        (0, 0, nz + 1),
        (big, big, big),
        (nx - 1, ny - 1, nz + 1),
    ] {
        assert_eq!(g.w(i, j, k), Fix128::ZERO, "w({i},{j},{k})");
        assert!(!g.is_w_solid(i, j, k), "is_w_solid({i},{j},{k})");
        assert_eq!(g.w_bc(i, j, k), FaceBc::Fluid, "w_bc({i},{j},{k})");
    }
    for (i, j, k) in [(nx, 0, 0), (0, ny, 0), (0, 0, nz), (big, big, big)] {
        assert_eq!(g.pressure(i, j, k), Fix128::ZERO, "pressure({i},{j},{k})");
    }
    // Out-of-range writes change nothing at all.
    let clear = FaceBc::Fluid;
    g.set_u_bc(nx + 1, 0, 0, clear);
    g.set_u_bc(0, ny, 0, clear);
    g.set_u_bc(0, 0, nz, clear);
    g.set_v_bc(nx, 0, 0, clear);
    g.set_v_bc(0, ny + 1, 0, clear);
    g.set_v_bc(0, 0, nz, clear);
    g.set_w_bc(nx, 0, 0, clear);
    g.set_w_bc(0, ny, 0, clear);
    g.set_w_bc(0, 0, nz + 1, clear);
    g.set_u_solid(nx + 1, 0, 0, false);
    g.set_v_solid(0, ny + 1, 0, false);
    g.set_w_solid(0, 0, nz + 1, false);
    assert_eq!(g.u_solid, snapshot.u_solid);
    assert_eq!(g.v_solid, snapshot.v_solid);
    assert_eq!(g.w_solid, snapshot.w_solid);
    for comp in 0..3 {
        for f in all_faces(&g, comp) {
            assert_eq!(bc_at(&g, comp, f), wall, "comp {comp} face {f:?}");
        }
    }
}

/// Oracle A-4: every condition on every axis round-trips and the legacy
/// solid flag equals `is_wall()`, at the corner faces of the grid (the
/// largest valid index on each axis), where an off-by-one in the range check
/// or in the face index would alias or drop the write.
#[test]
fn face_conditions_round_trip_at_the_corner_faces() {
    let mut rng = Lcg(7);
    for (nx, ny, nz) in [(1, 1, 1), (3, 2, 2), (2, 3, 1)] {
        let mut g = MacGrid::new(nx, ny, nz, q(1, 2));
        let mut expected: Vec<(usize, [usize; 3], FaceBc)> = Vec::new();
        for comp in 0..3 {
            for f in all_faces(&g, comp) {
                let bc = random_bc(&mut rng);
                set_bc_at(&mut g, comp, f, bc);
                expected.push((comp, f, bc));
            }
        }
        for (comp, f, bc) in expected {
            assert_eq!(
                bc_at(&g, comp, f),
                bc,
                "grid {nx}x{ny}x{nz} comp {comp} {f:?}"
            );
            let solid = match comp {
                0 => g.is_u_solid(f[0], f[1], f[2]),
                1 => g.is_v_solid(f[0], f[1], f[2]),
                _ => g.is_w_solid(f[0], f[1], f[2]),
            };
            assert_eq!(solid, bc.is_wall(), "solid flag of {bc:?}");
        }
    }
}

// ---------------------------------------------------------------------------
// set_closed_box_walls
// ---------------------------------------------------------------------------

/// Oracle A-5: `set_closed_box_walls` marks exactly the two boundary layers
/// of each component's own axis, as `Wall` at rest, except an axis one cell
/// thick which gets `SlipWall`; every interior face is untouched (so an
/// obstacle set beforehand survives).
#[test]
fn closed_box_marks_exactly_the_boundary_layers() {
    let rest = FaceBc::Wall {
        velocity: Vec3Fix::ZERO,
    };
    for (nx, ny, nz) in [(4, 3, 2), (4, 3, 1), (1, 3, 2), (3, 1, 2), (1, 1, 1)] {
        let mut g = MacGrid::new(nx, ny, nz, q(1, 2));
        // An interior obstacle on each component, set beforehand.
        let n = [nx, ny, nz];
        let obstacle = [1usize.min(nx), 1usize.min(ny), 1usize.min(nz)];
        let mut obstacles = Vec::new();
        for comp in 0..3 {
            // interior along its own axis: needs n >= 2 on that axis.
            if n[comp] >= 2 {
                let mut f = [0usize; 3];
                f[comp] = obstacle[comp];
                obstacles.push((comp, f));
                match comp {
                    0 => g.set_u_solid(f[0], f[1], f[2], true),
                    1 => g.set_v_solid(f[0], f[1], f[2], true),
                    _ => g.set_w_solid(f[0], f[1], f[2], true),
                }
            }
        }
        g.set_closed_box_walls();
        for comp in 0..3 {
            let want_edge = if n[comp] == 1 { FaceBc::SlipWall } else { rest };
            for f in all_faces(&g, comp) {
                let on_edge = f[comp] == 0 || f[comp] == n[comp];
                let got = bc_at(&g, comp, f);
                if on_edge {
                    assert_eq!(got, want_edge, "{nx}x{ny}x{nz} comp {comp} edge {f:?}");
                } else if obstacles.contains(&(comp, f)) {
                    assert_eq!(got, rest, "obstacle {f:?} must survive");
                } else {
                    assert_eq!(
                        got,
                        FaceBc::Fluid,
                        "{nx}x{ny}x{nz} comp {comp} interior {f:?}"
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// enforce_solid_faces / enforce_face_boundaries
// ---------------------------------------------------------------------------

fn random_grid(seed: u64, nx: usize, ny: usize, nz: usize) -> MacGrid {
    let mut rng = Lcg(seed);
    let mut g = MacGrid::new(nx, ny, nz, q(1, 2));
    for comp in 0..3 {
        for f in all_faces(&g, comp) {
            let bc = random_bc(&mut rng);
            set_bc_at(&mut g, comp, f, bc);
            // distinct non-zero dyadic velocity on every face
            let x = Fix128::from_int(1 + rng.below(7) as i64) * q(1, 8);
            set_val_at(&mut g, comp, f, x);
        }
    }
    g
}

/// Oracle A-6: `enforce_solid_faces` zeroes the faces that are walls (Wall,
/// SlipWall) and nothing else: Inflow, Outflow and Fluid keep their value
/// ("Walls only").
#[test]
fn enforce_solid_faces_zeroes_walls_only() {
    for seed in 0..6 {
        let before = random_grid(seed, 3, 2, 3);
        let mut g = before.clone();
        g.enforce_solid_faces();
        for comp in 0..3 {
            for f in all_faces(&g, comp) {
                let bc = bc_at(&before, comp, f);
                let want = if bc.is_wall() {
                    Fix128::ZERO
                } else {
                    val_at(&before, comp, f)
                };
                assert_eq!(
                    val_at(&g, comp, f),
                    want,
                    "seed {seed} comp {comp} {f:?} {bc:?}"
                );
            }
        }
    }
}

/// Oracle A-7: `enforce_face_boundaries`: walls to 0, Inflow to its value,
/// Fluid untouched, Outflow takes "the value of the face next to it" along
/// the face normal. Patterns avoid two Outflow faces next to each other so
/// that the neighbour value is unambiguous, and the expectation takes the
/// neighbour's *enforced* value (the face a caller will read afterwards).
#[test]
fn enforce_face_boundaries_imposes_every_condition() {
    for seed in 0..40 {
        let mut before = random_grid(seed, 3, 3, 3);
        // No two Outflow faces adjacent along their normal, and an Outflow on
        // the low (index 0) layer only borders a plain fluid face: see
        // `outflow_on_the_low_layer_reads_a_stale_neighbour` for the case this
        // pattern leaves out.
        for comp in 0..3 {
            for f in all_faces(&before, comp) {
                if bc_at(&before, comp, f) == FaceBc::Outflow {
                    let mut nb = f;
                    nb[comp] = if f[comp] > 0 {
                        f[comp] - 1
                    } else {
                        f[comp] + 1
                    };
                    let nb_bc = bc_at(&before, comp, nb);
                    if nb_bc == FaceBc::Outflow || (f[comp] == 0 && nb_bc != FaceBc::Fluid) {
                        set_bc_at(&mut before, comp, nb, FaceBc::Fluid);
                    }
                }
            }
        }
        let mut g = before.clone();
        g.enforce_face_boundaries();
        for comp in 0..3 {
            let enforced = |f: [usize; 3]| -> Fix128 {
                match bc_at(&before, comp, f) {
                    FaceBc::Fluid | FaceBc::Outflow => val_at(&before, comp, f),
                    FaceBc::Inflow { normal_velocity } => normal_velocity,
                    _ => Fix128::ZERO,
                }
            };
            for f in all_faces(&g, comp) {
                let bc = bc_at(&before, comp, f);
                let want = match bc {
                    FaceBc::Outflow => {
                        let mut nb = f;
                        nb[comp] = if f[comp] > 0 {
                            f[comp] - 1
                        } else {
                            f[comp] + 1
                        };
                        enforced(nb)
                    }
                    _ => enforced(f),
                };
                assert_eq!(
                    val_at(&g, comp, f),
                    want,
                    "seed {seed} comp {comp} face {f:?} {bc:?}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// *_wall_across_*
// ---------------------------------------------------------------------------

/// The wall velocity the doc describes, derived from MAC geometry.
///
/// A face of component `c` at index `idx` sits at half-cell coordinate
/// `2 idx + 1` on every axis except its own, where it is `2 idx`. Looking one
/// cell away along axis `a`, the shared plane passes through that face's
/// position shifted by half a cell along `a`; the two faces of component `a`
/// flanking that point along `c` are the ones that must both be no-slip walls
/// for the mirror to cross a wall. A flanking face outside the grid does not
/// veto; if there is none at all the answer is `None`.
fn wall_across_model(
    g: &MacGrid,
    c: usize,
    idx: [usize; 3],
    a: usize,
    fwd: bool,
) -> Option<Vec3Fix> {
    let mut m = [0i64; 3];
    for ax in 0..3 {
        m[ax] = 2 * idx[ax] as i64 + i64::from(ax != c);
    }
    m[a] += if fwd { 1 } else { -1 };
    let mut found: Vec<Vec3Fix> = Vec::new();
    for d in [-1i64, 1] {
        let mut p = m;
        p[c] += d;
        // index of the component-`a` face at half-cell position `p`
        let mut ix = [0i64; 3];
        let mut inside = true;
        for ax in 0..3 {
            let r = p[ax] - i64::from(ax != a);
            assert!(r % 2 == 0, "model geometry: position is not on an {a}-face");
            ix[ax] = r / 2;
            let max = face_extent(g, a, ax) as i64 - 1;
            if ix[ax] < 0 || ix[ax] > max {
                inside = false;
            }
        }
        if !inside {
            continue;
        }
        let f = [ix[0] as usize, ix[1] as usize, ix[2] as usize];
        match bc_at(g, a, f) {
            FaceBc::Wall { velocity } => found.push(velocity),
            _ => return None,
        }
    }
    match found.len() {
        0 => None,
        1 => Some(found[0]),
        _ => Some(v3(
            (found[0].x + found[1].x) * q(1, 2),
            (found[0].y + found[1].y) * q(1, 2),
            (found[0].z + found[1].z) * q(1, 2),
        )),
    }
}

fn call_wall_across(g: &MacGrid, c: usize, a: usize, i: [usize; 3], fwd: bool) -> Option<Vec3Fix> {
    match (c, a) {
        (0, 1) => g.u_wall_across_y(i[0], i[1], i[2], fwd),
        (0, 2) => g.u_wall_across_z(i[0], i[1], i[2], fwd),
        (1, 0) => g.v_wall_across_x(i[0], i[1], i[2], fwd),
        (1, 2) => g.v_wall_across_z(i[0], i[1], i[2], fwd),
        (2, 0) => g.w_wall_across_x(i[0], i[1], i[2], fwd),
        (2, 1) => g.w_wall_across_y(i[0], i[1], i[2], fwd),
        _ => unreachable!(),
    }
}

/// Oracle A-8: all six `*_wall_across_*` accessors against the geometric
/// model, on random boundary patterns where every face is Fluid, no-slip
/// wall (moving or at rest), SlipWall, Inflow or Outflow, for every query
/// face and both directions (grid edges included).
#[test]
fn wall_across_accessors_match_the_mac_geometry() {
    let mut some = 0usize;
    let mut none = 0usize;
    for seed in 0..12 {
        // wall-rich pattern: re-roll Fluid/Outflow/Inflow into walls half the time
        let mut rng = Lcg(1000 + seed);
        let mut g = MacGrid::new(3, 2, 3, q(1, 2));
        for comp in 0..3 {
            for f in all_faces(&g, comp) {
                let mut bc = random_bc(&mut rng);
                if rng.below(2) == 0 && !bc.is_wall() {
                    bc = FaceBc::Wall {
                        velocity: v3(
                            q(rng.below(5) as i64, 8),
                            q(rng.below(5) as i64, 8),
                            q(rng.below(5) as i64, 8),
                        ),
                    };
                }
                set_bc_at(&mut g, comp, f, bc);
            }
        }
        for c in 0..3 {
            for a in 0..3 {
                if a == c {
                    continue;
                }
                for f in all_faces(&g, c) {
                    for fwd in [false, true] {
                        let want = wall_across_model(&g, c, f, a, fwd);
                        let got = call_wall_across(&g, c, a, f, fwd);
                        assert_eq!(
                            got, want,
                            "seed {seed} comp {c} axis {a} face {f:?} fwd {fwd}"
                        );
                        if want.is_some() {
                            some += 1;
                        } else {
                            none += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(
        some > 100 && none > 100,
        "the patterns must exercise both outcomes ({some}/{none})"
    );
}

// ---------------------------------------------------------------------------
// divergence / cell_velocity
// ---------------------------------------------------------------------------

/// Oracle A-9: a field linear in position, `u = a x`, `v = b y`, `w = c z`
/// sampled at the face positions, has `div = a + b + c` in every cell and a
/// cell-centre velocity `(a, b, c) * centre` (the average of the two faces
/// of a linear field is its value at the centre). `dx` is not 1 so a missing
/// or doubled `/ dx` shows.
#[test]
fn divergence_and_cell_velocity_of_a_linear_field() {
    let dx = q(1, 4);
    let (a, b, c) = (
        Fix128::from_int(3),
        Fix128::from_int(-2),
        Fix128::from_int(5),
    );
    let mut g = MacGrid::new(3, 4, 2, dx);
    for f in all_faces(&g, 0) {
        let x = Fix128::from_int(f[0] as i64) * dx;
        set_val_at(&mut g, 0, f, a * x);
    }
    for f in all_faces(&g, 1) {
        let y = Fix128::from_int(f[1] as i64) * dx;
        set_val_at(&mut g, 1, f, b * y);
    }
    for f in all_faces(&g, 2) {
        let z = Fix128::from_int(f[2] as i64) * dx;
        set_val_at(&mut g, 2, f, c * z);
    }
    for k in 0..2 {
        for j in 0..4 {
            for i in 0..3 {
                assert_eq!(g.divergence(i, j, k), a + b + c, "div at {i},{j},{k}");
                let (uc, vc, wc) = g.cell_velocity(i, j, k);
                let half = q(1, 2);
                assert_eq!(
                    uc,
                    a * (Fix128::from_int(i as i64) + half) * dx,
                    "uc at {i},{j},{k}"
                );
                assert_eq!(
                    vc,
                    b * (Fix128::from_int(j as i64) + half) * dx,
                    "vc at {i},{j},{k}"
                );
                assert_eq!(
                    wc,
                    c * (Fix128::from_int(k as i64) + half) * dx,
                    "wc at {i},{j},{k}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// g2p_velocity
// ---------------------------------------------------------------------------

/// Oracle A-10: `g2p_velocity` reproduces a field linear in position exactly
/// when each component is stored at its own face positions
/// (`u` at `(i, j+1/2, k+1/2) dx`, `v` at `(i+1/2, j, k+1/2) dx`, `w` at
/// `(i+1/2, j+1/2, k) dx`), anywhere inside the staggered interior, with
/// different gradients per component so a swapped stagger offset shows.
#[test]
fn g2p_reproduces_a_linear_field_with_per_component_staggering() {
    let dx = q(1, 4);
    let n = 4usize;
    let mut g = MacGrid::new(n, n, n, dx);
    let half = q(1, 2);
    // field components: u = 1 + 2x + 3y - z ; v = -1 + x - 2y + 4z ; w = 2 - 3x + y + 2z
    let fu = |x: Fix128, y: Fix128, z: Fix128| {
        Fix128::ONE + x * Fix128::from_int(2) + y * Fix128::from_int(3) - z
    };
    let fv = |x: Fix128, y: Fix128, z: Fix128| {
        -Fix128::ONE + x - y * Fix128::from_int(2) + z * Fix128::from_int(4)
    };
    let fw = |x: Fix128, y: Fix128, z: Fix128| {
        Fix128::from_int(2) - x * Fix128::from_int(3) + y + z * Fix128::from_int(2)
    };
    let c = |i: usize, off: Fix128| (Fix128::from_int(i as i64) + off) * dx;
    for f in all_faces(&g, 0) {
        set_val_at(
            &mut g,
            0,
            f,
            fu(c(f[0], Fix128::ZERO), c(f[1], half), c(f[2], half)),
        );
    }
    for f in all_faces(&g, 1) {
        set_val_at(
            &mut g,
            1,
            f,
            fv(c(f[0], half), c(f[1], Fix128::ZERO), c(f[2], half)),
        );
    }
    for f in all_faces(&g, 2) {
        set_val_at(
            &mut g,
            2,
            f,
            fw(c(f[0], half), c(f[1], half), c(f[2], Fix128::ZERO)),
        );
    }
    // positions in sixteenths of a metre inside [dx/2, 1 - dx/2] = [2/16, 14/16]
    let coords: Vec<Fix128> = [2, 3, 5, 7, 8, 11, 13, 14]
        .iter()
        .map(|&s| q(s, 16))
        .collect();
    for &x in &coords {
        for &y in &coords {
            for &z in &coords {
                let got = g2p_velocity(&g, v3(x, y, z));
                assert_eq!(got.x, fu(x, y, z), "u at {x:?},{y:?},{z:?}");
                assert_eq!(got.y, fv(x, y, z), "v at {x:?},{y:?},{z:?}");
                assert_eq!(got.z, fw(x, y, z), "w at {x:?},{y:?},{z:?}");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Known defect reproducers (kept red for the repair pass)
// ---------------------------------------------------------------------------

/// Oracle A-11: an Outflow face takes the value of the interior face next to
/// it, and that value is the one the neighbour has *after* the conditions
/// are imposed, whichever end of the axis the Outflow is on.
///
/// Measured: at the high end (index `n`) the neighbour `n - 1` has already
/// been processed when the Outflow face is reached, so an Inflow neighbour
/// is read at its prescribed value; at the low end (index 0) the neighbour
/// `1` has not been processed yet, so the same Inflow (or a wall) is read at
/// its stale value, and a second call gives a different field.
#[test]
#[ignore = "known defect: AUD-A-S1W1-001: enforce_face_boundaries Outflow at index 0 reads the not-yet-enforced neighbour (x: Inflow 3/8 neighbour gives 0 instead of 3/8, second call changes the field), high end reads the enforced one"]
fn outflow_on_the_low_layer_reads_a_stale_neighbour() {
    let inflow = FaceBc::Inflow {
        normal_velocity: q(3, 8),
    };
    for comp in 0..3 {
        let n = 3;
        let mut low = MacGrid::new(n, n, n, q(1, 2));
        let mut high = MacGrid::new(n, n, n, q(1, 2));
        let mut at = [1usize, 1, 1];
        at[comp] = 0;
        let mut nb_low = at;
        nb_low[comp] = 1;
        set_bc_at(&mut low, comp, at, FaceBc::Outflow);
        set_bc_at(&mut low, comp, nb_low, inflow);
        let mut at_hi = [1usize, 1, 1];
        at_hi[comp] = n;
        let mut nb_high = at_hi;
        nb_high[comp] = n - 1;
        set_bc_at(&mut high, comp, at_hi, FaceBc::Outflow);
        set_bc_at(&mut high, comp, nb_high, inflow);
        low.enforce_face_boundaries();
        high.enforce_face_boundaries();
        // the high end (control) reads the enforced Inflow value
        assert_eq!(val_at(&high, comp, at_hi), q(3, 8), "high end, comp {comp}");
        // the low end must do the same
        assert_eq!(val_at(&low, comp, at), q(3, 8), "low end, comp {comp}");
    }
}

// ---------------------------------------------------------------------------
// set_*_solid on every axis
// ---------------------------------------------------------------------------

/// Oracle A-12: `set_u_solid` / `set_v_solid` / `set_w_solid` are the old
/// spelling of "wall at rest" / "plain fluid": `true` gives
/// `Wall { velocity: 0 }`, `false` gives `Fluid`, either one replaces
/// whatever was on the face (an inflow, a moving wall).
#[test]
fn solid_setters_mean_wall_at_rest_or_fluid_on_every_axis() {
    let rest = FaceBc::Wall {
        velocity: Vec3Fix::ZERO,
    };
    let moving = FaceBc::Wall {
        velocity: v3(q(1, 2), q(1, 4), q(1, 8)),
    };
    let inflow = FaceBc::Inflow {
        normal_velocity: q(3, 8),
    };
    let mut g = MacGrid::new(3, 3, 3, q(1, 2));
    for prior in [
        FaceBc::Fluid,
        moving,
        inflow,
        FaceBc::SlipWall,
        FaceBc::Outflow,
    ] {
        for comp in 0..3 {
            let f = [1usize, 2, 1];
            set_bc_at(&mut g, comp, f, prior);
            match comp {
                0 => g.set_u_solid(f[0], f[1], f[2], true),
                1 => g.set_v_solid(f[0], f[1], f[2], true),
                _ => g.set_w_solid(f[0], f[1], f[2], true),
            }
            assert_eq!(
                bc_at(&g, comp, f),
                rest,
                "solid=true over {prior:?}, comp {comp}"
            );
            assert!(g.u_solid[g.idx_u_for_test(f)] || comp != 0);
            match comp {
                0 => g.set_u_solid(f[0], f[1], f[2], false),
                1 => g.set_v_solid(f[0], f[1], f[2], false),
                _ => g.set_w_solid(f[0], f[1], f[2], false),
            }
            assert_eq!(
                bc_at(&g, comp, f),
                FaceBc::Fluid,
                "solid=false, comp {comp}"
            );
            let solid = match comp {
                0 => g.is_u_solid(f[0], f[1], f[2]),
                1 => g.is_v_solid(f[0], f[1], f[2]),
                _ => g.is_w_solid(f[0], f[1], f[2]),
            };
            assert!(!solid);
        }
    }
}

// ---------------------------------------------------------------------------
// project_pressure
// ---------------------------------------------------------------------------

/// Oracle A-13: `project_pressure` solves `div grad p = (rho/dt) div u*` and
/// then subtracts `(dt/rho) grad p`. Hand-built exact case: pick any cell
/// potential `phi`, set `u* = (dt/rho) grad phi` with the discrete gradient
/// of the open box (zero outside on every side), so `p = phi` solves the
/// Poisson problem and the corrected velocity is exactly zero. `rho = 8`,
/// `dt = 1/4`, `dx = 1/2` so a wrong power of `rho`, `dt` or `dx` in either
/// stage leaves a large residue (the potential is O(1) and the velocity
/// O(1/32)).
#[test]
fn project_pressure_removes_a_gradient_field_and_recovers_its_potential() {
    let n = 4usize;
    let dx = q(1, 2);
    let (dt, rho) = (q(1, 4), Fix128::from_int(8));
    let coeff = dt / rho / dx; // (dt / rho) / dx, exact (dyadic)
    let mut g = MacGrid::new(n, n, n, dx);
    let mut rng = Lcg(99);
    let mut phi = vec![Fix128::ZERO; n * n * n];
    for p in phi.iter_mut() {
        *p = Fix128::from_int(rng.below(9) as i64 - 4) * q(1, 4);
    }
    let at = |i: i64, j: i64, k: i64| -> Fix128 {
        let nn = n as i64;
        if i < 0 || j < 0 || k < 0 || i >= nn || j >= nn || k >= nn {
            Fix128::ZERO
        } else {
            phi[(i + nn * (j + nn * k)) as usize]
        }
    };
    for f in all_faces(&g, 0) {
        let (i, j, k) = (f[0] as i64, f[1] as i64, f[2] as i64);
        set_val_at(&mut g, 0, f, coeff * (at(i, j, k) - at(i - 1, j, k)));
    }
    for f in all_faces(&g, 1) {
        let (i, j, k) = (f[0] as i64, f[1] as i64, f[2] as i64);
        set_val_at(&mut g, 1, f, coeff * (at(i, j, k) - at(i, j - 1, k)));
    }
    for f in all_faces(&g, 2) {
        let (i, j, k) = (f[0] as i64, f[1] as i64, f[2] as i64);
        set_val_at(&mut g, 2, f, coeff * (at(i, j, k) - at(i, j, k - 1)));
    }
    let tol = q(1, 1_000_000);
    alice_physics::eulerian_grid::project_pressure(&mut g, dt, rho, 600);
    for comp in 0..3 {
        for f in all_faces(&g, comp) {
            let x = val_at(&g, comp, f).abs();
            assert!(x < tol, "comp {comp} face {f:?} left {x:?}");
        }
    }
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let d = (g.pressure(i, j, k) - at(i as i64, j as i64, k as i64)).abs();
                assert!(d < tol, "pressure at {i},{j},{k} off by {d:?}");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Multigrid refusal contract (module doc: "any extent is not a power of two,
// including 0")
// ---------------------------------------------------------------------------

/// Oracle A-14: `project_pressure_multigrid` leaves the grid bit-identical
/// (no wall enforcement, no pressure write) when **any one** axis extent is
/// not a power of two, zero included. A control grid of power-of-two extents
/// with the same seed does change, so "untouched" is observable.
#[test]
fn multigrid_refuses_each_axis_that_is_not_a_power_of_two() {
    let seeded = |nx: usize, ny: usize, nz: usize| {
        let mut g = MacGrid::new(nx, ny, nz, q(1, 2));
        let mut rng = Lcg(5);
        for comp in 0..3 {
            for f in all_faces(&g, comp) {
                set_val_at(
                    &mut g,
                    comp,
                    f,
                    Fix128::from_int(rng.below(9) as i64 - 4) * q(1, 8),
                );
            }
        }
        // a wall face carrying a non-zero value: a solver that enforced walls
        // would zero it, so "untouched" shows up there as well
        if nx > 0 && ny > 0 && nz > 0 {
            g.set_u_bc(
                0,
                0,
                0,
                FaceBc::Wall {
                    velocity: Vec3Fix::ZERO,
                },
            );
            let ix = g.idx_u_for_test([0, 0, 0]);
            g.u[ix] = Fix128::ONE;
        }
        g
    };
    for (nx, ny, nz) in [
        (3, 4, 4),
        (4, 3, 4),
        (4, 4, 3),
        (6, 4, 4),
        (4, 4, 5),
        (0, 4, 4),
        (4, 0, 4),
        (4, 4, 0),
    ] {
        let mut g = seeded(nx, ny, nz);
        let before = g.clone();
        alice_physics::eulerian_grid::project_pressure_multigrid(
            &mut g,
            q(1, 4),
            Fix128::from_int(2),
            3,
        );
        assert_eq!(g.u, before.u, "u of {nx}x{ny}x{nz}");
        assert_eq!(g.v, before.v, "v of {nx}x{ny}x{nz}");
        assert_eq!(g.w, before.w, "w of {nx}x{ny}x{nz}");
        assert_eq!(g.pressure, before.pressure, "pressure of {nx}x{ny}x{nz}");
    }
    let mut control = seeded(4, 4, 4);
    let before = control.clone();
    alice_physics::eulerian_grid::project_pressure_multigrid(
        &mut control,
        q(1, 4),
        Fix128::from_int(2),
        3,
    );
    assert!(
        control.u != before.u || control.pressure != before.pressure,
        "the control grid must be solved"
    );
}

// ---------------------------------------------------------------------------
// Prescribed inflow on every axis (module doc: Inflow drops out of the
// pressure stencil like a wall and keeps its prescribed value)
// ---------------------------------------------------------------------------

/// Oracle A-15: a box closed on every side except one axis whose low layer is
/// a uniform `Inflow` and whose high layer is `Outflow` has the closed-form
/// projected state `normal velocity = inflow everywhere` (a uniform flow
/// through a duct is divergence free, and it is the only solution with the
/// prescribed inflow), with zero tangential velocity and zero divergence.
/// A face condition that fails to drop out of the stencil on one axis leaves
/// a pressure gradient across the duct. All three axes, so each of the
/// `u_` / `v_` / `w_blocks_pressure` and `*_is_inflow` paths is exercised.
#[test]
fn a_uniform_inflow_duct_projects_to_the_uniform_flow_on_every_axis() {
    let n = 4usize;
    let dx = q(1, 2);
    let (dt, rho) = (q(1, 4), Fix128::from_int(2));
    let speed = q(1, 2);
    for axis in 0..3 {
        let mut g = MacGrid::new(n, n, n, dx);
        g.set_closed_box_walls();
        for f in all_faces(&g, axis) {
            if f[axis] == 0 {
                set_bc_at(
                    &mut g,
                    axis,
                    f,
                    FaceBc::Inflow {
                        normal_velocity: speed,
                    },
                );
            } else if f[axis] == n {
                set_bc_at(&mut g, axis, f, FaceBc::Outflow);
            }
        }
        alice_physics::eulerian_grid::project_pressure(&mut g, dt, rho, 1500);
        let tol = q(1, 100_000);
        for comp in 0..3 {
            for f in all_faces(&g, comp) {
                let want = if comp == axis { speed } else { Fix128::ZERO };
                let got = val_at(&g, comp, f);
                assert!(
                    (got - want).abs() < tol,
                    "axis {axis} comp {comp} face {f:?}: {got:?} vs {want:?}"
                );
            }
        }
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let d = g.divergence(i, j, k).abs();
                    assert!(d < tol, "axis {axis} divergence at {i},{j},{k}: {d:?}");
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// BiCGStab residual history against the textbook algorithm
// ---------------------------------------------------------------------------

use alice_physics::cfd_solver::{CfdSolver, PressureSolver};

const BN: usize = 4;

fn bicgstab_scene() -> (CfdSolver, Fix128) {
    let dx = q(1, 4);
    let mut s = CfdSolver::new(BN, BN, BN, dx);
    s.gravity = Vec3Fix::ZERO;
    s.dynamic_viscosity_pas = Fix128::ZERO;
    // u(i, j, k) = i dx: a linear profile with div u = 1 in every cell
    for k in 0..BN {
        for j in 0..BN {
            for i in 0..=BN {
                let ix = s.grid.idx_u_for_test([i, j, k]);
                s.grid.u[ix] = Fix128::from_int(i as i64) * dx;
            }
        }
    }
    (s, q(1, 100))
}

/// Residual infinity norms of `A p = b` after 1..=`k` iterations of
/// right-preconditioned BiCGStab (van der Vorst 1992, diagonal preconditioner
/// `K = diag(A)`) in `f64`, for the open box `n^3` (every face fluid, so every
/// cell has six open faces: `A = -6 I + sum of the six neighbours`, the
/// exterior counts as `p = 0`), with `b = rho dx^2 / dt * div` and `div`
/// uniform, starting from `p = 0`.
fn textbook_bicgstab_residuals(n: usize, b_value: f64, k: usize) -> Vec<f64> {
    let cells = n * n * n;
    let idx = |i: usize, j: usize, l: usize| i + n * (j + n * l);
    let apply = |x: &[f64]| -> Vec<f64> {
        let mut out = vec![0.0; cells];
        for l in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let mut acc = -6.0 * x[idx(i, j, l)];
                    if i > 0 {
                        acc += x[idx(i - 1, j, l)];
                    }
                    if i + 1 < n {
                        acc += x[idx(i + 1, j, l)];
                    }
                    if j > 0 {
                        acc += x[idx(i, j - 1, l)];
                    }
                    if j + 1 < n {
                        acc += x[idx(i, j + 1, l)];
                    }
                    if l > 0 {
                        acc += x[idx(i, j, l - 1)];
                    }
                    if l + 1 < n {
                        acc += x[idx(i, j, l + 1)];
                    }
                    out[idx(i, j, l)] = acc;
                }
            }
        }
        out
    };
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    let linf = |a: &[f64]| a.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    let b = vec![b_value; cells];
    let mut r: Vec<f64> = b.clone(); // b - A 0
    let r_hat = r.clone();
    let (mut rho_prev, mut alpha, mut omega) = (1.0, 1.0, 1.0);
    let mut v = vec![0.0; cells];
    let mut p = vec![0.0; cells];
    let mut history = Vec::new();
    for _ in 0..k {
        let rho = dot(&r_hat, &r);
        let beta = (rho / rho_prev) * (alpha / omega);
        for c in 0..cells {
            p[c] = r[c] + beta * (p[c] - omega * v[c]);
        }
        let y: Vec<f64> = p.iter().map(|x| x / -6.0).collect();
        v = apply(&y);
        alpha = rho / dot(&r_hat, &v);
        let s: Vec<f64> = (0..cells).map(|c| r[c] - alpha * v[c]).collect();
        let z: Vec<f64> = s.iter().map(|x| x / -6.0).collect();
        let t = apply(&z);
        omega = dot(&t, &s) / dot(&t, &t);
        for c in 0..cells {
            r[c] = s[c] - omega * t[c];
        }
        history.push(linf(&r));
        rho_prev = rho;
    }
    history
}

/// Oracle A-16: the BiCGStab pressure solver's reported residual after each
/// iteration equals the textbook preconditioned BiCGStab recurrence on the
/// same system (an `f64` re-derivation, not the solver's own arithmetic).
///
/// The scene is `u = i dx` with no body force and no viscosity: the
/// semi-Lagrangian back-trace of a linear profile lands at `x (1 - dt)`, so
/// the divergence entering the projection is `1 - dt` in every cell (derived
/// by hand here, confirmed by the match). Iterations 1..=3 are compared; the
/// fourth is the breakdown case of `bicgstab_breakdown_keeps_a_stale_iterate`.
#[test]
fn bicgstab_residual_history_matches_the_textbook_recurrence() {
    let tol = Fix128::from_raw(0, 1); // one ulp: never reached
    let (probe, dt) = bicgstab_scene();
    let scale = probe.density_kg_m3.to_f64() * probe.grid.dx.to_f64() * probe.grid.dx.to_f64()
        / dt.to_f64();
    let reference = textbook_bicgstab_residuals(BN, scale * (1.0 - dt.to_f64()), 3);
    for (k, want) in reference.iter().enumerate() {
        let (mut s, dt) = bicgstab_scene();
        let report = s
            .step_with_pressure_solver(
                dt,
                PressureSolver::BiCgStab {
                    max_iterations: k as u32 + 1,
                    tolerance: tol,
                },
            )
            .expect("valid request");
        let stats = report.bicgstab.expect("reports");
        let got = stats.final_residual.to_f64();
        assert_eq!(stats.iterations, k as u32 + 1);
        assert!(
            (got - want).abs() <= 1e-9 * want.abs().max(1.0),
            "iteration {}: solver {got:e} vs textbook {want:e}",
            k + 1
        );
    }
}

/// Oracle A-17: the stop test is strict (`||r||_inf < tolerance`, the doc's
/// words), so a tolerance equal to the residual of the first iteration does
/// not count as converged and a tolerance one ulp larger does.
#[test]
fn bicgstab_stop_test_is_strict() {
    let run = |tolerance: Fix128| {
        let (mut s, dt) = bicgstab_scene();
        s.step_with_pressure_solver(
            dt,
            PressureSolver::BiCgStab {
                max_iterations: 1,
                tolerance,
            },
        )
        .expect("valid request")
        .bicgstab
        .expect("reports")
    };
    let first = run(Fix128::from_raw(0, 1));
    assert!(!first.converged);
    let r1 = first.final_residual;
    let at = run(r1);
    assert!(!at.converged, "residual == tolerance is not below it");
    assert_eq!(at.final_residual, r1);
    let above = run(r1 + Fix128::from_raw(0, 1));
    assert!(above.converged, "residual < tolerance converges");
}

/// Oracle A-18: with a budget that is not the limit, the solver reaches the
/// residual the textbook recurrence reaches on this scene (`~7e-15` at the
/// fourth iteration, where Krylov space exhausts the problem).
///
/// Measured: the fourth iteration computes `t = A z` with `z ~ 1e-15`, so
/// `t . t` is below one `Fix128` ulp, `tt.is_zero()` ends the loop before the
/// pending `x += alpha y`, and the solver returns `converged = false` with the
/// iteration-3 residual `2.986` (1.9e-3 of `|b|`) although 196 iterations of
/// budget remain and one more update would reach `1e-14`.
#[test]
#[ignore = "known defect: AUD-A-S1W1-002: BiCGStab breaks down on Fix128 underflow of t.t (tolerance below ~1e-10) and returns the iteration-3 iterate, converged=false, residual 2.986 instead of ~7e-15"]
fn bicgstab_breakdown_keeps_a_stale_iterate() {
    let (mut s, dt) = bicgstab_scene();
    let stats = s
        .step_with_pressure_solver(
            dt,
            PressureSolver::BiCgStab {
                max_iterations: 200,
                tolerance: Fix128::from_raw(0, 1),
            },
        )
        .expect("valid request")
        .bicgstab
        .expect("reports");
    assert!(
        stats.final_residual.to_f64() < 1e-9,
        "stopped after {} iterations at residual {:e}",
        stats.iterations,
        stats.final_residual.to_f64()
    );
}
