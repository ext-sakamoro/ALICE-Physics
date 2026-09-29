//! Element quality invariants for `alice_physics::sdf_fem_mesh`.
//!
//! Conformity (`mesh_conformity.rs`) says the elements fit together. It says
//! nothing about their *shape*, and a mesh can be perfectly conforming and still
//! useless: a sliver — a tetrahedron flattened almost into a plane — has a
//! stiffness matrix whose condition number blows up as its smallest dihedral
//! angle goes to zero, and the gradients it reconstructs are worst exactly where
//! it is thinnest.
//!
//! # The measure has to be the dihedral angle
//!
//! The usual quality number is the **radius-edge ratio** (circumradius over
//! shortest edge), because Delaunay refinement bounds it. It does not see
//! slivers: a sliver can have four well-spaced vertices, hence a perfectly
//! ordinary radius-edge ratio, while being flat. Measured on this crate's
//! marching-tets output at cell 0.1875, 1,448 tetrahedra had a radius-edge
//! ratio under 2 — nominally good — while the worst dihedral angle in the mesh
//! was 4.59°.
//!
//! So this file gates the **minimum dihedral angle** and reports the
//! radius-edge distribution alongside it, rather than the other way round.
//!
//! # What this gate is for, now that the solver can be asked directly
//!
//! The minimum dihedral angle is a *proxy*. What actually hurts a finite element
//! solve is the condition number of the stiffness matrix, and
//! `tests/mesh_to_fem_stress.rs` measures that end to end by running the FEM and
//! reading how far its answer lands from an exact one. That file carries the
//! claim "element shape is not breaking the solve"; this one does not.
//!
//! Two jobs are left here, and both are worth a gate:
//!
//! - **The upper side.** Warping corners too far collapses the mesh in a way the
//!   stress test does not see, because its scene is a box: at
//!   `SNAP_CELL_FRACTION = 0.49` corners on opposite sides of the torus tube warp
//!   towards each other and the worst angle falls to 3.80°, while the volume
//!   convergence stays clean. Nothing else in the suite is looking at that.
//! - **Cheap regression detection across more scenes** than it is worth running
//!   a full solve on.
//!
//! # The threshold is measured, not conventional
//!
//! **10° was the target declared before the fix**, taken from finite element
//! practice, and the mesh now clears it — but only by 0.20° on the torus at the
//! finest level, which is too thin to gate on and says more about the convention
//! than about the mesh.
//!
//! So the gate is placed between the two measured populations instead:
//!
//! | | worst minimum dihedral angle |
//! |---|---|
//! | warped lattice (ball and torus, 3 levels) | **10.20°** |
//! | warp disabled | 4.59° |
//! | warp far too wide (`0.49`) | 3.80° |
//!
//! `MIN_DIHEDRAL_DEG` sits 1.28 times under the worst good mesh, 1.74 times over
//! the unwarped one and 2.11 times over the over-warped one. The 10° convention
//! stays written down here as the reference it is, and not as the number a test
//! asserts.
//!
//! # Why this stays a gate and not a report
//!
//! Measured by rebuilding the mesher at three warp settings and running both
//! files:
//!
//! | warp | this file | `mesh_to_fem_stress.rs` |
//! |---|---|---|
//! | disabled | red, 4.59° | red, amplification 347.9 |
//! | 0.30 (shipped) | green | green, 7.1 to 20.7 |
//! | 0.49 (too wide) | red, 3.80° | **green, 5.2 to 12.3** |
//!
//! The solver-side measurement is the better instrument for the lower side and
//! is where the primary claim lives. It cannot see the upper side at all,
//! because its scene is a box and a box only improves as the warp widens.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::sdf_collider::ClosureSdf;
use alice_physics::sdf_fem_mesh::{generate, generate_marching_tets, SdfTetMesh};

/// Where the gate sits, in degrees — between the measured good and bad
/// populations, not on the engineering convention.
///
/// See the module documentation for both populations and all three margins.
const MIN_DIHEDRAL_DEG: f64 = 8.0;

/// The minimum dihedral angle finite element practice usually asks for.
///
/// Reported, never asserted. Kept so that the distance between what the mesh
/// achieves and what the convention wants stays visible: at the finest torus
/// level it is 0.20°.
const CONVENTIONAL_MIN_DIHEDRAL_DEG: f64 = 10.0;

/// A scene to mesh, with a closed form for the volume it encloses.
///
/// One shape is not enough to choose a threshold on. A sphere presents every
/// surface orientation to the lattice, but always convex and always at the same
/// curvature, and a value tuned to it would be tuned to that. The torus adds
/// concave curvature and, at the inner equator, two surfaces a couple of cells
/// apart — the configuration where a warp is most likely to close a gap that
/// should stay open.
struct Scene {
    name: &'static str,
    sdf: ClosureSdf,
    half_extent: f32,
    volume: f64,
}

/// A slab and a square rod, both only a few cells thick.
///
/// These are where the warp is expected to hurt most, and they are measured
/// apart from the closed shapes because the quantity that matters is different.
/// The warp moves a corner by up to `SNAP_CELL_FRACTION * cell`, and when the
/// body is two or three cells thick that is a large fraction of the thickness —
/// with a corner on each face free to move *inwards*, the mesh can come out
/// systematically thinner than the shape while every element in it stays
/// well formed. No existing gate would notice: the dihedral angle is a shape
/// measure, and the volume gate reads a convergence order, which a uniform
/// shrink barely disturbs.
///
/// So `thin_scenes` are gated on measured thickness, not on angles.
fn thin_scenes() -> Vec<ThinScene> {
    vec![
        ThinScene {
            name: "plate",
            half: [2.0, 2.0, 0.31],
            centre: [0.07, -0.05, 0.03],
        },
        ThinScene {
            name: "rod",
            half: [2.0, 0.31, 0.27],
            centre: [0.07, -0.05, 0.03],
        },
    ]
}

/// A box that is thin in one or two directions, with its exact half-extents.
struct ThinScene {
    name: &'static str,
    half: [f32; 3],
    centre: [f32; 3],
}

impl ThinScene {
    fn sdf(&self) -> ClosureSdf {
        let (half, centre) = (self.half, self.centre);
        ClosureSdf::new(
            move |x, y, z| {
                let d = [
                    (x - centre[0]).abs() - half[0],
                    (y - centre[1]).abs() - half[1],
                    (z - centre[2]).abs() - half[2],
                ];
                let out = [d[0].max(0.0), d[1].max(0.0), d[2].max(0.0)];
                (out[0] * out[0] + out[1] * out[1] + out[2] * out[2]).sqrt()
                    + d[0].max(d[1]).max(d[2]).min(0.0)
            },
            |_, _, _| (0.0, 0.0, 1.0),
        )
    }
}

/// The extent of the meshed vertices along each axis.
///
/// Read off the vertices rather than the volume, because the question is whether
/// the *surface* moved: a plate that keeps its volume by bulging in the middle
/// while its faces pull in is still the wrong plate.
fn measured_extent(mesh: &SdfTetMesh) -> [f64; 3] {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for v in &mesh.vertices {
        for k in 0..3 {
            lo[k] = lo[k].min(f64::from(v[k]));
            hi[k] = hi[k].max(f64::from(v[k]));
        }
    }
    [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]]
}

fn scenes() -> Vec<Scene> {
    vec![
        Scene {
            name: "ball",
            sdf: ball_sdf(1.0),
            half_extent: 1.5,
            volume: 4.0 / 3.0 * std::f64::consts::PI,
        },
        Scene {
            name: "torus",
            sdf: torus_sdf(0.8, 0.35),
            half_extent: 1.5,
            // 2 pi^2 R r^2
            volume: 2.0 * std::f64::consts::PI * std::f64::consts::PI * 0.8 * 0.35 * 0.35,
        },
    ]
}

/// Torus of major radius `major` about the z axis, minor radius `minor`.
///
/// The gradient is written out rather than taken from a finite difference so
/// that the field is exact where the mesher samples it.
fn torus_sdf(major: f32, minor: f32) -> ClosureSdf {
    ClosureSdf::new(
        move |x, y, z| {
            let q = (x * x + y * y).sqrt() - major;
            (q * q + z * z).sqrt() - minor
        },
        move |x, y, z| {
            let radial = (x * x + y * y).sqrt().max(1.0e-6);
            let q = radial - major;
            let len = (q * q + z * z).sqrt().max(1.0e-6);
            (q * x / (radial * len), q * y / (radial * len), z / len)
        },
    )
}

fn ball_sdf(radius: f32) -> ClosureSdf {
    ClosureSdf::new(
        move |x, y, z| (x * x + y * y + z * z).sqrt() - radius,
        |x, y, z| {
            let len = (x * x + y * y + z * z).sqrt().max(1.0e-6);
            (x / len, y / len, z / len)
        },
    )
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

/// The six dihedral angles of a tetrahedron, in degrees.
///
/// Along the edge `(i, j)`, the angle between the two faces that share it. Both
/// face normals are taken as `edge × (other vertex − i)`, which puts them on the
/// same side of the edge, so the angle between them is the interior one.
fn dihedral_angles_deg(p: [[f64; 3]; 4]) -> Vec<f64> {
    let mut out = Vec::with_capacity(6);
    for i in 0..4 {
        for j in (i + 1)..4 {
            let others: Vec<usize> = (0..4).filter(|m| *m != i && *m != j).collect();
            let e = sub(p[j], p[i]);
            let n1 = cross(e, sub(p[others[0]], p[i]));
            let n2 = cross(e, sub(p[others[1]], p[i]));
            let denom = norm(n1) * norm(n2);
            if denom <= 0.0 {
                out.push(0.0);
                continue;
            }
            out.push((dot(n1, n2) / denom).clamp(-1.0, 1.0).acos().to_degrees());
        }
    }
    out
}

fn circumradius(p: [[f64; 3]; 4]) -> f64 {
    let a = [sub(p[1], p[0]), sub(p[2], p[0]), sub(p[3], p[0])];
    let rhs = [
        dot(a[0], a[0]) / 2.0,
        dot(a[1], a[1]) / 2.0,
        dot(a[2], a[2]) / 2.0,
    ];
    let det3 = |m: [[f64; 3]; 3]| {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    let base = det3(a);
    if base == 0.0 {
        return f64::INFINITY;
    }
    let mut centre = [0.0; 3];
    for (i, c) in centre.iter_mut().enumerate() {
        let mut m = a;
        for (r, row) in m.iter_mut().enumerate() {
            row[i] = rhs[r];
        }
        *c = det3(m) / base;
    }
    norm(centre)
}

fn shortest_edge(p: [[f64; 3]; 4]) -> f64 {
    let mut best = f64::INFINITY;
    for i in 0..4 {
        for j in (i + 1)..4 {
            best = best.min(norm(sub(p[j], p[i])));
        }
    }
    best
}

fn signed_volume(p: [[f64; 3]; 4]) -> f64 {
    dot(sub(p[1], p[0]), cross(sub(p[2], p[0]), sub(p[3], p[0]))) / 6.0
}

struct Quality {
    tets: usize,
    degenerate: usize,
    /// Elements whose signed volume is negative, i.e. whose vertex order winds
    /// the opposite way from the rest.
    inverted: usize,
    min_dihedral_deg: f64,
    max_dihedral_deg: f64,
    worst_radius_edge: f64,
    /// Counts of the radius-edge ratio under 1, 2, 5, 20.
    radius_edge_buckets: [usize; 4],
    total_volume: f64,
    /// Tetrahedra whose minimum dihedral angle is under the target.
    below_target: usize,
    /// The four corners of the element holding `min_dihedral_deg`, and its
    /// volume.
    ///
    /// A quality gate that reports only a number tells you that something is
    /// wrong and nothing about what. The offending element is what says whether
    /// the mesher is producing a thin wedge at the surface, which is expected
    /// and bounded, or a flat element with four unrelated corners, which is a
    /// defect.
    worst: Option<([[f64; 3]; 4], f64)>,
}

fn measure(mesh: &SdfTetMesh) -> Quality {
    let mut q = Quality {
        tets: mesh.tet_count(),
        degenerate: 0,
        inverted: 0,
        min_dihedral_deg: 180.0,
        max_dihedral_deg: 0.0,
        worst_radius_edge: 0.0,
        radius_edge_buckets: [0; 4],
        total_volume: 0.0,
        below_target: 0,
        worst: None,
    };
    for tet in &mesh.tets {
        let p: [[f64; 3]; 4] = tet.vertices.map(|v| {
            let w = mesh.vertices[v as usize];
            [f64::from(w[0]), f64::from(w[1]), f64::from(w[2])]
        });
        let vol = signed_volume(p);
        q.total_volume += vol.abs();
        if vol < 0.0 {
            q.inverted += 1;
        }
        if vol.abs() <= 0.0 {
            q.degenerate += 1;
            continue;
        }
        let angles = dihedral_angles_deg(p);
        let lo = angles.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = angles.iter().copied().fold(0.0, f64::max);
        if lo <= q.min_dihedral_deg {
            q.min_dihedral_deg = lo;
            q.worst = Some((p, vol));
        }
        q.max_dihedral_deg = q.max_dihedral_deg.max(hi);
        if lo < CONVENTIONAL_MIN_DIHEDRAL_DEG {
            q.below_target += 1;
        }
        let re = circumradius(p) / shortest_edge(p);
        q.worst_radius_edge = q.worst_radius_edge.max(re);
        let bucket = if re < 1.0 {
            0
        } else if re < 2.0 {
            1
        } else if re < 5.0 {
            2
        } else {
            3
        };
        q.radius_edge_buckets[bucket] += 1;
    }
    q
}

fn report(name: &str, cell: f32, q: &Quality, analytic_volume: f64) {
    eprintln!(
        "[quality] {name:<10} cell {cell:<7} tets {:>5}  degenerate {:>3}  \
         inverted {:>5}  min dihedral {:>6.2}°  max {:>6.2}°  worst r/e {:>7.2}  \
         buckets {:?}  volume {:.4} / {:.4}  below {}°: {}",
        q.tets,
        q.degenerate,
        q.inverted,
        q.min_dihedral_deg,
        q.max_dihedral_deg,
        q.worst_radius_edge,
        q.radius_edge_buckets,
        q.total_volume,
        analytic_volume,
        CONVENTIONAL_MIN_DIHEDRAL_DEG,
        q.below_target
    );
    if let Some((p, vol)) = q.worst {
        eprintln!(
            "[worst]    {name:<10} cell {cell:<7} volume {vol:.3e}  \
             corners {:?}",
            p.map(|c| c.map(|x| (x * 1.0e6).round() / 1.0e6))
        );
    }
}

/// `generate` dices whole cubes, so every element is one of the two 5-tet
/// patterns and the quality is a property of the pattern, not of the surface:
/// a uniform 54.74° whatever the shape or the cell size.
///
/// This is the control for the test below. If clipping had no cost, the
/// marching-tets mesh would look like this one.
#[test]
fn whole_cube_dicing_has_uniform_quality() {
    let scene = &scenes()[0];
    for cell in [0.375_f32, 0.25, 0.1875] {
        let mesh = generate(&scene.sdf, [-1.5, -1.5, -1.5], [1.5, 1.5, 1.5], cell);
        let q = measure(&mesh);
        report("generate", cell, &q, scene.volume);
        assert_eq!(q.degenerate, 0, "cell {cell}: no element may be degenerate");
        assert!(
            q.min_dihedral_deg > 54.0 && q.min_dihedral_deg < 55.0,
            "cell {cell}: the 5-tet pattern's worst dihedral angle is 54.74° by construction, \
             got {:.2}°",
            q.min_dihedral_deg
        );
    }
}

/// Marching tetrahedra clip cells against the surface, and a zero crossing that
/// lands close to a lattice corner makes a very short edge — which is where the
/// slivers come from. `sdf_fem_mesh` warps the corner onto the surface instead;
/// this is the measurement that says whether it still is.
///
/// Both the gate and the convention it replaced are in the module documentation.
#[test]
fn marching_tets_stays_out_of_the_sliver_population() {
    for scene in scenes() {
        check_dihedral_population(&scene);
    }
}

fn check_dihedral_population(scene: &Scene) {
    let series = marching_series(scene);
    // Every level is measured and printed before anything is asserted. Asserting
    // inside the loop would stop at the first level that fails and hide the rest
    // of the series, which is the part that says whether refinement helps or
    // hurts — and here it hurts, so that is exactly the part worth seeing.
    let worst = series
        .iter()
        .map(|(_, q)| q.min_dihedral_deg)
        .fold(180.0_f64, f64::min);
    let degenerate: usize = series.iter().map(|(_, q)| q.degenerate).sum();
    let inverted: usize = series.iter().map(|(_, q)| q.inverted).sum();
    assert_eq!(
        inverted, 0,
        "{}: {inverted} elements are wound the opposite way from the rest. Both generators used \
         to emit a mix — `CUBE_FIVE_TETS` has one entry of one parity wound backwards, putting a \
         tenth of `generate`'s output at negative signed volume — and every consumer took \
         `abs()`, so the sign carried no information and nothing could be asserted about it. Now \
         that `push_tet` normalises it, a negative element means the mesher folded one through \
         its own face, which conformity (face counts), the dihedral angle (unsigned), the volume \
         (summed as `abs()`) and the FEM (`det.abs()`) would all report as healthy",
        scene.name
    );
    assert_eq!(
        degenerate, 0,
        "{}: {degenerate} elements enclose no volume. A zero-volume element is not a \
         tetrahedron, and linear_elastic_fem rejects the whole mesh over one of them",
        scene.name
    );
    assert!(
        worst >= MIN_DIHEDRAL_DEG,
        "{}: worst dihedral angle over the series is {worst:.2}°, under the {MIN_DIHEDRAL_DEG}° \
         gate. That gate is where the measured good and bad meshes separate (10.20° against \
         4.59° and 3.80°), so landing under it means the mesh has joined the bad population — \
         in either direction, since warping too far collapses elements as surely as not warping \
         at all. Do not move the gate to fit the number",
        scene.name
    );
}

/// Every scene at three resolutions, measured, with every level reported.
fn marching_series(scene: &Scene) -> Vec<(f32, Quality)> {
    let h = scene.half_extent;
    [0.375_f32, 0.25, 0.1875]
        .into_iter()
        .map(|cell| {
            let mesh = generate_marching_tets(&scene.sdf, [-h, -h, -h], [h, h, h], cell);
            let q = measure(&mesh);
            report(scene.name, cell, &q, scene.volume);
            (cell, q)
        })
        .collect()
}

/// Whatever the clipping does to element shape, the volume it encloses has to
/// keep converging to the shape's — that is the bound on any geometric error a
/// quality fix introduces.
#[test]
fn marching_tets_volume_converges() {
    for scene in scenes() {
        check_volume_converges(&scene);
    }
}

fn check_volume_converges(scene: &Scene) {
    let analytic = scene.volume;
    let series = marching_series(scene);
    let cells: Vec<f32> = series.iter().map(|(cell, _)| *cell).collect();
    let errors: Vec<f64> = series
        .into_iter()
        .map(|(cell, q)| {
            let err = (q.total_volume - analytic).abs() / analytic;
            eprintln!(
                "[volume]  {:<9} cell {cell:<7} tets {:>5}  volume {:.6}  analytic {:.6}  \
                 error {:.3}%",
                scene.name,
                q.tets,
                q.total_volume,
                analytic,
                100.0 * err
            );
            err
        })
        .collect();
    // The claim under test is that the warp's surface movement vanishes with the
    // cell size, not that it is under some particular percentage. An absolute
    // bound would be a number picked to sit just above whatever was measured,
    // and on the torus — whose tube is only about four cells across at the
    // finest level — it would sit within a whisker of the measurement and turn
    // any later change into a failure for the wrong reason.
    //
    // The observed order between two levels is
    // `ln(e_coarse / e_fine) / ln(h_coarse / h_fine)`. First order is what the
    // `SNAP_CELL_FRACTION * cell` bound entitles this to; both scenes measure
    // above second order, so the requirement has real room under it.
    for (w, hw) in errors.windows(2).zip(cells.windows(2)) {
        assert!(
            w[1] < w[0],
            "{}: refinement must reduce the volume error: {:.4}% -> {:.4}%",
            scene.name,
            100.0 * w[0],
            100.0 * w[1]
        );
        let order = (w[0] / w[1]).ln() / (f64::from(hw[0]) / f64::from(hw[1])).ln();
        eprintln!(
            "[order]   {:<9} cell {:<7} -> {:<7} error {:.3}% -> {:.3}%  order {order:.2}",
            scene.name,
            hw[0],
            hw[1],
            100.0 * w[0],
            100.0 * w[1]
        );
        assert!(
            order >= 1.0,
            "{}: the volume error has to vanish at least as fast as the cell size, since that \
             is what bounds how far the warp moves the surface; observed order {order:.2} \
             between cell {} and {}",
            scene.name,
            hw[0],
            hw[1]
        );
    }
}

/// Bodies only a few cells thick, where the warp moves corners by a large
/// fraction of the thickness.
///
/// The concern is that corners on the two faces of a thin body are drawn towards
/// each other, so the mesh comes out thinner than the shape while every element
/// in it stays well formed — which the dihedral gate would pass and the closed
/// shapes' volume gate would not be looking at.
///
/// Two quantities answer it, because either alone can be fooled:
///
/// - **The extent**, from the extreme vertices. It comes back exactly right at
///   every level, so the faces of the box are where they should be. On its own
///   that proves little: extremes say nothing about a face dimpled between them.
/// - **The volume**, which has a closed form on a box and does see the dimple.
///
/// The volume is gated on convergence rather than on a level, for the reason the
/// closed shapes are: any fixed percentage would be a number read off the
/// measurement. Measured deficits at the shipped warp — plate 4.80%, 2.44%,
/// 1.11% and rod 21.35%, 11.92%, 4.25% — which are orders of 1.9 to 2.5. The rod
/// starts far off because its cross-section is barely two cells across, and that
/// is a statement about asking for a mesh that coarse, not about the warp.
///
/// The warp's own share is visible by rebuilding at other widths: at cell 0.25
/// the plate holds 96.92% with no warp, 95.20% at the shipped 0.30 and 93.55% at
/// 0.49. So widening the warp does trade volume for element shape here, in the
/// direction and roughly the amount the `SNAP_CELL_FRACTION * cell` bound
/// allows, and it converges away.
#[test]
fn thin_bodies_keep_their_thickness() {
    for scene in thin_scenes() {
        let sdf = scene.sdf();
        let exact = scene.half.map(|h| 2.0 * f64::from(h));
        let exact_volume = exact[0] * exact[1] * exact[2];
        let cells = [0.25_f32, 0.1875, 0.125];
        let mut deficits = Vec::new();
        // Measured first, asserted afterwards, so that a failure at the coarsest
        // level does not hide how the series was going.
        for cell in cells {
            let mesh = generate_marching_tets(&sdf, [-3.0, -3.0, -3.0], [3.0, 3.0, 3.0], cell);
            let q = measure(&mesh);
            let extent = measured_extent(&mesh);
            let ratios: [f64; 3] = std::array::from_fn(|k| extent[k] / exact[k]);
            let volume_ratio = q.total_volume / exact_volume;
            eprintln!(
                "[thin]    {:<9} cell {cell:<7} tets {:>6}  inverted {:>4}  \
                 min dihedral {:>6.2}°  extent ratio {:.3} / {:.3} / {:.3}  \
                 volume {:.4}  thickness in cells {:.1}",
                scene.name,
                q.tets,
                q.inverted,
                q.min_dihedral_deg,
                ratios[0],
                ratios[1],
                ratios[2],
                volume_ratio,
                exact[2] / f64::from(cell)
            );
            assert_eq!(q.inverted, 0, "{}: cell {cell}", scene.name);
            assert_eq!(q.degenerate, 0, "{}: cell {cell}", scene.name);
            assert!(
                q.min_dihedral_deg >= MIN_DIHEDRAL_DEG,
                "{}: cell {cell}: {:.2}° is under the {MIN_DIHEDRAL_DEG}° gate",
                scene.name,
                q.min_dihedral_deg
            );
            for (k, ratio) in ratios.iter().enumerate() {
                let lost = (1.0 - ratio) * exact[k];
                assert!(
                    lost < f64::from(cell),
                    "{}: cell {cell}, axis {k}: the meshed body is {lost:.4} shorter than the \
                     shape, more than one cell",
                    scene.name
                );
            }
            deficits.push(1.0 - volume_ratio);
        }
        for (w, hw) in deficits.windows(2).zip(cells.windows(2)) {
            let order = (w[0] / w[1]).ln() / (f64::from(hw[0]) / f64::from(hw[1])).ln();
            eprintln!(
                "[thin]    {:<9} cell {:<7} -> {:<7} deficit {:.2}% -> {:.2}%  order {order:.2}",
                scene.name,
                hw[0],
                hw[1],
                100.0 * w[0],
                100.0 * w[1]
            );
            assert!(
                order >= 1.0,
                "{}: the volume the mesh is missing has to vanish at least as fast as the cell \
                 size, since that is what bounds how far the warp moves a face. Observed order \
                 {order:.2} between cell {} and {}, deficits {:.2}% and {:.2}%",
                scene.name,
                hw[0],
                hw[1],
                100.0 * w[0],
                100.0 * w[1]
            );
        }
    }
}
