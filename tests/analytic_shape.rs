//! Oracles for the solid-shape primitives' mass properties and for the shaped
//! bodies a `PhysicsWorld` builds from them.
//!
//! # Two independent expectations
//!
//! Every shape is checked against **two** things that do not share an
//! expression with the code under test:
//!
//! - the textbook closed form, written out below (and not obtained by calling the
//!   primitive's own method), and
//! - a brute-force **quadrature** of the solid: a midpoint grid over its bounding
//!   box with an `inside` predicate, giving volume, centre of mass and the inertia
//!   about the centre of mass directly from the definition `∫ ρ (|r|² − r rᵀ) dV`.
//!   It knows nothing about the formulas, so a swapped coefficient cannot hide
//!   behind a matching copy of itself.
//!
//! Author: Moroya Sakamoto

#![allow(clippy::disallowed_methods)]

use alice_physics::cone::Cone;
use alice_physics::cylinder::Cylinder;
use alice_physics::ellipsoid::Ellipsoid;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::torus::Torus;
use alice_physics::wedge::Wedge;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn arr(v: Vec3Fix) -> [f64; 3] {
    [v.x.to_f64(), v.y.to_f64(), v.z.to_f64()]
}

/// Volume, centre of mass and the diagonal of the inertia about the centre of
/// mass of the solid `inside` over `lo..hi`, at unit density, by a midpoint grid
/// of `n³` cells.
fn quadrature(
    lo: [f64; 3],
    hi: [f64; 3],
    n: usize,
    inside: impl Fn(f64, f64, f64) -> bool,
) -> (f64, [f64; 3], [f64; 3]) {
    let step = [
        (hi[0] - lo[0]) / n as f64,
        (hi[1] - lo[1]) / n as f64,
        (hi[2] - lo[2]) / n as f64,
    ];
    let cell = step[0] * step[1] * step[2];
    let (mut m, mut c) = (0.0, [0.0; 3]);
    let (mut sxx, mut syy, mut szz) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let x = lo[0] + (i as f64 + 0.5) * step[0];
        for j in 0..n {
            let y = lo[1] + (j as f64 + 0.5) * step[1];
            for k in 0..n {
                let z = lo[2] + (k as f64 + 0.5) * step[2];
                if inside(x, y, z) {
                    m += cell;
                    c[0] += x * cell;
                    c[1] += y * cell;
                    c[2] += z * cell;
                    sxx += x * x * cell;
                    syy += y * y * cell;
                    szz += z * z * cell;
                }
            }
        }
    }
    let com = [c[0] / m, c[1] / m, c[2] / m];
    // ∫ x² dV about the COM = ∫ x² dV − m c_x².
    let (qx, qy, qz) = (
        sxx - m * com[0] * com[0],
        syy - m * com[1] * com[1],
        szz - m * com[2] * com[2],
    );
    (m, com, [qy + qz, qx + qz, qx + qy])
}

fn rel(got: f64, want: f64) -> f64 {
    (got - want).abs() / want.abs().max(1e-300)
}

fn assert_close(got: f64, want: f64, tol: f64, what: &str) {
    assert!(
        rel(got, want) <= tol,
        "{what}: got {got:.12e}, expected {want:.12e} (relative error {:.3e} > {tol:.1e})",
        rel(got, want)
    );
}

const PI: f64 = std::f64::consts::PI;
/// The closed forms are exact; what is left is `Fix128` rounding.
const EXACT: f64 = 1e-12;
/// A 48³ midpoint grid over a curved solid.
const GRID: f64 = 0.03;
const N: usize = 48;

// ---------------------------------------------------------------------------
// Cone — apex at +half_height, base at −half_height, axis Y
// ---------------------------------------------------------------------------

/// Cone `r = 3`, `half_height = 2` (h = 4): `r ≠ h`, so the two inertia terms
/// cannot be swapped without showing.
#[test]
fn the_cone_volume_uses_pi_not_a_rational_approximation() {
    let c = Cone::new(Vec3Fix::ZERO, fx(3.0), fx(2.0));
    // V = (1/3) π r² h, h = 4.
    assert_close(
        c.volume().to_f64(),
        PI * 9.0 * 4.0 / 3.0,
        EXACT,
        "cone volume",
    );
}

#[test]
fn the_cone_inertia_about_the_centre_of_mass_is_the_textbook_one() {
    let (r, hh) = (3.0, 2.0);
    let h = 2.0 * hh;
    let c = Cone::new(Vec3Fix::ZERO, fx(r), fx(hh));
    let m = 5.0;
    let i = arr(c.inertia_diagonal(fx(m)));
    // About the COM: Iyy = 3/10 m r², Ixx = Izz = 3/20 m r² + 3/80 m h².
    let ixx = m * (3.0 / 20.0 * r * r + 3.0 / 80.0 * h * h);
    assert_close(i[0], ixx, EXACT, "cone Ixx");
    assert_close(i[1], 0.3 * m * r * r, EXACT, "cone Iyy");
    assert_close(i[2], ixx, EXACT, "cone Izz");
}

#[test]
fn the_cone_inertia_matches_a_quadrature_of_the_solid() {
    let (r, hh) = (3.0, 2.0);
    let (vol, com, q) = quadrature([-r, -hh, -r], [r, hh, r], N, |x, y, z| {
        // axis Y, base at y = −hh, apex at y = +hh: radius shrinks linearly.
        let t = (hh - y) / (2.0 * hh);
        (-hh..=hh).contains(&y) && (x * x + z * z).sqrt() <= r * t
    });
    let c = Cone::new(Vec3Fix::ZERO, fx(r), fx(hh));
    assert_close(vol, c.volume().to_f64(), GRID, "cone volume vs quadrature");
    assert!(
        (com[1] - (-hh / 2.0)).abs() < 0.02,
        "centre of mass at {com:?}"
    );
    let m = vol;
    let i = arr(c.inertia_diagonal(fx(m)));
    for k in 0..3 {
        assert_close(
            i[k],
            q[k],
            GRID,
            &format!("cone inertia axis {k} vs quadrature"),
        );
    }
}

// ---------------------------------------------------------------------------
// Cylinder, ellipsoid, torus — the formulas are right, the π was not
// ---------------------------------------------------------------------------

#[test]
fn the_cylinder_volume_and_inertia_are_the_textbook_ones() {
    let (r, hh) = (1.5, 2.5);
    let h = 2.0 * hh;
    let c = Cylinder::new(Vec3Fix::ZERO, fx(hh), fx(r));
    assert_close(
        c.volume().to_f64(),
        PI * r * r * h,
        EXACT,
        "cylinder volume",
    );
    let m = 7.0;
    let i = arr(c.inertia_diagonal(fx(m)));
    assert_close(
        i[0],
        m * (3.0 * r * r + h * h) / 12.0,
        EXACT,
        "cylinder Ixx",
    );
    assert_close(i[1], 0.5 * m * r * r, EXACT, "cylinder Iyy");
    let (vol, _, q) = quadrature([-r, -hh, -r], [r, hh, r], N, |x, _, z| {
        (x * x + z * z).sqrt() <= r
    });
    let i = arr(c.inertia_diagonal(fx(vol)));
    for k in 0..3 {
        assert_close(
            i[k],
            q[k],
            GRID,
            &format!("cylinder inertia axis {k} vs quadrature"),
        );
    }
}

#[test]
fn the_ellipsoid_volume_and_inertia_are_the_textbook_ones() {
    let (a, b, c) = (2.0, 1.0, 3.0);
    let e = Ellipsoid::new(Vec3Fix::ZERO, v3(a, b, c));
    assert_close(
        e.volume().to_f64(),
        4.0 / 3.0 * PI * a * b * c,
        EXACT,
        "ellipsoid volume",
    );
    let m = 3.0;
    let i = arr(e.inertia_diagonal(fx(m)));
    assert_close(i[0], m * (b * b + c * c) / 5.0, EXACT, "ellipsoid Ixx");
    assert_close(i[1], m * (a * a + c * c) / 5.0, EXACT, "ellipsoid Iyy");
    assert_close(i[2], m * (a * a + b * b) / 5.0, EXACT, "ellipsoid Izz");
    let (vol, _, q) = quadrature([-a, -b, -c], [a, b, c], N, |x, y, z| {
        (x / a).powi(2) + (y / b).powi(2) + (z / c).powi(2) <= 1.0
    });
    let i = arr(e.inertia_diagonal(fx(vol)));
    for k in 0..3 {
        assert_close(
            i[k],
            q[k],
            GRID,
            &format!("ellipsoid inertia axis {k} vs quadrature"),
        );
    }
}

#[test]
fn the_torus_volume_and_inertia_are_the_textbook_ones() {
    let (big, small) = (3.0, 1.0);
    let t = Torus::new(Vec3Fix::ZERO, fx(big), fx(small));
    assert_close(
        t.volume().to_f64(),
        2.0 * PI * PI * big * small * small,
        EXACT,
        "torus volume",
    );
    let m = 4.0;
    let i = arr(t.inertia_diagonal(fx(m)));
    assert_close(
        i[0],
        m * (0.625 * small * small + 0.5 * big * big),
        EXACT,
        "torus Ixx",
    );
    assert_close(
        i[1],
        m * (0.75 * small * small + big * big),
        EXACT,
        "torus Iyy",
    );
    let ext = big + small;
    let (vol, _, q) = quadrature([-ext, -small, -ext], [ext, small, ext], 64, |x, y, z| {
        ((x * x + z * z).sqrt() - big).powi(2) + y * y <= small * small
    });
    let i = arr(t.inertia_diagonal(fx(vol)));
    for k in 0..3 {
        assert_close(
            i[k],
            q[k],
            0.05,
            &format!("torus inertia axis {k} vs quadrature"),
        );
    }
}

// ---------------------------------------------------------------------------
// Wedge — triangular prism, apex at +height/2, centroid at −height/6
// ---------------------------------------------------------------------------

#[test]
fn the_wedge_inertia_about_the_centre_of_mass_is_the_textbook_one() {
    let (w, h, d) = (4.0, 3.0, 2.0);
    let wedge = Wedge::new(Vec3Fix::ZERO, fx(w), fx(h), fx(d));
    assert_close(
        wedge.volume().to_f64(),
        0.5 * w * h * d,
        EXACT,
        "wedge volume",
    );
    let m = 6.0;
    let i = arr(wedge.inertia_diagonal(fx(m)));
    // Isosceles triangle b = w, height h, about its centroid: Ihoriz = h²/18 per
    // unit mass, Ivert = w²/24, Iperp = Ihoriz + Ivert; the prism adds d²/12 about
    // the two axes in the cross-section's plane.
    assert_close(i[0], m * (h * h / 18.0 + d * d / 12.0), EXACT, "wedge Ixx");
    assert_close(i[1], m * (w * w / 24.0 + d * d / 12.0), EXACT, "wedge Iyy");
    assert_close(i[2], m * (w * w / 24.0 + h * h / 18.0), EXACT, "wedge Izz");
}

#[test]
fn the_wedge_inertia_matches_a_quadrature_of_the_solid() {
    let (w, h, d) = (4.0, 3.0, 2.0);
    let (vol, com, q) = quadrature(
        [-w / 2.0, -h / 2.0, -d / 2.0],
        [w / 2.0, h / 2.0, d / 2.0],
        N,
        |x, y, z| {
            // triangle (−w/2, −h/2) (w/2, −h/2) (0, h/2): half-width shrinks with y.
            let t = (h / 2.0 - y) / h;
            z.abs() <= d / 2.0 && x.abs() <= w / 2.0 * t
        },
    );
    let wedge = Wedge::new(Vec3Fix::ZERO, fx(w), fx(h), fx(d));
    assert_close(
        vol,
        wedge.volume().to_f64(),
        GRID,
        "wedge volume vs quadrature",
    );
    assert!(
        (com[1] - (-h / 6.0)).abs() < 0.02,
        "centre of mass at {com:?}"
    );
    let i = arr(wedge.inertia_diagonal(fx(vol)));
    for k in 0..3 {
        assert_close(
            i[k],
            q[k],
            GRID,
            &format!("wedge inertia axis {k} vs quadrature"),
        );
    }
}

// ===========================================================================
// `Shape` and `PhysicsWorld::add_shaped_body`
// ===========================================================================

use alice_physics::shape::{Shape, ShapeError};
use alice_physics::solver::{PhysicsWorld, SolverConfig};

/// The six shapes, with dimensions chosen so no two of them coincide
/// (`r ≠ h`, three different semi-axes), and the `inside` predicate and bounding
/// box of the same solid for the quadrature. The predicate is written here from
/// the geometry, not taken from the primitive.
type Inside = Box<dyn Fn(f64, f64, f64) -> bool>;

/// `(name, shape, bounding box low corner, high corner, inside predicate)`.
type Solid = (&'static str, Shape, [f64; 3], [f64; 3], Inside);

fn solids() -> Vec<Solid> {
    let (bx, by, bz) = (1.0, 2.0, 3.0);
    let (cr, chh) = (1.5, 2.5);
    let (kr, khh) = (3.0, 2.0);
    let (ea, eb, ec) = (2.0, 1.0, 3.0);
    let (ww, wh, wd) = (4.0, 3.0, 2.0);
    let (tr, tm) = (3.0, 1.0);
    vec![
        (
            "box",
            Shape::Box {
                half_extents: v3(bx, by, bz),
            },
            [-bx, -by, -bz],
            [bx, by, bz],
            Box::new(|_, _, _| true),
        ),
        (
            "cylinder",
            Shape::Cylinder {
                radius: fx(cr),
                half_height: fx(chh),
            },
            [-cr, -chh, -cr],
            [cr, chh, cr],
            Box::new(move |x, _, z| (x * x + z * z).sqrt() <= cr),
        ),
        (
            "cone",
            Shape::Cone {
                radius: fx(kr),
                half_height: fx(khh),
            },
            [-kr, -khh, -kr],
            [kr, khh, kr],
            Box::new(move |x, y, z| {
                let t = (khh - y) / (2.0 * khh);
                (x * x + z * z).sqrt() <= kr * t
            }),
        ),
        (
            // Tall and thin: the apex, not the base rim, is the farthest point from
            // the centre of mass, so the apex term of the bounding radius decides.
            "tall cone",
            Shape::Cone {
                radius: fx(1.0),
                half_height: fx(4.0),
            },
            [-1.0, -4.0, -1.0],
            [1.0, 4.0, 1.0],
            Box::new(|x, y, z| {
                let t = (4.0 - y) / 8.0;
                (x * x + z * z).sqrt() <= t
            }),
        ),
        (
            "ellipsoid",
            Shape::Ellipsoid {
                radii: v3(ea, eb, ec),
            },
            [-ea, -eb, -ec],
            [ea, eb, ec],
            Box::new(move |x, y, z| (x / ea).powi(2) + (y / eb).powi(2) + (z / ec).powi(2) <= 1.0),
        ),
        (
            "wedge",
            Shape::Wedge {
                width: fx(ww),
                height: fx(wh),
                depth: fx(wd),
            },
            [-ww / 2.0, -wh / 2.0, -wd / 2.0],
            [ww / 2.0, wh / 2.0, wd / 2.0],
            Box::new(move |x, y, _| {
                let t = (wh / 2.0 - y) / wh;
                x.abs() <= ww / 2.0 * t
            }),
        ),
        (
            // Tall and narrow: an apex vertex, not a base corner, is the farthest.
            "tall wedge",
            Shape::Wedge {
                width: fx(1.0),
                height: fx(6.0),
                depth: fx(1.0),
            },
            [-0.5, -3.0, -0.5],
            [0.5, 3.0, 0.5],
            Box::new(|x, y, _| {
                let t = (3.0 - y) / 6.0;
                x.abs() <= 0.5 * t
            }),
        ),
        (
            "torus",
            Shape::Torus {
                major_radius: fx(tr),
                minor_radius: fx(tm),
            },
            [-(tr + tm), -tm, -(tr + tm)],
            [tr + tm, tm, tr + tm],
            Box::new(move |x, y, z| ((x * x + z * z).sqrt() - tr).powi(2) + y * y <= tm * tm),
        ),
    ]
}

/// Volume, centre-of-mass offset and inertia of every shape agree with the
/// quadrature of the solid — the offset is measured from the geometric centre.
#[test]
fn every_shape_matches_a_quadrature_of_its_solid() {
    for (name, shape, lo, hi, inside) in solids() {
        let (vol, com, q) = quadrature(lo, hi, 56, &inside);
        assert_close(
            shape.volume().to_f64(),
            vol,
            GRID,
            &format!("{name} volume"),
        );
        let off = arr(shape.center_of_mass_offset());
        for k in 0..3 {
            assert!(
                (off[k] - com[k]).abs() < 0.03,
                "{name}: centre of mass offset axis {k} is {} but the solid's is {}",
                off[k],
                com[k]
            );
        }
        let i = arr(shape.inertia_diagonal(fx(vol)));
        for k in 0..3 {
            assert_close(i[k], q[k], 0.05, &format!("{name} inertia axis {k}"));
        }
    }
}

/// The bounding radius is about the **centre of mass**: it contains every point of
/// the solid and is reached by one (to the grid's resolution).
#[test]
fn the_bounding_radius_encloses_the_solid_about_its_centre_of_mass_and_is_tight() {
    for (name, shape, lo, hi, inside) in solids() {
        let com = arr(shape.center_of_mass_offset());
        let radius = shape.bounding_radius().to_f64();
        let n = 64;
        let step = [
            (hi[0] - lo[0]) / n as f64,
            (hi[1] - lo[1]) / n as f64,
            (hi[2] - lo[2]) / n as f64,
        ];
        let mut farthest = 0.0f64;
        for i in 0..=n {
            for j in 0..=n {
                for k in 0..=n {
                    let p = [
                        lo[0] + i as f64 * step[0],
                        lo[1] + j as f64 * step[1],
                        lo[2] + k as f64 * step[2],
                    ];
                    if inside(p[0], p[1], p[2]) {
                        let d = ((p[0] - com[0]).powi(2)
                            + (p[1] - com[1]).powi(2)
                            + (p[2] - com[2]).powi(2))
                        .sqrt();
                        farthest = farthest.max(d);
                    }
                }
            }
        }
        assert!(
            farthest <= radius + 1e-9,
            "{name}: a point of the solid is {farthest} from the centre of mass, outside the \
             bounding radius {radius}"
        );
        assert!(
            radius - farthest < 0.1,
            "{name}: the bounding radius {radius} is not tight (farthest sampled point {farthest})"
        );
    }
}

/// The textbook volume of a shape, from its dimensions — not the shape's own
/// method.
fn closed_volume(shape: &Shape) -> f64 {
    let f = Fix128::to_f64;
    match *shape {
        Shape::Box { half_extents } => {
            8.0 * f(half_extents.x) * f(half_extents.y) * f(half_extents.z)
        }
        Shape::Cylinder {
            radius,
            half_height,
        } => PI * f(radius).powi(2) * 2.0 * f(half_height),
        Shape::Cone {
            radius,
            half_height,
        } => PI * f(radius).powi(2) * 2.0 * f(half_height) / 3.0,
        Shape::Ellipsoid { radii } => 4.0 / 3.0 * PI * f(radii.x) * f(radii.y) * f(radii.z),
        Shape::Wedge {
            width,
            height,
            depth,
        } => 0.5 * f(width) * f(height) * f(depth),
        Shape::Torus {
            major_radius,
            minor_radius,
        } => 2.0 * PI * PI * f(major_radius) * f(minor_radius).powi(2),
    }
}

fn world() -> PhysicsWorld {
    PhysicsWorld::new(SolverConfig::default())
}

/// The body a world builds has the closed-form mass and the closed-form inverse
/// inertia, and a torque spins it at `τ·dt / I` — not at the unit-sphere value
/// every body used to get.
#[test]
fn a_shaped_body_has_the_mass_and_inertia_of_its_shape() {
    let density = 2.5;
    let mut w = world();
    for (name, shape, ..) in solids() {
        let idx = w
            .add_shaped_body(&shape, fx(density), v3(1.0, 2.0, 3.0))
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        let body = w.get_body(idx).expect("the body just added");
        let mass = density * shape.volume().to_f64();
        assert_close(body.mass().to_f64(), mass, EXACT, &format!("{name} mass"));
        let inertia = arr(shape.inertia_diagonal(fx(mass)));
        let inv = arr(body.inv_inertia);
        for k in 0..3 {
            assert_close(
                inv[k],
                1.0 / inertia[k],
                1e-11,
                &format!("{name} inverse inertia axis {k}"),
            );
        }
        assert_eq!(
            arr(body.position),
            [1.0, 2.0, 3.0],
            "{name}: the body sits where it was put"
        );
        // A torque about each principal axis for dt = 1/64 gives ω = τ dt / I.
        let dt = fx(1.0 / 64.0);
        for k in 0..3 {
            let mut b = *w.get_body(idx).expect("body");
            let mut torque = [0.0; 3];
            torque[k] = 10.0;
            b.add_torque(v3(torque[0], torque[1], torque[2]), dt);
            let omega = arr(b.angular_velocity);
            assert_close(
                omega[k],
                10.0 / 64.0 / inertia[k],
                1e-11,
                &format!("{name} ω axis {k}"),
            );
        }
    }
}

/// A ray from outside meets the body at its bounding sphere: the world was handed
/// the shape's bounding radius as the collision radius.
#[test]
fn the_world_uses_the_bounding_radius_as_the_collision_radius() {
    for (name, shape, ..) in solids() {
        let mut w = world();
        w.add_shaped_body(&shape, fx(1.0), Vec3Fix::ZERO)
            .expect("valid shape");
        let (idx, t) = w
            .raycast(v3(50.0, 0.0, 0.0), v3(-1.0, 0.0, 0.0), fx(100.0))
            .unwrap_or_else(|| panic!("{name}: the ray missed the body"));
        assert_eq!(idx, 0);
        assert_close(
            t.to_f64(),
            50.0 - shape.bounding_radius().to_f64(),
            1e-9,
            &format!("{name} hit distance"),
        );
    }
}

#[test]
fn degenerate_shapes_and_densities_are_refused() {
    let mut w = world();
    let unit = Shape::Box {
        half_extents: v3(1.0, 1.0, 1.0),
    };
    for density in [0.0, -1.0] {
        assert_eq!(
            w.add_shaped_body(&unit, fx(density), Vec3Fix::ZERO),
            Err(ShapeError::NonPositiveDensity),
            "density {density}"
        );
    }
    let flat = [
        Shape::Box {
            half_extents: v3(1.0, 0.0, 1.0),
        },
        Shape::Box {
            half_extents: v3(1.0, -1.0, 1.0),
        },
        Shape::Cylinder {
            radius: fx(0.0),
            half_height: fx(1.0),
        },
        Shape::Cylinder {
            radius: fx(1.0),
            half_height: fx(-1.0),
        },
        Shape::Cone {
            radius: fx(0.0),
            half_height: fx(1.0),
        },
        Shape::Ellipsoid {
            radii: v3(1.0, 0.0, 1.0),
        },
        Shape::Wedge {
            width: fx(1.0),
            height: fx(0.0),
            depth: fx(1.0),
        },
        Shape::Torus {
            major_radius: fx(1.0),
            minor_radius: fx(0.0),
        },
        // a torus whose tube is wider than its ring has no hole to speak of
        Shape::Torus {
            major_radius: fx(1.0),
            minor_radius: fx(2.0),
        },
    ];
    for shape in &flat {
        assert_eq!(
            w.add_shaped_body(shape, fx(1.0), Vec3Fix::ZERO),
            Err(ShapeError::DegenerateShape),
            "{shape:?}"
        );
    }
    assert_eq!(w.body_count(), 0, "a refused shape must not add a body");
}

/// `Fix128` wraps: a shape so large its mass overflows would otherwise come back
/// negative or tiny, and a body with that mass would be silently wrong.
#[test]
fn a_mass_that_does_not_fit_is_refused_not_wrapped() {
    let mut w = world();
    let huge = Shape::Box {
        half_extents: v3(1.0e6, 1.0e6, 1.0e6),
    };
    assert_eq!(
        w.add_shaped_body(&huge, fx(1.0e3), Vec3Fix::ZERO),
        Err(ShapeError::MassNotRepresentable)
    );
    assert_eq!(w.body_count(), 0);
}

/// Whatever the size, a shape either becomes a body whose mass is the closed-form
/// one, or is refused — it is never accepted with a mass that wrapped. Swept over
/// many decades of size and density, because the sizes at which `Fix128` wraps
/// depend on the shape.
#[test]
fn a_shape_is_never_accepted_with_a_mass_that_wrapped() {
    let mut accepted = 0;
    let mut refused = 0;
    for exp in -3i32..=8 {
        let side = 10f64.powi(exp);
        for density in [1.0e-3, 1.0, 7.8e3, 1.0e9] {
            for (name, shape, ..) in solids() {
                // Scale the solid's dimensions by `side`.
                let scaled = match shape {
                    Shape::Box { half_extents } => Shape::Box {
                        half_extents: v3(
                            half_extents.x.to_f64() * side,
                            half_extents.y.to_f64() * side,
                            half_extents.z.to_f64() * side,
                        ),
                    },
                    Shape::Cylinder {
                        radius,
                        half_height,
                    } => Shape::Cylinder {
                        radius: fx(radius.to_f64() * side),
                        half_height: fx(half_height.to_f64() * side),
                    },
                    Shape::Cone {
                        radius,
                        half_height,
                    } => Shape::Cone {
                        radius: fx(radius.to_f64() * side),
                        half_height: fx(half_height.to_f64() * side),
                    },
                    Shape::Ellipsoid { radii } => Shape::Ellipsoid {
                        radii: v3(
                            radii.x.to_f64() * side,
                            radii.y.to_f64() * side,
                            radii.z.to_f64() * side,
                        ),
                    },
                    Shape::Wedge {
                        width,
                        height,
                        depth,
                    } => Shape::Wedge {
                        width: fx(width.to_f64() * side),
                        height: fx(height.to_f64() * side),
                        depth: fx(depth.to_f64() * side),
                    },
                    Shape::Torus {
                        major_radius,
                        minor_radius,
                    } => Shape::Torus {
                        major_radius: fx(major_radius.to_f64() * side),
                        minor_radius: fx(minor_radius.to_f64() * side),
                    },
                };
                let mut w = world();
                match w.add_shaped_body(&scaled, fx(density), Vec3Fix::ZERO) {
                    Ok(idx) => {
                        accepted += 1;
                        let body = w.get_body(idx).expect("body");
                        let mass = body.mass().to_f64();
                        let want = density * closed_volume(&scaled);
                        // The body stores `1 / mass`, quantised to 2⁻⁶⁴, so a mass this
                        // large is known to a few parts in 2⁶⁴ / mass⁻¹ — a few ulps of
                        // the reciprocal — not to `Fix128`'s own resolution.
                        let tol = 1e-6 + want * 4.0 * 2f64.powi(-64);
                        assert!(
                            mass > 0.0 && rel(mass, want) < tol,
                            "{name} at size 1e{exp}, density {density}: accepted with mass {mass}, \
                             closed form {want}"
                        );
                        let inv = arr(body.inv_inertia);
                        assert!(
                            inv.iter().all(|v| *v > 0.0),
                            "{name} at size 1e{exp}: accepted with inverse inertia {inv:?}"
                        );
                    }
                    Err(e) => {
                        refused += 1;
                        assert!(
                            matches!(e, ShapeError::MassNotRepresentable),
                            "{name} at size 1e{exp}: refused as {e:?}, but the dimensions are valid"
                        );
                    }
                }
            }
        }
    }
    assert!(
        accepted > 0 && refused > 0,
        "the sweep must cross the limit both ways"
    );
}

/// Below the resolution of `Fix128` the inertia rounds to zero, and the reciprocal
/// would divide by it: refused, with the one error that says the numbers do not fit.
#[test]
fn a_shape_too_small_for_the_arithmetic_is_refused() {
    let mut w = world();
    let tiny = Shape::Box {
        half_extents: v3(1.0e-5, 1.0e-5, 1.0e-5),
    };
    assert_eq!(
        w.add_shaped_body(&tiny, fx(1.0), Vec3Fix::ZERO),
        Err(ShapeError::MassNotRepresentable)
    );
    assert_eq!(w.body_count(), 0);
}
