//! Oracles for the weight-normalised particle-to-grid transfer
//! `eulerian_grid::p2g_normalized`.
//!
//! The pre-existing `p2g_trilinear` only accumulates `weight * v` per face and
//! keeps no weight sum, so a particle of velocity `v` leaves `w_i * v` on a
//! face rather than `v`. A transfer that cannot reproduce a uniform field is
//! not a transfer, so every oracle here is a field the answer is known for:
//!
//! 1. a uniform velocity is reproduced exactly on every reached face
//! 2. a field linear in position is reproduced at the face centre. The
//!    particle lattice is symmetric about every face, so the weighted mean of
//!    a linear function is its value at the face — and the three components
//!    use **different** gradients, so a swapped axis or a wrong stagger offset
//!    moves the answer
//! 3. unequal weights: the mean is weighted, not a plain average, and not the
//!    unnormalised sum
//! 4. P2G followed by G2P returns the particle velocity at an off-grid point
//!
//! Every position and velocity is a dyadic rational, so each product and sum
//! is exact in `Fix128` and the asserts are `assert_eq!`, not a tolerance.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]

use alice_physics::eulerian_grid::{g2p_velocity, p2g_normalized, MacGrid};
use alice_physics::math::{Fix128, Vec3Fix};

fn q(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn v3(x: Fix128, y: Fix128, z: Fix128) -> Vec3Fix {
    Vec3Fix::new(x, y, z)
}

/// Particles at `a/2 + 1/4` for `a` in `0..2n` along each axis: spacing 1/2,
/// spanning `[1/4, n - 1/4]`, symmetric about every integer and half-integer
/// coordinate in `[1, n - 1]`. Eight particles per cell, so the weight sum on
/// an interior face is 8 and a missing normalisation shows as a factor of 8.
fn lattice(n: i64) -> Vec<Vec3Fix> {
    let m = 2 * n;
    let mut out = Vec::new();
    for a in 0..m {
        for b in 0..m {
            for c in 0..m {
                out.push(v3(q(2 * a + 1, 4), q(2 * b + 1, 4), q(2 * c + 1, 4)));
            }
        }
    }
    out
}

#[test]
fn a_uniform_velocity_is_reproduced_exactly_on_every_interior_face() {
    let n = 6usize;
    let mut g = MacGrid::new(n, n, n, Fix128::ONE);
    let u0 = v3(Fix128::from_int(3), Fix128::from_int(-2), q(5, 2));
    let ps: Vec<_> = lattice(n as i64).into_iter().map(|p| (p, u0)).collect();
    p2g_normalized(&mut g, &ps);
    for i in 1..n {
        for j in 1..n - 1 {
            for k in 1..n - 1 {
                assert_eq!(g.u(i, j, k), u0.x, "u({i},{j},{k})");
            }
        }
    }
    for i in 1..n - 1 {
        for j in 1..n {
            for k in 1..n - 1 {
                assert_eq!(g.v(i, j, k), u0.y, "v({i},{j},{k})");
            }
        }
    }
    for i in 1..n - 1 {
        for j in 1..n - 1 {
            for k in 1..n {
                assert_eq!(g.w(i, j, k), u0.z, "w({i},{j},{k})");
            }
        }
    }
}

#[test]
fn a_linear_field_is_reproduced_at_the_face_centre_with_distinct_gradients() {
    // u = 1 + 2x + 3y + 5z,  v = -1 + 7x - 2y + 4z,  w = 2 + 3x + 6y - 8z
    let field = |p: Vec3Fix| {
        let (x, y, z) = (p.x, p.y, p.z);
        let k = Fix128::from_int;
        v3(
            k(1) + k(2) * x + k(3) * y + k(5) * z,
            k(-1) + k(7) * x - k(2) * y + k(4) * z,
            k(2) + k(3) * x + k(6) * y - k(8) * z,
        )
    };
    let n = 6usize;
    let mut g = MacGrid::new(n, n, n, Fix128::ONE);
    let ps: Vec<_> = lattice(n as i64)
        .into_iter()
        .map(|p| (p, field(p)))
        .collect();
    p2g_normalized(&mut g, &ps);
    let half = q(1, 2);
    let f = |i: usize| Fix128::from_int(i as i64);
    for i in 1..n {
        for j in 1..n - 1 {
            for k in 1..n - 1 {
                let c = v3(f(i), f(j) + half, f(k) + half);
                assert_eq!(g.u(i, j, k), field(c).x, "u({i},{j},{k})");
            }
        }
    }
    for i in 1..n - 1 {
        for j in 1..n {
            for k in 1..n - 1 {
                let c = v3(f(i) + half, f(j), f(k) + half);
                assert_eq!(g.v(i, j, k), field(c).y, "v({i},{j},{k})");
            }
        }
    }
    for i in 1..n - 1 {
        for j in 1..n - 1 {
            for k in 1..n {
                let c = v3(f(i) + half, f(j) + half, f(k));
                assert_eq!(g.w(i, j, k), field(c).z, "w({i},{j},{k})");
            }
        }
    }
}

#[test]
fn unequal_weights_give_the_weighted_mean_not_the_sum_or_the_plain_average() {
    // Both particles sit at y = z = 2.5 so only the x weights matter for u.
    // B at x = 0.5 (faces 0 and 1, weight 1/2 each), A at x = 1.25 (face 1
    // weight 3/4, face 2 weight 1/4).  Face 1: (3/4 * 2 + 1/2 * 7) / (5/4) = 4.
    // The plain average of 2 and 7 is 4.5; the unnormalised sum is 5.
    let mut g = MacGrid::new(6, 6, 6, Fix128::ONE);
    let a = (
        v3(q(5, 4), q(5, 2), q(5, 2)),
        v3(Fix128::from_int(2), Fix128::ZERO, Fix128::ZERO),
    );
    let b = (
        v3(q(1, 2), q(5, 2), q(5, 2)),
        v3(Fix128::from_int(7), Fix128::ZERO, Fix128::ZERO),
    );
    p2g_normalized(&mut g, &[a, b]);
    assert_eq!(g.u(1, 2, 2), Fix128::from_int(4));
    // Faces reached by one particle only carry that particle's velocity.
    assert_eq!(g.u(0, 2, 2), Fix128::from_int(7));
    assert_eq!(g.u(2, 2, 2), Fix128::from_int(2));
}

#[test]
fn p2g_then_g2p_returns_the_particle_velocity_off_the_grid() {
    let mut g = MacGrid::new(6, 6, 6, Fix128::ONE);
    let pos = v3(q(37, 16), q(59, 16), q(23, 16));
    let vel = v3(
        Fix128::from_int(3),
        Fix128::from_int(-5),
        Fix128::from_int(7),
    );
    p2g_normalized(&mut g, &[(pos, vel)]);
    assert_eq!(g2p_velocity(&g, pos), vel);
}
