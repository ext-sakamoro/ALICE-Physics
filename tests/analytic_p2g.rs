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

// ---------------------------------------------------------------------------
// 退化入力・panic 耐性
// ---------------------------------------------------------------------------

/// 全 face に 0 以外の既知値 (面ごとに異なる) を入れた格子。
/// 値は整数なので dyadic で厳密。
fn seeded_grid(n: usize, dx: Fix128) -> MacGrid {
    let mut g = MacGrid::new(n, n, n, dx);
    for (i, x) in g.u.iter_mut().enumerate() {
        *x = Fix128::from_int(100 + i as i64);
    }
    for (i, x) in g.v.iter_mut().enumerate() {
        *x = Fix128::from_int(-200 - i as i64);
    }
    for (i, x) in g.w.iter_mut().enumerate() {
        *x = Fix128::from_int(300 + 2 * i as i64);
    }
    g
}

fn assert_same_faces(a: &MacGrid, b: &MacGrid) {
    for (name, x, y) in [("u", &a.u, &b.u), ("v", &a.v, &b.v), ("w", &a.w, &b.w)] {
        assert_eq!(x.len(), y.len(), "{name} length");
        for (i, (p, r)) in x.iter().zip(y).enumerate() {
            assert_eq!(p, r, "{name} face index {i} changed");
        }
    }
}

fn vel(x: i64, y: i64, z: i64) -> Vec3Fix {
    v3(
        Fix128::from_int(x),
        Fix128::from_int(y),
        Fix128::from_int(z),
    )
}

#[test]
fn an_empty_particle_slice_leaves_every_face_untouched() {
    let mut g = seeded_grid(4, Fix128::ONE);
    let before = g.clone();
    p2g_normalized(&mut g, &[]);
    assert_same_faces(&g, &before);
}

#[test]
fn a_zero_dx_grid_is_left_untouched_without_panicking() {
    let mut g = seeded_grid(4, Fix128::ZERO);
    let before = g.clone();
    let ps = [(v3(q(3, 2), q(3, 2), q(3, 2)), vel(7, 8, 9))];
    p2g_normalized(&mut g, &ps);
    assert_same_faces(&g, &before);
}

#[test]
fn particles_beyond_the_far_side_change_nothing() {
    // 正の側に領域外: x だけ / y だけ / z だけ / 全軸 / 遠方。
    // 各 particle は 1 軸でも face の範囲外なので全 deposit が範囲条件で落ちる
    let n = 4usize;
    let inside = q(5, 4);
    let out = Fix128::from_int(n as i64 + 2);
    let cases = [
        v3(out, inside, inside),
        v3(inside, out, inside),
        v3(inside, inside, out),
        v3(out, out, out),
        v3(
            Fix128::from_int(1 << 20),
            Fix128::from_int(1 << 20),
            Fix128::from_int(1 << 20),
        ),
    ];
    for pos in cases {
        let mut g = seeded_grid(n, Fix128::ONE);
        let before = g.clone();
        p2g_normalized(&mut g, &[(pos, vel(7, 8, 9))]);
        assert_same_faces(&g, &before);
    }
}

#[test]
fn far_side_particles_do_not_disturb_values_from_in_range_particles() {
    // 領域外が混ざっても結果は領域内の粒子だけのときと同一
    let n = 6usize;
    let inner = (v3(q(5, 4), q(5, 2), q(5, 2)), vel(2, 3, 4));
    let mut alone = seeded_grid(n, Fix128::ONE);
    p2g_normalized(&mut alone, &[inner]);
    let far = (
        v3(
            Fix128::from_int(50),
            Fix128::from_int(50),
            Fix128::from_int(50),
        ),
        vel(1000, 1000, 1000),
    );
    let mut mixed = seeded_grid(n, Fix128::ONE);
    p2g_normalized(&mut mixed, &[far, inner, far]);
    assert_same_faces(&mixed, &alone);
    assert_eq!(mixed.u(1, 2, 2), Fix128::from_int(2));
}

#[test]
fn negative_coordinate_particles_do_not_panic_and_keep_in_range_values() {
    // 負座標は split が face 0 に clamp する (frac = 0) ので face 0 近傍には
    // deposit されうる。panic しないこと、face 0 から離れた領域内の値を壊さない
    // ことを見る
    let n = 6usize;
    let inner = (v3(q(5, 4), q(5, 2), q(5, 2)), vel(2, 3, 4));
    let neg = (
        v3(
            Fix128::from_int(-5),
            Fix128::from_int(-7),
            Fix128::from_int(-9),
        ),
        vel(1000, 1000, 1000),
    );
    let mut alone = MacGrid::new(n, n, n, Fix128::ONE);
    p2g_normalized(&mut alone, &[inner]);
    let mut mixed = MacGrid::new(n, n, n, Fix128::ONE);
    p2g_normalized(&mut mixed, &[neg, inner]);
    assert_eq!(mixed.u(1, 2, 2), alone.u(1, 2, 2));
    assert_eq!(mixed.u(2, 2, 2), alone.u(2, 2, 2));
}

#[test]
fn negative_coordinate_only_particles_leave_the_grid_untouched() {
    // 領域外の粒子のみのとき格子不変であるべき
    let mut g = seeded_grid(4, Fix128::ONE);
    let before = g.clone();
    let ps = [(
        v3(
            Fix128::from_int(-5),
            Fix128::from_int(-7),
            Fix128::from_int(-9),
        ),
        vel(7, 8, 9),
    )];
    p2g_normalized(&mut g, &ps);
    assert_same_faces(&g, &before);
}

#[test]
fn a_particle_negative_on_one_axis_alone_is_dropped() {
    // 1 軸だけ負で、他の軸は領域内。除外条件が 3 軸それぞれに効いていること
    // (全軸が負の粒子だけでは、どれか 1 軸の条件が欠けても通ってしまう)
    let inside = q(9, 4);
    for axis in 0..3 {
        let mut c = [inside, inside, inside];
        c[axis] = q(-1, 4);
        let mut g = seeded_grid(4, Fix128::ONE);
        let before = g.clone();
        p2g_normalized(&mut g, &[(v3(c[0], c[1], c[2]), vel(7, 8, 9))]);
        assert_same_faces(&g, &before);
    }
}

#[test]
fn faces_no_particle_reaches_keep_their_previous_value() {
    // 全 face に 0 以外を seed、粒子は 1 個。届く face は粒子速度に上書き、
    // 届かない face は seed のまま
    let n = 6usize;
    let mut g = seeded_grid(n, Fix128::ONE);
    let before = g.clone();
    p2g_normalized(&mut g, &[(v3(q(5, 4), q(5, 2), q(5, 2)), vel(2, 3, 4))]);
    // u の y,z 重みは (5/2 - 1/2) = 2 ちょうどで j = k = 2 のみ、x は face 1, 2
    for i in 0..=n {
        for j in 0..n {
            for k in 0..n {
                let reached = (i == 1 || i == 2) && j == 2 && k == 2;
                let want = if reached {
                    Fix128::from_int(2)
                } else {
                    before.u(i, j, k)
                };
                assert_eq!(g.u(i, j, k), want, "u({i},{j},{k})");
            }
        }
    }
    // v の x 重みは 5/4 - 1/2 = 3/4 で i = 0, 1、z は 2 ちょうどで k = 2、
    // y は 5/2 で j = 2, 3
    for i in 0..n {
        for j in 0..=n {
            for k in 0..n {
                let reached = (i == 0 || i == 1) && (j == 2 || j == 3) && k == 2;
                let want = if reached {
                    Fix128::from_int(3)
                } else {
                    before.v(i, j, k)
                };
                assert_eq!(g.v(i, j, k), want, "v({i},{j},{k})");
            }
        }
    }
    // w の x 重みは i = 0, 1、y は j = 2、z は 5/2 で k = 2, 3
    for i in 0..n {
        for j in 0..n {
            for k in 0..=n {
                let reached = (i == 0 || i == 1) && j == 2 && (k == 2 || k == 3);
                let want = if reached {
                    Fix128::from_int(4)
                } else {
                    before.w(i, j, k)
                };
                assert_eq!(g.w(i, j, k), want, "w({i},{j},{k})");
            }
        }
    }
}

/// 1 軸ぶんの届く face index 集合。`p4` は位置 (4 倍した整数)、`off4` は stagger
/// offset (4 倍)、`faces` はその軸の face 数。重み 0 の face は届かない
fn reach_axis(p4: i64, off4: i64, faces: usize) -> Vec<usize> {
    let s = p4 - off4;
    let base = s.div_euclid(4);
    let frac = s.rem_euclid(4);
    let mut out = Vec::new();
    if (base as usize) < faces {
        out.push(base as usize);
    }
    if frac > 0 && ((base + 1) as usize) < faces {
        out.push((base + 1) as usize);
    }
    out
}

#[test]
fn a_single_particle_reaches_exactly_the_faces_its_weights_and_the_bounds_allow() {
    // 位置は 1/4 刻み。軸ごとに独立な参照モデル (reach_axis) が届く face を決め、
    // 届く face は粒子速度、届かない face は seed のまま。上端をまたぐ位置
    // (x = n + 1/4 など) では、範囲条件が外れると 1 つ先の face が行を折り返して
    // 別の face (u(0, j+1, k) など) を壊す
    let n = 4usize;
    let nn = n as i64;
    let positions: [(i64, i64, i64); 8] = [
        (5, 10, 10),          // 内部
        (4 * nn + 1, 10, 10), // x が上端をまたぐ
        (10, 4 * nn + 1, 10), // y
        (10, 10, 4 * nn + 1), // z
        (4 * nn + 1, 4 * nn + 1, 4 * nn + 1),
        (4 * nn - 1, 4 * nn - 1, 4 * nn - 1),
        (0, 0, 0),                // 下端ちょうど
        (4 * nn, 4 * nn, 4 * nn), // 上端ちょうど
    ];
    for (px, py, pz) in positions {
        let mut g = seeded_grid(n, Fix128::ONE);
        let before = g.clone();
        p2g_normalized(&mut g, &[(v3(q(px, 4), q(py, 4), q(pz, 4)), vel(2, 3, 4))]);
        let (nu, nv, nw) = (n + 1, n + 1, n + 1);
        let ax = |p: i64, off: i64, f: usize| reach_axis(p, off, f);
        let u_ix = (ax(px, 0, nu), ax(py, 2, n), ax(pz, 2, n));
        let v_ix = (ax(px, 2, n), ax(py, 0, nv), ax(pz, 2, n));
        let w_ix = (ax(px, 2, n), ax(py, 2, n), ax(pz, 0, nw));
        for i in 0..=n {
            for j in 0..n {
                for k in 0..n {
                    let r = u_ix.0.contains(&i) && u_ix.1.contains(&j) && u_ix.2.contains(&k);
                    let want = if r {
                        Fix128::from_int(2)
                    } else {
                        before.u(i, j, k)
                    };
                    assert_eq!(g.u(i, j, k), want, "u({i},{j},{k}) @ ({px},{py},{pz})/4");
                }
            }
        }
        for i in 0..n {
            for j in 0..=n {
                for k in 0..n {
                    let r = v_ix.0.contains(&i) && v_ix.1.contains(&j) && v_ix.2.contains(&k);
                    let want = if r {
                        Fix128::from_int(3)
                    } else {
                        before.v(i, j, k)
                    };
                    assert_eq!(g.v(i, j, k), want, "v({i},{j},{k}) @ ({px},{py},{pz})/4");
                }
            }
        }
        for i in 0..n {
            for j in 0..n {
                for k in 0..=n {
                    let r = w_ix.0.contains(&i) && w_ix.1.contains(&j) && w_ix.2.contains(&k);
                    let want = if r {
                        Fix128::from_int(4)
                    } else {
                        before.w(i, j, k)
                    };
                    assert_eq!(g.w(i, j, k), want, "w({i},{j},{k}) @ ({px},{py},{pz})/4");
                }
            }
        }
    }
}

#[test]
fn many_coincident_particles_with_zero_velocity_give_exactly_zero() {
    // 同位置 512 個 (重み和が大きい) 速度 0。seed を入れた格子でも届いた face は 0
    let n = 6usize;
    let pos = v3(q(5, 4), q(5, 2), q(5, 2));
    let ps: Vec<_> = (0..512).map(|_| (pos, vel(0, 0, 0))).collect();
    let mut g = seeded_grid(n, Fix128::ONE);
    let before = g.clone();
    p2g_normalized(&mut g, &ps);
    assert_eq!(g.u(1, 2, 2), Fix128::ZERO);
    assert_eq!(g.u(2, 2, 2), Fix128::ZERO);
    assert_eq!(g.v(0, 2, 2), Fix128::ZERO);
    assert_eq!(g.v(1, 2, 2), Fix128::ZERO);
    assert_eq!(g.w(0, 2, 2), Fix128::ZERO);
    assert_eq!(g.w(1, 2, 2), Fix128::ZERO);
    assert_eq!(g.u(0, 0, 0), before.u(0, 0, 0));
}

#[test]
fn many_coincident_particles_still_give_the_exact_mean() {
    // 同位置 512 個、速度 2, 6, 4, 4 の繰返し。平均 = 4
    let n = 6usize;
    let pos = v3(q(5, 4), q(5, 2), q(5, 2));
    let ps: Vec<_> = (0..512)
        .map(|i| {
            let x = match i % 4 {
                0 => 2,
                1 => 6,
                _ => 4,
            };
            (pos, vel(x, 0, 0))
        })
        .collect();
    let mut g = MacGrid::new(n, n, n, Fix128::ONE);
    p2g_normalized(&mut g, &ps);
    assert_eq!(g.u(1, 2, 2), Fix128::from_int(4));
    assert_eq!(g.u(2, 2, 2), Fix128::from_int(4));
}

/// `f` が panic したらその文言を返す。出力を汚さないよう hook を一時的に外す。
fn panic_message<F: FnOnce() + std::panic::UnwindSafe>(f: F) -> Option<String> {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let r = std::panic::catch_unwind(f);
    std::panic::set_hook(prev);
    r.err().map(|e| {
        e.downcast_ref::<String>()
            .cloned()
            .or_else(|| e.downcast_ref::<&str>().map(|s| (*s).to_string()))
            .unwrap_or_else(|| "<non-string payload>".to_string())
    })
}

#[test]
fn huge_positions_do_not_panic() {
    let big = [
        ("i64::MAX/4", Fix128::from_int(i64::MAX / 4)),
        ("1<<40", Fix128::from_int(1 << 40)),
        ("-(i64::MAX/4)", Fix128::from_int(-(i64::MAX / 4))),
        ("-(1<<40)", Fix128::from_int(-(1 << 40))),
    ];
    for (name, b) in big {
        let r = panic_message(|| {
            let mut g = MacGrid::new(4, 4, 4, Fix128::ONE);
            p2g_normalized(&mut g, &[(v3(b, b, b), vel(1, 2, 3))]);
        });
        assert_eq!(r, None, "position {name}: panicked");
    }
}

#[test]
fn huge_velocities_do_not_panic() {
    let big = [
        ("i64::MAX/4", Fix128::from_int(i64::MAX / 4)),
        ("1<<40", Fix128::from_int(1 << 40)),
    ];
    for (name, b) in big {
        let r = panic_message(|| {
            let mut g = MacGrid::new(4, 4, 4, Fix128::ONE);
            let pos = v3(q(5, 4), q(5, 4), q(5, 4));
            p2g_normalized(&mut g, &[(pos, v3(b, b, b)), (pos, v3(b, b, b))]);
        });
        assert_eq!(r, None, "velocity {name}: panicked");
    }
}
