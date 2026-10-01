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

/// 格子の全 face が bit 一致するか (`MacGrid` は `PartialEq` を持たない)。
fn same_faces(a: &MacGrid, b: &MacGrid) -> bool {
    a.u == b.u && a.v == b.v && a.w == b.w
}

/// `Fix128` の加算・乗算は mod 2^128 の wrapping なので、巨大値は panic しない
/// ことより「どの値になるか」が仕様。領域外の位置は無視される (`p2g_normalized`
/// の左側 skip と `split` の遠側 skip) ので、
/// (a) 巨大位置だけの粒子は格子を 1 bit も変えない
/// (b) 領域内の粒子と同居しても、領域内の粒子だけの結果と bit 一致する
/// (wrap して領域内に落ちて deposit されると (a) (b) のどちらかが崩れる)。
#[test]
fn huge_positions_are_ignored_bit_for_bit() {
    let big = [
        ("i64::MAX/4", Fix128::from_int(i64::MAX / 4)),
        ("1<<40", Fix128::from_int(1 << 40)),
        ("1<<32", Fix128::from_int(1 << 32)),
        ("-(i64::MAX/4)", Fix128::from_int(-(i64::MAX / 4))),
        ("-(1<<40)", Fix128::from_int(-(1 << 40))),
        ("-(1<<32)", Fix128::from_int(-(1 << 32))),
    ];
    let inside = (v3(q(5, 4), q(5, 4), q(5, 4)), vel(1, 2, 3));
    let mut only_inside = MacGrid::new(4, 4, 4, Fix128::ONE);
    p2g_normalized(&mut only_inside, &[inside]);
    let empty = MacGrid::new(4, 4, 4, Fix128::ONE);
    assert!(
        !same_faces(&only_inside, &empty),
        "the in-domain particle must deposit something, or (b) is vacuous"
    );
    for (name, b) in big {
        // 全成分が巨大、および 1 成分だけ巨大 (他は領域内) の両方。
        let mid = q(5, 4);
        for (axis, pos) in [
            ("xyz", v3(b, b, b)),
            ("x", v3(b, mid, mid)),
            ("y", v3(mid, b, mid)),
            ("z", v3(mid, mid, b)),
        ] {
            let mut g = MacGrid::new(4, 4, 4, Fix128::ONE);
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                p2g_normalized(&mut g, &[(pos, vel(1, 2, 3))]);
            }));
            assert!(r.is_ok(), "position {name} on {axis}: panicked");
            assert!(
                same_faces(&g, &empty),
                "position {name} on {axis}: the grid changed"
            );

            let mut g = MacGrid::new(4, 4, 4, Fix128::ONE);
            p2g_normalized(
                &mut g,
                &[(pos, vel(7, 8, 9)), inside, (pos, vel(-5, 6, -7))],
            );
            assert!(
                same_faces(&g, &only_inside),
                "position {name} on {axis}: changed the in-domain particle's result"
            );
        }
    }
}

/// 巨大な速度は wrap しない範囲 (同じ face に載る粒子 2 個で |v|·Σw < 2^63) なら
/// 閉形式: 正規化 P2G は到達した face に粒子速度そのものを置く。到達する face の
/// 集合は速度 1 の粒子と同じ (速度で変わらない) ので、基準の集合を 1 で取り、
/// 巨大速度の格子はその集合上で厳密に `b`、集合外で 0。`1<<32` は二乗が 2^64 で
/// 0 に巻き戻る境目、`i64::MAX/4` は 2 粒子の和が 2^62 に届く上限側。
#[test]
fn huge_velocities_reproduce_the_particle_velocity_exactly() {
    let big = [
        ("i64::MAX/4", Fix128::from_int(i64::MAX / 4)),
        ("-(i64::MAX/4)", Fix128::from_int(-(i64::MAX / 4))),
        ("1<<40", Fix128::from_int(1 << 40)),
        ("1<<32", Fix128::from_int(1 << 32)),
    ];
    let pos = v3(q(5, 4), q(5, 4), q(5, 4));
    let mut reference = MacGrid::new(4, 4, 4, Fix128::ONE);
    p2g_normalized(&mut reference, &[(pos, vel(1, 1, 1)), (pos, vel(1, 1, 1))]);
    let reached = |a: &[Fix128]| a.iter().filter(|x| !x.is_zero()).count();
    assert_eq!(
        (
            reached(&reference.u),
            reached(&reference.v),
            reached(&reference.w)
        ),
        (8, 8, 8),
        "the unit-velocity reference must reach 8 faces per component"
    );
    for (name, b) in big {
        let mut g = MacGrid::new(4, 4, 4, Fix128::ONE);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            p2g_normalized(&mut g, &[(pos, v3(b, b, b)), (pos, v3(b, b, b))]);
        }));
        assert!(r.is_ok(), "velocity {name}: panicked");
        for (comp, got, refv) in [
            ("u", &g.u, &reference.u),
            ("v", &g.v, &reference.v),
            ("w", &g.w, &reference.w),
        ] {
            for (i, (x, r)) in got.iter().zip(refv).enumerate() {
                let want = if r.is_zero() { Fix128::ZERO } else { b };
                assert_eq!(*x, want, "velocity {name}: {comp}[{i}]");
            }
        }
    }
}

// ============================================================================
// 厳密積累積 (項を丸めず 256 bit に足して 1 回だけ割る) の oracle
//
// 期待値は実装関数を呼んで作らない 一様場は入力そのもの、2 粒子の平均は
// i128 の組み込み除算 (0 方向切り捨て)、凸包は入力の min / max
// ============================================================================

fn raw_of(x: Fix128) -> i128 {
    (i128::from(x.hi) << 64) | i128::from(x.lo)
}

fn from_raw(r: i128) -> Fix128 {
    Fix128 {
        hi: (r >> 64) as i64,
        lo: r as u64,
    }
}

fn ulp(n: i64) -> Fix128 {
    from_raw(i128::from(n))
}

struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// [lo, hi) の位置 (raw の下位 64 bit が全域で動く非 dyadic)
    fn pos(&mut self, lo: i64, hi: i64) -> Fix128 {
        let span = (hi - lo) as u64;
        Fix128 {
            hi: lo + (self.next() % span) as i64,
            lo: self.next(),
        }
    }
    fn raw128(&mut self) -> i128 {
        (i128::from(self.next()) << 64) | i128::from(self.next())
    }
}

const SENT1: i128 = 0x1234_5678_9abc_def0_1357_9bdf_0246_8ace;
const SENT2: i128 = -0x0fed_cba9_8765_4321_0246_8ace_1357_9bdf;

fn filled(n: usize, s: i128) -> MacGrid {
    let mut g = MacGrid::new(n, n, n, Fix128::ONE);
    let s = from_raw(s);
    g.u.iter_mut().for_each(|x| *x = s);
    g.v.iter_mut().for_each(|x| *x = s);
    g.w.iter_mut().for_each(|x| *x = s);
    g
}

/// 粒子が届かない face は sentinel のまま 届いた face は初期値に依らず同じ値
/// なので sentinel を 2 通りで走らせ、両方で sentinel のままの face だけを
/// 未到達とみなす (届いたのに誤値の face は両 sentinel のどちらでもない)
fn run_with_sentinels(n: usize, ps: &[(Vec3Fix, Vec3Fix)]) -> (MacGrid, Vec<bool>) {
    let mut g1 = filled(n, SENT1);
    let mut g2 = filled(n, SENT2);
    p2g_normalized(&mut g1, ps);
    p2g_normalized(&mut g2, ps);
    let mut reached = Vec::new();
    for (a, b) in [(&g1.u, &g2.u), (&g1.v, &g2.v), (&g1.w, &g2.w)] {
        for (x, y) in a.iter().zip(b.iter()) {
            reached.push(!(raw_of(*x) == SENT1 && raw_of(*y) == SENT2));
        }
    }
    (g1, reached)
}

/// 到達した face の全部が成分ごとの `want` と bit 一致 かつ 1 face 以上届く
fn assert_uniform(n: usize, ps: &[(Vec3Fix, Vec3Fix)], want: Vec3Fix, what: &str) {
    let (g, reached) = run_with_sentinels(n, ps);
    let mut hit = 0usize;
    let mut idx = 0usize;
    for (comp, arr, w) in [
        ("u", &g.u, want.x),
        ("v", &g.v, want.y),
        ("w", &g.w, want.z),
    ] {
        for (i, x) in arr.iter().enumerate() {
            if reached[idx] {
                hit += 1;
                assert_eq!(
                    raw_of(*x),
                    raw_of(w),
                    "{what}: {comp}[{i}] raw {} != {}",
                    raw_of(*x),
                    raw_of(w)
                );
            }
            idx += 1;
        }
    }
    assert!(hit > 0, "{what}: no face reached");
}

fn x_speeds() -> Vec<(&'static str, Fix128)> {
    vec![
        ("i64::MAX", Fix128::from_int(i64::MAX)),
        ("i64::MIN", Fix128::from_int(i64::MIN)),
        (
            "raw max",
            Fix128 {
                hi: i64::MAX,
                lo: u64::MAX,
            },
        ),
        ("1/3", q(1, 3)),
        ("-7/10", q(-7, 10)),
        ("1 ulp", ulp(1)),
        ("-1 ulp", ulp(-1)),
        ("2^61", Fix128::from_int(1 << 61)),
        ("-5/2", q(-5, 2)),
    ]
}

#[test]
fn x1_a_uniform_velocity_is_exact_at_any_position_and_any_count() {
    let mut rng = XorShift(0x9e37_79b9_7f4a_7c15);
    for (name, s) in x_speeds() {
        let input = v3(s, s, s);
        // 同一点 n = 1..16
        for n in 1..=16usize {
            let p = v3(rng.pos(0, 5), rng.pos(0, 5), rng.pos(0, 5));
            let ps: Vec<_> = (0..n).map(|_| (p, input)).collect();
            assert_uniform(6, &ps, input, &format!("{name} same point n={n}"));
        }
        // 非 dyadic の乱数位置 多数
        let ps: Vec<_> = (0..300)
            .map(|_| (v3(rng.pos(0, 5), rng.pos(0, 5), rng.pos(0, 5)), input))
            .collect();
        assert_uniform(6, &ps, input, &format!("{name} scattered"));
    }
}

#[test]
fn x2_a_single_particle_gives_its_velocity_even_where_the_weight_is_one_ulp() {
    // 重み raw = 2^k の face (k = 0..=56) と v = ±1, ±1000 ulp
    for k in 0..=56u32 {
        let e = from_raw(1i128 << k);
        let coords = [
            Fix128::from_int(2) + e,
            Fix128::from_int(2) - e,
            q(5, 2) + e,
            q(5, 2) - e,
        ];
        for &vv in &[1i64, -1, 1000, -1000] {
            let s = ulp(vv);
            for &cx in &coords {
                for &cy in &coords {
                    for &cz in &coords {
                        assert_uniform(
                            5,
                            &[(v3(cx, cy, cz), v3(s, -s, s))],
                            v3(s, -s, s),
                            &format!("k={k} v={vv} ulp"),
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn x3_the_wrap_region_returns_the_particle_velocity_on_every_reached_face() {
    let speeds = [
        ("2^60", Fix128::from_int(1 << 60)),
        ("2^61", Fix128::from_int(1 << 61)),
        ("2^62", Fix128::from_int(1 << 62)),
        ("i64::MAX", Fix128::from_int(i64::MAX)),
        ("i64::MIN", Fix128::from_int(i64::MIN)),
    ];
    // 各 face に重み 0.5 ずつ届く配置 (従来の閾値の測定と同じ)
    let p = v3(q(5, 2), q(5, 2), q(5, 2));
    for (name, s) in speeds {
        for n in 1..=10usize {
            let ps: Vec<_> = (0..n).map(|_| (p, v3(s, s, s))).collect();
            assert_uniform(5, &ps, v3(s, s, s), &format!("{name} n={n}"));
        }
    }
}

#[test]
fn x4_negating_every_velocity_negates_every_face_bit_for_bit() {
    let mut rng = XorShift(0x2545_f491_4f6c_dd1d);
    for round in 0..40 {
        let n = 1 + (rng.next() % 40) as usize;
        let mut ps = Vec::new();
        let mut ngs = Vec::new();
        for _ in 0..n {
            let p = v3(rng.pos(0, 5), rng.pos(0, 5), rng.pos(0, 5));
            // 全域の raw (i128::MIN は避ける) を、大きさもばらして
            let sh = (rng.next() % 100) as u32;
            let mk = |r: &mut XorShift| {
                let x = r.raw128() >> sh;
                if x == i128::MIN {
                    0
                } else {
                    x
                }
            };
            let (a, b, c) = (mk(&mut rng), mk(&mut rng), mk(&mut rng));
            ps.push((p, v3(from_raw(a), from_raw(b), from_raw(c))));
            ngs.push((p, v3(from_raw(-a), from_raw(-b), from_raw(-c))));
        }
        let mut g = MacGrid::new(6, 6, 6, Fix128::ONE);
        let mut h = MacGrid::new(6, 6, 6, Fix128::ONE);
        p2g_normalized(&mut g, &ps);
        p2g_normalized(&mut h, &ngs);
        for (comp, a, b) in [("u", &g.u, &h.u), ("v", &g.v, &h.v), ("w", &g.w, &h.w)] {
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                assert_eq!(
                    raw_of(*x),
                    -raw_of(*y),
                    "round {round}: {comp}[{i}] not antisymmetric"
                );
            }
        }
    }
}

#[test]
fn x5_the_result_does_not_depend_on_the_particle_order() {
    let mut rng = XorShift(0x1234_5678_dead_beef);
    let ps: Vec<_> = (0..200)
        .map(|_| {
            let sh = (rng.next() % 90) as u32;
            let c = |r: &mut XorShift| from_raw(r.raw128() >> sh);
            (
                v3(rng.pos(0, 5), rng.pos(0, 5), rng.pos(0, 5)),
                v3(c(&mut rng), c(&mut rng), c(&mut rng)),
            )
        })
        .collect();
    let mut base = MacGrid::new(6, 6, 6, Fix128::ONE);
    p2g_normalized(&mut base, &ps);
    let mut rev = ps.clone();
    rev.reverse();
    let mut shuf = ps.clone();
    for i in (1..shuf.len()).rev() {
        let j = (rng.next() % (i as u64 + 1)) as usize;
        shuf.swap(i, j);
    }
    for (name, order) in [("reverse", rev), ("shuffle", shuf)] {
        let mut g = MacGrid::new(6, 6, 6, Fix128::ONE);
        p2g_normalized(&mut g, &order);
        assert_eq!(g.u, base.u, "{name}: u");
        assert_eq!(g.v, base.v, "{name}: v");
        assert_eq!(g.w, base.w, "{name}: w");
    }
}

#[test]
fn x6_every_face_stays_inside_the_min_max_of_the_particle_velocities() {
    let mut rng = XorShift(0xabcd_ef01_2345_6789);
    for round in 0..30 {
        let n = 2 + (rng.next() % 60) as usize;
        let sh = (rng.next() % 100) as u32;
        let ps: Vec<_> = (0..n)
            .map(|_| {
                let c = |r: &mut XorShift| from_raw(r.raw128() >> sh);
                (
                    v3(rng.pos(0, 5), rng.pos(0, 5), rng.pos(0, 5)),
                    v3(c(&mut rng), c(&mut rng), c(&mut rng)),
                )
            })
            .collect();
        let (g, reached) = run_with_sentinels(6, &ps);
        let range = |f: fn(&Vec3Fix) -> Fix128| {
            let all: Vec<i128> = ps.iter().map(|p| raw_of(f(&p.1))).collect();
            (*all.iter().min().unwrap(), *all.iter().max().unwrap())
        };
        let ranges = [range(|v| v.x), range(|v| v.y), range(|v| v.z)];
        let mut idx = 0;
        for (c, arr) in [&g.u, &g.v, &g.w].into_iter().enumerate() {
            for (i, x) in arr.iter().enumerate() {
                if reached[idx] {
                    let r = raw_of(*x);
                    assert!(
                        r >= ranges[c].0 && r <= ranges[c].1,
                        "round {round}: comp {c} face {i} raw {r} outside [{}, {}]",
                        ranges[c].0,
                        ranges[c].1
                    );
                }
                idx += 1;
            }
        }
    }
}

#[test]
fn x7_two_particles_give_the_exact_weighted_mean_truncated_toward_zero() {
    // A は軸方向に 2.5 (面 2 に 1/2, 面 3 に 1/2)、B は 2.25 (3/4, 1/4)
    // 面 2: (a/2 + 3b/4) / (5/4) = (2a + 3b) / 5
    // 面 3: (a/2 + b/4)  / (3/4) = (2a + b) / 3
    // 期待値は i128 の組み込み除算 (0 方向切り捨て)
    let mut rng = XorShift(0x0f0f_1234_5678_9abc);
    for round in 0..200 {
        let sh = 4 + (rng.next() % 4) as u32;
        let a = rng.raw128() >> (sh + 1);
        let b = rng.raw128() >> (sh + 1);
        for axis in 0..3 {
            let pa = [q(5, 2), q(5, 2), q(5, 2)];
            let mut pb = pa;
            pb[axis] = q(9, 4);
            let vel_of = |r: i128| v3(from_raw(r), from_raw(r), from_raw(r));
            let ps = [
                (v3(pa[0], pa[1], pa[2]), vel_of(a)),
                (v3(pb[0], pb[1], pb[2]), vel_of(b)),
            ];
            let mut g = MacGrid::new(5, 5, 5, Fix128::ONE);
            p2g_normalized(&mut g, &ps);
            let (f2, f3) = match axis {
                0 => (g.u(2, 2, 2), g.u(3, 2, 2)),
                1 => (g.v(2, 2, 2), g.v(2, 3, 2)),
                _ => (g.w(2, 2, 2), g.w(2, 2, 3)),
            };
            assert_eq!(
                raw_of(f2),
                (2 * a + 3 * b) / 5,
                "round {round} axis {axis} face 2"
            );
            assert_eq!(
                raw_of(f3),
                (2 * a + b) / 3,
                "round {round} axis {axis} face 3"
            );
        }
    }
}

#[test]
fn x10_degenerate_inputs_keep_their_bits_or_their_value() {
    let mut rng = XorShift(0x77);
    let sent = SENT1;
    let all_sent = |g: &MacGrid| {
        g.u.iter()
            .chain(&g.v)
            .chain(&g.w)
            .all(|x| raw_of(*x) == sent)
    };
    let s = Fix128::from_int(i64::MAX);
    let big = v3(s, s, s);
    // 粒子 0 個
    let mut g = filled(4, sent);
    p2g_normalized(&mut g, &[]);
    assert!(all_sent(&g));
    // 1 軸だけ負の座標 3 通り
    for axis in 0..3 {
        let mut p = [q(3, 2), q(3, 2), q(3, 2)];
        p[axis] = ulp(-1);
        let mut g = filled(4, sent);
        p2g_normalized(&mut g, &[(v3(p[0], p[1], p[2]), big)]);
        assert!(all_sent(&g), "axis {axis}");
    }
    // 領域のはるか外
    let mut g = filled(4, sent);
    p2g_normalized(
        &mut g,
        &[(v3(Fix128::from_int(1000), q(3, 2), q(3, 2)), big)],
    );
    assert!(all_sent(&g));
    // dx = 0
    let mut g = MacGrid::new(4, 4, 4, Fix128::ZERO);
    g.u.iter_mut().for_each(|x| *x = from_raw(sent));
    p2g_normalized(&mut g, &[(v3(q(3, 2), q(3, 2), q(3, 2)), big)]);
    assert!(g.u.iter().all(|x| raw_of(*x) == sent));
    // 巨大 N の全粒子同一点 (届いた face は粒子速度そのもの)
    let p = v3(rng.pos(1, 3), rng.pos(1, 3), rng.pos(1, 3));
    let ps: Vec<_> = (0..20_000).map(|_| (p, big)).collect();
    assert_uniform(4, &ps, big, "N=20000 same point");
}
