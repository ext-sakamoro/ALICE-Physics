"""Run-time oracles of the Python binding (`--features python`), against the
installed extension module.

Run against a built wheel: `pip install dist/*.whl && python python/tests/test_binding.py`.
Every value is a closed form or a property of the engine (bit-identical
replay), not a value read back from an earlier run. The runner fails when no
test ran, so an import that silently finds nothing cannot pass.
"""
from __future__ import annotations

import math
import pathlib
import re
import struct
import sys
import traceback

import numpy as np

import alice_physics as ap

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_version_is_the_crate_version():
    cargo = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    want = re.search(r'^version\s*=\s*"([^"]+)"', cargo, re.M).group(1)
    assert ap.version() == want, (ap.version(), want)


def test_free_fall_matches_the_pinned_closed_form():
    # dynamic body at y = 10, mass 1, 60 steps of 1/60 s, default config (no damping):
    # 480 substeps of h = 1/480 under g = 10, y = 10 - g h^2 n(n+1)/2 = 479/96
    # (the closed form the FFI golden pins). The binding returns f32, so y is that
    # value rounded to f32: compare with the f32 nearest 479/96, exactly.
    w = ap.PhysicsWorld()
    b = w.add_dynamic_body(0.0, 10.0, 0.0, 1.0)
    for _ in range(60):
        w.step(1.0 / 60.0)
    x, y, z = w.get_position(b)
    assert (x, z) == (0.0, 0.0), (x, z)
    want = struct.unpack("<f", struct.pack("<f", 479.0 / 96.0))[0]
    assert y == want, (y, want)


def test_a_static_body_does_not_move():
    w = ap.PhysicsWorld()
    s = w.add_static_body(1.0, 2.0, 3.0)
    w.step_n(1.0 / 60.0, 30)
    assert w.get_position(s) == (1.0, 2.0, 3.0)


def _jointed_scene():
    w = ap.PhysicsWorld()
    w.add_static_plane(0.0, 1.0, 0.0, 0.0)
    a = w.add_dynamic_body(0.0, 5.0, 0.0, 1.0)
    b = w.add_dynamic_body(1.0, 5.0, 0.0, 2.0)
    w.add_ball_joint(a, b, (0.0, 0.0, 0.0), (-1.0, 0.0, 0.0))
    return w


def test_state_digest_is_sha256_of_the_domain_and_the_snapshot():
    # oracle: hashlib over the 32 domain bytes (the SHA-256 of the domain text), then snapshot_world()
    import hashlib
    w = _jointed_scene()
    for _ in range(30):
        w.step(1.0 / 60.0)
    domain = hashlib.sha256(b"alice-physics/state-digest/v1").digest()
    want = hashlib.sha256(domain + bytes(w.snapshot_world())).digest()
    assert bytes(w.state_digest()) == want
    assert len(bytes(w.law_id())) == 32


def test_restore_world_brings_back_the_joint_and_refuses_with_its_kind():
    w = _jointed_scene()
    blob = bytes(w.snapshot_world())
    fresh = ap.PhysicsWorld()
    fresh.restore_world(blob)
    assert bytes(fresh.state_digest()) == bytes(w.state_digest())
    bad = bytearray(blob)
    bad[-20] ^= 1
    before = bytes(fresh.state_digest())
    try:
        fresh.restore_world(bytes(bad))
    except ValueError as e:
        assert str(e).startswith("ChecksumMismatch:"), e
    else:
        raise AssertionError("a corrupted snapshot was accepted")
    assert bytes(fresh.state_digest()) == before


def test_a_fixed_joint_takes_a_rotation_of_any_size():
    w = ap.PhysicsWorld()
    a = w.add_dynamic_body(0.0, 0.0, 0.0, 1.0)
    b = w.add_dynamic_body(0.0, 0.0, 0.0, 1.0)
    origin = (0.0, 0.0, 0.0)
    assert w.add_fixed_joint(a, b, origin, origin, (8.3e15, -1.5e16, 3.1e15, -2.2e17)) >= 0
    try:
        w.add_fixed_joint(a, b, origin, origin, (0.0, 0.0, 0.0, 0.0))
    except ValueError:
        pass
    else:
        raise AssertionError("the zero quaternion was accepted")


def test_a_spin_wakes_a_sleeping_body():
    # a box at rest on a floor is asleep after 240 steps; a spin makes it turn
    w = ap.PhysicsWorld()
    w.add_static_plane(0.0, 1.0, 0.0, 0.0)
    b = w.add_shaped_body(0, 0.5, 0.5, 0.5, 1000.0, 0.0, 0.5, 0.0)
    for _ in range(240):
        w.step(1.0 / 60.0)
    w.set_angular_velocity(b, 0.0, 3.0, 0.0)
    for _ in range(10):
        w.step(1.0 / 60.0)
    assert abs(w.observe_body(b)[2][3] - 1.0) > 1e-6, w.observe_body(b)


def test_set_rotation_normalises_and_refuses():
    w = ap.PhysicsWorld()
    b = w.add_dynamic_body(0.0, 5.0, 0.0, 1.0)
    w.set_rotation(b, 0.0, 0.0, 2.0, 0.0)
    q = np.asarray(w.observe_body(b)[2])  # (position, velocity, rotation (x, y, z, w), ...)
    assert np.allclose(q, [0.0, 0.0, 1.0, 0.0], atol=1e-12), q
    for bad in ((0.0, 0.0, 0.0, 0.0), (math.nan, 0.0, 0.0, 1.0)):
        try:
            w.set_rotation(b, *bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad} was accepted")
    try:
        w.set_angular_velocity(99, 0.0, 0.0, 1.0)
    except IndexError:
        pass
    else:
        raise AssertionError("an unknown body was accepted")


def test_serialized_state_replays_bit_for_bit():
    a = ap.PhysicsWorld()
    for i in range(4):
        a.add_dynamic_body(float(i), 5.0 + i, 0.0, 1.0 + i)
    a.set_velocity(1, 0.5, 0.0, -0.25)
    a.step_n(1.0 / 60.0, 10)
    blob = bytes(a.serialize_state())
    b = ap.PhysicsWorld()
    for i in range(4):
        b.add_dynamic_body(0.0, 0.0, 0.0, 1.0 + i)
    assert b.deserialize_state(blob)
    a.step_n(1.0 / 60.0, 20)
    b.step_n(1.0 / 60.0, 20)
    assert bytes(a.serialize_state()) == bytes(b.serialize_state())


def test_positions_is_an_n_by_3_array_of_the_bodies():
    w = ap.PhysicsWorld()
    w.add_dynamic_body(1.0, 2.0, 3.0, 1.0)
    w.add_static_body(-4.0, 5.0, -6.0)
    p = w.positions()
    assert isinstance(p, np.ndarray) and p.shape == (2, 3), (type(p), getattr(p, "shape", None))
    assert p[0].tolist() == [1.0, 2.0, 3.0] and p[1].tolist() == [-4.0, 5.0, -6.0]
    assert w.body_count() == 2


def test_add_bodies_batch_adds_one_body_per_row():
    w = ap.PhysicsWorld()
    rows = np.array([[0.0, 1.0, 0.0, 1.0], [2.0, 3.0, 4.0, 2.0]], dtype=np.float64)
    ids = w.add_bodies_batch(rows)
    assert list(ids) == [0, 1], ids
    assert w.positions()[1].tolist() == [2.0, 3.0, 4.0]


def test_a_ray_down_onto_a_plane_hits_at_the_height():
    w = ap.PhysicsWorld()
    w.add_static_plane(0.0, 1.0, 0.0, 0.0)
    hit = w.cast_ray((0.0, 5.0, 0.0), (0.0, -1.0, 0.0), 100.0)
    assert hit is not None
    t, point, normal = hit[0], hit[1], hit[2]
    assert abs(t - 5.0) < 1e-9, t
    assert all(abs(a - b) < 1e-9 for a, b in zip(point, (0.0, 0.0, 0.0))), point
    assert all(abs(a - b) < 1e-9 for a, b in zip(normal, (0.0, 1.0, 0.0))), normal
    assert w.cast_ray((0.0, 5.0, 0.0), (0.0, 1.0, 0.0), 100.0) is None


def test_an_unknown_body_id_raises_index_error():
    w = ap.PhysicsWorld()
    for call in (lambda: w.get_position(0), lambda: w.set_velocity(3, 0.0, 0.0, 0.0)):
        try:
            call()
        except IndexError:
            continue
        raise AssertionError("no IndexError for an unknown body id")


def test_a_non_finite_query_length_raises_value_error():
    w = ap.PhysicsWorld()
    try:
        w.cast_ray((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), math.inf)
    except ValueError:
        return
    raise AssertionError("no ValueError for max_t = inf")


def test_a_magnet_pulls_only_its_material():
    # a magnet at the origin with its moment along x; a ball of the registered material 6 cm to
    # its right and a ball of the default material 6 cm to its left fall side by side for 0.5 s:
    # the first moves toward the magnet (x below 0.06), the second keeps its x
    w = ap.PhysicsWorld()
    a = w.add_dynamic_body(0.06, 0.0, 0.0, 0.00411)
    b = w.add_dynamic_body(-0.06, 0.0, 0.0, 0.00411)
    for i in (a, b):
        w.set_collision_radius(i, 0.005)
    m = w.register_material(0.3, 0.15)
    w.set_body_material(a, m)
    assert w.add_magnet((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), 0.01, m, 3.0, 1.7e6, 7850.0) == 0
    w.step_n(1.0 / 120.0, 60)
    assert w.get_position(a)[0] < 0.06, w.get_position(a)
    # positions come back as f32: the unmoved ball is at the f32 nearest -0.06
    assert w.get_position(b)[0] == struct.unpack("<f", struct.pack("<f", -0.06))[0], w.get_position(b)


def _windy():
    # a wind straight up, U = 10 m/s, with k = 1/2 rho Cd A = 0.5 on a 2 kg ball under gravity
    w = ap.PhysicsWorld()
    ball = w.add_dynamic_body(0.0, 0.0, 0.0, 2.0)
    i = w.add_wind_zone((-1e3, -1e3, -1e3), (1e3, 1e3, 1e3), (0.0, 1.0, 0.0), 10.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    assert i == 0
    return w, ball


def test_a_ball_in_a_rising_wind_zone_reaches_the_terminal_velocity():
    # k (U - v)^2 = m g, so v = U - sqrt(m g / k), with the default world's g = 10
    w, ball = _windy()
    w.step_n(1.0 / 60.0, 900)
    want = 10.0 - math.sqrt(2.0 * 10.0 / 0.5)
    got = w.velocities()[ball]
    assert abs(got[1] - want) < 0.01 * want, got
    assert got[0] == 0.0 and got[2] == 0.0, got


def test_a_wind_zone_rewinds_and_is_part_of_the_snapshot():
    w, _ = _windy()
    w.step_n(1.0 / 60.0, 30)
    blob = bytes(w.snapshot_world())
    w.step_n(1.0 / 60.0, 30)
    first = bytes(w.snapshot_world())
    w.restore_world(blob)
    assert bytes(w.snapshot_world()) == blob
    w.step_n(1.0 / 60.0, 30)
    assert bytes(w.snapshot_world()) == first
    plain = ap.PhysicsWorld()
    plain.add_dynamic_body(0.0, 0.0, 0.0, 2.0)
    try:
        plain.restore_world(blob)
    except ValueError:
        return
    raise AssertionError("a world without the zone took its snapshot")


def test_a_bad_wind_zone_is_refused():
    w = ap.PhysicsWorld()
    box = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1.0, 0.0, 0.0))
    for bad in [
        ((1.0, 0.0, 0.0), (0.0, 1.0, 1.0), (1.0, 0.0, 0.0), 1.0, 0.0, 0.0, 1.0, 1.0, 1.0),
        ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), 1.0, 0.0, 0.0, 1.0, 1.0, 1.0),
        (*box, -1.0, 0.0, 0.0, 1.0, 1.0, 1.0),
        (*box, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0),
        (*box, float("nan"), 0.0, 0.0, 1.0, 1.0, 1.0),
    ]:
        try:
            w.add_wind_zone(*bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")
    # past the range of the fixed point, on every argument and on either side
    good = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    for i in range(len(good)):
        for value in (1e19, 1e30, sys.float_info.max, -1e19, -1e30, -sys.float_info.max):
            p = list(good)
            p[i] = value
            try:
                w.add_wind_zone(tuple(p[0:3]), tuple(p[3:6]), tuple(p[6:9]), *p[9:])
            except ValueError:
                continue
            raise AssertionError(f"accepted argument {i} = {value}")
    assert w.add_wind_zone(*box, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0) == 0


def _floating():
    # a ball of radius 1 and half the density of the fluid, whose surface is at y = 0
    w = ap.PhysicsWorld()
    ball = w.add_dynamic_body(0.0, 0.25, 0.0, 0.5 * 4.0 * math.pi / 3.0)
    w.set_collision_radius(ball, 1.0)
    assert w.add_buoyancy_zone((-1e3, -1e3, -1e3), (1e3, 0.0, 1e3), 1.0, 10.0, 8.0, 0.0) == 0
    return w, ball


def test_a_ball_of_half_the_density_floats_on_its_centre():
    # m g = rho (V / 2) g: the submerged cap is half the ball with the centre on the surface
    w, ball = _floating()
    w.step_n(1.0 / 60.0, 1800)
    got = w.get_position(ball)
    assert abs(got[1]) < 1e-3, got
    assert got[0] == 0.0 and got[2] == 0.0, got


def test_a_buoyancy_zone_rewinds_and_is_part_of_the_snapshot():
    w, _ = _floating()
    w.step_n(1.0 / 60.0, 30)
    blob = bytes(w.snapshot_world())
    w.step_n(1.0 / 60.0, 30)
    first = bytes(w.snapshot_world())
    w.restore_world(blob)
    assert bytes(w.snapshot_world()) == blob
    w.step_n(1.0 / 60.0, 30)
    assert bytes(w.snapshot_world()) == first
    plain = ap.PhysicsWorld()
    plain.add_dynamic_body(0.0, 0.25, 0.0, 1.0)
    try:
        plain.restore_world(blob)
    except ValueError:
        return
    raise AssertionError("a world without the zone took its snapshot")


def _hanging_ball(mass):
    w = ap.PhysicsWorld()
    top = w.add_static_body(0.0, 0.0, 0.0)
    ball = w.add_dynamic_body(0.8, -1.8, 0.0, mass)
    tie = (0.0, 0.0, 0.0)
    rope = w.add_rope((0.0, 0.0, 0.0), (0.8, -1.8, 0.0), 8, 1.0, 1.0, (top, tie), (ball, tie))
    return w, ball, rope


def test_a_rope_holds_a_ball_at_its_length_and_swings_it():
    w, ball, rope = _hanging_ball(4.0)
    length = math.hypot(0.8, 1.8)
    lowest = 0.0
    for frame in range(128):
        w.step(1.0 / 64.0)
        x, y, z = w.get_position(ball)
        assert abs(math.sqrt(x * x + y * y + z * z) - length) < length * 0.05, frame
        lowest = min(lowest, x)
    assert lowest < -0.1, lowest
    particles = w.rope_positions(rope)
    assert len(particles) == 9
    assert particles[0] == (0.0, 0.0, 0.0)


def test_a_rope_rewinds_and_is_part_of_the_snapshot():
    w, _, _ = _hanging_ball(4.0)
    for _ in range(32):
        w.step(1.0 / 64.0)
    blob = bytes(w.snapshot_world())
    for _ in range(32):
        w.step(1.0 / 64.0)
    end = bytes(w.snapshot_world())
    w.restore_world(blob)
    for _ in range(32):
        w.step(1.0 / 64.0)
    assert bytes(w.snapshot_world()) == end
    try:
        ap.PhysicsWorld().restore_world(blob)
    except ValueError:
        return
    raise AssertionError("a world without the rope took its snapshot")


def test_a_bad_rope_is_refused():
    w = ap.PhysicsWorld()
    o = (0.0, 0.0, 0.0)
    x = (1.0, 0.0, 0.0)
    bad = [
        (o, x, 0, 1.0),
        (o, x, 2000, 1.0),
        (o, o, 2, 1.0),
        (o, x, 2, 0.0),
        (o, x, 2, 1.0, 1.5),
        (o, x, 2, 1.0, 1.0, (7, o)),
    ]
    for value in (1e19, 1e30, sys.float_info.max, -1e19, -1e30, -sys.float_info.max, float("nan")):
        bad += [
            ((value, 0.0, 0.0), x, 2, 1.0),
            (o, (0.0, value, 0.0), 2, 1.0),
            (o, x, 2, value),
            (o, x, 2, 1.0, value),
        ]
    for args in bad:
        try:
            w.add_rope(*args)
        except ValueError:
            continue
        raise AssertionError(f"accepted {args}")
    assert w.add_rope(o, x, 2, 1.0) == 0


def test_a_bad_buoyancy_zone_is_refused():
    w = ap.PhysicsWorld()
    box = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    for bad in [
        ((1.0, 0.0, 0.0), (0.0, 1.0, 1.0), 1.0, 1.0, 0.0, 0.0),
        (*box, 0.0, 1.0, 0.0, 0.0),
        (*box, 1.0, 0.0, 0.0, 0.0),
        (*box, 1.0, 1.0, -1.0, 0.0),
        (*box, float("nan"), 1.0, 0.0, 0.0),
    ]:
        try:
            w.add_buoyancy_zone(*bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")
    # past the range of the fixed point, on every argument and on either side
    good = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
    for i in range(len(good)):
        for value in (1e19, 1e30, sys.float_info.max, -1e19, -1e30, -sys.float_info.max):
            p = list(good)
            p[i] = value
            try:
                w.add_buoyancy_zone(tuple(p[0:3]), tuple(p[3:6]), *p[6:])
            except ValueError:
                continue
            raise AssertionError(f"accepted argument {i} = {value}")
    assert w.add_buoyancy_zone(*box, 1.0, 1.0, 0.0, 0.0) == 0


def _pull(chi_eff, saturation):
    # how far a ball of the magnet's material, 6 cm from a magnet along its moment, moves toward
    # it in 0.5 s (no other force acts along x)
    w = ap.PhysicsWorld()
    a = w.add_dynamic_body(0.06, 0.0, 0.0, 0.00411)
    w.set_collision_radius(a, 0.005)
    m = w.register_material(0.3, 0.15)
    w.set_body_material(a, m)
    w.add_magnet((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), 0.01, m, chi_eff, saturation, 7850.0)
    w.step_n(1.0 / 120.0, 60)
    return 0.06 - w.get_position(a)[0]


def test_magnet_arguments_reach_their_fields():
    # a low saturation caps the pull, a low susceptibility weakens it: with the arguments in
    # place, steel (chi 3, 1.7e6 A/m) pulls far more than a material saturating at 1 A/m; with
    # susceptibility and saturation swapped both would barely move
    strong = _pull(3.0, 1.7e6)
    capped = _pull(3.0, 1.0)
    assert strong > 0.0 and strong > 100.0 * abs(capped), (strong, capped)


def _bounce(friction, restitution):
    # x velocity of a resting ball after a ball of a registered material hits it head on
    w = ap.PhysicsWorld()
    a = w.add_shaped_body(3, 0.2, 0.2, 0.2, 7800.0, 0.0, 5.0, 0.0)
    b = w.add_shaped_body(3, 0.2, 0.2, 0.2, 7800.0, 0.9, 5.0, 0.0)
    w.set_velocity(a, 2.0, 0.0, 0.0)
    w.set_body_material(a, w.register_material(friction, restitution))
    w.step_n(1.0 / 120.0, 60)
    return w.get_position(b)[0]


def test_registered_material_arguments_reach_their_fields():
    # restitution changes a head-on hit, friction barely does: with the arguments in place,
    # (friction 0, restitution 0.9) and (0, 0) differ; swapped they would both have restitution 0
    assert _bounce(0.0, 0.9) != _bounce(0.0, 0.0)
    assert _bounce(0.0, 0.9) == _bounce(0.0, 0.9)


def test_bad_magnets_and_materials_raise():
    w = ap.PhysicsWorld()
    body = w.add_dynamic_body(0.0, 0.0, 0.0, 1.0)
    m = w.register_material(0.3, 0.15)
    for bad in [
        lambda: w.register_material(-0.1, 0.2),
        lambda: w.set_body_material(body, 70000),
        lambda: w.add_magnet((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0.0, m, 3.0, 1.7e6, 7850.0),
        lambda: w.add_magnet((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0.01, m, -3.0, 1.7e6, 7850.0),
        lambda: w.add_magnet((float("nan"), 0.0, 0.0), (0.0, 1.0, 0.0), 0.01, m, 3.0, 1.7e6, 7850.0),
    ]:
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("a bad value was accepted")
    try:
        w.set_body_material(99, m)
    except IndexError:
        pass
    else:
        raise AssertionError("an unknown body was accepted")


def test_an_axis_locked_body_keeps_its_locked_coordinates():
    # a body thrown out of the x-y plane and locked to it (z position, x and y rotations: 4 + 8
    # + 16) keeps z = 0 exactly and falls under gravity; a sixth bit is refused
    w = ap.PhysicsWorld()
    b = w.add_dynamic_body(0.0, 10.0, 0.0, 1.0)
    w.set_body_axis_lock(b, 28)
    assert w.body_axis_lock(b) == 28
    try:
        w.set_body_axis_lock(b, 64)
    except ValueError:
        pass
    else:
        raise AssertionError("a seventh bit was accepted")
    try:
        w.set_body_axis_lock(9, 4)
    except IndexError:
        pass
    else:
        raise AssertionError("an unknown body was accepted")
    w.set_velocity(b, 1.0, 0.0, 3.0)
    w.step_n(1.0 / 60.0, 30)
    x, y, z = w.get_position(b)
    assert z == 0.0
    assert x > 0.4 and y < 10.0, (x, y)


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"ok   {name}")
        except Exception:  # noqa: BLE001 - report every failure, then fail the run
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"{len(tests)} tests, {failed} failed")
    if not tests:
        print("no test ran")
        return 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

