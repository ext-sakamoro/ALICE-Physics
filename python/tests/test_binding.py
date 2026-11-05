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
