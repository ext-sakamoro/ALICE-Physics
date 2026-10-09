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
    # dynamic body at y = 10, mass 1, 60 steps of 1/60 s, default config:
    # y = 10 - 4.1406 = 5.8594 (the frame-damping closed form the FFI golden pins)
    w = ap.PhysicsWorld()
    b = w.add_dynamic_body(0.0, 10.0, 0.0, 1.0)
    for _ in range(60):
        w.step(1.0 / 60.0)
    x, y, z = w.get_position(b)
    assert (x, z) == (0.0, 0.0), (x, z)
    assert abs(y - 5.8594) < 1e-3, y


def test_a_static_body_does_not_move():
    w = ap.PhysicsWorld()
    s = w.add_static_body(1.0, 2.0, 3.0)
    w.step_n(1.0 / 60.0, 30)
    assert w.get_position(s) == (1.0, 2.0, 3.0)


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
