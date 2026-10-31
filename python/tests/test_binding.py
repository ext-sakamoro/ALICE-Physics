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


# ---- every float parameter refuses a non-finite value ------------------------------
#
# The rows come from scripts/binding-float-entries.tsv (kept equal to the binding's code by
# scripts/binding_float_entries.py); a row without a case fails. Each parameter, given NaN,
# +inf or -inf (the others finite), raises ValueError at its explicit check and never a
# PanicException (the debug assertion of Fix128::from_f64, on in the dev-profile wheel CI
# builds for this run); the same call with finite values succeeds.

_FLOAT_ROWS = [
    line.split("\t")[1:3]
    for line in (ROOT / "scripts" / "binding-float-entries.tsv").read_text(encoding="utf-8").splitlines()
    if line and not line.startswith("#") and line.split("\t")[0] == "src/python.rs"
]


def _scene():
    # two unit-radius bodies on the z axis; a ray from z = -5 along +z hits body 0
    w = ap.PhysicsWorld()
    w.add_dynamic_body(0.0, 0.0, 0.0, 1.0)
    w.add_dynamic_body(0.0, 0.0, 5.0, 1.0)
    w.set_collision_radius(0, 1.0)
    w.set_collision_radius(1, 1.0)
    return w


def _choose(param, bad):
    """x(name, good): `good`, except for `param` when `bad` is given (a tuple or an array
    carries it in its first element)"""
    def x(name, good):
        if name != param or bad is None:
            return good
        if isinstance(good, tuple):
            return (bad,) + good[1:]
        if isinstance(good, list):
            return [(good[0][0], bad) + tuple(good[0][2:])] + good[1:]
        if isinstance(good, np.ndarray):
            a = good.copy()
            a.flat[0] = bad
            return a
        return bad
    return x


_W = "PyPhysicsWorld::"
_S = "PyDeterministicSimulation::"
_V0 = (0.0, 0.0, 0.0)
_FLOAT_CASES = {
    _W + "add_dynamic_body": lambda x: _scene().add_dynamic_body(x("x", 0.0), x("y", 2.0), x("z", 0.0), x("mass", 1.0)),
    _W + "add_static_body": lambda x: _scene().add_static_body(x("x", 0.0), x("y", 2.0), x("z", 0.0)),
    _W + "step": lambda x: _scene().step(x("dt", 1.0 / 60.0)),
    _W + "step_n": lambda x: _scene().step_n(x("dt", 1.0 / 60.0), 2),
    _W + "set_velocity": lambda x: _scene().set_velocity(0, x("vx", 1.0), x("vy", 0.0), x("vz", 0.0)),
    _W + "set_positions_batch": lambda x: _scene().set_positions_batch(x("data", np.array([[1.0, 0, 0], [0, 0, 5.0]]))),
    _W + "set_velocities_batch": lambda x: _scene().set_velocities_batch(x("data", np.array([[1.0, 0, 0], [0, 0, 0.0]]))),
    _W + "apply_impulses_batch": lambda x: _scene().apply_impulses_batch(_impulse_rows(x)),
    _W + "add_bodies_batch": lambda x: _scene().add_bodies_batch(x("data", np.array([[0.0, 2.0, 0, 1.0]]))),
    _W + "add_shaped_body": lambda x: _scene().add_shaped_body(
        0, x("a", 0.5), x("b", 0.5), x("c", 0.5), x("density", 1.0), x("x", 0.0), x("y", 2.0), x("z", 0.0)),
    _W + "set_body_shape": lambda x: _scene().set_body_shape(0, 0, x("a", 0.5), x("b", 0.5), x("c", 0.5)),
    _W + "set_collision_radius": lambda x: _scene().set_collision_radius(0, x("radius", 0.5)),
    _W + "add_static_plane": lambda x: _scene().add_static_plane(x("nx", 0.0), x("ny", 1.0), x("nz", 0.0), x("offset", -10.0)),
    _W + "add_static_heightfield": lambda x: _scene().add_static_heightfield(
        x("heights", np.zeros((2, 2))), x("spacing", 1.0), x("origin", (0.0, -5.0, 0.0))),
    _W + "add_static_trimesh": lambda x: _scene().add_static_trimesh(
        x("vertices", np.array([[0.0, -5.0, 0.0], [1.0, -5.0, 0.0], [0.0, -5.0, 1.0]])), [0, 1, 2]),
    _W + "add_ball_joint": lambda x: _scene().add_ball_joint(0, 1, x("anchor_a", _V0), x("anchor_b", _V0)),
    _W + "add_fixed_joint": lambda x: _scene().add_fixed_joint(
        0, 1, x("anchor_a", _V0), x("anchor_b", _V0), x("relative_rotation", (0.0, 0.0, 0.0, 1.0))),
    _W + "add_hinge_joint": lambda x: _scene().add_hinge_joint(
        0, 1, x("anchor_a", _V0), x("anchor_b", _V0), x("axis_a", (0.0, 1.0, 0.0)), x("axis_b", (0.0, 1.0, 0.0))),
    _W + "add_slider_joint": lambda x: _scene().add_slider_joint(
        0, 1, x("axis", (1.0, 0.0, 0.0)), x("anchor_a", _V0), x("anchor_b", _V0)),
    _W + "add_spring_joint": lambda x: _scene().add_spring_joint(
        0, 1, x("anchor_a", _V0), x("anchor_b", _V0), x("rest_length", 1.0), x("stiffness", 10.0), x("damping", 0.1)),
    _W + "cast_ray": lambda x: _hit(_scene().cast_ray(
        x("origin", (0.0, 0.0, -5.0)), x("direction", (0.0, 0.0, 1.0)), x("max_t", 100.0))),
    _W + "cast_sphere": lambda x: _hit(_scene().cast_sphere(
        x("center", (0.0, 0.0, -5.0)), x("radius", 0.5), x("direction", (0.0, 0.0, 1.0)), x("max_t", 100.0))),
    _W + "cast_capsule": lambda x: _hit(_scene().cast_capsule(
        x("a", (0.0, 0.0, -5.0)), x("b", (0.0, 0.1, -5.0)), x("radius", 0.5), x("direction", (0.0, 0.0, 1.0)), x("max_t", 100.0))),
    _W + "overlap_sphere": lambda x: _hit(_scene().overlap_sphere(x("center", _V0), x("radius", 0.5))),
    _S + "add_body": lambda x: ap.DeterministicSimulation().add_body(x("x", 0.0), x("y", 2.0), x("z", 0.0), x("mass", 1.0)),
    _S + "advance_frame": lambda x: _advance(x("inputs", [(0, 1.0, 0.0, 0.0, 0)])),
    "encode_frame_input": lambda x: ap.encode_frame_input(
        0, x("move_x", 1.0), x("move_y", 0.0), x("move_z", 0.0), 0, x("aim_x", 0.0), x("aim_y", 0.0), x("aim_z", 1.0)),
}


def _impulse_rows(x):
    # the non-finite value goes in a component (column 1), not in the body id (column 0)
    a = np.array([[0.0, 1.0, 0.0, 0.0]])
    a[0, 1] = x("data", 1.0)
    return a


def _hit(r):
    assert r, f"the finite query must hit ({r})"
    return r


def _advance(inputs):
    s = ap.DeterministicSimulation()
    b = s.add_body(0.0, 2.0, 0.0, 1.0)
    s.assign_player(0, b)
    return s.advance_frame(inputs)


def test_every_float_parameter_refuses_a_non_finite_value():
    assert _FLOAT_ROWS, "no python row in scripts/binding-float-entries.tsv"
    missing = [r for r in _FLOAT_ROWS if r[0] not in _FLOAT_CASES]
    assert not missing, f"rows without a case: {missing}"
    compared = 0
    for entry, param in _FLOAT_ROWS:
        case = _FLOAT_CASES[entry]
        case(_choose(param, None))
        for bad in (math.nan, math.inf, -math.inf):
            try:
                case(_choose(param, bad))
            except ValueError:
                compared += 1
                continue
            except BaseException as e:  # noqa: BLE001 - a PanicException is a BaseException
                raise AssertionError(f"{entry} {param} = {bad}: {type(e).__name__}: {e}") from e
            raise AssertionError(f"{entry} {param} = {bad}: accepted")
    assert compared > 0, "nothing was compared"


def test_a_refused_call_names_the_method_and_the_parameter():
    w = _scene()
    before = bytes(w.serialize_state())
    for call, want in [
        (lambda: w.step(math.nan), "step: dt is not finite"),
        (lambda: w.add_dynamic_body(0.0, math.inf, 0.0, 1.0), "add_dynamic_body: position is not finite"),
        (lambda: w.set_velocities_batch(np.array([[1.0, 0, 0], [0, math.nan, 0]])),
         "set_velocities_batch: data is not finite"),
    ]:
        try:
            call()
        except ValueError as e:
            assert str(e) == want, (str(e), want)
        else:
            raise AssertionError(f"no ValueError ({want})")
    assert bytes(w.serialize_state()) == before, "a refused call changed the world"


def test_a_finite_value_beyond_the_f32_range_is_refused():
    # this binding narrows positions and velocities to f32: 1e39 is a finite f64 that
    # narrows to inf, so it is refused like inf (before, it became the largest Fix128)
    w = _scene()
    for call in (lambda: w.add_dynamic_body(1e39, 0.0, 0.0, 1.0),
                 lambda: w.set_velocity(0, 0.0, -1e39, 0.0),
                 lambda: ap.encode_frame_input(0, move_x=1e39)):
        try:
            call()
        except ValueError as e:
            assert str(e).endswith(" is not finite"), str(e)
        else:
            raise AssertionError("1e39 was accepted")
    # the largest f32 itself is finite and accepted
    w.add_dynamic_body(float(np.finfo(np.float32).max), 0.0, 0.0, 1.0)


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
