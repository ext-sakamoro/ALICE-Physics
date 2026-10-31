// Every float parameter of the WebAssembly binding, given NaN, +Infinity or -Infinity,
// reaches the explicit check (binding_api::checked*) and never the debug assertion of
// Fix128::from_f64: the call throws an Error "<method>: <parameter> is not finite", or
// returns its documented failure value (false / undefined / an empty array), and never a
// WebAssembly.RuntimeError (a panic). The same call with finite values succeeds.
//
// The rows come from scripts/binding-float-entries.tsv (kept equal to the binding's code
// by scripts/binding_float_entries.py); a row without a case below fails. Run against the
// package built with --dev (scripts/run_js_binding_tests.sh), where the assertion is on.

import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const pkgDir = process.env.ALICE_JS_PKG;
if (!pkgDir) throw new Error("set ALICE_JS_PKG to the wasm-pack output directory");
const { WasmPhysicsWorld } = createRequire(import.meta.url)(path.resolve(pkgDir, "alice_physics.js"));

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const rows = readFileSync(path.join(root, "scripts", "binding-float-entries.tsv"), "utf8")
  .split("\n")
  .filter((l) => l && !l.startsWith("#"))
  .map((l) => l.split("\t"))
  .filter((c) => c[0] === "src/wasm.rs")
  .map((c) => ({ entry: c[1].replace(/^WasmPhysicsWorld::/, ""), param: c[2] }));

// two unit-radius bodies on the z axis; a ray from z = -5 along +z hits body 0
function scene() {
  const w = WasmPhysicsWorld.withConfig(0, -10, 0, 8, 4);
  w.addDynamicBody(0, 0, 0, 1);
  w.addDynamicBody(0, 0, 5, 1);
  w.setCollisionRadius(0, 1);
  w.setCollisionRadius(1, 1);
  return w;
}

const isUnit = (r) => r === undefined;
const isTrue = (r) => r === true;
const isIndex = (r) => typeof r === "number";
const isHit = (r) => r.length > 0;

// each case: (w, x) => result, where x(name, good) is `good` except for the parameter
// under test, which is the non-finite value (a typed array carries it in element 0);
// `ok` says whether a result is the success value
const cases = {
  "with_config": { params: ["gravity_x", "gravity_y", "gravity_z"], ok: (r) => r instanceof WasmPhysicsWorld,
    call: (w, x) => WasmPhysicsWorld.withConfig(x("gravity_x", 0), x("gravity_y", -10), x("gravity_z", 0), 8, 4) },
  "add_dynamic_body": { params: ["x", "y", "z", "mass"], ok: isIndex,
    call: (w, x) => w.addDynamicBody(x("x", 0), x("y", 2), x("z", 0), x("mass", 1)) },
  "add_static_body": { params: ["x", "y", "z"], ok: isIndex,
    call: (w, x) => w.addStaticBody(x("x", 0), x("y", 2), x("z", 0)) },
  "add_kinematic_body": { params: ["x", "y", "z"], ok: isIndex,
    call: (w, x) => w.addKinematicBody(x("x", 0), x("y", 2), x("z", 0)) },
  "step": { params: ["dt"], ok: isUnit, call: (w, x) => w.step(x("dt", 1 / 60)) },
  "step_n": { params: ["dt"], ok: isUnit, call: (w, x) => w.stepN(x("dt", 1 / 60), 2) },
  "set_position": { params: ["x", "y", "z"], ok: isUnit,
    call: (w, x) => w.setPosition(0, x("x", 1), x("y", 0), x("z", 0)) },
  "set_velocity": { params: ["vx", "vy", "vz"], ok: isUnit,
    call: (w, x) => w.setVelocity(0, x("vx", 1), x("vy", 0), x("vz", 0)) },
  "apply_impulse": { params: ["ix", "iy", "iz"], ok: isUnit,
    call: (w, x) => w.applyImpulse(0, x("ix", 1), x("iy", 0), x("iz", 0)) },
  "set_restitution": { params: ["restitution"], ok: isUnit, call: (w, x) => w.setRestitution(0, x("restitution", 0.5)) },
  "set_friction": { params: ["friction"], ok: isUnit, call: (w, x) => w.setFriction(0, x("friction", 0.5)) },
  "set_gravity_scale": { params: ["scale"], ok: isUnit, call: (w, x) => w.setGravityScale(0, x("scale", 0.5)) },
  "set_kinematic_target": { params: ["x", "y", "z", "qx", "qy", "qz", "qw"], ok: isUnit,
    call: (w, x) => w.setKinematicTarget(0, x("x", 0), x("y", 1), x("z", 0), x("qx", 0), x("qy", 0), x("qz", 0), x("qw", 1)) },
  "set_velocities_batch": { params: ["data"], ok: isUnit,
    call: (w, x) => w.setVelocitiesBatch(x("data", new Float64Array([1, 0, 0, 0, 0, 0]))) },
  "apply_impulses_batch": { params: ["data"], ok: isUnit,
    call: (w, x) => w.applyImpulsesBatch(x("data", new Float64Array([1, 0, 0, 0]), 1)) },
  "set_gravity": { params: ["x", "y", "z"], ok: isUnit,
    call: (w, x) => w.setGravity(x("x", 0), x("y", -1), x("z", 0)) },
  "raycast": { params: ["ox", "oy", "oz", "dx", "dy", "dz", "max_distance", "body_radius"], ok: isHit,
    call: (w, x) => w.raycast(x("ox", 0), x("oy", 0), x("oz", -5), x("dx", 0), x("dy", 0), x("dz", 1),
      x("max_distance", 100), x("body_radius", 1)) },
  "cast_ray": { params: ["origin", "direction", "max_t"], ok: isHit,
    call: (w, x) => w.castRay(x("origin", new Float64Array([0, 0, -5])), x("direction", new Float64Array([0, 0, 1])),
      x("max_t", 100), undefined) },
  "cast_sphere": { params: ["center", "radius", "direction", "max_t"], ok: isHit,
    call: (w, x) => w.castSphere(x("center", new Float64Array([0, 0, -5])), x("radius", 0.5),
      x("direction", new Float64Array([0, 0, 1])), x("max_t", 100), undefined) },
  "cast_capsule": { params: ["a", "b", "radius", "direction", "max_t"], ok: isHit,
    call: (w, x) => w.castCapsule(x("a", new Float64Array([0, 0, -5])), x("b", new Float64Array([0, 0.1, -5])),
      x("radius", 0.5), x("direction", new Float64Array([0, 0, 1])), x("max_t", 100), undefined) },
  "overlap_sphere": { params: ["center", "radius"], ok: isHit,
    call: (w, x) => w.overlapSphere(x("center", new Float64Array([0, 0, 0])), x("radius", 0.5), undefined) },
  "set_collision_radius": { params: ["radius"], ok: isTrue, call: (w, x) => w.setCollisionRadius(0, x("radius", 0.5)) },
  "add_shaped_body": { params: ["a", "b", "c", "density", "x", "y", "z"], ok: isIndex,
    call: (w, x) => w.addShapedBody(0, x("a", 0.5), x("b", 0.5), x("c", 0.5), x("density", 1), x("x", 0), x("y", 2), x("z", 0)) },
  "set_body_shape": { params: ["a", "b", "c"], ok: isTrue,
    call: (w, x) => w.setBodyShape(0, 0, x("a", 0.5), x("b", 0.5), x("c", 0.5)) },
  "add_static_plane": { params: ["nx", "ny", "nz", "offset"], ok: isIndex,
    call: (w, x) => w.addStaticPlane(x("nx", 0), x("ny", 1), x("nz", 0), x("offset", -10)) },
  "add_static_heightfield": { params: ["heights", "spacing", "ox", "oy", "oz"], ok: isIndex,
    call: (w, x) => w.addStaticHeightField(x("heights", new Float64Array([0, 0, 0, 0])), 2, 2, x("spacing", 1),
      x("ox", 0), x("oy", -5), x("oz", 0)) },
  "add_static_trimesh": { params: ["vertices"], ok: isIndex,
    call: (w, x) => w.addStaticTriMesh(x("vertices", new Float64Array([0, -5, 0, 1, -5, 0, 0, -5, 1])), new Uint32Array([0, 1, 2])) },
  "add_ball_joint": { params: ["anchors"], ok: isIndex,
    call: (w, x) => w.addBallJoint(0, 1, x("anchors", new Float64Array([0, 0, 0, 0, 0, 0]))) },
  "add_hinge_joint": { params: ["anchors", "axes"], ok: isIndex,
    call: (w, x) => w.addHingeJoint(0, 1, x("anchors", new Float64Array([0, 0, 0, 0, 0, 0])), x("axes", new Float64Array([0, 1, 0, 0, 1, 0]))) },
  "add_fixed_joint": { params: ["anchors", "relative_rotation"], ok: isIndex,
    call: (w, x) => w.addFixedJoint(0, 1, x("anchors", new Float64Array([0, 0, 0, 0, 0, 0])), x("relative_rotation", new Float64Array([0, 0, 0, 1]))) },
  "add_slider_joint": { params: ["axis", "anchors"], ok: isIndex,
    call: (w, x) => w.addSliderJoint(0, 1, x("axis", new Float64Array([1, 0, 0])), x("anchors", new Float64Array([0, 0, 0, 0, 0, 0]))) },
  "add_spring_joint": { params: ["anchors", "rest_length", "stiffness", "damping"], ok: isIndex,
    call: (w, x) => w.addSpringJoint(0, 1, x("anchors", new Float64Array([0, 0, 0, 0, 0, 0])), x("rest_length", 1), x("stiffness", 10), x("damping", 0.1)) },
};

// the value of parameter `name` for a call testing `param` with `bad` (undefined: all finite);
// `slot` places it in a typed array (default element 0)
function chooser(param, bad) {
  return (name, good, slot = 0) => {
    if (name !== param || bad === undefined) return good;
    if (good instanceof Float64Array) {
      const a = Float64Array.from(good);
      a[slot] = bad;
      return a;
    }
    return bad;
  };
}

test("the table has wasm rows and every row has a case", () => {
  assert.ok(rows.length > 0, "no wasm row in scripts/binding-float-entries.tsv");
  const missing = rows.filter((r) => !(cases[r.entry] && cases[r.entry].params.includes(r.param)));
  assert.deepEqual(missing, [], "rows without a case");
});

test("every float parameter refuses NaN and ±Infinity at its explicit check", () => {
  let compared = 0;
  for (const { entry, param } of rows) {
    const c = cases[entry];
    const good = c.call(scene(), chooser(param, undefined));
    assert.ok(c.ok(good), `${entry} ${param}: the finite call must succeed (${good})`);
    for (const bad of [Number.NaN, Infinity, -Infinity]) {
      let result;
      let thrown;
      try {
        result = c.call(scene(), chooser(param, bad));
      } catch (e) {
        thrown = e;
      }
      if (thrown !== undefined) {
        assert.ok(!(thrown instanceof WebAssembly.RuntimeError), `${entry} ${param} = ${bad}: a panic (${thrown})`);
        assert.ok(thrown instanceof Error && / is not finite$/.test(thrown.message),
          `${entry} ${param} = ${bad}: unexpected error ${thrown}`);
      } else {
        assert.ok(!c.ok(result), `${entry} ${param} = ${bad}: accepted (${result})`);
      }
      compared += 1;
    }
  }
  assert.ok(compared > 0, "nothing was compared");
});

test("a refused call names the method and the parameter and changes nothing", () => {
  const w = scene();
  const before = Array.from(w.serializeState());
  assert.throws(() => w.step(Number.NaN), { message: "step: dt is not finite" });
  assert.throws(() => w.setVelocitiesBatch(new Float64Array([1, 0, 0, Number.NaN, 0, 0])),
    { message: "setVelocitiesBatch: data is not finite" });
  assert.deepEqual(Array.from(w.serializeState()), before);
});
