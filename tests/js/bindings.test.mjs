// End-to-end tests of the WebAssembly binding called from JavaScript, through the
// package wasm-bindgen generates (the same path a browser or Node application
// takes). Build the package first:
//   wasm-pack build --dev --target nodejs --out-dir <dir> -- --features wasm
// then run:
//   ALICE_JS_PKG=<dir> node --test tests/js/
//
// Expected values are closed forms (rest height = radius, arm length = anchor
// distance), not values produced by calling the binding.

import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import path from "node:path";

const pkgDir = process.env.ALICE_JS_PKG;
if (!pkgDir) throw new Error("set ALICE_JS_PKG to the wasm-pack output directory");
const { WasmPhysicsWorld } = createRequire(import.meta.url)(path.resolve(pkgDir, "alice_physics.js"));

const DT = 1 / 60;
const world = () => WasmPhysicsWorld.withConfig(0, -10, 0, 8, 4);
const pos = (w, id) => Array.from(w.getPositions().slice(3 * id, 3 * id + 3));
const vel = (w, id) => Array.from(w.getVelocities().slice(3 * id, 3 * id + 3));

test("a sphere dropped on a static plane rests at its radius", () => {
  const w = world();
  assert.equal(w.addStaticPlane(0, 1, 0, 0), 0);
  const ball = w.addDynamicBody(0, 3, 0, 1);
  assert.equal(w.setCollisionRadius(ball, 0.5), true);
  let lowest = Infinity;
  for (let i = 0; i < 240; i++) {
    w.step(DT);
    lowest = Math.min(lowest, pos(w, ball)[1]);
  }
  const [, y] = pos(w, ball);
  assert.ok(Math.abs(y - 0.5) <= 0.01, `rest height ${y}, expected 0.5`);
  assert.ok(lowest >= 0.5 - 0.01, `sank to ${lowest}`);
  assert.ok(Math.abs(vel(w, ball)[1]) <= 0.01, `vertical velocity ${vel(w, ball)[1]}`);
});

test("a sphere dropped on a flat height field rests at its radius above it", () => {
  const w = world();
  const h = 2;
  const heights = new Float64Array(4 * 4).fill(h);
  assert.equal(w.addStaticHeightField(heights, 4, 4, 1, -1.5, 0, -1.5), 0);
  const ball = w.addDynamicBody(0, 5, 0, 1);
  w.setCollisionRadius(ball, 0.5);
  for (let i = 0; i < 240; i++) w.step(DT);
  const [, y] = pos(w, ball);
  assert.ok(Math.abs(y - (h + 0.5)) <= 0.02, `rest height ${y}, expected ${h + 0.5}`);
});

test("a ball-jointed pendulum keeps its arm and swings", () => {
  const w = world();
  const pivot = w.addStaticBody(0, 0, 0);
  const bob = w.addDynamicBody(2, 0, 0, 1);
  assert.equal(w.addBallJoint(pivot, bob, new Float64Array([0, 0, 0, -2, 0, 0])), 0);
  assert.equal(w.jointCount(), 1);
  let lowest = 0;
  for (let i = 0; i < 180; i++) {
    w.step(DT);
    const [x, y, z] = pos(w, bob);
    const arm = Math.hypot(x, y, z);
    assert.ok(Math.abs(arm - 2) <= 0.05, `frame ${i}: arm ${arm}, expected 2`);
    lowest = Math.min(lowest, y);
  }
  assert.ok(lowest < -1, `the bob must swing down (lowest y ${lowest})`);
});

test("the same calls give bit-identical states", () => {
  const run = () => {
    const w = world();
    w.addStaticPlane(0, 1, 0, 0);
    const a = w.addShapedBody(0, 0.5, 0.5, 0.5, 2, 0, 4, 0);
    const b = w.addDynamicBody(0.3, 7, 0.1, 1);
    w.setCollisionRadius(b, 0.4);
    w.addSpringJoint(a, b, new Float64Array(6), 2, 40, 0.5);
    for (let i = 0; i < 120; i++) w.step(DT);
    return Array.from(w.getPositions());
  };
  assert.deepEqual(run(), run());
});

test("refused arguments return undefined / false and change nothing", () => {
  const w = world();
  const s = w.addStaticBody(0, 0, 0);
  const d = w.addDynamicBody(2, 0, 0, 1);
  const six = new Float64Array(6);
  assert.equal(w.setCollisionRadius(d, 0), false);
  assert.equal(w.setCollisionRadius(d, Number.NaN), false);
  assert.equal(w.setCollisionRadius(99, 1), false);
  assert.equal(w.addShapedBody(9, 1, 1, 1, 1, 0, 0, 0), undefined, "unknown shape kind");
  assert.equal(w.addShapedBody(0, 1, 1, 1, -1, 0, 0, 0), undefined, "negative density");
  assert.equal(w.addStaticPlane(0, 0, 0, 0), undefined, "zero normal");
  assert.equal(w.addStaticTriMesh(new Float64Array(9), new Uint32Array([0, 1, 3])), undefined, "index out of range");
  assert.equal(w.addBallJoint(d, d, six), undefined, "joint to itself");
  assert.equal(w.addBallJoint(s, 99, six), undefined, "unknown body");
  assert.equal(w.addHingeJoint(s, d, six, six), undefined, "zero axes");
  assert.equal(w.addSpringJoint(s, d, six, 1, 0, 0), undefined, "zero stiffness");
  assert.equal(w.removeJoint(0), false);
  assert.equal(w.jointCount(), 0);
  assert.equal(w.staticColliderCount(), 0);
  assert.equal(w.bodyCount(), 2);
});
