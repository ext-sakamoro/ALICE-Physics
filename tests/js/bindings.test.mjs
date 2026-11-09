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

// oracle: SHA-256 over the 32 domain bytes (themselves the SHA-256 of "alice-physics/state-digest/v1"), then
// snapshotWorld(), recomputed with WebCrypto (an implementation independent of the crate's)
async function digestOf(snapshot) {
  const text = new TextEncoder().encode("alice-physics/state-digest/v1");
  const domain = new Uint8Array(await crypto.subtle.digest("SHA-256", text));
  const all = new Uint8Array(domain.length + snapshot.length);
  all.set(domain, 0);
  all.set(snapshot, domain.length);
  return new Uint8Array(await crypto.subtle.digest("SHA-256", all));
}

const jointedScene = () => {
  const w = world();
  w.addStaticPlane(0, 1, 0, 0);
  const a = w.addDynamicBody(0, 5, 0, 1);
  const b = w.addDynamicBody(1, 5, 0, 2);
  w.addBallJoint(a, b, [0, 0, 0, -1, 0, 0]);
  return w;
};

test("stateDigest is SHA-256 of the domain and snapshotWorld (WebCrypto recomputation)", async () => {
  const w = jointedScene();
  for (let i = 0; i < 30; i++) w.step(DT);
  const d = w.stateDigest();
  assert.equal(d.length, 32);
  assert.deepEqual(Array.from(d), Array.from(await digestOf(w.snapshotWorld())));
  assert.equal(w.lawId().length, 32);
});

test("restoreWorld brings back joints that deserializeState drops, with the same digest", () => {
  const w = jointedScene();
  for (let i = 0; i < 20; i++) w.step(DT);
  const blob = w.snapshotWorld();
  const digest = Array.from(w.stateDigest());
  const fresh = world();
  fresh.restoreWorld(blob);
  assert.equal(fresh.jointCount(), 1);
  assert.deepEqual(Array.from(fresh.stateDigest()), digest);
  for (let i = 0; i < 40; i++) {
    w.step(DT);
    fresh.step(DT);
  }
  assert.deepEqual(Array.from(fresh.stateDigest()), Array.from(w.stateDigest()));
});

// A ball in a steady wind, no gravity, in a world built by withConfig, which keeps 0.99 of the
// velocity at the end of every frame. Within a frame of length T the drag u' = -c·u² (u = U - v,
// c = ½ρCdA/m) takes u to u/(1 + c·u·T), so the steady speed is the fixed point of
// v = 0.99·(U - u/(1 + c·u·T)). Here U = 10 m/s and c = 0.25 /m.
test("a ball in a wind zone reaches the speed where drag balances the frame damping", () => {
  const w = WasmPhysicsWorld.withConfig(0, 0, 0, 8, 4);
  const ball = w.addDynamicBody(0, 0, 0, 2);
  assert.equal(w.addWindZone(-1e3, -1e3, -1e3, 1e3, 1e3, 1e3, 1, 0, 0, 10, 0, 0, 1, 1, 1), 0);
  for (let i = 0; i < 1200; i++) w.step(DT);
  let want = 0;
  for (let i = 0; i < 5000; i++) {
    const u = 10 - want;
    want = 0.99 * (10 - u / (1 + 0.25 * u * DT));
  }
  const [vx, vy, vz] = vel(w, ball);
  assert.ok(Math.abs(vx - want) < 0.001 * want, `vx ${vx}, want ${want}`);
  assert.equal(vy, 0);
  assert.equal(vz, 0);
});

test("a gusting wind zone rewinds to the same bits through snapshotWorld", () => {
  const w = WasmPhysicsWorld.withConfig(0, -10, 0, 8, 4);
  w.addDynamicBody(0, 5, 0, 1);
  assert.equal(w.addWindZone(-50, -50, -50, 50, 50, 50, 1, 0, 0.5, 6, 3, 1.5, 1.2, 0.9, 0.4), 0);
  for (let i = 0; i < 30; i++) w.step(DT);
  const blob = w.snapshotWorld();
  const run = () => {
    for (let i = 0; i < 90; i++) w.step(DT);
    return [Array.from(w.stateDigest()), Array.from(w.getPositions())];
  };
  const first = run();
  w.restoreWorld(blob);
  assert.deepEqual(Array.from(w.snapshotWorld()), Array.from(blob));
  assert.deepEqual(run(), first);
});

test("a wind zone's arguments are checked and a world without it refuses its snapshot", () => {
  const w = WasmPhysicsWorld.withConfig(0, 0, 0, 8, 4);
  assert.equal(w.addWindZone(1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1), undefined, "inside out");
  assert.equal(w.addWindZone(0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1), undefined, "no direction");
  assert.equal(w.addWindZone(0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1), undefined, "no air");
  // past the range of the fixed point, on every argument and on either side
  const good = [0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1];
  for (let i = 0; i < good.length; i++) {
    for (const value of [1e19, 1e30, Number.MAX_VALUE, -1e19, -1e30, -Number.MAX_VALUE]) {
      const p = good.slice();
      p[i] = value;
      assert.equal(w.addWindZone(...p), undefined, `argument ${i} = ${value}`);
    }
  }
  assert.equal(w.addWindZone(0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1), 0);
  const blob = w.snapshotWorld();
  const plain = WasmPhysicsWorld.withConfig(0, 0, 0, 8, 4);
  assert.throws(() => plain.restoreWorld(blob));
});

// A sphere of half the fluid's density floats with its centre on the surface, where the submerged
// cap is half the ball: m·g = ρ·(V/2)·g. A rest height does not depend on the frame damping.
test("a ball of half the density floats on its centre, from the radius the world knows", () => {
  const w = WasmPhysicsWorld.withConfig(0, -10, 0, 8, 4);
  const volume = (4 * Math.PI) / 3;
  const ball = w.addDynamicBody(0, 0.25, 0, 0.5 * volume);
  assert.equal(w.setCollisionRadius(ball, 1), true);
  assert.equal(w.addBuoyancyZone(-1e3, -1e3, -1e3, 1e3, 0, 1e3, 1, 10, 8, 0), 0);
  for (let i = 0; i < 1800; i++) w.step(DT);
  const [x, y, z] = pos(w, ball);
  assert.ok(Math.abs(y) < 1e-3, `y ${y}`);
  assert.equal(x, 0);
  assert.equal(z, 0);
  // the same ball with half the radius displaces an eighth and sinks
  assert.equal(w.setCollisionRadius(ball, 0.5), true);
  for (let i = 0; i < 60; i++) w.step(DT);
  assert.ok(pos(w, ball)[1] < -0.1, `y ${pos(w, ball)[1]}`);
});

test("a buoyancy zone rewinds to the same bits through snapshotWorld", () => {
  const w = WasmPhysicsWorld.withConfig(0, -10, 0, 8, 4);
  const ball = w.addDynamicBody(0.5, 2, 0, 1);
  w.setCollisionRadius(ball, 0.6);
  assert.equal(w.addBuoyancyZone(-50, -50, -50, 50, 0, 50, 2, 10, 1.5, 0.5), 0);
  for (let i = 0; i < 30; i++) w.step(DT);
  const blob = w.snapshotWorld();
  const run = () => {
    for (let i = 0; i < 90; i++) w.step(DT);
    return [Array.from(w.stateDigest()), Array.from(w.getPositions())];
  };
  const first = run();
  w.restoreWorld(blob);
  assert.deepEqual(Array.from(w.snapshotWorld()), Array.from(blob));
  assert.deepEqual(run(), first);
});

test("a buoyancy zone's arguments are checked and a world without it refuses its snapshot", () => {
  const w = WasmPhysicsWorld.withConfig(0, 0, 0, 8, 4);
  assert.equal(w.addBuoyancyZone(1, 0, 0, 0, 1, 1, 1, 1, 0, 0), undefined, "inside out");
  assert.equal(w.addBuoyancyZone(0, 0, 0, 1, 1, 1, 0, 1, 0, 0), undefined, "no density");
  assert.equal(w.addBuoyancyZone(0, 0, 0, 1, 1, 1, 1, 0, 0, 0), undefined, "no gravity");
  assert.equal(w.addBuoyancyZone(0, 0, 0, 1, 1, 1, 1, 1, -1, 0), undefined, "negative drag");
  // past the range of the fixed point, on every argument and on either side
  const good = [0, 0, 0, 1, 1, 1, 1, 1, 0, 0];
  for (let i = 0; i < good.length; i++) {
    for (const value of [1e19, 1e30, Number.MAX_VALUE, -1e19, -1e30, -Number.MAX_VALUE]) {
      const p = good.slice();
      p[i] = value;
      assert.equal(w.addBuoyancyZone(...p), undefined, `argument ${i} = ${value}`);
    }
  }
  assert.equal(w.addBuoyancyZone(0, 0, 0, 1, 1, 1, 1, 1, 0, 0), 0);
  const blob = w.snapshotWorld();
  const plain = WasmPhysicsWorld.withConfig(0, 0, 0, 8, 4);
  assert.throws(() => plain.restoreWorld(blob));
});

function hangingBall(mass) {
  const w = WasmPhysicsWorld.withConfig(0, -10, 0, 8, 4);
  const top = w.addStaticBody(0, 0, 0);
  const ball = w.addDynamicBody(0.8, -1.8, 0, mass);
  const rope = w.addRope([0, 0, 0], [0.8, -1.8, 0], 8, 1, 1, top, [0, 0, 0], ball, [0, 0, 0]);
  return { w, ball, rope };
}

// oracle: a rope from JS holds the ball at its length while the ball swings past the bottom
test("a rope tied to a ball holds it at the rope's length and swings it", () => {
  const { w, ball, rope } = hangingBall(4);
  const length = Math.hypot(0.8, 1.8);
  let lowestX = Infinity;
  for (let i = 0; i < 128; i++) {
    w.step(1 / 64);
    const all = w.getPositions();
    const [x, y, z] = [all[ball * 3], all[ball * 3 + 1], all[ball * 3 + 2]];
    assert.ok(Math.abs(Math.hypot(x, y, z) - length) < length * 0.05, `frame ${i}`);
    lowestX = Math.min(lowestX, x);
  }
  assert.ok(lowestX < -0.1, `the ball swung to x = ${lowestX}`);
  const p = w.ropePositions(rope);
  assert.equal(p.length, 9 * 3);
  assert.deepEqual(Array.from(p.slice(0, 3)), [0, 0, 0], "the tied end stays on the anchor");
});

test("a rope rewinds to the same bits through snapshotWorld", () => {
  const { w } = hangingBall(4);
  for (let i = 0; i < 32; i++) w.step(1 / 64);
  const blob = w.snapshotWorld();
  for (let i = 0; i < 32; i++) w.step(1 / 64);
  const end = Array.from(w.snapshotWorld());
  w.restoreWorld(blob);
  for (let i = 0; i < 32; i++) w.step(1 / 64);
  assert.deepEqual(Array.from(w.snapshotWorld()), end);
});

test("a rope's arguments are checked and a world without it refuses its snapshot", () => {
  const w = WasmPhysicsWorld.withConfig(0, -10, 0, 8, 4);
  const o = [0, 0, 0];
  const x = [1, 0, 0];
  const add = (start, end, segments, mass, damping, body) =>
    w.addRope(start, end, segments, mass, damping, body, o, undefined, o);
  assert.equal(add(o, x, 0, 1, 1), undefined, "no segments");
  assert.equal(add(o, x, 2000, 1, 1), undefined, "too many segments");
  assert.equal(add(o, o, 2, 1, 1), undefined, "ends coincide");
  assert.equal(add(o, x, 2, 0, 1), undefined, "no mass");
  assert.equal(add(o, x, 2, 1, 1.5), undefined, "damping above 1");
  assert.equal(add(o, x, 2, 1, 1, 7), undefined, "no such body");
  for (const value of [1e19, 1e30, Number.MAX_VALUE, -1e19, -1e30, -Number.MAX_VALUE, NaN]) {
    assert.equal(add([value, 0, 0], x, 2, 1, 1), undefined, `start ${value}`);
    assert.equal(add(o, [0, value, 0], 2, 1, 1), undefined, `end ${value}`);
    assert.equal(add(o, x, 2, value, 1), undefined, `mass ${value}`);
    assert.equal(add(o, x, 2, 1, value), undefined, `damping ${value}`);
  }
  assert.equal(add(o, x, 2, 1, 1), 0);
  assert.equal(w.ropePositions(1), undefined, "no such participant");
  const blob = w.snapshotWorld();
  const plain = WasmPhysicsWorld.withConfig(0, -10, 0, 8, 4);
  assert.throws(() => plain.restoreWorld(blob));
});

test("a refused restoreWorld throws an Error named by its kind and changes nothing", () => {
  const w = jointedScene();
  const before = Array.from(w.stateDigest());
  const blob = jointedScene().snapshotWorld();
  blob[blob.length - 20] ^= 1;
  assert.throws(() => w.restoreWorld(blob), (e) => e instanceof Error && e.name === "ChecksumMismatch");
  assert.throws(() => w.restoreWorld(new Uint8Array(3)), (e) => e.name === "Truncated");
  assert.deepEqual(Array.from(w.stateDigest()), before);
});

test("one impulse changes the digest, the same calls do not", () => {
  const a = jointedScene();
  const b = jointedScene();
  assert.deepEqual(Array.from(a.stateDigest()), Array.from(b.stateDigest()));
  b.applyImpulse(0, 0, 1, 0);
  assert.notDeepEqual(Array.from(a.stateDigest()), Array.from(b.stateDigest()));
});

// golden (regenerated on purpose only, with the Rust test that pins the same two values,
// tests/state_digest.rs GOLDEN_STATE_DIGEST / GOLDEN_LAW_ID): the jointed scene after 120 steps
// must give these exact bytes in WebAssembly as on every native platform
test("the jointed scene's digest and law id are the golden the native platforms compute", () => {
  const w = jointedScene();
  for (let i = 0; i < 120; i++) w.step(DT);
  const hex = (b) => Array.from(b, (x) => x.toString(16).padStart(2, "0")).join("");
  assert.equal(hex(w.stateDigest()), "ae339480def0fcb4e1d1668888eea3a5180d4cab8cf8863ff050031c1c290f14");
  assert.equal(hex(w.lawId()), "696e21114645842b8a3b6db7346eae96f2f126b44fc5b8444a4da04d9726afd0");
});

test("setRotation normalises, refuses the zero quaternion and a bad id, and is saved by snapshotWorld", () => {
  const w = world();
  const b = w.addDynamicBody(0, 5, 0, 1);
  const before = Array.from(w.stateDigest());
  assert.equal(w.setRotation(b, 0, 0, 2, 0), true);
  assert.deepEqual(Array.from(w.getRotations().slice(4 * b, 4 * b + 4)), [0, 0, 1, 0]);
  assert.notDeepEqual(Array.from(w.stateDigest()), before);
  assert.equal(w.setRotation(b, 0, 0, 0, 0), false);
  assert.equal(w.setRotation(b, NaN, 0, 0, 1), false);
  assert.equal(w.setRotation(99, 0, 0, 0, 1), false);
  assert.deepEqual(Array.from(w.getRotations().slice(4 * b, 4 * b + 4)), [0, 0, 1, 0]);
});

test("a fixed joint takes a rotation of any size", () => {
  const w = WasmPhysicsWorld.withConfig(0, 0, 0, 8, 4);
  const a = w.addDynamicBody(0, 0, 0, 1);
  const b = w.addDynamicBody(0, 0, 0, 1);
  const anchors = new Float64Array(6);
  assert.notEqual(w.addFixedJoint(a, b, anchors, new Float64Array([8.3e15, -1.5e16, 3.1e15, -2.2e17])), undefined);
  assert.equal(w.addFixedJoint(a, b, anchors, new Float64Array(4)), undefined);
});

// a box at rest on a floor is asleep after 240 steps; a spin makes it turn
test("a spin wakes a sleeping body", () => {
  const w = new WasmPhysicsWorld();
  w.addStaticPlane(0, 1, 0, 0);
  const b = w.addShapedBody(0, 0.5, 0.5, 0.5, 1000, 0, 0.5, 0);
  for (let i = 0; i < 240; i++) w.step(DT);
  assert.equal(w.setAngularVelocity(b, 0, 3, 0), true);
  for (let i = 0; i < 10; i++) w.step(DT);
  assert.ok(Math.abs(w.getRotations()[4 * b + 3] - 1) > 1e-6);
});

// withConfig keeps the world's per-step damping, so the closed form (a half turn for pi rad/s
// over 1 s with no damping) is checked in the Rust test of binding_api; here: the turn is about
// z only, in the positive sense, and grows with time
test("setAngularVelocity turns a free body about the given axis", () => {
  const w = WasmPhysicsWorld.withConfig(0, 0, 0, 8, 4);
  const b = w.addDynamicBody(0, 0, 0, 1);
  assert.equal(w.setAngularVelocity(b, 0, 0, Math.PI), true);
  assert.equal(w.setAngularVelocity(99, 0, 0, 1), false);
  assert.equal(w.setAngularVelocity(b, Infinity, 0, 0), false);
  const angleAt = () => {
    const [qx, qy, qz, qw] = Array.from(w.getRotations().slice(4 * b, 4 * b + 4));
    assert.ok(Math.abs(qx) < 1e-9 && Math.abs(qy) < 1e-9, `${qx} ${qy}`);
    return 2 * Math.atan2(qz, qw);
  };
  for (let i = 0; i < 10; i++) w.step(DT);
  const early = angleAt();
  for (let i = 0; i < 10; i++) w.step(DT);
  const later = angleAt();
  assert.ok(early > 0 && later > early, `${early} then ${later}`);
});

test("an axis-locked body keeps its locked coordinates", () => {
  // a body thrown out of the x-y plane and locked to it (z position, x and y rotations: 4 + 8
  // + 16) keeps z = 0 exactly and falls under gravity as before; a seventh bit is refused
  const w = world();
  const b = w.addDynamicBody(0, 10, 0, 1);
  assert.equal(w.setBodyAxisLock(b, 28), true);
  assert.equal(w.bodyAxisLock(b), 28);
  assert.equal(w.setBodyAxisLock(b, 64), false);
  assert.equal(w.setBodyAxisLock(9, 4), false);
  assert.equal(w.bodyAxisLock(9), undefined);
  w.setVelocity(b, 1, 0, 3);
  for (let k = 0; k < 30; k++) w.step(DT);
  const [x, y, z] = pos(w, b);
  assert.equal(z, 0);
  assert.equal(vel(w, b)[2], 0);
  assert.ok(x > 0.4 && y < 10, `${x} ${y}`);
});
