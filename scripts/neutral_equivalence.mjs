// The neutral package behaves as the package it was made from, bit for bit.
//
//   node scripts/neutral_equivalence.mjs <original-dir> <neutral-dir> <name-map.json>
//
// <original-dir> holds the wasm-bindgen `--target web --out-name p` output, <neutral-dir> the
// package scripts/neutral_pkg.py made from it, <name-map.json> its private name map. The same
// scene (a static plane, two dynamic bodies on a ball joint, 120 steps of 1/60) runs through both,
// (and a second scene through the rest of the API, `wide`), and the positions, rotations and the
// whole serialized state (`serializeState`: positions,
// velocities, rotations, angular velocities, sleep state) must be equal byte for byte: renaming and
// overwriting text must not change behaviour. The wasm API exports no law identifier, so a string
// that feeds one is kept out by the build's source check (neutral_pkg.py --data-literals), not
// here. Exit 0 when equal, 1 when not, 2 when nothing could be compared.

import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const [origDir, neutralDir, mapFile] = process.argv.slice(2);
if (!origDir || !neutralDir || !mapFile) {
  console.error("usage: neutral_equivalence.mjs <original-dir> <neutral-dir> <name-map.json>");
  process.exit(2);
}
const map = JSON.parse(fs.readFileSync(mapFile, "utf8")).api;

async function load(dir, base) {
  const glue = await import(pathToFileURL(path.join(dir, `${base}.js`)).href);
  glue.initSync({ module: fs.readFileSync(path.join(dir, `${base}${base === "p" ? "_bg" : ""}.wasm`)) });
  return glue;
}

// calls a method of the original name on either package
function api(glue, neutral) {
  const name = (n) => (neutral ? map[n] : n);
  const World = glue[name("WasmPhysicsWorld")];
  return {
    world: () => World[name("withConfig")](0, -10, 0, 8, 4),
    call: (w, n, ...args) => (typeof w[name(n)] === "function" ? w[name(n)](...args) : undefined),
  };
}

function run(a) {
  const w = a.world();
  a.call(w, "addStaticPlane", 0, 1, 0, 0);
  const b0 = a.call(w, "addDynamicBody", 0, 5, 0, 1);
  const b1 = a.call(w, "addDynamicBody", 1, 5, 0, 2);
  a.call(w, "addBallJoint", b0, b1, new Float64Array([0, 0, 0, -1, 0, 0]));
  for (let i = 0; i < 120; i++) a.call(w, "step", 1 / 60);
  return {
    positions: a.call(w, "getPositions"),
    rotations: a.call(w, "getRotations"),
    serializedState: a.call(w, "serializeState"),
  };
}

// a second scene through the rest of the API: every joint kind, shaped bodies, a height field and a
// triangle mesh, material and gravity setters, impulses, the scene queries, removal, and a save /
// step / load round trip; every value the API returns is compared
function wide(a) {
  const w = a.world();
  const f = (x) => new Float64Array(x);
  const num = (x) => f([x === undefined || x === null ? NaN : Number(x)]);
  const out = {};
  out.plane = num(a.call(w, "addStaticPlane", 0, 1, 0, 0));
  out.field = num(a.call(w, "addStaticHeightField", f([0, 0.1, 0.2, 0.1, 0.3, 0.1, 0, 0.2, 0]), 3, 3, 1, 4, 0, 4));
  out.mesh = num(a.call(w, "addStaticTriMesh", f([-8, 0.5, -8, -6, 0.5, -8, -8, 0.5, -6]), new Uint32Array([0, 1, 2])));
  const b = [];
  for (let i = 0; i < 6; i++) b.push(a.call(w, "addDynamicBody", i * 0.7, 2 + i, 0.3 * i, 1 + i));
  b.push(a.call(w, "addStaticBody", 3, 0.5, 3));
  b.push(a.call(w, "addKinematicBody", -3, 1, -3));
  b.push(a.call(w, "addShapedBody", 0, 0.3, 0.2, 0.4, 2, 1, 6, 1));
  b.push(a.call(w, "addShapedBody", 1, 0.2, 0.5, 0, 1, -1, 6, 1));
  b.push(a.call(w, "addShapedBody", 3, 0.3, 0.2, 0.25, 1, 2, 7, -1));
  out.bodies = f(b);
  out.hinge = num(a.call(w, "addHingeJoint", b[0], b[1], f([0.3, 0, 0, -0.3, 0, 0]), f([0, 0, 1, 0, 0, 1])));
  out.fixed = num(a.call(w, "addFixedJoint", b[1], b[2], f([0, 0.4, 0, 0, -0.4, 0]), f([0, 0, 0, 1])));
  out.slider = num(a.call(w, "addSliderJoint", b[2], b[3], f([1, 0, 0]), f([0, 0, 0, 0, 0, 0])));
  out.spring = num(a.call(w, "addSpringJoint", b[3], b[4], f([0, 0, 0, 0, 0, 0]), 1.2, 40, 0.5));
  out.ball = num(a.call(w, "addBallJoint", b[4], b[5], f([0, 0, 0, -0.5, 0, 0])));
  a.call(w, "setRestitution", b[0], 0.6);
  a.call(w, "setFriction", b[1], 0.2);
  a.call(w, "setGravityScale", b[2], 0.5);
  a.call(w, "setCollisionRadius", b[3], 0.45);
  a.call(w, "applyImpulse", b[5], 1, 2, -0.5);
  a.call(w, "setVelocity", b[4], 0.5, 0, 0.25);
  a.call(w, "setKinematicTarget", b[7], -2.5, 1.2, -3);
  a.call(w, "setSubsteps", 6);
  a.call(w, "stepN", 1 / 60, 90);
  a.call(w, "setGravity", 0, -9.5, 0.3);
  a.call(w, "setBodyShape", b[8], 2, 0.3, 0.6, 0);
  for (let i = 0; i < 60; i++) a.call(w, "step", 1 / 60);
  out.removed = num(a.call(w, "removeJoint", 1));
  out.removedStatic = num(a.call(w, "removeStaticCollider", 2));
  out.counts = f([a.call(w, "bodyCount"), a.call(w, "jointCount"), a.call(w, "staticColliderCount")]);
  out.types = f(b.map((i) => a.call(w, "getBodyType", i)));
  out.raycast = f(a.call(w, "raycast", 0, 10, 0, 0, -1, 0, 50, 4) ?? []);
  out.castRay = f(a.call(w, "castRay", f([0.2, 10, 0.1]), f([0, -1, 0]), 50, undefined) ?? []);
  out.castSphere = f(a.call(w, "castSphere", f([1, 10, 0]), 0.3, f([0, -1, 0]), 50, undefined) ?? []);
  out.castCapsule = f(a.call(w, "castCapsule", f([0, 10, 1]), f([0, 10.5, 1]), 0.2, f([0, -1, 0]), 50, undefined) ?? []);
  out.overlap = f(a.call(w, "overlapSphere", f([0, 1, 0]), 3, undefined) ?? []);
  for (const k of ["raycast", "castRay", "castSphere", "castCapsule", "overlap"]) {
    if (out[k].length === 0) {
      console.error(`neutral_equivalence: the scene query ${k} hit nothing (it would compare nothing)`);
      process.exit(2);
    }
  }
  out.observe = f(a.call(w, "observeBody", b[0]) ?? []);
  out.states = a.call(w, "getStates");
  out.velocities = a.call(w, "getVelocities");
  const saved = a.call(w, "serializeState");
  out.saved = saved;
  for (let i = 0; i < 30; i++) a.call(w, "step", 1 / 60);
  out.after = a.call(w, "serializeState");
  out.loaded = num(a.call(w, "deserializeState", saved));
  out.reloaded = a.call(w, "serializeState");
  out.version = new TextEncoder().encode(String(a.call(w, "alicePhysicsVersion")));
  return out;
}

const neutralBase = fs.readdirSync(neutralDir).find((f) => /^p-[0-9a-f]{12}\.js$/.test(f))?.slice(0, -3);
if (!neutralBase) {
  console.error("neutral_equivalence: no p-<h12>.js in the neutral directory");
  process.exit(2);
}
const origApi = api(await load(origDir, "p"), false);
const neutralApi = api(await load(neutralDir, neutralBase), true);
const a = { ...run(origApi), ...Object.fromEntries(Object.entries(wide(origApi)).map(([k, v]) => [`wide.${k}`, v])) };
const b = { ...run(neutralApi), ...Object.fromEntries(Object.entries(wide(neutralApi)).map(([k, v]) => [`wide.${k}`, v])) };
const bytes = (x) => Buffer.from(x.buffer, x.byteOffset, x.byteLength);
let compared = 0;
let differ = 0;
for (const k of Object.keys(a)) {
  // a query may return an empty list, but every value must be bytes (a missing one is a failure)
  if (!ArrayBuffer.isView(a[k]) || (!k.startsWith("wide.") && a[k].length === 0)) {
    console.error(`neutral_equivalence: the original package gave no ${k}`);
    process.exit(2);
  }
  compared += 1;
  const same = ArrayBuffer.isView(b[k]) && bytes(a[k]).equals(bytes(b[k]));
  if (!same) differ += 1;
  console.log(`${k}: ${same ? "equal" : "DIFFERENT"} (${bytes(a[k]).length} bytes)`);
}
if (compared === 0) {
  console.error("neutral_equivalence: compared nothing");
  process.exit(2);
}
console.log(`neutral equivalence: ${compared} values compared, ${differ} different`);
process.exit(differ ? 1 : 0);
