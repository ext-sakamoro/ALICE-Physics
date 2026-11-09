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

// the API of the original package: the methods and static functions of its world class, its free
// functions and the constructor (the glue's own helpers that free a world are not API)
function exportsOf(glue) {
  const World = glue.WasmPhysicsWorld;
  const own = new Set(["constructor", "free", "__destroy_into_raw", "__wrap"]);
  const fns = (o) => Object.getOwnPropertyNames(o).filter((n) => typeof o[n] === "function" && !own.has(n) && !n.startsWith("__"));
  const free = Object.keys(glue).filter((n) => typeof glue[n] === "function" && !["initSync", "default", "WasmPhysicsWorld"].includes(n) && !n.startsWith("__"));
  return new Set([...fns(World.prototype), ...fns(World), ...free, "new"]);
}

// calls a function of the original name on either package; a name the package does not have is an
// error (never a silent undefined, which would compare as equal on both sides), and every name
// called is recorded so that the run can require the whole API
function api(glue, neutral, called) {
  const name = (n) => {
    if (neutral && !(n in map)) throw new Error(`the name map has no ${n}`);
    return neutral ? map[n] : n;
  };
  const World = glue[name("WasmPhysicsWorld")];
  if (typeof World !== "function") throw new Error("the package has no world class");
  const fn = (holder, n) => {
    const f = holder[name(n)];
    if (typeof f !== "function") throw new Error(`the package has no ${n}`);
    called.add(n);
    return f;
  };
  return {
    world: () => fn(World, "withConfig").call(World, 0, -10, 0, 8, 4),
    fresh: () => {
      called.add("new");
      return new World();
    },
    call: (w, n, ...args) => fn(w, n).apply(w, args),
    free: (n, ...args) => fn(glue, n)(...args),
  };
}

const f = (x) => new Float64Array(x);
// an add must succeed: an id that is not a number (a refused add gives undefined) would compare as
// equal on both sides, so it stops the run
const id = (x, what) => {
  if (typeof x !== "number" || !Number.isFinite(x)) throw new Error(`${what} gave no id (${x})`);
  return x;
};
// a yes / no result or a count, as bytes
const val = (x, what) => {
  if (typeof x !== "boolean" && typeof x !== "number") throw new Error(`${what} gave ${x}`);
  return f([Number(x)]);
};

function run(a) {
  const w = a.world();
  const plane = id(a.call(w, "addStaticPlane", 0, 1, 0, 0), "addStaticPlane");
  const b0 = id(a.call(w, "addDynamicBody", 0, 5, 0, 1), "addDynamicBody");
  const b1 = id(a.call(w, "addDynamicBody", 1, 5, 0, 2), "addDynamicBody");
  const joint = id(a.call(w, "addBallJoint", b0, b1, f([0, 0, 0, -1, 0, 0])), "addBallJoint");
  for (let i = 0; i < 120; i++) a.call(w, "step", 1 / 60);
  return {
    ids: f([plane, b0, b1, joint]),
    positions: a.call(w, "getPositions"),
    rotations: a.call(w, "getRotations"),
    serializedState: a.call(w, "serializeState"),
  };
}

// a second scene through the rest of the API: every joint kind, shaped bodies, a height field and a
// triangle mesh, material and gravity setters, a magnet, impulses, the scene queries, removal, and a save /
// step / load round trip; every value the API returns is compared (the setters return nothing)
function wide(a) {
  const w = a.world();
  const out = {};
  const add = (n, ...args) => id(a.call(w, n, ...args), n);
  out.statics = f([
    add("addStaticPlane", 0, 1, 0, 0),
    add("addStaticHeightField", f([0, 0.1, 0.2, 0.1, 0.3, 0.1, 0, 0.2, 0]), 3, 3, 1, 4, 0, 4),
    add("addStaticTriMesh", f([-8, 0.5, -8, -6, 0.5, -8, -8, 0.5, -6]), new Uint32Array([0, 1, 2])),
  ]);
  const b = [];
  for (let i = 0; i < 6; i++) b.push(add("addDynamicBody", i * 0.7, 2 + i, 0.3 * i, 1 + i));
  b.push(add("addStaticBody", 3, 0.5, 3));
  b.push(add("addKinematicBody", -3, 1, -3));
  b.push(add("addShapedBody", 0, 0.3, 0.2, 0.4, 2, 1, 6, 1));
  b.push(add("addShapedBody", 1, 0.2, 0.5, 0, 1, -1, 6, 1));
  b.push(add("addShapedBody", 3, 0.3, 0.2, 0.25, 1, 2, 7, -1));
  out.bodies = f(b);
  out.joints = f([
    add("addHingeJoint", b[0], b[1], f([0.3, 0, 0, -0.3, 0, 0]), f([0, 0, 1, 0, 0, 1])),
    add("addFixedJoint", b[1], b[2], f([0, 0.4, 0, 0, -0.4, 0]), f([0, 0, 0, 1])),
    add("addSliderJoint", b[2], b[3], f([1, 0, 0]), f([0, 0, 0, 0, 0, 0])),
    add("addSpringJoint", b[3], b[4], f([0, 0, 0, 0, 0, 0]), 1.2, 40, 0.5),
    add("addBallJoint", b[4], b[5], f([0, 0, 0, -0.5, 0, 0])),
  ]);
  a.call(w, "setRestitution", b[0], 0.6);
  a.call(w, "setFriction", b[1], 0.2);
  a.call(w, "setGravityScale", b[2], 0.5);
  out.radiusSet = val(a.call(w, "setCollisionRadius", b[3], 0.45), "setCollisionRadius");
  // a registered material for body 0 and a magnet beside its fall, pulling that material
  const mat = id(a.call(w, "registerMaterial", 0.4, 0.3), "registerMaterial");
  out.material = f([mat]);
  out.materialSet = val(a.call(w, "setBodyMaterial", b[0], mat), "setBodyMaterial");
  out.magnet = f([id(a.call(w, "addMagnet", 0.3, 0.8, 0, 0, 400, 0, 0.1, mat, 3, 1.7e6, 7850), "addMagnet")]);
  // an orientation and a spin set directly (both report whether the body exists)
  out.rotationSet = val(a.call(w, "setRotation", b[1], 0, 0.3826834323650898, 0, 0.9238795325112867), "setRotation");
  out.spinSet = val(a.call(w, "setAngularVelocity", b[2], 0, 0, 1.5), "setAngularVelocity");
  // a body locked to the x-y plane (z position, x and y rotations), and the lock read back
  out.axisLockSet = val(a.call(w, "setBodyAxisLock", b[4], 28), "setBodyAxisLock");
  out.axisLock = val(a.call(w, "bodyAxisLock", b[4]), "bodyAxisLock");
  // a wind zone over the scene: a resident, in the snapshot and the digest from here on
  out.wind = f([id(a.call(w, "addWindZone", -9, -9, -9, 9, 9, 9, 1, 0, 0.5, 2, 0.5, 1.5, 1.2, 0.9, 0.3), "addWindZone")]);
  // a pool under the scene: another resident, its count in the snapshot and the digest
  out.pool = f([id(a.call(w, "addBuoyancyZone", -9, -9, -9, 9, 0.5, 9, 40, 10, 0.75, 0.3), "addBuoyancyZone")]);
  // a rope tied to one body: a resident whose particles are in the snapshot and the digest
  out.rope = f([id(a.call(w, "addRope", [0, 6, 0], [1, 5, 0], 4, 0.5, 0.99, null, [0, 0, 0], b[5], [0, 0, 0]), "addRope")]);
  a.call(w, "applyImpulse", b[5], 1, 2, -0.5);
  a.call(w, "setVelocity", b[4], 0.5, 0, 0.25);
  a.call(w, "setKinematicTarget", b[7], -2.5, 1.2, -3, 0, 0.3826834323650898, 0, 0.9238795325112867);
  a.call(w, "setPosition", b[6], 3, 0.6, 3.2);
  a.call(w, "setVelocitiesBatch", f(b.flatMap((_, i) => [0.1 * i, 0, -0.05 * i])));
  a.call(w, "applyImpulsesBatch", f([b[0], 0, 1, 0, b[2], 0.5, 0, 0, 99, 1, 1, 1]));
  out.cleared = val(a.call(w, "clearCollisionRadius", b[3]), "clearCollisionRadius");
  out.clearedAgain = val(a.call(w, "clearCollisionRadius", b[3]), "clearCollisionRadius");
  a.call(w, "setSubsteps", 6);
  a.call(w, "stepN", 1 / 60, 90);
  a.call(w, "setGravity", 0, -9.5, 0.3);
  out.shapeSet = val(a.call(w, "setBodyShape", b[8], 2, 0.3, 0.6, 0), "setBodyShape");
  // the contacts of a step that has some (an empty array compares nothing)
  for (let i = 0; i < 60; i++) {
    a.call(w, "step", 1 / 60);
    const events = a.call(w, "getContactEvents");
    if (out.contactEvents === undefined && events.length > 0) out.contactEvents = events;
  }
  if (out.contactEvents === undefined) throw new Error("getContactEvents returned nothing in 60 steps");
  out.removed = val(a.call(w, "removeJoint", 1), "removeJoint");
  out.removedStatic = val(a.call(w, "removeStaticCollider", 2), "removeStaticCollider");
  out.counts = f(["bodyCount", "jointCount", "staticColliderCount"].map((n) => val(a.call(w, n), n)[0]));
  out.types = f(b.map((i) => val(a.call(w, "getBodyType", i), "getBodyType")[0]));
  // the queries must hit something (an empty result compares nothing); checked with every value below
  out.raycast = a.call(w, "raycast", 0, 10, 0, 0, -1, 0, 50, 4);
  out.castRay = a.call(w, "castRay", f([0.2, 10, 0.1]), f([0, -1, 0]), 50, undefined);
  out.castSphere = a.call(w, "castSphere", f([1, 10, 0]), 0.3, f([0, -1, 0]), 50, undefined);
  out.castCapsule = a.call(w, "castCapsule", f([0, 10, 1]), f([0, 10.5, 1]), 0.2, f([0, -1, 0]), 50, undefined);
  out.overlap = a.call(w, "overlapSphere", f([0, 1, 0]), 3, undefined);
  out.observe = a.call(w, "observeBody", b[0]);
  // the rope's particles after the steps (the rope is the third resident)
  out.ropePositions = a.call(w, "ropePositions", out.rope[0]);
  out.states = a.call(w, "getStates");
  out.velocities = a.call(w, "getVelocities");
  const saved = a.call(w, "serializeState");
  out.saved = saved;
  for (let i = 0; i < 30; i++) a.call(w, "step", 1 / 60);
  out.after = a.call(w, "serializeState");
  out.loaded = val(a.call(w, "deserializeState", saved), "deserializeState");
  out.reloaded = a.call(w, "serializeState");
  // the whole-world snapshot and its identifiers, then a restore after more steps: the steps must move
  // the digest (a restore that does nothing would otherwise compare as equal on both sides)
  const snap = a.call(w, "snapshotWorld");
  out.snapshot = snap;
  out.digest = a.call(w, "stateDigest");
  out.lawId = a.call(w, "lawId");
  for (let i = 0; i < 10; i++) a.call(w, "step", 1 / 60);
  out.movedDigest = a.call(w, "stateDigest");
  if (out.movedDigest.every((x, i) => x === out.digest[i])) throw new Error("the steps after snapshotWorld did not move the digest");
  if (a.call(w, "restoreWorld", snap) !== undefined) throw new Error("restoreWorld returned a value");
  out.restoredDigest = a.call(w, "stateDigest");
  // a soft letter: a square outline cut into a cloth, grabbed at its centre, moved, read back,
  // released, grabbed again and thrown; every cloth export is called and every value compared
  const cloth = id(
    a.call(w, "addClothOutline", f([0, 0, 0, 1, 0, 0, 0, 1, 0]), f([0, 0, 0.3, 0, 0.3, 0.3, 0, 0.3]),
      new Uint32Array([4]), 0.1, 0.01, 0, 0, 4, 2, 0.98, 0, 0),
    "addClothOutline",
  );
  if (cloth < 0) throw new Error("addClothOutline refused the square");
  out.clothCount = val(a.call(w, "clothCount"), "clothCount");
  out.clothParticles = val(a.call(w, "clothParticleCount", cloth), "clothParticleCount");
  out.clothRest = a.call(w, "clothRestPositions", cloth);
  out.clothTriangles = a.call(w, "clothTriangles", cloth);
  const grab = id(a.call(w, "clothGrab", cloth, 0.15, 0.15, 0, 0.2), "clothGrab");
  if (grab < 0) throw new Error("clothGrab reached no particle");
  out.clothMoved = val(a.call(w, "clothMoveGrab", grab, 0.15, 0.25, 0.1), "clothMoveGrab");
  for (let i = 0; i < 10; i++) a.call(w, "step", 1 / 60);
  out.clothPositions = a.call(w, "clothPositions", cloth);
  out.clothAsleep = val(a.call(w, "clothAsleep", cloth), "clothAsleep");
  out.clothReleased = val(a.call(w, "clothRelease", grab), "clothRelease");
  const grab2 = id(a.call(w, "clothGrab", cloth, 0, 0, 0, 0.2), "clothGrab");
  if (grab2 < 0) throw new Error("the second clothGrab reached no particle");
  out.clothThrown = val(a.call(w, "clothReleaseWithVelocity", grab2, 0, 0.5, 0), "clothReleaseWithVelocity");
  for (let i = 0; i < 5; i++) a.call(w, "step", 1 / 60);
  out.clothBytes = a.call(w, "clothStateBytes", cloth);
  out.clothDigest = a.call(w, "clothStateDigest", cloth);
  const version = a.free("alicePhysicsVersion");
  if (typeof version !== "string") throw new Error(`alicePhysicsVersion gave ${version}`);
  out.version = new TextEncoder().encode(version);
  const sid = a.free("semanticsId");
  if (!(sid instanceof Uint8Array) || sid.length !== 32) throw new Error(`semanticsId gave ${sid}`);
  out.semanticsId = sid;
  const fresh = a.fresh();
  out.freshCount = val(a.call(fresh, "bodyCount"), "bodyCount");
  return out;
}

const neutralBase = fs.readdirSync(neutralDir).find((f) => /^p-[0-9a-f]{12}\.js$/.test(f))?.slice(0, -3);
if (!neutralBase) {
  console.error("neutral_equivalence: no p-<h12>.js in the neutral directory");
  process.exit(2);
}
const origGlue = await load(origDir, "p");
const calledA = new Set();
const calledB = new Set();
let a;
let b;
try {
  const origApi = api(origGlue, false, calledA);
  const neutralApi = api(await load(neutralDir, neutralBase), true, calledB);
  a = { ...run(origApi), ...Object.fromEntries(Object.entries(wide(origApi)).map(([k, v]) => [`wide.${k}`, v])) };
  b = { ...run(neutralApi), ...Object.fromEntries(Object.entries(wide(neutralApi)).map(([k, v]) => [`wide.${k}`, v])) };
} catch (e) {
  console.error(`neutral_equivalence: ${e.message}`);
  process.exit(2);
}
// the scenes must go through every export of the package, on both sides
const exported = exportsOf(origGlue);
const missed = [...exported].filter((n) => !calledA.has(n)).sort();
const sameCalls = calledA.size === calledB.size && [...calledA].every((n) => calledB.has(n));
console.log(`API exports called: ${exported.size - missed.length} of ${exported.size}`);
if (exported.size === 0 || missed.length || !sameCalls) {
  console.error(`neutral_equivalence: not called: ${missed.join(", ") || "(none)"}; same calls on both sides: ${sameCalls}`);
  process.exit(2);
}
const bytes = (x) => Buffer.from(x.buffer, x.byteOffset, x.byteLength);
let compared = 0;
let differ = 0;
for (const k of Object.keys(a)) {
  // every value must be bytes and not empty (a missing or empty one would compare nothing)
  if (!ArrayBuffer.isView(a[k]) || a[k].length === 0) {
    console.error(`neutral_equivalence: the original package gave no ${k} (or an empty one)`);
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
