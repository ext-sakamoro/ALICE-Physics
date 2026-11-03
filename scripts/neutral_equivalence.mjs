// The neutral package behaves as the package it was made from, bit for bit.
//
//   node scripts/neutral_equivalence.mjs <original-dir> <neutral-dir> <name-map.json>
//
// <original-dir> holds the wasm-bindgen `--target web --out-name p` output, <neutral-dir> the
// package scripts/neutral_pkg.py made from it, <name-map.json> its private name map. The same
// scene (a static plane, two dynamic bodies on a ball joint, 120 steps of 1/60) runs through both,
// and the positions, rotations and the whole serialized state (`serializeState`: positions,
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

const neutralBase = fs.readdirSync(neutralDir).find((f) => /^p-[0-9a-f]{12}\.js$/.test(f))?.slice(0, -3);
if (!neutralBase) {
  console.error("neutral_equivalence: no p-<h12>.js in the neutral directory");
  process.exit(2);
}
const a = run(api(await load(origDir, "p"), false));
const b = run(api(await load(neutralDir, neutralBase), true));
const bytes = (x) => Buffer.from(x.buffer, x.byteOffset, x.byteLength);
let compared = 0;
let differ = 0;
for (const k of Object.keys(a)) {
  if (!a[k] || a[k].length === 0) {
    console.error(`neutral_equivalence: the original package gave no ${k}`);
    process.exit(2);
  }
  compared += 1;
  const same = b[k] && bytes(a[k]).equals(bytes(b[k]));
  if (!same) differ += 1;
  console.log(`${k}: ${same ? "equal" : "DIFFERENT"} (${bytes(a[k]).length} bytes)`);
}
if (compared === 0) {
  console.error("neutral_equivalence: compared nothing");
  process.exit(2);
}
console.log(`neutral equivalence: ${compared} values compared, ${differ} different`);
process.exit(differ ? 1 : 0);
