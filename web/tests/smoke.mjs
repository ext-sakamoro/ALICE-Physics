// Smoke test of the built viewer (web/dist) in Chrome.
//
//   npm run build && npm run smoke
//
// Serves web/dist on 127.0.0.1, opens the viewer in test mode (?steps=N, see src/main.ts)
// with Playwright driving the installed Google Chrome (or CHROME_PATH), and checks:
//   1. the page loads the wasm package (its init log) with no page error;
//   2. there is at least one dynamic body, and after N steps every dynamic body rests on the
//      floor: its centre is not below the floor plane y = 0 by more than PENETRATION and its
//      vertical speed is under REST_SPEED;
//   3. two fresh loads with the same seed give the same state bit for bit (SHA-256 of the
//      positions, rotations and velocities as f64 bytes; the number of compared values is
//      checked to be > 0);
//   4. another seed gives another state (the digest is not a constant).
// Exit 0 when all hold, 1 on a failed check, 2 when it could not run (no dist, no Chrome).

import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';

const STEPS = 1200; // 20 s of simulated time at 1/60
const SPHERE_RADIUS = 0.5; // the viewer's body radius (src/main.ts)
const PENETRATION = 0.05;
const REST_SPEED = 0.1;
const LOAD_TIMEOUT_MS = 60_000;

const dist = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'dist');
if (!fs.existsSync(path.join(dist, 'index.html'))) {
  console.error(`smoke: ${dist}/index.html is missing (run npm run build first)`);
  process.exit(2);
}

const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.wasm': 'application/wasm', '.css': 'text/css' };
const server = http.createServer((req, res) => {
  const rel = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
  const file = path.join(dist, rel === '/' ? 'index.html' : rel);
  if (!file.startsWith(dist + path.sep)) {
    res.writeHead(403);
    res.end();
    return;
  }
  fs.readFile(file, (err, data) => {
    if (err) {
      res.writeHead(404);
      res.end();
      return;
    }
    res.writeHead(200, { 'content-type': MIME[path.extname(file)] ?? 'application/octet-stream' });
    res.end(data);
  });
});
await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
const base = `http://127.0.0.1:${server.address().port}`;

let browser;
try {
  browser = await chromium.launch(
    process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : { channel: 'chrome' },
  );
} catch (e) {
  console.error(`smoke: could not start Chrome (${e.message.split('\n')[0]})`);
  server.close();
  process.exit(2);
}
console.log(`smoke: Chrome ${browser.version()}, ${STEPS} steps`);

async function load(seed) {
  const page = await browser.newPage();
  const errors = [];
  const logs = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => logs.push(m.text()));
  await page.goto(`${base}/?seed=${seed}&steps=${STEPS}`);
  await page.waitForFunction(() => window.__alicePhysicsViewer, null, { timeout: LOAD_TIMEOUT_MS });
  const state = await page.evaluate(() => window.__alicePhysicsViewer);
  await page.close();
  return { state, errors, logs };
}

function digest(state) {
  const values = [...state.positions, ...state.rotations, ...state.velocities];
  const bytes = Buffer.from(new Float64Array(values).buffer);
  return { sha256: crypto.createHash('sha256').update(bytes).digest('hex'), count: values.length };
}

const failures = [];
const fail = (msg) => failures.push(msg);

try {
  const first = await load(1);
  const second = await load(1);
  const other = await load(2);

  for (const [name, run] of [['seed 1 (1st)', first], ['seed 1 (2nd)', second], ['seed 2', other]]) {
    if (run.errors.length) fail(`${name}: page errors: ${run.errors.join(' | ')}`);
    if (!run.logs.some((l) => l.startsWith('[ALICE-Physics] WASM initialized'))) fail(`${name}: no wasm init log`);
  }

  const s = first.state;
  const dynamic = s.bodyTypes.flatMap((t, i) => (t === 0 ? [i] : []));
  console.log(`smoke: version ${s.version}, ${s.bodyTypes.length} bodies, ${dynamic.length} dynamic`);
  if (dynamic.length === 0) fail('no dynamic body (compared nothing)');
  if (dynamic.length) {
    const ys = dynamic.map((i) => s.positions[i * 3 + 1]);
    const vys = dynamic.map((i) => Math.abs(s.velocities[i * 3 + 1]));
    console.log(`smoke: centre y ${Math.min(...ys)} .. ${Math.max(...ys)}, max |vy| ${Math.max(...vys)}`);
  }
  for (const i of dynamic) {
    const y = s.positions[i * 3 + 1];
    const vy = s.velocities[i * 3 + 1];
    if (!Number.isFinite(y) || y < SPHERE_RADIUS - PENETRATION) {
      fail(`body ${i}: centre y = ${y}, below the floor (needs >= ${SPHERE_RADIUS - PENETRATION})`);
    } else if (!(Math.abs(vy) < REST_SPEED)) {
      fail(`body ${i}: vertical speed ${vy} after ${STEPS} steps (needs < ${REST_SPEED})`);
    }
  }

  const a = digest(first.state);
  const b = digest(second.state);
  const c = digest(other.state);
  console.log(`smoke: digest seed 1 ${a.sha256} / ${b.sha256} (${a.count} values), seed 2 ${c.sha256}`);
  if (a.count === 0 || a.count !== b.count) fail(`compared ${a.count} / ${b.count} values`);
  else if (a.sha256 !== b.sha256) fail('two loads with the same seed differ');
  if (a.sha256 === c.sha256) fail('seed 2 gives the same state as seed 1 (the digest does not depend on the scene)');
} catch (e) {
  fail(`run: ${e.message.split('\n')[0]}`);
} finally {
  await browser.close();
  server.close();
}

if (failures.length) {
  for (const f of failures) console.error(`smoke: FAIL ${f}`);
  process.exit(1);
}
console.log('smoke: OK');
