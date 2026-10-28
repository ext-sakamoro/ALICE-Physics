// Smoke test of the built viewer (web/dist) in Chrome.
//
//   npm run build && npm run smoke
//
// Serves web/dist on 127.0.0.1, opens the viewer in test mode (?steps=N, see src/main.ts)
// with Playwright driving the installed Google Chrome (or CHROME_PATH), and checks:
//   1. the page loads the wasm package (its init log) with no page error;
//   2. there is at least one dynamic body, and after N steps every dynamic body rests: its
//      vertical speed is under REST_SPEED and its centre is within PENETRATION of resting height
//      on the floor (y = radius) or on top of another body (centre distance 2 radius, above it);
//      a body hanging in the air or sunk through the floor fails (the check is run on such
//      made-up states first, so a check that cannot fail fails the smoke);
//   3. two fresh loads with the same seed give the same state bit for bit (SHA-256 of the
//      positions, rotations and velocities as f64 bytes; the number of compared values is
//      checked to be > 0);
//   4. another seed gives another state (the digest is not a constant);
//   5. an unreadable ?steps does not enter the test mode.
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

// what is wrong with the resting state of the dynamic bodies `ids` (empty: all rest)
function restProblems(positions, velocities, ids) {
  const p = (i) => [positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]];
  const out = [];
  for (const i of ids) {
    const [x, y, z] = p(i);
    const vy = velocities[i * 3 + 1];
    if (![x, y, z, vy].every(Number.isFinite)) {
      out.push(`body ${i}: not finite`);
      continue;
    }
    const onFloor = Math.abs(y - SPHERE_RADIUS) <= PENETRATION;
    const onBody = ids.some((j) => {
      if (j === i) return false;
      const [a, b, c] = p(j);
      const d = Math.hypot(x - a, y - b, z - c);
      return y > b + PENETRATION && Math.abs(d - 2 * SPHERE_RADIUS) <= PENETRATION;
    });
    if (!onFloor && !onBody) out.push(`body ${i}: centre y = ${y}, neither on the floor (y = ${SPHERE_RADIUS} ± ${PENETRATION}) nor on another body`);
    else if (!(Math.abs(vy) < REST_SPEED)) out.push(`body ${i}: vertical speed ${vy} after ${STEPS} steps (needs < ${REST_SPEED})`);
  }
  return out;
}

// the rest check must fail on bodies hanging in the air (zero gravity), sunk through the floor or
// moving, and pass on bodies on the floor and on a body stacked on another
{
  const r = SPHERE_RADIUS;
  const cases = [
    ['hanging in zero gravity', [0, 10, 0, 2, 14, 0], [0, 0, 0, 0, 0, 0], true],
    ['sunk through the floor', [0, -288, 0], [0, -16, 0], true],
    ['bouncing on the floor', [0, r, 0], [0, 3, 0], true],
    ['resting on the floor', [0, r, 0, 3, r, 0], [0, 0, 0, 0, 0, 0], false],
    ['stacked on another body', [0, r, 0, 0, 3 * r, 0], [0, 0, 0, 0, 0, 0], false],
  ];
  for (const [what, pos, vel, bad] of cases) {
    const ids = [...Array(pos.length / 3).keys()];
    if ((restProblems(pos, vel, ids).length > 0) !== bad) {
      console.error(`smoke: FAIL the rest check ${bad ? 'passes' : 'fails'} bodies ${what}`);
      process.exit(1);
    }
  }
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
  for (const problem of restProblems(s.positions, s.velocities, dynamic)) fail(problem);
  if (dynamic.length) {
    const ys = dynamic.map((i) => s.positions[i * 3 + 1]);
    const vys = dynamic.map((i) => Math.abs(s.velocities[i * 3 + 1]));
    console.log(`smoke: centre y ${Math.min(...ys)} .. ${Math.max(...ys)}, max |vy| ${Math.max(...vys)}`);
  }

  const a = digest(first.state);
  const b = digest(second.state);
  const c = digest(other.state);
  console.log(`smoke: digest seed 1 ${a.sha256} / ${b.sha256} (${a.count} values), seed 2 ${c.sha256}`);
  if (a.count === 0 || a.count !== b.count) fail(`compared ${a.count} / ${b.count} values`);
  else if (a.sha256 !== b.sha256) fail('two loads with the same seed differ');
  if (a.sha256 === c.sha256) fail('seed 2 gives the same state as seed 1 (the digest does not depend on the scene)');

  // 5. an unreadable ?steps: no test state, and the page says why
  const page = await browser.newPage();
  const errors = [];
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
  await page.goto(`${base}/?steps=abc`);
  // the version label leaves its placeholder '-' once the wasm package has loaded (or failed)
  await page.waitForFunction(() => document.getElementById('version')?.textContent !== '-', null, { timeout: LOAD_TIMEOUT_MS });
  await page.waitForTimeout(500);
  if (await page.evaluate(() => window.__alicePhysicsViewer !== undefined)) {
    fail('?steps=abc entered the test mode');
  }
  if (!errors.some((e) => e.includes('is not a step count'))) fail('?steps=abc was not reported');
  await page.close();
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
