#!/usr/bin/env python3
"""Tests of scripts/neutral_pkg.py and scripts/neutral_leak_check.py (python3 scripts/test_neutral_pkg.py)

A made-up engine name stands in for the real one: it is planted in every place a name can
reach a package (the wasm export names, the glue's class and method names, a source path and
a home path in the wasm data, a panic location string, an error message, the custom sections
`name` and `producers`, a comment in the glue). The neutral package must hold none of them,
the leak check must find each of them in the input, and the same inputs give the same bytes.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import re
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


npk = load("neutral_pkg")
nlc = load("neutral_leak_check")

ENGINE = "Zorblat"  # made up; stands in for the engine's name
USER = "quuxuser"  # made up; stands in for a home directory's owner
KEY = b"0123456789abcdef0123456789abcdef-test-only"


def section(sid: int, payload: bytes) -> bytes:
    return bytes([sid]) + npk.enc_uleb(len(payload)) + payload


def custom(name: str, body: bytes) -> bytes:
    n = name.encode()
    return section(0, npk.enc_uleb(len(n)) + n + body)


def module() -> bytes:
    exports = npk.enc_exports([
        ("memory", 2, 0),
        ("zorblatworld_new", 0, 0),
        ("zorblatworld_addFrobBody", 0, 1),
        ("zorblatVersion", 0, 2),
        ("__wbg_zorblatworld_free", 0, 3),
        ("__wbindgen_malloc", 0, 4),
    ])
    data = (b"\x01\x00\x0b"  # one active segment (flags 0 would need an offset expr; passive = 1)
            + npk.enc_uleb(0)
            + b"/Users/" + USER.encode() + b"/.rustup/toolchains/x/lib/rustlib/src/rust/library/core/src/fmt/mod.rs"
            + b"\0src/frob_quux.rs\0assertion failed: zorblat index\0" + ENGINE.encode() + b"Error: bad\0kept text\0")
    # rebuild the data payload: a vec of 1 passive segment holding the bytes above
    blob = data[3 + 1:]
    data_payload = npk.enc_uleb(1) + b"\x01" + npk.enc_uleb(len(blob)) + blob
    return (b"\0asm\x01\0\0\0" + section(7, exports) + section(11, data_payload)
            + custom("name", b"\x00\x05" + ENGINE.encode()) + custom("producers", b"rustc " + ENGINE.encode()))


JS = """/** The Zorblat engine. */
export class ZorblatWorld {
    free() { wasm.__wbg_zorblatworld_free(this.__wbg_ptr); }
    static new() { return wasm.zorblatworld_new(); }
    /** add a body */
    addFrobBody(x) { return wasm.zorblatworld_addFrobBody(this.__wbg_ptr, x); }
}
export function zorblatVersion() { return wasm.zorblatVersion(); }
// load
const url = new URL('p_bg.wasm', import.meta.url);
"""
DTS = """/** The Zorblat engine. */
export class ZorblatWorld {
  free(): void;
  static new(): ZorblatWorld;
  addFrobBody(x: number): number;
}
export function zorblatVersion(): string;
"""


def package(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "p_bg.wasm").write_bytes(module())
    (d / "p.js").write_text(JS)
    (d / "p.d.ts").write_text(DTS)
    return d


def run_check(tokens: list[str], *paths: Path) -> tuple[int, str]:
    t = Path(tempfile.mkdtemp()) / "tokens.json"
    t.write_text(json.dumps(tokens))
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = nlc.main(["--tokens", str(t), *map(str, paths)])
    return code, out.getvalue() + err.getvalue()


def made_up_tokens(result_tokens: dict) -> dict:
    extra = [ENGINE.lower(), USER, "rustup", "frob_quux.rs", "mod.rs"]
    ins = sorted(set(result_tokens["insensitive"]) | {hashlib.sha256(t.encode()).hexdigest() for t in extra})
    return {"insensitive": ins, "sensitive": result_tokens["sensitive"]}


class NeutralPackage(unittest.TestCase):
    def setUp(self):
        root = Path(tempfile.mkdtemp())
        self.src = package(root / "in")
        self.out = root / "out"
        self.r = npk.process(self.src, self.out, KEY, [ENGINE], [USER, "rustup", "frob_quux.rs", "mod.rs"])
        self.tokens = made_up_tokens(self.r["tokens"])

    def test_the_output_is_four_hash_named_files(self):
        names = sorted(p.name for p in self.out.iterdir())
        base = self.r["base"]
        self.assertRegex(base, r"^p-[0-9a-f]{12}$")
        self.assertEqual(names, [f"{base}.d.ts", f"{base}.js", f"{base}.manifest.json", f"{base}.wasm"])
        wasm = (self.out / f"{base}.wasm").read_bytes()
        self.assertEqual(base[2:], hashlib.sha256(wasm).hexdigest()[:12])
        self.assertIn(f"{base}.wasm", (self.out / f"{base}.js").read_text())

    def test_the_leak_check_finds_every_planted_form_in_the_input(self):
        code, out = run_check(self.tokens, self.src)
        self.assertEqual(code, 1, out)
        # each kind of place is found: the glue (class, method, comment), the wasm (exports, paths, message)
        self.assertIn("p.js", out)
        self.assertIn("p.d.ts", out)
        self.assertIn("p_bg.wasm", out)

    def test_the_neutral_package_holds_none_of_them(self):
        code, out = run_check(self.tokens, self.out)
        self.assertEqual(code, 0, out)
        wasm = next(self.out.glob("*.wasm")).read_bytes()
        for planted in (b"zorblat", b"Zorblat", USER.encode(), b".rs", b"rustup", b"name", b"producers"):
            self.assertNotIn(planted, wasm.lower() if planted.islower() else wasm, planted)
        self.assertIn(b"kept text", wasm)  # data that names nothing is untouched

    def test_offsets_do_not_move_and_sections_are_dropped(self):
        before = dict((sid, p) for sid, p in npk.sections(module()) if sid == 11)
        after = dict((sid, p) for sid, p in npk.sections(next(self.out.glob("*.wasm")).read_bytes()) if sid == 11)
        self.assertEqual(len(before[11]), len(after[11]))
        self.assertEqual([sid for sid, _ in npk.sections(next(self.out.glob("*.wasm")).read_bytes())], [7, 11])

    def test_every_exported_identifier_is_renamed_consistently(self):
        js = next(self.out.glob("*.js")).read_text()
        dts = next(self.out.glob("*.d.ts")).read_text()
        m = self.r["mapping"]
        for name in ("ZorblatWorld", "addFrobBody", "zorblatVersion", "zorblatworld_addFrobBody", "__wbg_zorblatworld_free"):
            self.assertIn(name, m)
            self.assertRegex(m[name], r"^n[0-9a-f]{10}$")
        self.assertIn(m["addFrobBody"], dts)
        self.assertIn(f"wasm.{m['zorblatworld_addFrobBody']}(", js)
        exp = [n for n, _, _ in npk.exports(next(p for s, p in npk.sections(next(self.out.glob('*.wasm')).read_bytes()) if s == 7))]
        self.assertIn(m["zorblatworld_addFrobBody"], exp)
        self.assertIn("memory", exp)
        self.assertIn("__wbindgen_malloc", exp)
        self.assertNotIn("/*", js)

    def test_the_same_inputs_give_the_same_bytes_and_another_key_other_names(self):
        again = Path(tempfile.mkdtemp())
        r2 = npk.process(self.src, again, KEY, [ENGINE], [])
        self.assertEqual(r2["base"], self.r["base"])
        for f in self.out.iterdir():
            self.assertEqual(f.read_bytes(), (again / f.name).read_bytes(), f.name)
        r3 = npk.process(self.src, Path(tempfile.mkdtemp()), KEY + b"x", [ENGINE], [])
        self.assertNotEqual(r3["mapping"]["ZorblatWorld"], self.r["mapping"]["ZorblatWorld"])

    def test_the_name_map_is_private(self):
        p = Path(tempfile.mkdtemp()) / "sub" / "map.json"
        p.parent.mkdir()
        p.write_text("{}")
        p.chmod(0o644)
        npk.write_private(p, {"a": "n0"})
        self.assertEqual(stat.S_IMODE(p.stat().st_mode), 0o600)
        self.assertEqual(json.loads(p.read_text()), {"a": "n0"})

    def test_no_key_is_an_input_error(self):
        with self.assertRaises(npk.InputError):
            npk.keychain_key("no-such-service-for-a-test", "none")


class LeakCheck(unittest.TestCase):
    def test_each_form_of_a_token_is_found(self):
        d = Path(tempfile.mkdtemp())
        tokens = [hashlib.sha256(b"zorblat").hexdigest()]
        for text in ("x = zorblatVersion()", "ZORBLAT", "a_zorblat_b", "/srv/zorblat/src/a.rs", "init.zorblat",
                     "theZorblatWorld"):
            (d / "f.txt").write_text(text)
            code, out = run_check(tokens, d)
            self.assertEqual(code, 1, (text, out))
            self.assertNotIn("zorblat", out.lower())  # a finding never prints the token

    def test_a_binary_file_is_read_by_its_printable_runs(self):
        d = Path(tempfile.mkdtemp())
        (d / "m.wasm").write_bytes(b"\0\xff\x01xxzorblat_index\0\x80")
        self.assertEqual(run_check([hashlib.sha256(b"zorblat").hexdigest()], d)[0], 1)

    def test_a_clean_file_passes_and_nothing_compared_exits_2(self):
        d = Path(tempfile.mkdtemp())
        (d / "f.txt").write_text("plain words only")
        tokens = [hashlib.sha256(b"zorblat").hexdigest()]
        self.assertEqual(run_check(tokens, d)[0], 0)
        self.assertEqual(run_check(tokens, Path(tempfile.mkdtemp()))[0], 2)  # no file
        self.assertEqual(run_check([], d)[0], 2)  # no token
        self.assertEqual(run_check(["not-a-hash"], d)[0], 2)

    def test_a_malformed_token_set_exits_2(self):
        d = Path(tempfile.mkdtemp())
        (d / "f.txt").write_text("plain words only")
        for bad in ([["x"]], 5, {"insensitive": 5}, {"insensitive": [], "sensitive": [1]}):
            self.assertEqual(run_check(bad, d)[0], 2, bad)
        t = Path(tempfile.mkdtemp()) / "tokens.json"
        t.write_text("{not json")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(nlc.main(["--tokens", str(t), str(d)]), 2)


class DataLiterals(unittest.TestCase):
    """an engine identifier overwritten in the wasm data must not be a string the source uses,
    unless reviewed, and never one that is hashed"""

    SOURCES = {
        "src/scene.rs": 'let a = read(obj, "frob_index")?;\n',
        "src/law.rs": 'let id = law_id_of("frob_tag");\n',
    }

    def test_a_reviewed_literal_and_a_derived_name_pass(self):
        r = npk.check_data_literals(["frob_index", "FrobOutOfRange"], self.SOURCES, {("frob_index", "src/scene.rs")})
        self.assertEqual(r, {"checked": 2, "literals": 1})

    def test_an_unreviewed_literal_stops(self):
        with self.assertRaisesRegex(npk.CheckError, "not on the reviewed list"):
            npk.check_data_literals(["frob_index"], self.SOURCES, {("other_name", "src/x.rs")} | set())

    def test_a_hashed_literal_stops_even_when_reviewed(self):
        with self.assertRaisesRegex(npk.CheckError, "reaches a hash"):
            npk.check_data_literals(["frob_tag"], self.SOURCES, {("frob_tag", "src/law.rs")})

    def test_a_stale_reviewed_line_stops(self):
        with self.assertRaisesRegex(npk.CheckError, "stale"):
            npk.check_data_literals(["frob_index"], self.SOURCES,
                                    {("frob_index", "src/scene.rs"), ("frob_gone", "src/scene.rs")})

    def test_an_identifier_inside_a_longer_literal_counts(self):
        sources = {"src/law.rs": 'h.update(b"world/frob_key/v1");\n'}
        with self.assertRaisesRegex(npk.CheckError, "not on the reviewed list"):
            npk.check_data_literals(["frob_key"], sources, {("x_y", "src/x.rs")})
        with self.assertRaisesRegex(npk.CheckError, "reaches a hash at line 1"):
            npk.check_data_literals(["frob_key"], sources, {("frob_key", "src/law.rs")})

    def test_a_bound_literal_hashed_on_another_line_stops(self):
        sources = {"src/law.rs": 'const TAG: &str = "frob_tag";\nfn id() -> u64 { law_id_of(TAG) }\n'}
        with self.assertRaisesRegex(npk.CheckError, "reaches a hash at line 2"):
            npk.check_data_literals(["frob_tag"], sources, {("frob_tag", "src/law.rs")})
        labelled = {"src/x.rs": 'const S: &str = "frob_tag";\nlet n = r.len(S, 8)?;\n'}
        self.assertEqual(npk.check_data_literals(["frob_tag"], labelled, {("frob_tag", "src/x.rs")}),
                         {"checked": 1, "literals": 1})

    def test_each_binding_form_and_hash_word_is_seen(self):
        for bind in ('let tag = "frob_tag";', 'static TAG: &str = "frob_tag";', 'let mut tag = "frob_tag";'):
            name = bind.split("=")[0].split()[-1].rstrip(":") if ":" not in bind.split("=")[0] else bind.split(":")[0].split()[-1]
            src = {"src/a.rs": f"{bind}\nlet h = digest_of({name});\n"}
            with self.assertRaisesRegex(npk.CheckError, "reaches a hash at line 2", msg=bind):
                npk.check_data_literals(["frob_tag"], src, {("frob_tag", "src/a.rs")})
        for call in ("hash_of", "law_id", "lawid", "digest", "Sha256::new", "sha3", "checksum", "fingerprint", "blake3",
                     "fnv1a", "crc32", "h.update"):
            src = {"src/a.rs": f'let x = {call}("frob_tag");\n'}
            with self.assertRaisesRegex(npk.CheckError, "reaches a hash", msg=call):
                npk.check_data_literals(["frob_tag"], src, {("frob_tag", "src/a.rs")})

    def test_a_list_is_checked_also_when_no_identifier_is_overwritten(self):
        with self.assertRaisesRegex(npk.CheckError, "stale"):
            npk.check_data_literals([], self.SOURCES, {("frob_index", "src/scene.rs")})

    def test_no_source_is_an_input_error(self):
        with self.assertRaises(npk.InputError):
            npk.check_data_literals(["frob_index"], {}, {("frob_index", "src/scene.rs")})

    def test_the_reviewed_list_needs_a_role_and_an_entry(self):
        self.assertEqual(npk.read_reviewed("# c\nfrob_index src/scene.rs json key\n"), {("frob_index", "src/scene.rs")})
        with self.assertRaises(npk.InputError):
            npk.read_reviewed("frob_index src/scene.rs\n")
        with self.assertRaises(npk.InputError):
            npk.read_reviewed("# only a comment\n")

    def test_the_package_refuses_data_identifiers_without_a_review(self):
        root = Path(tempfile.mkdtemp())
        src = package(root / "in")
        with self.assertRaises(npk.InputError):
            npk.process(src, root / "out", KEY, [], [USER], {"ZorblatError"})
        seen = []
        npk.process(src, root / "out2", KEY, [], [USER], {"ZorblatError"}, seen.append)
        self.assertEqual(seen, [["ZorblatError"]])
        seen.clear()
        npk.process(src, root / "out3", KEY, [], [USER], set(), seen.append)
        self.assertEqual(seen, [[]])  # the list is still checked


class CommandLine(unittest.TestCase):
    """main() wires the source check in; the key is replaced by a test key"""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.src = package(self.root / "in")
        (self.root / "repo" / "src").mkdir(parents=True)
        (self.root / "repo" / "src" / "err.rs").write_text('let k = "ZorblatError";\n')
        self.ids = self.root / "api.txt"
        self.ids.write_text("pub enum zorblat_core::ZorblatError\n")
        self.orig = npk.keychain_key
        npk.keychain_key = lambda *a: KEY

    def tearDown(self):
        npk.keychain_key = self.orig

    def main(self, *extra: str) -> int:
        n = len(list(self.root.glob("out*")))
        args = ["--in", str(self.src), "--out", str(self.root / f"out{n}"), "--map", str(self.root / "m" / "map.json"),
                "--tokens", str(self.root / f"t{n}.json"), "--token", USER, "--identifiers", str(self.ids),
                "--crate", "zorblat_core", *extra]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return npk.main(args)

    def test_the_check_runs_from_the_command_line(self):
        listed = self.root / "ok.txt"
        listed.write_text("ZorblatError src/err.rs error kind returned as text\n")
        other = self.root / "other.txt"
        other.write_text("ZorblatOther src/err.rs not this one\n")
        repo = str(self.root / "repo")
        self.assertEqual(self.main("--source-root", repo, "--data-literals", str(listed)), 0)
        self.assertEqual(self.main("--source-root", repo, "--data-literals", str(other)), 1)
        self.assertEqual(self.main("--source-root", repo), 2)  # one without the other
        self.assertEqual(self.main(), 2)  # identifiers in the data and no list
        empty = self.root / "empty"
        empty.mkdir()
        self.assertEqual(self.main("--source-root", str(empty), "--data-literals", str(listed)), 2)


class CommittedList(unittest.TestCase):
    """the repository's reviewed list matches its source (each line still names a literal, none hashed)"""

    def test_the_committed_list_matches_the_source(self):
        repo = HERE.parent
        reviewed = npk.read_reviewed((HERE / "neutral_data_literals.txt").read_text(encoding="utf-8"))
        sources = npk.read_sources(repo)
        self.assertGreater(len(sources), 100)
        # the identifiers the list names, as if each were overwritten: no stale line, no hash
        r = npk.check_data_literals(sorted({i for i, _ in reviewed}), sources, reviewed)
        self.assertEqual(r["literals"], len(reviewed))


def fake_glue(names: list[str], rename, overrides: dict[str, str] = {}, extra: str = "", module_extra: str = "") -> str:
    """a stand-in for the wasm-bindgen glue with the engine's API (method bodies return values of the
    kind the real API returns); `rename` maps an API name to the name the glue uses"""
    def body(n: str) -> str:
        if n in overrides:
            return overrides[n]
        if n.startswith(("add", "register")):
            return "return this.ids++;"
        if n == "getBodyType":
            return "return 0;"
        if n in ("serializeState",):
            return "return new Uint8Array([1, 2, this.ids]);"
        # the whole-world state follows a step counter, so a digest moves with steps and a restore rewinds it
        if n == "step":
            return "this.steps = (this.steps || 0) + 1;"
        if n in ("snapshotWorld", "stateDigest"):
            return "return new Uint8Array([7, this.steps || 0]);"
        if n == "lawId":
            return "return new Uint8Array(32).fill(3);"
        if n == "restoreWorld":
            return "this.steps = a[0][1];"
        if n.startswith(("get", "cast", "observe", "raycast", "overlap")):
            return "return new Float64Array([this.ids, 0.5]);"
        # the cloth exports: a grab gives a handle, the readings arrays, the switches booleans
        if n == "clothGrab":
            return "return this.ids++;"
        if n in ("clothPositions", "clothRestPositions"):
            return "return new Float64Array([this.ids, 0.25, 0.5]);"
        if n == "clothTriangles":
            return "return new Uint32Array([0, 1, 2]);"
        if n in ("clothStateBytes", "clothStateDigest"):
            return "return new Uint8Array([5, this.ids]);"
        if n in ("clothAsleep", "clothMoveGrab", "clothRelease", "clothReleaseWithVelocity"):
            return "return true;"
        if n.endswith("Count"):
            return "return this.ids;"
        if n.startswith(("remove", "clear", "deserialize")) or n in ("setCollisionRadius", "setBodyShape", "setBodyMaterial"):
            return "return true;"
        return "return undefined;"
    methods = "\n".join(f"  {rename(n)}(...a) {{ {body(n)} }}" for n in names
                         if n not in ("withConfig", "alicePhysicsVersion", "WasmPhysicsWorld"))
    return (f"export function initSync() {{}}\n"
            f"export class {rename('WasmPhysicsWorld')} {{\n  constructor() {{ this.ids = 0; }}\n"
            f"  static {rename('withConfig')}() {{ return new this(); }}\n{methods}\n{extra}\n  free() {{}}\n}}\n"
            + (f"export function {rename('alicePhysicsVersion')}() {{ return \"0.0.0\"; }}\n"
               if "alicePhysicsVersion" in names else "") + module_extra)


class Equivalence(unittest.TestCase):
    """scripts/neutral_equivalence.mjs against a stand-in package: it passes only when every export is
    called and every value is present, non-empty and equal"""

    NAMES = sorted(set(re.findall(r'js_name = "([A-Za-z]+)"', (HERE.parent / "src" / "wasm.rs").read_text(encoding="utf-8")))
                   | {"step", "WasmPhysicsWorld"})

    def run_pair(self, orig: dict = {}, neutral: dict = {}, orig_extra: str = "", drop: str = "", orig_module: str = "") -> int:
        d = Path(tempfile.mkdtemp())
        mapping = {n: "n" + hashlib.sha256(n.encode()).hexdigest()[:10] for n in self.NAMES}
        (d / "o").mkdir()
        (d / "o" / "p.js").write_text(fake_glue(self.NAMES, lambda n: n, orig, orig_extra, orig_module))
        (d / "o" / "p_bg.wasm").write_bytes(b"\0asm")
        (d / "n").mkdir()
        names = [n for n in self.NAMES if n != drop]
        (d / "n" / "p-0123456789ab.js").write_text(fake_glue(names, mapping.get, neutral))
        (d / "n" / "p-0123456789ab.wasm").write_bytes(b"\0asm")
        (d / "map.json").write_text(json.dumps({"api": mapping}))
        r = subprocess.run(["node", str(HERE / "neutral_equivalence.mjs"), str(d / "o"), str(d / "n"), str(d / "map.json")],
                           capture_output=True, text=True)
        self.last = r.stdout + r.stderr
        return r.returncode

    def test_the_stand_in_names_are_the_engine_api(self):
        self.assertGreater(len(self.NAMES), 40)

    def test_equal_packages_pass_with_every_export_called(self):
        self.assertEqual(self.run_pair(), 0, self.last)
        self.assertRegex(self.last, r"API exports called: (\d+) of \1")

    def test_a_different_value_fails(self):
        self.assertEqual(self.run_pair(neutral={"getPositions": "return new Float64Array([9]);"}), 1, self.last)

    def test_a_missing_name_stops(self):
        self.assertEqual(self.run_pair(drop="castCapsule"), 2, self.last)
        self.assertEqual(self.run_pair(drop="alicePhysicsVersion"), 2, self.last)
        self.assertEqual(self.run_pair(drop="setFriction"), 2, self.last)  # a setter returns nothing

    def test_an_export_no_scene_calls_stops(self):
        self.assertEqual(self.run_pair(orig_extra="  neverCalled() { return 1; }"), 2, self.last)
        self.assertIn("neverCalled", self.last)
        self.assertEqual(self.run_pair(orig_extra="  static neverStatic() { return 1; }"), 2, self.last)
        self.assertIn("neverStatic", self.last)
        self.assertEqual(self.run_pair(orig_module="export function neverFree() { return 1; }\n"), 2, self.last)
        self.assertIn("neverFree", self.last)

    def test_a_refused_add_or_an_empty_value_stops(self):
        self.assertEqual(self.run_pair(orig={"addHingeJoint": "return undefined;"},
                                       neutral={"addHingeJoint": "return undefined;"}), 2, self.last)
        self.assertEqual(self.run_pair(orig={"castSphere": "return new Float64Array([]);"},
                                       neutral={"castSphere": "return new Float64Array([]);"}), 2, self.last)
        self.assertEqual(self.run_pair(orig={"setBodyShape": "return undefined;"},
                                       neutral={"setBodyShape": "return undefined;"}), 2, self.last)


if __name__ == "__main__":
    unittest.main(verbosity=1)
