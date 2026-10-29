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
import stat
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
        for text in ("x = zorblatVersion()", "ZORBLAT", "a_zorblat_b", "/home/u/zorblat/src/a.rs", "init.zorblat",
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


if __name__ == "__main__":
    unittest.main(verbosity=1)
