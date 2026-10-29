#!/usr/bin/env python3
"""Turn a wasm-bindgen `--target web --out-name p` package into a neutral one.

    neutral_pkg.py --in DIR --out DIR --map FILE --tokens FILE
                   [--name NAME ...] [--key-service SERVICE --key-account ACCOUNT]

The input directory holds `p.js`, `p_bg.wasm` and `p.d.ts`. The output directory receives
exactly four files, named by the content hash h12 (first 12 hex digits of the SHA-256 of
the final wasm): `p-<h12>.wasm`, `p-<h12>.js`, `p-<h12>.d.ts`, `p-<h12>.manifest.json`.

  * every identifier the package exports (the wasm export names, and the class, method
    and function names of the glue and its declarations) becomes `n` + the first 10 hex
    digits of HMAC-SHA256(key, name): the engine's API names are public, so a plain hash
    could be reversed by hashing them; the key comes from the macOS Keychain (`security
    find-generic-password -s SERVICE -a ACCOUNT -w`) and the run fails without it
  * the wasm custom sections `name` and `producers` are dropped
  * every source path (a run ending in `.rs`) inside the wasm data is overwritten in
    place with filler of the same length, and every `--name` (an identifier or word the
    package must not carry: the engine's name, its type names) found in the wasm data is
    overwritten in place too, so no offset moves and no behaviour changes except the
    text of a panic message
  * comments are removed from the glue and the declarations (they carry the engine's
    documentation)
  * the engine name -> neutral name map is written to FILE with mode 600 (keep it out of
    every repository); the manifest holds only the hashes, sizes and the neutral API
  * TOKENS gets the SHA-256 set the leak check reads (`neutral_leak_check.py`)

The output is a pure function of the input bytes, the names and the key: the same inputs
give the same bytes. Exit 0 on success, 1 on a check that failed, 2 on a usage or input
problem (no key, a missing input file, nothing renamed).
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import subprocess
import sys
from pathlib import Path


class InputError(Exception):
    pass


# ----------------------------------------------------------------------------------------
# wasm sections


def uleb(data: bytes, i: int) -> tuple[int, int]:
    n = shift = 0
    while True:
        b = data[i]
        i += 1
        n |= (b & 0x7F) << shift
        shift += 7
        if not b & 0x80:
            return n, i


def enc_uleb(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def sections(wasm: bytes) -> list[tuple[int, bytes]]:
    if wasm[:8] != b"\0asm\x01\0\0\0":
        raise InputError("not a wasm module (version 1)")
    out, i = [], 8
    while i < len(wasm):
        sid = wasm[i]
        size, j = uleb(wasm, i + 1)
        out.append((sid, wasm[j:j + size]))
        i = j + size
    return out


def build(secs: list[tuple[int, bytes]]) -> bytes:
    return b"\0asm\x01\0\0\0" + b"".join(bytes([sid]) + enc_uleb(len(p)) + p for sid, p in secs)


def custom_name(payload: bytes) -> str:
    n, i = uleb(payload, 0)
    return payload[i:i + n].decode("utf-8", "replace")


def exports(payload: bytes) -> list[tuple[str, int, int]]:
    count, i = uleb(payload, 0)
    out = []
    for _ in range(count):
        n, i = uleb(payload, i)
        name = payload[i:i + n].decode("utf-8")
        i += n
        kind = payload[i]
        idx, i = uleb(payload, i + 1)
        out.append((name, kind, idx))
    return out


def enc_exports(entries: list[tuple[str, int, int]]) -> bytes:
    out = bytearray(enc_uleb(len(entries)))
    for name, kind, idx in entries:
        b = name.encode("utf-8")
        out += enc_uleb(len(b)) + b + bytes([kind]) + enc_uleb(idx)
    return bytes(out)


SOURCE_PATH = re.compile(rb"[A-Za-z0-9_./\\~@+-]*[A-Za-z0-9_]\.rs(?![A-Za-z0-9_])")


def scrub(data: bytes, names: list[str], key: bytes, exact: list[str] = ()) -> tuple[bytes, int]:
    """source paths, the given names (any case) and the `exact` identifiers (as written) overwritten
    in place with filler of the same length; (data, count)"""
    count = 0

    def filler(m: re.Match) -> bytes:
        nonlocal count
        count += 1
        return b"_" * (m.end() - m.start())

    data = SOURCE_PATH.sub(filler, data)
    for name in sorted(set(names), key=len, reverse=True):
        raw = name.encode("utf-8")
        pat = re.compile(re.escape(raw), re.I)
        rep = neutral(key, name).encode("ascii")
        rep = (rep * (len(raw) // len(rep) + 1))[: len(raw)]

        def same_len(m: re.Match, rep: bytes = rep) -> bytes:
            nonlocal count
            count += 1
            return rep

        data = pat.sub(same_len, data)
    for name in sorted(set(exact), key=len, reverse=True):
        raw = name.encode("utf-8")
        rep = neutral(key, name).encode("ascii")
        rep = (rep * (len(raw) // len(rep) + 1))[: len(raw)]
        n = data.count(raw)
        if n:
            count += n
            data = data.replace(raw, rep)
    return data, count


# ----------------------------------------------------------------------------------------
# names


def neutral(key: bytes, name: str) -> str:
    return "n" + hmac.new(key, name.encode("utf-8"), hashlib.sha256).hexdigest()[:10]


JS_BUILTINS = {
    "constructor", "length", "prototype", "free", "toString", "valueOf", "then", "default",
    "memory", "__wbg_ptr", "__destroy_into_raw", "__wrap", "__getRandomValues", "Symbol",
    # wasm-bindgen's own loader API: generic, not the engine's
    "initSync", "InitInput", "InitOutput", "SyncInitInput", "__wbg_init",
}


def api_names(dts: str, wasm_exports: list[str]) -> list[str]:
    """the identifiers the package exports, from its declarations and its wasm export section"""
    names = set(re.findall(r"^export class (\w+)", dts, re.M))
    names |= set(re.findall(r"^export function (\w+)", dts, re.M))
    for body in re.findall(r"^export class \w+ \{(.*?)^\}", dts, re.M | re.S):
        names |= set(re.findall(r"^\s+(?:static\s+)?(\w+)\(", body, re.M))
    names |= {e for e in wasm_exports if not e.startswith("__wbindgen") and e != "memory"}
    return sorted(n for n in names if n not in JS_BUILTINS)


BLOCK_COMMENTS = re.compile(r"/\*.*?\*/", re.S)
LINE_COMMENTS = re.compile(r"^[ \t]*//[^\n]*$", re.M)


def rename_text(text: str, mapping: dict[str, str]) -> str:
    text = LINE_COMMENTS.sub("", BLOCK_COMMENTS.sub("", text))
    if not mapping:
        return text
    pat = re.compile(r"(?<![\w$])(" + "|".join(re.escape(k) for k in sorted(mapping, key=len, reverse=True)) + r")(?![\w$])")
    return pat.sub(lambda m: mapping[m.group(1)], text)


def keychain_key(service: str, account: str) -> bytes:
    try:
        r = subprocess.run(["security", "find-generic-password", "-s", service, "-a", account, "-w"],
                           capture_output=True, text=True)
    except FileNotFoundError:
        raise InputError("no Keychain on this machine (the `security` command is missing)") from None
    key = r.stdout.strip()
    if r.returncode != 0 or len(key) < 32:
        raise InputError(f"no key in the Keychain item (service {service}): create it first")
    return key.encode("utf-8")


def token_hashes(names: list[str], extra: list[str]) -> list[str]:
    """SHA-256 of the lower-cased tokens the leak check refuses"""
    tokens = {t.lower() for t in names + extra if t}
    return sorted(hashlib.sha256(t.encode("utf-8")).hexdigest() for t in tokens)


DISTINCTIVE = re.compile(r"^(?=.{6,}$)(?:[A-Za-z0-9]*[a-z0-9][A-Z][A-Za-z0-9]*|[a-z0-9]+(?:_[a-z0-9]+)+)$")


def distinctive(name: str) -> bool:
    """an identifier that names something rather than being an ordinary word: 6 characters or
    more, and camelCase / PascalCase with an inner capital, or snake_case with an underscore"""
    return bool(DISTINCTIVE.match(name))


def snapshot_identifiers(text: str, crate: str) -> set[str]:
    """the path segments of the crate's own items in a public-API listing (`cargo public-api`)"""
    out = set()
    for path in re.findall(re.escape(crate) + r"((?:::[A-Za-z_][A-Za-z0-9_]*)+)", text):
        out |= {seg for seg in path.split("::") if seg}
    return out


def identifier_hashes(names: set[str]) -> list[str]:
    """SHA-256 of the distinctive identifiers as written (the leak check matches them by exact case)"""
    return sorted(hashlib.sha256(n.encode("utf-8")).hexdigest() for n in names if distinctive(n))


# ----------------------------------------------------------------------------------------


def process(src: Path, out: Path, key: bytes, names: list[str], extra_tokens: list[str],
            identifiers: set[str] = frozenset()) -> dict:
    files = {k: src / f for k, f in (("js", "p.js"), ("wasm", "p_bg.wasm"), ("dts", "p.d.ts"))}
    for f in files.values():
        if not f.is_file():
            raise InputError(f"{f} is missing")
    wasm = files["wasm"].read_bytes()
    js = files["js"].read_text(encoding="utf-8")
    dts = files["dts"].read_text(encoding="utf-8")

    secs = sections(wasm)
    exp = next((p for sid, p in secs if sid == 7), None)
    if exp is None:
        raise InputError("the wasm has no export section")
    entries = exports(exp)
    api = api_names(dts, [e[0] for e in entries])
    if not api:
        raise InputError("no exported identifier found (nothing to rename)")
    mapping = {n: neutral(key, n) for n in api}

    new_secs, scrubbed, data_names = [], 0, {}
    for sid, p in secs:
        if sid == 0 and custom_name(p) in ("name", "producers"):
            continue
        if sid == 7:
            p = enc_exports([(mapping.get(n, n), k, i) for n, k, i in entries])
        if sid == 11:
            # only source paths and the given names: API names also occur inside strings the code
            # uses (a solver tag hashed into a law identifier, an error kind returned to the caller),
            # and changing those would change behaviour
            # and the engine's own identifiers that appear as text (error kinds, field labels in
            # error messages): exact case only, never an ordinary word (`distinctive`); the caller
            # checks they are not hash inputs (see the build script)
            in_data = sorted(i for i in identifiers if distinctive(i) and i.encode("utf-8") in p)
            p, scrubbed = scrub(p, names, key, in_data)
            data_names = {i: neutral(key, i) for i in in_data}
        new_secs.append((sid, p))
    wasm2 = build(new_secs)
    h12 = hashlib.sha256(wasm2).hexdigest()[:12]
    base = f"p-{h12}"

    # the engine's identifiers the glue carries besides its API (parameter names come from the
    # Rust signatures): renamed in the text only, where they are local names
    words = set(re.findall(r"[A-Za-z0-9_$]+", BLOCK_COMMENTS.sub("", js + dts)))
    params = {i: neutral(key, i) for i in identifiers if distinctive(i) and i in words and i not in mapping}
    js2 = rename_text(js, {**mapping, **params}).replace("p_bg.wasm", f"{base}.wasm")
    dts2 = rename_text(dts, {**mapping, **params})

    out.mkdir(parents=True, exist_ok=True)
    written = {f"{base}.wasm": wasm2, f"{base}.js": js2.encode("utf-8"), f"{base}.d.ts": dts2.encode("utf-8")}
    for name, data in written.items():
        (out / name).write_bytes(data)
    manifest = {
        "sha256": hashlib.sha256(wasm2).hexdigest(),
        "size": len(wasm2),
        "files": [{"name": n, "sha256": hashlib.sha256(d).hexdigest(), "size": len(d)} for n, d in sorted(written.items())],
        "api": sorted(mapping.values()),
    }
    (out / f"{base}.manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return {"base": base, "mapping": mapping, "data_names": data_names, "scrubbed": scrubbed,
            "tokens": {"insensitive": token_hashes(names, extra_tokens),
                       "sensitive": identifier_hashes(set(api) | set(identifiers))}}


def write_private(path: Path, mapping: dict[str, str]) -> None:
    """the name map, readable by its owner only (mode 600, also when the file existed)"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=1, sort_keys=True)
    os.chmod(path, 0o600)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--name", action="append", default=[])
    ap.add_argument("--token", action="append", default=[], help="an extra token the leak check refuses")
    ap.add_argument("--identifiers", help="a public-API listing whose crate items the package must not name")
    ap.add_argument("--crate", default="alice_physics_core", help="the crate path prefix in --identifiers")
    ap.add_argument("--key-service", default="p-artifact-key")
    ap.add_argument("--key-account", default="k1")
    a = ap.parse_args(argv)
    try:
        key = keychain_key(a.key_service, a.key_account)
        ids = snapshot_identifiers(Path(a.identifiers).read_text(encoding="utf-8"), a.crate) if a.identifiers else set()
        r = process(Path(a.src), Path(a.out), key, a.name, a.token, ids)
    except (InputError, OSError) as e:
        print(f"neutral_pkg: {e}", file=sys.stderr)
        return 2
    write_private(Path(a.map), {"api": r["mapping"], "text": r["data_names"]})
    Path(a.tokens).write_text(json.dumps(r["tokens"], indent=0) + "\n", encoding="utf-8")
    print(f"neutral package {r['base']}: {len(r['mapping'])} names renamed, {r['scrubbed']} data strings overwritten, "
          f"{len(r['tokens']['insensitive'])} token hashes, {len(r['tokens']['sensitive'])} identifier hashes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
