#!/usr/bin/env python3
"""Check that no refused token appears in a set of files (a copyable, standard-library-only checker).

    neutral_leak_check.py --tokens TOKENS.json FILE_OR_DIR [...]

TOKENS.json holds SHA-256 hex digests: either a JSON list (tokens matched without regard to
case, hashed lower-cased) or an object {"insensitive": [...], "sensitive": [...]}, where the
sensitive tokens (identifiers, hashed as written) match only with their exact case, so an
identifier like `lawId` is not found inside `LawIdMismatch`. The checker holds no token in plain
text, so it can be copied into any repository. Every file is read: a text file
whole, any other file (a .wasm) as its runs of 4 or more printable ASCII bytes, where every
substring of 4 to 48 characters is a candidate (compiled code packs strings with no separator
between them). In a text file each run of
letters, digits, `_`, `-`, `.`, `/`, `\\` and `$` is split into path components, and each
component gives these candidates, all lower-cased: the component, the component without a
leading dot, the component with `_` / `-` / `.` removed, its `_` / `-` / `.` parts, and its
camelCase parts (so `fooBarBaz`, `foo_bar`, `src/foo_bar.rs` and `.foo` all reach `foo`).
A candidate whose SHA-256 is in the set is a finding; the finding names the file and the
first 8 hex digits of the hash, never the token.

Exit 0 when nothing is found, 1 on a finding, 2 when nothing was compared (no file, an empty
token set, no candidate at all) or an input cannot be read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

RUN = re.compile(r"[A-Za-z0-9_\-./\\$]+")
PRINTABLE = re.compile(rb"[\x20-\x7e]{4,}")
CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")


def candidates(text: str):
    for run in RUN.findall(text):
        for comp in re.split(r"[/\\]+", run):
            if not comp:
                continue
            yield comp.lower()
            yield comp.lstrip(".").lower()
            yield re.sub(r"[_\-.]", "", comp).lower()
            for part in re.split(r"[_\-.$]+", comp):
                if part:
                    yield part.lower()
                    for c in CAMEL.findall(part):
                        yield c.lower()


SUBSTRING_LENGTHS = range(4, 49)


def substrings(run: str, lower: bool = True):
    """every substring of 4 to 48 characters (lower-cased unless `lower` is false): in compiled
    code strings are packed next to each other with no separator (`...mod.rsassertion failed...`),
    so a token can sit inside a longer run"""
    text = run.lower() if lower else run
    for n in SUBSTRING_LENGTHS:
        for i in range(len(text) - n + 1):
            yield text[i:i + n]


def exact_candidates(text: str):
    """identifier-shaped runs as written (for the case-sensitive tokens)"""
    yield from re.findall(r"[A-Za-z0-9_$]+", text)


def file_texts(path: Path):
    """(text, binary): a text file whole, any other file as its printable runs"""
    data = path.read_bytes()
    try:
        yield data.decode("utf-8"), False
    except UnicodeDecodeError:
        for m in PRINTABLE.findall(data):
            yield m.decode("ascii"), True


def check(paths: list[Path], hashes: set[str], sensitive: set[str] = frozenset()) -> tuple[list[str], int, int]:
    """(findings, files read, candidates compared); `hashes` match lower-cased, `sensitive` as written"""
    files = []
    for p in paths:
        files += sorted(f for f in p.rglob("*") if f.is_file()) if p.is_dir() else [p]
    findings, compared = [], 0
    for f in files:
        seen = set()
        for text, binary in file_texts(f):
            for c in (substrings(text) if binary else candidates(text)):
                compared += 1
                h = hashlib.sha256(c.encode("utf-8")).hexdigest()
                if h in hashes and h not in seen:
                    seen.add(h)
                    findings.append(f"{f}: refused token {h[:8]}")
            if sensitive:
                for c in (substrings(text, lower=False) if binary else exact_candidates(text)):
                    compared += 1
                    h = hashlib.sha256(c.encode("utf-8")).hexdigest()
                    if h in sensitive and h not in seen:
                        seen.add(h)
                        findings.append(f"{f}: refused identifier {h[:8]}")
    return findings, len(files), compared


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tokens", required=True)
    ap.add_argument("paths", nargs="+")
    a = ap.parse_args(argv)
    try:
        data = json.loads(Path(a.tokens).read_text(encoding="utf-8"))
        if isinstance(data, dict):
            hashes, sensitive = set(data.get("insensitive", [])), set(data.get("sensitive", []))
        else:
            hashes, sensitive = set(data), set()
        every = hashes | sensitive
        if not hashes or not all(isinstance(h, str) and re.fullmatch(r"[0-9a-f]{64}", h) for h in every):
            print("neutral_leak_check: the token set is empty or not SHA-256 hex digests", file=sys.stderr)
            return 2
        findings, nfiles, compared = check([Path(p) for p in a.paths], hashes, sensitive)
    except (OSError, ValueError) as e:
        print(f"neutral_leak_check: {e}", file=sys.stderr)
        return 2
    for f in findings:
        print(f)
    print(f"neutral leak check: {nfiles} files, {compared} candidates against {len(hashes)} token hashes "
          f"and {len(sensitive)} identifier hashes, "
          f"{len(findings)} findings")
    if nfiles == 0 or compared == 0:
        print("neutral_leak_check: compared nothing", file=sys.stderr)
        return 2
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
