#!/usr/bin/env python3
"""The browser WebAssembly build does not carry the text of the constants the engine keeps masked.

`src/law_id.rs` hashes `LAW_ID_DOMAIN` through a masked copy computed at compile time, so that
the constant's text (which names the engine) is not in a binary built for distribution. This check
reads each listed constant's byte string from its source file and fails when that text occurs in
any `.wasm` file under the given directory.

usage: python3 scripts/wasm_literal_check.py <dir with .wasm files>
exit 0 = no listed text found; 1 = found; 2 = nothing to check (no .wasm file, or a listed
constant is missing from its source)
"""
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# (source file, constant): byte-string constants that must not appear in the browser build
MASKED_CONSTANTS = [("src/law_id.rs", "LAW_ID_DOMAIN")]


def constant_bytes(source: str, name: str) -> bytes | None:
    m = re.search(rf'^pub const {re.escape(name)}: &\[u8\] = b"([^"\\]+)";$', source, re.M)
    return m.group(1).encode("ascii") if m else None


def check(wasm_dir: Path, repo: Path = REPO, constants=MASKED_CONSTANTS) -> tuple[int, list[str]]:
    lines = []
    texts = []
    for rel, name in constants:
        b = constant_bytes((repo / rel).read_text(encoding="utf-8"), name)
        if b is None:
            return 2, [f"{rel} has no byte-string constant {name}"]
        texts.append((name, b))
    if not texts:
        return 2, ["no constant to check"]
    files = sorted(wasm_dir.rglob("*.wasm"))
    if not files:
        return 2, [f"no .wasm file under {wasm_dir}"]
    hits = [f"{f}: holds the text of {name}" for f in files for name, b in texts if b in f.read_bytes()]
    lines += hits
    lines.append(f"wasm literal check: {len(files)} file(s), {len(texts)} constant(s), {len(hits)} found")
    return (1 if hits else 0), lines


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[-4], file=sys.stderr)
        return 2
    code, lines = check(Path(argv[1]))
    for line in lines:
        print(line, file=sys.stderr if code else sys.stdout)
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv))
