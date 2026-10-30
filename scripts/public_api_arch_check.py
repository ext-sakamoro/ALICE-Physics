#!/usr/bin/env python3
"""Both public API snapshots exist, and each has the items only its architecture compiles.

The public API differs by architecture: items under `#[cfg(... target_arch = "x86_64" ...)]` (the
x86_64 SIMD methods of math.rs) exist on x86_64 only. scripts/ledgers.py names one snapshot per
target (PUBLIC_API_SNAPSHOTS); a snapshot regenerated for the wrong target, or copied from the other
one, would still diff clean against itself. So the `pub` items of src/ whose cfg attributes require an
architecture (a `target_arch = "<arch>"` outside any `not(...)`) are read from the source, and each must
appear in the snapshot of that architecture; at least one of the x86_64 ones must be absent from the
aarch64 snapshot (the two files differ where the source says they do).

Fails (exit 1) on a missing or empty snapshot, an architecture-only item missing from its snapshot, two
snapshots that do not differ on those items, and when no x86_64-only item was read from src/ (compared
nothing).

  python3 scripts/public_api_arch_check.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ledgers import PUBLIC_API_SNAPSHOTS  # noqa: E402

ROOT = HERE.parent
ITEM = re.compile(r"^\s*pub\s+(?:const\s+|unsafe\s+|extern\s+\"C\"\s+)*(fn|struct|enum|trait|const|static|type|mod)\s+"
                  r"([A-Za-z_][A-Za-z0-9_]*)")


def without_not(text: str) -> str:
    """the text with every `not( ... )` group (balanced parentheses) removed"""
    out, i = [], 0
    while i < len(text):
        if text.startswith("not(", i) and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")):
            depth, i = 1, i + 4
            while i < len(text) and depth:
                depth += {"(": 1, ")": -1}.get(text[i], 0)
                i += 1
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def arch_items(src: Path) -> list[tuple[str, str, str, str]]:
    """(architecture, item kind, name, file:line) of each `pub` item of src/ whose attributes require one
    architecture"""
    out = []
    for f in sorted(src.rglob("*.rs")):
        lines = f.read_text(encoding="utf-8").splitlines()
        attrs: list[str] = []
        for n, line in enumerate(lines, 1):
            s = line.strip()
            if s.startswith("#[") or (attrs and not attrs[-1].rstrip().endswith("]")):
                attrs.append(s)
                continue
            if s.startswith("///") or s.startswith("//") or not s:
                if not s.startswith("///"):
                    attrs = []
                continue
            m = ITEM.match(line)
            if m:
                cfg = without_not(" ".join(a for a in attrs if a.startswith("#[cfg(") or not a.startswith("#[")))
                archs = set(re.findall(r'target_arch\s*=\s*"([a-z0-9_]+)"', cfg))
                if len(archs) == 1:
                    out.append((archs.pop(), m.group(1), m.group(2), f"{f.relative_to(src.parent)}:{n}"))
            attrs = []
    return out


def appears(kind: str, name: str, text: str) -> bool:
    if kind == "fn":
        return re.search(rf"::{re.escape(name)}\(", text) is not None
    return re.search(rf"\b{kind} [\w:]*::{re.escape(name)}\b", text) is not None


def check(root: Path = ROOT) -> tuple[list[str], dict[str, int]]:
    errors: list[str] = []
    texts: dict[str, str] = {}
    for target, path in PUBLIC_API_SNAPSHOTS:
        arch = target.split("-")[0]
        p = root / path
        if not p.is_file() or not p.read_text(encoding="utf-8").strip():
            errors.append(f"{path}: the {arch} public API snapshot is missing or empty "
                          "(python3 scripts/ledgers.py --write)")
            continue
        texts[arch] = p.read_text(encoding="utf-8")
    items = arch_items(root / "src")
    counts = {a: 0 for a in texts}
    for arch, kind, name, where in items:
        if arch not in texts:
            continue
        counts[arch] += 1
        if not appears(kind, name, texts[arch]):
            path = dict((t.split("-")[0], p) for t, p in PUBLIC_API_SNAPSHOTS)[arch]
            errors.append(f"{where}: `pub {kind} {name}` is {arch}-only and not in {path} (regenerated for "
                          "another target?)")
    x86 = [(k, n) for a, k, n, _ in items if a == "x86_64"]
    if not x86:
        errors.append("read no x86_64-only pub item from src/ (compared nothing)")
    elif "aarch64" in texts and all(appears(k, n, texts["aarch64"]) for k, n in x86):
        errors.append("every x86_64-only item is also in the aarch64 snapshot: the two files do not differ where "
                      "the source says they do (one copied from the other?)")
    return errors, counts


def main() -> int:
    errors, counts = check()
    for e in errors:
        print(f"error: {e}", file=sys.stderr)
    print("public API snapshots: architecture-only items compared " +
          ", ".join(f"{a} {n}" for a, n in sorted(counts.items())))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
