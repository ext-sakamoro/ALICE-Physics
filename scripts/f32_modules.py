#!/usr/bin/env python3
"""List the modules whose public API carries `f32` / `f64` values.

README § Determinism scope has a row "`f32` / `f64` field modules (N: …)"
that must match the source. This script is the source of truth for that
row: it scans every `src/*.rs` (test modules stripped), keeps the modules
where a `pub` item signature or a struct field mentions `f32` / `f64`
(conversion helpers such as `to_f64` / `from_f64` are ignored), and splits
them into

  * arithmetic modules — the value is computed with float arithmetic and
    must go through `det_math` for cross-platform bit-exactness (this is
    the README list, pinned by `tests/determinism_golden_f32.rs`), and
  * boundary modules — the float only crosses an I/O / binding boundary
    (FFI, Python, replay, analytics, metrics ratios) and no simulation
    arithmetic happens in `f32`; listed in BOUNDARY below with the reason.

Usage: `python3 scripts/f32_modules.py` prints the README row fragment and
exits 1 when README.md disagrees with the source (`--check`), so
`alice-strict-eval` / CI can gate it. A new module that uses floats lands
in the arithmetic group by default; add it to BOUNDARY only with a reason.
"""

from __future__ import annotations

import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Modules where f32 / f64 appears in the public API without float
# arithmetic on simulation state (reason per entry).
BOUNDARY = {
    "ffi": "C ABI takes/returns f32 for the host, converts to Fix128",
    "pipeline": "print pipeline I/O (mm / °C values passed through)",
    "analytics_bridge": "alice-analytics sketches take f64 samples (I/O)",
    "db_bridge": "alice-db pass-through I/O, no arithmetic",
    "replay": "alice-db positions read back as f32 tuples (I/O)",
    "fluid_netcode": "compression_ratio metric only",
    "character_state": "state-machine timers, no simulation arithmetic",
    "solver_tgs": "pub(crate) cache hit-rate metric only",
}

# Files that are not simulation modules.
SKIP = {"lib", "det_math", "math", "math_util"}

CONVERSION = re.compile(r"fn (to|from|as)_f(32|64)\b|\bto_f64\(\)|\bto_f32\(\)")
PUB_ITEM = re.compile(
    r"^\s*pub(\s*\([a-z]+\))?\s+(fn|struct|enum|const|static|type|trait)\b"
    r"|^\s*pub(\s*\([a-z]+\))?\s+\w+\s*:"  # pub field
)
FLOAT = re.compile(r"\bf(32|64)\b")


def strip_tests(src: str) -> str:
    i = src.find("#[cfg(test)]")
    return src[:i] if i > 0 else src


def scan() -> tuple[list[str], list[str]]:
    arithmetic: list[str] = []
    boundary: list[str] = []
    for path in sorted(glob.glob(os.path.join(ROOT, "src", "*.rs"))):
        module = os.path.basename(path)[:-3]
        if module in SKIP:
            continue
        body = strip_tests(open(path, encoding="utf-8").read())
        hit = False
        for line in body.split("\n"):
            if line.lstrip().startswith("//"):
                continue
            if PUB_ITEM.search(line) and FLOAT.search(line) and not CONVERSION.search(line):
                hit = True
                break
        if not hit:
            continue
        (boundary if module in BOUNDARY else arithmetic).append(module)
    return arithmetic, boundary


def readme_count(readme: str) -> int | None:
    m = re.search(r"\*\*`f32` / `f64` field modules\*\* \((\d+):", readme)
    return int(m.group(1)) if m else None


def main() -> int:
    arithmetic, boundary = scan()
    print(f"arithmetic ({len(arithmetic)}): " + ", ".join(f"`{m}`" for m in arithmetic))
    print(f"boundary ({len(boundary)}): " + ", ".join(f"`{m}`" for m in boundary))
    stale = set(BOUNDARY) - set(boundary)
    if stale:
        print(f"BOUNDARY entries with no float in their public API any more: {sorted(stale)}")
    if "--check" in sys.argv:
        readme = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
        want = readme_count(readme)
        if want != len(arithmetic):
            print(
                f"README.md says {want} f32/f64 field modules, source has {len(arithmetic)}; "
                "regenerate the row from the list above",
                file=sys.stderr,
            )
            return 1
        missing = [m for m in arithmetic if f"`{m}`" not in readme]
        if missing:
            print(f"README.md does not mention: {missing}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
