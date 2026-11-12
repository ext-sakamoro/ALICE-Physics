#!/usr/bin/env python3
"""Every contact pairing is checked against every contact law.

The pairings are what a dynamic body can touch without moving it: each variant of
`pub enum StaticCollider` (read from src/static_collider.rs, so a new variant is a new
row), a static body with a shape (the body-body contact) and a static SDF collider.
The laws are restitution (bounce height), friction (sliding stop), rolling (rolling
without slip) and torque (a tipping box).

tests/analytic_static_contact_response.rs lists the pairings in `fn floors()`, one
`("<name>", <constructor>)` entry each, and each law is a `#[test]` whose body loops
over `floors()`. The check fails when

  - a `StaticCollider` variant has no `StaticCollider::<Variant>(` in `floors()`
    (a plane may be built through `plane()`, which must itself build
    `StaticCollider::Plane(`),
  - the static body (`Floor::Body`) or the SDF collider (`Floor::Sdf`) is not in
    `floors()`,
  - a law has no test, the test is ignored for another reason than a recorded
    `known defect:` (kept red on purpose until the law is fixed, listed in
    docs/oracle-status.md), or does not loop over `floors()`,
  - nothing was compared (no variant read, no test found).

  python3 scripts/contact_law_matrix.py [--root DIR]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ENUM = "src/static_collider.rs"
TESTS = "tests/analytic_static_contact_response.rs"
LAWS = {
    "restitution": "a_ball_bounces_to_e_squared_h_on_every_floor_and_backend",
    "friction": "a_sliding_box_stops_after_v0_squared_over_two_mu_g",
    "rolling": "a_sliding_ball_rolls_at_five_sevenths_of_v0",
    "torque": "a_domino_falls_over_only_past_atan_w_over_h",
}
EXTRA = {"static body": "Floor::Body", "sdf collider": "Floor::Sdf"}


def variants(text: str) -> list[str]:
    m = re.search(r"pub enum StaticCollider\s*\{(.*?)\n\}", text, re.S)
    if not m:
        return []
    body = re.sub(r"//[^\n]*", "", m.group(1))
    return re.findall(r"^\s*([A-Z][A-Za-z0-9]*)\s*\(", body, re.M)


def fn_body(text: str, name: str) -> str | None:
    m = re.search(rf"\bfn {re.escape(name)}\s*\(", text)
    if not m:
        return None
    start = text.index("{", m.end())
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def check(root: Path) -> tuple[list[str], int]:
    problems: list[str] = []
    compared = 0
    found = variants((root / ENUM).read_text(encoding="utf-8"))
    if not found:
        return [f"{ENUM}: no variant of StaticCollider read"], 0
    tests = (root / TESTS).read_text(encoding="utf-8")
    floors = fn_body(tests, "floors")
    if floors is None:
        return [f"{TESTS}: no fn floors()"], 0
    plane = fn_body(tests, "plane") or ""
    for v in found:
        compared += 1
        built = f"StaticCollider::{v}(" in floors or (
            "plane()" in floors and f"StaticCollider::{v}(" in plane and v == "Plane"
        )
        if not built:
            problems.append(f"StaticCollider::{v} is not a floor in {TESTS} fn floors()")
    for name, ctor in EXTRA.items():
        compared += 1
        if ctor not in floors:
            problems.append(f"the {name} pairing ({ctor}) is not a floor in {TESTS} fn floors()")
    for law, test in LAWS.items():
        compared += 1
        m = re.search(rf"((?:#\[[^\]]*\]\s*)*)fn {test}\s*\(", tests)
        if not m:
            problems.append(f"law {law}: no test {test} in {TESTS}")
            continue
        attrs = m.group(1)
        if "#[test]" not in attrs:
            problems.append(f"law {law}: {test} is not a #[test]")
        elif "#[ignore" in attrs and '#[ignore = "known defect:' not in attrs:
            problems.append(f"law {law}: {test} is ignored without a `known defect:` reason")
        body = fn_body(tests, test) or ""
        if "floors()" not in body:
            problems.append(f"law {law}: {test} does not loop over floors()")
    return problems, compared


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    a = ap.parse_args()
    problems, compared = check(Path(a.root))
    for p in problems:
        print(f"contact-law-matrix: {p}", file=sys.stderr)
    print(f"contact-law-matrix: compared {compared} (variants, extra pairings, laws), {len(problems)} problem(s)")
    if compared == 0:
        print("contact-law-matrix: compared nothing", file=sys.stderr)
        return 1
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
