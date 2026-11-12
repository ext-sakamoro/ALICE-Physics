#!/usr/bin/env python3
"""A constraint's bodies are read only through the accessor.

A contact against something that does not move (a static collider, an SDF collider)
carries `body_b = STATIC_BODY` (`usize::MAX`), which is no index of the world's
bodies. Every read or write of a constraint's bodies must go through
`PhysicsWorld::contact_body` / `contact_set_velocity` / `contact_set_position` (or
`contact_body_in` over a slice), which stand a static body in for the sentinel and
refuse to write into it. A direct `bodies[<x>.body_a]`, `bodies.get(<x>.body_b)` or
`bodies.get_mut(..)` would index past the end (a panic) or, with `.get`, silently skip
the contact.

The check reads src/solver.rs, src/solver/*.rs and src/solver_tgs_backend.rs (the
world-level code; the TGS hooks index the TGS engine's own body arrays) and fails when

  - a direct access by `.body_a` / `.body_b` is found (0 allowed),
  - the accessors are used nowhere (compared nothing).

  python3 scripts/contact_body_access.py [--root DIR]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DIRECT = re.compile(r"\bbodies\s*(?:\[|\.get\(|\.get_mut\()\s*[^\]\)]*?\.body_[ab]\b")
ACCESSOR = re.compile(r"\b(?:contact_body|contact_body_in|contact_set_velocity|contact_set_position)\s*\(")


def files(root: Path) -> list[Path]:
    out = [root / "src/solver.rs", root / "src/solver_tgs_backend.rs"]
    out += sorted((root / "src/solver").glob("*.rs"))
    return [p for p in out if p.is_file()]


def check(root: Path) -> tuple[list[str], int]:
    problems: list[str] = []
    uses = 0
    for p in files(root):
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("//", 1)[0]
            if DIRECT.search(code):
                problems.append(f"{p.relative_to(root)}:{n}: direct body access by a constraint index: {line.strip()}")
            uses += len(ACCESSOR.findall(code))
    return problems, uses


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    a = ap.parse_args()
    problems, uses = check(Path(a.root))
    for p in problems:
        print(f"contact-body-access: {p}", file=sys.stderr)
    print(f"contact-body-access: {uses} accessor use(s), {len(problems)} direct access(es)")
    if uses == 0:
        print("contact-body-access: no accessor use found (compared nothing)", file=sys.stderr)
        return 1
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
