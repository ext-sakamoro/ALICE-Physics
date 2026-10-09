#!/usr/bin/env python3
"""Exit 0 iff argv[1] (a literal substring, not a regex) is in the text
of the file at argv[2], else exit 1.

Used only by scripts/test_gate_oracle_check.py's own self-tests, as an
OS-independent stand-in for `grep -q PATTERN FILE`: grep is not
guaranteed to exist on PATH on every OS, and gate_oracle_check.py runs
every gate's `run` directly, without a shell, on every OS.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__":
    pattern, path = sys.argv[1], sys.argv[2]
    sys.exit(0 if pattern in Path(path).read_text(encoding="utf-8") else 1)
