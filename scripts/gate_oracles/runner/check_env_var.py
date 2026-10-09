#!/usr/bin/env python3
"""Exit 0 iff environment variable argv[1] equals argv[2], else exit 1.

Used only by scripts/test_gate_oracle_check.py's own self-tests, as an
OS-independent stand-in for a shell `test "$VAR" = value`: that syntax
needs a POSIX shell (cmd.exe does not understand it), and gate_oracle_check.py
runs every gate's `run` directly, without a shell, on every OS.
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    name, want = sys.argv[1], sys.argv[2]
    sys.exit(0 if os.environ.get(name) == want else 1)
