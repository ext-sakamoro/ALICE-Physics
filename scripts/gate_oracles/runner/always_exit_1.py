#!/usr/bin/env python3
"""Stub gate for scripts/test_gate_oracle_check.py's self-test: always exits
1, whatever its input. Used to confirm scripts/gate_oracle_check.py reports
a control failure when a gate does not actually turn green on its
must-green control (see docs/GATE_ORACLES.md section 4)."""
import sys

print("always-exit-1: compared nothing, as designed", file=sys.stderr)
sys.exit(1)
