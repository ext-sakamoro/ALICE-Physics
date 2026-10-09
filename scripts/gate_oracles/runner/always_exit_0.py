#!/usr/bin/env python3
"""Stub gate for scripts/test_gate_oracle_check.py's self-test: always exits
0, whatever its input. Used to confirm scripts/gate_oracle_check.py reports
a control failure when a gate does not actually turn red on its must-red
control (see docs/GATE_ORACLES.md section 4)."""
import sys

print("always-exit-0: compared nothing, as designed")
sys.exit(0)
