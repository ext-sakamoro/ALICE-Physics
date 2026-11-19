#!/usr/bin/env python3
"""Generate tests/fixtures/fix128_reference.txt, the reference table for the
Fix128 transcendental functions (tests/analytic_fix128_transcendentals.rs).

Every input is an exact Fix128 value (a raw 128-bit integer over 2^64), so the
table measures the functions alone. The expected value is the exact result,
computed with 400-bit arithmetic (mpmath), rounded to the nearest raw value
(sqrt and cbrt: floored, as those functions promise the floor); a result at or
beyond 2^63 is stored as the saturated maximum, one below 2^-65 as 0.

Line format (decimal raw i128 values, space separated):

    <fn> <input raw>... = <expected raw>...

fn is one of sqrt, sincos (two outputs), atan, atan2 (inputs y x), exp, ln,
powf (inputs base exponent), cbrt. The inputs come from a fixed seed, so the
file is reproducible: `python3 scripts/gen_fix128_reference.py` rewrites it
byte for byte (needs mpmath).
"""
import math
import pathlib
import random

import mpmath as M

M.mp.prec = 400
S = 1 << 64
MAXRAW = (1 << 127) - 1
OUT = pathlib.Path(__file__).resolve().parent.parent / "tests/fixtures/fix128_reference.txt"


def raw(v):
    return int(round(v * S))


def val(n):
    return M.mpf(n) / S


def nearest(e):
    r = e * S
    if r >= MAXRAW:
        return MAXRAW
    return int(M.floor(r + M.mpf(1) / 2))


def floor_raw(e):
    return int(M.floor(e * S))


def logspace(a, b, k):
    return [10 ** (a + (b - a) * i / (k - 1)) for i in range(k)]


def main():
    random.seed(20261119)
    rows = []

    def add(fn, ins, outs):
        rows.append(f"{fn} {' '.join(map(str, ins))} = {' '.join(map(str, outs))}")

    for v in logspace(-19, 18, 120):
        n = raw(v)
        add("sqrt", [n], [floor_raw(M.sqrt(val(n)))])
    for n in [1, 2, 3, S, 2 * S, S - 1, 1 << 126]:
        add("sqrt", [n], [floor_raw(M.sqrt(val(n)))])

    def sincos(n):
        x = val(n)
        add("sincos", [n], [nearest(M.sin(x)), nearest(M.cos(x))])

    for i in range(240):
        sincos(raw(-math.pi + 2 * math.pi * i / 239))
    for v in logspace(-18, -1, 40):
        sincos(raw(v))
        sincos(raw(-v))
    for v in [0, 10, 100, 1000, 1e6, 1e9, 4e18, -9.2e18]:
        sincos(raw(v))
    for n in [MAXRAW, -MAXRAW]:
        sincos(n)

    for v in logspace(-18, 18, 100):
        for n in (raw(v), raw(-v)):
            add("atan", [n], [nearest(M.atan(val(n)))])
    add("atan", [0], [0])
    for _ in range(120):
        a = random.uniform(-math.pi, math.pi)
        r = 10 ** random.uniform(-9, 9)
        y, x = raw(r * math.sin(a)), raw(r * math.cos(a))
        if x == 0:
            continue
        add("atan2", [y, x], [nearest(M.atan2(val(y), val(x)))])
    for y, x in [(0, -S), (S, -S), (-S, -S), (1, S), (-1, -S)]:
        add("atan2", [y, x], [nearest(M.atan2(val(y), val(x)))])

    def exp_row(n):
        x = val(n)
        e = M.exp(x)
        add("exp", [n], [nearest(e)])

    for i in range(200):
        exp_row(raw(-45 + 88.6 * i / 199))
    for v in logspace(-18, 0, 50):
        exp_row(raw(v))
        exp_row(raw(-v))
    for v in [0, 43.66, 43.67, 50, -44.3, -44.4, -46]:
        exp_row(raw(v))

    for v in logspace(-19, 18, 150):
        n = raw(v)
        add("ln", [n], [nearest(M.log(val(n)))])
    for v in [1 + 1e-15, 1 - 1e-15, 2, 0.5, 1]:
        n = raw(v)
        add("ln", [n], [nearest(M.log(val(n)))])

    def powf(b, e):
        x, y = val(b), val(e)
        add("powf", [b, e], [nearest(x ** y)])

    for _ in range(200):
        powf(raw(10 ** random.uniform(-3, 3)), raw(random.uniform(0, 8)))
    for _ in range(80):
        powf(raw(10 ** random.uniform(-6, 6)), raw(random.uniform(0, 3)))
    for b, e in [(1.0001, 1000), (0.9999, 1000), (1.5, 40), (1.01, 100), (0.5, 60),
                 (3, 39), (2, 62), (2, 63), (1.1, 400), (1000, 7), (614, 6.6), (614, 6.9)]:
        powf(raw(b), raw(e) if e != int(e) else int(e) * S)
    for b in [1.5, 2.7, 10, 0.3, 1.0001, 600]:
        for n in [1, 2, 5, 10]:
            for d in (-16, 0, 16):  # y = n - 2^-60, n, n + 2^-60
                powf(raw(b), n * S + d)

    for v in logspace(-19, 18, 140):
        n = raw(v)
        add("cbrt", [n], [floor_raw(M.cbrt(val(n)))])

    # exp's zero boundary: the largest raw x with e^x >= 2^-65 (half an ulp)
    # rounds to 1 ulp, the next raw value below rounds to 0
    edge = int(M.floor(-65 * M.log(2) * S))
    # e^x increases with x: step up while edge is below half an ulp, then down
    # while the value below edge is still at or above it
    while M.exp(val(edge)) < M.mpf(2) ** -65:
        edge += 1
    while M.exp(val(edge - 1)) >= M.mpf(2) ** -65:
        edge -= 1
    header = [
        "# Fix128 transcendental reference table (scripts/gen_fix128_reference.py)",
        f"# mpmath {M.__version__}, precision {M.mp.prec} bits, seed 20261119",
        f"# rows: {len(rows)}",
        f"# exp-zero-boundary-raw: {edge}",
    ]
    OUT.write_text("\n".join(header + rows) + "\n")
    print(f"{OUT.name}: {len(rows)} rows")


if __name__ == "__main__":
    main()
