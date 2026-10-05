#!/usr/bin/env bash
# Build the WebAssembly binding with wasm-pack and run tests/js/ against it from
# Node, the path a JavaScript application takes. Fails when no test ran.
set -euo pipefail
cd "$(dirname "$0")/.."
target="${CARGO_TARGET_DIR:-target}"
mkdir -p "$target"
pkg="$(cd "$target" && pwd)/js-pkg"   # wasm-pack resolves a relative --out-dir from the crate
wasm-pack build --dev --target nodejs --out-dir "$pkg" -- --features wasm
out=$(ALICE_JS_PKG="$pkg" node --test tests/js/*.test.mjs 2>&1) || { echo "$out"; exit 1; }
echo "$out"
n=$(printf '%s\n' "$out" | sed -n 's/^ℹ tests \([0-9][0-9]*\)$/\1/p' | tail -1)
[ -n "$n" ] && [ "$n" -gt 0 ] || { echo "error: no JavaScript test ran (compared nothing)" >&2; exit 1; }
echo "js binding tests: $n"
