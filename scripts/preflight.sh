#!/usr/bin/env bash
# Local reproduction of the CI gates before `git push` (ci.yml + the
# public-api job of security-audit.yml). Every command is the one CI runs;
# a step this script does not cover is a step that can only fail remotely.
#
# 2026-09-15: three pushes in a row were red on steps that were never run
# locally (workflow YAML parse, wasm32 dev-dep build, no_std `vec!`); this
# file is the checklist so that cannot repeat.
#
# usage: scripts/preflight.sh [--quick]   (--quick skips the slow test suites)
set -euo pipefail
cd "$(dirname "$0")/.."

quick=0
[[ "${1:-}" == "--quick" ]] && quick=1
NATIVE='std,simd,parallel,ffi,gpu-solver-bridge'

step() { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }

step "actionlint (workflow YAML)"
actionlint .github/workflows/*.yml

step "cargo fmt --check"
cargo fmt -- --check

step "README f32 module row = src/"
python3 scripts/f32_modules.py --check

step "clippy -D warnings (default, all targets)"
cargo clippy --all-targets -- -D warnings

step "clippy -D warnings (native feature set, all targets)"
cargo clippy --all-targets --features "$NATIVE" -- -D warnings

step "no_std rlib build (cdylib crate-type needs std, so rustc --crate-type rlib)"
cargo rustc --lib --no-default-features --crate-type rlib

step "wasm32-wasip1 golden tests build (dev-deps are built for the target too)"
rustup target list --installed | grep -q wasm32-wasip1 || rustup target add wasm32-wasip1
cargo test --test determinism_golden --target wasm32-wasip1 --no-run
cargo test --test determinism_golden_f32 --target wasm32-wasip1 --no-run

step "rustdoc -D warnings (default + docs.rs feature set)"
RUSTDOCFLAGS="-Dwarnings" cargo doc --lib --no-deps
RUSTDOCFLAGS="-Dwarnings" cargo doc --lib --no-deps --features "$NATIVE"

step "public API snapshot (needs nightly + cargo-public-api)"
if cargo +nightly public-api --version >/dev/null 2>&1; then
  tmp=$(mktemp)
  cargo +nightly public-api --features "$NATIVE" --simplified > "$tmp" 2>/dev/null
  if ! diff -u docs/PUBLIC_API_SNAPSHOT.txt "$tmp"; then
    echo "public API drift: regenerate docs/PUBLIC_API_SNAPSHOT.txt (diff above)" >&2
    exit 1
  fi
  rm -f "$tmp"
else
  echo "skip: cargo +nightly public-api not installed" >&2
fi

if [[ $quick -eq 1 ]]; then
  echo; echo "preflight --quick OK (test suites skipped)"; exit 0
fi

step "cargo test (default features, all targets)"
cargo test --no-fail-fast

step "cargo test --lib (native feature set)"
cargo test --lib --features "$NATIVE"

step "cargo test --lib (ffi module)"
cargo test --lib --features "ffi" "ffi::"

step "cargo test --lib (neural / replay / analytics via crates.io siblings)"
cargo test --lib --features "neural,replay,analytics"

echo; echo "preflight OK"
