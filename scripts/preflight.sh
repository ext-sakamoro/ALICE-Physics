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

step "README sync (README.md / README_JP.md / docs/MODULES.md = Cargo.toml + src/)"
python3 scripts/test_readme_sync.py
python3 scripts/readme_sync.py --check

step "SCIP reach analysis oracle (docs/integration-status.md の解析器)"
python3 scripts/test_scip_reach.py
python3 scripts/test_audit_refs.py

step "oracle ledger links (PIN / root external)"
python3 scripts/gen-oracle-status.py --check

step "wiring-guard (oracle + 新規の未配線 / 理由の無い dead_code が無い)"
python3 scripts/test_wiring_guard.py
python3 scripts/wiring_guard.py

step "status generators oracle (docs/wiring-status.md / docs/oracle-status.md の生成器)"
python3 scripts/test_gen_status.py

# Fast static gates that failed late in practice (no_std build, public API drift)
step "no_std rlib build (cdylib crate-type needs std, so rustc --crate-type rlib)"
cargo rustc --lib --no-default-features --crate-type rlib

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

# Cheap, failure-prone gates run first so a red shows up in minutes, not after
# the whole test suite (2026-10-04: the L0 ratchet used to be the LAST step, so a
# new unreached pub item cost a full cargo test run before it was reported).
# --quick keeps its old coverage (it does not need rust-analyzer).
if [[ $quick -eq 0 ]]; then
  # CI job `scip`: rust-analyzer index (about a minute), known-defect symbol
  # resolution, and the L0 ratchet (scripts/integration-baseline.txt)
  step "SCIP checks (audit_refs --check, scip_reach --check-baseline)"
  if command -v rust-analyzer >/dev/null && rust-analyzer --version >/dev/null 2>&1; then
    scripts/scip_index.sh
    python3 scripts/audit_refs.py --check
    python3 scripts/scip_reach.py --check-baseline
  else
    echo "rust-analyzer not installed: rustup component add rust-analyzer" >&2
    exit 1
  fi
fi

step "clippy -D warnings (default, all targets)"
cargo clippy --all-targets -- -D warnings

step "clippy -D warnings (native feature set, all targets)"
cargo clippy --all-targets --features "$NATIVE" -- -D warnings

# CI's "Clippy (full native feature set, lib + all targets)" job adds
# neural,replay,analytics on top of $NATIVE. These sibling-bridge features
# pull in rkyv (via alice-db) which can add a competing trait impl (e.g.
# PartialEq<rkyv::ArchivedBTreeSet> alongside the stdlib's) that only
# surfaces as an ambiguity error at this exact feature combination — missed
# here before and only caught by CI (2026-10-03, program item 16,
# tests/analytic_multi_world_wiring.rs:339 `[0,2].into_iter().collect()`).
step "clippy -D warnings (full native feature set incl. neural/replay/analytics, all targets)"
cargo clippy --all-targets --features "$NATIVE,neural,replay,analytics" -- -D warnings

step "wasm32-wasip1 golden tests build (dev-deps are built for the target too)"
rustup target list --installed | grep -q wasm32-wasip1 || rustup target add wasm32-wasip1
cargo test --test determinism_golden --target wasm32-wasip1 --no-run
cargo test --test determinism_golden_f32 --target wasm32-wasip1 --no-run

step "rustdoc -D warnings (default + docs.rs feature set)"
RUSTDOCFLAGS="-Dwarnings" cargo doc --lib --no-deps
RUSTDOCFLAGS="-Dwarnings" cargo doc --lib --no-deps --features "$NATIVE"

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
