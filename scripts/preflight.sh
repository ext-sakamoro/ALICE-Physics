#!/usr/bin/env bash
# Local reproduction of the CI gates before `git push` (ci.yml + the
# public-api job of security-audit.yml). Every command is the one CI runs;
# a step this script does not cover is a step that can only fail remotely.
#
# 2026-09-15: three pushes in a row were red on steps that were never run
# locally (workflow YAML parse, wasm32 dev-dep build, no_std `vec!`); this
# file is the checklist so that cannot repeat.
#
# usage: scripts/preflight.sh [--quick | --fast]
#   (none)   every static gate + the full test suites (what CI runs on 5 OS)
#            + every example built in release and executed (ci.yml job examples)
#   --fast   every static gate (incl. the SCIP / L0 ratchet) + only the tests that
#            reference the modules changed since origin/main + the determinism goldens
#            -- the local push gate; CI runs the full suites (2026-10-04: no full-suite
#            run ever found anything the static gates had not, and it costs tens of minutes)
#   --quick  every static gate, no test at all
set -euo pipefail
cd "$(dirname "$0")/.."

quick=0
fast=0
case "${1:-}" in
  --quick) quick=1 ;;
  --fast) fast=1 ;;
  "") ;;
  *) echo "usage: scripts/preflight.sh [--quick | --fast]" >&2; exit 2 ;;
esac
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

step "docs lint (public vocabulary / CHANGELOG structure)"
python3 scripts/test_docs_lint.py
python3 scripts/docs_lint.py --check

step "SCIP reach analysis oracle (docs/integration-status.md の解析器)"
python3 scripts/test_scip_reach.py
python3 scripts/test_audit_refs.py
python3 scripts/test_run_feature_gated_tests.py

step "integration levels oracle + C ABI coverage (docs/integration-levels.md)"
python3 scripts/test_integration_levels.py
python3 scripts/test_land.py
python3 scripts/integration_levels.py --no-index --check

step "oracle ledger links (PIN / root external)"
python3 scripts/gen-oracle-status.py --check

step "wiring-guard (oracle + 新規の未配線 / 理由の無い dead_code が無い)"
python3 scripts/test_wiring_guard.py
python3 scripts/wiring_guard.py

step "coverage tables (docs/coverage/*.toml = src/ LIMITATION comments + tests)"
python3 scripts/test_coverage_check.py
python3 scripts/test_line_coverage_ratchet.py
python3 scripts/test_mutants_ratchet.py
python3 scripts/test_mutants_in_diff_plan.py
python3 scripts/test_bench_counts_check.py
python3 scripts/test_ci_load_check.py
python3 scripts/ci_load_check.py
python3 scripts/test_downstream_select_tag.py
python3 scripts/test_downstream_test_counts.py
python3 scripts/test_coverage_refs_to_symbols.py
python3 scripts/coverage_check.py

step "status generators oracle (docs/wiring-status.md / docs/oracle-status.md の生成器)"
python3 scripts/test_gen_status.py

step "affected-test selector oracle (preflight --fast)"
python3 scripts/test_affected_tests.py

step "example runner oracle (scripts/run_examples.py, run in full mode and ci.yml job examples)"
python3 scripts/test_run_examples.py

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
# CI job `scip`: rust-analyzer index (about a minute), known-defect symbol
# resolution, and the L0 ratchet (scripts/integration-baseline.txt).
# Without rust-analyzer, --quick / --fast warn and skip; the full run fails.
step "SCIP checks (audit_refs --check, scip_reach --check-baseline)"
if command -v rust-analyzer >/dev/null && rust-analyzer --version >/dev/null 2>&1; then
  scripts/scip_index.sh
  python3 scripts/audit_refs.py --check
  python3 scripts/scip_reach.py --check-baseline
  python3 scripts/integration_levels.py --check
elif [[ $quick -eq 1 || $fast -eq 1 ]]; then
  echo "skip: rust-analyzer not installed (rustup component add rust-analyzer); CI runs this check" >&2
else
  echo "rust-analyzer not installed: rustup component add rust-analyzer" >&2
  exit 1
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
cargo test --test determinism_golden_contacts --target wasm32-wasip1 --no-run

step "wasm32-unknown-unknown build with every browser feature (CI job js-binding)"
rustup target list --installed | grep -q wasm32-unknown-unknown || rustup target add wasm32-unknown-unknown
cargo build --lib --target wasm32-unknown-unknown --no-default-features --features "std,wasm,neural,replay,analytics,gpu-solver-bridge,simd"

step "rustdoc -D warnings (default + docs.rs feature set)"
RUSTDOCFLAGS="-Dwarnings" cargo doc --lib --no-deps
RUSTDOCFLAGS="-Dwarnings" cargo doc --lib --no-deps --features "$NATIVE"

if [[ $quick -eq 1 ]]; then
  echo; echo "preflight --quick OK (test suites skipped)"; exit 0
fi

if [[ $fast -eq 1 ]]; then
  step "affected tests (modules changed since origin/main + determinism goldens)"
  python3 scripts/affected_tests.py --features "$NATIVE" --run
  echo; echo "preflight --fast OK (static gates + affected tests; the full suites run in CI)"; exit 0
fi

step "cargo test (default features, all targets)"
cargo test --no-fail-fast

step "cargo test --lib (native feature set)"
cargo test --lib --features "$NATIVE"

# same command as ci.yml: the semantics identifier with the gated stepping
# paths' features (`cargo test` above runs the file with default features)
step "semantics identifier (parallel + gpu-solver-bridge)"
cargo test --test physics_semantics_id --features "parallel,gpu-solver-bridge"

step "cargo test --lib (ffi module)"
cargo test --lib --features "ffi" "ffi::"

step "cargo test --lib (wasm module) + cargo check (python)"
cargo test --lib --features "wasm" "wasm::"
cargo check --lib --features "python"

step "WebAssembly binding from JavaScript (wasm-pack + node; CI job js-binding)"
command -v wasm-pack >/dev/null && command -v node >/dev/null \
  || { echo "wasm-pack and node are needed (cargo install wasm-pack; node 18+)" >&2; exit 1; }
bash scripts/run_js_binding_tests.sh

step "cargo test --lib (neural / replay / analytics via crates.io siblings)"
cargo test --lib --features "neural,replay,analytics"

step "feature-gated integration tests (same script and arguments as ci.yml)"
python3 scripts/run_feature_gated_tests.py

step "run every example in release (same script and arguments as ci.yml job examples)"
python3 scripts/run_examples.py

echo; echo "preflight OK"
