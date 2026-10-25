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

# A gate skipped for a missing local tool must stay visible: printing once to
# stderr and moving on is easy to miss in a long run, so every skip is also
# collected here and repeated in a summary right before the final "OK" line.
declare -a skipped_gates=()
skip() {
  echo "SKIPPED: $*" >&2
  skipped_gates+=("$*")
}
print_skip_summary() {
  if [[ ${#skipped_gates[@]} -gt 0 ]]; then
    echo
    echo "Skipped (not verified locally -- CI runs these):"
    printf '  - %s\n' "${skipped_gates[@]}"
  fi
}

step "actionlint (workflow YAML)"
actionlint .github/workflows/*.yml

step "workflow-timeouts (every job has timeout-minutes; CI job actionlint)"
step "eol.yml / line endings (tests, then check)"
# text files are LF in the index and the working tree (.github/workflows/eol.yml, shared by the ALICE crates)
python3 scripts/test_check_eol.py
python3 scripts/check_eol.py
python3 scripts/test_workflow_timeouts.py
python3 scripts/workflow_timeouts.py

step "cargo fmt --check (every workspace member)"
cargo fmt --all -- --check

step "ci.yml / test: no negated fma result (G3, syntax tree)"
cargo fmt --manifest-path tools/no-outer-fneg-of-fma/Cargo.toml -- --check
cargo test --quiet --locked --manifest-path tools/no-outer-fneg-of-fma/Cargo.toml
cargo run --quiet --locked --release --manifest-path tools/no-outer-fneg-of-fma/Cargo.toml -- --root .

step "facade check (features both ways, glob-only lib, generated path test)"
python3 scripts/test_facade_check.py
python3 scripts/facade_check.py

# every generated file and generated table at once (each later check stops the run at
# the first stale one; this lists all of them before the slow steps)
step "generated files are current (scripts/ledgers.py --check, all at once)"
python3 scripts/test_ledgers.py
if [[ $quick -eq 1 || $fast -eq 1 ]]; then
  python3 scripts/ledgers.py --check --allow-missing-tools
else
  python3 scripts/ledgers.py --check
fi

step "README f32 module row = src/"
python3 scripts/f32_modules.py --check

step "README sync (README.md / README_JP.md / docs/MODULES.md = Cargo.toml + src/)"
python3 scripts/test_readme_sync.py
python3 scripts/readme_sync.py --check
python3 scripts/test_version_sync.py
python3 scripts/version_sync.py --check
python3 scripts/test_shared_files.py
scripts/shared_files.sh
python3 scripts/test_lib_test_gate.py
python3 scripts/test_preflight_ci_parity.py
python3 scripts/preflight_ci_parity.py
python3 scripts/test_fuzz_targets.py
python3 scripts/fuzz_targets.py --check
python3 scripts/test_semver_checked.py
python3 scripts/test_semver_baseline_has.py
python3 scripts/test_ansi.py
python3 scripts/test_changed_paths.py
python3 scripts/changed_paths.py sync
python3 scripts/test_ci_triggers_check.py
python3 scripts/ci_triggers_check.py
python3 scripts/test_package_contents_check.py
python3 scripts/package_contents_check.py
python3 scripts/test_package_git_patches.py
python3 scripts/package_git_patches.py args >/dev/null
# CI only (not reproduced here): the cargo semver-checks run of security-audit.yml
# (needs the previous commit built as a baseline) and the `cargo +nightly build --bins`
# of every fuzz target in fuzz.yml; their log readers and target list are tested above

step "identifier moves (G9, against the published release)"
python3 scripts/test_id_move_check.py
python3 scripts/id_move_check.py

step "docs lint (public vocabulary / CHANGELOG structure)"
# 予測されて困る値が予測可能な源から来ていないか (検査器の test を先に走らせる)
python3 scripts/test_csprng_guard.py
python3 scripts/csprng_guard.py
python3 scripts/test_docs_lint.py
python3 scripts/docs_lint.py --check

step "tracked binaries (no built library or executable is tracked)"
python3 scripts/test_tracked_binaries.py
python3 scripts/tracked_binaries.py

# security-audit.yml job deny, the same two commands (a package renamed without its
# licence exception passed every other local gate and failed only there).
# Without cargo-deny, --quick / --fast warn and skip; the full run fails.
step "cargo-deny (license + advisories + bans + sources, security-audit.yml job deny)"
if command -v cargo-deny >/dev/null; then
  cargo deny --all-features check all
  cargo deny --manifest-path scripts/cfg-items/Cargo.toml --config deny.toml check all
elif [[ $quick -eq 1 || $fast -eq 1 ]]; then
  echo "skip: cargo-deny not installed (cargo install --locked cargo-deny); CI runs this check" >&2
else
  echo "cargo-deny not installed: cargo install --locked cargo-deny" >&2
  exit 1
fi

step "SCIP reach analysis oracle (docs/integration-status.md の解析器)"
python3 scripts/test_scip_reach.py
python3 scripts/test_audit_refs.py
python3 scripts/test_run_feature_gated_tests.py

step "integration levels oracle + C ABI coverage (docs/integration-levels.md)"
python3 scripts/test_integration_levels.py
python3 scripts/integration_levels.py --no-index --check
# the landing script's tests and its child tools from a minimal environment, after the toolchain
# (as ci.yml orders them on every OS)
python3 scripts/test_land.py
# the cargo of this shell's PATH is named to land.py, as ci.yml does (the emptied environment
# cannot find a cargo outside $CARGO_HOME/bin otherwise)
cargo="$(command -v cargo)" || { echo "cargo is not on the PATH of this shell" >&2; exit 1; }
cargo_dir="$(dirname "$cargo")"
env -i HOME="$HOME" USER="${USER:-}" LAND_CARGO_DIR="$cargo_dir" PATH=/usr/bin:/bin "$(python3 -c 'import sys; print(sys.executable)')" scripts/land.py --env-smoke

step "oracle ledger links (PIN / root external)"
python3 scripts/gen-oracle-status.py --check

# ⚠️ 上の `--check` は PIN / 外部原因の link だけを見る 生成物が古いことは見ない
#    (stale でも exit 0) stale 判定は oracle-status.yml / wiring-status.yml の
#    「再生成して git diff --exit-code」だけが持っており、preflight に無かった
#    ⇒ 新しい test file を足した push が **preflight exit 0 のまま CI で red** になった
#    (2026-10-09 実測、tests/ に 1 file 足しただけで Oracle Status が failure)
step "generated ledgers are not stale (oracle-status / wiring-status)"
python3 scripts/gen-oracle-status.py
python3 scripts/gen-wiring-status.py
if ! git diff --exit-code docs/oracle-status.md docs/wiring-status.md; then
  echo "生成物が古い: 上の差分を commit に含める (scripts/land.py が再生成する)" >&2
  exit 1
fi

step "wiring-guard (oracle + 新規の未配線 / 理由の無い dead_code が無い)"
python3 scripts/test_wiring_guard.py
python3 scripts/wiring_guard.py
python3 scripts/test_entropy_guard.py
python3 scripts/entropy_guard.py
python3 scripts/test_no_private_minmax.py
python3 scripts/no_private_minmax.py
python3 scripts/test_g2_rewrite_check.py
python3 scripts/g2_rewrite_check.py
# g2-comparisons.yml と対: 比較を全部 compiler の型で判定し、浮動小数点の比較は全部 select / predicate の行を持つ
python3 scripts/test_g2_comparisons.py
python3 scripts/g2_comparisons.py
python3 scripts/test_libm_dependency_guard.py
python3 scripts/libm_dependency_guard.py
python3 scripts/test_allow_reason_guard.py
python3 scripts/allow_reason_guard.py
python3 scripts/test_changelog_dup_check.py
python3 scripts/changelog_dup_check.py

step "coverage tables (docs/coverage/*.toml = src/ LIMITATION comments + tests)"
python3 scripts/test_coverage_check.py
python3 scripts/test_line_coverage_ratchet.py
python3 scripts/test_mutants_ratchet.py
python3 scripts/test_mutants_exclude_check.py
python3 scripts/test_mutants_equivalent.py
python3 scripts/mutants_equivalent.py --check
python3 scripts/test_mutants_in_diff_plan.py
python3 scripts/test_intel_nightly_gate.py
python3 scripts/test_run_ignored.py
python3 scripts/run_ignored.py --list > /dev/null

python3 scripts/test_bench_counts_check.py
python3 scripts/test_ci_load_check.py
python3 scripts/ci_load_check.py
python3 scripts/test_downstream_select_tag.py
python3 scripts/test_downstream_siblings.py
python3 scripts/test_downstream_major_rewrite.py
python3 scripts/test_downstream_path_rewrite.py
python3 scripts/test_downstream_test_counts.py
python3 scripts/test_downstream_trt_probe.py
python3 scripts/test_downstream_resolve.py
python3 scripts/test_downstream_patches.py
python3 scripts/test_coverage_refs_to_symbols.py
python3 scripts/coverage_check.py

step "build configuration guard (G14) + effective-flags check and run_tests.sh tests"
python3 scripts/test_build_config_guard.py
python3 scripts/build_config_guard.py
python3 scripts/test_build_flags_check.py
python3 scripts/test_run_tests.py

step "mutants.toml exclude_re entries still match a real mutant (needs cargo-mutants)"
if command -v cargo-mutants >/dev/null 2>&1; then
  python3 scripts/mutants_exclude_check.py
elif [[ $quick -eq 1 || $fast -eq 1 ]]; then
  skip "mutants_exclude_check (cargo-mutants not installed) -- CI runs this check"
else
  echo "cargo-mutants not installed: cargo install cargo-mutants --locked" >&2
  exit 1
fi

step "status generators oracle (docs/wiring-status.md / docs/oracle-status.md の生成器)"
python3 scripts/test_gen_status.py

step "affected-test selector oracle (preflight --fast)"
python3 scripts/test_affected_tests.py

# CI's "Determinism tests (release, ...)" job: the same tests optimised (the sign of a zero
# result can differ from the unoptimised build on a target with a fused multiply-add)
step "determinism tests (release)"
python3 scripts/test_determinism_release.py
python3 scripts/determinism_release.py

step "effective rustc flags (G14, release profile, fresh target directory)"
python3 scripts/build_flags_check.py -- cargo check --release --lib
python3 scripts/build_flags_check.py -- cargo check --release --lib --features "std,simd,parallel,ffi,gpu-solver-bridge"
python3 scripts/build_flags_check.py -- cargo check --release --lib --no-default-features

step "example runner oracle (scripts/run_examples.py, run in full mode and ci.yml job examples)"
python3 scripts/test_run_examples.py

# ci.yml jobs msrv and msrv-privacy-crypto: the declared rust-version (1.85)
# without privacy-crypto, and the effective minimum with it (1.87, its
# dependency ctutils), each built and its library tests run from the committed
# lock (--locked). Same argument sets as CI; a missing toolchain fails the full
# run and is skipped by --quick / --fast.
step "MSRV lanes (1.85 without privacy-crypto, 1.87 with it, --locked)"
if rustup toolchain list | grep -q '^1\.85' && rustup toolchain list | grep -q '^1\.87'; then
  if [[ $quick -eq 0 && $fast -eq 0 ]]; then
    cargo +1.85 build --lib --locked --no-default-features
    cargo +1.85 test --lib --locked
    cargo +1.85 test --lib --locked --features "std,simd,parallel,ffi,gpu-solver-bridge"
    cargo +1.87 test --lib --locked --features privacy-crypto
    cargo +1.87 test --lib --locked --features neural
  else
    cargo +1.85 check --lib --locked
    cargo +1.87 check --lib --locked --features privacy-crypto
    cargo +1.87 check --lib --locked --features neural
  fi
elif [[ $quick -eq 1 || $fast -eq 1 ]]; then
  echo "warn: rustup toolchains 1.85 and 1.87 are needed for the MSRV lanes (skipped)"
else
  echo "missing toolchain: rustup toolchain install 1.85 1.87" >&2
  exit 1
fi

# Fast static gates that failed late in practice (no_std build, public API drift)
step "no_std build, and a no_std dependent (consumers/no_std, warnings denied)"
cargo build --lib --no-default-features
python3 scripts/test_no_std_consumer.py
python3 scripts/no_std_consumer.py
python3 scripts/test_binding_build_commands.py
python3 scripts/binding_build_commands.py --check

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
  # 生成物の stale 検査 (integration-status.yml と同じ形) index がある時だけ測れる
  python3 scripts/scip_reach.py --write docs/integration-status.md
  if ! git diff --exit-code docs/integration-status.md; then
    echo "docs/integration-status.md が古い: 上の差分を commit に含める" >&2
    exit 1
  fi
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

step "clippy -D warnings (the facade, native feature set, all targets) + each forwarded feature builds"
cargo clippy -p alice-physics --all-targets --features "$NATIVE" -- -D warnings
python3 scripts/facade_check.py --build-features

# CI's "Clippy (full native feature set, lib + all targets)" job adds
# neural,analytics on top of $NATIVE. A sibling-bridge feature can pull in a
# dependency that adds a competing trait impl, which only surfaces as an
# ambiguity error at this exact feature combination (missed here once and
# only caught by CI, 2026-10-03), so the combination is linted here too.
step "clippy -D warnings (full native feature set incl. neural/analytics/privacy-crypto, all targets)"
cargo clippy --all-targets --features "$NATIVE,neural,analytics,privacy-crypto" -- -D warnings

# the library code also may not call f32 / f64 max and min (CI's "Clippy
# (library code, ...)" step; .github/clippy-lib/clippy.toml is generated from
# clippy.toml and the script checks it is current and fires on the canary)
step "clippy -D warnings (library code, f32 / f64 max and min forbidden)"
CLIPPY_CONF_DIR=.github/clippy-lib cargo clippy --lib --features "$NATIVE,neural,analytics,privacy-crypto" -- -D warnings
CLIPPY_CONF_DIR=.github/clippy-lib cargo clippy --lib --features "wasm" -- -D clippy::disallowed_methods -D clippy::disallowed_types
CLIPPY_CONF_DIR=.github/clippy-lib cargo clippy --lib --features "python" -- -D clippy::disallowed_methods -D clippy::disallowed_types
CLIPPY_CONF_DIR=.github/clippy-lib cargo clippy --lib --no-default-features -- -D clippy::disallowed_methods -D clippy::disallowed_types
step "library clippy configuration (tests, then current + canary)"
python3 scripts/test_clippy_lib_gate.py
python3 scripts/clippy_lib_gate.py
step "allow reasons (canary for the manifest lint + clock reads under wide allows)"
python3 scripts/allow_reason_guard.py --clippy

step "wasm32-wasip1 golden tests build (dev-deps are built for the target too)"
rustup target list --installed | grep -q wasm32-wasip1 || rustup target add wasm32-wasip1
cargo test --test determinism_golden --target wasm32-wasip1 --no-run
cargo test --test determinism_golden_f32 --target wasm32-wasip1 --no-run
cargo test --test determinism_golden_contacts --target wasm32-wasip1 --no-run

step "wasm32-unknown-unknown build with every browser feature (CI job js-binding)"
rustup target list --installed | grep -q wasm32-unknown-unknown || rustup target add wasm32-unknown-unknown
cargo build --lib --target wasm32-unknown-unknown
cargo build --lib --target wasm32-unknown-unknown --no-default-features --features std
cargo build --lib --target wasm32-unknown-unknown --no-default-features --features "std,wasm,neural,analytics,gpu-solver-bridge,simd"
cargo build --lib --target wasm32-unknown-unknown --no-default-features --features "std,wasm,privacy-crypto"

step "rustdoc -D warnings (default + docs.rs feature set)"
RUSTDOCFLAGS="-Dwarnings" cargo doc --lib --no-deps
RUSTDOCFLAGS="-Dwarnings" cargo doc --lib --no-deps --features "$NATIVE"

step "public-text.yml / public-text: Public text (ALICE-Law checker at the pin; tree, then the commits)"
python3 scripts/test_public_text.py
scripts/public_text.sh

if [[ $quick -eq 1 ]]; then
  print_skip_summary
  echo; echo "preflight --quick OK (test suites skipped)"; exit 0
fi

if [[ $fast -eq 1 ]]; then
  step "affected tests (modules changed since origin/main + determinism goldens)"
  python3 scripts/affected_tests.py --features "$NATIVE" --run
  print_skip_summary
  echo; echo "preflight --fast OK (static gates + affected tests; the full suites run in CI)"; exit 0
fi

step "cargo test (default features, all targets)"
cargo test --no-fail-fast

step "cargo test --lib (native feature set)"
cargo test --lib --features "$NATIVE"
cargo test --lib --features privacy-crypto privacy

# same commands as ci.yml: the integration tests gated on `parallel` (in 29 test
# files) and the SIMD paths with simd alone
step "cargo test (parallel, all targets)"
cargo test --no-fail-fast --features "parallel"

step "cargo test --lib (simd)"
cargo test --lib --features "simd"

step "float environment (G14, debug and release, both tests must pass)"
scripts/run_tests.sh float_environment=2 -- cargo test --test float_environment
scripts/run_tests.sh float_environment=2 -- cargo test --release --test float_environment
scripts/run_tests.sh facade_paths=1 -- cargo test -p alice-physics --features "std,simd,parallel,ffi,gpu-solver-bridge" --test facade_paths
scripts/run_tests.sh facade_identity=3 -- cargo test -p alice-physics --features "std,simd,parallel,ffi,gpu-solver-bridge" --test facade_identity
cargo test -p alice-physics --doc

# same command as ci.yml: the semantics identifier with the gated stepping
# paths' features (`cargo test` above runs the file with default features)
step "semantics identifier (parallel + gpu-solver-bridge)"
cargo test --test physics_semantics_id --features "parallel,gpu-solver-bridge"

step "ALICE-Law G7 handshakes (analytics + gpu-solver-bridge)"
scripts/run_tests.sh physics_semantics_id=21 -- cargo test --test physics_semantics_id --features "analytics,neural"
scripts/run_tests.sh law_g7_boundaries=4 -- cargo test --test law_g7_boundaries --features "analytics,gpu-solver-bridge"

step "cargo test --lib (ffi module)"
cargo test --lib --features "ffi" "ffi::"

step "participant panic leaves the world faulted (ffi)"
scripts/run_tests.sh participant_panic_leaves_world_faulted=2,faulted_world_refuses_changes=5,panic_paths_fault_the_world=3,c_world_fault_codes=1 -- cargo test --test participant_panic_leaves_world_faulted --test faulted_world_refuses_changes --test panic_paths_fault_the_world --test c_world_fault_codes --features "ffi,gpu-solver-bridge"

step "cargo test --lib (wasm module) + cargo check (python)"
cargo test --lib --features "wasm" "wasm::"
cargo check --lib --features "python"

step "WebAssembly binding from JavaScript (cargo rustc + wasm-bindgen CLI + node; CI job js-binding)"
command -v wasm-bindgen >/dev/null && command -v node >/dev/null \
  || { echo "the wasm-bindgen CLI (Cargo.lock's version) and node are needed (cargo install wasm-bindgen-cli --version <ver> --locked; node 18+)" >&2; exit 1; }
bash scripts/run_js_binding_tests.sh
python3 scripts/facade_wasm_check.py

step "C ABI libraries: the facade's cdylib / staticlib export the engine's symbols (release; CI job facade-artifacts)"
python3 scripts/facade_artifacts.py

step "Python binding (maturin wheel + python/tests/test_binding.py; CI python-binding.yml)"
# a venv under the target dir with maturin and numpy, made once
py_target="${CARGO_TARGET_DIR:-target}"
py_venv="$py_target/python-binding-venv"
[ -x "$py_venv/bin/maturin" ] || { python3 -m venv "$py_venv" && "$py_venv/bin/pip" install -q maturin numpy; }
rm -rf "$py_target/python-wheels"
"$py_venv/bin/maturin" build -m crates/alice-physics/Cargo.toml --features python --out "$py_target/python-wheels"
"$py_venv/bin/pip" install -q --force-reinstall "$py_target"/python-wheels/*.whl
"$py_venv/bin/python" python/tests/test_binding.py

step "cargo test --lib (neural / analytics via crates.io siblings)"
cargo test --lib --features "neural,analytics"

step "feature-gated integration tests (same script and arguments as ci.yml)"
python3 scripts/run_feature_gated_tests.py

step "run every example in release (same script and arguments as ci.yml job examples)"
python3 scripts/run_examples.py

print_skip_summary
echo; echo "preflight OK"
