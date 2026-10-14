# shellcheck shell=bash
# Sourced by downstream_check.sh (and by test_downstream_trt_probe.py).
#
# trt_gpu_probe asks the placed ALICE-TRT for a GPU adapter the way its tests do
# (`GpuDevice::new`); its GPU tests return early, and pass, without one. Two
# separate failures, reported apart:
#
#   - the probe does not build (a compiler error, an MSRV error of a dependency,
#     a resolution error): the message is the compiler's, and no statement is
#     made about the adapter, which was never asked
#   - the probe builds and runs, and prints nothing or exits non-zero: no adapter
#
# Needs from the caller: ROOT, TRT, and the arrays failed and summary.
trt_gpu_probe() {
    local probe="$ROOT/trt-gpu-probe" log="$ROOT/logs/trt-gpu-probe.log"
    local build_log="$ROOT/logs/trt-gpu-probe-build.log" out msg esc plain
    if [ "${TRT_GPU_PROBE:-1}" = 0 ]; then
        echo "trt: GPU probe skipped (TRT_GPU_PROBE=0); its GPU tests return early without an adapter"
        summary+=("trt GPU adapter: not probed (TRT_GPU_PROBE=0)")
        return
    fi
    mkdir -p "$probe/src"
    cat > "$probe/Cargo.toml" <<'TOML'
[package]
name = "trt-gpu-probe"
version = "0.0.0"
edition = "2021"
publish = false

[dependencies]
alice-trt = { path = "../ALICE-TRT", features = ["physics-solver"] }

[workspace]
TOML
    cat > "$probe/src/main.rs" <<'RUST'
fn main() {
    match alice_trt::GpuDevice::new() {
        Ok(device) => println!("{}", device.info()),
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}
RUST
    # the versions cargo tree just resolved for ALICE-TRT, so the probe reuses its build
    if [ -f "$TRT/Cargo.lock" ]; then
        cp "$TRT/Cargo.lock" "$probe/Cargo.lock"
    fi
    # run from ALICE-TRT so its rust-toolchain.toml picks the toolchain of the tests
    echo "==> trt-gpu-probe: build"
    if ! (cd "$TRT" && cargo build -q --color never --manifest-path "$probe/Cargo.toml") >"$build_log" 2>&1; then
        cat "$build_log" >&2
        # CARGO_TERM_COLOR=always may still colour; strip SGR codes, and never let the
        # extraction fail the script (set -e, pipefail): fall back to the log's last lines
        esc=$(printf '\033')
        plain=$(sed "s/${esc}\[[0-9;]*[A-Za-z]//g" "$build_log" || true)
        msg=$(printf '%s\n' "$plain" | grep -m1 -A1 '^error' | tr '\n' ' ' || true)
        [ -n "$msg" ] || msg=$(printf '%s\n' "$plain" | grep -m1 -A1 '^warning' | tr '\n' ' ' || true)
        [ -n "$msg" ] || msg=$(printf '%s\n' "$plain" | tail -n 5 | tr '\n' ' ' || true)
        echo "error: trt: the GPU probe failed to build, so the adapter was not asked: ${msg:-no compiler output}" >&2
        failed+=("trt (GPU probe build failed)")
        summary+=("trt GPU adapter: not probed, the probe did not build")
        return
    fi
    echo "==> trt-gpu-probe: alice_trt::GpuDevice::new()"
    if out=$(cd "$TRT" && cargo run -q --manifest-path "$probe/Cargo.toml" 2>"$log") && [ -n "$out" ]; then
        echo "trt: GPU adapter: $out"
        summary+=("trt GPU adapter: $out")
    else
        cat "$log" >&2
        echo "error: trt: ALICE-TRT finds no GPU adapter (${out:-no output}); its GPU tests would return early and pass without running" >&2
        failed+=("trt (no GPU adapter)")
        summary+=("trt GPU adapter: none")
    fi
}
