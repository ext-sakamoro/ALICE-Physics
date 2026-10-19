#!/usr/bin/env bash
# Build the WebAssembly package (the `wasm` feature) without wasm-pack.
#
# `[lib] crate-type` is rlib only (3.0.1: a cdylib there is built by every dependent,
# no_std ones included), so the cdylib that wasm-bindgen reads is asked for here, and the
# JavaScript glue comes from the wasm-bindgen CLI. The CLI must be the same version as the
# `wasm-bindgen` crate in Cargo.lock (the glue and the custom sections it reads are tied to
# it); a different version fails here instead of producing a package that breaks at load.
#
# usage: scripts/build_wasm_pkg.sh <nodejs|web|bundler> <out-dir> [--dev]
# Writes <out-dir>/alice_physics.js, alice_physics_bg.wasm, alice_physics.d.ts,
# alice_physics_bg.wasm.d.ts and package.json (the same files wasm-pack wrote).
set -euo pipefail
cd "$(dirname "$0")/.."

js_target=${1:?usage: build_wasm_pkg.sh <nodejs|web|bundler> <out-dir> [--dev]}
out=${2:?usage: build_wasm_pkg.sh <nodejs|web|bundler> <out-dir> [--dev]}
profile=release
cargo_profile=(--release)
bindgen_flags=()
if [ "${3:-}" = "--dev" ]; then
  profile=debug
  cargo_profile=()
  bindgen_flags=(--debug)   # what wasm-pack passed for --dev (the debug JS glue)
fi

want=$(awk '/^\[\[package\]\]/{p=0} $0=="name = \"wasm-bindgen\""{p=1} p&&/^version = /{gsub(/"/,"",$3); print $3; exit}' Cargo.lock)
[ -n "$want" ] || { echo "error: no wasm-bindgen in Cargo.lock" >&2; exit 1; }
have=$(wasm-bindgen --version 2>/dev/null | awk '{print $2}') \
  || { echo "error: wasm-bindgen CLI not found (cargo install wasm-bindgen-cli --version $want --locked)" >&2; exit 1; }
if [ "$have" != "$want" ]; then
  echo "error: wasm-bindgen CLI $have, Cargo.lock has wasm-bindgen $want (install the same version)" >&2
  exit 1
fi

target_dir=${CARGO_TARGET_DIR:-target}
cargo rustc -p alice-physics --lib --target wasm32-unknown-unknown ${cargo_profile[@]+"${cargo_profile[@]}"} --features wasm --crate-type cdylib
wasm="$target_dir/wasm32-unknown-unknown/$profile/alice_physics.wasm"
[ -f "$wasm" ] || { echo "error: $wasm was not produced" >&2; exit 1; }

mkdir -p "$out"
wasm-bindgen --target "$js_target" --out-dir "$out" --out-name alice_physics ${bindgen_flags[@]+"${bindgen_flags[@]}"} "$wasm"

# package.json and the copied files, as wasm-pack wrote them (from Cargo.toml's [package])
python3 - "$out" "$js_target" <<'PY'
import glob, json, shutil, sys, tomllib
out, target = sys.argv[1:]
pkg_toml = tomllib.load(open("Cargo.toml", "rb"))["package"]
files = ["alice_physics_bg.wasm", "alice_physics.js", "alice_physics.d.ts"]
pkg = {"name": pkg_toml["name"]}
if pkg_toml.get("authors"):
    pkg["collaborators"] = pkg_toml["authors"]
pkg["description"] = pkg_toml["description"]
pkg["version"] = pkg_toml["version"]
pkg["license"] = pkg_toml["license"]
if pkg_toml.get("repository"):
    pkg["repository"] = {"type": "git", "url": pkg_toml["repository"]}
pkg["files"] = files
if target == "nodejs":
    pkg["main"] = "alice_physics.js"
else:
    pkg["type"] = "module"
    pkg["main" if target == "web" else "module"] = "alice_physics.js"
if pkg_toml.get("homepage"):
    pkg["homepage"] = pkg_toml["homepage"]
pkg["types"] = "alice_physics.d.ts"
if target != "nodejs":
    pkg["sideEffects"] = ["./snippets/*"]
if pkg_toml.get("keywords"):
    pkg["keywords"] = pkg_toml["keywords"]
with open(f"{out}/package.json", "w", encoding="utf-8") as f:
    json.dump(pkg, f, indent=2)
    f.write("\n")
for extra in ["README.md"] + sorted(glob.glob("LICENSE*")):
    shutil.copy(extra, out)
PY
for f in alice_physics.js alice_physics_bg.wasm alice_physics.d.ts package.json; do
  [ -f "$out/$f" ] || { echo "error: $out/$f was not produced" >&2; exit 1; }
done
echo "wasm package: $out ($js_target, $profile, wasm-bindgen $have)"
