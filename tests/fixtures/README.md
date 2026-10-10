# Test fixtures

## World snapshot blobs (`world_snapshot_v*.bin`)

Blobs of `PhysicsWorld::snapshot_world`, kept byte for byte so that the
reader of every later build is tested against what an earlier writer
actually produced. The file name gives the format version (`v3`, `v4`,
`v5`) and the scene; `_later` is the same world stepped
`FIXTURE_LATER_STEPS` (20) more times by the same writer.

| files | written by | read by |
|---|---|---|
| `world_snapshot_v1_stacked.bin` | commit `686bf7d4` (no release wrote version 1) | `tests/analytic_world_step_ccd.rs`, `tests/analytic_world_participant_wiring.rs`, `tests/world_snapshot_v4.rs` |
| `world_snapshot_v2_stacked.bin` | commit `d4c4f125` (no release wrote version 2) | `tests/analytic_world_step_ccd.rs`, `tests/world_snapshot_v4.rs` |
| `world_snapshot_v3_*.bin` | tag `v2.0.0` (`c85cd1db7298ec7a7f6723160505890c0f39b8eb`), the first release that wrote version 3 | `tests/world_snapshot_v5.rs`, `tests/world_snapshot_v4.rs`, `tests/world_snapshot_dynamic_tree_canonical.rs` |
| `world_snapshot_v4_*.bin` | tag `v2.1.0` (`9e852775755600ecba1ad5fab1bd06d5e51f4310`), the only release that wrote version 4 | `tests/world_snapshot_v5.rs` |
| `world_snapshot_v5_*.bin` | the current writer (pinned: `tests/world_snapshot_v5.rs` asserts the writer still emits these bytes) | `tests/world_snapshot_v5.rs` |

`world_snapshot_v3_joint_pair.bin` and `world_snapshot_v3_dynamic_tree_freed.bin`
were first committed from a development build; the `v2.0.0` tag build
writes the same bytes (checked with `cmp` when the other version 3 files
were generated), so they are the 2.0.0 output.

The scenes are `world_snapshot_scenes.rs` (public API only, compiles
against 2.0.0, 2.1.0 and the current tree) and the program is
`world_snapshot_gen.rs`. Neither is built by `cargo test` here;
`tests/world_snapshot_v5.rs` includes the scenes file to rebuild the same
worlds.

### How to regenerate

Version 3 (tag `v2.0.0`) and version 4 (tag `v2.1.0`), from the repository
root; `<tag>` is `v2.0.0` or `v2.1.0`:

```sh
dir=$(mktemp -d)
git archive <tag> | tar -x -C "$dir"
cp tests/fixtures/world_snapshot_scenes.rs tests/fixtures/world_snapshot_gen.rs "$dir/examples/"
(cd "$dir" && CARGO_TARGET_DIR="$dir/target" cargo run --example world_snapshot_gen -- "$OLDPWD/tests/fixtures")
```

Version 5 (the current tree):

```sh
cp tests/fixtures/world_snapshot_scenes.rs tests/fixtures/world_snapshot_gen.rs examples/
cargo run --example world_snapshot_gen -- tests/fixtures
rm examples/world_snapshot_scenes.rs examples/world_snapshot_gen.rs
```

Each tag's `rust-toolchain.toml` pins Rust 1.98.1; the fixtures were
written with that toolchain, default features, the `dev` profile, on
arm64 (macOS). The format is little endian and `Fix128` arithmetic is
integer only, so the bytes do not depend on the target.

Version 5 files change only when the format or a scene changes; the older
ones never do (they record what a release wrote). Regenerating them from
their tags must give the SHA-256 below.

| file | SHA-256 |
|---|---|
| `world_snapshot_v3_bvh_later.bin` | `809856d3755bc36f3237b560bd0e106409d71b546690019e2ae29fcab4d79a5d` |
| `world_snapshot_v3_bvh.bin` | `ee798edf2f3cdebd13d42fd7ba55a5f65dadc196bf54c74b03b2dfba10cac9b6` |
| `world_snapshot_v3_dynamic_tree_freed_later.bin` | `6b332eeb56dead7b32342ddcf4d2ecfd80ce6174f6b24bbb2821130a0fbe2cab` |
| `world_snapshot_v3_dynamic_tree_freed.bin` | `0eae4e709ceb5590f043800f8d3c566c64cdfa613bbbc1a2effbf4bc5c0427db` |
| `world_snapshot_v3_dynamic_tree_later.bin` | `7d93bc0b05a92638c51bb5b6430ae15dfa04eaf2db914782321c779640eaa64d` |
| `world_snapshot_v3_dynamic_tree.bin` | `3efcf96f7700058e88b9123d546c18422314a7d9d84b3d7aadddc67ff81ab2a1` |
| `world_snapshot_v3_hybrid_later.bin` | `1bfb58542049f85365ee96fb3462cd0e131cd25872163ee205f07fc469be8b51` |
| `world_snapshot_v3_hybrid.bin` | `0d3aaa22ca5aa0e8353f9ce3f163d574f31004aa30bbf3e512ed2dc74713673e` |
| `world_snapshot_v3_joint_pair_later.bin` | `2f198dbdbd505c985ea256fb021bc3b12736b7f4b00b853b4bed71ab54914cbf` |
| `world_snapshot_v3_joint_pair.bin` | `26e61c07777589218f628c013bbca558d885081786ceb83f18cdfd3fb406d36f` |
| `world_snapshot_v4_bvh_later.bin` | `0d6553de7cf13672d49df68ef72979d0d9a7ae8ac0ae7d414b4a16f7cecbbab1` |
| `world_snapshot_v4_bvh.bin` | `d5ac1e8f7aa36129304a4c84630d3dda4126b754ca11708ecf5e9d45e88f21c9` |
| `world_snapshot_v4_dynamic_tree_freed_later.bin` | `bf76c885c7f96c1e854f92204b9896a289e59d9f7fa784d09dd65aad36dcb285` |
| `world_snapshot_v4_dynamic_tree_freed.bin` | `d1cb41404809abf066da316b6a154743dfd1469826931e9042fb4ff00a2ed1de` |
| `world_snapshot_v4_dynamic_tree_later.bin` | `3c242fe54fbf4f6fa32938998bbb63e9bedb8a263965613f43e267ac48168a42` |
| `world_snapshot_v4_dynamic_tree.bin` | `5020f4ee0a7fd997c53612261e6e8f7a9fe50ce4a8fdc2fef81905d21d1ea99a` |
| `world_snapshot_v4_hybrid_later.bin` | `4926849b6e39af4d73db9a7e8cd0f8ba48ef369b9f11f1e6cba9a2e60b9914be` |
| `world_snapshot_v4_hybrid.bin` | `fd22406d3e3e1c6912d4cbf10c8778d93cbeeac5b0e406c07ba23c2c81dd9710` |
| `world_snapshot_v4_joint_pair_later.bin` | `7133f609b9450001738ddb62b5d85fc169ec5228eb188a9b1bd36c8ab2ae5648` |
| `world_snapshot_v4_joint_pair.bin` | `1cef52ac12a7ac6c9dea1804cbee7ba4eb1bc1dd9b3fa5a668cca9d9ac69c304` |
| `world_snapshot_v5_bvh.bin` | `e57c40aea25e0c7f22d5d575134ab6827637185e0452f5ead5b0ed5dd923f8c8` |
| `world_snapshot_v5_dynamic_tree_freed.bin` | `8dc1f775a6277f9f558e49a7379db2b41ffeb4ea02ffb9e03a8eec0dd3282219` |
| `world_snapshot_v5_dynamic_tree.bin` | `5843f78cad2f70fe29d72664e48f5e6314f202ea5fb8624da27a851ba40c44f9` |
| `world_snapshot_v5_hybrid.bin` | `7bc5866bb16cda345aeaa2564026bfb7f7e0387a94213a1132edce070ee9b44b` |
| `world_snapshot_v5_joint_pair.bin` | `3081f1fab9c5761eb20d7666f8685cbc4f185a3fcb79b079f86a305b00609fa3` |
