// Print the export names of a .wasm file, one per line, sorted.
// usage: node scripts/wasm_exports.mjs <file.wasm>
// Used to compare two builds of the WebAssembly package by what they export, not by bytes.
import { readFileSync } from "node:fs";

const file = process.argv[2];
if (!file) {
  console.error("usage: node scripts/wasm_exports.mjs <file.wasm>");
  process.exit(2);
}
const names = WebAssembly.Module.exports(new WebAssembly.Module(readFileSync(file)))
  .map((e) => `${e.kind} ${e.name}`)
  .sort();
if (names.length === 0) {
  console.error(`error: ${file} exports nothing`);
  process.exit(1);
}
console.log(names.join("\n"));
