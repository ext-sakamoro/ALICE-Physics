# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.5.x   | Yes       |
| < 0.5   | No        |

## Reporting a Vulnerability

If you discover a security vulnerability in ALICE-Physics, please report it responsibly:

1. **Do NOT** open a public GitHub issue for security vulnerabilities.
2. Email: ext-sakamoro@github.com with subject `[SECURITY] ALICE-Physics: <brief description>`.
3. Include: version, steps to reproduce, potential impact.
4. We will acknowledge receipt within 48 hours and provide a fix timeline.

## Security Considerations

### Deterministic Arithmetic

ALICE-Physics uses 128-bit fixed-point arithmetic (`Fix128`) for all computations.
This eliminates floating-point non-determinism but introduces the following considerations:

- **Overflow**: `Fix128` operations use wrapping arithmetic. Extremely large values
  (> 9.2 x 10^18) will silently wrap. Ensure inputs are within reasonable bounds.
- **Division by zero**: Returns `Fix128::ZERO` instead of panicking. This is by design
  for physics stability, but callers should validate inputs when precision matters.

### Deserialization

- Binary scene format (`.aphys`) reads raw bytes. Only load files from trusted sources.
- JSON scene format validates structure but does not limit recursion depth.
- The `DeserializationFailed` error is returned for malformed input.

### Memory Safety

- All `unsafe` code is limited to:
  - SIMD intrinsics (behind `#[cfg(feature = "simd")]`)
  - Parallel constraint solver pointer wrappers (behind `#[cfg(feature = "parallel")]`)
  - FFI boundary (behind `#[cfg(feature = "ffi")]`)
- Graph-coloring invariant guarantees safe parallel access to disjoint body slots.

### Dependency Auditing

Run `cargo audit` periodically to check for known vulnerabilities in dependencies.
