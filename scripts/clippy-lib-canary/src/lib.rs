//! Each function uses one form that `.github/clippy-lib/clippy.toml` forbids
//! `scripts/clippy_lib_gate.py` runs clippy with that configuration and
//! requires a `disallowed_methods` diagnostic for every one of them

pub fn method_f32(a: f32, b: f32) -> f32 {
    a.max(b)
}

pub fn path_f32(a: f32, b: f32) -> f32 {
    f32::min(a, b)
}

pub fn method_f64(a: f64, b: f64) -> f64 {
    a.min(b)
}

pub fn fold_f64(v: &[f64]) -> f64 {
    v.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

pub fn clamp_f32(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

pub fn libm_f32(x: f32) -> f32 {
    x.sin()
}

// the wall clock read through `elapsed` (Instant and SystemTime): five forms
use std::time::Instant as Clock;

pub fn elapsed_method(started: std::time::Instant) -> std::time::Duration {
    started.elapsed()
}

pub fn elapsed_path(t0: std::time::Instant) -> std::time::Duration {
    std::time::Instant::elapsed(&t0)
}

pub fn elapsed_system(stamp: std::time::SystemTime) -> bool {
    stamp.elapsed().is_ok()
}

pub fn elapsed_alias(clock_t: Clock) -> std::time::Duration {
    clock_t.elapsed()
}

pub fn elapsed_qualified_self(t1: std::time::Instant) -> std::time::Duration {
    <std::time::Instant>::elapsed(&t1)
}

// must stay clean: the difference of two values already held reads no clock
pub fn duration_between(earlier: std::time::Instant, later: std::time::Instant) -> std::time::Duration {
    later.duration_since(earlier)
}
