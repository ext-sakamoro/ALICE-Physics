//! Audit oracles for coupled_field: `reconcile_mean` on a reused channel
//! returns the mean of the current participants only, whatever the channel
//! held before (AUD-C-S1W5-006).
//!
//! A channel built by `CoupledScalar::coupled_channel` starts zero-filled, so
//! the reset at the top of `reconcile_mean` is only visible when the same
//! channel is passed a second time, or when it arrives holding a value. The
//! expected means are dyadic, so the comparison is exact.

use alice_physics::coupled_field::{
    reconcile_mean, CoupledField, CoupledFieldError, CoupledScalar,
};
use alice_physics::math::Fix128;

fn t3(a: i64, b: i64, c: i64) -> (Fix128, Fix128, Fix128) {
    (
        Fix128::from_int(a),
        Fix128::from_int(b),
        Fix128::from_int(c),
    )
}

const N: usize = 4;

struct Part {
    field: Vec<Fix128>,
}

impl Part {
    fn new(values: [f64; N]) -> Self {
        Self {
            field: values.iter().map(|v| Fix128::from_f64(*v)).collect(),
        }
    }
    fn values(&self) -> Vec<f64> {
        self.field.iter().map(|v| v.to_f64()).collect()
    }
}

impl CoupledScalar for Part {
    fn coupled_name(&self) -> &'static str {
        "temperature"
    }
    fn coupled_channel(&self) -> Result<CoupledField, CoupledFieldError> {
        CoupledField::try_new(N, 1, 1, t3(0, 0, 0), t3(1, 1, 1))
    }
    fn publish(&self, out: &mut CoupledField) -> Result<(), CoupledFieldError> {
        out.as_mut_slice().copy_from_slice(&self.field);
        Ok(())
    }
    fn adopt(&mut self, src: &CoupledField) -> Result<(), CoupledFieldError> {
        self.field.copy_from_slice(src.as_slice());
        Ok(())
    }
}

fn channel_values(c: &CoupledField) -> Vec<f64> {
    c.as_slice().iter().map(|v| v.to_f64()).collect()
}

fn mean(a: [f64; N], b: [f64; N]) -> Vec<f64> {
    (0..N).map(|i| (a[i] + b[i]) / 2.0).collect()
}

#[test]
fn reused_channel_holds_the_mean_of_the_second_reconcile_only() {
    let (a1, b1) = ([1.0, 2.0, 3.0, 4.0], [3.0, 6.0, 9.0, 12.0]);
    let (a2, b2) = ([10.0, 20.0, 30.0, 40.0], [-2.0, 0.5, 6.25, 8.0]);
    let mut channel = CoupledField::try_new(N, 1, 1, t3(0, 0, 0), t3(1, 1, 1)).unwrap();

    let (mut p, mut q) = (Part::new(a1), Part::new(b1));
    reconcile_mean(&mut [&mut p, &mut q], &mut channel).expect("first reconcile");
    assert_eq!(channel_values(&channel), mean(a1, b1));

    let (mut r, mut s) = (Part::new(a2), Part::new(b2));
    reconcile_mean(&mut [&mut r, &mut s], &mut channel).expect("second reconcile");
    let want = mean(a2, b2);
    assert_eq!(
        channel_values(&channel),
        want,
        "the channel must hold (a2 + b2) / 2, not include the first mean"
    );
    assert_eq!(r.values(), want);
    assert_eq!(s.values(), want);
}

#[test]
fn reconciling_the_same_participants_twice_is_idempotent() {
    // After the first call every participant holds the mean m; the mean of
    // two copies of m is m. A channel that kept its contents would give 2m.
    let (a, b) = ([0.5, 1.5, -4.0, 7.0], [2.5, -1.5, 8.0, 1.0]);
    let (mut p, mut q) = (Part::new(a), Part::new(b));
    let mut channel = CoupledField::try_new(N, 1, 1, t3(0, 0, 0), t3(1, 1, 1)).unwrap();
    let want = mean(a, b);
    for round in 0..3 {
        reconcile_mean(&mut [&mut p, &mut q], &mut channel).expect("reconcile");
        assert_eq!(channel_values(&channel), want, "round {round}");
        assert_eq!(p.values(), want, "round {round}");
    }
}

#[test]
fn a_channel_arriving_with_a_value_is_overwritten_by_the_mean() {
    // A pre-filled channel (here 1000 in every cell) is a caller's scratch
    // buffer: its contents are not an input to the mean.
    let (a, b, c) = (
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
    );
    let (mut p, mut q, mut r) = (Part::new(a), Part::new(b), Part::new(c));
    let mut channel =
        CoupledField::try_new_filled(N, 1, 1, t3(0, 0, 0), t3(1, 1, 1), Fix128::from_int(1000))
            .unwrap();
    reconcile_mean(&mut [&mut p, &mut q, &mut r], &mut channel).expect("reconcile");
    let want: Vec<f64> = (0..N).map(|i| (a[i] + b[i] + c[i]) / 3.0).collect();
    // (a + b + c) / 3 is an integer per cell here (5, 6, 7, 8), so exact
    assert_eq!(want, vec![5.0, 6.0, 7.0, 8.0]);
    assert_eq!(channel_values(&channel), want);
    assert_eq!(q.values(), want);
}
