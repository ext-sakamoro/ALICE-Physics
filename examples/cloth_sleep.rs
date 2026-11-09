//! A soft sheet that is pulled, let go, and goes to sleep once it is exactly at rest
//!
//! Reaches `Cloth::pin`, `Cloth::unpin`, `Cloth::set_sleep` and `Cloth::is_asleep`.
//!
//! A 7×7 sheet is tethered to its rest shape (period 0.5 s, damping ratio 0.42) with no
//! gravity. A corner is held and dragged for half a second, then released: the sheet
//! wobbles back. With sleep on, the first `step` that changes no bit of the state puts it to
//! sleep and the following steps are skipped; a tap wakes it, and from then on it follows
//! the same trajectory as a sheet that never slept. The example runs both and checks that.
//!
//! Author: Moroya Sakamoto

use alice_physics_core::cloth::{Cloth, ClothConfig};
use alice_physics_core::math::{Fix128, Vec3Fix};

fn sheet(sleep: bool) -> Cloth {
    let mut c = Cloth::new_grid(Vec3Fix::ZERO, Fix128::ONE, Fix128::ONE, 7, 7, Fix128::ONE);
    c.config = ClothConfig {
        iterations: 4,
        substeps: 1,
        gravity: Vec3Fix::ZERO,
        damping: Fix128::from_ratio(985, 1000),
        stretch_compliance: Fix128::from_ratio(1, 100),
        bend_compliance: Fix128::from_ratio(1, 4),
        ..ClothConfig::default()
    };
    let omega =
        Fix128::from_ratio(4, 1) * Fix128::from_ratio(314_159_265_358_979, 100_000_000_000_000);
    c.set_rest_tether(omega * omega, Fix128::from_ratio(84, 100) * omega);
    c.set_sleep(sleep);
    c
}

fn main() {
    let dt = Fix128::from_ratio(1, 60);
    let tap = 700;
    let mut runs = [sheet(true), sheet(false)];
    let corner = 48;
    let start = runs[0].rest_positions()[corner];
    let (mut fell_asleep, mut skipped) = (None, 0);
    for f in 0..tap + 120 {
        for c in &mut runs {
            if (10..40).contains(&f) {
                c.pin(corner);
                let t = Fix128::from_ratio(i64::from(f) - 10, 30);
                c.positions[corner] = start
                    + Vec3Fix::new(
                        Fix128::from_ratio(6, 10) * t,
                        Fix128::ZERO,
                        Fix128::from_ratio(3, 10) * t,
                    );
                c.velocities[corner] = Vec3Fix::ZERO;
            }
            if f == 40 {
                c.unpin(corner);
            }
            if f == tap {
                c.velocities[3] = c.velocities[3]
                    + Vec3Fix::new(Fix128::ZERO, Fix128::from_ratio(1, 2), Fix128::ZERO);
            }
        }
        if runs[0].is_asleep() {
            skipped += 1;
        }
        for c in &mut runs {
            c.step(dt);
        }
        if f > 40 && fell_asleep.is_none() && runs[0].is_asleep() {
            fell_asleep = Some(f);
        }
        assert_eq!(runs[0].positions, runs[1].positions, "frame {f}");
        assert_eq!(runs[0].velocities, runs[1].velocities, "frame {f}");
    }
    println!("released at frame 40, asleep after frame {fell_asleep:?}, {skipped} steps skipped,");
    println!("tapped at frame {tap}: the woken sheet matched the never-slept one on every frame");
}
