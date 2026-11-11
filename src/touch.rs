//! A visitor's touch on a body: take hold of it at a point, carry that point, let go. The touches are inputs a log
//! carries ([`crate::input_log::InputLog`]), so a world touched, logged and replayed comes to the same state.
//!
//! # MODEL — taking hold of a body's point, by a rigid hand or a finite grip
//!
//! **Equations.** A grab takes the body's point nearest the touch within the hand's reach. A rigid hand
//! ([`Hand::new`](crate::touch::Hand::new)) pins it where the hand is: it moves with the hand and nothing else moves it;
//! a drag carries it; letting go returns it to the body as it was (its mass, or pinned if it was pinned before). A
//! finite grip ([`Hand::with_grip`](crate::touch::Hand::with_grip)) leaves the point the body's and pulls it toward the
//! hand by a spring `F = k_g (h − x)`, taken each step as the backward Euler impulse
//! `J = k_g dt (h − x − dt v) / (1 + k_g w dt²)` (`w` the point's inverse mass) before the body's own step
//! ([`Hand::pull`](crate::touch::Hand::pull)); when `|k_g (h − x)|` exceeds the grip's strength `F_max` it slips and the
//! hand lets go. Against a body that holds the point by `k_t`, the point rests at `x = k_g d / (k_g + k_t)` for a hand
//! `d` away (two springs in series), and a slow drag lets go at `d = F_max (k_g + k_t) / (k_g k_t)`. What the body does
//! is its own law: a cloth held to its rest shape by its tether
//! ([`Cloth::set_rest_tether`](crate::cloth::Cloth::set_rest_tether)) springs back as a damped oscillator. The body says
//! what a point is ([`Grabbable`](crate::touch::Grabbable)): for a cloth, one of its particles.
//!
//! **Units.** SI: positions and reach m, `k_g` N/m, `F_max` N, `J` N·s, `w` 1/kg, `dt` s. Every quantity is `Fix128`
//! (Q64.64); touches travel on a grid of 2⁻¹⁶ m (15 µm), so a log is short and what is read back is what was played.
//!
//! **Conservation.** A hand is an external force: while it holds, the body's momentum changes by the grip's impulses
//! (`Σ J`) and its energy by their work; a rigid hand moves a pinned point without regard to energy. Letting go adds
//! or takes nothing. The grip has no damping of its own; the backward Euler step damps it numerically (it never adds
//! energy), the body's own law does the rest.
//!
//! **Range.** Any stiffness, mass and step: the backward Euler impulse is unconditionally stable (an explicit
//! `F dt` kick is stable only while `√(k_g w) dt < 2`, which a 0.1 kg cloth point under the law's grip already
//! exceeds at 1/60 s). A grip of no stiffness or no strength, or one not finite, takes nothing; a touch out of reach
//! takes nothing; a drag or a let go with nothing held does nothing.
//!
//! **Approximations.** The grip is a point spring (no contact patch, no torque, no friction across the hand); it
//! acts once per step before the body's step (a splitting whose rest is within O(dt²) of the continuous balance);
//! positions are quantised to 2⁻¹⁶ m; the slip is all at once (no partial slip).
//!
//! **Validation.** `tests/analytic_touch.rs::a_finite_grip_holds_the_point_where_the_springs_balance::a_finite_grip_slips_where_its_pull_reaches_its_strength::the_grip_comes_to_the_continuous_balance_as_the_step_shrinks::a_light_point_under_a_stiff_grip_stays_where_the_springs_balance::a_letter_let_go_springs_back_by_its_tether::a_touch_out_of_reach_takes_nothing::letting_go_of_a_pinned_point_leaves_it_pinned::a_grip_of_no_strength_takes_nothing::touches_are_written_on_their_grid_and_read_back_exactly::the_digest_covers_what_the_hand_did`
//!
//! **Tiers.** L0 only.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::cloth::Cloth;
use crate::input_log::LogInput;
use crate::math::{Fix128, Vec3Fix};

/// Bits below the grid a touch's position is carried on (2⁻¹⁶ of a unit)
const GRID_BITS: u32 = 16;

/// What a hand does
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Touch {
    /// Take hold of body `body` at its point nearest `at`
    Grab {
        /// The body's index among those the hand reaches
        body: u32,
        /// Where the touch is
        at: Vec3Fix,
    },
    /// Carry the point held to `to`
    Drag {
        /// Where the point goes
        to: Vec3Fix,
    },
    /// Let go
    Release,
}

/// `x` on the grid: rounded to the nearest 2⁻¹⁶, ties away from zero
fn on_grid(x: Fix128) -> Fix128 {
    let q = grid_units(x);
    from_grid_units(q)
}

/// `x` in grid steps, rounded to the nearest
fn grid_units(x: Fix128) -> i64 {
    // the 128-bit value is hi·2⁶⁴ + lo; in grid steps that is hi·2¹⁶ + lo / 2⁴⁸
    let raw = (i128::from(x.hi) << 64) | i128::from(x.lo);
    let half = 1i128 << (63 - GRID_BITS);
    let q = (raw + if raw >= 0 { half } else { -half }) / (1i128 << (64 - GRID_BITS));
    q.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64
}

fn from_grid_units(q: i64) -> Fix128 {
    let raw = i128::from(q) << (64 - GRID_BITS);
    Fix128 {
        hi: (raw >> 64) as i64,
        lo: raw as u64,
    }
}

fn grid(v: Vec3Fix) -> Vec3Fix {
    Vec3Fix::new(on_grid(v.x), on_grid(v.y), on_grid(v.z))
}

impl Touch {
    /// Take hold of `body` at `at` (put on the grid)
    #[must_use]
    pub fn grab(body: u32, at: Vec3Fix) -> Self {
        Self::Grab { body, at: grid(at) }
    }

    /// Carry the point held to `to` (put on the grid)
    #[must_use]
    pub fn drag(to: Vec3Fix) -> Self {
        Self::Drag { to: grid(to) }
    }
}

fn write_varint(mut n: u64, out: &mut Vec<u8>) {
    while n >= 0x80 {
        out.push((n as u8 & 0x7f) | 0x80);
        n >>= 7;
    }
    out.push(n as u8);
}

fn read_varint(bytes: &mut &[u8]) -> Option<u64> {
    let mut n = 0u64;
    let mut shift = 0u32;
    loop {
        let (&b, after) = bytes.split_first()?;
        *bytes = after;
        if shift > 63 {
            return None;
        }
        n |= u64::from(b & 0x7f).checked_shl(shift)?;
        shift += 7;
        if b & 0x80 == 0 {
            return Some(n);
        }
    }
}

fn write_point(p: Vec3Fix, out: &mut Vec<u8>) {
    for c in [p.x, p.y, p.z] {
        // zigzag: small magnitudes of either sign stay short
        let q = grid_units(c);
        write_varint(((q << 1) ^ (q >> 63)) as u64, out);
    }
}

fn read_point(bytes: &mut &[u8]) -> Option<Vec3Fix> {
    let mut c = [Fix128::ZERO; 3];
    for x in &mut c {
        let z = read_varint(bytes)?;
        *x = from_grid_units(((z >> 1) as i64) ^ -((z & 1) as i64));
    }
    Some(Vec3Fix::new(c[0], c[1], c[2]))
}

impl LogInput for Touch {
    fn write(&self, out: &mut Vec<u8>) {
        match *self {
            Self::Grab { body, at } => {
                out.push(1);
                write_varint(u64::from(body), out);
                write_point(at, out);
            }
            Self::Drag { to } => {
                out.push(2);
                write_point(to, out);
            }
            Self::Release => out.push(3),
        }
    }

    fn read(bytes: &mut &[u8]) -> Option<Self> {
        let (&tag, after) = bytes.split_first()?;
        *bytes = after;
        match tag {
            1 => {
                let body = u32::try_from(read_varint(bytes)?).ok()?;
                Some(Self::Grab {
                    body,
                    at: read_point(bytes)?,
                })
            }
            2 => Some(Self::Drag {
                to: read_point(bytes)?,
            }),
            3 => Some(Self::Release),
            _ => None,
        }
    }
}

/// What a hand holds of a body: the point, and what the point was before it was taken
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grip {
    /// The body's own index of the point (a cloth's particle, a soft body's vertex)
    pub point: usize,
    inv_mass: Fix128,
    was_pinned: bool,
    /// Held where the hand is (a rigid hand), or only pulled toward it (a finite grip)
    pinned: bool,
}

/// A body a hand can take hold of
pub trait Grabbable {
    /// Take hold of the point nearest `at` within `reach`, and hold it at `at`; `None` when none is that near
    fn grab(&mut self, at: Vec3Fix, reach: Fix128) -> Option<Grip>;
    /// Take hold of the point nearest `at` within `reach` without holding it in place (a finite grip pulls it)
    fn take(&mut self, at: Vec3Fix, reach: Fix128) -> Option<Grip>;
    /// Carry the point held to `to` (a rigid hand)
    fn drag(&mut self, grip: &Grip, to: Vec3Fix);
    /// Where the point is
    fn point_at(&self, grip: &Grip) -> Vec3Fix;
    /// How fast the point moves (m/s)
    fn velocity_at(&self, grip: &Grip) -> Vec3Fix;
    /// Give the point the impulse `j` (N·s)
    fn push(&mut self, grip: &Grip, j: Vec3Fix);
    /// Let go of it: it is the body's again, as it was
    fn release(&mut self, grip: Grip);
}

impl Cloth {
    /// The particle nearest `at` within `reach` (the lowest index of equals)
    fn nearest_within(&self, at: Vec3Fix, reach: Fix128) -> Option<usize> {
        if reach <= Fix128::ZERO {
            return None;
        }
        let r2 = reach * reach;
        let mut best: Option<(usize, Fix128)> = None;
        for (i, p) in self.positions.iter().enumerate() {
            let d = *p - at;
            let d2 = d.dot(d);
            if d2 <= r2 && best.is_none_or(|(_, b)| d2 < b) {
                best = Some((i, d2));
            }
        }
        best.map(|(i, _)| i)
    }
}

impl Grabbable for Cloth {
    fn grab(&mut self, at: Vec3Fix, reach: Fix128) -> Option<Grip> {
        let point = self.nearest_within(at, reach)?;
        let grip = Grip {
            point,
            inv_mass: self.inv_masses[point],
            was_pinned: self.pinned.contains(&point),
            pinned: true,
        };
        self.pin(point);
        self.drag(&grip, at);
        Some(grip)
    }

    fn take(&mut self, at: Vec3Fix, reach: Fix128) -> Option<Grip> {
        let point = self.nearest_within(at, reach)?;
        Some(Grip {
            point,
            inv_mass: self.inv_masses[point],
            was_pinned: self.pinned.contains(&point),
            pinned: false,
        })
    }

    fn drag(&mut self, grip: &Grip, to: Vec3Fix) {
        let i = grip.point;
        self.positions[i] = to;
        self.prev_positions[i] = to;
        self.velocities[i] = Vec3Fix::ZERO;
    }

    fn point_at(&self, grip: &Grip) -> Vec3Fix {
        self.positions[grip.point]
    }

    fn velocity_at(&self, grip: &Grip) -> Vec3Fix {
        self.velocities[grip.point]
    }

    fn push(&mut self, grip: &Grip, j: Vec3Fix) {
        let i = grip.point;
        self.velocities[i] = self.velocities[i] + j * self.inv_masses[i];
    }

    fn release(&mut self, grip: Grip) {
        if !grip.pinned || grip.was_pinned {
            return;
        }
        self.inv_masses[grip.point] = grip.inv_mass;
        self.pinned.retain(|&j| j != grip.point);
    }
}

/// A hand among bodies: what it holds, how far it reaches, and how it holds
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hand {
    reach: Fix128,
    /// A finite grip: its stiffness (N/m) and its strength (N); none for a rigid hand
    grip: Option<(Fix128, Fix128)>,
    held: Option<(u32, Grip)>,
    /// Where the hand is, while it holds with a finite grip
    at: Option<Vec3Fix>,
}

impl Hand {
    /// A rigid hand that takes a point at most `reach` from where it touches and holds it where the hand is
    #[must_use]
    pub fn new(reach: Fix128) -> Self {
        Self {
            reach,
            grip: None,
            held: None,
            at: None,
        }
    }

    /// A hand with a finite grip: it pulls the point it takes toward itself with a spring of `stiffness` (N/m), and
    /// slips when the pull would exceed `strength` (N). A grip of no stiffness or no strength takes nothing
    #[must_use]
    pub fn with_grip(reach: Fix128, stiffness: Fix128, strength: Fix128) -> Self {
        Self {
            reach,
            grip: Some((stiffness, strength)),
            held: None,
            at: None,
        }
    }

    /// Do `touch` to `bodies`. Whether it did anything: a grab out of reach or of no such body, or a drag or a release
    /// with nothing held, does nothing. A grab lets go of what was held first
    pub fn apply<B: Grabbable>(&mut self, bodies: &mut [B], touch: &Touch) -> bool {
        match *touch {
            Touch::Grab { body, at } => {
                self.let_go(bodies);
                if self
                    .grip
                    .is_some_and(|(k, f)| k <= Fix128::ZERO || f <= Fix128::ZERO)
                {
                    return false;
                }
                let Some(b) = usize::try_from(body).ok().and_then(|i| bodies.get_mut(i)) else {
                    return false;
                };
                let taken = if self.grip.is_some() {
                    b.take(at, self.reach)
                } else {
                    b.grab(at, self.reach)
                };
                match taken {
                    Some(grip) => {
                        self.held = Some((body, grip));
                        self.at = Some(at);
                        true
                    }
                    None => false,
                }
            }
            Touch::Drag { to } => match self.held {
                Some((body, grip)) => {
                    self.at = Some(to);
                    if self.grip.is_none() {
                        bodies[body as usize].drag(&grip, to);
                    }
                    true
                }
                None => false,
            },
            Touch::Release => self.let_go(bodies),
        }
    }

    /// A finite grip's pull for a step of `dt` (s), before the bodies step: the backward Euler impulse
    /// `J = k_g dt (h − x − dt v) / (1 + k_g w dt²)` on the point held, or, when that pull would exceed the grip's strength, the grip slips and lets go. Whether it still holds
    pub fn pull<B: Grabbable>(&mut self, bodies: &mut [B], dt: Fix128) -> bool {
        let (Some((k, fmax)), Some((body, grip)), Some(at)) = (self.grip, self.held, self.at)
        else {
            return self.held.is_some();
        };
        let Some(b) = bodies.get_mut(body as usize) else {
            return false;
        };
        let x = b.point_at(&grip);
        // the grip's pull where the point is: past its strength the grip slips
        let f = (at - x) * k;
        if f.dot(f) > fmax * fmax {
            self.let_go(bodies);
            return false;
        }
        // the spring's backward Euler impulse: J = k dt (h − x − dt v) / (1 + k w dt²), the pull at where the point
        // will be after the step, so any stiffness, mass and step are stable
        let w = grip.inv_mass;
        let j = (at - x - b.velocity_at(&grip) * dt) * (k * dt) / (Fix128::ONE + k * w * dt * dt);
        b.push(&grip, j);
        true
    }

    fn let_go<B: Grabbable>(&mut self, bodies: &mut [B]) -> bool {
        self.at = None;
        match self.held.take() {
            Some((body, grip)) => {
                if let Some(b) = bodies.get_mut(body as usize) {
                    b.release(grip);
                }
                true
            }
            None => false,
        }
    }

    /// The body and the point held
    #[must_use]
    pub fn held(&self) -> Option<(u32, usize)> {
        self.held.map(|(b, g)| (b, g.point))
    }

    /// The hand's state as bytes, for a digest: what it holds and what the point was before
    #[must_use]
    pub fn state_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&self.reach.hi.to_le_bytes());
        out.extend_from_slice(&self.reach.lo.to_le_bytes());
        match self.grip {
            None => out.push(0),
            Some((k, f)) => {
                out.push(1);
                for x in [k, f] {
                    out.extend_from_slice(&x.hi.to_le_bytes());
                    out.extend_from_slice(&x.lo.to_le_bytes());
                }
            }
        }
        if let Some(h) = self.at {
            for x in [h.x, h.y, h.z] {
                out.extend_from_slice(&x.hi.to_le_bytes());
                out.extend_from_slice(&x.lo.to_le_bytes());
            }
        }
        match self.held {
            None => out.push(0),
            Some((body, g)) => {
                out.push(1);
                out.extend_from_slice(&body.to_le_bytes());
                out.extend_from_slice(&(g.point as u64).to_le_bytes());
                out.extend_from_slice(&g.inv_mass.hi.to_le_bytes());
                out.extend_from_slice(&g.inv_mass.lo.to_le_bytes());
                out.push(u8::from(g.was_pinned));
                out.push(u8::from(g.pinned));
            }
        }
        out
    }
}
