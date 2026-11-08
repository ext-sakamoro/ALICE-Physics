//! A visitor's touch on a body: take hold of it at a point, carry that point, let go. The touches are inputs a log
//! carries ([`crate::input_log::InputLog`]), so a world touched, logged and replayed comes to the same state.
//!
//! MODEL: a grab takes the body's point nearest the touch within the hand's reach. A rigid hand ([`Hand::new`]) holds
//! it where the hand is (the point is pinned: it moves with the hand and nothing else moves it); a drag carries it;
//! letting go returns it to the body as it was (its mass, or pinned if it was pinned before). A hand with a finite
//! grip ([`Hand::with_grip`]) leaves the point the body's and pulls it toward the hand with a spring, `F = k_g (h − x)`
//! (N, with `h` the hand and `x` the point, m), applied as the impulse `F dt` before each step ([`Hand::pull`]); when
//! `|F|` would exceed the grip's strength `F_max` (N) the grip slips and the hand lets go. At rest against a body that
//! holds the point by `k_t`, the point sits at `x = k_g d / (k_g + k_t)` for a hand `d` away, and a slow drag lets go at
//! `d = F_max (k_g + k_t) / (k_g k_t)` (`tests/analytic_touch.rs`). What the body does then is its own law: a
//! cloth held to its rest shape by its tether ([`Cloth::set_rest_tether`]) springs back as a damped oscillator. The
//! body says what a point is ([`Grabbable`]): for a cloth, one of its particles; for a soft body, a surface vertex.
//!
//! Positions travel on a grid of 2⁻¹⁶ m (15 µm), so a log is short and what is read back is what was played.

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

    /// A finite grip's pull for a step of `dt` (s), before the bodies step: the impulse `k_g (h − x) dt` on the point
    /// held, or, when that pull would exceed the grip's strength, the grip slips and lets go. Whether it still holds
    pub fn pull<B: Grabbable>(&mut self, bodies: &mut [B], dt: Fix128) -> bool {
        let (Some((k, fmax)), Some((body, grip)), Some(at)) = (self.grip, self.held, self.at)
        else {
            return self.held.is_some();
        };
        let Some(b) = bodies.get_mut(body as usize) else {
            return false;
        };
        let f = (at - b.point_at(&grip)) * k;
        if f.dot(f) > fmax * fmax {
            self.let_go(bodies);
            return false;
        }
        b.push(&grip, f * dt);
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
