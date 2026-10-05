//! Two-body problem: Kepler's equation, classical orbital elements and the
//! closed-form laws of an elliptic orbit.
//!
//! A body of negligible mass moving under the gravity of a point mass with
//! gravitational parameter `μ = G·M` follows a conic section. This module
//! covers the **elliptic** case (`0 ≤ e < 1`, `a > 0`):
//!
//! | law | function |
//! |-----|----------|
//! | Kepler's equation `M = E − e·sin E` | [`solve_kepler`] |
//! | anomaly conversions `ν ↔ E ↔ M` | [`true_from_eccentric_anomaly`], [`eccentric_from_true_anomaly`], [`mean_from_true_anomaly`], [`true_from_mean_anomaly`] |
//! | Kepler's third law `T = 2π·√(a³/μ)` | [`orbital_period`], [`mean_motion`] |
//! | vis-viva `v² = μ·(2/r − 1/a)` | [`vis_viva_speed`] |
//! | energy `ε = −μ/(2a)`, angular momentum `h = √(μ·a·(1−e²))` | [`specific_orbital_energy`], [`specific_angular_momentum`] |
//! | elements ↔ state vector | [`OrbitalElements::to_state`], [`OrbitalElements::from_state`] |
//! | propagation in time | [`OrbitalElements::propagate`] |
//! | secular J2 drift of the node and the periapsis | [`j2_raan_rate`], [`j2_arg_periapsis_rate`] |
//!
//! Parabolic and hyperbolic orbits (`e ≥ 1`, non-negative energy) are
//! **rejected** with [`KeplerError::EccentricityOutOfRange`] /
//! [`KeplerError::NotElliptic`] rather than solved: the hyperbolic Kepler
//! equation `M = e·sinh F − F` needs `sinh`, a different initial guess and a
//! different state ↔ element map, and returning an elliptic answer for them
//! would be silently wrong.
//!
//! The N-body counterpart (mutual gravity of many bodies, integrated in time)
//! is [`crate::nbody`]; a fixed attractor acting on `PhysicsWorld` bodies is
//! [`crate::force::ForceField::Point`].
//!
//! # Units and range
//!
//! The functions are unit-agnostic: any consistent length / time unit works.
//! Values are [`Fix128`] (64 integer bits), so every intermediate product must
//! stay below about `9.2·10¹⁸`. The widest products are `|r × v|²` and
//! `|r|·|v|²` in [`OrbitalElements::from_state`]. For Earth orbits use km and s
//! (`μ ≈ 3.986·10⁵ km³/s²`, `|h|² ≈ 3·10⁹`) or canonical units (`μ = 1`); SI
//! metres overflow `|h|²` already in low Earth orbit (`≈ 2.8·10²¹ m⁴/s²`).
//! [`orbital_period`] and [`specific_angular_momentum`] are written as
//! `a·√(a/μ)` and `√μ·√(a(1−e²))` so that they do not form `a³` or `μ·a`.
//!
//! # Determinism
//!
//! Everything is [`Fix128`] arithmetic: `sin` / `cos` / `atan2` are the
//! crate's CORDIC (48 iterations, about `2⁻⁴⁸` absolute error) and `sqrt` is
//! the integer square root, so results are bit-identical on every platform.
//! The Newton iteration of [`solve_kepler`] stops on a test evaluated in
//! integer arithmetic, so its iteration count is a function of the input bits
//! only.
//!
//! # References
//!
//! - D. A. Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed.,
//!   §2.2 (Kepler's equation), §2.6 (COE2RV / RV2COE), §9.6 (J2 secular rates).
//! - J. M. A. Danby, *Fundamentals of Celestial Mechanics*, §6.6 (the starting
//!   value `E₀ = M + 0.85·e·sign(sin M)`).

use crate::math::{Fix128, Vec3Fix};

/// Most Newton steps [`solve_kepler`] takes before it reports
/// [`KeplerError::NotConverged`].
///
/// Every step that would leave the bracket `[M − e, M + e]` (which always
/// contains the root, since `|E − M| = e·|sin E| ≤ e`) is replaced by a
/// bisection of the bracket, so 64 steps reduce a bracket of width `≤ 2` below
/// the `2⁻⁶⁴` resolution of [`Fix128`] even when no Newton step is accepted.
pub const KEPLER_MAX_ITERATIONS: u32 = 64;

/// [`solve_kepler`] stops when a Newton step moves `E` by at most this much:
/// `2⁻⁴⁴ ≈ 5.7·10⁻¹⁴` rad.
///
/// Newton converges quadratically, so the error left after a step of size
/// `δ` is about `δ²·e/(2(1−e))`, far below `δ`. The floor of the whole
/// computation is the CORDIC `sin` / `cos` (about `2⁻⁴⁸`); a tolerance 16×
/// above that floor keeps the stopping test from chasing rounding noise.
pub const KEPLER_TOLERANCE: Fix128 = Fix128::from_raw(0, 1u64 << 20);

/// Below this, [`OrbitalElements::from_state`] treats the orbit as circular
/// (`e`) or equatorial (`sin i`): `2⁻³⁰ ≈ 9.3·10⁻¹⁰`.
///
/// For a circular orbit the argument of periapsis `ω` is undefined; for an
/// equatorial orbit the right ascension of the ascending node `Ω` is. See
/// [`OrbitalElements::from_state`] for the convention used in each case. The
/// value sits well above the `≈ 10⁻¹⁴` noise a round trip through
/// [`OrbitalElements::to_state`] leaves in `e` and `sin i`, and well below any
/// eccentricity or inclination that is meant as one.
pub const DEGENERACY_TOLERANCE: Fix128 = Fix128::from_raw(0, 1u64 << 34);

/// Why a function of this module refused its input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum KeplerError {
    /// The gravitational parameter `μ` is not strictly positive.
    NonPositiveGravitationalParameter,
    /// The semi-major axis `a` is not strictly positive.
    NonPositiveSemiMajorAxis,
    /// The eccentricity is outside `[0, 1)`: negative, or a parabolic /
    /// hyperbolic orbit (not supported, see the module documentation).
    EccentricityOutOfRange,
    /// The inclination is outside `[0, π]`.
    InclinationOutOfRange,
    /// A radius is not strictly positive.
    NonPositiveRadius,
    /// No elliptic orbit with this semi-major axis reaches the radius
    /// (`r ≥ 2a`, where `v²` from vis-viva would be `≤ 0`).
    RadiusUnreachable,
    /// The position vector is zero.
    ZeroPosition,
    /// `r × v = 0`: a radial (rectilinear) trajectory, which has no orbital
    /// plane.
    RadialTrajectory,
    /// The specific energy `v²/2 − μ/r` is not negative: the state is on a
    /// parabolic or hyperbolic trajectory.
    NotElliptic,
    /// The equatorial radius of [`j2_raan_rate`] / [`j2_arg_periapsis_rate`]
    /// is not strictly positive.
    NonPositiveEquatorialRadius,
    /// [`solve_kepler`] did not meet [`KEPLER_TOLERANCE`] in
    /// [`KEPLER_MAX_ITERATIONS`] steps (not observed; reported instead of
    /// returning an unconverged value).
    NotConverged,
}

impl core::fmt::Display for KeplerError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let msg = match self {
            Self::NonPositiveGravitationalParameter => {
                "the gravitational parameter must be positive"
            }
            Self::NonPositiveSemiMajorAxis => "the semi-major axis must be positive",
            Self::EccentricityOutOfRange => {
                "the eccentricity must be in [0, 1) (elliptic orbits only)"
            }
            Self::InclinationOutOfRange => "the inclination must be in [0, pi]",
            Self::NonPositiveRadius => "the radius must be positive",
            Self::RadiusUnreachable => {
                "no elliptic orbit with this semi-major axis reaches the radius"
            }
            Self::ZeroPosition => "the position vector is zero",
            Self::RadialTrajectory => "r x v = 0: a radial trajectory has no orbital plane",
            Self::NotElliptic => "the specific energy is not negative: not an elliptic orbit",
            Self::NonPositiveEquatorialRadius => "the equatorial radius must be positive",
            Self::NotConverged => "Kepler's equation did not converge",
        };
        f.write_str(msg)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for KeplerError {}

fn check_mu(mu: Fix128) -> Result<(), KeplerError> {
    if mu > Fix128::ZERO {
        Ok(())
    } else {
        Err(KeplerError::NonPositiveGravitationalParameter)
    }
}

fn check_a(a: Fix128) -> Result<(), KeplerError> {
    if a > Fix128::ZERO {
        Ok(())
    } else {
        Err(KeplerError::NonPositiveSemiMajorAxis)
    }
}

fn check_e(e: Fix128) -> Result<(), KeplerError> {
    if e.is_negative() || e >= Fix128::ONE {
        Err(KeplerError::EccentricityOutOfRange)
    } else {
        Ok(())
    }
}

/// `x − 2π·k` in `[0, 2π)`.
fn wrap_two_pi(x: Fix128) -> Fix128 {
    let k = (x / Fix128::TWO_PI).floor();
    let mut r = x - Fix128::TWO_PI * k;
    // `TWO_PI * k` is floored, so `r` can land one unit outside either end.
    if r.is_negative() {
        r = r + Fix128::TWO_PI;
    }
    if r >= Fix128::TWO_PI {
        r = r - Fix128::TWO_PI;
    }
    r
}

/// The integer `k` with `x − 2π·k` in `[−π, π)`.
fn turns_to_pi_range(x: Fix128) -> Fix128 {
    ((x + Fix128::PI) / Fix128::TWO_PI).floor()
}

/// Solve Kepler's equation `M = E − e·sin E` for the eccentric anomaly `E`.
///
/// Valid for `0 ≤ e < 1` and any mean anomaly `M` (radians). The result
/// satisfies the equation for the `M` given, not only modulo `2π`: `M` is
/// reduced to `[−π, π)` as `M = M′ + 2πk`, the root `E′` of the reduced
/// equation is found and `E′ + 2πk` is returned.
///
/// # Method
///
/// Newton's method on `f(E) = E − e·sin E − M′`, `f′(E) = 1 − e·cos E ≥ 1 − e`,
/// from Danby's starting value `E₀ = M′ + 0.85·e·sign(sin M′)`. `f` is strictly
/// increasing and its root lies in `[M′ − e, M′ + e]`; the bracket is narrowed
/// with the sign of `f` at every iterate, and a Newton step that would leave
/// it is replaced by its midpoint (safeguarded Newton), which makes the
/// iteration converge for every `e < 1`, including `e → 1` near `M = 0` where
/// plain Newton overshoots.
///
/// Stops when a step moves `E` by at most [`KEPLER_TOLERANCE`], when `f` is
/// exactly zero, or when the bracket is narrower than the tolerance. `e = 0`
/// returns `M` exactly.
///
/// # Errors
///
/// [`KeplerError::EccentricityOutOfRange`] for `e < 0` or `e ≥ 1`;
/// [`KeplerError::NotConverged`] after [`KEPLER_MAX_ITERATIONS`] steps without
/// meeting the tolerance (see that constant for why it does not happen).
///
/// # Examples
///
/// ```
/// use alice_physics::kepler::solve_kepler;
/// use alice_physics::Fix128;
///
/// // Vallado Example 2-1: M = 235.4°, e = 0.4 → E = 220.512 074 767 522°
/// let m = Fix128::from_f64(235.4_f64.to_radians());
/// let e = solve_kepler(m, Fix128::from_ratio(2, 5)).unwrap();
/// assert!((e.to_f64().to_degrees() - 220.512_074_767_522).abs() < 1e-9);
/// ```
pub fn solve_kepler(mean_anomaly: Fix128, eccentricity: Fix128) -> Result<Fix128, KeplerError> {
    check_e(eccentricity)?;
    let e = eccentricity;
    let k = turns_to_pi_range(mean_anomaly);
    let turns = Fix128::TWO_PI * k;
    let m = mean_anomaly - turns;

    let mut lo = m - e;
    let mut hi = m + e;
    let danby = Fix128::from_ratio(85, 100) * e;
    let mut ecc = if m.sin().is_negative() {
        m - danby
    } else if m.is_zero() {
        m
    } else {
        m + danby
    };

    for _ in 0..KEPLER_MAX_ITERATIONS {
        let (s, c) = ecc.sin_cos();
        let f = ecc - e * s - m;
        if f.is_zero() {
            return Ok(ecc + turns);
        }
        if f.is_negative() {
            lo = ecc;
        } else {
            hi = ecc;
        }
        let fp = Fix128::ONE - e * c;
        let mut next = ecc - f / fp;
        if next <= lo || next >= hi {
            next = (lo + hi).half();
        }
        let step = (next - ecc).abs();
        ecc = next;
        if step <= KEPLER_TOLERANCE || hi - lo <= KEPLER_TOLERANCE {
            return Ok(ecc + turns);
        }
    }
    Err(KeplerError::NotConverged)
}

/// True anomaly `ν` from the eccentric anomaly `E`:
/// `ν = 2·atan2(√(1+e)·sin(E/2), √(1−e)·cos(E/2))`, in `(−π, π]` for `E` in
/// `(−π, π]` (the half-angle form keeps the quadrant and is regular at
/// `E = π`).
///
/// # Errors
///
/// [`KeplerError::EccentricityOutOfRange`] for `e` outside `[0, 1)`.
pub fn true_from_eccentric_anomaly(
    eccentric_anomaly: Fix128,
    eccentricity: Fix128,
) -> Result<Fix128, KeplerError> {
    check_e(eccentricity)?;
    let (s, c) = eccentric_anomaly.half().sin_cos();
    let y = (Fix128::ONE + eccentricity).sqrt() * s;
    let x = (Fix128::ONE - eccentricity).sqrt() * c;
    Ok(Fix128::atan2(y, x).double())
}

/// Eccentric anomaly `E` from the true anomaly `ν`:
/// `E = 2·atan2(√(1−e)·sin(ν/2), √(1+e)·cos(ν/2))`, the inverse of
/// [`true_from_eccentric_anomaly`].
///
/// # Errors
///
/// [`KeplerError::EccentricityOutOfRange`] for `e` outside `[0, 1)`.
pub fn eccentric_from_true_anomaly(
    true_anomaly: Fix128,
    eccentricity: Fix128,
) -> Result<Fix128, KeplerError> {
    check_e(eccentricity)?;
    let (s, c) = true_anomaly.half().sin_cos();
    let y = (Fix128::ONE - eccentricity).sqrt() * s;
    let x = (Fix128::ONE + eccentricity).sqrt() * c;
    Ok(Fix128::atan2(y, x).double())
}

/// Mean anomaly `M = E − e·sin E` of the true anomaly `ν`, in `[0, 2π)`.
///
/// # Errors
///
/// [`KeplerError::EccentricityOutOfRange`] for `e` outside `[0, 1)`.
pub fn mean_from_true_anomaly(
    true_anomaly: Fix128,
    eccentricity: Fix128,
) -> Result<Fix128, KeplerError> {
    let ecc = eccentric_from_true_anomaly(true_anomaly, eccentricity)?;
    Ok(wrap_two_pi(ecc - eccentricity * ecc.sin()))
}

/// True anomaly `ν` in `[0, 2π)` of the mean anomaly `M` (solves
/// [`solve_kepler`], then [`true_from_eccentric_anomaly`]).
///
/// # Errors
///
/// As [`solve_kepler`].
pub fn true_from_mean_anomaly(
    mean_anomaly: Fix128,
    eccentricity: Fix128,
) -> Result<Fix128, KeplerError> {
    let ecc = solve_kepler(wrap_two_pi(mean_anomaly), eccentricity)?;
    Ok(wrap_two_pi(true_from_eccentric_anomaly(ecc, eccentricity)?))
}

/// Mean motion `n = √(μ/a³)` (rad per time unit), computed as `√(μ/a)/a`.
///
/// # Errors
///
/// [`KeplerError::NonPositiveGravitationalParameter`] for `μ ≤ 0`,
/// [`KeplerError::NonPositiveSemiMajorAxis`] for `a ≤ 0`.
pub fn mean_motion(mu: Fix128, semi_major_axis: Fix128) -> Result<Fix128, KeplerError> {
    check_mu(mu)?;
    check_a(semi_major_axis)?;
    Ok((mu / semi_major_axis).sqrt() / semi_major_axis)
}

/// Kepler's third law: orbital period `T = 2π·√(a³/μ)`, computed as
/// `2π·a·√(a/μ)` so that `a³` is never formed.
///
/// # Errors
///
/// As [`mean_motion`].
pub fn orbital_period(mu: Fix128, semi_major_axis: Fix128) -> Result<Fix128, KeplerError> {
    check_mu(mu)?;
    check_a(semi_major_axis)?;
    Ok(Fix128::TWO_PI * semi_major_axis * (semi_major_axis / mu).sqrt())
}

/// Vis-viva: the speed `v = √(μ·(2/r − 1/a))` at radius `r` on an elliptic
/// orbit of semi-major axis `a`.
///
/// # Errors
///
/// [`KeplerError::NonPositiveGravitationalParameter`] for `μ ≤ 0`,
/// [`KeplerError::NonPositiveSemiMajorAxis`] for `a ≤ 0`,
/// [`KeplerError::NonPositiveRadius`] for `r ≤ 0`,
/// [`KeplerError::RadiusUnreachable`] for `r ≥ 2a` (every elliptic orbit stays
/// inside `r ≤ a(1+e) < 2a`).
pub fn vis_viva_speed(
    mu: Fix128,
    radius: Fix128,
    semi_major_axis: Fix128,
) -> Result<Fix128, KeplerError> {
    check_mu(mu)?;
    check_a(semi_major_axis)?;
    if radius <= Fix128::ZERO {
        return Err(KeplerError::NonPositiveRadius);
    }
    let bracket = Fix128::from_int(2) / radius - Fix128::ONE / semi_major_axis;
    if bracket <= Fix128::ZERO {
        return Err(KeplerError::RadiusUnreachable);
    }
    Ok((mu * bracket).sqrt())
}

/// Specific orbital energy `ε = v²/2 − μ/r = −μ/(2a)` of an elliptic orbit.
///
/// # Errors
///
/// As [`mean_motion`].
pub fn specific_orbital_energy(mu: Fix128, semi_major_axis: Fix128) -> Result<Fix128, KeplerError> {
    check_mu(mu)?;
    check_a(semi_major_axis)?;
    Ok(-(mu / semi_major_axis.double()))
}

/// Specific angular momentum `h = |r × v| = √(μ·a·(1 − e²))`, computed as
/// `√μ·√(a(1−e²))` so that `μ·a` is never formed.
///
/// # Errors
///
/// As [`mean_motion`], plus [`KeplerError::EccentricityOutOfRange`] for `e`
/// outside `[0, 1)`.
pub fn specific_angular_momentum(
    mu: Fix128,
    semi_major_axis: Fix128,
    eccentricity: Fix128,
) -> Result<Fix128, KeplerError> {
    check_mu(mu)?;
    check_a(semi_major_axis)?;
    check_e(eccentricity)?;
    let p = semi_major_axis * (Fix128::ONE - eccentricity * eccentricity);
    Ok(mu.sqrt() * p.sqrt())
}

/// Secular J2 drift of the right ascension of the ascending node:
///
/// ```text
/// dΩ/dt = −(3/2)·n·J2·(R/p)²·cos i,   n = √(μ/a³),  p = a(1 − e²)
/// ```
///
/// (rad per time unit; Vallado §9.6). Negative (westward regression) for
/// prograde orbits, zero for polar orbits, positive for retrograde orbits —
/// the drift a sun-synchronous orbit tunes `i` for. `j2` (dimensionless) and
/// `equatorial_radius` describe the central body and are the caller's values;
/// this crate carries no body constants. Only the secular (orbit-averaged)
/// first-order term: short-periodic J2 terms and higher zonal harmonics are not
/// modelled, and [`OrbitalElements::propagate`] does not apply this drift.
///
/// # Errors
///
/// As [`specific_angular_momentum`], plus
/// [`KeplerError::NonPositiveEquatorialRadius`] for `R ≤ 0`.
pub fn j2_raan_rate(
    mu: Fix128,
    semi_major_axis: Fix128,
    eccentricity: Fix128,
    inclination: Fix128,
    j2: Fix128,
    equatorial_radius: Fix128,
) -> Result<Fix128, KeplerError> {
    let base = j2_base(mu, semi_major_axis, eccentricity, j2, equatorial_radius)?;
    let three_halves = Fix128::from_ratio(3, 2);
    Ok(-(three_halves * base * inclination.cos()))
}

/// Secular J2 drift of the argument of periapsis:
///
/// ```text
/// dω/dt = (3/4)·n·J2·(R/p)²·(5·cos² i − 1)
/// ```
///
/// (rad per time unit; Vallado §9.6). Zero at the critical inclination
/// `cos² i = 1/5` (63.43° / 116.57°), used by frozen-perigee orbits. Same
/// scope and inputs as [`j2_raan_rate`].
///
/// # Errors
///
/// As [`j2_raan_rate`].
pub fn j2_arg_periapsis_rate(
    mu: Fix128,
    semi_major_axis: Fix128,
    eccentricity: Fix128,
    inclination: Fix128,
    j2: Fix128,
    equatorial_radius: Fix128,
) -> Result<Fix128, KeplerError> {
    let base = j2_base(mu, semi_major_axis, eccentricity, j2, equatorial_radius)?;
    let c = inclination.cos();
    let three_quarters = Fix128::from_ratio(3, 4);
    Ok(three_quarters * base * (Fix128::from_int(5) * c * c - Fix128::ONE))
}

/// `n·J2·(R/p)²`, shared by the two J2 rates.
fn j2_base(
    mu: Fix128,
    semi_major_axis: Fix128,
    eccentricity: Fix128,
    j2: Fix128,
    equatorial_radius: Fix128,
) -> Result<Fix128, KeplerError> {
    check_e(eccentricity)?;
    let n = mean_motion(mu, semi_major_axis)?;
    if equatorial_radius <= Fix128::ZERO {
        return Err(KeplerError::NonPositiveEquatorialRadius);
    }
    let p = semi_major_axis * (Fix128::ONE - eccentricity * eccentricity);
    let ratio = equatorial_radius / p;
    Ok(n * j2 * ratio * ratio)
}

/// Position and velocity of the orbiting body relative to the central body,
/// in the inertial frame the elements are measured in (`z` is the pole the
/// inclination is measured from, `x` the reference direction of `Ω`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateVector {
    /// Position `r`
    pub position: Vec3Fix,
    /// Velocity `v`
    pub velocity: Vec3Fix,
}

/// Classical (Keplerian) orbital elements of an elliptic orbit.
///
/// Angles are in radians. [`Self::new`] validates the ranges; the fields are
/// public for reading and for callers that keep them valid themselves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OrbitalElements {
    /// Semi-major axis `a > 0`
    pub semi_major_axis: Fix128,
    /// Eccentricity `0 ≤ e < 1`
    pub eccentricity: Fix128,
    /// Inclination `i` in `[0, π]`, measured from the `z` axis to the angular
    /// momentum vector
    pub inclination: Fix128,
    /// Right ascension of the ascending node `Ω` (from the `x` axis, about `z`)
    pub raan: Fix128,
    /// Argument of periapsis `ω` (from the ascending node, in the direction of
    /// motion)
    pub arg_periapsis: Fix128,
    /// True anomaly `ν` (from periapsis, in the direction of motion)
    pub true_anomaly: Fix128,
}

impl OrbitalElements {
    /// Validated elements; `Ω`, `ω`, `ν` are wrapped to `[0, 2π)`.
    ///
    /// # Errors
    ///
    /// [`KeplerError::NonPositiveSemiMajorAxis`] for `a ≤ 0`,
    /// [`KeplerError::EccentricityOutOfRange`] for `e` outside `[0, 1)`,
    /// [`KeplerError::InclinationOutOfRange`] for `i` outside `[0, π]`.
    pub fn new(
        semi_major_axis: Fix128,
        eccentricity: Fix128,
        inclination: Fix128,
        raan: Fix128,
        arg_periapsis: Fix128,
        true_anomaly: Fix128,
    ) -> Result<Self, KeplerError> {
        check_a(semi_major_axis)?;
        check_e(eccentricity)?;
        if inclination.is_negative() || inclination > Fix128::PI {
            return Err(KeplerError::InclinationOutOfRange);
        }
        Ok(Self {
            semi_major_axis,
            eccentricity,
            inclination,
            raan: wrap_two_pi(raan),
            arg_periapsis: wrap_two_pi(arg_periapsis),
            true_anomaly: wrap_two_pi(true_anomaly),
        })
    }

    fn validate(&self) -> Result<(), KeplerError> {
        Self::new(
            self.semi_major_axis,
            self.eccentricity,
            self.inclination,
            self.raan,
            self.arg_periapsis,
            self.true_anomaly,
        )
        .map(|_| ())
    }

    /// Semi-latus rectum `p = a(1 − e²)`.
    #[must_use]
    pub fn semi_latus_rectum(&self) -> Fix128 {
        self.semi_major_axis * (Fix128::ONE - self.eccentricity * self.eccentricity)
    }

    /// Mean anomaly `M` in `[0, 2π)` of the current true anomaly.
    ///
    /// # Errors
    ///
    /// [`KeplerError::EccentricityOutOfRange`] if `e` was set outside `[0, 1)`.
    pub fn mean_anomaly(&self) -> Result<Fix128, KeplerError> {
        mean_from_true_anomaly(self.true_anomaly, self.eccentricity)
    }

    /// Orbital period under `μ` ([`orbital_period`]).
    ///
    /// # Errors
    ///
    /// As [`orbital_period`].
    pub fn period(&self, mu: Fix128) -> Result<Fix128, KeplerError> {
        orbital_period(mu, self.semi_major_axis)
    }

    /// State vector (COE2RV, Vallado Algorithm 10):
    ///
    /// ```text
    /// r_pqw = p/(1 + e·cos ν) · (cos ν, sin ν, 0)
    /// v_pqw = √(μ/p) · (−sin ν, e + cos ν, 0)
    /// r, v  = R₃(−Ω)·R₁(−i)·R₃(−ω) · (r_pqw, v_pqw)
    /// ```
    ///
    /// # Errors
    ///
    /// [`KeplerError::NonPositiveGravitationalParameter`] for `μ ≤ 0`, and the
    /// range errors of [`Self::new`] if a field was set out of range.
    pub fn to_state(&self, mu: Fix128) -> Result<StateVector, KeplerError> {
        check_mu(mu)?;
        self.validate()?;
        let e = self.eccentricity;
        let p = self.semi_latus_rectum();
        let (sn, cn) = self.true_anomaly.sin_cos();
        let radius = p / (Fix128::ONE + e * cn);
        let speed = (mu / p).sqrt();
        let r_pqw = (radius * cn, radius * sn);
        let v_pqw = (-(speed * sn), speed * (e + cn));

        let (so, co) = self.raan.sin_cos();
        let (si, ci) = self.inclination.sin_cos();
        let (sw, cw) = self.arg_periapsis.sin_cos();
        // Columns P and Q of the perifocal → inertial rotation.
        let p_axis = Vec3Fix::new(co * cw - so * sw * ci, so * cw + co * sw * ci, sw * si);
        let q_axis = Vec3Fix::new(-(co * sw) - so * cw * ci, co * cw * ci - so * sw, cw * si);
        Ok(StateVector {
            position: p_axis * r_pqw.0 + q_axis * r_pqw.1,
            velocity: p_axis * v_pqw.0 + q_axis * v_pqw.1,
        })
    }

    /// Elements of a state vector (RV2COE, Vallado Algorithm 9).
    ///
    /// ```text
    /// h = r × v,  n = ẑ × h,  e⃗ = ((v² − μ/r)·r − (r·v)·v)/μ,  a = −μ/(2ε)
    /// i = atan2(|n|, h_z),  Ω = atan2(n_y, n_x)
    /// ω = ∠(n, e⃗),  ν = ∠(e⃗, r)   (angles measured about ĥ)
    /// ```
    ///
    /// # Degenerate orbits
    ///
    /// With `e < DEGENERACY_TOLERANCE` (circular) `ω` is undefined; with
    /// `|n| < DEGENERACY_TOLERANCE·|h|` (`sin i` below it: equatorial,
    /// prograde or retrograde) `Ω` is undefined. Then:
    ///
    /// | case | `Ω` | `ω` | `ν` |
    /// |------|-----|-----|-----|
    /// | circular, inclined | `atan2(n_y, n_x)` | `0` | argument of latitude `u = ∠(n, r)` |
    /// | elliptic, equatorial | `0` | longitude of periapsis `∠(x̂, e⃗)` | `∠(e⃗, r)` |
    /// | circular, equatorial | `0` | `0` | true longitude `∠(x̂, r)` |
    ///
    /// With these conventions [`Self::to_state`] of the result reproduces the
    /// state. `e` is returned as computed (not snapped to `0`).
    ///
    /// # Errors
    ///
    /// [`KeplerError::NonPositiveGravitationalParameter`] for `μ ≤ 0`,
    /// [`KeplerError::ZeroPosition`] for `r = 0`,
    /// [`KeplerError::RadialTrajectory`] for `r × v = 0`,
    /// [`KeplerError::NotElliptic`] for non-negative energy (or a computed
    /// `e ≥ 1`).
    pub fn from_state(state: &StateVector, mu: Fix128) -> Result<Self, KeplerError> {
        check_mu(mu)?;
        let r_vec = state.position;
        let v_vec = state.velocity;
        let r = r_vec.length();
        if r.is_zero() {
            return Err(KeplerError::ZeroPosition);
        }
        let h_vec = r_vec.cross(v_vec);
        let h = h_vec.length();
        if h.is_zero() {
            return Err(KeplerError::RadialTrajectory);
        }
        let v2 = v_vec.length_squared();
        let mu_over_r = mu / r;
        let energy = v2.half() - mu_over_r;
        if !energy.is_negative() {
            return Err(KeplerError::NotElliptic);
        }
        let a = -(mu / energy.double());
        let e_vec = (r_vec * (v2 - mu_over_r) - v_vec * r_vec.dot(v_vec)) / mu;
        let e = e_vec.length();
        if e >= Fix128::ONE {
            return Err(KeplerError::NotElliptic);
        }
        let h_hat = h_vec / h;
        let n_vec = Vec3Fix::new(-h_vec.y, h_vec.x, Fix128::ZERO);
        let n = n_vec.length();
        let inclination = Fix128::atan2(n, h_vec.z);

        // Angle from `from` to `to` about ĥ.
        let angle = |from: Vec3Fix, to: Vec3Fix| -> Fix128 {
            wrap_two_pi(Fix128::atan2(h_hat.dot(from.cross(to)), from.dot(to)))
        };

        let circular = e < DEGENERACY_TOLERANCE;
        let equatorial = n < DEGENERACY_TOLERANCE * h;
        let (raan, arg_periapsis, true_anomaly) = match (circular, equatorial) {
            (false, false) => (
                wrap_two_pi(Fix128::atan2(n_vec.y, n_vec.x)),
                angle(n_vec, e_vec),
                angle(e_vec, r_vec),
            ),
            (true, false) => (
                wrap_two_pi(Fix128::atan2(n_vec.y, n_vec.x)),
                Fix128::ZERO,
                angle(n_vec, r_vec),
            ),
            (false, true) => (
                Fix128::ZERO,
                angle(Vec3Fix::UNIT_X, e_vec),
                angle(e_vec, r_vec),
            ),
            (true, true) => (Fix128::ZERO, Fix128::ZERO, angle(Vec3Fix::UNIT_X, r_vec)),
        };

        Ok(Self {
            semi_major_axis: a,
            eccentricity: e,
            inclination,
            raan,
            arg_periapsis,
            true_anomaly,
        })
    }

    /// The elements after time `dt` of unperturbed two-body motion: the mean
    /// anomaly advances by `n·dt` (`n` = [`mean_motion`]), Kepler's equation
    /// gives the new true anomaly, and `a, e, i, Ω, ω` are unchanged. `dt` may
    /// be negative (propagation backward in time).
    ///
    /// # Errors
    ///
    /// [`KeplerError::NonPositiveGravitationalParameter`] for `μ ≤ 0`, the
    /// range errors of [`Self::new`], and those of [`solve_kepler`].
    pub fn propagate(&self, mu: Fix128, dt: Fix128) -> Result<Self, KeplerError> {
        self.validate()?;
        let n = mean_motion(mu, self.semi_major_axis)?;
        let m0 = self.mean_anomaly()?;
        let m1 = wrap_two_pi(m0 + n * dt);
        Ok(Self {
            true_anomaly: true_from_mean_anomaly(m1, self.eccentricity)?,
            ..*self
        })
    }

    /// State vector after time `dt` ([`Self::propagate`] then
    /// [`Self::to_state`]).
    ///
    /// # Errors
    ///
    /// As [`Self::propagate`].
    pub fn state_at(&self, mu: Fix128, dt: Fix128) -> Result<StateVector, KeplerError> {
        self.propagate(mu, dt)?.to_state(mu)
    }
}
