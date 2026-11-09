//! Soft letters: a soft body cut from a planar outline, living inside
//! [`PhysicsWorld`](crate::solver::PhysicsWorld).
//!
//! A [`SoftBodyDesc`] names a plane (`origin`, `u`, `v`), closed contours in that plane
//! (outer boundaries and holes, filled by the even-odd rule) and a lattice `spacing`.
//! [`PhysicsWorld::add_soft_body`](crate::solver::PhysicsWorld::add_soft_body) keeps the
//! lattice points that are inside the fill or within `spacing / 2` of the outline, keeps
//! a lattice cell's two triangles when all three corners are kept, and drops the points
//! no triangle uses. The result is a [`Cloth`] whose rest shape is the lattice, tethered
//! back to it by [`Cloth::set_rest_tether`]. There is no triangulation: strokes thinner
//! than the spacing keep a one-point-wide line of particles and so no triangle; an
//! outline that yields no triangle at all is refused ([`SoftBodyError::NoTriangles`]).
//!
//! Grabs: [`PhysicsWorld::soft_grab`](crate::solver::PhysicsWorld::soft_grab) pins the
//! nearest free particle within reach to a pointer and returns a [`GrabId`] (unique in
//! the world); any number of grabs may hold one body. Each world step places every
//! grabbed particle at its target (position, previous position; velocity zero) before
//! the cloth steps. Releasing a grab gives the particle back its mass and the pointer's
//! velocity over the last step, `(target applied at the last step − target applied at
//! the step before) / dt`, so a moving pointer throws the letter; the explicit form sets
//! the velocity given. The tether then pulls the letter back to its rest shape and the
//! exact sleep of [`Cloth::set_sleep`] stops the work once a step changes no bit.
//!
//! Soft bodies do not collide with rigid or static bodies, and a body's gravity is its
//! own [`ClothConfig::gravity`], not the world's. Every quantity is `Fix128`: the same
//! inputs replay bit for bit on every target.

use crate::cloth::{Cloth, ClothConfig};
use crate::math::{Fix128, Vec3Fix};
#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

/// What a soft body is cut from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SoftBodyDesc {
    /// Plane origin: lattice point `(x, y)` is placed at `origin + u·x + v·y`.
    pub origin: Vec3Fix,
    /// Plane axis of the contour's first coordinate.
    pub u: Vec3Fix,
    /// Plane axis of the contour's second coordinate.
    pub v: Vec3Fix,
    /// Closed polygons in plane coordinates (last point joins the first), any
    /// winding; the fill is the even-odd rule over all of them.
    pub contours: Vec<Vec<[Fix128; 2]>>,
    /// Lattice spacing in plane units (`> 0`).
    pub spacing: Fix128,
    /// Mass of every particle (`> 0`).
    pub mass_per_particle: Fix128,
    /// The cloth's configuration (iterations, substeps, gravity, damping, compliances).
    pub config: ClothConfig,
    /// `(stiffness, damping)` of the rest tether ([`Cloth::set_rest_tether`]), or none.
    pub tether: Option<(Fix128, Fix128)>,
}

/// Why a [`SoftBodyDesc`] was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SoftBodyError {
    /// No contour, or every contour has fewer than three points.
    EmptyOutline,
    /// `spacing` is not positive.
    SpacingNotPositive,
    /// `mass_per_particle` is not positive.
    MassNotPositive,
    /// The lattice keeps no triangle (the outline is thinner than the spacing).
    NoTriangles,
}

impl core::fmt::Display for SoftBodyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::EmptyOutline => "empty outline",
            Self::SpacingNotPositive => "spacing is not positive",
            Self::MassNotPositive => "mass per particle is not positive",
            Self::NoTriangles => "the lattice keeps no triangle",
        })
    }
}

#[cfg(feature = "std")]
impl std::error::Error for SoftBodyError {}

/// Handle of one grab, unique within its world.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GrabId(pub u32);

/// One pointer holding one particle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grab {
    /// The handle [`PhysicsWorld::soft_grab`](crate::solver::PhysicsWorld::soft_grab) returned.
    pub id: GrabId,
    /// The particle held.
    pub particle: usize,
    /// Where the next step places the particle.
    pub target: Vec3Fix,
    /// Targets applied at the last step and at the step before (the throw velocity).
    pub(crate) applied: [Vec3Fix; 2],
}

/// A soft body inside a world: a [`Cloth`] plus the grabs holding it.
pub struct SoftBody {
    cloth: Cloth,
    grabs: Vec<Grab>,
    /// `dt` of the last step (the throw velocity divides by it); zero before any step.
    last_dt: Fix128,
}

impl SoftBody {
    /// Build the lattice of `desc` (see the module doc).
    ///
    /// # Errors
    ///
    /// [`SoftBodyError`] when the outline, spacing or mass is unusable, or no
    /// triangle is kept.
    pub fn from_desc(desc: &SoftBodyDesc) -> Result<Self, SoftBodyError> {
        let contours: Vec<&[[Fix128; 2]]> = desc
            .contours
            .iter()
            .filter(|c| c.len() >= 3)
            .map(Vec::as_slice)
            .collect();
        if contours.is_empty() {
            return Err(SoftBodyError::EmptyOutline);
        }
        if desc.spacing.is_negative() || desc.spacing.is_zero() {
            return Err(SoftBodyError::SpacingNotPositive);
        }
        if desc.mass_per_particle.is_negative() || desc.mass_per_particle.is_zero() {
            return Err(SoftBodyError::MassNotPositive);
        }
        let (mut lo, mut hi) = (contours[0][0], contours[0][0]);
        for c in &contours {
            for p in c.iter() {
                for k in 0..2 {
                    if p[k] < lo[k] {
                        lo[k] = p[k];
                    }
                    if p[k] > hi[k] {
                        hi[k] = p[k];
                    }
                }
            }
        }
        let s = desc.spacing;
        let count = |span: Fix128| -> usize {
            // lattice points lo, lo + s, … while ≤ hi
            let mut n = 1usize;
            let mut x = s;
            while x <= span {
                n += 1;
                x = x + s;
            }
            n
        };
        let (nx, ny) = (count(hi[0] - lo[0]), count(hi[1] - lo[1]));
        let half = s.half();
        let r2 = half * half;
        // kept lattice points, row-major; `index[j * nx + i]` is the particle index or usize::MAX
        let mut index = Vec::with_capacity(nx * ny);
        let mut lattice: Vec<[Fix128; 2]> = Vec::new();
        for j in 0..ny {
            let y = lo[1] + s * Fix128::from_int(j as i64);
            for i in 0..nx {
                let x = lo[0] + s * Fix128::from_int(i as i64);
                let p = [x, y];
                let keep = inside_even_odd(p, &contours) || within_of_outline(p, &contours, r2);
                if keep {
                    index.push(lattice.len());
                    lattice.push(p);
                } else {
                    index.push(usize::MAX);
                }
            }
        }
        let mut triangles: Vec<[usize; 3]> = Vec::new();
        for j in 0..ny.saturating_sub(1) {
            for i in 0..nx.saturating_sub(1) {
                let k00 = index[j * nx + i];
                let k10 = index[j * nx + i + 1];
                let k01 = index[(j + 1) * nx + i];
                let k11 = index[(j + 1) * nx + i + 1];
                if k00 != usize::MAX && k10 != usize::MAX && k01 != usize::MAX {
                    triangles.push([k00, k10, k01]);
                }
                if k10 != usize::MAX && k11 != usize::MAX && k01 != usize::MAX {
                    triangles.push([k10, k11, k01]);
                }
            }
        }
        if triangles.is_empty() {
            return Err(SoftBodyError::NoTriangles);
        }
        // drop the particles no triangle uses, keeping the order
        let mut used = vec![false; lattice.len()];
        for t in &triangles {
            for &k in t {
                used[k] = true;
            }
        }
        let mut remap = vec![usize::MAX; lattice.len()];
        let mut positions = Vec::new();
        for (k, p) in lattice.iter().enumerate() {
            if used[k] {
                remap[k] = positions.len();
                positions.push(desc.origin + desc.u * p[0] + desc.v * p[1]);
            }
        }
        for t in &mut triangles {
            for k in t.iter_mut() {
                *k = remap[*k];
            }
        }
        let inv_mass = Fix128::ONE / desc.mass_per_particle;
        let mut cloth = Cloth::from_mesh(positions, triangles, inv_mass);
        cloth.config = desc.config;
        let rest = cloth.positions.clone();
        cloth
            .set_rest_positions(&rest)
            .expect("rest positions are the particles");
        if let Some((k, c)) = desc.tether {
            cloth.set_rest_tether(k, c);
        }
        cloth.set_sleep(true);
        Ok(Self {
            cloth,
            grabs: Vec::new(),
            last_dt: Fix128::ZERO,
        })
    }

    /// Wrap a cloth as a soft body with no grabs (restore path).
    pub(crate) fn from_parts(cloth: Cloth, grabs: Vec<Grab>, last_dt: Fix128) -> Self {
        Self {
            cloth,
            grabs,
            last_dt,
        }
    }

    /// The cloth: particles, triangles, configuration, rest shape.
    #[must_use]
    pub fn cloth(&self) -> &Cloth {
        &self.cloth
    }

    /// The cloth, to write to directly; the next step sees every change (a sleeping
    /// body wakes when a bit differs from what it fell asleep with).
    pub fn cloth_mut(&mut self) -> &mut Cloth {
        &mut self.cloth
    }

    /// Number of particles.
    #[must_use]
    pub fn particle_count(&self) -> usize {
        self.cloth.particle_count()
    }

    /// Whether the last step was the identity and the next will be skipped
    /// ([`Cloth::is_asleep`]).
    #[must_use]
    pub fn is_asleep(&self) -> bool {
        self.cloth.is_asleep()
    }

    /// Turn the exact sleep on or off ([`Cloth::set_sleep`]; on for a body added by
    /// [`PhysicsWorld::add_soft_body`](crate::solver::PhysicsWorld::add_soft_body)).
    pub fn set_sleep(&mut self, enabled: bool) {
        self.cloth.set_sleep(enabled);
    }

    /// Whether the exact sleep is on.
    #[must_use]
    pub fn sleep_enabled(&self) -> bool {
        self.cloth.sleep_enabled()
    }

    /// The grabs holding this body, in the order they were taken.
    #[must_use]
    pub fn grabs(&self) -> &[Grab] {
        &self.grabs
    }

    /// The particle a grab holds, if the grab is on this body.
    #[must_use]
    pub fn grab_particle(&self, id: GrabId) -> Option<usize> {
        self.grabs.iter().find(|g| g.id == id).map(|g| g.particle)
    }

    /// `dt` of the last step (zero before any).
    #[must_use]
    pub fn last_dt(&self) -> Fix128 {
        self.last_dt
    }

    /// Nearest particle to `point` within `reach` that no grab holds (ties: the lowest
    /// index), or `None`.
    #[must_use]
    pub fn nearest_free(&self, point: Vec3Fix, reach: Fix128) -> Option<usize> {
        let mut best: Option<(usize, Fix128)> = None;
        for (i, p) in self.cloth.positions.iter().enumerate() {
            if self.grabs.iter().any(|g| g.particle == i) {
                continue;
            }
            let d = (*p - point).length();
            if d <= reach && best.is_none_or(|(_, bd)| d < bd) {
                best = Some((i, d));
            }
        }
        best.map(|(i, _)| i)
    }

    pub(crate) fn grab(&mut self, id: GrabId, point: Vec3Fix, reach: Fix128) -> Option<GrabId> {
        let particle = self.nearest_free(point, reach)?;
        self.cloth.pin(particle);
        self.grabs.push(Grab {
            id,
            particle,
            target: point,
            applied: [point, point],
        });
        Some(id)
    }

    pub(crate) fn move_grab(&mut self, id: GrabId, point: Vec3Fix) -> bool {
        match self.grabs.iter_mut().find(|g| g.id == id) {
            Some(g) => {
                g.target = point;
                true
            }
            None => false,
        }
    }

    pub(crate) fn release(&mut self, id: GrabId, velocity: Option<Vec3Fix>) -> bool {
        let Some(k) = self.grabs.iter().position(|g| g.id == id) else {
            return false;
        };
        let g = self.grabs.remove(k);
        self.cloth.unpin(g.particle);
        let v = velocity.unwrap_or_else(|| {
            if self.last_dt.is_zero() {
                Vec3Fix::ZERO
            } else {
                (g.applied[0] - g.applied[1]) / self.last_dt
            }
        });
        self.cloth.velocities[g.particle] = v;
        true
    }

    pub(crate) fn has_grab(&self, id: GrabId) -> bool {
        self.grabs.iter().any(|g| g.id == id)
    }

    /// One world step: place the grabbed particles, step the cloth, record the targets.
    pub(crate) fn step(&mut self, dt: Fix128) {
        for g in &self.grabs {
            let i = g.particle;
            self.cloth.positions[i] = g.target;
            self.cloth.prev_positions[i] = g.target;
            self.cloth.velocities[i] = Vec3Fix::ZERO;
        }
        self.cloth.step(dt);
        for g in &mut self.grabs {
            g.applied = [g.target, g.applied[0]];
        }
        self.last_dt = dt;
    }
}

/// FNV-1a 64 over `bytes` into `hash`.
fn fold(hash: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *hash ^= u64::from(b);
        *hash = hash.wrapping_mul(0x0000_0001_0000_01b3);
    }
}

fn fold_fix(hash: &mut u64, f: Fix128) {
    fold(hash, &f.hi.to_le_bytes());
    fold(hash, &f.lo.to_le_bytes());
}

fn fold_vec3(hash: &mut u64, v: Vec3Fix) {
    fold_fix(hash, v.x);
    fold_fix(hash, v.y);
    fold_fix(hash, v.z);
}

fn push_fix(out: &mut Vec<u8>, f: Fix128) {
    out.extend_from_slice(&f.hi.to_le_bytes());
    out.extend_from_slice(&f.lo.to_le_bytes());
}

fn push_vec3(out: &mut Vec<u8>, v: Vec3Fix) {
    push_fix(out, v.x);
    push_fix(out, v.y);
    push_fix(out, v.z);
}

/// A cursor over a rollback blob; every read is bounds-checked.
pub(crate) struct Cursor<'a> {
    data: &'a [u8],
    pub(crate) pos: usize,
}

impl<'a> Cursor<'a> {
    pub(crate) fn new(data: &'a [u8], pos: usize) -> Self {
        Self { data, pos }
    }
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.pos.checked_add(n)?;
        let s = self.data.get(self.pos..end)?;
        self.pos = end;
        Some(s)
    }
    pub(crate) fn u32(&mut self) -> Option<u32> {
        self.take(4)
            .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }
    fn u64(&mut self) -> Option<u64> {
        let s = self.take(8)?;
        let mut b = [0u8; 8];
        b.copy_from_slice(s);
        Some(u64::from_le_bytes(b))
    }
    fn fix(&mut self) -> Option<Fix128> {
        let s = self.take(16)?;
        let mut hi = [0u8; 8];
        let mut lo = [0u8; 8];
        hi.copy_from_slice(&s[..8]);
        lo.copy_from_slice(&s[8..]);
        Some(Fix128 {
            hi: i64::from_le_bytes(hi),
            lo: u64::from_le_bytes(lo),
        })
    }
    fn vec3(&mut self) -> Option<Vec3Fix> {
        Some(Vec3Fix::new(self.fix()?, self.fix()?, self.fix()?))
    }
    fn index(&mut self, below: usize) -> Option<usize> {
        let i = usize::try_from(self.u32()?).ok()?;
        (i < below).then_some(i)
    }
}

impl SoftBody {
    /// FNV-1a 64 of what the rollback blob does not carry: triangles, rest shape,
    /// configuration, tether.
    pub(crate) fn structure_hash(&self) -> u64 {
        let c = &self.cloth;
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        fold(&mut h, &(c.triangles.len() as u64).to_le_bytes());
        for t in &c.triangles {
            for &k in t {
                fold(&mut h, &(k as u64).to_le_bytes());
            }
        }
        fold(&mut h, &(c.rest_positions().len() as u64).to_le_bytes());
        for p in c.rest_positions() {
            fold_vec3(&mut h, *p);
        }
        fold(&mut h, &(c.config.iterations as u64).to_le_bytes());
        fold(&mut h, &(c.config.substeps as u64).to_le_bytes());
        fold_vec3(&mut h, c.config.gravity);
        fold_fix(&mut h, c.config.damping);
        fold_fix(&mut h, c.config.stretch_compliance);
        fold_fix(&mut h, c.config.bend_compliance);
        fold_fix(&mut h, c.config.sdf_friction);
        fold_fix(&mut h, c.config.thickness);
        fold(&mut h, &[u8::from(c.config.self_collision)]);
        fold_fix(&mut h, c.config.self_collision_distance);
        match c.rest_tether() {
            Some((k, d)) => {
                fold(&mut h, &[1]);
                fold_fix(&mut h, k);
                fold_fix(&mut h, d);
            }
            None => fold(&mut h, &[0]),
        }
        h
    }

    /// Append this body's motion state to a rollback blob
    /// ([`PhysicsWorld::serialize_state`](crate::solver::PhysicsWorld::serialize_state)):
    /// particle count `u32`, [`Self::structure_hash`] `u64`, per particle position /
    /// previous position / velocity / inverse mass, the pinned list, the inverse masses
    /// pins replaced, the grabs, `last_dt`. The sleep flag is not carried (it changes
    /// no bit): a restored body has the exact sleep on.
    pub(crate) fn write_rollback(&self, out: &mut Vec<u8>) {
        let c = &self.cloth;
        out.extend_from_slice(&(c.positions.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.structure_hash().to_le_bytes());
        for i in 0..c.positions.len() {
            push_vec3(out, c.positions[i]);
            push_vec3(out, c.prev_positions[i]);
            push_vec3(out, c.velocities[i]);
            push_fix(out, c.inv_masses[i]);
        }
        out.extend_from_slice(&(c.pinned.len() as u32).to_le_bytes());
        for &i in &c.pinned {
            out.extend_from_slice(&(i as u32).to_le_bytes());
        }
        out.extend_from_slice(&(c.pin_released().len() as u32).to_le_bytes());
        for &(i, m) in c.pin_released() {
            out.extend_from_slice(&(i as u32).to_le_bytes());
            push_fix(out, m);
        }
        out.extend_from_slice(&(self.grabs.len() as u32).to_le_bytes());
        for g in &self.grabs {
            out.extend_from_slice(&g.id.0.to_le_bytes());
            out.extend_from_slice(&(g.particle as u32).to_le_bytes());
            push_vec3(out, g.target);
            push_vec3(out, g.applied[0]);
            push_vec3(out, g.applied[1]);
        }
        push_fix(out, self.last_dt);
    }

    /// Read one body's section written by [`Self::write_rollback`]. With `apply`
    /// false nothing is written: the section is only checked against this body
    /// (particle count, structure hash, every index in range). Returns `None` when
    /// the blob is short, mismatched or holds an index out of range.
    pub(crate) fn read_rollback(&mut self, cur: &mut Cursor<'_>, apply: bool) -> Option<()> {
        let n = self.cloth.positions.len();
        if usize::try_from(cur.u32()?).ok()? != n || cur.u64()? != self.structure_hash() {
            return None;
        }
        for i in 0..n {
            let (p, q, v, m) = (cur.vec3()?, cur.vec3()?, cur.vec3()?, cur.fix()?);
            if apply {
                self.cloth.positions[i] = p;
                self.cloth.prev_positions[i] = q;
                self.cloth.velocities[i] = v;
                self.cloth.inv_masses[i] = m;
            }
        }
        let k = usize::try_from(cur.u32()?).ok()?;
        let mut pinned = Vec::with_capacity(if apply { k } else { 0 });
        for _ in 0..k {
            let i = cur.index(n)?;
            if apply {
                pinned.push(i);
            }
        }
        let k = usize::try_from(cur.u32()?).ok()?;
        let mut released = Vec::with_capacity(if apply { k } else { 0 });
        for _ in 0..k {
            let (i, m) = (cur.index(n)?, cur.fix()?);
            if apply {
                released.push((i, m));
            }
        }
        let k = usize::try_from(cur.u32()?).ok()?;
        let mut grabs = Vec::with_capacity(if apply { k } else { 0 });
        for _ in 0..k {
            let id = GrabId(cur.u32()?);
            let particle = cur.index(n)?;
            let (target, a0, a1) = (cur.vec3()?, cur.vec3()?, cur.vec3()?);
            if apply {
                grabs.push(Grab {
                    id,
                    particle,
                    target,
                    applied: [a0, a1],
                });
            }
        }
        let last_dt = cur.fix()?;
        if apply {
            self.cloth.pinned = pinned;
            self.cloth.set_pin_released(released);
            self.grabs = grabs;
            self.last_dt = last_dt;
            // a restored body wakes with the exact sleep on; if its state is the
            // fixed point the next step is the identity and puts it back to sleep
            self.cloth.set_sleep(false);
            self.cloth.set_sleep(true);
        }
        Some(())
    }
}

/// Even-odd fill: a horizontal ray from `p` crosses an odd number of edges. The
/// crossing test is exact (`(p.x − a.x)(b.y − a.y) < (p.y − a.y)(b.x − a.x)` with the
/// edge oriented upward), half-open in `y` so a vertex is counted once.
fn inside_even_odd(p: [Fix128; 2], contours: &[&[[Fix128; 2]]]) -> bool {
    let mut odd = false;
    for c in contours {
        let n = c.len();
        for i in 0..n {
            let (mut a, mut b) = (c[i], c[(i + 1) % n]);
            if a[1] == b[1] {
                continue;
            }
            if a[1] > b[1] {
                core::mem::swap(&mut a, &mut b);
            }
            // upward edge a → b: crosses the ray when a.y <= p.y < b.y
            if !(a[1] <= p[1] && p[1] < b[1]) {
                continue;
            }
            let lhs = (p[0] - a[0]) * (b[1] - a[1]);
            let rhs = (p[1] - a[1]) * (b[0] - a[0]);
            if lhs < rhs {
                odd = !odd;
            }
        }
    }
    odd
}

/// Whether `p` is within `sqrt(r2)` of some edge of the outline.
fn within_of_outline(p: [Fix128; 2], contours: &[&[[Fix128; 2]]], r2: Fix128) -> bool {
    for c in contours {
        let n = c.len();
        for i in 0..n {
            let (a, b) = (c[i], c[(i + 1) % n]);
            let e = [b[0] - a[0], b[1] - a[1]];
            let d = [p[0] - a[0], p[1] - a[1]];
            let len2 = e[0] * e[0] + e[1] * e[1];
            let t = if len2.is_zero() {
                Fix128::ZERO
            } else {
                let t = (d[0] * e[0] + d[1] * e[1]) / len2;
                if t.is_negative() {
                    Fix128::ZERO
                } else if t > Fix128::ONE {
                    Fix128::ONE
                } else {
                    t
                }
            };
            let q = [d[0] - e[0] * t, d[1] - e[1] * t];
            if q[0] * q[0] + q[1] * q[1] <= r2 {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square() -> Vec<[Fix128; 2]> {
        let r = |n: i64| Fix128::from_ratio(n, 10);
        vec![[r(0), r(0)], [r(10), r(0)], [r(10), r(10)], [r(0), r(10)]]
    }

    #[test]
    fn even_odd_counts_a_vertex_once() {
        let sq = square();
        let c: Vec<&[[Fix128; 2]]> = vec![sq.as_slice()];
        let r = |n: i64| Fix128::from_ratio(n, 10);
        assert!(inside_even_odd([r(5), r(5)], &c));
        assert!(!inside_even_odd([r(15), r(5)], &c));
        // the ray through the top vertices' y = 1.0 is outside (half-open)
        assert!(!inside_even_odd([r(5), r(10)], &c));
        assert!(inside_even_odd([r(5), r(0)], &c));
    }

    #[test]
    fn outline_distance_is_a_band() {
        let sq = square();
        let c: Vec<&[[Fix128; 2]]> = vec![sq.as_slice()];
        let r = |n: i64| Fix128::from_ratio(n, 100);
        let r2 = r(5) * r(5);
        assert!(within_of_outline([r(-4), r(50)], &c, r2));
        assert!(!within_of_outline([r(-6), r(50)], &c, r2));
        assert!(within_of_outline([r(103), r(103)], &c, r2));
        assert!(!within_of_outline([r(104), r(104)], &c, r2));
    }
}
