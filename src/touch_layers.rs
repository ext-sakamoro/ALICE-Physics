//! Layered touch playback: a set of soft bodies, one hand per layer of touches, and one rule for which layer
//! moves a body. The same playback runs while touches are recorded step by step and when the logs are replayed,
//! so a recording and its replay go through the same states, frame for frame, on every target.
//!
//! # MODEL — soft bodies held to their rest shape, touched by layered hands, under an active-body budget
//!
//! **Equations.** Each body is a caller-supplied [`Cloth`](crate::cloth::Cloth) whose own law moves it; a cloth held to its rest shape
//! by its tether ([`Cloth::set_rest_tether`](crate::cloth::Cloth::set_rest_tether)) springs back as a damped oscillator. Layer `L` (0 the lowest) has
//! its own [`Hand`](crate::touch::Hand); a touch is what [`Hand::apply`](crate::touch::Hand::apply) does and a finite grip pulls by [`Hand::pull`](crate::touch::Hand::pull) before the step
//! ([`TouchBodies::input`](crate::touch_layers::TouchBodies::input), [`TouchBodies::step`](crate::touch_layers::TouchBodies::step)). The override rule ([`LayeredPlayback::advance`](crate::touch_layers::LayeredPlayback::advance),
//! `LayeredPlayback::apply`): each step's touches are given from the highest layer down, then the bodies step; a
//! grab of body `b` by layer `L` at step `s` makes every lower hand holding `b` let go at `s`, and every later span
//! (grab … release) of a lower layer on `b` is skipped, the skipped grab still letting go of what that hand held;
//! the lower hands let go and the later spans are skipped only when `L`'s hand takes `b` (a grab that misses or that the
//! budget refuses takes nothing, so the lower hands keep `b` and their spans on it go on).
//! The active-body budget ([`TouchBodies::input`](crate::touch_layers::TouchBodies::input)): only active bodies are stepped; a grab of an inactive body,
//! after the hand lets go of what it held, makes it active; when `max_active` bodies are already active, the active
//! body that started moving earliest and that no hand holds is first put exactly at rest (`TouchBodies::settle`:
//! positions and previous positions on the rest shape, velocities zero); when every active body is held, the grab
//! takes nothing. After a step, an active body no hand holds whose every particle is within `rest_tolerance` of its
//! rest place on each axis and whose every velocity component is within `rest_tolerance` is put exactly at rest and
//! leaves the active list ([`TouchBodies::step`](crate::touch_layers::TouchBodies::step)). The state digest ([`TouchBodies::digest`](crate::touch_layers::TouchBodies::digest)) is SHA-256 over
//! every body's [`Cloth::state_digest`](crate::cloth::Cloth::state_digest), every hand's [`Hand::state_bytes`](crate::touch::Hand::state_bytes) and the active list; the history digest
//! ([`history_digest`](crate::touch_layers::history_digest)) is SHA-256 over every frame's state digest in order. The final state cannot tell two
//! histories apart, because bodies settle exactly to rest: two different touch logs that both end at rest reach
//! the same last frame, so a recording is identified by its history, not its last state.
//!
//! **Units.** SI: positions, reach and `rest_tolerance` m (and m/s for the speed test), velocities m/s, `dt` s;
//! steps are counts. Every quantity is `Fix128` (Q64.64); touches are on the 2⁻¹⁶ m grid of [`Touch`](crate::touch::Touch).
//!
//! **Conservation.** Nothing beyond each body's own law: a hand is an external force (momentum and energy change by
//! what it does, as in [`crate::touch`]), and settling to rest removes the residual energy below the rest tolerance
//! at once (at most `½ m (√3 ε)²` per particle for tolerance `ε`), the one place the playback itself takes energy
//! out ([`TouchBodies::step`](crate::touch_layers::TouchBodies::step)). A body that is not active is not stepped, so it neither gains nor loses anything.
//!
//! **Range.** Any `dt > 0`, any tolerance `≥ 0`, any budget (0 means every grab of a body at rest takes nothing),
//! any number of bodies and layers ([`PlaybackConfig::new`](crate::touch_layers::PlaybackConfig::new), [`TouchBodies::new`](crate::touch_layers::TouchBodies::new)); refused with [`LayerError`](crate::touch_layers::LayerError):
//! `dt ≤ 0`, a negative tolerance, a hand template that holds something, a body without a rest shape (its rest
//! positions not one per particle), a touch at or after the playback's step count and a top-layer grab of a body
//! that does not exist ([`LayeredPlayback::touch`](crate::touch_layers::LayeredPlayback::touch)). A lower layer's grab of a body that does not exist takes
//! nothing (the hand lets go). A body whose rest shape is not its equilibrium (gravity on, no tether) never comes
//! within the tolerance and stays active until the budget settles it.
//!
//! **Approximations.** Bodies do not touch each other (each steps alone, [`TouchBodies::step`](crate::touch_layers::TouchBodies::step)); settling is all at
//! once (no blend); the override is by body, not by point (a higher layer takes the whole body,
//! `LayeredPlayback::apply`); a body put to rest by the budget loses its motion without a transition.
//!
//! **Validation.** `tests/analytic_touch_layers.rs::a_live_recording_and_its_replay_from_the_encoded_logs_agree_frame_for_frame`
//! `::a_higher_layer_takes_a_body_over_and_leaves_the_others_bit_equal`
//! `::a_skipped_grab_still_lets_go_of_what_the_lower_hand_held`
//! `::a_grab_past_the_budget_settles_the_earliest_free_body_exactly_to_rest`
//! `::no_more_than_the_budget_ever_move`
//! `::a_released_body_comes_back_to_rest_and_is_put_exactly_there`
//! `::the_history_digest_tells_apart_two_logs_that_end_at_the_same_rest`
//! `::degenerate_scenes_give_their_stated_results`
//! `::a_fixed_scene_reaches_the_pinned_history_digest`
//!
//! **Tiers.** L0 only.

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use sha2::{Digest, Sha256};

use crate::cloth::Cloth;
use crate::input_log::InputLog;
use crate::math::{Fix128, Vec3Fix};
use crate::touch::{Hand, Touch};

const STATE_DOMAIN: &[u8] = b"alice-physics touch-layers state v1";
const HISTORY_DOMAIN: &[u8] = b"alice-physics touch-layers history v1";

/// Why a playback could not be built or a touch could not be given
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerError {
    /// The step length is not above zero
    BadStep,
    /// The rest tolerance is negative
    BadTolerance,
    /// A limit is out of its range (see [`crate::touch_limits::TouchLimits::new`])
    BadLimits,
    /// The hand given as the template already holds a body
    HandHolds,
    /// This body has no rest shape (its rest positions are not one per particle)
    NoRestShape(usize),
    /// A touch at or after the playback's last step
    TooLong,
    /// A top-layer grab of a body that does not exist
    NoSuchBody,
}

/// How long a playback runs and how its bodies are stepped
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlaybackConfig {
    steps: u64,
    dt: Fix128,
    max_active: usize,
    rest_tolerance: Fix128,
}

impl PlaybackConfig {
    /// `steps` steps of `dt` (s); at most `max_active` bodies move at once; a released body within
    /// `rest_tolerance` (m, m/s) of rest in every particle's place and speed is put exactly at rest.
    ///
    /// # Errors
    ///
    /// [`LayerError::BadStep`] for `dt ≤ 0`, [`LayerError::BadTolerance`] for a negative tolerance.
    pub fn new(
        steps: u64,
        dt: Fix128,
        max_active: usize,
        rest_tolerance: Fix128,
    ) -> Result<Self, LayerError> {
        if dt <= Fix128::ZERO {
            return Err(LayerError::BadStep);
        }
        if rest_tolerance.is_negative() {
            return Err(LayerError::BadTolerance);
        }
        Ok(Self {
            steps,
            dt,
            max_active,
            rest_tolerance,
        })
    }
}

/// Soft bodies and one hand per layer touching them; only the active bodies are stepped
#[derive(Clone)]
pub struct TouchBodies {
    cloths: Vec<Cloth>,
    hands: Vec<Hand>,
    /// Bodies being stepped, in the order they started moving
    active: Vec<usize>,
    config: PlaybackConfig,
}

impl TouchBodies {
    /// `cloths` as they are, and `layers` hands like `hand` (at least one).
    ///
    /// # Errors
    ///
    /// [`LayerError::HandHolds`] when `hand` holds a body, [`LayerError::NoRestShape`] for a cloth whose rest
    /// positions are not one per particle.
    pub fn new(
        cloths: Vec<Cloth>,
        hand: &Hand,
        layers: usize,
        config: PlaybackConfig,
    ) -> Result<Self, LayerError> {
        if hand.held().is_some() {
            return Err(LayerError::HandHolds);
        }
        if let Some(b) = cloths
            .iter()
            .position(|c| c.rest_positions().len() != c.positions.len())
        {
            return Err(LayerError::NoRestShape(b));
        }
        Ok(Self {
            cloths,
            hands: vec![hand.clone(); layers.max(1)],
            active: Vec::new(),
            config,
        })
    }

    /// How many bodies there are
    #[must_use]
    pub fn len(&self) -> usize {
        self.cloths.len()
    }

    /// Whether there are none
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cloths.is_empty()
    }

    /// The bodies moving now, in the order they started
    #[must_use]
    pub fn active(&self) -> &[usize] {
        &self.active
    }

    /// Body `b` (`None` when there is no such body)
    #[must_use]
    pub fn cloth(&self, b: usize) -> Option<&Cloth> {
        self.cloths.get(b)
    }

    /// Body `b`'s centre: the mean of its particles (`None` when there is no such body or it has none)
    #[must_use]
    pub fn centre(&self, b: usize) -> Option<Vec3Fix> {
        mean(&self.cloths.get(b)?.positions)
    }

    /// The body hand `layer` holds
    fn held_by(&self, layer: usize) -> Option<usize> {
        self.hands
            .get(layer)
            .and_then(Hand::held)
            .and_then(|(b, _)| usize::try_from(b).ok())
    }

    /// Whether `layer`'s grab `touch` of body `b` would take a point, without touching anything (the budget aside)
    fn would_take(&self, layer: usize, b: usize, touch: &Touch) -> bool {
        let Touch::Grab { at, .. } = *touch else {
            return false;
        };
        match (self.hands.get(layer), self.cloths.get(b)) {
            (Some(hand), Some(cloth)) => hand.would_take(cloth, at),
            _ => false,
        }
    }

    fn is_held(&self, b: usize) -> bool {
        (0..self.hands.len()).any(|h| self.held_by(h) == Some(b))
    }

    /// Put body `b` exactly at rest and stop stepping it
    fn settle(&mut self, b: usize) {
        let cloth = &mut self.cloths[b];
        let rest = cloth.rest_positions().to_vec();
        cloth.positions.clone_from(&rest);
        cloth.prev_positions = rest;
        cloth.velocities.iter_mut().for_each(|v| *v = Vec3Fix::ZERO);
        self.active.retain(|&a| a != b);
    }

    /// One touch by hand `layer`, given before the step. A grab first lets go of what the hand held; a grab of a
    /// body not moving makes it move, and when the budget is full the active body that started moving earliest and
    /// that no hand holds is put exactly at rest first; when every active body is held the grab takes nothing.
    /// A hand that does not exist does nothing. Whether the touch did anything
    pub fn input(&mut self, layer: usize, touch: &Touch) -> bool {
        if layer >= self.hands.len() {
            return false;
        }
        let Touch::Grab { body, .. } = *touch else {
            return self.hands[layer].apply(&mut self.cloths, touch);
        };
        let b = usize::try_from(body).unwrap_or(usize::MAX);
        if b >= self.cloths.len() || self.active.contains(&b) {
            return self.hands[layer].apply(&mut self.cloths, touch);
        }
        // the hand lets go of what it held before it takes another body (as `Hand::apply` does)
        self.hands[layer].apply(&mut self.cloths, &Touch::Release);
        let evict = if self.active.len() >= self.config.max_active {
            match self.active.iter().copied().find(|&a| !self.is_held(a)) {
                Some(a) => Some(a),
                None => return false,
            }
        } else {
            None
        };
        if !self.hands[layer].apply(&mut self.cloths, touch) {
            return false;
        }
        if let Some(a) = evict {
            self.settle(a);
        }
        self.active.push(b);
        true
    }

    /// One step of `dt` over the active bodies (finite grips pull first); a body no hand holds that is within the
    /// rest tolerance is put exactly at rest and stops
    pub fn step(&mut self) {
        let dt = self.config.dt;
        for hand in &mut self.hands {
            hand.pull(&mut self.cloths, dt);
        }
        let held: Vec<usize> = (0..self.hands.len())
            .filter_map(|h| self.held_by(h))
            .collect();
        let eps = self.config.rest_tolerance;
        let near = |v: Vec3Fix| v.x.abs() <= eps && v.y.abs() <= eps && v.z.abs() <= eps;
        let mut done = Vec::new();
        for &b in &self.active {
            let cloth = &mut self.cloths[b];
            cloth.step(dt);
            if held.contains(&b) {
                continue;
            }
            let still = cloth
                .positions
                .iter()
                .zip(cloth.rest_positions())
                .all(|(p, r)| near(*p - *r))
                && cloth.velocities.iter().all(|v| near(*v));
            if still {
                done.push(b);
            }
        }
        for b in done {
            self.settle(b);
        }
    }

    /// SHA-256 of the whole state: every body, every hand in order, which bodies move
    #[must_use]
    pub fn digest(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(STATE_DOMAIN);
        h.update((self.cloths.len() as u64).to_le_bytes());
        for c in &self.cloths {
            h.update(c.state_digest());
        }
        h.update((self.hands.len() as u64).to_le_bytes());
        for hand in &self.hands {
            let bytes = hand.state_bytes();
            h.update((bytes.len() as u64).to_le_bytes());
            h.update(&bytes);
        }
        h.update((self.active.len() as u64).to_le_bytes());
        for &a in &self.active {
            h.update((a as u64).to_le_bytes());
        }
        h.finalize().into()
    }
}

/// The mean of `points` (`None` for none)
pub(crate) fn mean(points: &[Vec3Fix]) -> Option<Vec3Fix> {
    if points.is_empty() {
        return None;
    }
    let n = Fix128::from_int(i64::try_from(points.len()).unwrap_or(i64::MAX));
    let s = points.iter().fold(Vec3Fix::ZERO, |a, p| a + *p);
    Some(Vec3Fix::new(s.x / n, s.y / n, s.z / n))
}

/// SHA-256 over every frame's state digest, in order: what identifies a playback (its last frame does not, since
/// bodies settle exactly to rest)
#[must_use]
pub fn history_digest(frames: &[[u8; 32]]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(HISTORY_DOMAIN);
    h.update((frames.len() as u64).to_le_bytes());
    for f in frames {
        h.update(f);
    }
    h.finalize().into()
}

/// What a playback went through: the top layer's touches, every frame's state digest, their history digest
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Recording {
    /// The top layer's touches, by step
    pub top: InputLog<Touch>,
    /// Each frame's state digest, in order
    pub frames: Vec<[u8; 32]>,
    /// [`history_digest`] of the frames
    pub history: [u8; 32],
}

/// Layers of touches played on bodies under the override rule; the top layer is fed step by step
/// ([`Self::touch`]), the lower layers come from logs. Used for recording and for replay alike.
pub struct LayeredPlayback {
    bodies: TouchBodies,
    /// The layers under the top one, lowest first
    lower: Vec<InputLog<Touch>>,
    /// Each lower layer's next entry
    next: Vec<usize>,
    /// Per body: 1 + the highest layer that has grabbed it (0 for none)
    owner: Vec<usize>,
    /// Per layer (the top one last): inside a span that was taken over
    skipping: Vec<bool>,
    top: InputLog<Touch>,
    frames: Vec<[u8; 32]>,
    step: u64,
}

impl LayeredPlayback {
    /// `cloths` under `lower` (lowest first) and a top layer fed by [`Self::touch`], each layer with a hand like
    /// `hand`.
    ///
    /// # Errors
    ///
    /// As [`TouchBodies::new`], and [`LayerError::TooLong`] for a lower-layer touch at or after the last step.
    pub fn new(
        cloths: Vec<Cloth>,
        hand: &Hand,
        lower: Vec<InputLog<Touch>>,
        config: PlaybackConfig,
    ) -> Result<Self, LayerError> {
        if lower
            .iter()
            .any(|l| l.entries().last().is_some_and(|(s, _)| *s >= config.steps))
        {
            return Err(LayerError::TooLong);
        }
        let k = lower.len() + 1;
        let n = cloths.len();
        Ok(Self {
            bodies: TouchBodies::new(cloths, hand, k, config)?,
            next: vec![0; lower.len()],
            lower,
            owner: vec![0; n],
            skipping: vec![false; k],
            top: InputLog::new(),
            frames: Vec::new(),
            step: 0,
        })
    }

    /// The bodies now
    #[must_use]
    pub fn bodies(&self) -> &TouchBodies {
        &self.bodies
    }

    /// The step the next touches are given at
    #[must_use]
    pub fn step_now(&self) -> u64 {
        self.step
    }

    /// A top-layer touch at the current step, recorded into the top layer's log.
    ///
    /// # Errors
    ///
    /// [`LayerError::TooLong`] at or after the last step, [`LayerError::NoSuchBody`] for a grab of a body that does
    /// not exist (nothing is recorded or done).
    pub fn touch(&mut self, touch: &Touch) -> Result<(), LayerError> {
        if self.step >= self.bodies.config.steps {
            return Err(LayerError::TooLong);
        }
        if let Touch::Grab { body, .. } = *touch {
            if !usize::try_from(body).is_ok_and(|b| b < self.bodies.len()) {
                return Err(LayerError::NoSuchBody);
            }
        }
        self.top
            .record(self.step, *touch)
            .map_err(|_| LayerError::TooLong)?;
        self.apply(self.lower.len(), touch);
        Ok(())
    }

    /// The override rule for one touch of `layer`
    fn apply(&mut self, layer: usize, touch: &Touch) {
        match *touch {
            Touch::Grab { body, .. } => {
                if let Some(b) = usize::try_from(body).ok().filter(|&b| b < self.owner.len()) {
                    if self.owner[b] > layer + 1 {
                        // a higher layer has taken this body: skip the span (the grab still lets go of what the
                        // hand held, as a grab does)
                        self.skipping[layer] = true;
                        self.bodies.input(layer, &Touch::Release);
                        return;
                    }
                    // a grab that misses takes nothing, so the lower hands keep the body (whether it misses is
                    // known before anyone lets go: letting go does not move a point)
                    let takes = self.bodies.would_take(layer, b, touch);
                    for below in 0..layer {
                        if takes && self.bodies.held_by(below) == Some(b) {
                            self.bodies.input(below, &Touch::Release);
                            self.skipping[below] = true;
                        }
                    }
                    self.skipping[layer] = false;
                    // the layer owns the body only when its hand took it: a grab the budget refuses (every
                    // moving body held) or that misses leaves the lower layers' later spans on the body as they are
                    if self.bodies.input(layer, touch) {
                        self.owner[b] = self.owner[b].max(layer + 1);
                    }
                    return;
                }
                self.skipping[layer] = false;
                self.bodies.input(layer, touch);
            }
            Touch::Drag { .. } | Touch::Release => {
                if self.skipping[layer] {
                    if *touch == Touch::Release {
                        self.skipping[layer] = false;
                    }
                    return;
                }
                self.bodies.input(layer, touch);
            }
        }
    }

    /// Give the lower layers' touches of this step (highest first), step the bodies, and return the frame's digest.
    ///
    /// # Errors
    ///
    /// [`LayerError::TooLong`] at or after the last step.
    pub fn advance(&mut self) -> Result<[u8; 32], LayerError> {
        if self.step >= self.bodies.config.steps {
            return Err(LayerError::TooLong);
        }
        for layer in (0..self.lower.len()).rev() {
            while let Some(&(at, touch)) = self.lower[layer].entries().get(self.next[layer]) {
                if at > self.step {
                    break;
                }
                self.next[layer] += 1;
                self.apply(layer, &touch);
            }
        }
        self.bodies.step();
        self.step += 1;
        let d = self.bodies.digest();
        self.frames.push(d);
        Ok(d)
    }

    /// Run on to the last step; what the playback went through and the bodies at the end
    #[must_use]
    pub fn finish(mut self) -> (Recording, TouchBodies) {
        while self.advance().is_ok() {}
        let history = history_digest(&self.frames);
        (
            Recording {
                top: self.top,
                frames: self.frames,
                history,
            },
            self.bodies,
        )
    }
}

/// Replay `layers` (lowest first, the top last) on `cloths`, from the start to the last step.
///
/// # Errors
///
/// As [`LayeredPlayback::new`] and [`LayeredPlayback::touch`].
pub fn replay(
    cloths: &[Cloth],
    hand: &Hand,
    layers: &[InputLog<Touch>],
    config: PlaybackConfig,
) -> Result<Recording, LayerError> {
    replay_with(cloths, hand, layers, config, |_, _| {})
}

/// [`replay`], calling `observe(step, bodies)` after every step.
///
/// # Errors
///
/// As [`replay`].
pub fn replay_with(
    cloths: &[Cloth],
    hand: &Hand,
    layers: &[InputLog<Touch>],
    config: PlaybackConfig,
    mut observe: impl FnMut(u64, &TouchBodies),
) -> Result<Recording, LayerError> {
    let (top, lower) = match layers.split_last() {
        Some((top, lower)) => (top.entries(), lower.to_vec()),
        None => (&[][..], Vec::new()),
    };
    if top.last().is_some_and(|(s, _)| *s >= config.steps) {
        return Err(LayerError::TooLong);
    }
    let mut p = LayeredPlayback::new(cloths.to_vec(), hand, lower, config)?;
    let mut next = 0;
    while p.step_now() < config.steps {
        let step = p.step_now();
        while next < top.len() && top[next].0 <= step {
            p.touch(&top[next].1)?;
            next += 1;
        }
        p.advance()?;
        observe(step, p.bodies());
    }
    Ok(p.finish().0)
}
