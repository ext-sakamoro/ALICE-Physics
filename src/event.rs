//! Physics Event System
//!
//! Provides collision event reporting (begin/persist/end), trigger events and
//! joint break events.
//! Events are collected during `step()` and can be consumed after each frame.

use crate::math::{Fix128, Vec3Fix};

#[cfg(not(feature = "std"))]
use alloc::collections::{BTreeMap, BTreeSet};
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::collections::{BTreeMap, BTreeSet};

/// Type of contact event
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContactEventType {
    /// First frame of contact
    Begin,
    /// Contact persists from previous frame
    Persist,
    /// Contact ended (bodies separated)
    End,
}

/// A contact event between two bodies
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContactEvent {
    /// First body index
    pub body_a: usize,
    /// Second body index
    pub body_b: usize,
    /// Event type
    pub event_type: ContactEventType,
    /// Contact normal, unit length, pointing from `body_b` toward `body_a`
    /// (the crate-wide [`Contact::normal`](crate::collider::Contact::normal)
    /// contract): moving `body_a` along `+normal` separates the pair.
    /// Zero for [`ContactEventType::End`].
    pub normal: Vec3Fix,
    /// Contact point (world space)
    pub point: Vec3Fix,
    /// Penetration depth
    pub depth: Fix128,
    /// Relative velocity along the normal, `(v_a − v_b) · normal`: negative
    /// while the bodies approach each other
    pub relative_velocity: Fix128,
}

/// A trigger event (overlap without physics response)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TriggerEvent {
    /// Trigger body index
    pub trigger_body: usize,
    /// Other body index
    pub other_body: usize,
    /// Whether this is an enter or exit event
    pub entered: bool,
}

/// A joint removed by [`crate::solver::PhysicsWorld::step`] because its
/// reaction force exceeded its `break_force`
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JointBreakEvent {
    /// Index the joint had in `PhysicsWorld::joints` when it broke. Removal is
    /// a swap-remove (as [`crate::solver::PhysicsWorld::remove_joint`]), so the
    /// joint that was last now has this index.
    pub index: usize,
    /// The removed joint, as it was configured
    pub joint: crate::joint::Joint,
    /// The reaction force its last solve transmitted
    /// ([`crate::joint::solve_joints_with_reaction_forces`], N),
    /// strictly greater than its `break_force`
    pub force: Fix128,
}

/// Manages physics events for one simulation step
pub struct EventCollector {
    /// Contact events this frame
    pub(crate) contact_events: Vec<ContactEvent>,
    /// Trigger events this frame
    pub(crate) trigger_events: Vec<TriggerEvent>,
    /// Joints broken this frame
    pub(crate) joint_break_events: Vec<JointBreakEvent>,
    /// Active contact pairs from previous frame (for begin/persist/end tracking)
    pub(crate) prev_pairs: Vec<(usize, usize)>,
    /// Active contact pairs this frame (ordered set: membership tests are
    /// O(log n); the linear `Vec::contains` made `report_contact` O(pairs²) —
    /// 3 ms per detection at 2700 contacts, run 8× per frame since 1.2.0)
    /// Active contact pairs this frame
    pub(crate) curr_pairs: BTreeSet<(usize, usize)>,
    /// Active trigger overlaps from previous frame: (normalized pair, (trigger, other) roles of the enter frame)
    pub(crate) prev_triggers: Vec<((usize, usize), (usize, usize))>,
    /// Active trigger overlaps this frame (normalized pair -> roles as reported)
    pub(crate) curr_triggers: BTreeMap<(usize, usize), (usize, usize)>,
}

impl EventCollector {
    /// Create a new event collector
    #[must_use]
    pub const fn new() -> Self {
        Self {
            contact_events: Vec::new(),
            trigger_events: Vec::new(),
            joint_break_events: Vec::new(),
            prev_pairs: Vec::new(),
            curr_pairs: BTreeSet::new(),
            prev_triggers: Vec::new(),
            curr_triggers: BTreeMap::new(),
        }
    }

    /// Begin a new frame: swap previous/current pair tracking
    pub fn begin_frame(&mut self) {
        self.contact_events.clear();
        self.trigger_events.clear();
        self.joint_break_events.clear();
        // BTreeSet iterates in sorted order, so `prev_*` are sorted for
        // `binary_search` without an explicit sort.
        self.prev_pairs.clear();
        self.prev_pairs.extend(self.curr_pairs.iter().copied());
        self.curr_pairs.clear();
        self.prev_triggers.clear();
        self.prev_triggers
            .extend(self.curr_triggers.iter().map(|(k, v)| (*k, *v)));
        self.curr_triggers.clear();
    }

    /// Report a contact between two bodies
    ///
    /// `normal` points from `body_b` toward `body_a` as given here. The event
    /// stores the pair as `(min, max)`; when that swaps the bodies the normal
    /// is negated so it still points from the event's `body_b` toward its
    /// `body_a`. `relative_velocity` (`(v_a − v_b) · normal`) and `depth` are
    /// unchanged by the swap.
    pub fn report_contact(
        &mut self,
        body_a: usize,
        body_b: usize,
        normal: Vec3Fix,
        point: Vec3Fix,
        depth: Fix128,
        relative_velocity: Fix128,
    ) {
        let pair = normalize_pair(body_a, body_b);
        // B→A of the stored pair: negate when the pair was swapped.
        let normal = if pair.0 == body_a { normal } else { -normal };
        let was_active = self.prev_pairs.binary_search(&pair).is_ok();
        // Collision detection runs once per substep since 1.2.0; a pair is
        // reported once per frame (first substep that sees it).
        if !self.curr_pairs.insert(pair) {
            return;
        }

        let event_type = if was_active {
            ContactEventType::Persist
        } else {
            ContactEventType::Begin
        };

        self.contact_events.push(ContactEvent {
            body_a: pair.0,
            body_b: pair.1,
            event_type,
            normal,
            point,
            depth,
            relative_velocity,
        });
    }

    /// Report a trigger overlap
    pub fn report_trigger(&mut self, trigger_body: usize, other_body: usize) {
        let pair = normalize_pair(trigger_body, other_body);
        // One report per pair per frame (detection runs once per substep since
        // 1.2.0); same rule as `report_contact`.
        if self.curr_triggers.contains_key(&pair) {
            return;
        }

        let prev = self
            .prev_triggers
            .binary_search_by_key(&pair, |&(k, _)| k)
            .ok()
            .map(|i| self.prev_triggers[i].1);
        // Roles stay those of the enter frame while the overlap lasts, so the
        // exit event names the same trigger body as the enter event even if a
        // later frame reports the pair in the other order.
        self.curr_triggers
            .insert(pair, prev.unwrap_or((trigger_body, other_body)));

        if prev.is_none() {
            self.trigger_events.push(TriggerEvent {
                trigger_body,
                other_body,
                entered: true,
            });
        }
    }

    /// Finalize frame: generate End events for contacts/triggers that stopped
    pub fn end_frame(&mut self) {
        // Contact end events
        for &pair in &self.prev_pairs {
            if !self.curr_pairs.contains(&pair) {
                self.contact_events.push(ContactEvent {
                    body_a: pair.0,
                    body_b: pair.1,
                    event_type: ContactEventType::End,
                    normal: Vec3Fix::ZERO,
                    point: Vec3Fix::ZERO,
                    depth: Fix128::ZERO,
                    relative_velocity: Fix128::ZERO,
                });
            }
        }

        // Trigger exit events
        for &(pair, (trigger_body, other_body)) in &self.prev_triggers {
            if !self.curr_triggers.contains_key(&pair) {
                // Roles as given in the frame that reported the overlap, so
                // the exit event names the same trigger body as the enter event.
                self.trigger_events.push(TriggerEvent {
                    trigger_body,
                    other_body,
                    entered: false,
                });
            }
        }
    }

    /// Get all contact events for this frame
    #[inline]
    #[must_use]
    pub fn contact_events(&self) -> &[ContactEvent] {
        &self.contact_events
    }

    /// Get all trigger events for this frame
    #[inline]
    #[must_use]
    pub fn trigger_events(&self) -> &[TriggerEvent] {
        &self.trigger_events
    }

    /// Get the joints broken this frame, in the order they were removed
    /// (substep by substep; within one substep from the highest index down)
    #[inline]
    #[must_use]
    pub fn joint_break_events(&self) -> &[JointBreakEvent] {
        &self.joint_break_events
    }

    /// Record a joint broken this frame
    pub(crate) fn report_joint_break(&mut self, event: JointBreakEvent) {
        self.joint_break_events.push(event);
    }

    /// Drain joint break events (consumes them)
    #[inline]
    pub fn drain_joint_break_events(&mut self) -> Vec<JointBreakEvent> {
        core::mem::take(&mut self.joint_break_events)
    }

    /// Drain contact events (consumes them)
    #[inline]
    pub fn drain_contact_events(&mut self) -> Vec<ContactEvent> {
        core::mem::take(&mut self.contact_events)
    }

    /// Drain trigger events (consumes them)
    #[inline]
    pub fn drain_trigger_events(&mut self) -> Vec<TriggerEvent> {
        core::mem::take(&mut self.trigger_events)
    }

    /// Check if there are any events this frame
    #[inline]
    #[must_use]
    pub fn has_events(&self) -> bool {
        !self.contact_events.is_empty()
            || !self.trigger_events.is_empty()
            || !self.joint_break_events.is_empty()
    }
}

impl Default for EventCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Normalize a body pair so that the smaller index is first (deterministic ordering)
#[inline]
const fn normalize_pair(a: usize, b: usize) -> (usize, usize) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 2 kg body hung from a static anchor under g = 8 transmits exactly
    /// m·g = 16 N on the first solve (substeps 1, damping 1), so a joint rated
    /// 15 N breaks through `PhysicsWorld::step` and the world's collector
    /// reports it once, with that force
    #[test]
    fn step_reports_a_broken_joint_with_its_reaction_force() {
        use crate::joint::{BallJoint, Joint};
        use crate::solver::{PhysicsConfig, PhysicsWorld, RigidBody};
        let config = PhysicsConfig {
            substeps: 1,
            gravity: Vec3Fix::new(Fix128::ZERO, Fix128::from_int(-8), Fix128::ZERO),
            damping: Fix128::ONE,
            ..PhysicsConfig::default()
        };
        let mut w = PhysicsWorld::new(config);
        w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
        w.add_body(RigidBody::new(Vec3Fix::ZERO, Fix128::from_int(2)));
        let joint = Joint::Ball(
            BallJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO)
                .with_break_force(Fix128::from_int(15)),
        );
        w.add_joint(joint);
        w.step(Fix128::from_ratio(1, 64));
        assert!(w.joints.is_empty());
        assert!(w.events.has_events());
        assert_eq!(
            w.events.joint_break_events(),
            &[JointBreakEvent {
                index: 0,
                joint,
                force: Fix128::from_int(16)
            }]
        );
        // the next frame starts empty: nothing is left to break
        w.step(Fix128::from_ratio(1, 64));
        assert!(w.events.joint_break_events().is_empty());
    }

    /// The collector keeps joint breaks for one frame: reported events are
    /// readable and drainable, `begin_frame` clears them, and they alone make
    /// `has_events` true
    #[test]
    fn joint_break_events_live_for_one_frame() {
        use crate::joint::{BallJoint, Joint};
        let event = JointBreakEvent {
            index: 3,
            joint: Joint::Ball(BallJoint::new(1, 2, Vec3Fix::ZERO, Vec3Fix::UNIT_Y)),
            force: Fix128::from_int(7),
        };
        let mut events = EventCollector::default();
        events.begin_frame();
        assert!(!events.has_events());
        events.report_joint_break(event);
        events.report_joint_break(JointBreakEvent { index: 1, ..event });
        assert!(events.has_events(), "a joint break alone is an event");
        assert_eq!(events.joint_break_events().len(), 2);
        assert_eq!(events.joint_break_events()[1].index, 1);
        assert_eq!(
            events.drain_joint_break_events(),
            vec![event, JointBreakEvent { index: 1, ..event }]
        );
        assert!(events.joint_break_events().is_empty() && !events.has_events());
        events.report_joint_break(event);
        events.begin_frame();
        assert!(
            events.joint_break_events().is_empty(),
            "begin_frame clears them"
        );
    }

    #[test]
    fn test_contact_begin() {
        let mut events = EventCollector::new();
        events.begin_frame();
        events.report_contact(
            0,
            1,
            Vec3Fix::UNIT_Y,
            Vec3Fix::ZERO,
            Fix128::ONE,
            Fix128::ZERO,
        );
        events.end_frame();

        assert_eq!(events.contact_events().len(), 1);
        assert_eq!(
            events.contact_events()[0].event_type,
            ContactEventType::Begin
        );
    }

    #[test]
    fn test_contact_persist() {
        let mut events = EventCollector::new();

        // Frame 1: begin
        events.begin_frame();
        events.report_contact(
            0,
            1,
            Vec3Fix::UNIT_Y,
            Vec3Fix::ZERO,
            Fix128::ONE,
            Fix128::ZERO,
        );
        events.end_frame();

        // Frame 2: persist
        events.begin_frame();
        events.report_contact(
            0,
            1,
            Vec3Fix::UNIT_Y,
            Vec3Fix::ZERO,
            Fix128::ONE,
            Fix128::ZERO,
        );
        events.end_frame();

        assert_eq!(
            events.contact_events()[0].event_type,
            ContactEventType::Persist
        );
    }

    #[test]
    fn test_contact_end() {
        let mut events = EventCollector::new();

        // Frame 1: begin
        events.begin_frame();
        events.report_contact(
            0,
            1,
            Vec3Fix::UNIT_Y,
            Vec3Fix::ZERO,
            Fix128::ONE,
            Fix128::ZERO,
        );
        events.end_frame();

        // Frame 2: no contact => end event
        events.begin_frame();
        events.end_frame();

        let end_events: Vec<_> = events
            .contact_events()
            .iter()
            .filter(|e| e.event_type == ContactEventType::End)
            .collect();
        assert_eq!(end_events.len(), 1);
        assert_eq!(end_events[0].body_a, 0);
        assert_eq!(end_events[0].body_b, 1);
    }

    #[test]
    fn test_trigger_enter_exit() {
        let mut events = EventCollector::new();

        // Frame 1: trigger enter
        events.begin_frame();
        events.report_trigger(0, 1);
        events.end_frame();

        assert_eq!(events.trigger_events().len(), 1);
        assert!(events.trigger_events()[0].entered);

        // Frame 2: still overlapping - no new event
        events.begin_frame();
        events.report_trigger(0, 1);
        events.end_frame();

        // Only trigger exit events would be new; no enter since it persists
        let enters = events.trigger_events().iter().filter(|e| e.entered).count();
        assert_eq!(enters, 0);

        // Frame 3: trigger exit
        events.begin_frame();
        events.end_frame();

        let exits = events
            .trigger_events()
            .iter()
            .filter(|e| !e.entered)
            .count();
        assert_eq!(exits, 1);
    }

    #[test]
    fn test_pair_normalization() {
        let mut events = EventCollector::new();
        events.begin_frame();
        // Report in both orders
        events.report_contact(
            3,
            1,
            Vec3Fix::UNIT_Y,
            Vec3Fix::ZERO,
            Fix128::ONE,
            Fix128::ZERO,
        );
        events.end_frame();

        // Should be normalized to (1, 3)
        assert_eq!(events.contact_events()[0].body_a, 1);
        assert_eq!(events.contact_events()[0].body_b, 3);
    }

    #[test]
    fn has_events_tracks_contact_and_trigger_queues_across_frame_lifecycle() {
        let mut events = EventCollector::new();
        assert!(!events.has_events());
        events.begin_frame();
        assert!(!events.has_events());

        // contact だけ → true、drain で空になれば false
        events.report_contact(
            0,
            1,
            Vec3Fix::UNIT_Y,
            Vec3Fix::ZERO,
            Fix128::ONE,
            Fix128::ZERO,
        );
        assert!(events.has_events());
        assert_eq!(events.drain_contact_events().len(), 1);
        assert!(!events.has_events());

        // trigger だけ → true (contact 側が空でも)
        events.report_trigger(5, 6);
        assert!(events.has_events());
        assert!(events.contact_events().is_empty());
        assert_eq!(events.drain_trigger_events().len(), 1);
        assert!(!events.has_events());

        // 同じ pair の再報告は event を生まない (curr_pairs 重複) → false のまま
        events.report_contact(
            1,
            0,
            Vec3Fix::UNIT_Y,
            Vec3Fix::ZERO,
            Fix128::ONE,
            Fix128::ZERO,
        );
        assert!(!events.has_events());
        events.end_frame();
        assert!(!events.has_events(), "全 pair 継続中なので End/exit なし");

        // 次 frame: 何も報告せず end_frame → End 1 + exit 1 が生成され true
        events.begin_frame();
        assert!(!events.has_events(), "begin_frame は queue を空にする");
        events.end_frame();
        assert!(events.has_events());
        assert_eq!(events.contact_events().len(), 1);
        assert_eq!(events.contact_events()[0].event_type, ContactEventType::End);
        assert_eq!(events.trigger_events().len(), 1);
        assert!(!events.trigger_events()[0].entered);

        // 更に次 frame: 完全に静か → false
        events.begin_frame();
        events.end_frame();
        assert!(!events.has_events());
    }
}
