//! Physics Event System
//!
//! Provides collision event reporting (begin/persist/end) and trigger events.
//! Events are collected during `step()` and can be consumed after each frame.

use crate::math::{Fix128, Vec3Fix};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

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
#[derive(Clone, Copy, Debug)]
pub struct ContactEvent {
    /// First body index
    pub body_a: usize,
    /// Second body index
    pub body_b: usize,
    /// Event type
    pub event_type: ContactEventType,
    /// Contact normal (A to B)
    pub normal: Vec3Fix,
    /// Contact point (world space)
    pub point: Vec3Fix,
    /// Penetration depth
    pub depth: Fix128,
    /// Relative velocity along normal at contact point
    pub relative_velocity: Fix128,
}

/// A trigger event (overlap without physics response)
#[derive(Clone, Copy, Debug)]
pub struct TriggerEvent {
    /// Trigger body index
    pub trigger_body: usize,
    /// Other body index
    pub other_body: usize,
    /// Whether this is an enter or exit event
    pub entered: bool,
}

/// Manages physics events for one simulation step
pub struct EventCollector {
    /// Contact events this frame
    contact_events: Vec<ContactEvent>,
    /// Trigger events this frame
    trigger_events: Vec<TriggerEvent>,
    /// Active contact pairs from previous frame (for begin/persist/end tracking)
    prev_pairs: Vec<(usize, usize)>,
    /// Active contact pairs this frame
    curr_pairs: Vec<(usize, usize)>,
    /// Active trigger overlaps from previous frame
    prev_triggers: Vec<(usize, usize)>,
    /// Active trigger overlaps this frame
    curr_triggers: Vec<(usize, usize)>,
}

impl EventCollector {
    /// Create a new event collector
    pub fn new() -> Self {
        Self {
            contact_events: Vec::new(),
            trigger_events: Vec::new(),
            prev_pairs: Vec::new(),
            curr_pairs: Vec::new(),
            prev_triggers: Vec::new(),
            curr_triggers: Vec::new(),
        }
    }

    /// Begin a new frame: swap previous/current pair tracking
    pub fn begin_frame(&mut self) {
        self.contact_events.clear();
        self.trigger_events.clear();
        core::mem::swap(&mut self.prev_pairs, &mut self.curr_pairs);
        self.curr_pairs.clear();
        core::mem::swap(&mut self.prev_triggers, &mut self.curr_triggers);
        self.curr_triggers.clear();
        // Sort prev lists for binary_search lookups
        self.prev_pairs.sort_unstable();
        self.prev_triggers.sort_unstable();
    }

    /// Report a contact between two bodies
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
        let was_active = self.prev_pairs.binary_search(&pair).is_ok();
        let already_reported = self.curr_pairs.contains(&pair);

        if !already_reported {
            self.curr_pairs.push(pair);
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
        if !self.curr_triggers.contains(&pair) {
            self.curr_triggers.push(pair);
        }

        let was_active = self.prev_triggers.binary_search(&pair).is_ok();
        if !was_active {
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
        for &pair in &self.prev_triggers {
            if !self.curr_triggers.contains(&pair) {
                self.trigger_events.push(TriggerEvent {
                    trigger_body: pair.0,
                    other_body: pair.1,
                    entered: false,
                });
            }
        }
    }

    /// Get all contact events for this frame
    #[inline]
    pub fn contact_events(&self) -> &[ContactEvent] {
        &self.contact_events
    }

    /// Get all trigger events for this frame
    #[inline]
    pub fn trigger_events(&self) -> &[TriggerEvent] {
        &self.trigger_events
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
    pub fn has_events(&self) -> bool {
        !self.contact_events.is_empty() || !self.trigger_events.is_empty()
    }
}

impl Default for EventCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Normalize a body pair so that the smaller index is first (deterministic ordering)
#[inline]
fn normalize_pair(a: usize, b: usize) -> (usize, usize) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let enters: Vec<_> = events
            .trigger_events()
            .iter()
            .filter(|e| e.entered)
            .collect();
        assert_eq!(enters.len(), 0);

        // Frame 3: trigger exit
        events.begin_frame();
        events.end_frame();

        let exits: Vec<_> = events
            .trigger_events()
            .iter()
            .filter(|e| !e.entered)
            .collect();
        assert_eq!(exits.len(), 1);
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
}
