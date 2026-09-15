//! Audio-Physics Integration
//!
//! Generates physically-based audio parameters from collision events.
//! Maps impact velocity, material, surface area, and contact duration
//! to audio parameters (volume, pitch, decay, timbre).
//!
//! # Output
//!
//! This module does NOT generate audio samples. It computes parameters
//! that an audio engine can use to trigger and modulate sounds.
//!
//! Author: Moroya Sakamoto

use crate::collider::Contact;
use crate::math::{Fix128, Vec3Fix};

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ============================================================================
// Material
// ============================================================================

/// Physical material properties for audio synthesis
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AudioMaterial {
    /// Material type identifier
    pub material_type: MaterialType,
    /// Density (kg/m^3) — affects pitch/resonance
    pub density: Fix128,
    /// Hardness (0..1) — affects attack/brightness
    pub hardness: Fix128,
    /// Resonance (0..1) — how much the material rings
    pub resonance: Fix128,
    /// Damping (0..1) — how quickly sound decays
    pub damping: Fix128,
}

/// Material type for sound bank selection
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaterialType {
    /// Metal (bright, ringing)
    Metal,
    /// Wood (warm, short decay)
    Wood,
    /// Glass (bright, fragile)
    Glass,
    /// Stone/Concrete (dull, heavy)
    Stone,
    /// Rubber (soft, muffled)
    Rubber,
    /// Fabric/Cloth (very soft)
    Fabric,
    /// Water/Liquid (splash)
    Liquid,
    /// Custom material
    Custom(u32),
}

impl AudioMaterial {
    /// Metal preset
    pub const METAL: Self = Self {
        material_type: MaterialType::Metal,
        density: Fix128 { hi: 7800, lo: 0 },
        hardness: Fix128 {
            hi: 0,
            lo: 0xCCCCCCCCCCCCCCCC,
        }, // 0.8
        resonance: Fix128 {
            hi: 0,
            lo: 0xCCCCCCCCCCCCCCCC,
        }, // 0.8
        damping: Fix128 {
            hi: 0,
            lo: 0x3333333333333333,
        }, // 0.2
    };

    /// Wood preset
    pub const WOOD: Self = Self {
        material_type: MaterialType::Wood,
        density: Fix128 { hi: 600, lo: 0 },
        hardness: Fix128 {
            hi: 0,
            lo: 0x999999999999999A,
        }, // 0.6
        resonance: Fix128 {
            hi: 0,
            lo: 0x6666666666666666,
        }, // 0.4
        damping: Fix128 {
            hi: 0,
            lo: 0x999999999999999A,
        }, // 0.6
    };

    /// Stone preset
    pub const STONE: Self = Self {
        material_type: MaterialType::Stone,
        density: Fix128 { hi: 2500, lo: 0 },
        hardness: Fix128 {
            hi: 0,
            lo: 0xE666666666666666,
        }, // 0.9
        resonance: Fix128 {
            hi: 0,
            lo: 0x3333333333333333,
        }, // 0.2
        damping: Fix128 {
            hi: 0,
            lo: 0xB333333333333333,
        }, // 0.7
    };

    /// Rubber preset
    pub const RUBBER: Self = Self {
        material_type: MaterialType::Rubber,
        density: Fix128 { hi: 1100, lo: 0 },
        hardness: Fix128 {
            hi: 0,
            lo: 0x3333333333333333,
        }, // 0.2
        resonance: Fix128 {
            hi: 0,
            lo: 0x1999999999999999,
        }, // 0.1
        damping: Fix128 {
            hi: 0,
            lo: 0xE666666666666666,
        }, // 0.9
    };
}

// ============================================================================
// Audio Event
// ============================================================================

/// Audio parameters generated from a physics event
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AudioEvent {
    /// World-space position for 3D spatialization
    pub position: Vec3Fix,
    /// Volume (0..1, linear)
    pub volume: Fix128,
    /// Pitch multiplier (1.0 = normal, 0.5 = octave down, 2.0 = octave up)
    pub pitch: Fix128,
    /// Decay time in seconds
    pub decay: Fix128,
    /// Brightness (0..1, controls high-frequency content)
    pub brightness: Fix128,
    /// Roughness (0..1, 0 = clean, 1 = gritty/textured)
    pub roughness: Fix128,
    /// Impact type
    pub event_type: AudioEventType,
    /// Material A
    pub material_a: MaterialType,
    /// Material B
    pub material_b: MaterialType,
}

/// Type of audio event
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AudioEventType {
    /// Single impact (collision begin)
    Impact,
    /// Sliding contact (persistent)
    Slide,
    /// Rolling contact
    Roll,
    /// Scraping (high-friction slide)
    Scrape,
    /// Breaking/fracture
    Break,
    /// Splash (fluid)
    Splash,
}

// ============================================================================
// Audio Generator
// ============================================================================

/// Audio event generator configuration
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AudioConfig {
    /// Minimum impact velocity to generate sound
    pub min_velocity: Fix128,
    /// Maximum volume impact velocity (velocities above this = max volume)
    pub max_velocity: Fix128,
    /// Velocity to pitch mapping factor
    pub velocity_pitch_factor: Fix128,
    /// Minimum volume threshold (events below this are discarded)
    pub min_volume: Fix128,
    /// Sliding sound velocity threshold
    pub slide_velocity_threshold: Fix128,
    /// Maximum simultaneous audio events per frame
    pub max_events_per_frame: usize,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            min_velocity: Fix128::from_ratio(1, 10),
            max_velocity: Fix128::from_int(20),
            velocity_pitch_factor: Fix128::from_ratio(1, 20),
            min_volume: Fix128::from_ratio(1, 100),
            slide_velocity_threshold: Fix128::from_ratio(1, 2),
            max_events_per_frame: 32,
        }
    }
}

/// Audio event generator
pub struct AudioGenerator {
    /// Configuration
    pub config: AudioConfig,
    /// Per-body material (indexed by body ID)
    pub materials: Vec<AudioMaterial>,
    /// Generated audio events this frame
    pub events: Vec<AudioEvent>,
}

impl AudioGenerator {
    /// Create a new audio generator for N bodies
    #[must_use]
    pub fn new(num_bodies: usize, config: AudioConfig) -> Self {
        Self {
            config,
            materials: vec![AudioMaterial::WOOD; num_bodies],
            events: Vec::new(),
        }
    }

    /// Set material for a body
    pub fn set_material(&mut self, body_idx: usize, material: AudioMaterial) {
        if body_idx >= self.materials.len() {
            self.materials.resize(body_idx + 1, AudioMaterial::WOOD);
        }
        self.materials[body_idx] = material;
    }

    /// Clear events for new frame
    pub fn begin_frame(&mut self) {
        self.events.clear();
    }

    /// Process a collision contact and generate audio event
    pub fn process_contact(
        &mut self,
        body_a: usize,
        body_b: usize,
        contact: &Contact,
        relative_velocity: Vec3Fix,
        is_new_contact: bool,
    ) {
        if self.events.len() >= self.config.max_events_per_frame {
            return;
        }

        let speed = relative_velocity.length();
        if speed < self.config.min_velocity {
            return;
        }

        let mat_a = self.get_material(body_a);
        let mat_b = self.get_material(body_b);

        // Determine event type
        let normal_speed = relative_velocity.dot(contact.normal).abs();
        let tangential_sq = speed * speed - normal_speed * normal_speed;
        let tangential_speed = if tangential_sq > Fix128::ZERO {
            tangential_sq.sqrt()
        } else {
            Fix128::ZERO
        };

        let event_type = if is_new_contact {
            AudioEventType::Impact
        } else if tangential_speed > self.config.slide_velocity_threshold {
            AudioEventType::Slide
        } else {
            AudioEventType::Roll
        };

        // Compute audio parameters
        let volume = self.compute_volume(speed, &mat_a, &mat_b);
        let pitch = self.compute_pitch(speed, &mat_a, &mat_b);
        let decay = self.compute_decay(&mat_a, &mat_b);
        let brightness = self.compute_brightness(speed, &mat_a, &mat_b);
        let roughness = self.compute_roughness(tangential_speed, &mat_a, &mat_b);

        if volume < self.config.min_volume {
            return;
        }

        self.events.push(AudioEvent {
            position: contact.point_a,
            volume,
            pitch,
            decay,
            brightness,
            roughness,
            event_type,
            material_a: mat_a.material_type,
            material_b: mat_b.material_type,
        });
    }

    fn get_material(&self, body_idx: usize) -> AudioMaterial {
        if body_idx < self.materials.len() {
            self.materials[body_idx]
        } else {
            AudioMaterial::WOOD
        }
    }

    /// Volume: impact velocity mapped to 0..1 with sqrt curve
    fn compute_volume(
        &self,
        speed: Fix128,
        _mat_a: &AudioMaterial,
        _mat_b: &AudioMaterial,
    ) -> Fix128 {
        let normalized = speed / self.config.max_velocity;
        let clamped = if normalized > Fix128::ONE {
            Fix128::ONE
        } else {
            normalized
        };
        clamped.sqrt()
    }

    /// Pitch: higher velocity = slightly higher pitch, adjusted by material density
    fn compute_pitch(&self, speed: Fix128, mat_a: &AudioMaterial, mat_b: &AudioMaterial) -> Fix128 {
        let base = Fix128::ONE + speed * self.config.velocity_pitch_factor;

        // Denser materials = lower pitch
        let avg_density = (mat_a.density + mat_b.density).half();
        if avg_density.is_zero() {
            return base;
        }
        let density_factor = Fix128::from_int(1000) / avg_density;
        let density_factor = if density_factor > Fix128::from_int(2) {
            Fix128::from_int(2)
        } else if density_factor < Fix128::from_ratio(1, 2) {
            Fix128::from_ratio(1, 2)
        } else {
            density_factor
        };

        base * density_factor
    }

    /// Decay: based on material resonance and damping
    #[allow(clippy::unused_self)]
    fn compute_decay(&self, mat_a: &AudioMaterial, mat_b: &AudioMaterial) -> Fix128 {
        let avg_resonance = (mat_a.resonance + mat_b.resonance).half();
        let avg_damping = (mat_a.damping + mat_b.damping).half();

        // High resonance + low damping = long decay
        let base_decay = Fix128::from_ratio(1, 10); // 0.1s minimum
        let resonance_bonus = avg_resonance * Fix128::from_int(2);
        let damping_reduction = avg_damping;

        base_decay + resonance_bonus * (Fix128::ONE - damping_reduction)
    }

    /// Brightness: hardness and impact speed
    fn compute_brightness(
        &self,
        speed: Fix128,
        mat_a: &AudioMaterial,
        mat_b: &AudioMaterial,
    ) -> Fix128 {
        let avg_hardness = (mat_a.hardness + mat_b.hardness).half();
        let speed_factor = speed / self.config.max_velocity;
        let speed_factor = if speed_factor > Fix128::ONE {
            Fix128::ONE
        } else {
            speed_factor
        };

        avg_hardness * (Fix128::from_ratio(1, 2) + speed_factor.half())
    }

    /// Roughness: tangential velocity and surface hardness
    fn compute_roughness(
        &self,
        tangential_speed: Fix128,
        mat_a: &AudioMaterial,
        mat_b: &AudioMaterial,
    ) -> Fix128 {
        let avg_hardness = (mat_a.hardness + mat_b.hardness).half();
        let speed_factor = tangential_speed / self.config.max_velocity;
        let speed_factor = if speed_factor > Fix128::ONE {
            Fix128::ONE
        } else {
            speed_factor
        };

        speed_factor * avg_hardness
    }

    /// Get all audio events for this frame
    #[must_use]
    pub fn get_events(&self) -> &[AudioEvent] {
        &self.events
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_generator() {
        let mut gen = AudioGenerator::new(2, AudioConfig::default());
        gen.set_material(0, AudioMaterial::METAL);
        gen.set_material(1, AudioMaterial::STONE);

        gen.begin_frame();

        let contact = Contact {
            depth: Fix128::from_ratio(1, 10),
            normal: Vec3Fix::UNIT_Y,
            point_a: Vec3Fix::ZERO,
            point_b: Vec3Fix::ZERO,
        };

        gen.process_contact(0, 1, &contact, Vec3Fix::from_f32(0.0, -5.0, 0.0), true);

        assert_eq!(gen.events.len(), 1);
        let event = &gen.events[0];
        assert_eq!(event.event_type, AudioEventType::Impact);
        assert!(event.volume > Fix128::ZERO);
    }

    #[test]
    fn test_material_presets() {
        let metal = AudioMaterial::METAL;
        let rubber = AudioMaterial::RUBBER;

        // Metal should be harder and more resonant than rubber
        assert!(metal.hardness > rubber.hardness);
        assert!(metal.resonance > rubber.resonance);
    }

    #[test]
    fn test_volume_scaling() {
        let gen = AudioGenerator::new(2, AudioConfig::default());
        let mat = AudioMaterial::WOOD;

        let soft = gen.compute_volume(Fix128::from_int(1), &mat, &mat);
        let hard = gen.compute_volume(Fix128::from_int(10), &mat, &mat);

        assert!(hard > soft, "Harder impact should be louder");
    }

    #[test]
    fn test_below_threshold() {
        let mut gen = AudioGenerator::new(2, AudioConfig::default());
        gen.begin_frame();

        let contact = Contact {
            depth: Fix128::from_ratio(1, 100),
            normal: Vec3Fix::UNIT_Y,
            point_a: Vec3Fix::ZERO,
            point_b: Vec3Fix::ZERO,
        };

        // Very slow impact (below threshold)
        gen.process_contact(0, 1, &contact, Vec3Fix::from_f32(0.0, -0.01, 0.0), true);

        assert!(
            gen.events.is_empty(),
            "Below-threshold impacts should be silent"
        );
    }

    #[test]
    fn test_slide_detection() {
        let mut gen = AudioGenerator::new(2, AudioConfig::default());
        gen.begin_frame();

        let contact = Contact {
            depth: Fix128::from_ratio(1, 10),
            normal: Vec3Fix::UNIT_Y,
            point_a: Vec3Fix::ZERO,
            point_b: Vec3Fix::ZERO,
        };

        // Tangential velocity (sliding)
        gen.process_contact(
            0,
            1,
            &contact,
            Vec3Fix::from_f32(5.0, 0.0, 0.0), // X velocity, Y normal = sliding
            false,                            // not new contact
        );

        assert_eq!(gen.events.len(), 1);
        assert_eq!(gen.events[0].event_type, AudioEventType::Slide);
    }

    #[test]
    fn get_events_returns_this_frames_events_in_order_and_respects_cap() {
        let config = AudioConfig {
            max_events_per_frame: 2,
            ..AudioConfig::default()
        };
        let mut gen = AudioGenerator::new(3, config);
        gen.set_material(1, AudioMaterial::METAL);
        assert!(gen.get_events().is_empty());
        gen.begin_frame();

        let contact_at = |x: i64| Contact {
            depth: Fix128::from_ratio(1, 10),
            normal: Vec3Fix::UNIT_Y,
            point_a: Vec3Fix::from_int(x, 0, 0),
            point_b: Vec3Fix::from_int(x, -1, 0),
        };
        // 1: impact、speed 5 → volume = sqrt(5/20) = 1/2
        gen.process_contact(0, 1, &contact_at(1), Vec3Fix::from_int(0, -5, 0), true);
        // 2: slide (接線 5 > 1/2)、材質 WOOD × WOOD (body 2 は default)
        gen.process_contact(0, 2, &contact_at(2), Vec3Fix::from_int(5, 0, 0), false);
        // 3: cap (2) 超過で捨てられる
        gen.process_contact(1, 2, &contact_at(3), Vec3Fix::from_int(0, -9, 0), true);

        let events = gen.get_events();
        assert_eq!(events.len(), 2);
        assert_eq!(events, gen.events.as_slice());
        // 挿入順 (position = contact.point_a)
        assert_eq!(events[0].position, Vec3Fix::from_int(1, 0, 0));
        assert_eq!(events[1].position, Vec3Fix::from_int(2, 0, 0));
        assert_eq!(events[0].event_type, AudioEventType::Impact);
        assert_eq!(events[1].event_type, AudioEventType::Slide);
        assert_eq!(
            (events[0].material_a, events[0].material_b),
            (MaterialType::Wood, MaterialType::Metal)
        );
        assert_eq!(
            (events[1].material_a, events[1].material_b),
            (MaterialType::Wood, MaterialType::Wood)
        );
        let vol_err = (events[0].volume - Fix128::from_ratio(1, 2)).abs();
        assert!(
            vol_err < Fix128::from_ratio(1, 1_000_000),
            "volume {:?}",
            events[0].volume
        );
        // slide の roughness = (5/20) * avg_hardness、impact (接線 0) は 0
        assert_eq!(events[0].roughness, Fix128::ZERO);
        let wood = AudioMaterial::WOOD;
        assert_eq!(
            events[1].roughness,
            Fix128::from_ratio(1, 4) * wood.hardness
        );

        // 次 frame で空になり、再度 process すれば新しい slice
        gen.begin_frame();
        assert!(gen.get_events().is_empty());
        gen.process_contact(1, 2, &contact_at(3), Vec3Fix::from_int(0, -9, 0), true);
        assert_eq!(gen.get_events().len(), 1);
        assert_eq!(gen.get_events()[0].position, Vec3Fix::from_int(3, 0, 0));
    }
}
