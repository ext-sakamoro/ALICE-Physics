//! Scene Serialization and Deserialization
//!
//! Save and load physics scenes in binary or JSON format.
//! All data is stored as raw fixed-point values (hi/lo pairs) to preserve
//! bit-exact determinism across platforms.
//!
//! # Binary Format
//!
//! ```text
//! Magic:  "APHYS\0" (6 bytes)
//! Version: u32 LE
//! Body count: u32 LE
//! Joint count: u32 LE
//! Config: substeps(u32), iterations(u32), gravity(6xi64), damping(i64+u64)
//! Bodies: [SerializedBody; body_count]
//! Joints: [SerializedJoint; joint_count]
//! ```
//!
//! # JSON Format
//!
//! Manual JSON formatting without serde dependency. Each Fix128 value is
//! stored as a `[hi, lo]` array for deterministic round-tripping.
//!
//! # Feature Gate
//!
//! This module requires the `std` feature (file I/O).

use crate::math::{Fix128, Vec3Fix};

use std::io::{Read, Write};

// ============================================================================
// Scene Types
// ============================================================================

/// A complete physics scene for serialization.
#[derive(Clone, Debug)]
pub struct PhysicsScene {
    /// Serialized rigid bodies
    pub bodies: Vec<SerializedBody>,
    /// Serialized joints
    pub joints: Vec<SerializedJoint>,
    /// Solver configuration
    pub config: PhysicsConfig,
    /// Format version
    pub version: u32,
}

/// Serialized rigid body (raw fixed-point data).
///
/// Position and velocity are stored as 6 i64 values:
/// `[x.hi, x.lo_as_i64, y.hi, y.lo_as_i64, z.hi, z.lo_as_i64]`
#[derive(Clone, Debug)]
pub struct SerializedBody {
    /// Position: x.hi, x.lo, y.hi, y.lo, z.hi, z.lo
    pub position: [i64; 6],
    /// Velocity: x.hi, x.lo, y.hi, y.lo, z.hi, z.lo
    pub velocity: [i64; 6],
    /// Rotation quaternion: x.hi, x.lo, y.hi, y.lo, z.hi, z.lo, w.hi, w.lo
    pub rotation: [i64; 8],
    /// Mass: hi, lo
    pub mass: [i64; 2],
    /// Body type: 0=Dynamic, 1=Static, 2=Kinematic
    pub body_type: u8,
}

/// Serialized joint (raw fixed-point data).
#[derive(Clone, Debug)]
pub struct SerializedJoint {
    /// Index of body A
    pub body_a: u32,
    /// Index of body B
    pub body_b: u32,
    /// Joint type: 0=Ball, 1=Hinge, 2=Fixed, 3=Slider, 4=Spring
    pub joint_type: u8,
    /// Anchor on body A: x.hi, x.lo, y.hi, y.lo, z.hi, z.lo
    pub anchor_a: [i64; 6],
    /// Anchor on body B: x.hi, x.lo, y.hi, y.lo, z.hi, z.lo
    pub anchor_b: [i64; 6],
}

/// Serialized physics configuration.
#[derive(Clone, Debug)]
pub struct PhysicsConfig {
    /// Number of substeps
    pub substeps: u32,
    /// Number of iterations per substep
    pub iterations: u32,
    /// Gravity vector (6 i64 values)
    pub gravity: [i64; 6],
    /// Damping (hi, lo)
    pub damping: [i64; 2],
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        let grav = vec3fix_to_raw(Vec3Fix::new(
            Fix128::ZERO,
            Fix128::from_int(-10),
            Fix128::ZERO,
        ));
        let damp = fix128_to_raw(Fix128::from_ratio(99, 100));
        Self {
            substeps: 8,
            iterations: 4,
            gravity: grav,
            damping: damp,
        }
    }
}

/// Magic bytes for the binary format header.
const MAGIC: &[u8; 6] = b"APHYS\0";

/// Current format version.
const CURRENT_VERSION: u32 = 1;

// ============================================================================
// Conversion Helpers
// ============================================================================

fn fix128_to_raw(v: Fix128) -> [i64; 2] {
    [v.hi, v.lo as i64]
}

#[cfg(test)]
fn raw_to_fix128(r: &[i64; 2]) -> Fix128 {
    Fix128 {
        hi: r[0],
        lo: r[1] as u64,
    }
}

fn vec3fix_to_raw(v: Vec3Fix) -> [i64; 6] {
    [
        v.x.hi,
        v.x.lo as i64,
        v.y.hi,
        v.y.lo as i64,
        v.z.hi,
        v.z.lo as i64,
    ]
}

#[cfg(test)]
fn raw_to_vec3fix(r: &[i64; 6]) -> Vec3Fix {
    Vec3Fix::new(
        Fix128 {
            hi: r[0],
            lo: r[1] as u64,
        },
        Fix128 {
            hi: r[2],
            lo: r[3] as u64,
        },
        Fix128 {
            hi: r[4],
            lo: r[5] as u64,
        },
    )
}

// ============================================================================
// Binary I/O Helpers
// ============================================================================

fn write_u32(w: &mut dyn Write, v: u32) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_u8(w: &mut dyn Write, v: u8) -> std::io::Result<()> {
    w.write_all(&[v])
}

fn write_i64(w: &mut dyn Write, v: i64) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn read_u32(r: &mut dyn Read) -> std::io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u8(r: &mut dyn Read) -> std::io::Result<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_i64(r: &mut dyn Read) -> std::io::Result<i64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(i64::from_le_bytes(buf))
}

fn write_i64_array(w: &mut dyn Write, arr: &[i64]) -> std::io::Result<()> {
    for &v in arr {
        write_i64(w, v)?;
    }
    Ok(())
}

fn read_i64_array<const N: usize>(r: &mut dyn Read) -> std::io::Result<[i64; N]> {
    let mut arr = [0i64; N];
    for item in &mut arr {
        *item = read_i64(r)?;
    }
    Ok(arr)
}

// ============================================================================
// Binary Format
// ============================================================================

/// Save a physics scene to a binary file.
///
/// Format: magic, version, counts, config, bodies, joints.
pub fn save_scene(scene: &PhysicsScene, path: &std::path::Path) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    write_scene_binary(&mut file, scene)
}

/// Load a physics scene from a binary file.
pub fn load_scene(path: &std::path::Path) -> std::io::Result<PhysicsScene> {
    let mut file = std::fs::File::open(path)?;
    read_scene_binary(&mut file)
}

fn write_scene_binary(w: &mut dyn Write, scene: &PhysicsScene) -> std::io::Result<()> {
    // Magic
    w.write_all(MAGIC)?;

    // Version
    write_u32(w, scene.version)?;

    // Counts
    write_u32(w, scene.bodies.len() as u32)?;
    write_u32(w, scene.joints.len() as u32)?;

    // Config
    write_u32(w, scene.config.substeps)?;
    write_u32(w, scene.config.iterations)?;
    write_i64_array(w, &scene.config.gravity)?;
    write_i64_array(w, &scene.config.damping)?;

    // Bodies
    for body in &scene.bodies {
        write_i64_array(w, &body.position)?;
        write_i64_array(w, &body.velocity)?;
        write_i64_array(w, &body.rotation)?;
        write_i64_array(w, &body.mass)?;
        write_u8(w, body.body_type)?;
    }

    // Joints
    for joint in &scene.joints {
        write_u32(w, joint.body_a)?;
        write_u32(w, joint.body_b)?;
        write_u8(w, joint.joint_type)?;
        write_i64_array(w, &joint.anchor_a)?;
        write_i64_array(w, &joint.anchor_b)?;
    }

    Ok(())
}

fn read_scene_binary(r: &mut dyn Read) -> std::io::Result<PhysicsScene> {
    // Magic
    let mut magic = [0u8; 6];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Invalid magic bytes: expected APHYS\\0",
        ));
    }

    // Version
    let version = read_u32(r)?;

    // Counts
    let body_count = read_u32(r)? as usize;
    let joint_count = read_u32(r)? as usize;

    // Config
    let substeps = read_u32(r)?;
    let iterations = read_u32(r)?;
    let gravity = read_i64_array::<6>(r)?;
    let damping = read_i64_array::<2>(r)?;

    let config = PhysicsConfig {
        substeps,
        iterations,
        gravity,
        damping,
    };

    // Bodies
    let mut bodies = Vec::with_capacity(body_count);
    for _ in 0..body_count {
        let position = read_i64_array::<6>(r)?;
        let velocity = read_i64_array::<6>(r)?;
        let rotation = read_i64_array::<8>(r)?;
        let mass = read_i64_array::<2>(r)?;
        let body_type = read_u8(r)?;
        bodies.push(SerializedBody {
            position,
            velocity,
            rotation,
            mass,
            body_type,
        });
    }

    // Joints
    let mut joints = Vec::with_capacity(joint_count);
    for _ in 0..joint_count {
        let body_a = read_u32(r)?;
        let body_b = read_u32(r)?;
        let joint_type = read_u8(r)?;
        let anchor_a = read_i64_array::<6>(r)?;
        let anchor_b = read_i64_array::<6>(r)?;
        joints.push(SerializedJoint {
            body_a,
            body_b,
            joint_type,
            anchor_a,
            anchor_b,
        });
    }

    Ok(PhysicsScene {
        bodies,
        joints,
        config,
        version,
    })
}

// ============================================================================
// JSON Format (manual, no serde)
// ============================================================================

/// Save a physics scene to a JSON file (human-readable).
pub fn save_scene_json(scene: &PhysicsScene, path: &std::path::Path) -> std::io::Result<()> {
    let json = scene_to_json(scene);
    std::fs::write(path, json)
}

/// Load a physics scene from a JSON file.
pub fn load_scene_json(path: &std::path::Path) -> std::io::Result<PhysicsScene> {
    let json = std::fs::read_to_string(path)?;
    parse_scene_json(&json).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

fn i64_array_to_json(arr: &[i64]) -> String {
    let items: Vec<String> = arr.iter().map(|v| format!("{}", v)).collect();
    format!("[{}]", items.join(", "))
}

fn scene_to_json(scene: &PhysicsScene) -> String {
    let mut s = String::new();
    s.push_str("{\n");
    s.push_str(&format!("  \"version\": {},\n", scene.version));

    // Config
    s.push_str("  \"config\": {\n");
    s.push_str(&format!("    \"substeps\": {},\n", scene.config.substeps));
    s.push_str(&format!(
        "    \"iterations\": {},\n",
        scene.config.iterations
    ));
    s.push_str(&format!(
        "    \"gravity\": {},\n",
        i64_array_to_json(&scene.config.gravity)
    ));
    s.push_str(&format!(
        "    \"damping\": {}\n",
        i64_array_to_json(&scene.config.damping)
    ));
    s.push_str("  },\n");

    // Bodies
    s.push_str("  \"bodies\": [\n");
    for (i, body) in scene.bodies.iter().enumerate() {
        s.push_str("    {\n");
        s.push_str(&format!(
            "      \"position\": {},\n",
            i64_array_to_json(&body.position)
        ));
        s.push_str(&format!(
            "      \"velocity\": {},\n",
            i64_array_to_json(&body.velocity)
        ));
        s.push_str(&format!(
            "      \"rotation\": {},\n",
            i64_array_to_json(&body.rotation)
        ));
        s.push_str(&format!(
            "      \"mass\": {},\n",
            i64_array_to_json(&body.mass)
        ));
        s.push_str(&format!("      \"body_type\": {}\n", body.body_type));
        if i < scene.bodies.len() - 1 {
            s.push_str("    },\n");
        } else {
            s.push_str("    }\n");
        }
    }
    s.push_str("  ],\n");

    // Joints
    s.push_str("  \"joints\": [\n");
    for (i, joint) in scene.joints.iter().enumerate() {
        s.push_str("    {\n");
        s.push_str(&format!("      \"body_a\": {},\n", joint.body_a));
        s.push_str(&format!("      \"body_b\": {},\n", joint.body_b));
        s.push_str(&format!("      \"joint_type\": {},\n", joint.joint_type));
        s.push_str(&format!(
            "      \"anchor_a\": {},\n",
            i64_array_to_json(&joint.anchor_a)
        ));
        s.push_str(&format!(
            "      \"anchor_b\": {}\n",
            i64_array_to_json(&joint.anchor_b)
        ));
        if i < scene.joints.len() - 1 {
            s.push_str("    },\n");
        } else {
            s.push_str("    }\n");
        }
    }
    s.push_str("  ]\n");

    s.push_str("}\n");
    s
}

// ============================================================================
// JSON Parser (minimal, no external dependencies)
// ============================================================================

fn parse_scene_json(json: &str) -> Result<PhysicsScene, String> {
    let json = json.trim();
    if !json.starts_with('{') || !json.ends_with('}') {
        return Err("Expected JSON object".into());
    }

    let version = extract_u32(json, "version").unwrap_or(CURRENT_VERSION);

    // Config
    let config_str = extract_object(json, "config").unwrap_or_default();
    let substeps = extract_u32(&config_str, "substeps").unwrap_or(8);
    let iterations = extract_u32(&config_str, "iterations").unwrap_or(4);
    let gravity = extract_i64_array(&config_str, "gravity", 6)?;
    let damping = extract_i64_array(&config_str, "damping", 2)?;

    let config = PhysicsConfig {
        substeps,
        iterations,
        gravity: [
            gravity[0], gravity[1], gravity[2], gravity[3], gravity[4], gravity[5],
        ],
        damping: [damping[0], damping[1]],
    };

    // Bodies
    let bodies_str = extract_array(json, "bodies").unwrap_or_default();
    let body_objects = split_array_objects(&bodies_str);
    let mut bodies = Vec::new();
    for obj in &body_objects {
        let position_v = extract_i64_array(obj, "position", 6)?;
        let velocity_v = extract_i64_array(obj, "velocity", 6)?;
        let rotation_v = extract_i64_array(obj, "rotation", 8)?;
        let mass_v = extract_i64_array(obj, "mass", 2)?;
        let body_type = extract_u32(obj, "body_type").unwrap_or(0) as u8;

        let mut position = [0i64; 6];
        let mut velocity = [0i64; 6];
        let mut rotation = [0i64; 8];
        let mut mass = [0i64; 2];
        position.copy_from_slice(&position_v);
        velocity.copy_from_slice(&velocity_v);
        rotation.copy_from_slice(&rotation_v);
        mass.copy_from_slice(&mass_v);

        bodies.push(SerializedBody {
            position,
            velocity,
            rotation,
            mass,
            body_type,
        });
    }

    // Joints
    let joints_str = extract_array(json, "joints").unwrap_or_default();
    let joint_objects = split_array_objects(&joints_str);
    let mut joints = Vec::new();
    for obj in &joint_objects {
        let body_a = extract_u32(obj, "body_a").unwrap_or(0);
        let body_b = extract_u32(obj, "body_b").unwrap_or(0);
        let joint_type = extract_u32(obj, "joint_type").unwrap_or(0) as u8;
        let anchor_a_v = extract_i64_array(obj, "anchor_a", 6)?;
        let anchor_b_v = extract_i64_array(obj, "anchor_b", 6)?;

        let mut anchor_a = [0i64; 6];
        let mut anchor_b = [0i64; 6];
        anchor_a.copy_from_slice(&anchor_a_v);
        anchor_b.copy_from_slice(&anchor_b_v);

        joints.push(SerializedJoint {
            body_a,
            body_b,
            joint_type,
            anchor_a,
            anchor_b,
        });
    }

    Ok(PhysicsScene {
        bodies,
        joints,
        config,
        version,
    })
}

// ============================================================================
// Minimal JSON extraction helpers
// ============================================================================

fn extract_u32(json: &str, key: &str) -> Option<u32> {
    let pattern = format!("\"{}\"", key);
    let idx = json.find(&pattern)?;
    let rest = &json[idx + pattern.len()..];
    let colon = rest.find(':')?;
    let after_colon = rest[colon + 1..].trim_start();
    // Read digits (possibly negative, but u32 so just digits)
    let end = after_colon
        .find(|c: char| !c.is_ascii_digit() && c != '-')
        .unwrap_or(after_colon.len());
    after_colon[..end].trim().parse().ok()
}

fn extract_object(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{}\"", key);
    let idx = json.find(&pattern)?;
    let rest = &json[idx + pattern.len()..];
    let brace = rest.find('{')?;
    let start = brace;
    let mut depth = 0i32;
    let bytes = rest.as_bytes();
    for (i, &b) in bytes[start..].iter().enumerate() {
        if b == b'{' {
            depth += 1;
        }
        if b == b'}' {
            depth -= 1;
        }
        if depth == 0 {
            return Some(rest[start..start + i + 1].to_string());
        }
    }
    None
}

fn extract_array(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{}\"", key);
    let idx = json.find(&pattern)?;
    let rest = &json[idx + pattern.len()..];
    // Find the opening [ that follows the colon
    let colon = rest.find(':')?;
    let after_colon = &rest[colon + 1..];
    let bracket = after_colon.find('[')?;
    let start = bracket;
    let mut depth = 0i32;
    let bytes = after_colon.as_bytes();
    for (i, &b) in bytes[start..].iter().enumerate() {
        if b == b'[' {
            depth += 1;
        }
        if b == b']' {
            depth -= 1;
        }
        if depth == 0 {
            return Some(after_colon[start..start + i + 1].to_string());
        }
    }
    None
}

fn extract_i64_array(json: &str, key: &str, expected_len: usize) -> Result<Vec<i64>, String> {
    let arr_str = extract_array(json, key).ok_or_else(|| format!("Missing key: {}", key))?;
    // Parse [n1, n2, ...]
    let inner = arr_str.trim_start_matches('[').trim_end_matches(']');
    let values: Result<Vec<i64>, _> = inner.split(',').map(|s| s.trim().parse::<i64>()).collect();
    let values = values.map_err(|e| format!("Parse error for {}: {}", key, e))?;
    if values.len() != expected_len {
        return Err(format!(
            "Expected {} values for {}, got {}",
            expected_len,
            key,
            values.len()
        ));
    }
    Ok(values)
}

fn split_array_objects(arr_str: &str) -> Vec<String> {
    let inner = arr_str.trim_start_matches('[').trim_end_matches(']').trim();
    if inner.is_empty() {
        return Vec::new();
    }

    let mut objects = Vec::new();
    let mut depth = 0i32;
    let mut start = 0;
    let bytes = inner.as_bytes();

    for (i, &b) in bytes.iter().enumerate() {
        if b == b'{' {
            if depth == 0 {
                start = i;
            }
            depth += 1;
        }
        if b == b'}' {
            depth -= 1;
            if depth == 0 {
                objects.push(inner[start..=i].to_string());
            }
        }
    }

    objects
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_scene() -> PhysicsScene {
        PhysicsScene {
            version: CURRENT_VERSION,
            config: PhysicsConfig::default(),
            bodies: vec![
                SerializedBody {
                    position: [0, 0, 10, 0, 0, 0],
                    velocity: [0, 0, 0, 0, 0, 0],
                    rotation: [0, 0, 0, 0, 0, 0, 1, 0],
                    mass: [1, 0],
                    body_type: 0,
                },
                SerializedBody {
                    position: [5, 0, 0, 0, -3, 0],
                    velocity: [1, 0, -1, 0, 0, 0],
                    rotation: [0, 0, 0, 0, 0, 0, 1, 0],
                    mass: [2, 0],
                    body_type: 1,
                },
            ],
            joints: vec![SerializedJoint {
                body_a: 0,
                body_b: 1,
                joint_type: 0,
                anchor_a: [0, 0, 0, 0, 0, 0],
                anchor_b: [1, 0, 0, 0, 0, 0],
            }],
        }
    }

    #[test]
    fn test_binary_roundtrip() {
        let scene = make_test_scene();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.aphys");

        save_scene(&scene, &path).unwrap();
        let loaded = load_scene(&path).unwrap();

        assert_eq!(loaded.version, scene.version);
        assert_eq!(loaded.bodies.len(), scene.bodies.len());
        assert_eq!(loaded.joints.len(), scene.joints.len());
        assert_eq!(loaded.config.substeps, scene.config.substeps);
    }

    #[test]
    fn test_binary_body_data_preserved() {
        let scene = make_test_scene();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test2.aphys");

        save_scene(&scene, &path).unwrap();
        let loaded = load_scene(&path).unwrap();

        assert_eq!(loaded.bodies[0].position, scene.bodies[0].position);
        assert_eq!(loaded.bodies[0].mass, scene.bodies[0].mass);
        assert_eq!(loaded.bodies[0].body_type, scene.bodies[0].body_type);
        assert_eq!(loaded.bodies[1].velocity, scene.bodies[1].velocity);
    }

    #[test]
    fn test_binary_joint_data_preserved() {
        let scene = make_test_scene();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test3.aphys");

        save_scene(&scene, &path).unwrap();
        let loaded = load_scene(&path).unwrap();

        assert_eq!(loaded.joints[0].body_a, 0);
        assert_eq!(loaded.joints[0].body_b, 1);
        assert_eq!(loaded.joints[0].joint_type, 0);
        assert_eq!(loaded.joints[0].anchor_b, scene.joints[0].anchor_b);
    }

    #[test]
    fn test_binary_invalid_magic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.aphys");
        std::fs::write(&path, b"BADMG\0").unwrap();

        let result = load_scene(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_json_roundtrip() {
        let scene = make_test_scene();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.json");

        save_scene_json(&scene, &path).unwrap();
        let loaded = load_scene_json(&path).unwrap();

        assert_eq!(loaded.version, scene.version);
        assert_eq!(loaded.bodies.len(), scene.bodies.len());
        assert_eq!(loaded.joints.len(), scene.joints.len());
    }

    #[test]
    fn test_json_body_data_preserved() {
        let scene = make_test_scene();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test2.json");

        save_scene_json(&scene, &path).unwrap();
        let loaded = load_scene_json(&path).unwrap();

        assert_eq!(loaded.bodies[0].position, scene.bodies[0].position);
        assert_eq!(loaded.bodies[1].velocity, scene.bodies[1].velocity);
        assert_eq!(loaded.bodies[0].rotation, scene.bodies[0].rotation);
    }

    #[test]
    fn test_json_config_preserved() {
        let scene = make_test_scene();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test3.json");

        save_scene_json(&scene, &path).unwrap();
        let loaded = load_scene_json(&path).unwrap();

        assert_eq!(loaded.config.substeps, scene.config.substeps);
        assert_eq!(loaded.config.iterations, scene.config.iterations);
        assert_eq!(loaded.config.gravity, scene.config.gravity);
        assert_eq!(loaded.config.damping, scene.config.damping);
    }

    #[test]
    fn test_json_empty_scene() {
        let scene = PhysicsScene {
            version: CURRENT_VERSION,
            config: PhysicsConfig::default(),
            bodies: vec![],
            joints: vec![],
        };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.json");

        save_scene_json(&scene, &path).unwrap();
        let loaded = load_scene_json(&path).unwrap();

        assert!(loaded.bodies.is_empty());
        assert!(loaded.joints.is_empty());
    }

    #[test]
    fn test_binary_empty_scene() {
        let scene = PhysicsScene {
            version: CURRENT_VERSION,
            config: PhysicsConfig::default(),
            bodies: vec![],
            joints: vec![],
        };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.aphys");

        save_scene(&scene, &path).unwrap();
        let loaded = load_scene(&path).unwrap();

        assert!(loaded.bodies.is_empty());
        assert!(loaded.joints.is_empty());
    }

    #[test]
    fn test_fix128_raw_roundtrip() {
        let val = Fix128::from_ratio(355, 113); // approx pi
        let raw = fix128_to_raw(val);
        let restored = raw_to_fix128(&raw);
        assert_eq!(val.hi, restored.hi);
        assert_eq!(val.lo, restored.lo);
    }

    #[test]
    fn test_vec3fix_raw_roundtrip() {
        let val = Vec3Fix::new(Fix128::from_int(42), Fix128::from_ratio(-7, 3), Fix128::PI);
        let raw = vec3fix_to_raw(val);
        let restored = raw_to_vec3fix(&raw);
        assert_eq!(val.x.hi, restored.x.hi);
        assert_eq!(val.x.lo, restored.x.lo);
        assert_eq!(val.y.hi, restored.y.hi);
        assert_eq!(val.z.hi, restored.z.hi);
    }

    #[test]
    fn test_binary_negative_values() {
        let scene = PhysicsScene {
            version: CURRENT_VERSION,
            config: PhysicsConfig::default(),
            bodies: vec![SerializedBody {
                position: [-100, 0, -200, 0, -300, 0],
                velocity: [-1, 0, -2, 0, -3, 0],
                rotation: [0, 0, 0, 0, 0, 0, 1, 0],
                mass: [5, 0],
                body_type: 2,
            }],
            joints: vec![],
        };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("neg.aphys");

        save_scene(&scene, &path).unwrap();
        let loaded = load_scene(&path).unwrap();

        assert_eq!(loaded.bodies[0].position[0], -100);
        assert_eq!(loaded.bodies[0].position[2], -200);
        assert_eq!(loaded.bodies[0].body_type, 2);
    }

    #[test]
    fn test_json_negative_values() {
        let scene = PhysicsScene {
            version: CURRENT_VERSION,
            config: PhysicsConfig::default(),
            bodies: vec![SerializedBody {
                position: [-10, 0, -20, 0, -30, 0],
                velocity: [0, 0, 0, 0, 0, 0],
                rotation: [0, 0, 0, 0, 0, 0, 1, 0],
                mass: [1, 0],
                body_type: 0,
            }],
            joints: vec![],
        };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("neg.json");

        save_scene_json(&scene, &path).unwrap();
        let loaded = load_scene_json(&path).unwrap();

        assert_eq!(loaded.bodies[0].position[0], -10);
        assert_eq!(loaded.bodies[0].position[2], -20);
    }

    #[test]
    fn test_multiple_joints_roundtrip() {
        let scene = PhysicsScene {
            version: CURRENT_VERSION,
            config: PhysicsConfig::default(),
            bodies: vec![SerializedBody {
                position: [0; 6],
                velocity: [0; 6],
                rotation: [0, 0, 0, 0, 0, 0, 1, 0],
                mass: [1, 0],
                body_type: 0,
            }],
            joints: vec![
                SerializedJoint {
                    body_a: 0,
                    body_b: 0,
                    joint_type: 1,
                    anchor_a: [1, 0, 2, 0, 3, 0],
                    anchor_b: [4, 0, 5, 0, 6, 0],
                },
                SerializedJoint {
                    body_a: 0,
                    body_b: 0,
                    joint_type: 4,
                    anchor_a: [7, 0, 8, 0, 9, 0],
                    anchor_b: [10, 0, 11, 0, 12, 0],
                },
            ],
        };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("multi_joint.aphys");

        save_scene(&scene, &path).unwrap();
        let loaded = load_scene(&path).unwrap();

        assert_eq!(loaded.joints.len(), 2);
        assert_eq!(loaded.joints[0].joint_type, 1);
        assert_eq!(loaded.joints[1].joint_type, 4);
        assert_eq!(loaded.joints[1].anchor_a[0], 7);
    }

    #[test]
    fn test_scene_to_json_format() {
        let scene = make_test_scene();
        let json = scene_to_json(&scene);
        assert!(json.contains("\"version\""));
        assert!(json.contains("\"bodies\""));
        assert!(json.contains("\"joints\""));
        assert!(json.contains("\"config\""));
    }
}
