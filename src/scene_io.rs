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

use std::fmt::Write as FmtWrite;
use std::io::{Read, Write};

// ============================================================================
// Scene Types
// ============================================================================

/// A complete physics scene for serialization.
///
/// Marked `#[non_exhaustive]` so additional scene sections (particle
/// systems, CFD grids, structural analysis) can be added post-v1.0
/// without a breaking API change.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
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

impl PhysicsScene {
    /// Assemble a scene from its parts.
    ///
    /// This is the only way to construct a `PhysicsScene` from outside the
    /// crate: the struct is `#[non_exhaustive]`, so a struct literal is
    /// rejected downstream (E0639). Pass [`CURRENT_SCENE_VERSION`] as
    /// `version` unless you are deliberately writing an older format.
    ///
    /// ```
    /// use alice_physics::scene_io::{PhysicsConfig, PhysicsScene, CURRENT_SCENE_VERSION};
    ///
    /// let scene = PhysicsScene::new(Vec::new(), Vec::new(), PhysicsConfig::default(), CURRENT_SCENE_VERSION);
    /// assert_eq!(scene.version, CURRENT_SCENE_VERSION);
    /// assert!(scene.bodies.is_empty());
    /// ```
    #[must_use]
    pub fn new(
        bodies: Vec<SerializedBody>,
        joints: Vec<SerializedJoint>,
        config: PhysicsConfig,
        version: u32,
    ) -> Self {
        Self {
            bodies,
            joints,
            config,
            version,
        }
    }
}

/// Serialized rigid body (raw fixed-point data).
///
/// Position and velocity are stored as 6 i64 values:
/// `[x.hi, x.lo_as_i64, y.hi, y.lo_as_i64, z.hi, z.lo_as_i64]`
#[derive(Clone, Debug, PartialEq, Eq)]
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
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SerializedJoint {
    /// Index of body A
    pub body_a: u32,
    /// Index of body B
    pub body_b: u32,
    // LIMITATION(COV-ENGINE-150): five of the seven joint kinds (no D6, no ConeTwist), and only the kind and two anchors are stored: axes, limits, motors and spring constants are not.
    /// Joint type: 0=Ball, 1=Hinge, 2=Fixed, 3=Slider, 4=Spring
    pub joint_type: u8,
    /// Anchor on body A: x.hi, x.lo, y.hi, y.lo, z.hi, z.lo
    pub anchor_a: [i64; 6],
    /// Anchor on body B: x.hi, x.lo, y.hi, y.lo, z.hi, z.lo
    pub anchor_b: [i64; 6],
}

/// Serialized physics configuration.
///
/// Marked `#[non_exhaustive]` so new solver parameters can be added
/// post-v1.0 without a breaking API change.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
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

impl PhysicsConfig {
    /// Build a configuration from raw serialized values.
    ///
    /// `gravity` and `damping` use the same raw `Fix128` limb layout as
    /// [`SerializedBody::position`] (`[hi, lo_as_i64, ...]`). Downstream
    /// crates need this because the struct is `#[non_exhaustive]`; for the
    /// engine defaults use [`PhysicsConfig::default`].
    #[must_use]
    pub const fn new(substeps: u32, iterations: u32, gravity: [i64; 6], damping: [i64; 2]) -> Self {
        Self {
            substeps,
            iterations,
            gravity,
            damping,
        }
    }
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

/// Upper bound on the entries reserved up front when reading a binary scene.
const MAX_PREALLOC: usize = 4096;

/// Current `.aphys` / JSON scene format version, for [`PhysicsScene::new`].
pub const CURRENT_SCENE_VERSION: u32 = CURRENT_VERSION;

/// Scene format versions this crate can actually read, as accepted by
/// [`load_scene`] / [`load_scene_json`].
///
/// Only version 1 exists today: the writer has never produced another layout,
/// so every other value names a format this reader does not know.
pub const SUPPORTED_SCENE_VERSIONS: &[u32] = &[CURRENT_VERSION];

/// Error payload for a scene whose `version` is not in
/// [`SUPPORTED_SCENE_VERSIONS`], returned by [`load_scene`] /
/// [`load_scene_json`].
///
/// It travels inside a [`std::io::Error`] of kind
/// [`std::io::ErrorKind::InvalidData`]; recover it with
/// `err.get_ref().and_then(|e| e.downcast_ref::<UnsupportedSceneVersion>())`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct UnsupportedSceneVersion {
    /// The version recorded in the scene.
    pub found: u32,
}

impl core::fmt::Display for UnsupportedSceneVersion {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "unsupported scene version {} (supported: {:?})",
            self.found, SUPPORTED_SCENE_VERSIONS
        )
    }
}

impl std::error::Error for UnsupportedSceneVersion {}

/// `Ok` when `version` is one this crate can read, otherwise an
/// `InvalidData` error carrying [`UnsupportedSceneVersion`].
fn check_scene_version(version: u32) -> std::io::Result<()> {
    if SUPPORTED_SCENE_VERSIONS.contains(&version) {
        Ok(())
    } else {
        Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            UnsupportedSceneVersion { found: version },
        ))
    }
}

/// Error payload for a JSON scene whose top-level `version` member cannot be
/// read, returned by [`load_scene_json`].
///
/// The loader reads `version` only as a member of the top-level object, after
/// checking that the whole document is JSON (RFC 8259). A `version` key inside
/// a nested object (for example inside `config`) is not the scene version; when
/// the top-level object has no `version` member the version is 1.
///
/// It travels inside a [`std::io::Error`] of kind
/// [`std::io::ErrorKind::InvalidData`]; recover it with
/// `err.get_ref().and_then(|e| e.downcast_ref::<InvalidSceneJsonVersion>())`.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum InvalidSceneJsonVersion {
    /// The document is not JSON, so its top-level `version` member cannot be
    /// located. `offset` is the byte offset of the first character that does
    /// not fit the JSON grammar (the document length when it ends too early).
    /// Non-JSON numbers such as `+1` or `01` land here.
    MalformedJson {
        /// Byte offset of the offending character.
        offset: usize,
    },
    /// The document nests arrays / objects deeper than the loader accepts
    /// ([`MAX_SCENE_JSON_DEPTH`]).
    TooDeep {
        /// Byte offset of the bracket that exceeded the limit.
        offset: usize,
    },
    /// The top-level object has more than one `version` member (JSON leaves
    /// the meaning of duplicate keys open, so neither one is chosen).
    DuplicateVersion,
    /// The top-level `version` member is JSON but not a non-negative integer
    /// written without fraction or exponent (`1.0`, `1e0`, `-1`, `"1"`, `true`,
    /// `null`, an array or an object).
    NotAnUnsignedInteger {
        /// The member's value as written in the document.
        value: String,
    },
    /// The top-level `version` member is a non-negative integer above
    /// `u32::MAX`.
    OutOfRange {
        /// The member's value as written in the document.
        value: String,
    },
}

impl core::fmt::Display for InvalidSceneJsonVersion {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::MalformedJson { offset } => {
                write!(f, "scene JSON is malformed at byte {offset}")
            }
            Self::TooDeep { offset } => write!(
                f,
                "scene JSON nests deeper than {MAX_SCENE_JSON_DEPTH} levels at byte {offset}"
            ),
            Self::DuplicateVersion => {
                write!(
                    f,
                    "scene JSON has more than one top-level \"version\" member"
                )
            }
            Self::NotAnUnsignedInteger { value } => write!(
                f,
                "scene JSON top-level \"version\" is not a non-negative integer: {value}"
            ),
            Self::OutOfRange { value } => write!(
                f,
                "scene JSON top-level \"version\" does not fit in u32: {value}"
            ),
        }
    }
}

impl std::error::Error for InvalidSceneJsonVersion {}

/// Deepest array / object nesting [`load_scene_json`] accepts. The writer
/// nests at most 3 levels (`bodies` → body → field array).
pub const MAX_SCENE_JSON_DEPTH: usize = 64;

/// Strict RFC 8259 scanner over a whole document. It checks the grammar and,
/// for the top-level object only, records each member's decoded key and the
/// byte range of its value.
struct JsonScanner<'a> {
    text: &'a str,
    pos: usize,
}

impl<'a> JsonScanner<'a> {
    fn malformed(&self) -> InvalidSceneJsonVersion {
        InvalidSceneJsonVersion::MalformedJson { offset: self.pos }
    }

    fn peek(&self) -> Option<u8> {
        self.text.as_bytes().get(self.pos).copied()
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    fn expect(&mut self, b: u8) -> Result<(), InvalidSceneJsonVersion> {
        if self.peek() == Some(b) {
            self.pos += 1;
            Ok(())
        } else {
            Err(self.malformed())
        }
    }

    /// `members`: `Some` only for the top-level object, collecting
    /// `(decoded key, value text)`.
    fn object(
        &mut self,
        depth: usize,
        mut members: Option<&mut Vec<(String, &'a str)>>,
    ) -> Result<(), InvalidSceneJsonVersion> {
        self.expect(b'{')?;
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(());
        }
        loop {
            self.skip_ws();
            let key = self.string()?;
            self.skip_ws();
            self.expect(b':')?;
            self.skip_ws();
            let start = self.pos;
            self.value(depth)?;
            if let Some(m) = members.as_deref_mut() {
                let text: &'a str = self.text;
                m.push((key, &text[start..self.pos]));
            }
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(());
                }
                _ => return Err(self.malformed()),
            }
        }
    }

    fn array(&mut self, depth: usize) -> Result<(), InvalidSceneJsonVersion> {
        self.expect(b'[')?;
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(());
        }
        loop {
            self.skip_ws();
            self.value(depth)?;
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Ok(());
                }
                _ => return Err(self.malformed()),
            }
        }
    }

    /// A value at nesting `depth` (the enclosing container's depth).
    fn value(&mut self, depth: usize) -> Result<(), InvalidSceneJsonVersion> {
        match self.peek() {
            Some(b'{' | b'[') if depth >= MAX_SCENE_JSON_DEPTH => {
                Err(InvalidSceneJsonVersion::TooDeep { offset: self.pos })
            }
            Some(b'{') => self.object(depth + 1, None),
            Some(b'[') => self.array(depth + 1),
            Some(b'"') => self.string().map(drop),
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(b't') => self.literal("true"),
            Some(b'f') => self.literal("false"),
            Some(b'n') => self.literal("null"),
            _ => Err(self.malformed()),
        }
    }

    fn literal(&mut self, word: &str) -> Result<(), InvalidSceneJsonVersion> {
        if self.text[self.pos..].starts_with(word) {
            self.pos += word.len();
            Ok(())
        } else {
            Err(self.malformed())
        }
    }

    fn digits(&mut self) -> Result<(), InvalidSceneJsonVersion> {
        if !matches!(self.peek(), Some(b'0'..=b'9')) {
            return Err(self.malformed());
        }
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.pos += 1;
        }
        Ok(())
    }

    /// `-? (0 | [1-9][0-9]*) (. [0-9]+)? ([eE] [+-]? [0-9]+)?`
    fn number(&mut self) -> Result<(), InvalidSceneJsonVersion> {
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        match self.peek() {
            Some(b'0') => {
                self.pos += 1;
                if matches!(self.peek(), Some(b'0'..=b'9')) {
                    return Err(self.malformed());
                }
            }
            Some(b'1'..=b'9') => self.digits()?,
            _ => return Err(self.malformed()),
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            self.digits()?;
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            self.digits()?;
        }
        Ok(())
    }

    fn hex4(&mut self) -> Result<u32, InvalidSceneJsonVersion> {
        let Some(digits) = self.text.get(self.pos..self.pos + 4) else {
            return Err(self.malformed());
        };
        if !digits.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(self.malformed());
        }
        let v = u32::from_str_radix(digits, 16).map_err(|_| self.malformed())?;
        self.pos += 4;
        Ok(v)
    }

    /// A string, decoded. A lone surrogate escape decodes to U+FFFD (it is
    /// grammatical JSON but names no character).
    fn string(&mut self) -> Result<String, InvalidSceneJsonVersion> {
        self.expect(b'"')?;
        let mut out = String::new();
        loop {
            let run_start = self.pos;
            while matches!(self.peek(), Some(b) if b != b'"' && b != b'\\' && b >= 0x20) {
                self.pos += 1;
            }
            // the run stops at an ASCII byte, so both ends are char boundaries
            out.push_str(&self.text[run_start..self.pos]);
            match self.peek() {
                Some(b'"') => {
                    self.pos += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.pos += 1;
                    let c = match self.peek() {
                        Some(b'"') => '"',
                        Some(b'\\') => '\\',
                        Some(b'/') => '/',
                        Some(b'b') => '\u{8}',
                        Some(b'f') => '\u{c}',
                        Some(b'n') => '\n',
                        Some(b'r') => '\r',
                        Some(b't') => '\t',
                        Some(b'u') => {
                            self.pos += 1;
                            let hi = self.hex4()?;
                            let code = if (0xD800..0xDC00).contains(&hi)
                                && self.text[self.pos..].starts_with("\\u")
                            {
                                let save = self.pos;
                                self.pos += 2;
                                let lo = self.hex4()?;
                                if (0xDC00..0xE000).contains(&lo) {
                                    0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                                } else {
                                    // not a pair: decode the next escape on its own
                                    self.pos = save;
                                    hi
                                }
                            } else {
                                hi
                            };
                            out.push(char::from_u32(code).unwrap_or('\u{FFFD}'));
                            continue;
                        }
                        _ => return Err(self.malformed()),
                    };
                    self.pos += 1;
                    out.push(c);
                }
                // end of input or an unescaped control character
                _ => return Err(self.malformed()),
            }
        }
    }
}

/// The scene version recorded in a JSON document: the value of the top-level
/// object's `version` member, 1 when the top-level object has none.
///
/// The whole document must be JSON whose top level is an object; anything
/// else is an [`InvalidSceneJsonVersion`] (see its variants).
fn scene_json_version(json: &str) -> Result<u32, InvalidSceneJsonVersion> {
    let mut scan = JsonScanner { text: json, pos: 0 };
    let mut members = Vec::new();
    scan.skip_ws();
    scan.object(1, Some(&mut members))?;
    scan.skip_ws();
    if scan.pos != json.len() {
        return Err(scan.malformed());
    }
    let mut versions = members.iter().filter(|(k, _)| k == "version");
    let Some(&(_, value)) = versions.next() else {
        return Ok(CURRENT_VERSION);
    };
    if versions.next().is_some() {
        return Err(InvalidSceneJsonVersion::DuplicateVersion);
    }
    // the scanner accepted `value` as one JSON value; a non-negative integer
    // without fraction or exponent is exactly a run of ASCII digits (the
    // grammar already ruled out a leading `+` and leading zeros)
    if value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) {
        return Err(InvalidSceneJsonVersion::NotAnUnsignedInteger {
            value: value.to_string(),
        });
    }
    value
        .parse::<u32>()
        .map_err(|_| InvalidSceneJsonVersion::OutOfRange {
            value: value.to_string(),
        })
}

// ============================================================================
// Conversion Helpers
// ============================================================================

const fn fix128_to_raw(v: Fix128) -> [i64; 2] {
    [v.hi, v.lo as i64]
}

#[cfg(test)]
const fn raw_to_fix128(r: &[i64; 2]) -> Fix128 {
    Fix128 {
        hi: r[0],
        lo: r[1] as u64,
    }
}

const fn vec3fix_to_raw(v: Vec3Fix) -> [i64; 6] {
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
const fn raw_to_vec3fix(r: &[i64; 6]) -> Vec3Fix {
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
///
/// # Errors
///
/// Returns an error if the file cannot be created or written.
pub fn save_scene(scene: &PhysicsScene, path: &std::path::Path) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    write_scene_binary(&mut file, scene)
}

/// Load a physics scene from a binary file.
///
/// A header `version` outside [`SUPPORTED_SCENE_VERSIONS`] is rejected right
/// after the header is read (nothing after it is parsed): every other value
/// names a layout this reader does not know, and reading it with the
/// version 1 layout would misread it silently.
///
/// # Errors
///
/// Returns an error if the file cannot be opened, read, or contains invalid
/// data, including an [`std::io::ErrorKind::InvalidData`] error carrying
/// [`UnsupportedSceneVersion`] for an unknown version.
pub fn load_scene(path: &std::path::Path) -> std::io::Result<PhysicsScene> {
    let mut file = std::fs::File::open(path)?;
    read_scene_binary(&mut file)
}

/// Same as [`load_scene`], which now rejects unknown format versions itself.
///
/// # Errors
///
/// Exactly those of [`load_scene`].
#[deprecated(note = "load_scene rejects unsupported scene versions itself; call load_scene")]
pub fn load_scene_checked(path: &std::path::Path) -> std::io::Result<PhysicsScene> {
    load_scene(path)
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
    check_scene_version(version)?;

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
    // The counts come from the file: do not trust them for a reservation (a corrupt
    // header must end in `UnexpectedEof`, not in a multi-gigabyte allocation).
    let mut bodies = Vec::with_capacity(body_count.min(MAX_PREALLOC));
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
    let mut joints = Vec::with_capacity(joint_count.min(MAX_PREALLOC));
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
///
/// # Errors
///
/// Returns an error if the file cannot be written.
pub fn save_scene_json(scene: &PhysicsScene, path: &std::path::Path) -> std::io::Result<()> {
    let json = scene_to_json(scene);
    std::fs::write(path, json)
}

/// Load a physics scene from a JSON file.
///
/// The version is the top-level object's `version` member (an absent member
/// means 1; a `version` key inside a nested object is not the scene version).
/// A version outside [`SUPPORTED_SCENE_VERSIONS`] is rejected, as for
/// [`load_scene`], before the rest of the scene is read. The document must be
/// JSON (RFC 8259) with an object at the top level.
///
/// # Errors
///
/// Returns an error if the file cannot be read or contains invalid JSON
/// data, including an [`std::io::ErrorKind::InvalidData`] error carrying
/// [`UnsupportedSceneVersion`] for an unknown version, or
/// [`InvalidSceneJsonVersion`] when the document is not JSON or its top-level
/// `version` member is duplicated or not a `u32`.
pub fn load_scene_json(path: &std::path::Path) -> std::io::Result<PhysicsScene> {
    let json = std::fs::read_to_string(path)?;
    let version = scene_json_version(&json)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    check_scene_version(version)?;
    parse_scene_json(&json, version)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

/// Same as [`load_scene_json`], which now rejects unknown format versions
/// itself.
///
/// # Errors
///
/// Exactly those of [`load_scene_json`].
#[deprecated(
    note = "load_scene_json rejects unsupported scene versions itself; call load_scene_json"
)]
pub fn load_scene_json_checked(path: &std::path::Path) -> std::io::Result<PhysicsScene> {
    load_scene_json(path)
}

fn i64_array_to_json(arr: &[i64]) -> String {
    let items: Vec<String> = arr.iter().map(|v| format!("{v}")).collect();
    format!("[{}]", items.join(", "))
}

fn scene_to_json(scene: &PhysicsScene) -> String {
    fn write_json(s: &mut String, scene: &PhysicsScene) -> core::fmt::Result {
        s.push_str("{\n");
        writeln!(s, "  \"version\": {},", scene.version)?;

        // Config
        s.push_str("  \"config\": {\n");
        writeln!(s, "    \"substeps\": {},", scene.config.substeps)?;
        writeln!(s, "    \"iterations\": {},", scene.config.iterations)?;
        writeln!(
            s,
            "    \"gravity\": {},",
            i64_array_to_json(&scene.config.gravity)
        )?;
        writeln!(
            s,
            "    \"damping\": {}",
            i64_array_to_json(&scene.config.damping)
        )?;
        s.push_str("  },\n");

        // Bodies
        s.push_str("  \"bodies\": [\n");
        for (i, body) in scene.bodies.iter().enumerate() {
            s.push_str("    {\n");
            writeln!(
                s,
                "      \"position\": {},",
                i64_array_to_json(&body.position)
            )?;
            writeln!(
                s,
                "      \"velocity\": {},",
                i64_array_to_json(&body.velocity)
            )?;
            writeln!(
                s,
                "      \"rotation\": {},",
                i64_array_to_json(&body.rotation)
            )?;
            writeln!(s, "      \"mass\": {},", i64_array_to_json(&body.mass))?;
            writeln!(s, "      \"body_type\": {}", body.body_type)?;
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
            writeln!(s, "      \"body_a\": {},", joint.body_a)?;
            writeln!(s, "      \"body_b\": {},", joint.body_b)?;
            writeln!(s, "      \"joint_type\": {},", joint.joint_type)?;
            writeln!(
                s,
                "      \"anchor_a\": {},",
                i64_array_to_json(&joint.anchor_a)
            )?;
            writeln!(
                s,
                "      \"anchor_b\": {}",
                i64_array_to_json(&joint.anchor_b)
            )?;
            if i < scene.joints.len() - 1 {
                s.push_str("    },\n");
            } else {
                s.push_str("    }\n");
            }
        }
        s.push_str("  ]\n");

        s.push_str("}\n");
        Ok(())
    }

    let mut s = String::new();
    // fmt::Write for String is infallible, but we use ? for clean control flow
    let _ = write_json(&mut s, scene);
    s
}

// ============================================================================
// JSON Parser (minimal, no external dependencies)
// ============================================================================

/// The scene in `json` with the given `version` (read beforehand by
/// [`scene_json_version`], which also checked that `json` is JSON).
fn parse_scene_json(json: &str, version: u32) -> Result<PhysicsScene, String> {
    let json = json.trim();
    if !json.starts_with('{') || !json.ends_with('}') {
        return Err("Expected JSON object".into());
    }

    // Config
    let config_str = extract_object(json, "config").unwrap_or_default();
    let substeps = extract_u32(&config_str, "substeps")?.unwrap_or(8);
    let iterations = extract_u32(&config_str, "iterations")?.unwrap_or(4);
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
        let body_type = extract_u8(obj, "body_type")?.unwrap_or(0);

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
        let body_a = extract_u32(obj, "body_a")?.unwrap_or(0);
        let body_b = extract_u32(obj, "body_b")?.unwrap_or(0);
        let joint_type = extract_u8(obj, "joint_type")?.unwrap_or(0);
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

/// Read an unsigned integer value for `key`.
///
/// `Ok(None)` when the key (or its colon) is absent, so callers can apply their documented
/// default. A value that is present but is not a `u32` (negative, too large, not a number)
/// is an error: silently replacing it with the default would load a different scene than
/// the file describes.
fn extract_u32(json: &str, key: &str) -> Result<Option<u32>, String> {
    let pattern = format!("\"{key}\"");
    let Some(idx) = json.find(&pattern) else {
        return Ok(None);
    };
    let rest = &json[idx + pattern.len()..];
    let Some(colon) = rest.find(':') else {
        return Ok(None);
    };
    let after_colon = rest[colon + 1..].trim_start();
    // Token: everything up to the next JSON delimiter
    let end = after_colon
        .find([',', '}', ']', '\n', '\r'])
        .unwrap_or(after_colon.len());
    let token = after_colon[..end].trim();
    token
        .parse::<u32>()
        .map(Some)
        .map_err(|e| format!("Invalid value for {key}: {token:?} ({e})"))
}

/// Like [`extract_u32`] for fields stored as `u8`; a value above 255 is an error, not a truncation.
fn extract_u8(json: &str, key: &str) -> Result<Option<u8>, String> {
    match extract_u32(json, key)? {
        None => Ok(None),
        Some(v) => u8::try_from(v)
            .map(Some)
            .map_err(|_| format!("Value for {key} does not fit in u8: {v}")),
    }
}

fn extract_object(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{key}\"");
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
            return Some(rest[start..=(start + i)].to_string());
        }
    }
    None
}

fn extract_array(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{key}\"");
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
            return Some(after_colon[start..=(start + i)].to_string());
        }
    }
    None
}

fn extract_i64_array(json: &str, key: &str, expected_len: usize) -> Result<Vec<i64>, String> {
    let arr_str = extract_array(json, key).ok_or_else(|| format!("Missing key: {key}"))?;
    // Parse [n1, n2, ...]
    let inner = arr_str.trim_start_matches('[').trim_end_matches(']');
    let values: Result<Vec<i64>, _> = inner.split(',').map(|s| s.trim().parse::<i64>()).collect();
    let values = values.map_err(|e| format!("Parse error for {key}: {e}"))?;
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

    fn unsupported(e: &std::io::Error) -> Option<UnsupportedSceneVersion> {
        e.get_ref()
            .and_then(|i| i.downcast_ref::<UnsupportedSceneVersion>())
            .copied()
    }

    fn with_version(v: u32) -> PhysicsScene {
        let mut s = make_test_scene();
        s.version = v;
        s
    }

    /// 版 1 だけを読める版として受理し、0 / 2 / 0xDEAD_BEEF / u32::MAX は
    /// header を読んだ直後に `UnsupportedSceneVersion` で拒否する
    #[test]
    fn binary_loader_accepts_only_supported_versions() {
        assert_eq!(SUPPORTED_SCENE_VERSIONS, &[1]);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.aphys");
        save_scene(&with_version(1), &path).unwrap();
        assert_eq!(load_scene(&path).unwrap(), with_version(1));
        for v in [0u32, 2, 0xDEAD_BEEF, u32::MAX] {
            save_scene(&with_version(v), &path).unwrap();
            let e = load_scene(&path).unwrap_err();
            assert_eq!(e.kind(), std::io::ErrorKind::InvalidData, "{v}");
            assert_eq!(unsupported(&e), Some(UnsupportedSceneVersion { found: v }));
            assert!(e.to_string().contains(&format!("version {v}")), "{e}");
        }
    }

    /// 版の検査は header 直後に行う — 版 2 の header だけで本体の無い入力は
    /// `UnexpectedEof` でなく版の拒否になり、版 1 の同じ入力は本体を読みに行って
    /// `UnexpectedEof` になる
    #[test]
    fn binary_loader_rejects_version_before_reading_the_body() {
        let header = |v: u32| {
            let mut bytes = MAGIC.to_vec();
            bytes.extend_from_slice(&v.to_le_bytes());
            bytes
        };
        let e = read_scene_binary(&mut std::io::Cursor::new(&header(2))).unwrap_err();
        assert_eq!(e.kind(), std::io::ErrorKind::InvalidData);
        assert_eq!(unsupported(&e), Some(UnsupportedSceneVersion { found: 2 }));
        let e = read_scene_binary(&mut std::io::Cursor::new(&header(1))).unwrap_err();
        assert_eq!(e.kind(), std::io::ErrorKind::UnexpectedEof);
        assert_eq!(unsupported(&e), None);
    }

    /// JSON の loader も同じ集合で判定する (key が無ければ 1)
    #[test]
    fn json_loader_accepts_only_supported_versions() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.json");
        save_scene_json(&with_version(1), &path).unwrap();
        assert_eq!(load_scene_json(&path).unwrap(), with_version(1));
        for v in [0u32, 2, 0xDEAD_BEEF] {
            save_scene_json(&with_version(v), &path).unwrap();
            let e = load_scene_json(&path).unwrap_err();
            assert_eq!(e.kind(), std::io::ErrorKind::InvalidData, "{v}");
            assert_eq!(unsupported(&e), Some(UnsupportedSceneVersion { found: v }));
        }
        std::fs::write(
            &path,
            "{\"config\":{\"gravity\":[0,0,0,0,0,0],\"damping\":[1,0]}}",
        )
        .unwrap();
        assert_eq!(load_scene_json(&path).unwrap().version, 1);
        // 他の不正 (壊れた JSON) は版の error にならない
        std::fs::write(&path, "{\"version\": 1, \"bodies\": [").unwrap();
        let e = load_scene_json(&path).unwrap_err();
        assert_eq!(e.kind(), std::io::ErrorKind::InvalidData);
        assert_eq!(unsupported(&e), None);
    }

    /// 版は top-level の object の `version` member だけから読む 入れ子の
    /// `version` は版でなく、top-level に無ければ 1
    #[test]
    fn json_version_is_the_top_level_member_only() {
        use InvalidSceneJsonVersion as E;
        let v = scene_json_version;
        assert_eq!(v("{}"), Ok(1));
        assert_eq!(v("{\"config\":{\"version\":2}}"), Ok(1));
        assert_eq!(v("{\"bodies\":[{\"version\":2}],\"version\":1}"), Ok(1));
        assert_eq!(v("{\"note\":\"\\\"version\\\": 2\"}"), Ok(1));
        assert_eq!(v(" \t\r\n{ \"version\" \n:\t 1 \r}\n "), Ok(1));
        assert_eq!(v("{\"version\":2}"), Ok(2));
        assert_eq!(v("{\"version\":0}"), Ok(0));
        assert_eq!(v("{\"version\":4294967295}"), Ok(u32::MAX));
        // escape で綴った key も同じ key
        assert_eq!(v("{\"vers\\u0069on\":7}"), Ok(7));
        // 重複は順序に依らず拒否
        for t in [
            "{\"version\":1,\"version\":2}",
            "{\"version\":2,\"version\":1}",
            "{\"version\":1,\"version\":1}",
            "{\"version\":1,\"x\":0,\"vers\\u0069on\":1}",
        ] {
            assert_eq!(v(t), Err(E::DuplicateVersion), "{t}");
        }
        // JSON でない数 (先頭の `+`、先頭の 0) は文書ごと拒否
        for (t, at) in [
            ("{\"version\":+1}", 11),
            ("{\"version\":01}", 12),
            ("{\"version\":00}", 12),
            ("{\"version\":-01}", 13),
            ("{\"version\":1.}", 13),
            ("{\"version\":1e}", 13),
            ("{\"version\":.5}", 11),
            ("{\"version\":0x10}", 12),
            ("{\"version\":1 2}", 13),
            ("{\"version\":1,}", 13),
            ("{\"version\":1}x", 13),
            ("{\"version\":1", 12),
            ("[1]", 0),
            ("", 0),
            ("{\"a\":\"\u{1}\"}", 6),
            ("{\"a\":\"\\x\"}", 7),
            ("{\"a\":tru}", 5),
        ] {
            assert_eq!(v(t), Err(E::MalformedJson { offset: at }), "{t}");
        }
        // JSON だが非負の整数でない値
        for t in [
            "1.0", "1e0", "1E+0", "-1", "-0", "\"1\"", "true", "null", "[1]", "{}",
        ] {
            assert_eq!(
                v(&format!("{{\"version\":{t}}}")),
                Err(E::NotAnUnsignedInteger {
                    value: t.to_string()
                }),
                "{t}"
            );
        }
        for t in ["4294967296", "99999999999999999999"] {
            assert_eq!(
                v(&format!("{{\"version\":{t}}}")),
                Err(E::OutOfRange {
                    value: t.to_string()
                }),
                "{t}"
            );
        }
        // 入れ子の深さの上限
        let deep = |n: usize| format!("{{\"a\":{}{}}}", "[".repeat(n), "]".repeat(n));
        assert_eq!(v(&deep(MAX_SCENE_JSON_DEPTH - 1)), Ok(1));
        assert_eq!(
            v(&deep(MAX_SCENE_JSON_DEPTH)),
            Err(E::TooDeep {
                offset: 5 + MAX_SCENE_JSON_DEPTH - 1
            })
        );
    }

    /// 文字列の escape の復号 (surrogate pair、対でない surrogate は U+FFFD)
    #[test]
    fn json_scanner_decodes_string_escapes() {
        let dec = |t: &str| JsonScanner { text: t, pos: 0 }.string();
        assert_eq!(
            dec("\"a\\\"\\\\\\/\\b\\f\\n\\r\\t\""),
            Ok("a\"\\/\u{8}\u{c}\n\r\t".into())
        );
        assert_eq!(dec("\"\\ud83d\\ude00\""), Ok("\u{1F600}".into()));
        assert_eq!(dec("\"\\ud83d\\u0041\""), Ok("\u{FFFD}A".into()));
        assert_eq!(dec("\"\\ude00\""), Ok("\u{FFFD}".into()));
        assert_eq!(dec("\"é😀\""), Ok("é😀".into()));
        assert!(dec("\"\\u12\"").is_err());
        assert!(dec("\"\\u12G4\"").is_err());
        assert!(dec("\"abc").is_err());
    }

    /// 版の error は `InvalidData` の中身として取り出せ、表示に値を含む
    #[test]
    fn json_loader_reports_named_version_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.json");
        let named = |e: &std::io::Error| {
            e.get_ref()
                .and_then(|i| i.downcast_ref::<InvalidSceneJsonVersion>())
                .cloned()
        };
        for (text, want) in [
            (
                "{\"version\":1,\"version\":1}",
                InvalidSceneJsonVersion::DuplicateVersion,
            ),
            (
                "{\"version\":01}",
                InvalidSceneJsonVersion::MalformedJson { offset: 12 },
            ),
            (
                "{\"version\":1e0}",
                InvalidSceneJsonVersion::NotAnUnsignedInteger {
                    value: "1e0".into(),
                },
            ),
        ] {
            std::fs::write(&path, text).unwrap();
            let e = load_scene_json(&path).unwrap_err();
            assert_eq!(e.kind(), std::io::ErrorKind::InvalidData, "{text}");
            assert_eq!(named(&e), Some(want), "{text}");
            assert_eq!(unsupported(&e), None, "{text}");
        }
        let shown = [
            InvalidSceneJsonVersion::MalformedJson { offset: 3 }.to_string(),
            InvalidSceneJsonVersion::TooDeep { offset: 4 }.to_string(),
            InvalidSceneJsonVersion::DuplicateVersion.to_string(),
            InvalidSceneJsonVersion::NotAnUnsignedInteger {
                value: "1.0".into(),
            }
            .to_string(),
            InvalidSceneJsonVersion::OutOfRange {
                value: "4294967296".into(),
            }
            .to_string(),
        ];
        assert!(shown[0].contains("byte 3"));
        assert!(shown[1].contains("64") && shown[1].contains("byte 4"));
        assert!(shown[2].contains("more than one"));
        assert!(shown[3].contains("1.0"));
        assert!(shown[4].contains("4294967296"));
        // 入れ子だけに版 2 を持つ scene は版 1 として読む (入れ子の key は版でない)
        std::fs::write(
            &path,
            "{\"config\":{\"version\":2,\"gravity\":[0,0,0,0,0,0],\"damping\":[1,0]}}",
        )
        .unwrap();
        assert_eq!(load_scene_json(&path).unwrap().version, 1);
    }

    /// 非推奨の checked 版は既定の loader と同じ結果を返す (受理も拒否も)
    #[test]
    #[allow(deprecated)]
    fn deprecated_checked_loaders_are_the_default_loaders() {
        let dir = tempfile::tempdir().unwrap();
        let bin = dir.path().join("v.aphys");
        let json = dir.path().join("v.json");
        save_scene(&with_version(1), &bin).unwrap();
        save_scene_json(&with_version(1), &json).unwrap();
        assert_eq!(load_scene_checked(&bin).unwrap(), with_version(1));
        assert_eq!(load_scene_json_checked(&json).unwrap(), with_version(1));
        save_scene(&with_version(2), &bin).unwrap();
        save_scene_json(&with_version(2), &json).unwrap();
        for e in [
            load_scene_checked(&bin).unwrap_err(),
            load_scene_json_checked(&json).unwrap_err(),
        ] {
            assert_eq!(unsupported(&e), Some(UnsupportedSceneVersion { found: 2 }));
        }
    }
}
