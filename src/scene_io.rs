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
                "scene JSON top-level \"version\" is not a non-negative integer: {}",
                shown_value(value)
            ),
            Self::OutOfRange { value } => write!(
                f,
                "scene JSON top-level \"version\" does not fit in u32: {}",
                shown_value(value)
            ),
        }
    }
}

impl std::error::Error for InvalidSceneJsonVersion {}

/// Deepest array / object nesting [`load_scene_json`] accepts. The writer
/// nests at most 3 levels (`bodies` → body → field array).
pub const MAX_SCENE_JSON_DEPTH: usize = 64;

/// A JSON value parsed by [`JsonScanner`].
///
/// A number keeps its text (already checked against the RFC 8259 grammar) and
/// is evaluated only when a field reads it, so a token of any length is held
/// without overflow. Object members keep their document order, duplicates
/// included, so the reader can refuse them by name.
#[derive(Debug, PartialEq, Eq)]
enum JsonValue<'a> {
    Null,
    Bool(bool),
    Number(&'a str),
    String(String),
    Array(Vec<JsonValue<'a>>),
    Object(Vec<JsonMember<'a>>),
}

/// One member of a JSON object: the decoded key, the parsed value and the
/// value as written in the document.
#[derive(Debug, PartialEq, Eq)]
struct JsonMember<'a> {
    key: String,
    value: JsonValue<'a>,
    text: &'a str,
}

/// Strict RFC 8259 parser over a whole document, building a [`JsonValue`]
/// tree. Recursion goes one call deeper per nested array / object and stops
/// with [`InvalidSceneJsonVersion::TooDeep`] before it passes
/// [`MAX_SCENE_JSON_DEPTH`], so the stack use is bounded by that limit.
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

    /// An object whose members sit at nesting `depth`.
    fn object(&mut self, depth: usize) -> Result<Vec<JsonMember<'a>>, InvalidSceneJsonVersion> {
        self.expect(b'{')?;
        let mut members = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(members);
        }
        loop {
            self.skip_ws();
            let key = self.string()?;
            self.skip_ws();
            self.expect(b':')?;
            self.skip_ws();
            let start = self.pos;
            let value = self.value(depth)?;
            let text: &'a str = self.text;
            members.push(JsonMember {
                key,
                value,
                text: &text[start..self.pos],
            });
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(members);
                }
                _ => return Err(self.malformed()),
            }
        }
    }

    /// An array whose items sit at nesting `depth`.
    fn array(&mut self, depth: usize) -> Result<Vec<JsonValue<'a>>, InvalidSceneJsonVersion> {
        self.expect(b'[')?;
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(items);
        }
        loop {
            self.skip_ws();
            items.push(self.value(depth)?);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Ok(items);
                }
                _ => return Err(self.malformed()),
            }
        }
    }

    /// A value at nesting `depth` (the enclosing container's depth).
    fn value(&mut self, depth: usize) -> Result<JsonValue<'a>, InvalidSceneJsonVersion> {
        match self.peek() {
            Some(b'{' | b'[') if depth >= MAX_SCENE_JSON_DEPTH => {
                Err(InvalidSceneJsonVersion::TooDeep { offset: self.pos })
            }
            Some(b'{') => self.object(depth + 1).map(JsonValue::Object),
            Some(b'[') => self.array(depth + 1).map(JsonValue::Array),
            Some(b'"') => self.string().map(JsonValue::String),
            Some(b'-' | b'0'..=b'9') => self.number().map(JsonValue::Number),
            Some(b't') => self.literal("true").map(|()| JsonValue::Bool(true)),
            Some(b'f') => self.literal("false").map(|()| JsonValue::Bool(false)),
            Some(b'n') => self.literal("null").map(|()| JsonValue::Null),
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

    /// `-? (0 | [1-9][0-9]*) (. [0-9]+)? ([eE] [+-]? [0-9]+)?`, returned as written.
    fn number(&mut self) -> Result<&'a str, InvalidSceneJsonVersion> {
        let start = self.pos;
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
        let text: &'a str = self.text;
        Ok(&text[start..self.pos])
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

/// The members of the top-level object of `json`, which must be one JSON
/// (RFC 8259) object with nothing but whitespace around it.
fn parse_json_document(json: &str) -> Result<Vec<JsonMember<'_>>, InvalidSceneJsonVersion> {
    let mut scan = JsonScanner { text: json, pos: 0 };
    scan.skip_ws();
    if scan.peek() != Some(b'{') {
        return Err(scan.malformed());
    }
    let members = scan.object(1)?;
    scan.skip_ws();
    if scan.pos != json.len() {
        return Err(scan.malformed());
    }
    Ok(members)
}

/// The scene version recorded in a parsed JSON document: the value of the
/// top-level object's `version` member, 1 when the top-level object has none.
fn scene_json_version_of(top: &[JsonMember<'_>]) -> Result<u32, InvalidSceneJsonVersion> {
    let mut versions = top.iter().filter(|m| m.key == "version");
    let Some(member) = versions.next() else {
        return Ok(CURRENT_VERSION);
    };
    if versions.next().is_some() {
        return Err(InvalidSceneJsonVersion::DuplicateVersion);
    }
    // a non-negative integer without fraction or exponent is exactly a run of
    // ASCII digits (the grammar already ruled out a leading `+` and leading zeros)
    let value = member.text;
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

/// The scene version recorded in a JSON document (see
/// [`scene_json_version_of`]); the whole document must be JSON whose top level
/// is an object.
#[cfg(test)]
fn scene_json_version(json: &str) -> Result<u32, InvalidSceneJsonVersion> {
    scene_json_version_of(&parse_json_document(json)?)
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
/// JSON (RFC 8259) with an object at the top level, and no object in it may
/// repeat a key. Every field is read from its own object's members (see
/// [`InvalidSceneJson`]).
///
/// # Errors
///
/// Returns an error if the file cannot be read or contains invalid JSON
/// data, including an [`std::io::ErrorKind::InvalidData`] error carrying
/// [`UnsupportedSceneVersion`] for an unknown version,
/// [`InvalidSceneJsonVersion`] when the document is not JSON or its top-level
/// `version` member is duplicated or not a `u32`, or [`InvalidSceneJson`] when
/// any object repeats a key or a scene member is missing, of the wrong type or
/// out of range.
pub fn load_scene_json(path: &std::path::Path) -> std::io::Result<PhysicsScene> {
    let json = std::fs::read_to_string(path)?;
    let top = parse_json_document(&json)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let version = scene_json_version_of(&top)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    check_scene_version(version)?;
    scene_from_json(&top, version)
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
// JSON scene reader (over the parsed tree)
// ============================================================================

/// Error payload for a JSON scene whose members (other than the top-level
/// `version`, see [`InvalidSceneJsonVersion`]) do not describe a scene,
/// returned by [`load_scene_json`].
///
/// Every field is read from the members of its own object (the top-level
/// object, `config`, one entry of `bodies` / `joints`); a key that appears only
/// inside a nested value never stands in for the member being read. `path`
/// names the member, for example `config.gravity` or `bodies[2].mass[1]`.
///
/// It travels inside a [`std::io::Error`] of kind
/// [`std::io::ErrorKind::InvalidData`]; recover it with
/// `err.get_ref().and_then(|e| e.downcast_ref::<InvalidSceneJson>())`.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum InvalidSceneJson {
    /// An object, at any nesting level, has the same key more than once (JSON
    /// leaves the meaning of duplicate keys open, so neither one is chosen).
    /// `path` names the repeated member.
    DuplicateKey {
        /// The repeated member.
        path: String,
    },
    /// A required member is absent from its object.
    MissingMember {
        /// The absent member.
        path: String,
    },
    /// A member or array item has the wrong JSON type, or is a number that is
    /// not an integer written without fraction or exponent (for an unsigned
    /// field: a non-negative one).
    WrongType {
        /// The offending member or item.
        path: String,
        /// What the scene format needs there.
        expected: &'static str,
        /// What the document holds there.
        found: &'static str,
    },
    /// A fixed-length array has another number of items.
    WrongLength {
        /// The array.
        path: String,
        /// The number of items the field holds.
        expected: usize,
        /// The number of items in the document.
        found: usize,
    },
    /// An integer that does not fit the field's integer type.
    OutOfRange {
        /// The offending member or item.
        path: String,
        /// The integer as written in the document.
        value: String,
    },
}

/// Longest prefix of a value written into an error message; longer values are
/// shown as that prefix plus their length.
const SHOWN_VALUE_BYTES: usize = 32;

/// `value` for an error message: itself when short, else a bounded prefix and
/// its length in bytes.
fn shown_value(value: &str) -> String {
    if value.len() <= SHOWN_VALUE_BYTES {
        return value.to_string();
    }
    let mut end = SHOWN_VALUE_BYTES;
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}... ({} bytes)", &value[..end], value.len())
}

impl core::fmt::Display for InvalidSceneJson {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::DuplicateKey { path } => {
                write!(f, "scene JSON has more than one \"{path}\" member")
            }
            Self::MissingMember { path } => {
                write!(f, "scene JSON has no \"{path}\" member")
            }
            Self::WrongType {
                path,
                expected,
                found,
            } => write!(f, "scene JSON \"{path}\" is {found}, expected {expected}"),
            Self::WrongLength {
                path,
                expected,
                found,
            } => write!(
                f,
                "scene JSON \"{path}\" has {found} items, expected {expected}"
            ),
            Self::OutOfRange { path, value } => write!(
                f,
                "scene JSON \"{path}\" is out of range: {}",
                shown_value(value)
            ),
        }
    }
}

impl std::error::Error for InvalidSceneJson {}

/// `parent.key`, or `key` at the top level.
fn member_path(parent: &str, key: &str) -> String {
    if parent.is_empty() {
        key.to_string()
    } else {
        format!("{parent}.{key}")
    }
}

/// The path of the first repeated key found in any object of the document
/// (each object is checked before its children; the walk keeps its own stack,
/// so it does not recurse).
fn duplicate_key_path(top: &[JsonMember<'_>]) -> Option<String> {
    enum Node<'t, 'a> {
        Members(&'t [JsonMember<'a>]),
        Items(&'t [JsonValue<'a>]),
    }
    fn container<'t, 'a>(v: &'t JsonValue<'a>) -> Option<Node<'t, 'a>> {
        match v {
            JsonValue::Object(m) => Some(Node::Members(m)),
            JsonValue::Array(i) => Some(Node::Items(i)),
            _ => None,
        }
    }
    let mut stack = vec![(Node::Members(top), String::new())];
    while let Some((node, path)) = stack.pop() {
        match node {
            Node::Members(members) => {
                let mut seen = std::collections::BTreeSet::new();
                for m in members {
                    if !seen.insert(m.key.as_str()) {
                        return Some(member_path(&path, &m.key));
                    }
                }
                for m in members.iter().rev() {
                    if let Some(child) = container(&m.value) {
                        stack.push((child, member_path(&path, &m.key)));
                    }
                }
            }
            Node::Items(items) => {
                for (i, v) in items.iter().enumerate().rev() {
                    if let Some(child) = container(v) {
                        stack.push((child, format!("{path}[{i}]")));
                    }
                }
            }
        }
    }
    None
}

/// The JSON type of `v`, for [`InvalidSceneJson::WrongType`].
const fn json_type(v: &JsonValue<'_>) -> &'static str {
    match v {
        JsonValue::Null => "null",
        JsonValue::Bool(_) => "a boolean",
        JsonValue::Number(_) => "a number",
        JsonValue::String(_) => "a string",
        JsonValue::Array(_) => "an array",
        JsonValue::Object(_) => "an object",
    }
}

/// The members of one JSON object of the scene, with its path for errors.
/// Keys are unique (checked by [`duplicate_key_path`] beforehand).
struct SceneObject<'t, 'a> {
    members: &'t [JsonMember<'a>],
    path: String,
}

impl<'t, 'a> SceneObject<'t, 'a> {
    /// `v` as an object at `path`.
    fn new(v: &'t JsonValue<'a>, path: String) -> Result<Self, InvalidSceneJson> {
        match v {
            JsonValue::Object(members) => Ok(Self { members, path }),
            other => Err(InvalidSceneJson::WrongType {
                path,
                expected: "an object",
                found: json_type(other),
            }),
        }
    }

    fn path(&self, key: &str) -> String {
        member_path(&self.path, key)
    }

    /// This object's own `key` member (never one inside a nested value).
    fn get(&self, key: &str) -> Option<&'t JsonValue<'a>> {
        self.members.iter().find(|m| m.key == key).map(|m| &m.value)
    }

    fn require(&self, key: &str) -> Result<&'t JsonValue<'a>, InvalidSceneJson> {
        self.get(key)
            .ok_or_else(|| InvalidSceneJson::MissingMember {
                path: self.path(key),
            })
    }

    /// An optional integer member, `default` when absent.
    fn int_or<T: TryFrom<i128>>(&self, key: &str, default: T) -> Result<T, InvalidSceneJson> {
        self.get(key)
            .map_or(Ok(default), |v| json_integer(v, &self.path(key)))
    }

    /// A required array member of exactly `N` integers.
    fn i64_array<const N: usize>(&self, key: &str) -> Result<[i64; N], InvalidSceneJson> {
        json_i64_array(self.require(key)?, &self.path(key))
    }

    /// An optional array-of-objects member, empty when absent.
    fn objects(&self, key: &str) -> Result<Vec<SceneObject<'t, 'a>>, InvalidSceneJson> {
        let path = self.path(key);
        match self.get(key) {
            None => Ok(Vec::new()),
            Some(JsonValue::Array(items)) => items
                .iter()
                .enumerate()
                .map(|(i, v)| SceneObject::new(v, format!("{path}[{i}]")))
                .collect(),
            Some(other) => Err(InvalidSceneJson::WrongType {
                path,
                expected: "an array",
                found: json_type(other),
            }),
        }
    }
}

/// `v` as an integer of type `T`. The token must be a JSON number without
/// fraction or exponent (and without a minus sign when `T` is unsigned); its
/// value must fit `T`. A token of any length is an error, never a wrap.
fn json_integer<T: TryFrom<i128>>(v: &JsonValue<'_>, path: &str) -> Result<T, InvalidSceneJson> {
    let unsigned = T::try_from(-1i128).is_err();
    let expected = if unsigned {
        "a non-negative integer"
    } else {
        "an integer"
    };
    let wrong = |found| InvalidSceneJson::WrongType {
        path: path.to_string(),
        expected,
        found,
    };
    let JsonValue::Number(text) = v else {
        return Err(wrong(json_type(v)));
    };
    if text.bytes().any(|b| matches!(b, b'.' | b'e' | b'E')) {
        return Err(wrong("a number with a fraction or exponent"));
    }
    if unsigned && text.starts_with('-') {
        return Err(wrong("a negative number"));
    }
    text.parse::<i128>()
        .ok()
        .and_then(|n| T::try_from(n).ok())
        .ok_or_else(|| InvalidSceneJson::OutOfRange {
            path: path.to_string(),
            value: (*text).to_string(),
        })
}

/// `v` as an array of exactly `N` `i64` values.
fn json_i64_array<const N: usize>(
    v: &JsonValue<'_>,
    path: &str,
) -> Result<[i64; N], InvalidSceneJson> {
    let JsonValue::Array(items) = v else {
        return Err(InvalidSceneJson::WrongType {
            path: path.to_string(),
            expected: "an array",
            found: json_type(v),
        });
    };
    if items.len() != N {
        return Err(InvalidSceneJson::WrongLength {
            path: path.to_string(),
            expected: N,
            found: items.len(),
        });
    }
    let mut out = [0i64; N];
    for (i, (slot, item)) in out.iter_mut().zip(items).enumerate() {
        *slot = json_integer(item, &format!("{path}[{i}]"))?;
    }
    Ok(out)
}

/// The scene held by the top-level members `top` (parsed by
/// [`parse_json_document`]) with the given `version` (read beforehand by
/// [`scene_json_version_of`]).
///
/// Absent members take the documented defaults: `substeps` 8, `iterations` 4,
/// `bodies` / `joints` empty, `body_type` / `body_a` / `body_b` / `joint_type`
/// 0. `config`, its `gravity` / `damping`, each body's `position` /
/// `velocity` / `rotation` / `mass` and each joint's `anchor_a` / `anchor_b`
/// are required. Unknown members are ignored.
fn scene_from_json(top: &[JsonMember<'_>], version: u32) -> Result<PhysicsScene, InvalidSceneJson> {
    if let Some(path) = duplicate_key_path(top) {
        return Err(InvalidSceneJson::DuplicateKey { path });
    }
    let root = SceneObject {
        members: top,
        path: String::new(),
    };

    let cfg = SceneObject::new(root.require("config")?, "config".into())?;
    let config = PhysicsConfig {
        substeps: cfg.int_or("substeps", 8)?,
        iterations: cfg.int_or("iterations", 4)?,
        gravity: cfg.i64_array("gravity")?,
        damping: cfg.i64_array("damping")?,
    };

    let bodies = root
        .objects("bodies")?
        .iter()
        .map(|b| {
            Ok(SerializedBody {
                position: b.i64_array("position")?,
                velocity: b.i64_array("velocity")?,
                rotation: b.i64_array("rotation")?,
                mass: b.i64_array("mass")?,
                body_type: b.int_or("body_type", 0)?,
            })
        })
        .collect::<Result<Vec<_>, InvalidSceneJson>>()?;

    let joints = root
        .objects("joints")?
        .iter()
        .map(|j| {
            Ok(SerializedJoint {
                body_a: j.int_or("body_a", 0)?,
                body_b: j.int_or("body_b", 0)?,
                joint_type: j.int_or("joint_type", 0)?,
                anchor_a: j.i64_array("anchor_a")?,
                anchor_b: j.i64_array("anchor_b")?,
            })
        })
        .collect::<Result<Vec<_>, InvalidSceneJson>>()?;

    Ok(PhysicsScene {
        bodies,
        joints,
        config,
        version,
    })
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

/// The reader that `load_scene_json` used before the parse tree (substring
/// extractors that resolve a key to its first occurrence anywhere in the
/// text), kept verbatim only as the reference of the differential test
/// `json_tree_tests::tree_reader_matches_the_old_reader_on_every_valid_scene`.
#[cfg(test)]
#[allow(clippy::all, clippy::pedantic, clippy::nursery)]
mod legacy_json {
    use super::*;

    /// The scene in `json` with the given `version` (read beforehand by
    /// [`scene_json_version`], which also checked that `json` is JSON).
    pub(super) fn parse_scene_json(json: &str, version: u32) -> Result<PhysicsScene, String> {
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
        let values: Result<Vec<i64>, _> =
            inner.split(',').map(|s| s.trim().parse::<i64>()).collect();
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
}

/// 解析木による読み取りの試験 (旧 reader との差分、重複 key、入れ子の同名 key、
/// 文字列中の括弧、深さの上限、巨大な数、型の誤り)
#[cfg(test)]
mod json_tree_tests {
    use super::*;

    fn named(json: &str) -> Result<PhysicsScene, InvalidSceneJson> {
        let top = parse_json_document(json).expect("probe is JSON");
        let version = scene_json_version_of(&top).expect("probe version");
        scene_from_json(&top, version)
    }

    fn sb(
        position: [i64; 6],
        velocity: [i64; 6],
        rotation: [i64; 8],
        mass: [i64; 2],
        body_type: u8,
    ) -> SerializedBody {
        SerializedBody {
            position,
            velocity,
            rotation,
            mass,
            body_type,
        }
    }

    fn sj(
        body_a: u32,
        body_b: u32,
        joint_type: u8,
        anchor_a: [i64; 6],
        anchor_b: [i64; 6],
    ) -> SerializedJoint {
        SerializedJoint {
            body_a,
            body_b,
            joint_type,
            anchor_a,
            anchor_b,
        }
    }

    fn scene(
        bodies: Vec<SerializedBody>,
        joints: Vec<SerializedJoint>,
        config: PhysicsConfig,
        version: u32,
    ) -> PhysicsScene {
        PhysicsScene {
            bodies,
            joints,
            config,
            version,
        }
    }

    /// examples/scene_snapshot_roundtrip.rs が書く scene (同じ手順で world を進める)
    fn example_scene() -> PhysicsScene {
        use crate::math::QuatFix;
        use crate::solver::{PhysicsConfig as WorldConfig, PhysicsWorld, RigidBody};
        let limbs3 = |v: Vec3Fix| {
            [
                v.x.hi,
                v.x.lo as i64,
                v.y.hi,
                v.y.lo as i64,
                v.z.hi,
                v.z.lo as i64,
            ]
        };
        let limb = |v: Fix128| [v.hi, v.lo as i64];
        let mut world = PhysicsWorld::new(WorldConfig::default());
        world.add_body(RigidBody::new_static(Vec3Fix::from_int(0, -1, 0)));
        world.add_body(RigidBody::new_dynamic(
            Vec3Fix::new(
                Fix128::from_ratio(1, 3),
                Fix128::from_int(5),
                Fix128::from_ratio(-2, 7),
            ),
            Fix128::from_ratio(5, 2),
        ));
        for _ in 0..30 {
            world.step(Fix128::from_ratio(1, 60));
        }
        let q = QuatFix::IDENTITY;
        let bodies = world
            .bodies
            .iter()
            .map(|b| {
                sb(
                    limbs3(b.position),
                    limbs3(b.velocity),
                    [
                        q.x.hi,
                        q.x.lo as i64,
                        q.y.hi,
                        q.y.lo as i64,
                        q.z.hi,
                        q.z.lo as i64,
                        q.w.hi,
                        q.w.lo as i64,
                    ],
                    if b.inv_mass.is_zero() {
                        [0, 0]
                    } else {
                        limb(Fix128::ONE / b.inv_mass)
                    },
                    u8::from(b.inv_mass.is_zero()),
                )
            })
            .collect();
        let gravity = Vec3Fix::new(Fix128::ZERO, Fix128::from_ratio(-981, 100), Fix128::ZERO);
        let config = PhysicsConfig::new(
            world.config.substeps as u32,
            world.config.iterations as u32 + 1,
            limbs3(gravity),
            limb(Fix128::from_ratio(95, 100)),
        );
        scene(bodies, Vec::new(), config, CURRENT_VERSION)
    }

    /// 既存の試験と例が writer に書かせる scene をすべて列挙する
    /// (src/scene_io.rs の tests、tests/analytic_scene_io_wiring.rs、examples/scene_snapshot_roundtrip.rs)
    fn writer_scenes() -> Vec<(String, PhysicsScene)> {
        let mut out: Vec<(String, PhysicsScene)> = Vec::new();
        // src/scene_io.rs::tests
        let unit = tests_scene();
        for v in [1u32, 0, 2, 0xDEAD_BEEF, u32::MAX] {
            let mut s = unit.clone();
            s.version = v;
            out.push((format!("unit with_version({v})"), s));
        }
        out.push((
            "unit empty".into(),
            scene(vec![], vec![], PhysicsConfig::default(), 1),
        ));
        out.push((
            "unit negative".into(),
            scene(
                vec![sb(
                    [-10, 0, -20, 0, -30, 0],
                    [0; 6],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                    [1, 0],
                    0,
                )],
                vec![],
                PhysicsConfig::default(),
                1,
            ),
        ));
        out.push((
            "unit multiple joints".into(),
            scene(
                vec![sb([0; 6], [0; 6], [0, 0, 0, 0, 0, 0, 1, 0], [1, 0], 0)],
                vec![
                    sj(0, 0, 1, [1, 0, 2, 0, 3, 0], [4, 0, 5, 0, 6, 0]),
                    sj(0, 0, 4, [7, 0, 8, 0, 9, 0], [10, 0, 11, 0, 12, 0]),
                ],
                PhysicsConfig::default(),
                1,
            ),
        ));
        // tests/analytic_scene_io_wiring.rs
        let body = |seed: i64, ty: u8| {
            sb(
                [seed, -seed, i64::MAX, i64::MIN, 0, seed * 3],
                [1, 2, 3, 4, 5, 6],
                [0, 0, 0, 0, 0, 0, 1, seed],
                [seed + 1, -1],
                ty,
            )
        };
        let joint =
            |a: u32, b: u32, ty: u8| sj(a, b, ty, [1, 2, 3, 4, 5, 6], [-1, -2, -3, -4, -5, -6]);
        let config = || PhysicsConfig::new(2, 3, [10, 20, 30, 40, 50, 60], [7, 8]);
        out.push((
            "wiring scene()".into(),
            scene(
                vec![body(1, 0), body(2, 1)],
                vec![joint(0, 1, 4)],
                config(),
                1,
            ),
        ));
        for v in [1u32, 0, 7, u32::MAX] {
            out.push((
                format!("wiring empty v{v}"),
                scene(vec![], vec![], config(), v),
            ));
        }
        out.push((
            "wiring layout one".into(),
            scene(vec![body(1, 2)], vec![joint(3, 4, 1)], config(), 1),
        ));
        out.push((
            "wiring round trip 40".into(),
            scene(
                (0..40i64).map(|i| body(i - 20, (i % 3) as u8)).collect(),
                vec![joint(0, 39, 0), joint(u32::MAX, 0, 255), joint(5, 6, 2)],
                config(),
                1,
            ),
        ));
        out.push((
            "wiring default empty".into(),
            scene(vec![], vec![], PhysicsConfig::default(), 1),
        ));
        out.push((
            "wiring big".into(),
            scene(
                (0..10_000i64).map(|i| body(i, (i % 3) as u8)).collect(),
                (0..5_000u32)
                    .map(|i| joint(i, i + 1, (i % 5) as u8))
                    .collect(),
                config(),
                1,
            ),
        ));
        out.push((
            "wiring fixture_scene".into(),
            scene(
                vec![sb(
                    [1, 2, 3, 4, 5, 6],
                    [-1, 0, 0, 0, 0, 7],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                    [2, 0],
                    1,
                )],
                vec![sj(0, 0, 3, [1, 0, 0, 0, 0, 0], [0, 0, -1, 0, 0, 0])],
                PhysicsConfig::new(4, 6, [0, 0, -10, 0, 0, 0], [0, -2]),
                1,
            ),
        ));
        for v in [0u32, 2, 0xDEAD_BEEF] {
            out.push((
                format!("wiring deprecated v{v}"),
                scene(vec![body(1, 0)], vec![], config(), v),
            ));
        }
        // examples/scene_snapshot_roundtrip.rs (書いた版と、版 2 を書いた版)
        let ex = example_scene();
        let mut ex2 = ex.clone();
        ex2.version = CURRENT_VERSION + 1;
        out.push(("example snapshot".into(), ex));
        out.push(("example snapshot v2".into(), ex2));
        out
    }

    /// src/scene_io.rs::tests::make_test_scene と同じ scene
    fn tests_scene() -> PhysicsScene {
        scene(
            vec![
                sb(
                    [0, 0, 10, 0, 0, 0],
                    [0; 6],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                    [1, 0],
                    0,
                ),
                sb(
                    [5, 0, 0, 0, -3, 0],
                    [1, 0, -1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                    [2, 0],
                    1,
                ),
            ],
            vec![sj(0, 1, 0, [0; 6], [1, 0, 0, 0, 0, 0])],
            PhysicsConfig::default(),
            CURRENT_VERSION,
        )
    }

    /// 既存の試験が手で書いた有効な文書 (tests/analytic_scene_io_wiring.rs と本 file の tests)
    fn hand_written_documents() -> Vec<(String, String)> {
        let json_of = |a: &str, t: &str| {
            format!(
            "{{\"version\": 1, \"config\": {{\"substeps\": 2, \"iterations\": 3, \"gravity\": [0,0,0,0,0,0], \"damping\": [1, 0]}}, \"bodies\": [], \"joints\": [{{\"body_a\": {a}, \"body_b\": 1, \"joint_type\": {t}, \"anchor_a\": [0,0,0,0,0,0], \"anchor_b\": [0,0,0,0,0,0]}}]}}"
        )
        };
        let with_head = |head: &str| {
            let body = json_of("0", "0");
            format!("{{{head}{}", &body[body.find("\"config\"").unwrap()..])
        };
        let mut out: Vec<(String, String)> = Vec::new();
        for a in ["0", "1", "4294967295", "  7  "] {
            out.push((format!("json_of({a:?}, 0)"), json_of(a, "0")));
        }
        for t in ["0", "1", "255"] {
            out.push((format!("json_of(0, {t})"), json_of("0", t)));
        }
        out.push(("defaults".into(), "{\"config\": {\"gravity\": [1,2,3,4,5,6], \"damping\": [9, 9]}, \"bodies\": [{\"position\":[0,0,0,0,0,0],\"velocity\":[0,0,0,0,0,0],\"rotation\":[0,0,0,0,0,0,0,0],\"mass\":[1,0]}], \"joints\": [{\"anchor_a\": [0,0,0,0,0,0], \"anchor_b\": [0,0,0,0,0,1]}]}".into()));
        out.push((
            "minimal".into(),
            "{\"config\": {\"gravity\": [1,2,3,4,5,6], \"damping\": [9, 9]}}".into(),
        ));
        out.push(("compact".into(), "{\"config\":{\"gravity\":[1,2,3,4,5,6],\"damping\":[1,2]},\"bodies\":[{\"position\":[0,0,0,0,0,0],\"velocity\":[0,0,0,0,0,0],\"rotation\":[0,0,0,0,0,0,0,0],\"mass\":[1,0],\"body_type\":2}],\"joints\":[{\"anchor_a\":[0,0,0,0,0,0],\"anchor_b\":[0,0,0,0,0,0],\"body_b\":9}]}".into()));
        out.push((
            "stray".into(),
            "{\"config\":{\"gravity\":[1,2,3,4,5,6],\"damping\":[1,2]},\"note\":\"version\"}"
                .into(),
        ));
        out.push((
            "head ws".into(),
            with_head(" \n\t\"version\"\r\n :\n 1 \n,"),
        ));
        out.push((
            "nested version in config".into(),
            with_head("").replace("\"config\": {", "\"config\": {\"version\": 2, "),
        ));
        out.push((
            "nested version in meta".into(),
            with_head("").replace(
                "\"bodies\": []",
                "\"bodies\": [], \"meta\": {\"version\": 2}",
            ),
        ));
        out.push(("head version 1".into(), with_head("\"version\": 1, ")));
        out.push((
            "unit config only".into(),
            "{\"config\":{\"gravity\":[0,0,0,0,0,0],\"damping\":[1,0]}}".into(),
        ));
        out.push((
            "unit nested version".into(),
            "{\"config\":{\"version\":2,\"gravity\":[0,0,0,0,0,0],\"damping\":[1,0]}}".into(),
        ));
        out
    }

    /// 既存の有効な scene すべてで、旧 reader と解析木の reader が同じ値を返す
    #[test]
    fn tree_reader_matches_the_old_reader_on_every_valid_scene() {
        let mut docs: Vec<(String, String, Option<PhysicsScene>)> = Vec::new();
        for (name, s) in writer_scenes() {
            docs.push((name, scene_to_json(&s), Some(s)));
        }
        for name in ["scene_v1.json", "scene_v2.json"] {
            let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures")
                .join(name);
            docs.push((name.into(), std::fs::read_to_string(p).unwrap(), None));
        }
        for (name, text) in hand_written_documents() {
            docs.push((name, text, None));
        }
        let mut diffs = Vec::new();
        for (name, text, written) in &docs {
            let top = parse_json_document(text).unwrap();
            let version = scene_json_version_of(&top).unwrap();
            let new = scene_from_json(&top, version);
            let old = legacy_json::parse_scene_json(text, version);
            match (&new, &old) {
                (Ok(n), Ok(o)) if n == o => {}
                _ => diffs.push(format!("{name}: new {new:?} old {old:?}")),
            }
            if let (Some(w), Ok(n)) = (written, &new) {
                if w != n {
                    diffs.push(format!("{name}: round trip differs"));
                }
            }
        }
        println!(
            "scene JSON differential: {} documents, {} diffs",
            docs.len(),
            diffs.len()
        );
        assert!(docs.len() > 40, "{}", docs.len());
        assert!(diffs.is_empty(), "{diffs:#?}");
    }

    const CFG: &str = "\"config\": {\"substeps\": 2, \"iterations\": 3, \"gravity\": [1,2,3,4,5,6], \"damping\": [7,8]}";
    const BODY: &str = "{\"position\": [1,2,3,4,5,6], \"velocity\": [1,1,1,1,1,1], \"rotation\": [0,0,0,0,0,0,1,0], \"mass\": [2,0], \"body_type\": 1}";
    const JOINT: &str = "{\"body_a\": 3, \"body_b\": 4, \"joint_type\": 2, \"anchor_a\": [1,0,0,0,0,0], \"anchor_b\": [0,0,1,0,0,0]}";

    fn doc(cfg: &str, body: &str, joint: &str) -> String {
        format!("{{\"version\": 1, {cfg}, \"bodies\": [{body}], \"joints\": [{joint}]}}")
    }

    fn base() -> PhysicsScene {
        named(&doc(CFG, BODY, JOINT)).unwrap()
    }

    fn dup(path: &str) -> Result<PhysicsScene, InvalidSceneJson> {
        Err(InvalidSceneJson::DuplicateKey { path: path.into() })
    }

    /// 重複 key はどの階層でも名前つきで拒否 (先勝ちにしない)
    #[test]
    fn duplicate_keys_at_every_level_are_refused() {
        let d = doc(CFG, BODY, JOINT);
        assert_eq!(base().config.substeps, 2);
        // top-level
        assert_eq!(
            named(&d.replace("\"bodies\"", &format!("{CFG}, \"bodies\""))),
            dup("config")
        );
        assert_eq!(
            named(&d.replace("\"joints\"", "\"bodies\": [], \"joints\"")),
            dup("bodies")
        );
        assert_eq!(
            named(&d.replace("\"bodies\"", "\"joints\": [], \"bodies\"")),
            dup("joints")
        );
        let e = parse_json_document(&d.replace("\"bodies\"", "\"version\": 1, \"bodies\""))
            .and_then(|t| scene_json_version_of(&t));
        assert_eq!(e, Err(InvalidSceneJsonVersion::DuplicateVersion));
        // 両方とも完全な config の重複
        let both = format!("{{{CFG}, {CFG}}}");
        assert_eq!(named(&both), dup("config"));
        // config / body / joint の中 (両順序)
        for (k, v) in [
            ("substeps", "9"),
            ("iterations", "9"),
            ("gravity", "[0,0,0,0,0,0]"),
            ("damping", "[0,0]"),
        ] {
            let c = CFG.replace(&format!("\"{k}\""), &format!("\"{k}\": {v}, \"{k}\""));
            assert_eq!(
                named(&doc(&c, BODY, JOINT)),
                dup(&format!("config.{k}")),
                "{k}"
            );
        }
        for k in ["position", "velocity", "rotation", "mass", "body_type"] {
            let b = BODY.replace(&format!("\"{k}\""), &format!("\"{k}\": 0, \"{k}\""));
            assert_eq!(
                named(&doc(CFG, &b, JOINT)),
                dup(&format!("bodies[0].{k}")),
                "{k}"
            );
        }
        for k in ["body_a", "body_b", "joint_type", "anchor_a", "anchor_b"] {
            let j = JOINT.replace(&format!("\"{k}\""), &format!("\"{k}\": 0, \"{k}\""));
            assert_eq!(
                named(&doc(CFG, BODY, &j)),
                dup(&format!("joints[0].{k}")),
                "{k}"
            );
        }
        // 2 番目の body、読まない member の中、配列の中の object
        let two = d.replace(
            &format!("[{BODY}]"),
            &format!(
                "[{BODY}, {}]",
                BODY.replace("\"mass\"", "\"mass\": [1,0], \"mass\"")
            ),
        );
        assert_eq!(named(&two), dup("bodies[1].mass"));
        assert_eq!(
            named(&d.replace("\"bodies\"", "\"meta\": {\"a\": 1, \"a\": 2}, \"bodies\"")),
            dup("meta.a")
        );
        assert_eq!(
            named(&d.replace(
                "\"bodies\"",
                "\"meta\": [[{\"x\": {\"y\": 1, \"y\": 1}}]], \"bodies\""
            )),
            dup("meta[0][0].x.y")
        );
        // escape で綴った同じ key (BMP と surrogate pair)
        let esc = CFG.replace(
            "\"damping\"",
            "\"gr\\u0061vity\": [0,0,0,0,0,0], \"damping\"",
        );
        assert_eq!(named(&doc(&esc, BODY, JOINT)), dup("config.gravity"));
        let pair = d.replace(
            "\"bodies\"",
            "\"\\ud83d\\ude00\": 1, \"\u{1F600}\": 2, \"bodies\"",
        );
        assert_eq!(named(&pair), dup("\u{1F600}"));
    }

    /// 入れ子の中にだけある同名 key は、読んでいる member の代わりにならない
    #[test]
    fn nested_keys_never_stand_in_for_the_member() {
        let want = base();
        let extra = |obj: &str, k: &str, v: &str| {
            obj.replacen('{', &format!("{{\"extra\": {{\"{k}\": {v}}}, "), 1)
        };
        // config の各 field: 入れ子の値が先にあっても自分の member を読む
        for (k, v) in [
            ("substeps", "99"),
            ("iterations", "99"),
            ("gravity", "[9,9,9,9,9,9]"),
            ("damping", "[9,9]"),
        ] {
            let c = CFG.replacen("{", &format!("{{\"extra\": {{\"{k}\": {v}}}, "), 1);
            assert_eq!(named(&doc(&c, BODY, JOINT)), Ok(want.clone()), "config.{k}");
        }
        for (k, v) in [
            ("position", "[9,9,9,9,9,9]"),
            ("velocity", "[9,9,9,9,9,9]"),
            ("rotation", "[9,9,9,9,9,9,9,9]"),
            ("mass", "[9,9]"),
            ("body_type", "2"),
        ] {
            assert_eq!(
                named(&doc(CFG, &extra(BODY, k, v), JOINT)),
                Ok(want.clone()),
                "body.{k}"
            );
        }
        for (k, v) in [
            ("body_a", "9"),
            ("body_b", "9"),
            ("joint_type", "4"),
            ("anchor_a", "[9,9,9,9,9,9]"),
            ("anchor_b", "[9,9,9,9,9,9]"),
        ] {
            assert_eq!(
                named(&doc(CFG, BODY, &extra(JOINT, k, v))),
                Ok(want.clone()),
                "joint.{k}"
            );
        }
        // 自分の member が無く入れ子にだけある: 必須なら欠落、任意なら既定値
        let gone = |obj: &str, k: &str, v: &str| {
            let start = obj.find(&format!("\"{k}\"")).unwrap();
            let end = obj[start..].find([',', '}']).map(|i| start + i).unwrap();
            let end = if obj[start..].starts_with(&format!("\"{k}\": [")) {
                start + obj[start..].find(']').unwrap() + 1
            } else {
                end
            };
            let rest = obj[end..].trim_start_matches(',');
            let cut = format!("{}{}", &obj[..start], rest.trim_start());
            let cut = cut.replace(", }", "}").replace(",}", "}");
            extra(&cut, k, v)
        };
        let missing = |p: &str| Err(InvalidSceneJson::MissingMember { path: p.into() });
        let c = gone(&CFG["\"config\": ".len()..], "gravity", "[9,9,9,9,9,9]");
        assert_eq!(
            named(&doc(&format!("\"config\": {c}"), BODY, JOINT)),
            missing("config.gravity")
        );
        let c = gone(&CFG["\"config\": ".len()..], "substeps", "99");
        assert_eq!(
            named(&doc(&format!("\"config\": {c}"), BODY, JOINT))
                .unwrap()
                .config
                .substeps,
            8
        );
        assert_eq!(
            named(&doc(CFG, &gone(BODY, "mass", "[9,9]"), JOINT)),
            missing("bodies[0].mass")
        );
        assert_eq!(
            named(&doc(CFG, &gone(BODY, "body_type", "2"), JOINT))
                .unwrap()
                .bodies[0]
                .body_type,
            0
        );
        assert_eq!(
            named(&doc(CFG, BODY, &gone(JOINT, "anchor_b", "[9,9,9,9,9,9]"))),
            missing("joints[0].anchor_b")
        );
        assert_eq!(
            named(&doc(CFG, BODY, &gone(JOINT, "body_a", "9")))
                .unwrap()
                .joints[0]
                .body_a,
            0
        );
        // body の中の "gravity" が top-level の config より前にある
        let early = format!(
            "{{\"bodies\": [{}], {CFG}}}",
            BODY.replace("{", "{\"gravity\": [5,5,5,5,5,5], ")
        );
        assert_eq!(named(&early).unwrap().config.gravity, [1, 2, 3, 4, 5, 6]);
        let early_missing = format!(
            "{{\"bodies\": [{}], \"config\": {{\"damping\": [1,0]}}}}",
            BODY.replace("{", "{\"gravity\": [5,5,5,5,5,5], ")
        );
        assert_eq!(named(&early_missing), missing("config.gravity"));
        // config の中の "bodies" / "joints" は scene の bodies / joints でない
        let inner = format!(
            "{{{}}}",
            CFG.replace(
                "\"substeps\"",
                &format!("\"bodies\": [{BODY}], \"joints\": [{JOINT}], \"substeps\"")
            )
        );
        let s = named(&inner).unwrap();
        assert!(s.bodies.is_empty() && s.joints.is_empty());
        // body の中にだけある "config" は scene の config でない
        let only_nested = format!(
            "{{\"bodies\": [{}]}}",
            BODY.replace("{", &format!("{{{CFG}, "))
        );
        assert_eq!(named(&only_nested), missing("config"));
        // escape で綴った key は同じ key として読む
        let esc = doc(
            &CFG.replace("\"config\"", "\"\\u0063onfig\"")
                .replace("\"gravity\"", "\"gr\\u0061vity\""),
            BODY,
            JOINT,
        );
        assert_eq!(named(&esc), Ok(want));
    }

    /// 文字列の中の括弧と escape した引用符は構造に数えない
    #[test]
    fn brackets_and_quotes_inside_strings_are_not_structure() {
        let want = base();
        let noisy = "\"}{][,\\\"}\\\"{ \\\\\"";
        let b = BODY.replace("{", &format!("{{\"note\": {noisy}, "));
        let j = JOINT.replace('{', "{\"n\\\"}\": \"{[\", ");
        let d = doc(CFG, &b, &j).replace(
            "\"bodies\"",
            &format!("\"meta\": [{noisy}, \"]\"], \"bodies\""),
        );
        assert_eq!(named(&d), Ok(want.clone()));
        // body 配列の中の文字列 (旧 reader の split は "}" を object の終わりと数えた)
        let two = doc(
            CFG,
            &format!(
                "{b}, {}",
                BODY.replace("\"body_type\": 1", "\"body_type\": 2, \"s\": \"{\"")
            ),
            JOINT,
        );
        let s = named(&two).unwrap();
        assert_eq!(s.bodies.len(), 2);
        assert_eq!((s.bodies[0].body_type, s.bodies[1].body_type), (1, 2));
        assert_ne!(
            legacy_json::parse_scene_json(&two, 1).map(|s| s.bodies.len()),
            Ok(2)
        );
    }

    /// 深さの上限ちょうどは読め、1 つ超えると名前つき error (stack を溢れさせない)
    #[test]
    fn nesting_depth_limit_is_exact_and_does_not_overflow() {
        let d = doc(CFG, BODY, JOINT);
        // top-level object が深さ 1、"meta" の値の n 段の入れ子で深さ 1 + n
        let arrays = |n: usize| {
            d.replace(
                "\"bodies\"",
                &format!("\"meta\": {}{}, \"bodies\"", "[".repeat(n), "]".repeat(n)),
            )
        };
        let objects = |n: usize| {
            d.replace(
                "\"bodies\"",
                &format!(
                    "\"meta\": {}1{}, \"bodies\"",
                    "{\"a\": ".repeat(n),
                    "}".repeat(n)
                ),
            )
        };
        let at = |t: &str| t.find("\"meta\"").unwrap() + "\"meta\": ".len();
        for make in [&arrays as &dyn Fn(usize) -> String, &objects] {
            let ok = make(MAX_SCENE_JSON_DEPTH - 1);
            assert_eq!(named(&ok), Ok(base()));
            let over = make(MAX_SCENE_JSON_DEPTH);
            let step = if over.contains("{\"a\": {") {
                "{\"a\": ".len()
            } else {
                1
            };
            assert_eq!(
                parse_json_document(&over),
                Err(InvalidSceneJsonVersion::TooDeep {
                    offset: at(&over) + step * (MAX_SCENE_JSON_DEPTH - 1)
                })
            );
        }
        // 上限ちょうどの深さの重複 key も見つける
        let deep_dup = d.replace(
            "\"bodies\"",
            &format!(
                "\"meta\": {}{{\"k\": 1, \"k\": 2}}{}, \"bodies\"",
                "[".repeat(MAX_SCENE_JSON_DEPTH - 2),
                "]".repeat(MAX_SCENE_JSON_DEPTH - 2)
            ),
        );
        assert_eq!(
            named(&deep_dup),
            dup(&format!("meta{}.k", "[0]".repeat(MAX_SCENE_JSON_DEPTH - 2)))
        );
        // 非常に深い入力 (100 万段) も上限で止まる
        for open in ["[", "{\"a\":"] {
            let huge = format!("{{\"x\": {}", open.repeat(1_000_000));
            assert!(matches!(
                parse_json_document(&huge),
                Err(InvalidSceneJsonVersion::TooDeep { .. })
            ));
        }
    }

    /// 巨大な数は wrap も panic もせず名前つき error、message は先頭だけ
    #[test]
    fn huge_numbers_are_named_errors_with_bounded_messages() {
        let big = format!("1{}", "0".repeat(1000));
        let oor = |p: &str, v: &str| {
            Err(InvalidSceneJson::OutOfRange {
                path: p.into(),
                value: v.into(),
            })
        };
        let c = CFG.replace("\"substeps\": 2", &format!("\"substeps\": {big}"));
        let e = named(&doc(&c, BODY, JOINT));
        assert_eq!(e, oor("config.substeps", &big));
        let shown = e.unwrap_err().to_string();
        assert!(
            shown.len() < 120 && shown.contains("1001 bytes") && shown.contains("config.substeps"),
            "{shown}"
        );
        let neg = format!("-{big}");
        let c = CFG.replace("[1,2,3,4,5,6]", &format!("[1,2,{neg},4,5,6]"));
        assert_eq!(named(&doc(&c, BODY, JOINT)), oor("config.gravity[2]", &neg));
        let b = BODY.replace("\"mass\": [2,0]", "\"mass\": [9223372036854775808,0]");
        assert_eq!(
            named(&doc(CFG, &b, JOINT)),
            oor("bodies[0].mass[0]", "9223372036854775808")
        );
        let b = BODY.replace(
            "\"mass\": [2,0]",
            "\"mass\": [-9223372036854775808,9223372036854775807]",
        );
        assert_eq!(
            named(&doc(CFG, &b, JOINT)).unwrap().bodies[0].mass,
            [i64::MIN, i64::MAX]
        );
        let b = BODY.replace("\"mass\": [2,0]", "\"mass\": [1e999999999,0]");
        assert!(matches!(
            named(&doc(CFG, &b, JOINT)),
            Err(InvalidSceneJson::WrongType { .. })
        ));
        // 読まない member の巨大な数は無関係
        let d = doc(CFG, BODY, JOINT)
            .replace("\"bodies\"", &format!("\"meta\": {big}e{big}, \"bodies\""));
        assert_eq!(named(&d), Ok(base()));
        // 版の error の message も先頭だけ
        let v = InvalidSceneJsonVersion::OutOfRange { value: big.clone() }.to_string();
        assert!(v.len() < 120 && v.contains("1001 bytes"), "{v}");
        let v = InvalidSceneJsonVersion::NotAnUnsignedInteger {
            value: format!("{big}.5"),
        }
        .to_string();
        assert!(v.len() < 140 && v.contains("1003 bytes"), "{v}");
        assert_eq!(shown_value("12345"), "12345");
        assert_eq!(
            shown_value(&"é".repeat(20)),
            format!("{}... (40 bytes)", "é".repeat(16))
        );
        assert_eq!(shown_value(&"x".repeat(32)), "x".repeat(32));
        assert_eq!(
            shown_value(&format!("a{}", "é".repeat(16))),
            format!("a{}... (33 bytes)", "é".repeat(15))
        );
    }

    /// 型と範囲の誤りは名前つき error (path、期待、実際)
    #[test]
    fn wrong_types_and_ranges_are_named_errors() {
        use InvalidSceneJson as E;
        let ty = |p: &str, expected: &'static str, found: &'static str| {
            Err(E::WrongType {
                path: p.into(),
                expected,
                found,
            })
        };
        let oor = |p: &str, v: &str| {
            Err(E::OutOfRange {
                path: p.into(),
                value: v.into(),
            })
        };
        let d = doc(CFG, BODY, JOINT);
        let cases: Vec<(String, Result<PhysicsScene, E>)> = vec![
            (
                format!("{{\"config\": [1], \"bodies\": [{BODY}]}}"),
                ty("config", "an object", "an array"),
            ),
            (
                d.replace(&format!("[{BODY}]"), "{}"),
                ty("bodies", "an array", "an object"),
            ),
            (
                d.replace(&format!("[{JOINT}]"), "\"x\""),
                ty("joints", "an array", "a string"),
            ),
            (
                d.replace(&format!("[{BODY}]"), "[1]"),
                ty("bodies[0]", "an object", "a number"),
            ),
            (
                d.replace(&format!("[{JOINT}]"), &format!("[{JOINT}, null]")),
                ty("joints[1]", "an object", "null"),
            ),
            (
                d.replace("[1,2,3,4,5,6]", "\"g\""),
                ty("config.gravity", "an array", "a string"),
            ),
            (
                d.replace("[1,2,3,4,5,6]", "[1,2,true,4,5,6]"),
                ty("config.gravity[2]", "an integer", "a boolean"),
            ),
            (
                d.replace("[1,2,3,4,5,6]", "[1,2,3,4,5]"),
                Err(E::WrongLength {
                    path: "config.gravity".into(),
                    expected: 6,
                    found: 5,
                }),
            ),
            (
                d.replace("\"mass\": [2,0]", "\"mass\": [1.5,0]"),
                ty(
                    "bodies[0].mass[0]",
                    "an integer",
                    "a number with a fraction or exponent",
                ),
            ),
            (
                d.replace("\"mass\": [2,0]", "\"mass\": [[2],0]"),
                ty("bodies[0].mass[0]", "an integer", "an array"),
            ),
            (
                d.replace("\"body_type\": 1", "\"body_type\": \"1\""),
                ty("bodies[0].body_type", "a non-negative integer", "a string"),
            ),
            (
                d.replace("\"body_type\": 1", "\"body_type\": 300"),
                oor("bodies[0].body_type", "300"),
            ),
            (
                d.replace("\"body_type\": 1", "\"body_type\": -1"),
                ty(
                    "bodies[0].body_type",
                    "a non-negative integer",
                    "a negative number",
                ),
            ),
            (
                d.replace("\"body_a\": 3", "\"body_a\": 4294967296"),
                oor("joints[0].body_a", "4294967296"),
            ),
            (
                d.replace("\"body_a\": 3", "\"body_a\": -0"),
                ty(
                    "joints[0].body_a",
                    "a non-negative integer",
                    "a negative number",
                ),
            ),
            (
                d.replace("\"substeps\": 2", "\"substeps\": null"),
                ty("config.substeps", "a non-negative integer", "null"),
            ),
            (
                d.replace("\"iterations\": 3", "\"iterations\": 3e0"),
                ty(
                    "config.iterations",
                    "a non-negative integer",
                    "a number with a fraction or exponent",
                ),
            ),
            (
                d.replace("\"joint_type\": 2", "\"joint_type\": {}"),
                ty(
                    "joints[0].joint_type",
                    "a non-negative integer",
                    "an object",
                ),
            ),
            (
                "{\"bodies\": []}".into(),
                Err(E::MissingMember {
                    path: "config".into(),
                }),
            ),
        ];
        for (text, want) in cases {
            assert_eq!(named(&text), want, "{text}");
        }
        // -0 は i64 の 0
        assert_eq!(
            named(&d.replace("[1,2,3,4,5,6]", "[-0,2,3,4,5,6]"))
                .unwrap()
                .config
                .gravity[0],
            0
        );
        // 表示
        for (e, has) in [
            (
                E::DuplicateKey {
                    path: "config".into(),
                },
                "more than one \"config\"",
            ),
            (
                E::MissingMember {
                    path: "bodies[0].mass".into(),
                },
                "no \"bodies[0].mass\"",
            ),
            (
                E::WrongType {
                    path: "p".into(),
                    expected: "an array",
                    found: "null",
                },
                "\"p\" is null, expected an array",
            ),
            (
                E::WrongLength {
                    path: "p".into(),
                    expected: 6,
                    found: 5,
                },
                "has 5 items, expected 6",
            ),
            (
                E::OutOfRange {
                    path: "p".into(),
                    value: "300".into(),
                },
                "out of range: 300",
            ),
        ] {
            assert!(e.to_string().contains(has), "{e}");
        }
    }

    /// loader は field の error を `InvalidData` の中の `InvalidSceneJson` として返す
    #[test]
    fn loader_reports_field_errors_inside_invalid_data() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.json");
        std::fs::write(
            &path,
            doc(CFG, BODY, JOINT).replace("\"joints\"", "\"bodies\": [], \"joints\""),
        )
        .unwrap();
        let e = load_scene_json(&path).unwrap_err();
        assert_eq!(e.kind(), std::io::ErrorKind::InvalidData);
        assert_eq!(
            e.get_ref()
                .and_then(|i| i.downcast_ref::<InvalidSceneJson>())
                .cloned(),
            Some(InvalidSceneJson::DuplicateKey {
                path: "bodies".into()
            })
        );
        std::fs::write(&path, doc(CFG, BODY, JOINT)).unwrap();
        assert_eq!(load_scene_json(&path).unwrap(), base());
    }
}
