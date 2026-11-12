//! What a visitor did to a world, in order and by step, so that it can be done again: from the log and the same first
//! state, a deterministic world comes to the same `state_digest`. A log travels as a short code with the first bytes
//! of the digest it should reach, which is how two places show that they agree.

#[cfg(not(feature = "std"))]
use alloc::string::String;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::math::Fix128;
use crate::solver::PhysicsWorld;
use crate::touch::Touch;

/// The largest byte form of a touch log [`InputLog::from_bytes_bounded`] reads (1 MiB)
pub const TOUCH_LOG_MAX_BYTES: usize = 1 << 20;

/// An input that can be written into a log
pub trait LogInput: Sized {
    /// Append this input's bytes
    fn write(&self, out: &mut Vec<u8>);
    /// Read one input from the front of `bytes`, moving past it. `None` when they are not one
    fn read(bytes: &mut &[u8]) -> Option<Self>;
}

/// Bytes as an input, for a host that writes its own: their length, then them
impl LogInput for Vec<u8> {
    fn write(&self, out: &mut Vec<u8>) {
        let mut n = self.len();
        while n >= 0x80 {
            out.push((n as u8 & 0x7f) | 0x80);
            n >>= 7;
        }
        out.push(n as u8);
        out.extend_from_slice(self);
    }

    fn read(bytes: &mut &[u8]) -> Option<Self> {
        let mut n = 0usize;
        let mut shift = 0u32;
        loop {
            let (&b, after) = bytes.split_first()?;
            *bytes = after;
            if shift >= usize::MAX.count_ones() {
                return None;
            }
            n |= usize::from(b & 0x7f).checked_shl(shift)?;
            shift += 7;
            if b & 0x80 == 0 {
                break;
            }
        }
        if n > bytes.len() {
            return None;
        }
        let (mine, after) = bytes.split_at(n);
        *bytes = after;
        Some(mine.to_vec())
    }
}

/// Why a log or a code was refused
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogError {
    /// A step before the last one recorded
    Backwards,
    /// The text is not a code
    NotACode,
}

/// Why the byte form of a touch log was refused
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogDecodeError {
    /// The bytes end before the log does
    Truncated,
    /// Bytes are left after the log
    Trailing,
    /// A step at or after the bound
    StepOutOfRange,
    /// A grab of a body at or after the bound
    BodyOutOfRange,
    /// A step or a count that does not fit in 64 bits
    Overflow,
    /// More bytes than [`TOUCH_LOG_MAX_BYTES`]
    TooLarge,
    /// Bytes that are not a touch, or not in the one form [`InputLog::to_bytes`] writes
    Malformed,
}

fn write_varint(mut n: u64, out: &mut Vec<u8>) {
    while n >= 0x80 {
        out.push(n.to_le_bytes()[0] | 0x80);
        n >>= 7;
    }
    out.push(n.to_le_bytes()[0]);
}

/// A LEB128 number from the front of `bytes`: `Truncated` when they end inside it, `Overflow` past 64 bits
fn read_varint(bytes: &mut &[u8]) -> Result<u64, LogDecodeError> {
    let mut n = 0u64;
    let mut shift = 0u32;
    loop {
        let (&b, after) = bytes.split_first().ok_or(LogDecodeError::Truncated)?;
        *bytes = after;
        if shift > 63 || (shift == 63 && b & 0x7e != 0) {
            return Err(LogDecodeError::Overflow);
        }
        n |= u64::from(b & 0x7f) << shift;
        shift += 7;
        if b & 0x80 == 0 {
            return Ok(n);
        }
    }
}

/// Steps and inputs, in order
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InputLog<I> {
    entries: Vec<(u64, I)>,
}

impl<I> Default for InputLog<I> {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
        }
    }
}

impl<I: LogInput> InputLog<I> {
    /// An empty log
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// What happened at `step`
    pub fn record(&mut self, step: u64, input: I) -> Result<(), LogError> {
        if self.entries.last().is_some_and(|(last, _)| step < *last) {
            return Err(LogError::Backwards);
        }
        self.entries.push((step, input));
        Ok(())
    }

    /// The entries, in order
    #[must_use]
    pub fn entries(&self) -> &[(u64, I)] {
        &self.entries
    }

    /// Run `steps` steps of `dt`, giving `apply` each input before the step it was recorded at. An input recorded at
    /// or after `steps` is given after the last step
    pub fn replay(
        &self,
        world: &mut PhysicsWorld,
        steps: u64,
        dt: Fix128,
        mut apply: impl FnMut(&mut PhysicsWorld, &I),
    ) {
        let mut next = 0;
        for step in 0..steps {
            while next < self.entries.len() && self.entries[next].0 <= step {
                apply(world, &self.entries[next].1);
                next += 1;
            }
            world.step(dt);
        }
        for (_, input) in &self.entries[next..] {
            apply(world, input);
        }
    }

    /// The log as bytes: `[count varint][(step delta varint, input)…]`, the deltas from the step before (the first
    /// from 0)
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        write_varint(self.entries.len() as u64, &mut out);
        let mut last = 0u64;
        for (step, input) in &self.entries {
            write_varint(step - last, &mut out);
            last = *step;
            input.write(&mut out);
        }
        out
    }

    /// The log as a code that survives a link: the digest's first four bytes, then the inputs
    #[must_use]
    pub fn to_code(&self, digest: &[u8; 32]) -> String {
        let mut bytes = Vec::new();
        let mut last = 0u64;
        for (step, input) in &self.entries {
            let mut gap = step - last;
            last = *step;
            while gap >= 0x80 {
                bytes.push((gap as u8 & 0x7f) | 0x80);
                gap >>= 7;
            }
            bytes.push(gap as u8);
            input.write(&mut bytes);
        }
        let mut out = String::with_capacity(9 + bytes.len() * 4 / 3 + 4);
        for b in &digest[..4] {
            out.push(char::from(HEX[usize::from(b >> 4)]));
            out.push(char::from(HEX[usize::from(b & 15)]));
        }
        out.push('-');
        for chunk in bytes.chunks(3) {
            let n =
                chunk.iter().fold(0u32, |n, b| (n << 8) | u32::from(*b)) << (8 * (3 - chunk.len()));
            for k in 0..=chunk.len() {
                out.push(char::from(LETTERS[((n >> (18 - 6 * k)) & 63) as usize]));
            }
        }
        out
    }

    /// A code back into the digest's first bytes and the log
    pub fn from_code(code: &str) -> Result<([u8; 4], Self), LogError> {
        let (head, tail) = code.split_once('-').ok_or(LogError::NotACode)?;
        let head = head.as_bytes();
        if head.len() != 8 {
            return Err(LogError::NotACode);
        }
        let mut digest = [0u8; 4];
        for (k, d) in digest.iter_mut().enumerate() {
            *d = (hex_value(head[k * 2]).ok_or(LogError::NotACode)? << 4)
                | hex_value(head[k * 2 + 1]).ok_or(LogError::NotACode)?;
        }
        let mut bytes = Vec::with_capacity(tail.len() * 3 / 4);
        for chunk in tail.as_bytes().chunks(4) {
            // one letter alone cannot end a code: it would hold less than a byte
            if chunk.len() == 1 {
                return Err(LogError::NotACode);
            }
            let mut n = 0u32;
            for (k, c) in chunk.iter().enumerate() {
                n |= letter_value(*c).ok_or(LogError::NotACode)? << (18 - 6 * k);
            }
            for k in 0..chunk.len() - 1 {
                bytes.push((n >> (16 - 8 * k)) as u8);
            }
        }
        let mut rest = bytes.as_slice();
        let mut log = Self::new();
        let mut step = 0u64;
        while !rest.is_empty() {
            let mut gap = 0u64;
            let mut shift = 0u32;
            loop {
                let (&b, after) = rest.split_first().ok_or(LogError::NotACode)?;
                rest = after;
                if shift >= 64 || (shift == 63 && b & 0x7e != 0) {
                    return Err(LogError::NotACode);
                }
                gap |= u64::from(b & 0x7f) << shift;
                shift += 7;
                if b & 0x80 == 0 {
                    break;
                }
            }
            step = step.checked_add(gap).ok_or(LogError::NotACode)?;
            let input = I::read(&mut rest).ok_or(LogError::NotACode)?;
            log.entries.push((step, input));
        }
        Ok((digest, log))
    }
}

const HEX: &[u8; 16] = b"0123456789abcdef";
const LETTERS: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

fn hex_value(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        _ => None,
    }
}

fn letter_value(c: u8) -> Option<u32> {
    match c {
        b'A'..=b'Z' => Some(u32::from(c - b'A')),
        b'a'..=b'z' => Some(u32::from(c - b'a') + 26),
        b'0'..=b'9' => Some(u32::from(c - b'0') + 52),
        b'-' => Some(62),
        b'_' => Some(63),
        _ => None,
    }
}

impl InputLog<Touch> {
    /// A touch log from the bytes [`Self::to_bytes`] writes, read only in that exact form: every step below
    /// `max_step`, every grab of a body below `max_body`, no bytes left over, at most [`TOUCH_LOG_MAX_BYTES`].
    ///
    /// # Errors
    ///
    /// The [`LogDecodeError`] that applies (the first one met, reading from the front).
    pub fn from_bytes_bounded(
        bytes: &[u8],
        max_step: u64,
        max_body: u32,
    ) -> Result<Self, LogDecodeError> {
        if bytes.len() > TOUCH_LOG_MAX_BYTES {
            return Err(LogDecodeError::TooLarge);
        }
        let mut rest = bytes;
        let count = read_varint(&mut rest)?;
        // every entry takes at least two bytes (a delta and a tag), so a larger count is cut short
        if count > (rest.len() / 2) as u64 {
            return Err(LogDecodeError::Truncated);
        }
        let mut log = Self::new();
        let mut step = 0u64;
        for _ in 0..count {
            let delta = read_varint(&mut rest)?;
            step = step.checked_add(delta).ok_or(LogDecodeError::Overflow)?;
            if step >= max_step {
                return Err(LogDecodeError::StepOutOfRange);
            }
            if rest.is_empty() {
                return Err(LogDecodeError::Truncated);
            }
            let touch = Touch::read(&mut rest).ok_or(LogDecodeError::Malformed)?;
            if let Touch::Grab { body, .. } = touch {
                if body >= max_body {
                    return Err(LogDecodeError::BodyOutOfRange);
                }
            }
            log.entries.push((step, touch));
        }
        if !rest.is_empty() {
            return Err(LogDecodeError::Trailing);
        }
        // one form only: a longer spelling of the same numbers (a padded varint) reads as the same log
        if log.to_bytes() != bytes {
            return Err(LogDecodeError::Malformed);
        }
        Ok(log)
    }
}
