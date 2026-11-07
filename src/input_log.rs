//! What a visitor did to a world, in order and by step, so that it can be done again: from the log and the same first
//! state, a deterministic world comes to the same `state_digest`. A log travels as a short code with the first bytes
//! of the digest it should reach, which is how two places show that they agree.

#[cfg(not(feature = "std"))]
use alloc::string::String;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::math::Fix128;
use crate::solver::PhysicsWorld;

/// An input that can be written into a log
pub trait LogInput: Sized {
    /// Append this input's bytes
    fn write(&self, out: &mut Vec<u8>);
    /// Read one input from the front of `bytes`, moving past it. `None` when they are not one
    fn read(bytes: &mut &[u8]) -> Option<Self>;
}

/// Why a log or a code was refused
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogError {
    /// A step before the last one recorded
    Backwards,
    /// The text is not a code
    NotACode,
}

/// Steps and inputs, in order
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InputLog<I> {
    entries: Vec<(u64, I)>,
}

impl<I> Default for InputLog<I> {
    fn default() -> Self {
        Self { entries: Vec::new() }
    }
}

impl<I: LogInput> InputLog<I> {
    /// An empty log
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// What happened at `step`
    pub fn record(&mut self, _step: u64, _input: I) -> Result<(), LogError> {
        todo!("record")
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
        _world: &mut PhysicsWorld,
        _steps: u64,
        _dt: Fix128,
        _apply: impl FnMut(&mut PhysicsWorld, &I),
    ) {
        todo!("replay")
    }

    /// The log as a code that survives a link: the digest's first four bytes, then the inputs
    #[must_use]
    pub fn to_code(&self, _digest: &[u8; 32]) -> String {
        todo!("to code")
    }

    /// A code back into the digest's first bytes and the log
    pub fn from_code(_code: &str) -> Result<([u8; 4], Self), LogError> {
        todo!("from code")
    }
}
