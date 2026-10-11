//! Hash maps with no random seed.
//!
//! `std::collections::HashMap::new()` seeds its hasher from OS randomness, so
//! the same keys land in a different order in every process. The crate's maps
//! use [`DetHashMap`] instead: FNV-1a (64-bit) with the fixed offset basis, fed
//! little-endian bytes (and `usize` widened to `u64`), so a key hashes to the
//! same value in every process and on every platform. Nothing in the crate
//! lets a result depend on hash-map iteration order; where it iterates, it
//! sorts or uses a `BTreeMap`. The fixed hasher removes the OS entropy draw,
//! which `scripts/entropy_guard.py` and clippy's `disallowed-types` keep out.

use core::hash::{BuildHasherDefault, Hasher};

const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// FNV-1a over the bytes written, from the fixed offset basis
#[derive(Clone, Copy, Debug)]
pub(crate) struct DetHasher(u64);

impl Default for DetHasher {
    fn default() -> Self {
        Self(FNV_OFFSET_BASIS)
    }
}

impl Hasher for DetHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ u64::from(b)).wrapping_mul(FNV_PRIME);
        }
    }

    fn write_u16(&mut self, n: u16) {
        self.write(&n.to_le_bytes());
    }

    fn write_u32(&mut self, n: u32) {
        self.write(&n.to_le_bytes());
    }

    fn write_u64(&mut self, n: u64) {
        self.write(&n.to_le_bytes());
    }

    fn write_u128(&mut self, n: u128) {
        self.write(&n.to_le_bytes());
    }

    fn write_usize(&mut self, n: usize) {
        // the same bytes on 32- and 64-bit targets
        self.write_u64(n as u64);
    }

    fn write_i16(&mut self, n: i16) {
        self.write(&n.to_le_bytes());
    }

    fn write_i32(&mut self, n: i32) {
        self.write(&n.to_le_bytes());
    }

    fn write_i64(&mut self, n: i64) {
        self.write(&n.to_le_bytes());
    }

    fn write_i128(&mut self, n: i128) {
        self.write(&n.to_le_bytes());
    }

    fn write_isize(&mut self, n: isize) {
        self.write_i64(n as i64);
    }
}

/// The build hasher of [`DetHashMap`]
pub(crate) type DetBuildHasher = BuildHasherDefault<DetHasher>;

/// A `HashMap` whose hasher has no random seed
// the one place the std map type is named: everything else uses this alias
#[allow(clippy::disallowed_types)]
pub(crate) type DetHashMap<K, V> = std::collections::HashMap<K, V, DetBuildHasher>;

#[cfg(test)]
mod tests {
    use super::{DetHashMap, DetHasher};
    use core::hash::{BuildHasher, Hash, Hasher};

    fn fnv(bytes: &[u8]) -> u64 {
        let mut h = DetHasher::default();
        h.write(bytes);
        h.finish()
    }

    /// FNV-1a 64 reference values (Fowler, Noll, Vo): the empty input is the
    /// offset basis, "a" and "foobar" are the published test vectors
    #[test]
    fn fnv1a_matches_the_published_vectors() {
        assert_eq!(fnv(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv(b"foobar"), 0x8594_4171_f739_67e8);
    }

    /// Integers are fed little-endian and `usize` as `u64`, so the hash of a
    /// key is the same on every platform
    #[test]
    fn integers_hash_as_their_little_endian_bytes() {
        let mut h = DetHasher::default();
        h.write_u32(0x0403_0201);
        assert_eq!(h.finish(), fnv(&[1, 2, 3, 4]));
        let mut a = DetHasher::default();
        a.write_usize(7);
        let mut b = DetHasher::default();
        b.write_u64(7);
        assert_eq!(a.finish(), b.finish());
    }

    /// Two maps built from the same keys hash every key the same (no seed)
    #[test]
    fn two_maps_hash_a_key_the_same() {
        let a: DetHashMap<(i32, i32, i32), u32> = DetHashMap::default();
        let b: DetHashMap<(i32, i32, i32), u32> = DetHashMap::default();
        let key = (3, -1, 7);
        let hash = |m: &DetHashMap<(i32, i32, i32), u32>| {
            let mut h = m.hasher().build_hasher();
            key.hash(&mut h);
            h.finish()
        };
        assert_eq!(hash(&a), hash(&b));
    }
}
