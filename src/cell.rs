/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

//! Cell indices.
//!
//! The [`Cell`] trait abstracts over the integer types (`u32`, `u64`, and
//! `u128`) used to store cell indices, and [`cell_index`] and
//! [`decimate_once`] draw cell indices from a generator.

use voracious_radix_sort::{RadixSort, ska_sort};

use crate::prng::Prng;
use crate::util::parallelism;

/// An integer type that can store cell indices.
pub trait Cell:
    Copy
    + Eq
    + Ord
    + Send
    + Sync
    + bytemuck::Pod
    + std::ops::ShlAssign<usize>
    + std::ops::ShrAssign<usize>
    + std::ops::BitAnd<Output = Self>
    + std::ops::BitXorAssign
    + std::ops::Sub<Output = Self>
    + std::ops::SubAssign
    + std::ops::AddAssign
{
    const ZERO: Self;

    fn from_u64(x: u64) -> Self;
    fn from_u128(x: u128) -> Self;

    /// Returns a mask of the lowest *b* bits (all ones if *b* is at least the
    /// width of the type).
    ///
    /// The counting level of the birthday-spacings tradeoff classifies spacings
    /// by their low bits, as spacings cluster near zero, so their top bits are
    /// unbalanced.
    fn low_bits_mask(b: usize) -> Self;

    /// Sorts `v` in place using a multithreaded radix sort.
    fn sort_mt(v: &mut [Self]);

    /// Sorts `v` in place using a single-threaded radix sort.
    fn sort_st(v: &mut [Self]);
}

macro_rules! impl_cell {
    ($t:ty, $bits:literal) => {
        impl Cell for $t {
            const ZERO: Self = 0;

            #[inline(always)]
            fn from_u64(x: u64) -> Self {
                x as Self
            }
            #[inline(always)]
            fn from_u128(x: u128) -> Self {
                x as Self
            }
            #[inline(always)]
            fn low_bits_mask(b: usize) -> Self {
                if b >= $bits {
                    !0
                } else {
                    (1 as $t).wrapping_shl(b as u32).wrapping_sub(1)
                }
            }
            fn sort_mt(v: &mut [Self]) {
                v.voracious_mt_sort(parallelism());
            }
            fn sort_st(v: &mut [Self]) {
                // In-place MSD radix sort: unlike voracious_sort, it does not
                // allocate a scratch buffer as large as v, which would double
                // the peak memory usage.
                ska_sort(v, 8);
            }
        }
    };
}

impl_cell!(u32, 32);
impl_cell!(u64, 64);
impl_cell!(u128, 128);

/// Returns a cell index drawn from `prng`, without decimation.
///
/// The index is the concatenation of *t* chunks of *u* bits, each taken from
/// the top of an output of [`next_u64()`] shifted left by *s*. Decimation is
/// handled by [`decimate_once`].
///
/// # Implementation Details
///
/// If `DIM` is nonzero, it is used as the dimension *t* in place of `t_rt`, so
/// that the loop can be unrolled. `FULL` can be true only if *u* = 64 and
/// *s* = 0, in which case the extraction is the identity.
///
/// [`next_u64()`]: crate::prng::Prng::next_u64
#[inline]
pub fn cell_index<T: Cell, const DIM: usize, const FULL: bool>(
    prng: &mut Prng,
    t_rt: usize,
    u: usize,
    s: usize,
) -> T {
    let combined_shift = 64 - u - s;
    let extract_mask: u64 = if u >= 64 { !0 } else { (1u64 << u) - 1 };

    #[inline(always)]
    fn extract<const FULL: bool>(raw: u64, combined_shift: usize, extract_mask: u64) -> u64 {
        if FULL {
            raw
        } else {
            (raw >> combined_shift) & extract_mask
        }
    }

    let t = if DIM == 0 { t_rt } else { DIM };

    let mut x = T::from_u64(extract::<FULL>(
        prng.next_u64(),
        combined_shift,
        extract_mask,
    ));
    for _ in 1..t {
        x <<= u;
        x ^= T::from_u64(extract::<FULL>(
            prng.next_u64(),
            combined_shift,
            extract_mask,
        ));
    }
    x
}

/// Draws a candidate tuple from `prng` and returns its decimated cell index, or
/// `None` if the tuple is rejected.
///
/// The tuple is accepted if the lowest *d* bits of all its elements are zero,
/// and in this case the index is the concatenation of the elements without
/// their lowest *d* bits. Exactly *t* outputs are drawn in both cases, so
/// sample *j* starts at offset *j* · *t* of the orbit, which makes jump-ahead
/// possible. See [`cell_index`] for the other parameters.
#[inline]
pub fn decimate_once<T: Cell, const DIM: usize, const FULL: bool>(
    prng: &mut Prng,
    t_rt: usize,
    u: usize,
    s: usize,
    d: usize,
) -> Option<T> {
    let combined_shift = 64 - u - s;
    let extract_mask: u64 = if u >= 64 { !0 } else { (1u64 << u) - 1 };

    #[inline(always)]
    fn extract<const FULL: bool>(raw: u64, combined_shift: usize, extract_mask: u64) -> u64 {
        if FULL {
            raw
        } else {
            (raw >> combined_shift) & extract_mask
        }
    }

    let t = if DIM == 0 { t_rt } else { DIM };
    let dec_mask: u64 = (1u64 << d) - 1;
    let width = u - d;

    let first = extract::<FULL>(prng.next_u64(), combined_shift, extract_mask);
    let mut rejected = first & dec_mask != 0;
    let mut x = T::from_u64(first >> d);
    for _ in 1..t {
        let raw = extract::<FULL>(prng.next_u64(), combined_shift, extract_mask);
        rejected |= raw & dec_mask != 0;
        x <<= width;
        x ^= T::from_u64(raw >> d);
    }
    if rejected { None } else { Some(x) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prng::Prng;

    // Tests that decimate_once draws exactly t outputs, whether it accepts or
    // not, and that accepted indices are smaller than (2ᵘ⁻ᵈ)ᵗ. The incr
    // generator is excluded because both outcomes must occur.
    #[cfg(not(feature = "incr"))]
    #[test]
    fn test_decimate_once_consumes_t_draws_and_is_dense() {
        let (t, u, s, d) = (3usize, 12usize, 0usize, 3usize);
        let bound = 1u128 << ((u - d) * t);
        let mut a = Prng::new(0xABCD_1234);
        let (mut accepted, mut rejected) = (0usize, 0usize);
        for _ in 0..50_000 {
            let b = a; // Copy: capture position before the attempt
            let got: Option<u128> = decimate_once::<u128, 3, false>(&mut a, 0, u, s, d);
            // b advanced by t outputs must be in the same state as a.
            let mut aref = b;
            for _ in 0..t {
                aref.next_u64();
            }
            let (mut a2, mut b2) = (a, aref);
            for _ in 0..4 {
                assert_eq!(
                    a2.next_u64(),
                    b2.next_u64(),
                    "decimate_once must consume exactly t draws"
                );
            }
            match got {
                Some(x) => {
                    assert!(x < bound, "dense index {x} not below {bound}");
                    accepted += 1;
                }
                None => rejected += 1,
            }
        }
        assert!(
            accepted > 0 && rejected > 0,
            "expected both accepts and rejects (acc={accepted}, rej={rejected})"
        );
    }

    // A nonzero DIM must give the same indices as the runtime t.
    #[test]
    fn test_dim_specialization_matches_runtime_fallback() {
        let (u, s) = (12usize, 0usize);
        let mut a = Prng::new(12_345);
        let mut b = a; // Prng: Copy → identical stream
        for _ in 0..2_000 {
            let spec: u64 = cell_index::<u64, 3, false>(&mut a, 0, u, s);
            let rt: u64 = cell_index::<u64, 0, false>(&mut b, 3, u, s);
            assert_eq!(spec, rt);
        }
    }

    // FULL must give the same indices as the shift-and-mask extraction.
    #[test]
    fn test_full_matches_shift_when_u_is_64() {
        let (u, s) = (64usize, 0usize);
        let mut a = Prng::new(999);
        let mut b = a;
        for _ in 0..2_000 {
            let full: u64 = cell_index::<u64, 1, true>(&mut a, 1, u, s);
            let shift: u64 = cell_index::<u64, 1, false>(&mut b, 1, u, s);
            assert_eq!(full, shift);
        }
    }
}
