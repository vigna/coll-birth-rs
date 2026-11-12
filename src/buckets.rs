/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

//! Bucketed processing of large unsorted buffers.
//!
//! The tests do not need the points of a pass sorted as a whole: collisions
//! require only that equal points be adjacent, and spacings require that each
//! range of values be sorted, plus the extremes of each range. A buffer is thus
//! partitioned in place into 2ᵏ *buckets* by a *k*-bit digit of the points
//! (see [`partition`]), and then each bucket is gathered into a small
//! thread-local buffer, sorted, and processed (see
//! [`Buckets::for_each_sorted`]). Both phases are fully parallel, whereas
//! the global data movement of a parallel in-place radix sort of a large buffer
//! is, in practice, mostly sequential: on 192 cores, counting the collisions
//! of 1.36 · 10¹⁰ random `u128` takes 7.5 s instead of 15.9 s.

use crate::cell::Cell;
use crate::common::bin_overflow;
use crate::util::{Stopwatch, parallelism};
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

/// The base-2 logarithm of the target number of points in a bucket.
///
/// Buckets of about 2¹⁹ points are sorted mostly in cache, and with the
/// resulting number of buckets the partition is still fast.
const LOG_BUCKET_TARGET: usize = 19;

/// The maximum number *k* of digit bits.
const MAX_DIGIT_BITS: usize = 16;

/// The number of points below which buckets are not used.
const MIN_POINTS: usize = 1 << 24;

/// The number of threads below which buckets are not used.
///
/// Buckets pay off when the multithreaded radix sort cannot keep the cores
/// busy: on 192 cores they halve the time of the sort, but on 10 cores the sort
/// is faster (1.3 s instead of 2.2 s for 5 · 10⁸ random `u128`).
const MIN_THREADS: usize = 64;

/// The maximum number of digit bits handled by a single partition pass.
///
/// With few buckets the write positions stay in cache, and the partition runs
/// at memory speed; with many buckets each swap incurs a cache (and TLB) miss.
const PASS_BITS: usize = 7;

/// Returns the number *k* of digit bits to use for `n` points, given that at
/// most `avail` bits are available, or zero if buckets should not be used.
pub(crate) fn digit_bits(n: usize, avail: usize) -> usize {
    if n < MIN_POINTS || parallelism() < MIN_THREADS {
        return 0;
    }
    let k = (n >> LOG_BUCKET_TARGET).max(1).ilog2() as usize;
    let k = k.min(MAX_DIGIT_BITS).min(avail);
    // Very few buckets would make the thread-local buffers too large.
    if k < 8 { 0 } else { k }
}

/// Returns the digit of `x`, that is, its bits from `shift` (included) to
/// `shift` + *k* (excluded), where `mask` is 2ᵏ − 1.
#[inline(always)]
fn digit<T: Cell>(x: T, shift: usize, mask: usize) -> usize {
    let mut y = x;
    y >>= shift;
    y.low_u64() as usize & mask
}

/// A partition of a buffer into buckets, computed by [`partition`].
///
/// The buffer is divided into chunks of `chunk_len` points (the last one might
/// be shorter), and each chunk contains, in order, its points of each bucket.
pub(crate) struct Buckets {
    /// The number of buckets.
    num_buckets: usize,
    /// The length of a chunk.
    chunk_len: usize,
    /// For each chunk, the start of each bucket within the chunk, plus the
    /// length of the chunk.
    starts: Box<[Box<[usize]>]>,
}

/// Partitions `c` in place by the digit made of its bits from `shift`
/// (included) to `shift` + *k* (excluded), and returns the start of each of the
/// 2ᵏ buckets, plus the length of `c`.
///
/// # Implementation Notes
///
/// The partition is an American-flag sort (in-place cycle leader) on the top
/// [`PASS_BITS`] bits of the digit, followed, if there are more bits, by a
/// recursive partition of each resulting bucket on the remaining bits.
fn partition_chunk<T: Cell>(c: &mut [T], shift: usize, k: usize) -> Vec<usize> {
    let bits = k.min(PASS_BITS);
    let num_buckets = 1usize << bits;
    let mask = num_buckets - 1;
    let pass_shift = shift + k - bits;
    let mut start = vec![0usize; num_buckets + 1];
    for &x in c.iter() {
        start[digit(x, pass_shift, mask) + 1] += 1;
    }
    for g in 0..num_buckets {
        start[g + 1] += start[g];
    }
    let mut next = start.clone();
    for g in 0..num_buckets {
        while next[g] < start[g + 1] {
            let d = digit(c[next[g]], pass_shift, mask);
            if d == g {
                next[g] += 1;
            } else {
                c.swap(next[g], next[d]);
                next[d] += 1;
            }
        }
    }
    if bits == k {
        return start;
    }
    let mut starts = Vec::with_capacity((1 << k) + 1);
    for g in 0..num_buckets {
        let sub = partition_chunk(&mut c[start[g]..start[g + 1]], shift, k - bits);
        starts.extend(sub[..sub.len() - 1].iter().map(|&s| start[g] + s));
    }
    starts.push(c.len());
    starts
}

/// Partitions `v` in place into 2ᵏ buckets by the digit of its points made of
/// the bits from `shift` (included) to `shift` + *k* (excluded).
///
/// `v` is divided into one chunk per thread, and each chunk is partitioned
/// independently (see [`partition_chunk`]), so no data moves between chunks.
pub(crate) fn partition<T: Cell>(v: &mut [T], shift: usize, k: usize) -> Buckets {
    let chunk_len = v.len().div_ceil(parallelism()).max(1);
    let starts: Box<[Box<[usize]>]> = v
        .par_chunks_mut(chunk_len)
        .map(|c| partition_chunk(c, shift, k).into_boxed_slice())
        .collect();
    Buckets {
        num_buckets: 1 << k,
        chunk_len,
        starts,
    }
}

impl Buckets {
    /// Returns the number of points of bucket `g`.
    fn bucket_len(&self, g: usize) -> usize {
        self.starts.iter().map(|s| s[g + 1] - s[g]).sum()
    }

    /// Returns the number of points of the largest bucket.
    pub(crate) fn max_bucket_len(&self) -> usize {
        (0..self.num_buckets)
            .into_par_iter()
            .map(|g| self.bucket_len(g))
            .max()
            .unwrap_or(0)
    }

    /// Returns whether the buckets are balanced enough that each thread can
    /// gather a bucket in a buffer of its own.
    ///
    /// Buckets are unbalanced only if the digits of the points are very far
    /// from uniform; the buffers of all threads then take at most about 4% of
    /// the memory of the points.
    pub(crate) fn balanced(&self, n: usize) -> bool {
        self.max_bucket_len() <= 4 * n.div_ceil(self.num_buckets).max(1 << 12)
    }

    /// Calls `f(g, s)` in parallel for each bucket `g`, where `s` contains the
    /// points of the bucket sorted, and returns the results in bucket order.
    ///
    /// `s` is a thread-local buffer, which `f` can use as scratch space. `v`
    /// must be the buffer partitioned by [`partition`].
    pub(crate) fn for_each_sorted<T: Cell, R: Send>(
        &self,
        v: &[T],
        f: impl Fn(usize, &mut [T]) -> R + Sync,
    ) -> Vec<R> {
        (0..self.num_buckets)
            .into_par_iter()
            .map_init(Vec::<T>::new, |scratch, g| {
                scratch.clear();
                for (c, s) in self.starts.iter().enumerate() {
                    let base = c * self.chunk_len;
                    scratch.extend_from_slice(&v[base + s[g]..base + s[g + 1]]);
                }
                scratch.sort_unstable();
                f(g, scratch)
            })
            .collect()
    }
}

/// Returns the number of collisions (points minus distinct points) of `v`,
/// whose points are unsorted, possibly permuting `v`.
///
/// Points are bucketed by the top bits of their bits from `shift` (included)
/// to `shift` + `avail` (excluded), which should be as close to uniform as
/// possible; if there are too few points or threads, too few bits, or the
/// buckets are unbalanced, `v` is sorted with [`Cell::sort_mt`] instead.
///
/// If `sw` is not [`None`], the phases are reported on standard error as
/// `partition...[`*time*`] sort+count...` with buckets and as
/// `sort...[`*time*`] count...` otherwise, each preceded by the time of the
/// previous phase measured by `sw`, as in the other progress lines; the time of
/// the last phase is left to the caller.
pub(crate) fn count_collisions<T: Cell>(
    v: &mut [T],
    shift: usize,
    avail: usize,
    sw: Option<&mut Stopwatch>,
) -> usize {
    count_collisions_with(v, shift, avail, digit_bits(v.len(), avail), sw)
}

/// Like [`count_collisions`], but using `k` digit bits (zero for no buckets).
fn count_collisions_with<T: Cell>(
    v: &mut [T],
    shift: usize,
    avail: usize,
    k: usize,
    mut sw: Option<&mut Stopwatch>,
) -> usize {
    let mut report = |label: &str, first: bool| {
        if let Some(sw) = sw.as_deref_mut() {
            let lap = sw.lap();
            if first {
                eprint!("{label}...");
            } else {
                eprint!("[{lap:.3}s] {label}...");
            }
        }
    };
    if k > 0 {
        report("partition", true);
        let buckets = partition(v, shift + avail - k, k);
        if buckets.balanced(v.len()) {
            report("sort+count", false);
            return buckets
                .for_each_sorted(v, |_, s| s.windows(2).filter(|w| w[0] == w[1]).count())
                .into_iter()
                .sum();
        }
        report("sort", false);
    } else {
        report("sort", true);
    }
    T::sort_mt(v);
    report("count", false);
    crate::common::count_adjacent_equals(v)
}

/// A slice that threads can append to concurrently, in no particular order.
pub(crate) struct Sink<'a, T> {
    ptr: *mut T,
    cap: usize,
    len: AtomicUsize,
    _slice: std::marker::PhantomData<&'a mut [T]>,
}

impl<'a, T: Cell> Sink<'a, T> {
    /// Creates an empty sink storing at most `slice.len()` elements in `slice`.
    pub(crate) fn new(slice: &'a mut [T]) -> Self {
        Self {
            ptr: slice.as_mut_ptr(),
            cap: slice.len(),
            len: AtomicUsize::new(0),
            _slice: std::marker::PhantomData,
        }
    }

    /// Appends the elements of `s`, aborting if they do not fit.
    pub(crate) fn push_slice(&self, s: &[T], what: &str) {
        let start = self.len.fetch_add(s.len(), Ordering::Relaxed);
        if start + s.len() > self.cap {
            bin_overflow(what);
        }
        // SAFETY: ptr comes from a slice of cap elements that the sink borrows
        // mutably for its whole lifetime, so no one else accesses it; the
        // ranges reserved by fetch_add are pairwise disjoint and, as checked
        // above, within the slice, so no two threads write the same element.
        unsafe {
            std::ptr::copy_nonoverlapping(s.as_ptr(), self.ptr.add(start), s.len());
        }
    }

    /// Returns the number of elements appended.
    pub(crate) fn len(self) -> usize {
        self.len.into_inner()
    }
}

// SAFETY: see push_slice; concurrent appends write disjoint ranges.
unsafe impl<T: Send> Sync for Sink<'_, T> {}
// SAFETY: the sink owns its mutable borrow of the slice.
unsafe impl<T: Send> Send for Sink<'_, T> {}

#[cfg(test)]
mod tests {
    use super::*;

    fn xorshift(x: &mut u64) -> u64 {
        *x ^= *x << 13;
        *x ^= *x >> 7;
        *x ^= *x << 17;
        *x
    }

    // Bucketed collision counts must match the sorted ones, for uniform and
    // for very unbalanced digits (which trigger the fallback).
    #[test]
    fn test_count_collisions_matches_sort() {
        let mut x = 0x0123_4567_89ab_cdef_u64;
        // (points, value bits, shift, available bits, digit bits): with k = 0
        // and with a constant digit (the last case) the fallback is used.
        for &(n, bits, shift, avail, k) in &[
            (1usize << 20, 30u32, 0usize, 30usize, 10usize),
            (1 << 20, 24, 0, 24, 12),
            (1 << 20, 40, 10, 30, 8),
            (1 << 20, 64, 0, 64, 10),
            (1 << 20, 64, 0, 64, 0),
            (1 << 20, 20, 30, 10, 10),
        ] {
            let mask = if bits >= 64 { !0 } else { (1u64 << bits) - 1 };
            let mut v: Vec<u64> = (0..n).map(|_| xorshift(&mut x) & mask).collect();
            let mut w = v.clone();
            w.sort_unstable();
            let expected = crate::common::count_adjacent_equals(&w);
            assert_eq!(
                count_collisions_with(&mut v, shift, avail, k, None),
                expected,
                "n={n} bits={bits} k={k}"
            );
        }
    }

    // Each bucket must contain exactly the points with its digit, and the
    // buckets must be presented sorted.
    #[test]
    fn test_partition_and_sorted_buckets() {
        let mut x = 0xfeed_beef_u64;
        let n = 1 << 20;
        let mut v: Vec<u128> = (0..n).map(|_| u128::from(xorshift(&mut x) >> 20)).collect();
        let mut sorted = v.clone();
        sorted.sort_unstable();
        let (shift, k) = (34, 10);
        let b = partition(&mut v, shift, k);
        let parts = b.for_each_sorted(&v, |g, s| {
            assert!(s.windows(2).all(|w| w[0] <= w[1]));
            assert!(s.iter().all(|&y| digit(y, shift, (1 << k) - 1) == g));
            s.to_vec()
        });
        let concatenated: Vec<u128> = parts.into_iter().flatten().collect();
        assert_eq!(concatenated, sorted);
    }

    // Concurrent appends must store all the elements.
    #[test]
    fn test_sink() {
        let mut buf = vec![0u64; 10_000];
        let sink = Sink::new(&mut buf);
        (0..100u64).into_par_iter().for_each(|i| {
            let s: Vec<u64> = (0..i).map(|j| i * 1000 + j).collect();
            sink.push_slice(&s, "test sink");
        });
        let len = sink.len();
        assert_eq!(len, (0..100).sum::<usize>());
        let mut got = buf[..len].to_vec();
        got.sort_unstable();
        let mut expected: Vec<u64> = (0..100u64)
            .flat_map(|i| (0..i).map(move |j| i * 1000 + j))
            .collect();
        expected.sort_unstable();
        assert_eq!(got, expected);
    }
}
