/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

//! Infrastructure shared by the collision and birthday-spacings tests.

use std::mem::size_of;

use mmap_rs::{MmapFlags, MmapMut, MmapOptions};
use num::BigUint;
use num::traits::ToPrimitive;
use rayon::prelude::*;

use crate::birthday::{run_birthday, run_birthday_tradeoff};
use crate::cell::{Cell, cell_index, decimate_once};
use crate::cli::Args;
use crate::collision::{run_collision, run_collision_decimate, run_collision_tradeoff};
use crate::prng::Prng;
use crate::stats::{
    MAX_DENSITY, Null, NullKind, VARIANCE_MAXIMIZING_DENSITY, format_p_value, p_value,
};
use crate::util::{Stopwatch, parallelism, superscript};

/// Allocates a memory-mapped buffer of `n` elements of type `T`, prefaulting it
/// as transparent huge pages.
///
/// # Implementation Details
///
/// [`MmapFlags::POPULATE`] would prefault the buffer as base (4 KiB) pages, so
/// we use [`MmapFlags::TRANSPARENT_HUGE_PAGES`] and prefault the buffer by
/// touching one byte every 2 MiB. If transparent huge pages are disabled, the
/// buffer is still prefaulted, as base pages.
///
/// [`MmapFlags::POPULATE`]: mmap_rs::MmapFlags::POPULATE
/// [`MmapFlags::TRANSPARENT_HUGE_PAGES`]: mmap_rs::MmapFlags::TRANSPARENT_HUGE_PAGES
pub(crate) fn alloc_mmap<T>(n: usize) -> MmapMut {
    // A failed allocation is not a bug, so we exit with an explanation rather
    // than panicking.
    fn alloc_error(n: usize, detail: &dyn std::fmt::Display) -> ! {
        eprintln!(
            "\ncannot allocate a buffer of {n} memory locations: {detail}; \
             reduce m or use more tradeoff bits (-b)"
        );
        std::process::exit(1);
    }
    let bytes_len = n
        .checked_mul(size_of::<T>())
        .unwrap_or_else(|| alloc_error(n, &"the size in bytes overflows usize"));
    let mut mapped = MmapOptions::new(bytes_len)
        .and_then(|options| {
            options
                .with_flags(MmapFlags::TRANSPARENT_HUGE_PAGES)
                .map_mut()
        })
        .unwrap_or_else(|e| alloc_error(n, &e));
    const HUGE_PAGE: usize = 2 * 1024 * 1024;
    let bytes: &mut [u8] = &mut mapped;
    bytes
        .par_chunks_mut(HUGE_PAGE)
        .for_each(|chunk| chunk[0] = 0);
    mapped
}

/// Returns the size of a buffer for the points of a bin.
///
/// A pass keeps the samples whose `partition_bits` selection bits (*t* · *d*
/// decimation bits and *b* tradeoff bits) have a given value, that is, one of
/// the *n* = 2^`partition_bits` bins of a balls-into-bins experiment with
/// `points` balls. If `partition_bits` is zero, the result is `points`.
///
/// Since the load of a bin is a sum of independent indicators, by Bernstein's
/// inequality and a union bound the probability that some bin receives at
/// least *m* + λ balls, where *m* = `points` / *n*, is at most *n* ·
/// exp(−λ² / (2(*m* + λ / 3))). This bound is 10⁻¹⁰⁰⁰ for λ = *L* / 3 +
/// √(*L*² / 9 + 2*mL*), where *L* = ln *n* + 1000 ln 10.
///
/// The bound is essentially tight: by Theorem 1 of [Raab and Steger], in the
/// heavily loaded case the maximum load is *m* + √(2*m* ln *n*) (1 − *o*(1)).
///
/// [Raab and Steger]: https://doi.org/10.1007/3-540-49543-6_13
pub fn buffer_size(points: usize, partition_bits: usize) -> usize {
    if partition_bits == 0 {
        return points;
    }
    let mean = points as f64 / 2.0f64.powi(partition_bits as i32);
    let ln_n = (partition_bits as f64) * std::f64::consts::LN_2;
    let l = ln_n + 1000.0 * std::f64::consts::LN_10;
    // Exact inversion of the Bernstein exponent: λ²/(2(mean + λ/3)) = L.
    let dev = l / 3.0 + (l * l / 9.0 + 2.0 * mean * l).sqrt();
    ((mean + dev).ceil() as usize).min(points)
}

/// Exits the process when a bin (e.g., a tradeoff pass or a spacing class)
/// overflows its buffer.
///
/// Since buffers are sized by [`buffer_size`], for a uniform generator this
/// happens with probability less than 10⁻¹⁰⁰⁰, so the generator is grossly
/// non-uniform. We exit rather than panicking, also in worker threads, so that
/// the message is not buried under a backtrace.
pub(crate) fn bin_overflow(what: &str) -> ! {
    eprintln!(
        "\n{what} overflowed its buffer: a bin received more elements than its \
         balls-into-bins headroom, which happens with probability below 10⁻¹⁰⁰⁰ \
         for a uniform generator: this is overwhelming evidence that the generator under \
         test is grossly non-uniform. Rerun in plain mode (no -b/-d) for an exact \
         p-value."
    );
    std::process::exit(1);
}

/// Returns the number of samples scanned by a pass, that is, `points` · 2ᵗᵈ.
///
/// # Panics
///
/// Panics if the result does not fit in a `usize`. Note that [`checked_shl`]
/// checks only the shift amount, so the product must be checked separately.
///
/// [`checked_shl`]: usize::checked_shl
pub(crate) fn scan_samples(points: usize, t: usize, d: usize) -> usize {
    1usize
        .checked_shl((t * d) as u32)
        .and_then(|factor| points.checked_mul(factor))
        .expect("points · 2ᵗᵈ overflows usize")
}

/// Returns the header description of the bits of each output used by the test.
pub(crate) fn bits_read_desc(s: usize) -> String {
    if s == 0 {
        "full 64-bit output".to_string()
    } else {
        format!("lowest {} bits", 64 - s)
    }
}

/// Returns the header description of decimation.
pub(crate) fn decimation_desc(d: usize, t: usize) -> String {
    format!(
        "decimating {} bits per dimension (~2{} candidate samples per kept sample)",
        d,
        superscript(d * t)
    )
}

/// Returns the header description of parallel generation.
///
/// `num_cpus` must be the number of threads actually used, which might be
/// smaller than the size of the thread pool (see [`OrbitPartition::new`]).
pub(crate) fn generation_desc(num_cpus: usize, split_desc: &str) -> String {
    format!(
        "using {} parallel generator{} ({})",
        num_cpus,
        if num_cpus == 1 { "" } else { "s" },
        split_desc
    )
}

/// Returns the header note " (+*x*%)" on the headroom of buffers.
///
/// `capacity` is the size of the buffers, and `nominal` the load they would
/// carry if samples were split evenly among the 2^`partition_bits` bins. The
/// note is empty if `partition_bits` is zero, as there is no headroom.
///
/// `nominal` is an `f64` because 2^`partition_bits` might not fit in a `usize`
/// (*t* · *d* might exceed 63).
pub(crate) fn headroom_desc(capacity: usize, nominal: f64, partition_bits: usize) -> String {
    if partition_bits == 0 {
        return String::new();
    }
    format!(" (+{:.2}%)", (capacity as f64 / nominal - 1.0) * 100.0)
}

/// Returns the header suffix listing the given mode descriptions, each
/// preceded by a comma.
pub(crate) fn join_mode_parts(parts: &[String]) -> String {
    if parts.is_empty() {
        String::new()
    } else {
        format!(", {}", parts.join(", "))
    }
}

/// Returns the header note on the number of cells after decimation (empty if
/// *d* = 0).
pub(crate) fn effective_cells_suffix(d: usize, u: usize, t: usize) -> String {
    if d > 0 {
        format!(
            " (effective cells after decimation: 2{})",
            superscript((u - d) * t)
        )
    } else {
        String::new()
    }
}

/// The parameters of a test grid.
pub struct GridParams<'a> {
    /// The base-2 logarithm *u* of the number of subdivisions per dimension.
    pub u: usize,
    /// The number *t* of dimensions.
    pub t: usize,
    /// The left shift *s* applied to each output before extracting bits.
    pub s: usize,
    /// The number *d* of decimation bits (zero for no decimation).
    pub d: usize,
    /// The number of cells, used by the wrap-around spacing.
    pub cells: &'a BigUint,
}

impl GridParams<'_> {
    /// Returns a cell index drawn from `prng` (see [`cell_index`]).
    #[inline]
    pub fn draw<T: Cell, const DIM: usize, const FULL: bool>(&self, prng: &mut Prng) -> T {
        cell_index::<T, DIM, FULL>(prng, self.t, self.u, self.s)
    }

    /// Draws a candidate tuple from `prng` and returns its decimated cell
    /// index, or `None` if the tuple is rejected (see [`decimate_once`]).
    #[inline]
    pub fn draw_decimate_once<T: Cell, const DIM: usize, const FULL: bool>(
        &self,
        prng: &mut Prng,
    ) -> Option<T> {
        decimate_once::<T, DIM, FULL>(prng, self.t, self.u, self.s, self.d)
    }
}

/// Returns the number of pairs of adjacent equal elements of a sorted slice,
/// that is, the number of elements minus the number of distinct elements.
///
/// # Implementation Details
///
/// The slice is split into contiguous chunks, and pairs straddling chunk
/// boundaries are counted separately: on very large slices, this is much
/// faster than `par_windows`.
#[inline]
pub(crate) fn count_adjacent_equals<T: Cell>(v: &[T]) -> usize {
    if v.len() < 2 {
        return 0;
    }
    let chunk_size = (v.len() / (parallelism() * 10)).max(1024);
    // Pairs that lie wholly within a chunk.
    let within: usize = v
        .par_chunks(chunk_size)
        .map(|c| {
            let mut count = 0usize;
            for w in c.windows(2) {
                if w[0] == w[1] {
                    count += 1;
                }
            }
            count
        })
        .sum();
    // Pairs straddling a chunk boundary: this chunk's first vs. the previous tail.
    let borders: usize = v
        .par_chunks(chunk_size)
        .enumerate()
        .skip(1)
        .filter(|(i, c)| v[i * chunk_size - 1] == c[0])
        .count();
    within + borders
}

/// Merges the sorted slice `src` into the sorted prefix `buf[..prefix_len]`,
/// leaving the result in `buf[..prefix_len + src.len()]`.
///
/// # Implementation Details
///
/// The merge proceeds right to left, so it never overwrites an element of the
/// prefix that has not been read yet, and no additional space is needed. When
/// `src` is exhausted, the remaining elements of the prefix are in place.
pub(crate) fn merge_into<T: Cell>(buf: &mut [T], prefix_len: usize, src: &[T]) {
    let mut i = prefix_len;
    let mut j = src.len();
    let mut w = i + j;
    while i > 0 && j > 0 {
        w -= 1;
        if buf[i - 1] >= src[j - 1] {
            i -= 1;
            buf[w] = buf[i];
        } else {
            j -= 1;
            buf[w] = src[j];
        }
    }
    while j > 0 {
        w -= 1;
        j -= 1;
        buf[w] = src[j];
    }
}

/// A partition of the orbit of the generator among threads.
///
/// A scan of `scan_total` samples is split into `num_cpus` contiguous ranges.
/// The start of each range is reached by jump-ahead (`try_skip`) if the
/// generator supports it, or by a sequential pre-scan
/// ([`prescan_checkpoints`]) otherwise. In both cases, the result of a parallel
/// run is the same as that of a sequential run.
pub(crate) struct OrbitPartition {
    pub(crate) num_cpus: usize,
    pub(crate) scan_total: usize,
    pub(crate) base_chunk: usize,
    pub(crate) rem: usize,
    t: usize,
    skip_capable: bool,
    seed: u64,
    /// The start of the next repetition, used only by the pre-scan, which
    /// cannot reach an absolute offset.
    prescan_start: Prng,
}

impl OrbitPartition {
    pub(crate) fn new(seed: u64, num_cpus: usize, scan_total: usize, t: usize) -> Self {
        // try_skip succeeds for every offset or for none (see the prng module).
        let skip_capable = {
            let mut probe = Prng::new(seed);
            probe.try_skip(0).is_ok()
        };
        // There cannot be more threads than samples, as an empty chunk would
        // lead to a zero-size allocation, which alloc_mmap rejects.
        let num_cpus = num_cpus.min(scan_total).max(1);
        Self {
            num_cpus,
            scan_total,
            base_chunk: scan_total / num_cpus,
            rem: scan_total % num_cpus,
            t,
            skip_capable,
            seed,
            prescan_start: Prng::new(seed),
        }
    }

    /// Returns the first sample of thread `i`.
    pub(crate) fn start_sample(&self, i: usize) -> usize {
        i * self.base_chunk + i.min(self.rem)
    }

    /// Returns the header description of how the orbit is split.
    pub(crate) fn split_desc(&self) -> &'static str {
        if self.skip_capable {
            "jump-ahead"
        } else {
            "pre-scan"
        }
    }

    /// Returns the generator states at which threads start.
    ///
    /// Thread `i` starts at sample `base + boundaries[i]`, in a window of
    /// `scan_len` samples. With jump-ahead, the states are computed from the
    /// seed; otherwise, the window is pre-scanned starting from the end of the
    /// previous pre-scan. If `prescan_label` is not `None`, the pre-scan is
    /// announced and timed.
    pub(crate) fn snapshots(
        &mut self,
        base: usize,
        scan_len: usize,
        boundaries: &[usize],
        prescan_label: Option<&str>,
    ) -> Box<[Prng]> {
        if self.skip_capable {
            boundaries
                .iter()
                .map(|&b| {
                    let off = ((base + b) as u64)
                        .checked_mul(self.t as u64)
                        .expect("orbit offset overflows u64");
                    let mut p = Prng::new(self.seed);
                    p.try_skip(off)
                        .expect("try_skip must succeed for every offset or for none");
                    p
                })
                .collect()
        } else {
            let mut sw = prescan_label.map(|lbl| {
                eprint!("{lbl}");
                Stopwatch::new()
            });
            let (snaps, end) =
                prescan_checkpoints(self.prescan_start, self.t, scan_len, boundaries);
            self.prescan_start = end;
            if let Some(sw) = sw.as_mut() {
                eprintln!("[{:.3}s]", sw.lap());
            }
            snaps
        }
    }

    /// Returns the generator states at which threads start in repetition `rep`
    /// (one-based), in which each thread scans a contiguous chunk.
    pub(crate) fn rep_snapshots(&mut self, rep: usize) -> Box<[Prng]> {
        let boundaries: Box<[usize]> = (0..self.num_cpus).map(|i| self.start_sample(i)).collect();
        self.snapshots(
            (rep - 1) * self.scan_total,
            self.scan_total,
            &boundaries,
            Some("Pre-scan..."),
        )
    }
}

/// Scans sequentially `total_cells` samples of *t* outputs (also under
/// decimation, see [`decimate_once`]) starting from `start`, and returns the
/// generator states at the samples in `boundaries`, together with the final
/// state.
///
/// `boundaries` must be sorted, and its elements must be smaller than
/// `total_cells`. The state at `boundaries[i]` is the same that
/// `try_skip(boundaries[i] * t)` would yield, so this function makes it
/// possible to split the orbit among threads for generators that cannot jump
/// ahead. The final state is the start of the next repetition.
pub(crate) fn prescan_checkpoints(
    start: Prng,
    t: usize,
    total_cells: usize,
    boundaries: &[usize],
) -> (Box<[Prng]>, Prng) {
    let mut p = start;
    let mut snaps = Vec::with_capacity(boundaries.len());
    let mut next = 0usize;
    for cell in 0..total_cells {
        while next < boundaries.len() && boundaries[next] == cell {
            snaps.push(p);
            next += 1;
        }
        for _ in 0..t {
            p.next_u64();
        }
    }
    debug_assert_eq!(
        next,
        boundaries.len(),
        "every boundary must be < total_cells"
    );
    (snaps.into_boxed_slice(), p)
}

/// Generates the points of a thread in a pass, scanning `stream_len` samples
/// starting from state `snapshot`.
///
/// Returns the number of points written to `buf` and the final state. In
/// tradeoff mode, all passes start from the same state.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gen_pass_dispatch<T: Cell>(
    snapshot: Prng,
    params: &GridParams,
    buf: &mut [T],
    stream_len: usize,
    pass: u64,
    tradeoff_b: usize,
    decimating: bool,
    full: bool,
) -> (usize, Prng) {
    macro_rules! go {
        ($dim:literal) => {{
            if tradeoff_b > 0 {
                // FULL (u = 64 and s = 0) makes the extraction the identity.
                match (decimating, full) {
                    (true, true) => gen_pass_tradeoff::<T, $dim, true, true>(
                        snapshot, params, buf, stream_len, pass, tradeoff_b,
                    ),
                    (true, false) => gen_pass_tradeoff::<T, $dim, true, false>(
                        snapshot, params, buf, stream_len, pass, tradeoff_b,
                    ),
                    (false, true) => gen_pass_tradeoff::<T, $dim, false, true>(
                        snapshot, params, buf, stream_len, pass, tradeoff_b,
                    ),
                    (false, false) => gen_pass_tradeoff::<T, $dim, false, false>(
                        snapshot, params, buf, stream_len, pass, tradeoff_b,
                    ),
                }
            } else {
                // No tradeoff: a single pass.
                let mut prng = snapshot;
                let used = if decimating {
                    if full {
                        gen_plain::<T, $dim, true, true>(&mut prng, params, buf, stream_len)
                    } else {
                        gen_plain::<T, $dim, true, false>(&mut prng, params, buf, stream_len)
                    }
                } else if full {
                    gen_plain::<T, $dim, false, true>(&mut prng, params, buf, stream_len)
                } else {
                    gen_plain::<T, $dim, false, false>(&mut prng, params, buf, stream_len)
                };
                (used, prng)
            }
        }};
    }
    match params.t {
        1 => go!(1),
        2 => go!(2),
        3 => go!(3),
        4 => go!(4),
        5 => go!(5),
        6 => go!(6),
        7 => go!(7),
        8 => go!(8),
        _ => go!(0),
    }
}

/// Closes in place the gaps left by partially filled thread regions, and
/// returns the number of points.
///
/// `buf` is partitioned into regions of size `caps[i]`, and thread `i` wrote
/// `used[i]` ≤ `caps[i]` points at the start of its region. This function
/// moves the points so that they are contiguous at the start of `buf`.
///
/// # Implementation Details
///
/// Since the prefix sums of `used` are bounded by those of `caps`, every block
/// moves leftward. Blocks are moved left to right, and the destination of
/// block `i` ends before the source of block `i` + 1, so no move overwrites a
/// source that has not been moved yet; `copy_within` handles overlaps within a
/// block. If `used[i]` = `caps[i]` for all `i`, nothing is moved.
fn compact_blocks<T: Cell>(buf: &mut [T], caps: &[usize], used: &[usize]) -> usize {
    debug_assert_eq!(caps.len(), used.len());
    let mut src_base = 0usize;
    let mut dst = 0usize;
    for (&cap, &len) in caps.iter().zip(used) {
        debug_assert!(len <= cap);
        if dst != src_base {
            buf.copy_within(src_base..src_base + len, dst);
        }
        dst += len;
        src_base += cap;
    }
    dst
}

/// Generates in parallel the points of a pass into a contiguous buffer, and
/// returns the number of points.
///
/// Thread `i` scans `chunk(i)` samples starting from `snapshots[i]`, and writes
/// the points it keeps into a region of `buf` of size `caps[i]` (see
/// [`gen_pass_dispatch`]). The regions are then compacted (see
/// [`compact_blocks`]), so the points are stored, unsorted, at the start of
/// `buf`, whose length must be at least Σ `caps`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gen_unit_contiguous<T: Cell>(
    buf: &mut [T],
    caps: &[usize],
    snapshots: &[Prng],
    params: &GridParams,
    chunk: impl Fn(usize) -> usize + Sync,
    pass: u64,
    tradeoff_b: usize,
    decimating: bool,
    full: bool,
) -> usize {
    let num_cpus = caps.len();
    debug_assert_eq!(snapshots.len(), num_cpus);

    // Phase 1: fill the regions in parallel, one task per region.
    let regions: Vec<&mut [T]> = {
        let mut regions = Vec::with_capacity(num_cpus);
        let mut rest = &mut buf[..];
        for &cap in caps {
            let (region, tail) = rest.split_at_mut(cap);
            regions.push(region);
            rest = tail;
        }
        regions
    };
    let used: Vec<usize> = regions
        .into_par_iter()
        .enumerate()
        .map(|(i, region)| {
            gen_pass_dispatch::<T>(
                snapshots[i],
                params,
                region,
                chunk(i),
                pass,
                tradeoff_b,
                decimating,
                full,
            )
            .0
        })
        .collect();

    // Phase 2: close the gaps so the kept points are contiguous.
    compact_blocks(buf, caps, &used)
}

/// Generates the points of a pass without tradeoff, scanning `scan_len`
/// samples, and returns the number of points written to `buf`.
fn gen_plain<T: Cell, const DIM: usize, const DECIMATE: bool, const FULL: bool>(
    prng: &mut Prng,
    params: &GridParams,
    buf: &mut [T],
    scan_len: usize,
) -> usize {
    if DECIMATE {
        // Keep the accepted samples.
        let mut len = 0usize;
        for _ in 0..scan_len {
            if let Some(x) = params.draw_decimate_once::<T, DIM, FULL>(prng) {
                *buf.get_mut(len)
                    .unwrap_or_else(|| bin_overflow("a parallel decimation chunk")) = x;
                len += 1;
            }
        }
        len
    } else {
        for x in buf[..scan_len].iter_mut() {
            *x = params.draw::<T, DIM, FULL>(prng);
        }
        scan_len
    }
}

/// Generates the points of a tradeoff pass, scanning `stream_len` samples
/// starting from state `snapshot` and keeping the points whose top *b* bits
/// are equal to `pass`, as in [`run_collision_tradeoff`].
///
/// Returns the number of points written to `buf` and the final state.
///
/// [`run_collision_tradeoff`]: crate::collision::run_collision_tradeoff
fn gen_pass_tradeoff<T: Cell, const DIM: usize, const DECIMATE: bool, const FULL: bool>(
    snapshot: Prng,
    params: &GridParams,
    buf: &mut [T],
    stream_len: usize,
    pass: u64,
    b: usize,
) -> (usize, Prng) {
    let t = params.t;
    let u = params.u;
    let d = params.d;
    let elem_width = if DECIMATE { u - d } else { u };
    let key_shift = t * elem_width - b;

    let key_of = |x: T| -> T {
        let mut key = x;
        key >>= key_shift;
        key
    };

    let mut local = snapshot;
    let target = T::from_u64(pass);
    let mut len = 0usize;
    for _ in 0..stream_len {
        let x = if DECIMATE {
            match params.draw_decimate_once::<T, DIM, FULL>(&mut local) {
                Some(x) => x,
                None => continue,
            }
        } else {
            params.draw::<T, DIM, FULL>(&mut local)
        };
        if key_of(x) == target {
            *buf.get_mut(len)
                .unwrap_or_else(|| bin_overflow("a tradeoff bin")) = x;
            len += 1;
        }
    }
    // The final state does not depend on the pass.
    (len, local)
}

/// Returns the header note on the null distribution of a collision test: empty
/// for a Poisson null distribution, and otherwise the approximation and the
/// variance.
pub(crate) fn null_desc(points: usize, cells: f64) -> String {
    let null = Null::collisions(points as f64, cells);
    match null.kind() {
        NullKind::Poisson => String::new(),
        NullKind::Binomial => format!(" (binomial approximation, variance: {})", null.var),
        NullKind::Normal => format!(" (normal approximation, variance: {})", null.var),
    }
}

/// Returns the null distribution of a test with `points` points on `cells`
/// cells: [`Null::collisions`] for the collision test, and a Poisson
/// distribution with mean points³/(4 · cells) for birthday spacings.
///
/// It is used both with the nominal number of points (header line and default
/// sizing) and with the number of points actually kept, which differs only
/// under decimation.
pub fn test_null(points: usize, cells: f64, birthday_spacings: bool) -> Null {
    if birthday_spacings {
        // TestU01 long guide: lambda = n³ / (4k).
        Null::poisson(BigUint::from(points).pow(3).to_f64().unwrap() / (cells * 4.0))
    } else {
        Null::collisions(points as f64, cells)
    }
}

/// Returns the expected count (the mean of the null distribution) and the
/// number of points, applying the defaults of each test.
pub fn compute_lambda_and_points(args: &Args, cells: &BigUint) -> (f64, usize) {
    // cells is (2ᵘ⁻ᵈ)ᵗ, so it takes decimation into account.
    let effective_cells_f64 = cells.to_f64().unwrap();

    // With tradeoff, m is the memory per pass, and the number of points is
    // m · 2ᵇ.
    let pass_factor = match args.tradeoff_bits {
        Some(b) => 1usize.checked_shl(b as u32).expect("2ᵇ overflows usize"),
        None => 1,
    };

    let points;
    let null = if args.birthday_spacings {
        // The largest number of points allowed by TestU01 (8rλ ≤ k¹ᐟ⁴; long guide,
        // p. 132).
        let max_points = (effective_cells_f64.powf(5.0 / 12.0)
            / (2.0 * args.reps as f64).powf(1.0 / 3.0)) as usize;
        // As for collisions, the number of points is m · 2ᵇ.
        let m = args.m.unwrap_or(max_points / pass_factor.max(1));
        points = m.checked_mul(pass_factor).unwrap_or_else(|| {
            Args::die("the number of points m · 2ᵇ overflows the address space (reduce m or b)")
        });
        if points > max_points {
            Args::die(
                "the given combination of memory, repetitions and cells is out of range \
                 (omit -m to use the maximum)",
            );
        }
        test_null(points, effective_cells_f64, true)
    } else {
        // By default, ⌊VARIANCE_MAXIMIZING_DENSITY · cells⌋ points split across
        // the 2ᵇ passes, capped so that m · 2ᵇ fits a usize (an explicit, too
        // large m fails the checked multiplication below).
        let m_cap = usize::MAX / pass_factor;
        let m_max_var =
            (VARIANCE_MAXIMIZING_DENSITY * effective_cells_f64 / pass_factor as f64).floor();
        let m_default = if m_max_var >= m_cap as f64 {
            m_cap
        } else {
            m_max_var as usize
        };
        let m = args.m.unwrap_or(m_default);
        points = m.checked_mul(pass_factor).unwrap_or_else(|| {
            Args::die("the number of points m · 2ᵇ overflows the address space (reduce m or b)")
        });

        // Densities slightly above the one maximizing the variance are allowed.
        if points as f64 > MAX_DENSITY * effective_cells_f64 {
            Args::die(&format!(
                "more points ({}) than {} times the number of {}cells ({})",
                points,
                MAX_DENSITY,
                if args.decimation_bits.is_some() {
                    "effective "
                } else {
                    ""
                },
                cells
            ));
        }
        test_null(points, effective_cells_f64, false)
    };

    if points < 10000 {
        Args::die(&format!(
            "the number of points ({points}) is smaller than 10000"
        ));
    }

    (null.mean, points)
}

/// Runs a test sequentially.
///
/// Returns the total number of collisions and the sum of the null
/// distributions of the repetitions, each conditioned on the number of points
/// actually kept (which differs from `points` only under decimation).
///
/// # Implementation Details
///
/// The specialization of the runner is chosen once per test: `DIM` is the
/// dimension *t* if *t* ≤ 8, so that the draw loop can be unrolled, and zero
/// otherwise; `DECIMATE` is true if *d* > 0; `FULL` is true if *u* = 64 and
/// *s* = 0 (see [`cell_index`]).
///
/// [`cell_index`]: crate::cell::cell_index
pub fn run_test<T: Cell>(args: &Args, points: usize, cells: &BigUint, lambda: f64) -> (u128, Null) {
    let seed = args.seed;
    eprintln!("Seed: {:#018x}", seed);

    let mut prng = Prng::new(seed);

    let d = args.decimation_bits.unwrap_or(0);
    let tradeoff_b = args.tradeoff_bits(); // tradeoff bits b (0 when absent)
    // A pass scans points · 2ᵗᵈ samples, and keeps those selected by the
    // t · d + b bits of decimation and tradeoff.
    let partition_bits = args.t * d + tradeoff_b;
    let scan_len = scan_samples(points, args.t, d);
    // The birthday tradeoff uses this buffer for the spacings of a class, and
    // allocates internally a buffer for the points of an interval.
    let buf_len = if args.birthday_spacings && tradeoff_b > 0 {
        buffer_size(points, tradeoff_b)
    } else {
        buffer_size(scan_len, partition_bits)
    };

    let output_type = bits_read_desc(args.s);
    let test_type = if args.birthday_spacings {
        "birthday-spacings"
    } else {
        "collision"
    };

    // One buffer holds one bin of the sample scan.
    let headroom_suffix = headroom_desc(
        buf_len,
        (scan_len as f64) / 2.0f64.powi(partition_bits as i32),
        partition_bits,
    );

    let mut mode_parts: Vec<String> = Vec::new();
    if tradeoff_b > 0 {
        mode_parts.push(format!(
            "tradeoff on {} top bits over {} passes",
            tradeoff_b,
            1u64 << tradeoff_b
        ));
    }
    if d > 0 {
        mode_parts.push(decimation_desc(d, args.t));
    }
    let mode_suffix = join_mode_parts(&mode_parts);

    eprintln!(
        "Running a {}-dimensional {} test sequentially on the upper {} bits of the {} ({} points, {}-bit cells, {} memory locations, {:.3} GiB RAM{}{})",
        args.t,
        test_type,
        args.u,
        output_type,
        points,
        size_of::<T>() * 8,
        points >> tradeoff_b,
        // In floating point, as the size in bytes might overflow a usize (in
        // which case alloc_mmap will fail later).
        buf_len as f64 * size_of::<T>() as f64 / 2.0f64.powi(30),
        headroom_suffix,
        mode_suffix
    );

    let cells_suffix = effective_cells_suffix(d, args.u, args.t);
    eprintln!(
        "u: {} t: {} cells: {:.0} expected collisions: {}{}{}",
        args.u,
        args.t,
        cells,
        lambda,
        if args.birthday_spacings {
            String::new()
        } else {
            null_desc(points, cells.to_f64().unwrap())
        },
        cells_suffix
    );

    let mut mapped = alloc_mmap::<T>(buf_len);
    let buf: &mut [T] = bytemuck::try_cast_slice_mut(&mut mapped).unwrap();

    let params = GridParams {
        u: args.u,
        t: args.t,
        s: args.s,
        d,
        cells,
    };

    let full = args.u == 64 && args.s == 0;
    let decimating = d > 0;

    // cells already takes decimation into account.
    let effective_cells_f64 = cells.to_f64().unwrap();

    let mut sw = Stopwatch::new();
    let mut tot: u128 = 0;
    let mut null_sum = Null::ZERO;
    for _rep in 1..=args.reps {
        // go!(DIM) expands the FULL/DECIMATE/mode matrix for one DIM literal;
        // the outer match picks DIM (0 = runtime fallback).
        macro_rules! go {
            ($dim:literal) => {{
                if args.birthday_spacings {
                    if tradeoff_b > 0 {
                        match (decimating, full) {
                            (false, false) => run_birthday_tradeoff::<T, $dim, false, false>(
                                &mut prng,
                                &params,
                                buf,
                                points,
                                tradeoff_b,
                                args.pretty_p,
                                args.pass,
                            ),
                            (true, false) => run_birthday_tradeoff::<T, $dim, true, false>(
                                &mut prng,
                                &params,
                                buf,
                                points,
                                tradeoff_b,
                                args.pretty_p,
                                args.pass,
                            ),
                            (false, true) => run_birthday_tradeoff::<T, $dim, false, true>(
                                &mut prng,
                                &params,
                                buf,
                                points,
                                tradeoff_b,
                                args.pretty_p,
                                args.pass,
                            ),
                            (true, true) => run_birthday_tradeoff::<T, $dim, true, true>(
                                &mut prng,
                                &params,
                                buf,
                                points,
                                tradeoff_b,
                                args.pretty_p,
                                args.pass,
                            ),
                        }
                    } else {
                        match (decimating, full) {
                            (false, false) => run_birthday::<T, $dim, false, false>(
                                &mut prng, &params, buf, points,
                            ),
                            (true, false) => run_birthday::<T, $dim, true, false>(
                                &mut prng, &params, buf, points,
                            ),
                            (false, true) => run_birthday::<T, $dim, false, true>(
                                &mut prng, &params, buf, points,
                            ),
                            (true, true) => {
                                run_birthday::<T, $dim, true, true>(&mut prng, &params, buf, points)
                            }
                        }
                    }
                } else if tradeoff_b > 0 {
                    let cells_per_pass = effective_cells_f64 / (1u64 << tradeoff_b) as f64;
                    if decimating {
                        run_collision_tradeoff::<T, $dim, true>(
                            &mut prng,
                            &params,
                            buf,
                            points,
                            tradeoff_b,
                            cells_per_pass,
                            args.pretty_p,
                            args.pass,
                        )
                    } else {
                        run_collision_tradeoff::<T, $dim, false>(
                            &mut prng,
                            &params,
                            buf,
                            points,
                            tradeoff_b,
                            cells_per_pass,
                            args.pretty_p,
                            args.pass,
                        )
                    }
                } else if decimating {
                    if full {
                        run_collision_decimate::<T, $dim, true>(
                            &mut prng,
                            &params,
                            buf,
                            points,
                            effective_cells_f64,
                            args.checkpoints,
                            args.pretty_p,
                        )
                    } else {
                        run_collision_decimate::<T, $dim, false>(
                            &mut prng,
                            &params,
                            buf,
                            points,
                            effective_cells_f64,
                            args.checkpoints,
                            args.pretty_p,
                        )
                    }
                } else if full {
                    run_collision::<T, $dim, true>(&mut prng, &params, buf)
                } else {
                    run_collision::<T, $dim, false>(&mut prng, &params, buf)
                }
            }};
        }

        let (c, used) = match args.t {
            1 => go!(1),
            2 => go!(2),
            3 => go!(3),
            4 => go!(4),
            5 => go!(5),
            6 => go!(6),
            7 => go!(7),
            8 => go!(8),
            _ => go!(0),
        };

        tot += c as u128;
        // Condition the null distribution on the points actually kept, whose
        // number is random under decimation, to avoid overdispersion.
        let null_rep = test_null(used, effective_cells_f64, args.birthday_spacings);
        null_sum += null_rep;
        if args.pass.is_some() {
            // With --pass, used is the number of points of one pass, so a
            // p-value would be meaningless: main prints the count and the share
            // of the null distribution.
            if args.reps > 1 {
                eprintln!("{c}\tcombined: {tot}");
            } else {
                eprintln!("{c}");
            }
        } else {
            let rep_p = format_p_value(p_value(c as f64, null_rep), args.pretty_p);
            if args.reps > 1 {
                eprintln!(
                    "{c}\tp={rep_p}\tcombined: {tot}\tp={}",
                    format_p_value(p_value(tot as f64, null_sum), args.pretty_p)
                );
            } else {
                eprintln!("{c}\tp={rep_p}");
            }
        }
    }
    eprintln!("Test completed in {:.2} seconds", sw.lap());
    (tot, null_sum)
}

#[cfg(all(test, feature = "incr"))]
mod prescan_tests {
    use super::*;

    // Tests that prescan_checkpoints returns the states at the boundaries,
    // using the fact that after n steps the next output of incr is
    // seed + n + 1.
    #[test]
    fn test_prescan_lands_at_jump_targets() {
        let seed = 0x1234_5678_9abc_def0u64;
        let t = 3usize;
        let total = 1000usize;
        let boundaries = [0usize, 137, 500, 999];
        let (snaps, end) = prescan_checkpoints(Prng::new(seed), t, total, &boundaries);
        assert_eq!(snaps.len(), boundaries.len());
        for (k, &b) in boundaries.iter().enumerate() {
            let mut s = snaps[k];
            let expected = seed.wrapping_add((b * t) as u64).wrapping_add(1);
            assert_eq!(
                s.next_u64(),
                expected,
                "snapshot {k} at boundary {b} landed wrong"
            );
        }
        let mut e = end;
        let expected_end = seed.wrapping_add((total * t) as u64).wrapping_add(1);
        assert_eq!(e.next_u64(), expected_end, "chained end state landed wrong");
    }
}

// Tests of compact_blocks (no generator is involved).
#[cfg(test)]
mod compact_tests {
    use super::*;

    /// Checks that compact_blocks concatenates the used prefixes of the
    /// regions.
    fn check(caps: &[usize], used: &[usize]) {
        let cap_total: usize = caps.iter().sum();
        // Used elements are distinct, and the others are u64::MAX.
        let mut buf = vec![u64::MAX; cap_total];
        let mut base = 0usize;
        let mut expected: Vec<u64> = Vec::new();
        for (i, (&cap, &len)) in caps.iter().zip(used).enumerate() {
            for j in 0..len {
                let v = (i as u64) * 1000 + j as u64;
                buf[base + j] = v;
                expected.push(v);
            }
            base += cap;
        }
        let total = compact_blocks::<u64>(&mut buf, caps, used);
        assert_eq!(
            total,
            used.iter().sum::<usize>(),
            "total for {caps:?}/{used:?}"
        );
        assert_eq!(
            &buf[..total],
            &expected[..],
            "compacted for {caps:?}/{used:?}"
        );
    }

    #[test]
    fn test_compaction_cases() {
        check(&[5, 5, 5], &[5, 5, 5]); // all full: plain no-op
        check(&[5, 5, 5], &[3, 4, 2]); // generic gaps
        check(&[5, 5, 5], &[0, 4, 2]); // first empty
        check(&[5, 5, 5], &[3, 0, 2]); // middle empty
        check(&[5, 5, 5], &[3, 4, 0]); // last empty
        check(&[5, 5, 5], &[0, 0, 0]); // all empty
        check(&[10, 1, 7], &[1, 1, 7]); // big leftward shift, full last block
        check(&[4], &[2]); // single block
        check(&[0, 5, 0, 3], &[0, 5, 0, 3]); // zero-capacity sub-regions
    }
}
