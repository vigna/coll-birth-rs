/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

//! The birthday-spacings test.

use std::mem::size_of;

use num::BigUint;
use num::traits::ToPrimitive;
use rayon::prelude::*;

use crate::cell::Cell;
use crate::cli::Args;
use crate::common::{
    GridParams, OrbitPartition, alloc_mmap, bin_overflow, bits_read_desc, buffer_size,
    count_adjacent_equals, decimation_desc, gen_unit_contiguous, generation_desc, headroom_desc,
    join_mode_parts, scan_samples, test_null,
};
use crate::prng::Prng;
use crate::stats::{Null, format_p_value, p_value};
use crate::util::{Stopwatch, parallelism};

/// Runs a birthday-spacings test.
///
/// Without decimation, exactly `points` points are generated. With decimation
/// (`DECIMATE`), `points` · 2ᵗᵈ samples are scanned, keeping about `points`
/// points, as in [`run_collision_decimate`]. The parallel counterpart is
/// [`run_birthday_parallel`].
///
/// Returns the number of collisions among spacings and the number of points
/// actually kept.
///
/// [`run_collision_decimate`]: crate::collision::run_collision_decimate
pub fn run_birthday<T: Cell, const DIM: usize, const DECIMATE: bool, const FULL: bool>(
    prng: &mut Prng,
    params: &GridParams,
    buf: &mut [T],
    points: usize,
) -> (usize, usize) {
    let mut sw = Stopwatch::new();
    eprint!("Generating points...");
    let len = if DECIMATE {
        let scan_len = scan_samples(points, params.t, params.d);
        let mut len = 0usize;
        for _ in 0..scan_len {
            if let Some(x) = params.draw_decimate_once::<T, DIM, FULL>(prng) {
                *buf.get_mut(len)
                    .unwrap_or_else(|| bin_overflow("a decimated birthday run")) = x;
                len += 1;
            }
        }
        len
    } else {
        for x in buf[..points].iter_mut() {
            *x = params.draw::<T, DIM, FULL>(prng);
        }
        points
    };
    let pts = &mut buf[..len];

    eprint!("[{:.3}s] sorting...", sw.lap());
    T::sort_mt(pts);

    eprint!("[{:.3}s] computing deltas...", sw.lap());
    compute_spacings(pts, params.cells);

    eprint!("[{:.3}s] sorting deltas...", sw.lap());
    T::sort_mt(pts);

    eprint!("[{:.3}s] counting collisions...", sw.lap());
    let c = count_adjacent_equals(pts);

    eprintln!("[{:.3}s] {len} points done.", sw.lap());
    (c, len)
}

/// Replaces in parallel the elements of a sorted slice with their spacings.
///
/// Each element is replaced by its difference with its predecessor, and the
/// first element by the wrap-around spacing `cells` − max + min.
///
/// The wrap-around spacing lies in \[1 . . `cells`\], so it might not be
/// representable (e.g., 2⁶⁴ cells in a `u64`), and it is equal to `cells` only
/// if min = max. Thus, it is computed through `cells` − 1, and in the
/// degenerate case it is replaced by a nonzero value: since all other spacings
/// are then zero, the number of collisions does not change.
pub(crate) fn compute_spacings<T: Cell>(v: &mut [T], cells: &BigUint) {
    if v.is_empty() {
        return;
    }
    let global_min = v[0];
    let global_max = *v.last().unwrap();

    // About ten chunks per thread, but at least 1024 elements per chunk to
    // amortize scheduling.
    let chunk_size = (v.len() / (parallelism() * 10)).max(1024);

    let chunk_tails: Vec<T> = v.chunks(chunk_size).map(|c| *c.last().unwrap()).collect();

    v.par_chunks_mut(chunk_size).for_each(|c| {
        let mut prev = *c.last().unwrap();
        for i in (1..c.len()).rev() {
            let tmp = c[i - 1];
            c[i] = prev - tmp;
            prev = tmp;
        }
    });

    // Fix the first spacing of each chunk except the first one.
    v.par_chunks_mut(chunk_size)
        .enumerate()
        .skip(1)
        .for_each(|(i, c)| c[0] -= chunk_tails[i - 1]);

    // The first element is still the minimum: replace it with the wrap-around
    // spacing.
    if global_min == global_max {
        v[0] = T::from_u64(1);
    } else {
        // min + (cells - 1 - max) + 1: every intermediate fits, since min < max.
        let cells_m1 = T::from_u128((cells - BigUint::from(1u8)).to_u128().unwrap());
        v[0] += cells_m1 - global_max;
        v[0] += T::from_u64(1);
    }
}

/// Runs a birthday-spacings test using a space/time tradeoff with two levels,
/// each with 2ᵇ passes.
///
/// - The inner (distance) level partitions the cell index into 2ᵇ contiguous
///   intervals by its top *b* bits, and visits them in increasing order. The
///   points of each interval are generated, sorted, and replaced by their
///   spacings; the first spacing of an interval is computed using the maximum
///   of the previous interval, and the wrap-around spacing (`cells` − max +
///   min) is computed at the end. Since the intervals are contiguous and
///   visited in order, these are exactly the spacings of all points.
///
/// - The outer (counting) level partitions spacings into 2ᵇ classes by their
///   *lowest* *b* bits, as spacings cluster near zero, so their top bits are
///   unbalanced. Each pass keeps, sorts, and counts the spacings of a class.
///   Since equal spacings fall in the same class, the sum of the numbers of
///   collisions of the classes is the number of collisions of all spacings,
///   but only about `points` / 2ᵇ spacings (and `points` / 2ᵇ points) are in
///   memory at the same time.
///
/// The result is thus the same as that of [`run_birthday`].
///
/// This function runs a single repetition, and it is called by [`run_test`].
/// The parallel counterpart is [`run_birthday_parallel`], which returns the
/// same result.
///
/// Returns the number of collisions among spacings and the number of points
/// actually kept.
///
/// [`run_test`]: crate::common::run_test
pub fn run_birthday_tradeoff<T: Cell, const DIM: usize, const DECIMATE: bool, const FULL: bool>(
    prng: &mut Prng,
    params: &GridParams,
    class_buf: &mut [T],
    points: usize,
    b: usize,
    pretty_p: bool,
    pass: Option<u64>,
) -> (usize, usize) {
    let t = params.t;
    let u = params.u;
    let d = params.d;
    let num_passes: u64 = 1u64 << b;
    // With --pass K, only class K runs (counts of different classes can be
    // summed).
    let (pass_lo, pass_hi) = match pass {
        Some(k) => (k, k + 1),
        None => (0, num_passes),
    };
    // Cumulative statistics are printed only if more than one class runs.
    let multi_pass = pass_hi - pass_lo > 1;
    let elem_width = if DECIMATE { u - d } else { u };
    let point_key_shift = t * elem_width - b; // top b bits select the value interval
    let spacing_mask = T::low_bits_mask(b); // low b bits select the spacing class
    let scan_len = scan_samples(points, t, d);
    // cells itself may be unrepresentable (2ᴺ in N-bit storage); cells − 1 always
    // is, and the wrap-around spacing is evaluated through it.
    let cells_m1 = T::from_u128((params.cells - BigUint::from(1u8)).to_u128().unwrap());

    // The points of an interval (about points / 2ᵇ, plus headroom).
    let mut scratch: Vec<T> = vec![T::ZERO; buffer_size(scan_len, t * d + b)];

    let snapshot = *prng;
    let mut end_state = snapshot;
    let mut total_coll = 0usize;
    let mut total_points = 0usize;
    let mut sw = Stopwatch::new();
    eprintln!("Birthday tradeoff over {num_passes} spacing-classes");
    // Times a class, that is, a sweep over all intervals.
    let mut class_sw = Stopwatch::new();
    let cells_f64 = params.cells.to_f64().unwrap();
    // The nominal null distribution of a class, for the progress p-values.
    let null_class = Null::poisson((points as f64).powi(3) / (4.0 * cells_f64) / num_passes as f64);

    for j in pass_lo..pass_hi {
        let class_target = T::from_u64(j);
        let mut class_len = 0usize;
        let mut prev_max: Option<T> = None;
        let mut global_min: Option<T> = None;
        let mut global_max: Option<T> = None;

        for k in 0..num_passes {
            let interval_target = T::from_u64(k);
            // Each interval requires a full scan, so we report progress.
            let mut isw = Stopwatch::new();
            eprint!(
                "  Class {}/{} interval {}/{}: gen...",
                j + 1,
                num_passes,
                k + 1,
                num_passes
            );
            // Replay the same sample stream, keeping the points in interval k.
            let mut local = snapshot;
            let mut len = 0usize;
            for _ in 0..scan_len {
                let x = if DECIMATE {
                    match params.draw_decimate_once::<T, DIM, FULL>(&mut local) {
                        Some(x) => x,
                        None => continue,
                    }
                } else {
                    params.draw::<T, DIM, FULL>(&mut local)
                };
                let mut key = x;
                key >>= point_key_shift;
                if key == interval_target {
                    *scratch
                        .get_mut(len)
                        .unwrap_or_else(|| bin_overflow("a birthday value interval")) = x;
                    len += 1;
                }
            }
            end_state = local;
            // Every point lies in exactly one interval, so we count points
            // during the first sweep only.
            if j == pass_lo {
                total_points += len;
            }
            if len == 0 {
                eprintln!("[{:.3}s] empty", isw.lap());
                continue;
            }
            eprint!("[{:.3}s] sort...", isw.lap());
            T::sort_st(&mut scratch[..len]);
            let interval_min = scratch[0];
            let interval_max = scratch[len - 1];
            if global_min.is_none() {
                global_min = Some(interval_min);
            }
            global_max = Some(interval_max);

            // Scatter-back: replace each point by its spacing to its predecessor.
            for i in (1..len).rev() {
                scratch[i] = scratch[i] - scratch[i - 1];
            }
            // The first spacing is computed using the maximum of the previous
            // interval; in the first interval, it is the wrap-around spacing,
            // which is computed at the end.
            let start = match prev_max {
                Some(pm) => {
                    scratch[0] -= pm;
                    0
                }
                None => 1,
            };
            for &s in &scratch[start..len] {
                if s & spacing_mask == class_target {
                    *class_buf
                        .get_mut(class_len)
                        .unwrap_or_else(|| bin_overflow("a birthday-spacings class")) = s;
                    class_len += 1;
                }
            }
            eprintln!("[{:.3}s], {len} points", isw.lap());
            prev_max = Some(interval_max);
        }

        // The wrap-around spacing cells − max + min is equal to cells, which
        // might not be representable, only if all points coincide: in this
        // case all other spacings are zero, so we can drop it.
        if let (Some(gmin), Some(gmax)) = (global_min, global_max) {
            if gmin != gmax {
                let mut wrap = cells_m1 - gmax;
                wrap += gmin;
                wrap += T::from_u64(1);
                if wrap & spacing_mask == class_target {
                    *class_buf
                        .get_mut(class_len)
                        .unwrap_or_else(|| bin_overflow("a birthday-spacings class")) = wrap;
                    class_len += 1;
                }
            }
        }

        T::sort_st(&mut class_buf[..class_len]);
        let class_coll = count_adjacent_equals(&class_buf[..class_len]);
        total_coll += class_coll;
        let classes_done = (j - pass_lo + 1) as f64;
        let elapsed = class_sw.lap();
        let class_p = format_p_value(p_value(class_coll as f64, null_class), pretty_p);
        if multi_pass {
            eprintln!(
                "  Class {}/{} done: [{elapsed:.3}s], {class_len} spacings, {class_coll} collisions, p={class_p}; combined: {total_coll} collisions, p={}",
                j + 1,
                num_passes,
                format_p_value(
                    p_value(total_coll as f64, classes_done * null_class),
                    pretty_p
                ),
            );
        } else {
            eprintln!(
                "  Class {}/{} done: [{elapsed:.3}s], {class_len} spacings, {class_coll} collisions, p={class_p}",
                j + 1,
                num_passes,
            );
        }
    }
    *prng = end_state;
    eprintln!("[{:.3}s] done.", sw.lap());
    (total_coll, total_points)
}

/// Runs a birthday-spacings test in parallel.
///
/// The test uses the same two levels as [`run_birthday_tradeoff`], and splits
/// samples among threads as [`run_test_parallel`] does:
///
/// - At the inner (distance) level, for each interval the threads write points
///   into disjoint regions of a buffer, which are then compacted and sorted.
///   Spacings are computed on the fly, leaving the sorted interval unmodified;
///   the first spacing of an interval is computed using the maximum of the
///   previous interval, and the wrap-around spacing is computed at the end.
///
/// - At the outer (counting) level, the spacings of the current class are
///   copied in parallel into a buffer, which is then sorted and scanned to
///   count collisions. If *b* = 0, there is a single interval and a single
///   class.
///
/// The result is the same as that of [`run_birthday_tradeoff`] (or
/// [`run_birthday`] if *b* = 0) for every number of threads and repetitions.
///
/// Differently from the sequential runners called by [`run_test`], this
/// function runs all repetitions, prints the header, and selects modes at run
/// time rather than using const generics.
///
/// Returns the total number of collisions and the sum of the null
/// distributions of the repetitions, each conditioned on the number of points
/// actually kept.
///
/// [`run_test_parallel`]: crate::collision::run_test_parallel
/// [`run_test`]: crate::common::run_test
pub fn run_birthday_parallel<T: Cell>(
    args: &Args,
    points: usize,
    cells: &BigUint,
    lambda: f64,
    num_cpus: usize,
) -> (u128, Null) {
    let seed = args.seed;
    eprintln!("Seed: {:#018x}", seed);

    let d = args.decimation_bits.unwrap_or(0);
    let b = args.tradeoff_bits();
    let num_passes: u64 = 1u64 << b;
    let t = args.t;
    let decimating = d > 0;
    let full = args.u == 64 && args.s == 0;
    let partition_bits = t * d + b;
    let spacing_mask = T::low_bits_mask(b);
    // cells itself may be unrepresentable (2ᴺ in N-bit storage); cells − 1 always
    // is, and the wrap-around spacing is evaluated through it.
    let cells_m1 = T::from_u128((cells - BigUint::from(1u8)).to_u128().unwrap());

    let scan_total = scan_samples(points, t, d);
    let mut partition = OrbitPartition::new(seed, num_cpus, scan_total, t);
    let num_cpus = partition.num_cpus;
    let base_chunk = partition.base_chunk;
    let rem = partition.rem;
    let chunk = |i: usize| base_chunk + if i < rem { 1 } else { 0 };

    let block_cap = |i: usize| buffer_size(chunk(i), partition_bits).max(1);
    // The capacities of the thread regions of the interval buffer.
    let caps: Box<[usize]> = (0..num_cpus).map(block_cap).collect();
    let interval_cap: usize = caps.iter().sum();
    // The class buffer accumulates the spacings of a class over all intervals.
    // Under decimation their number is random, with mean points, so the buffer
    // needs the full t · d + b headroom (buffer_size(points, b) provides none
    // if b = 0).
    let class_cap = buffer_size(scan_total, partition_bits).max(1);

    let params = GridParams {
        u: args.u,
        t,
        s: args.s,
        d,
        cells,
    };

    let split_desc = partition.split_desc();

    let output_type = bits_read_desc(args.s);

    let mut mode_parts: Vec<String> = Vec::new();
    if b > 0 {
        // Two levels: 2ᵇ value intervals by 2ᵇ spacing classes.
        mode_parts.push(format!(
            "tradeoff on {} top bits over {} value intervals x {} spacing classes",
            b, num_passes, num_passes
        ));
    }
    if d > 0 {
        mode_parts.push(decimation_desc(d, t));
    }
    let mode_suffix = join_mode_parts(&mode_parts);

    // The interval and class buffers are both in memory during a repetition.
    let live_elems: usize = interval_cap + class_cap;

    // Each buffer holds one bin of the scan (the points of an interval, or the
    // spacings of a class), so the nominal load is twice the bin mean.
    let headroom_suffix = headroom_desc(
        live_elems,
        2.0 * (scan_total as f64) / 2.0f64.powi(partition_bits as i32),
        partition_bits,
    );
    eprintln!(
        "Running a {t}-dimensional birthday-spacings test {} on the upper {} bits of the {} \
         ({} points, {}-bit cells, {} memory locations, {:.3} GiB RAM{}{})",
        generation_desc(num_cpus, split_desc),
        args.u,
        output_type,
        points,
        size_of::<T>() * 8,
        points >> b,
        // In floating point, as the size in bytes might overflow a usize (in
        // which case alloc_mmap will fail later).
        live_elems as f64 * size_of::<T>() as f64 / 2.0f64.powi(30),
        headroom_suffix,
        mode_suffix
    );
    eprintln!(
        "u: {} t: {} cells: {:.0} expected collisions: {}",
        args.u, t, cells, lambda
    );

    let mut sw = Stopwatch::new();
    let mut tot: u128 = 0;
    let mut null_sum = Null::ZERO;
    let cells_f64 = cells.to_f64().unwrap();

    for rep in 1..=args.reps {
        let mut interval_buf = alloc_mmap::<T>(interval_cap);
        let mut class_buf = alloc_mmap::<T>(class_cap);

        // Per-thread orbit starts for the scan sub-ranges, reused for every interval.
        let snapshots = partition.rep_snapshots(rep);

        let mut rep_coll = 0usize;
        let mut rep_points = 0usize;
        let mut psw = Stopwatch::new();
        // With --pass K, only class K runs (counts of different classes can be
        // summed).
        let (pass_lo, pass_hi) = match args.pass {
            Some(k) => (k, k + 1),
            None => (0, num_passes),
        };
        // Cumulative statistics are printed only if more than one class runs.
        let multi_pass = pass_hi - pass_lo > 1;
        eprintln!(
            "Rep {}/{}: {} value intervals x {} spacing classes",
            rep, args.reps, num_passes, num_passes
        );
        // Times a class, that is, a sweep over all intervals.
        let mut class_sw = Stopwatch::new();
        // The nominal null distribution of a class, for the progress p-values
        // (the final p-value is conditioned on the number of points kept).
        let null_class = test_null(points, cells_f64, true) / num_passes as f64;

        for j in pass_lo..pass_hi {
            let class_target = T::from_u64(j);
            let class: &mut [T] = bytemuck::try_cast_slice_mut(&mut class_buf).unwrap();
            let mut class_len = 0usize;
            let mut prev_max: Option<T> = None;
            let mut global_min: Option<T> = None;
            let mut global_max: Option<T> = None;

            for k in 0..num_passes {
                // Each interval requires a full scan, so we report progress.
                let mut isw = Stopwatch::new();
                eprint!(
                    "  Class {}/{}, interval {}/{}: gen...",
                    j + 1,
                    num_passes,
                    k + 1,
                    num_passes
                );
                // Phase 1: generate the points of interval k into a contiguous
                // buffer.
                let unit: &mut [T] = bytemuck::try_cast_slice_mut(&mut interval_buf).unwrap();
                let total = gen_unit_contiguous::<T>(
                    unit, &caps, &snapshots, &params, &chunk, k, b, decimating, full,
                );
                // Every point lies in exactly one interval, so we count points
                // during the first sweep only.
                if j == pass_lo {
                    rep_points += total;
                }
                if total == 0 {
                    eprintln!("[{:.3}s] empty", isw.lap());
                    continue;
                }
                eprint!("[{:.3}s] sort...", isw.lap());
                let interval: &mut [T] = &mut unit[..total];
                T::sort_mt(interval);
                let interval_max = interval[total - 1];
                if global_min.is_none() {
                    global_min = Some(interval[0]);
                }
                global_max = Some(interval_max);

                eprint!("[{:.3}s] filter...", isw.lap());
                // Copy the spacings of the class into the class buffer, in two
                // parallel passes (count, then write). The first point of the
                // first interval is skipped, as its spacing is the wrap-around
                // one. Order is irrelevant, as the class buffer will be sorted.
                let start = if prev_max.is_none() { 1 } else { 0 };
                let interval: &[T] = interval;
                let span = total - start;
                let chunk_bounds = |c: usize| {
                    (
                        start + c * span / num_cpus,
                        start + (c + 1) * span / num_cpus,
                    )
                };
                let spacing_at = |i: usize| {
                    if i == 0 {
                        interval[0] - prev_max.unwrap()
                    } else {
                        interval[i] - interval[i - 1]
                    }
                };
                // Pass 1: count matching spacings per chunk.
                let counts: Vec<usize> = (0..num_cpus)
                    .into_par_iter()
                    .map(|c| {
                        let (lo, hi) = chunk_bounds(c);
                        (lo..hi)
                            .filter(|&i| spacing_at(i) & spacing_mask == class_target)
                            .count()
                    })
                    .collect();
                let matched: usize = counts.iter().sum();
                if class_len + matched > class.len() {
                    bin_overflow("a birthday-spacings class");
                }
                // Pass 2: each chunk recomputes its matching spacings and
                // writes them into its own slice of the class buffer.
                {
                    let mut rest = &mut class[class_len..class_len + matched];
                    let mut dsts: Vec<&mut [T]> = Vec::with_capacity(num_cpus);
                    for &cnt in &counts {
                        let (head, tail) = rest.split_at_mut(cnt);
                        dsts.push(head);
                        rest = tail;
                    }
                    dsts.into_par_iter().enumerate().for_each(|(c, dst)| {
                        let (lo, hi) = chunk_bounds(c);
                        let mut w = 0usize;
                        for i in lo..hi {
                            let s = spacing_at(i);
                            if s & spacing_mask == class_target {
                                dst[w] = s;
                                w += 1;
                            }
                        }
                    });
                }
                class_len += matched;
                eprintln!("[{:.3}s], {total} points", isw.lap());
                prev_max = Some(interval_max);
            }

            // The wrap-around spacing cells − max + min is equal to cells,
            // which might not be representable, only if all points coincide: in
            // this case all other spacings are zero, so we can drop it.
            if let (Some(gmin), Some(gmax)) = (global_min, global_max) {
                if gmin != gmax {
                    let mut wrap = cells_m1 - gmax;
                    wrap += gmin;
                    wrap += T::from_u64(1);
                    if wrap & spacing_mask == class_target {
                        *class
                            .get_mut(class_len)
                            .unwrap_or_else(|| bin_overflow("a birthday-spacings class")) = wrap;
                        class_len += 1;
                    }
                }
            }
            T::sort_mt(&mut class[..class_len]);
            let class_coll = count_adjacent_equals(&class[..class_len]);
            rep_coll += class_coll;
            let classes_done = (j - pass_lo + 1) as f64;
            let elapsed = class_sw.lap();
            let class_p = format_p_value(p_value(class_coll as f64, null_class), args.pretty_p);
            if multi_pass {
                eprintln!(
                    "  Class {}/{} done: [{elapsed:.3}s], {class_len} spacings, {class_coll} collisions, p={class_p}; combined: {rep_coll} collisions, p={}",
                    j + 1,
                    num_passes,
                    format_p_value(
                        p_value(rep_coll as f64, classes_done * null_class),
                        args.pretty_p
                    ),
                );
            } else {
                eprintln!(
                    "  Class {}/{} done: [{elapsed:.3}s], {class_len} spacings, {class_coll} collisions, p={class_p}",
                    j + 1,
                    num_passes,
                );
            }
        }

        tot += rep_coll as u128;
        // Condition the null distribution on the points actually kept.
        let null_rep = test_null(rep_points, cells_f64, true);
        null_sum += null_rep;
        let elapsed = psw.lap();
        if args.pass.is_some() {
            // With --pass, rep_coll is the count of one class, so a p-value
            // would be meaningless: main prints the count and the share of the
            // null distribution.
            if args.reps > 1 {
                eprintln!("[{elapsed:.3}s] {rep_coll}\tcombined: {tot}");
            } else {
                eprintln!("[{elapsed:.3}s] {rep_coll}");
            }
        } else {
            let rep_p = format_p_value(p_value(rep_coll as f64, null_rep), args.pretty_p);
            if args.reps > 1 {
                eprintln!(
                    "[{elapsed:.3}s] {rep_coll}\tp={rep_p}\tcombined: {tot}\tp={}",
                    format_p_value(p_value(tot as f64, null_sum), args.pretty_p)
                );
            } else {
                eprintln!("[{elapsed:.3}s] {rep_coll}\tp={rep_p}");
            }
        }
    }
    eprintln!("Test completed in {:.2} seconds", sw.lap());
    (tot, null_sum)
}

// Tests of the wrap-around spacing of compute_spacings with 2ᴺ cells in N-bit
// storage (no generator is involved).
#[cfg(test)]
mod spacing_tests {
    use super::*;
    use num::BigUint;

    // Points {3, 10, 2⁶⁴ − 1} on 2⁶⁴ cells have spacings {7, 2⁶⁴ − 11} and
    // wrap-around spacing 2⁶⁴ − (2⁶⁴ − 1) + 3 = 4, all representable in a u64.
    #[test]
    fn test_wrap_at_width_boundary_is_exact() {
        let cells = BigUint::from(1u8) << 64;
        let mut v = [3u64, 10, u64::MAX];
        compute_spacings(&mut v, &cells);
        assert_eq!(v[0], 4, "wrap-around spacing");
        assert_eq!(v[1], 7);
        assert_eq!(v[2], u64::MAX - 10);
    }

    // If all n points coincide, the wrap-around spacing would be 2⁶⁴; it is
    // replaced by a nonzero value, so the n − 1 zero spacings yield the correct
    // count n − 2.
    #[test]
    fn test_degenerate_equal_points_at_width_boundary() {
        let cells = BigUint::from(1u8) << 64;
        let mut v = [7u64, 7, 7, 7];
        compute_spacings(&mut v, &cells);
        v.sort_unstable();
        assert_eq!(count_adjacent_equals(&v), 2);
    }
}
