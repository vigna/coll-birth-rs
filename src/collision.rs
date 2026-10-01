/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

//! The collision test.

use std::mem::size_of;

use num::BigUint;
use num::traits::ToPrimitive;

use crate::cell::Cell;
use crate::cli::Args;
use crate::common::{
    GridParams, OrbitPartition, alloc_mmap, bin_overflow, bits_read_desc, buffer_size,
    count_adjacent_equals, decimation_desc, effective_cells_suffix, gen_unit_contiguous,
    generation_desc, headroom_desc, join_mode_parts, merge_into, null_desc, scan_samples,
    test_null,
};
use crate::prng::Prng;
use crate::stats::{Null, format_p_value, p_value};
use crate::util::Stopwatch;

/// Runs a collision test.
///
/// [`run_collision_tradeoff`] and [`run_collision_decimate`] are slower but
/// more powerful alternatives using the same amount of memory, and
/// [`run_test_parallel`] is the parallel counterpart of all three.
///
/// Returns the number of collisions and the number of points, which is always
/// `buf.len()` (other runners return the number of points actually kept, on
/// which the null distribution is conditioned).
pub fn run_collision<T: Cell, const DIM: usize, const FULL: bool>(
    prng: &mut Prng,
    params: &GridParams,
    buf: &mut [T],
) -> (usize, usize) {
    let mut sw = Stopwatch::new();
    eprint!("Generating points...");
    for x in buf.iter_mut() {
        *x = params.draw::<T, DIM, FULL>(prng);
    }

    eprint!("[{:.3}s] sorting...", sw.lap());
    T::sort_mt(buf);

    eprint!("[{:.3}s] counting collisions...", sw.lap());
    let c = count_adjacent_equals(buf);

    eprintln!("[{:.3}s] done.", sw.lap());
    (c, buf.len())
}

/// Runs a collision test using a space/time tradeoff on the top bits.
///
/// The cell index is partitioned into 2ᵇ contiguous intervals by its top *b*
/// bits, and pass *k* keeps the points in the *k*-th interval. Since equal
/// points fall in the same interval, the sum of the numbers of collisions of
/// the passes is the number of collisions of all points, but only about
/// `points` / 2ᵇ points are in memory at the same time. Decimation (if
/// `DECIMATE`) acts independently on the lowest *d* bits of each element.
///
/// A *p*-value is printed after each pass, so the test can be monitored while
/// it runs.
///
/// This function runs a single repetition, and it is called by [`run_test`].
/// The parallel counterpart is [`run_test_parallel`], which returns the same
/// result.
///
/// Returns the number of collisions and the number of points actually kept
/// (`points` without decimation).
///
/// [`run_test`]: crate::common::run_test
#[allow(clippy::too_many_arguments)]
pub fn run_collision_tradeoff<T: Cell, const DIM: usize, const DECIMATE: bool>(
    prng: &mut Prng,
    params: &GridParams,
    buf: &mut [T],
    points: usize,
    b: usize,
    cells_per_pass: f64,
    pretty_p: bool,
    pass: Option<u64>,
) -> (usize, usize) {
    let t = params.t;
    let u = params.u;
    let d = params.d;
    let num_passes: u64 = 1u64 << (b as u64);
    // With --pass K, only pass K runs (counts of different passes can be
    // summed).
    let (pass_lo, pass_hi) = match pass {
        Some(k) => (k, k + 1),
        None => (0, num_passes),
    };
    // Cumulative statistics are printed only if more than one pass runs.
    let multi_pass = pass_hi - pass_lo > 1;

    // Each pass scans points · 2ᵗᵈ samples of t outputs, and keeps those that
    // survive decimation and fall in the interval of the pass.
    let scan_len = scan_samples(points, t, d);

    // Decimation compacts each element to u − d bits.
    let elem_width = if DECIMATE { u - d } else { u };

    // Pass k keeps the points whose top b bits are equal to k.
    let key_shift = t * elem_width - b;
    let key_of = |x: T| -> T {
        let mut key = x;
        key >>= key_shift;
        key
    };

    let snapshot = *prng;
    let mut end_state = snapshot;
    let mut total_coll = 0usize;
    let mut total_len = 0usize;

    for k in pass_lo..pass_hi {
        eprint!("Pass {}/{}: gen...", k + 1, num_passes);
        let mut sw = Stopwatch::new();

        let mut local = snapshot;
        let target = T::from_u64(k);
        let mut len = 0usize;
        for _ in 0..scan_len {
            let x = if DECIMATE {
                match params.draw_decimate_once::<T, DIM, false>(&mut local) {
                    Some(x) => x,
                    None => continue,
                }
            } else {
                params.draw::<T, DIM, false>(&mut local)
            };
            if key_of(x) == target {
                *buf.get_mut(len)
                    .unwrap_or_else(|| bin_overflow("a collision tradeoff pass")) = x;
                len += 1;
            }
        }
        eprint!("[{:.3}s] sort...", sw.lap());

        let slice = &mut buf[..len];
        T::sort_mt(slice);
        eprint!("[{:.3}s] count...", sw.lap());

        let c = count_adjacent_equals(slice);
        let null_pass = Null::collisions(len as f64, cells_per_pass);
        total_coll += c;
        total_len += len;
        let elapsed = sw.lap();
        let local_p = format_p_value(p_value(c as f64, null_pass), pretty_p);
        if multi_pass {
            let null_so_far = Null::collisions(total_len as f64, (k + 1) as f64 * cells_per_pass);
            eprintln!(
                "[{elapsed:1.3}s], {len} points, {c} collisions, p={local_p}; combined: {total_len} points, {total_coll} collisions, p={}",
                format_p_value(p_value(total_coll as f64, null_so_far), pretty_p)
            );
        } else {
            eprintln!("[{elapsed:1.3}s], {len} points, {c} collisions, p={local_p}");
        }
        end_state = local;
    }
    *prng = end_state;
    (total_coll, total_len)
}

/// Runs a collision test with decimation, keeping only the tuples whose
/// elements have their lowest *d* bits equal to zero.
///
/// A fixed number `points` · 2ᵗᵈ of samples is scanned, so the number of points
/// kept is random, with mean `points`.
///
/// Decimation multiplies the expected number of collisions by 2ᵗᵈ, because the
/// number of cells is divided by the same amount. This can lead to stronger
/// results in detecting faulty generators. The idea of using decimation to
/// strengthen the collision test was proposed by [Melissa O'Neill].
///
/// If `checkpoints` is true, the scan is split into ⌊√2ᵗᵈ⌋ stages of the same
/// length, and a cumulative *p*-value is printed after each stage. The output
/// is thus directly comparable with that of [`run_collision_tradeoff`] with the
/// statistically equivalent choice *b* = *t* · *d* / 2.
///
/// This function runs a single repetition, and it is called by [`run_test`].
/// The parallel counterpart is [`run_test_parallel`], which returns the same
/// result, also with checkpoints.
///
/// Returns the number of collisions and the number of points actually kept.
///
/// # Implementation Details
///
/// With checkpoints, the points of each stage are sorted in an auxiliary buffer
/// and merged right to left into the sorted prefix of `buf`, so the cost of
/// each checkpoint is linear, rather than log-linear, in the number of points
/// accumulated so far.
///
/// [Melissa O'Neill]: https://www.pcg-random.org/posts/birthday-test.html
/// [`run_test`]: crate::common::run_test
pub fn run_collision_decimate<T: Cell, const DIM: usize, const FULL: bool>(
    prng: &mut Prng,
    params: &GridParams,
    buf: &mut [T],
    points: usize,
    effective_cells: f64,
    checkpoints: bool,
    pretty_p: bool,
) -> (usize, usize) {
    let t = params.t;
    let d = params.d;
    // The number of points kept is random, so buf has some headroom (see
    // buffer_size).
    let scan_len = scan_samples(points, t, d);

    if !checkpoints {
        let mut sw = Stopwatch::new();
        eprint!(
            "Scanning {scan_len} samples (decimating low {} bits per dimension)...",
            d
        );
        let mut len = 0usize;
        for _ in 0..scan_len {
            if let Some(x) = params.draw_decimate_once::<T, DIM, FULL>(prng) {
                *buf.get_mut(len)
                    .unwrap_or_else(|| bin_overflow("a decimated collision run")) = x;
                len += 1;
            }
        }
        eprint!("[{:.3}s] sort...", sw.lap());
        T::sort_mt(&mut buf[..len]);
        eprint!("[{:.3}s] count...", sw.lap());
        let c = count_adjacent_equals(&buf[..len]);
        eprintln!("[{:.3}s] {len} points done.", sw.lap());
        return (c, len);
    }

    // √2ᵗᵈ is computed in floating point because t · d might exceed 63.
    let num_checkpoints = (2.0f64.powf((t * d) as f64 / 2.0) as usize).clamp(1, scan_len);
    let aux_cap = buffer_size(scan_len.div_ceil(num_checkpoints), t * d).max(1);
    let mut aux: Vec<T> = vec![T::ZERO; aux_cap];

    let mut len = 0usize; // cumulative kept points in `buf`
    let mut scanned = 0usize; // cumulative samples scanned
    let mut c = 0usize;
    for k in 1..=num_checkpoints {
        let target_scanned = scan_len * k / num_checkpoints;
        let stage = target_scanned - scanned;
        let mut sw = Stopwatch::new();
        eprint!("Checkpoint {}/{}: gen...", k, num_checkpoints);

        let mut got = 0usize;
        for _ in 0..stage {
            if let Some(x) = params.draw_decimate_once::<T, DIM, FULL>(prng) {
                *aux.get_mut(got)
                    .unwrap_or_else(|| bin_overflow("a decimation checkpoint stage")) = x;
                got += 1;
            }
        }
        scanned = target_scanned;
        eprint!("[{:.3}s] sort...", sw.lap());

        T::sort_mt(&mut aux[..got]);
        eprint!("[{:.3}s] merge...", sw.lap());

        // The headroom of buf is limited, so the merge might write past its
        // end.
        if len + got > buf.len() {
            bin_overflow("the decimation checkpoint accumulator");
        }
        merge_into(buf, len, &aux[..got]);
        len += got;
        eprint!("[{:.3}s] count...", sw.lap());

        c = count_adjacent_equals(&buf[..len]);
        let null = Null::collisions(len as f64, effective_cells);
        eprintln!(
            "[{:.3}s], {len} points, {} collisions\tp={}",
            sw.lap(),
            c,
            format_p_value(p_value(c as f64, null), pretty_p)
        );
    }
    (c, len)
}

/// Runs a collision test in parallel.
///
/// The samples of each pass are split into `num_cpus` contiguous ranges, and
/// each thread writes the points of its range into a disjoint region of a
/// shared buffer. Threads reach the start of their range by jump-ahead
/// (`try_skip`) or, for generators that cannot jump, by a sequential pre-scan;
/// repetitions continue the same orbit. The result is thus the same as that of
/// [`run_test`]. Without tradeoff there is a single pass; with tradeoff there
/// are 2ᵇ passes, as in [`run_collision_tradeoff`].
///
/// After each pass, the regions are compacted, and the whole buffer is sorted
/// and scanned to count collisions, so no merge is necessary. Only one pass is
/// in memory at any time, as in the sequential case.
///
/// Differently from the sequential runners called by [`run_test`], this
/// function runs all repetitions, prints the header, and selects modes at run
/// time rather than using const generics.
///
/// Returns the total number of collisions and the sum of the null
/// distributions of the repetitions, each conditioned on the number of points
/// actually kept.
///
/// [`run_test`]: crate::common::run_test
pub fn run_test_parallel<T: Cell>(
    args: &Args,
    points: usize,
    cells: &BigUint,
    lambda: f64,
    num_cpus: usize,
) -> (u128, Null) {
    let seed = args.seed;
    eprintln!("Seed: {:#018x}", seed);

    let d = args.decimation_bits.unwrap_or(0);
    let tradeoff_b = args.tradeoff_bits();
    let num_passes: u64 = 1u64 << tradeoff_b; // tradeoff passes (1 when none)
    // Decimation and tradeoff select points using t · d + b bits.
    let partition_bits = args.t * d + tradeoff_b;

    let output_type = bits_read_desc(args.s);

    let mut mode_parts: Vec<String> = Vec::new();
    if tradeoff_b > 0 {
        mode_parts.push(format!(
            "tradeoff on {} top bits over {} passes",
            tradeoff_b, num_passes
        ));
    }
    if d > 0 {
        mode_parts.push(decimation_desc(d, args.t));
    }
    let mode_suffix = join_mode_parts(&mode_parts);

    let decimating = d > 0;

    // Each pass scans points · 2ᵗᵈ samples, split among threads into contiguous
    // ranges (see OrbitPartition).
    let scan_total = scan_samples(points, args.t, d);
    let mut partition = OrbitPartition::new(seed, num_cpus, scan_total, args.t);
    let num_cpus = partition.num_cpus;
    let base_chunk = partition.base_chunk;
    let rem = partition.rem;
    let chunk = |i: usize| base_chunk + if i < rem { 1 } else { 0 };
    let buf_len = |i: usize| buffer_size(chunk(i), partition_bits);
    let total_buf: usize = (0..num_cpus).map(buf_len).sum();
    let split_desc = partition.split_desc();
    // The buffer holds one bin of the scan, as in run_test, but the headroom is
    // larger because each thread region has its own, and smaller bins need
    // proportionally more headroom.
    let headroom_suffix = headroom_desc(
        total_buf,
        (scan_total as f64) / 2.0f64.powi(partition_bits as i32),
        partition_bits,
    );
    eprintln!(
        "Running a {}-dimensional collision test {} on the upper {} bits of the {} \
         ({} points, {}-bit cells, {} memory locations, {:.3} GiB RAM{}{})",
        args.t,
        generation_desc(num_cpus, split_desc),
        args.u,
        output_type,
        points,
        size_of::<T>() * 8,
        points >> tradeoff_b,
        // In floating point, as the size in bytes might overflow a usize (in
        // which case alloc_mmap will fail later).
        total_buf as f64 * size_of::<T>() as f64 / 2.0f64.powi(30),
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
        null_desc(points, cells.to_f64().unwrap()),
        cells_suffix
    );

    let full = args.u == 64 && args.s == 0;

    let params = GridParams {
        u: args.u,
        t: args.t,
        s: args.s,
        d,
        cells,
    };

    let mut sw = Stopwatch::new();
    let mut tot: u128 = 0;
    let mut null_sum = Null::ZERO;
    let cells_f64 = cells.to_f64().unwrap();
    // Cells covered by a pass, for the null distributions of passes.
    let cells_per_pass = cells_f64 / num_passes as f64;

    // Checkpoints require decimation and exclude tradeoff, so there is a single
    // pass. The samples of each stage are split among threads, and the sorted
    // points of the stage are merged into a sorted accumulator.
    if args.checkpoints {
        let effective_cells_f64 = cells.to_f64().unwrap();
        // ⌊√2ᵗᵈ⌋ stages, computed as in run_collision_decimate.
        let num_checkpoints =
            (2.0f64.powf((args.t * d) as f64 / 2.0) as usize).clamp(1, scan_total);
        let acc_cap = buffer_size(scan_total, partition_bits);
        let max_stage = scan_total.div_ceil(num_checkpoints);
        let thread_cap = buffer_size(max_stage.div_ceil(num_cpus) + 1, partition_bits).max(1);
        // All thread regions in the stage buffer have the same capacity.
        let stage_caps: Box<[usize]> = vec![thread_cap; num_cpus].into_boxed_slice();
        let stage_buf_len = thread_cap * num_cpus;
        for rep in 1..=args.reps {
            let mut acc = alloc_mmap::<T>(acc_cap);
            let mut acc_len = 0usize;
            let mut stage_mmap = alloc_mmap::<T>(stage_buf_len);
            let mut scanned = 0usize;
            let mut c = 0usize;
            for k in 1..=num_checkpoints {
                let mut psw = Stopwatch::new();
                eprint!("Checkpoint {}/{}: gen...", k, num_checkpoints);
                let target_scanned = scan_total * k / num_checkpoints;
                let stage = target_scanned - scanned;
                // There cannot be more threads than samples, as boundaries
                // must be smaller than the stage length.
                let stage_cpus = num_cpus.min(stage).max(1);
                let sbase = stage / stage_cpus;
                let srem = stage % stage_cpus;
                let schunk = |i: usize| sbase + if i < srem { 1 } else { 0 };
                let sstart = |i: usize| i * sbase + i.min(srem);

                // Per-thread orbit starts within this stage's sample-range.
                let boundaries: Box<[usize]> = (0..stage_cpus).map(sstart).collect();
                let snapshots =
                    partition.snapshots((rep - 1) * scan_total + scanned, stage, &boundaries, None);

                let stage_buf: &mut [T] = bytemuck::try_cast_slice_mut(&mut stage_mmap).unwrap();

                // Phase 1: generate the points of the stage into a contiguous
                // buffer.
                let stage_len = gen_unit_contiguous::<T>(
                    stage_buf,
                    &stage_caps[..stage_cpus],
                    &snapshots,
                    &params,
                    &schunk,
                    0,
                    0,
                    true,
                    full,
                );
                eprint!("[{:.3}s] sort...", psw.lap());

                // Phase 2: sort the contiguous stage run.
                T::sort_mt(&mut stage_buf[..stage_len]);
                eprint!("[{:.3}s] merge...", psw.lap());

                // Merge the stage into the sorted accumulator, so that each
                // checkpoint requires just a linear scan.
                let acc_slice: &mut [T] = bytemuck::try_cast_slice_mut(&mut acc).unwrap();
                // The headroom of the accumulator is limited, so the merge
                // might write past its end.
                if acc_len + stage_len > acc_slice.len() {
                    bin_overflow("the checkpoint accumulator");
                }
                merge_into(acc_slice, acc_len, &stage_buf[..stage_len]);
                acc_len += stage_len;
                scanned = target_scanned;
                eprint!("[{:.3}s] count...", psw.lap());

                c = count_adjacent_equals(&acc_slice[..acc_len]);
                let null_cp = Null::collisions(acc_len as f64, effective_cells_f64);
                eprintln!(
                    "[{:.3}s], {acc_len} points, {c} collisions\tp={}",
                    psw.lap(),
                    format_p_value(p_value(c as f64, null_cp), args.pretty_p),
                );
            }
            tot += c as u128;
            // Condition the null distribution on the points actually kept.
            let null_rep = test_null(acc_len, effective_cells_f64, false);
            null_sum += null_rep;
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
        eprintln!("Test completed in {:.2} seconds", sw.lap());
        return (tot, null_sum);
    }

    // The capacities of the thread regions of the buffer.
    let caps: Box<[usize]> = (0..num_cpus).map(buf_len).collect();

    for rep in 1..=args.reps {
        // A buffer of total_buf = Σ caps elements, reused by all passes.
        let mut buf_mmap = alloc_mmap::<T>(total_buf);

        // Per-thread orbit starts for this rep (jump-ahead or chained pre-scan).
        let snapshots = partition.rep_snapshots(rep);

        let mut rep_coll = 0usize;
        let mut total_points = 0usize;
        // With --pass K, only pass K runs (counts of different passes can be
        // summed).
        let (pass_lo, pass_hi) = match args.pass {
            Some(k) => (k, k + 1),
            None => (0, num_passes),
        };
        // Cumulative statistics are printed only if more than one pass runs.
        let multi_pass = pass_hi - pass_lo > 1;
        for pass in pass_lo..pass_hi {
            // The phases are timed separately, as in run_collision_tradeoff.
            let mut psw = Stopwatch::new();
            eprint!("Pass {}/{}: gen...", pass + 1, num_passes);

            let buf: &mut [T] = bytemuck::try_cast_slice_mut(&mut buf_mmap).unwrap();

            // Phase 1: generate the points of the pass into a contiguous
            // buffer.
            let pass_points = gen_unit_contiguous::<T>(
                buf, &caps, &snapshots, &params, &chunk, pass, tradeoff_b, decimating, full,
            );
            eprint!("[{:.3}s] sort...", psw.lap());

            // Phase 2: one sort over the whole contiguous unit.
            T::sort_mt(&mut buf[..pass_points]);
            eprint!("[{:.3}s] count...", psw.lap());

            // Phase 3: count; since all points are sorted together, collisions
            // between points generated by different threads are counted, too.
            let c = count_adjacent_equals(&buf[..pass_points]);

            // Statistics are printed as in run_collision_tradeoff.
            total_points += pass_points;
            rep_coll += c;
            let null_pass = Null::collisions(pass_points as f64, cells_per_pass);
            let elapsed = psw.lap();
            let local_p = format_p_value(p_value(c as f64, null_pass), args.pretty_p);
            if multi_pass {
                let null_so_far =
                    Null::collisions(total_points as f64, (pass + 1) as f64 * cells_per_pass);
                eprintln!(
                    "[{elapsed:1.3}s], {pass_points} points, {c} collisions, p={local_p}; combined: {total_points} points, {rep_coll} collisions, p={}",
                    format_p_value(p_value(rep_coll as f64, null_so_far), args.pretty_p),
                );
            } else {
                eprintln!("[{elapsed:1.3}s], {pass_points} points, {c} collisions, p={local_p}");
            }
        }

        tot += rep_coll as u128;
        // Condition the null distribution on the points actually kept.
        let null_rep = test_null(total_points, cells_f64, false);
        null_sum += null_rep;
        if args.pass.is_some() {
            // With --pass, total_points is the number of points of one pass, so
            // a p-value would be meaningless: main prints the count and the
            // share of the null distribution.
            if args.reps > 1 {
                eprintln!("{rep_coll}\tcombined: {tot}");
            } else {
                eprintln!("{rep_coll}");
            }
        } else {
            let rep_p = format_p_value(p_value(rep_coll as f64, null_rep), args.pretty_p);
            if args.reps > 1 {
                eprintln!(
                    "{rep_coll}\tp={rep_p}\tcombined: {tot}\tp={}",
                    format_p_value(p_value(tot as f64, null_sum), args.pretty_p)
                );
            } else {
                eprintln!("{rep_coll}\tp={rep_p}");
            }
        }
    }
    eprintln!("Test completed in {:.2} seconds", sw.lap());
    (tot, null_sum)
}
