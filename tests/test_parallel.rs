/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

//! Tests checking that parallel runs and single passes give the same results
//! as sequential runs.
//!
//! Depending on the generator, the tests exercise jump-ahead or pre-scan. The
//! incr generator is excluded, as buffers are sized assuming that points are
//! spread evenly among bins.
#![cfg(not(feature = "incr"))]

use num::BigUint;
use num::traits::ToPrimitive;

use coll_birth::birthday::run_birthday_parallel;
use coll_birth::cli::Args;
use coll_birth::collision::run_test_parallel;
use coll_birth::common::{compute_lambda_and_points, run_test, test_null};

fn make_args(u: usize, t: usize, m: usize, tradeoff: Option<usize>, seed: u64) -> Args {
    Args {
        u,
        t,
        m: Some(m),
        s: 0,
        tradeoff_bits: tradeoff,
        decimation_bits: None,
        checkpoints: false,
        birthday_spacings: false,
        reps: 1,
        seed,
        pretty_p: false,
        parallel: true,
        pass: None,
    }
}

fn cells_for(args: &Args) -> BigUint {
    BigUint::from(2u32).pow(args.u as u32).pow(args.t as u32)
}

// The counts of the 2ᵇ single passes (--pass K) must sum to the count of a
// full run, both sequentially and in parallel.
#[test]
fn test_single_pass_collision_sum_matches_full() {
    let seed = 0xABCD_1234_5678_9ABC;
    let b = 2u32;
    let mut args = make_args(16, 2, 1 << 16, Some(b as usize), seed);
    let cells = cells_for(&args);
    let (lambda, points) = compute_lambda_and_points(&args, &cells);

    let (full_seq, _) = run_test::<u64>(&args, points, &cells, lambda);
    let (full_par, _) = run_test_parallel::<u64>(&args, points, &cells, lambda, 4);

    let mut sum_seq = 0u128;
    let mut sum_par = 0u128;
    for k in 0..(1u64 << b) {
        args.pass = Some(k);
        sum_seq += run_test::<u64>(&args, points, &cells, lambda).0;
        sum_par += run_test_parallel::<u64>(&args, points, &cells, lambda, 4).0;
    }
    assert_eq!(
        full_seq, sum_seq,
        "sequential single-pass counts must sum to full"
    );
    assert_eq!(
        full_par, sum_par,
        "parallel single-pass counts must sum to full"
    );

    // The shares of the null distribution must sum exactly to the full one
    // (division by a power of two is exact).
    let num_passes = 1u64 << b;
    let lambda_k = test_null(points, cells.to_f64().unwrap(), false) / num_passes as f64;
    assert_eq!(
        num_passes as f64 * lambda_k,
        test_null(points, cells.to_f64().unwrap(), false),
        "single-pass null shares must sum to the full null distribution"
    );
}

// Same, for birthday spacings.
#[test]
fn test_single_pass_birthday_sum_matches_full() {
    let seed = 0x0BAD_F00D_DEAD_BEEF;
    let b = 2u32;
    let mut args = make_args(30, 2, 1 << 16, Some(b as usize), seed);
    args.birthday_spacings = true;
    let cells = cells_for(&args);
    let (lambda, points) = compute_lambda_and_points(&args, &cells);

    let (full_seq, _) = run_test::<u64>(&args, points, &cells, lambda);
    let (full_par, _) = run_birthday_parallel::<u64>(&args, points, &cells, lambda, 4);

    let mut sum_seq = 0u128;
    let mut sum_par = 0u128;
    for k in 0..(1u64 << b) {
        args.pass = Some(k);
        sum_seq += run_test::<u64>(&args, points, &cells, lambda).0;
        sum_par += run_birthday_parallel::<u64>(&args, points, &cells, lambda, 4).0;
    }
    assert_eq!(
        full_seq, sum_seq,
        "sequential single-pass birthday counts must sum to full"
    );
    assert_eq!(
        full_par, sum_par,
        "parallel single-pass birthday counts must sum to full"
    );
}

// A parallel run must give the same result as a sequential run.
#[test]
fn test_faithful_plain_matches_sequential() {
    let seed = 0x00C0_FFEE_1234_5678;
    let args = make_args(12, 2, 1 << 18, None, seed);
    let cells = cells_for(&args);
    let (lambda, points) = compute_lambda_and_points(&args, &cells);
    let seq = run_test::<u64>(&args, points, &cells, lambda);
    let par = run_test_parallel::<u64>(&args, points, &cells, lambda, 4);
    assert_eq!(
        seq, par,
        "faithful plain parallel must equal the sequential run"
    );
}

// Same, with tradeoff.
#[test]
fn test_faithful_tradeoff_matches_sequential() {
    let seed = 0x0D15_EA5E_0BAD_F00D;
    let args = make_args(12, 2, 1 << 14, Some(2), seed);
    let cells = cells_for(&args);
    let (lambda, points) = compute_lambda_and_points(&args, &cells);
    let seq = run_test::<u64>(&args, points, &cells, lambda);
    let par = run_test_parallel::<u64>(&args, points, &cells, lambda, 3);
    assert_eq!(
        seq, par,
        "faithful tradeoff parallel must equal the sequential run"
    );
}

// Same, with decimation.
#[test]
fn test_faithful_decimation_matches_sequential() {
    let seed = 0x0DEC_1A7E_0000_0001;
    let mut args = make_args(14, 2, 1 << 14, None, seed);
    args.decimation_bits = Some(2);
    let cells = BigUint::from(2u32)
        .pow((args.u - 2) as u32)
        .pow(args.t as u32);
    let (lambda, points) = compute_lambda_and_points(&args, &cells);
    let seq = run_test::<u64>(&args, points, &cells, lambda);
    let par = run_test_parallel::<u64>(&args, points, &cells, lambda, 4);
    assert_eq!(
        seq, par,
        "faithful parallel decimation must equal sequential"
    );
}

// Same, with decimation and tradeoff.
#[test]
fn test_faithful_decimation_tradeoff_matches_sequential() {
    let seed = 0x0DEC_1A7E_0000_0002;
    let mut args = make_args(14, 2, 1 << 12, Some(2), seed);
    args.decimation_bits = Some(2);
    let cells = BigUint::from(2u32)
        .pow((args.u - 2) as u32)
        .pow(args.t as u32);
    let (lambda, points) = compute_lambda_and_points(&args, &cells);
    let seq = run_test::<u64>(&args, points, &cells, lambda);
    let par = run_test_parallel::<u64>(&args, points, &cells, lambda, 3);
    assert_eq!(
        seq, par,
        "faithful parallel decimation+tradeoff must equal sequential"
    );
}

fn checkpoint_args(seed: u64) -> (Args, BigUint) {
    let mut args = make_args(16, 2, 1 << 14, None, seed);
    args.decimation_bits = Some(2);
    args.checkpoints = true;
    let cells = BigUint::from(2u32)
        .pow((args.u - 2) as u32)
        .pow(args.t as u32);
    (args, cells)
}

// Parallel checkpoints must give the same result for any number of threads.
#[test]
fn test_parallel_checkpoints_match_across_cpus() {
    let (args, cells) = checkpoint_args(0x0C0C_0C0C_0000_0001);
    let (lambda, points) = compute_lambda_and_points(&args, &cells);
    let r1 = run_test_parallel::<u64>(&args, points, &cells, lambda, 1);
    let r4 = run_test_parallel::<u64>(&args, points, &cells, lambda, 4);
    assert_eq!(r1, r4, "parallel checkpoints must match across CPU counts");
}

// Parallel checkpoints with one thread must give the same result as
// sequential checkpoints.
#[test]
fn test_parallel_checkpoints_p1_match_sequential_runner() {
    let (args, cells) = checkpoint_args(0x0C0C_0C0C_0000_0002);
    let (lambda, points) = compute_lambda_and_points(&args, &cells);
    let seq = run_test::<u64>(&args, points, &cells, lambda);
    let par1 = run_test_parallel::<u64>(&args, points, &cells, lambda, 1);
    assert_eq!(
        seq, par1,
        "P=1 checkpoints must equal the sequential runner"
    );
}

// A parallel birthday-spacings test must give the same result as a sequential
// one.
#[test]
fn test_faithful_birthday_plain_matches_sequential() {
    let seed = 0x0B17_4DA9_0000_0001;
    let mut args = make_args(20, 2, 40_000, None, seed);
    args.birthday_spacings = true;
    let cells = cells_for(&args); // 2⁴⁰
    let (lambda, points) = compute_lambda_and_points(&args, &cells);
    let seq = run_test::<u64>(&args, points, &cells, lambda);
    let par1 = run_birthday_parallel::<u64>(&args, points, &cells, lambda, 1);
    let par3 = run_birthday_parallel::<u64>(&args, points, &cells, lambda, 3);
    assert_eq!(seq, par1, "P=1 parallel birthday must equal sequential");
    assert_eq!(seq, par3, "P=3 parallel birthday must equal sequential");
}

// Same, with decimation. With this seed 40196 > 40000 points are kept, so the
// class buffer needs headroom even if b = 0.
//
// The 32-bit LCGs are excluded: their low bits are grossly non-uniform (bit i
// has period 2ⁱ⁺¹), and decimation keys on low bits, so the class buffer
// overflows, and both runs abort.
#[cfg(not(any(feature = "lcg_32_32_0xec65035", feature = "lcg_32_32_0x915f77f5")))]
#[test]
fn test_faithful_birthday_decimation_matches_sequential() {
    let seed = 3;
    let mut args = make_args(30, 2, 40_000, None, seed);
    args.birthday_spacings = true;
    args.decimation_bits = Some(2);
    let cells = BigUint::from(2u32)
        .pow((args.u - 2) as u32)
        .pow(args.t as u32); // effective cells 2^((u−d)·t)
    let (lambda, points) = compute_lambda_and_points(&args, &cells);
    let seq = run_test::<u64>(&args, points, &cells, lambda);
    let par1 = run_birthday_parallel::<u64>(&args, points, &cells, lambda, 1);
    let par3 = run_birthday_parallel::<u64>(&args, points, &cells, lambda, 3);
    assert_eq!(
        seq, par1,
        "P=1 parallel birthday decimation must equal sequential"
    );
    assert_eq!(
        seq, par3,
        "P=3 parallel birthday decimation must equal sequential"
    );
}

// Same, with tradeoff.
#[test]
fn test_faithful_birthday_tradeoff_matches_sequential() {
    let seed = 0x0B17_4DA9_0000_0002;
    let mut args = make_args(20, 2, 10_000, Some(2), seed);
    args.birthday_spacings = true;
    let cells = cells_for(&args); // 2⁴⁰
    let (lambda, points) = compute_lambda_and_points(&args, &cells); // points = 10000 · 4
    let seq = run_test::<u64>(&args, points, &cells, lambda);
    let par1 = run_birthday_parallel::<u64>(&args, points, &cells, lambda, 1);
    let par3 = run_birthday_parallel::<u64>(&args, points, &cells, lambda, 3);
    assert_eq!(
        seq, par1,
        "P=1 parallel birthday tradeoff must equal sequential"
    );
    assert_eq!(
        seq, par3,
        "P=3 parallel birthday tradeoff must equal sequential"
    );
}

// Same, with tradeoff and 2³² cells in u32 storage (the wrap-around spacing is
// computed through cells − 1).
#[test]
fn test_faithful_birthday_boundary_u32_matches_sequential() {
    let seed = 0x0B17_4DA9_0000_0003;
    let mut args = make_args(16, 2, 12_500, Some(2), seed);
    args.birthday_spacings = true;
    let cells = cells_for(&args); // exactly 2^32
    let points = 50_000usize; // m · 2ᵇ
    let lambda = (points as f64).powi(3) / (4.0 * cells.to_f64().unwrap());
    let seq = run_test::<u32>(&args, points, &cells, lambda);
    let par3 = run_birthday_parallel::<u32>(&args, points, &cells, lambda, 3);
    assert_eq!(
        seq, par3,
        "boundary-width parallel birthday must equal sequential"
    );
}
