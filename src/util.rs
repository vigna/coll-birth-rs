/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

//! Utilities.

use std::fmt::Display;
use std::time::Instant;

/// Returns the representation of an integer using Unicode superscripts.
///
/// For example, `superscript(64)` returns `"⁶⁴"`. Digits and `-` are replaced
/// by their superscript counterparts; other characters are left unchanged.
pub fn superscript(n: impl Display) -> String {
    n.to_string()
        .chars()
        .map(|c| match c {
            '0' => '⁰',
            '1' => '¹',
            '2' => '²',
            '3' => '³',
            '4' => '⁴',
            '5' => '⁵',
            '6' => '⁶',
            '7' => '⁷',
            '8' => '⁸',
            '9' => '⁹',
            '-' => '⁻',
            other => other,
        })
        .collect()
}

/// Returns the number of threads of the Rayon global thread pool.
///
/// All parallel phases (generation, sorting, and counting) use this number of
/// threads, so they all honor `RAYON_NUM_THREADS` (by default, the number of
/// available cores).
pub fn parallelism() -> usize {
    rayon::current_num_threads()
}

/// A stopwatch measuring the time elapsed between successive calls to
/// [`Stopwatch::lap`].
pub struct Stopwatch(Instant);

impl Stopwatch {
    pub fn new() -> Self {
        Self(Instant::now())
    }

    /// Returns the seconds elapsed since the previous lap (or since creation),
    /// and starts a new lap.
    pub fn lap(&mut self) -> f64 {
        let elapsed = self.0.elapsed().as_secs_f64();
        self.0 = Instant::now();
        elapsed
    }
}

impl Default for Stopwatch {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_superscript() -> anyhow::Result<()> {
        assert_eq!(superscript(0u32), "⁰");
        assert_eq!(superscript(64usize), "⁶⁴");
        assert_eq!(superscript(1234567890u64), "¹²³⁴⁵⁶⁷⁸⁹⁰");
        assert_eq!(superscript(-5i32), "⁻⁵");
        Ok(())
    }
}
