/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

//! Statistical helpers: expected collision count, adjusted *p*-values, and formatting.

use std::borrow::Cow;

use std::ops::{Add, AddAssign, Div, Mul};

use cdflib::special::try_beta_inc;
use cdflib::traits::{ContinuousCdf, DiscreteCdf};
use cdflib::{Normal, Poisson};

/// Variance-to-mean ratio at or above which we use the Poisson approximation
/// (see [`Null::kind`]).
pub const POISSON_VARIANCE_RATIO: f64 = 0.99;

/// Variance-to-mean ratio below which we use the normal approximation (see
/// [`Null::kind`]).
pub const NORMAL_VARIANCE_RATIO: f64 = 0.54;

/// Density below which we use the Poisson approximation without computing the
/// variance, which would lose precision.
const POISSON_DENSITY: f64 = 0.001;

/// The density that asymptotically maximizes the variance of the number of
/// collisions (the root of e^α = 1 + 2α), used by default.
pub const VARIANCE_MAXIMIZING_DENSITY: f64 = 1.256431;

/// The maximum density of a collision test.
pub const MAX_DENSITY: f64 = 1.26;

/// An approximating distribution (see [`Null::kind`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NullKind {
    /// Poisson with the given mean.
    Poisson,
    /// Binomial with the given mean and variance.
    Binomial,
    /// Normal with the given mean and variance.
    Normal,
}

/// The null distribution of a collision (or birthday-spacings) count, given by
/// its mean and variance.
///
/// Null distributions of independent tests can be added, and scaled to obtain
/// per-pass shares: the approximating distribution is chosen only when
/// computing tails (see [`Null::kind`]).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Null {
    /// The expected count.
    pub mean: f64,
    /// The variance of the count (equal to the mean in the Poisson case).
    pub var: f64,
}

impl Null {
    /// The null distribution of an empty test.
    pub const ZERO: Null = Null {
        mean: 0.0,
        var: 0.0,
    };

    /// A Poisson null distribution with the given mean.
    pub fn poisson(mean: f64) -> Self {
        Null { mean, var: mean }
    }

    /// The null distribution of the number of collisions of `points` balls
    /// thrown into `cells` bins.
    ///
    /// If the variance is at least [`POISSON_VARIANCE_RATIO`] times the mean,
    /// it is replaced by the mean, yielding a Poisson null distribution.
    pub fn collisions(points: f64, cells: f64) -> Self {
        let mean = expected_collisions(points, cells);
        if points / cells < POISSON_DENSITY {
            return Null::poisson(mean);
        }
        let var = collision_variance(points, cells);
        if var >= POISSON_VARIANCE_RATIO * mean {
            Null::poisson(mean)
        } else {
            Null { mean, var }
        }
    }

    /// Returns the approximating distribution: Poisson if the variance is at
    /// least [`POISSON_VARIANCE_RATIO`] times the mean, a binomial distribution
    /// with the same mean and variance if it is at least
    /// [`NORMAL_VARIANCE_RATIO`] times the mean, and normal otherwise.
    ///
    /// # Implementation Details
    ///
    /// The variance of the number of collisions is about 1 − 4α/3 times its
    /// mean when the density α is small, and 0.26 times its mean at α = 1. The
    /// Poisson approximation thus overestimates tails: at six standard
    /// deviations, by 20% at ratio 0.99, by a factor of 10 at α = 0.1, and by a
    /// factor of 10⁶ at α = 1. TestU01 uses it up to density 1.0001.
    ///
    /// The normal approximation ignores the skewness of the count: when the
    /// mean is small, at six standard deviations it underestimates the right
    /// tail by a factor of up to 50.
    ///
    /// The binomial approximation is within 0.5% of the exact distribution up
    /// to density 0.05. As the ratio decreases it deteriorates, whereas the
    /// normal approximation improves: they have the same error at ratio 0.54,
    /// independently of the number of points. There, for 10⁴ points, tails at
    /// six standard deviations are within 25% of the exact ones.
    pub fn kind(&self) -> NullKind {
        // Degenerate values (zero or NaN) fall through to the Poisson case.
        if self.var < NORMAL_VARIANCE_RATIO * self.mean {
            NullKind::Normal
        } else if self.var < POISSON_VARIANCE_RATIO * self.mean {
            NullKind::Binomial
        } else {
            NullKind::Poisson
        }
    }
}

impl Add for Null {
    type Output = Null;
    fn add(self, other: Null) -> Null {
        Null {
            mean: self.mean + other.mean,
            var: self.var + other.var,
        }
    }
}

impl AddAssign for Null {
    fn add_assign(&mut self, other: Null) {
        *self = *self + other;
    }
}

impl Mul<Null> for f64 {
    type Output = Null;
    fn mul(self, null: Null) -> Null {
        Null {
            mean: self * null.mean,
            var: self * null.var,
        }
    }
}

impl Div<f64> for Null {
    type Output = Null;
    fn div(self, d: f64) -> Null {
        Null {
            mean: self.mean / d,
            var: self.var / d,
        }
    }
}

/// Lower and upper Poisson tail probabilities at a given count.
#[derive(Clone, Copy, Debug)]
pub struct PoissonTails {
    /// `Pr[X <= coll]`
    pub p_left: f64,
    /// `Pr[X >= coll]`
    pub p_right: f64,
}

/// Computes the Poisson lower and upper tail probabilities at `coll` given mean `lambda`.
///
/// Returns `None` when `lambda` is not a valid Poisson rate or `coll` is not a
/// non-negative integer representable as `u64`.
///
/// # Implementation notes
///
/// [`DiscreteCdf::ccdf`] returns `Pr[X > s]`; computing `Pr[X >= coll]` therefore
/// evaluates the complementary CDF at `coll - 1`. The `coll == 0` case is
/// special-cased to avoid underflowing the `u64` argument.
pub fn poisson_tails(coll: f64, lambda: f64) -> Option<PoissonTails> {
    if !coll.is_finite() || coll < 0.0 || coll.fract() != 0.0 {
        return None;
    }
    let coll_u = coll as u64;

    // `Poisson::new` panics on an invalid rate (negative or non-finite); the
    // fallible constructor lets us honour the documented `None` contract instead.
    let poi = Poisson::try_new(lambda).ok()?;
    let p_left = poi.cdf(coll_u);
    let p_right = if coll_u == 0 {
        1.0
    } else {
        poi.ccdf(coll_u - 1)
    };

    Some(PoissonTails { p_left, p_right })
}

/// Expected number of collisions when throwing `points` balls into `cells` bins.
///
/// For sparse regimes (`points / cells <= 0.1`) the routine evaluates a truncated Maclaurin
/// expansion converging in at most 64 terms; otherwise it falls back to the closed form
/// `points - cells + cells * (1 - 1/cells)^points`, with `(1 - 1/cells)^points` evaluated as
/// `exp(points * log(1 - 1/cells))` and `log(1 - 1/cells)` itself expanded as a Maclaurin
/// series for accuracy when `cells` is large.
pub fn expected_collisions(points: f64, cells: f64) -> f64 {
    // Fewer than two points cannot collide; returning early also avoids a 0/0 NaN
    // in the sparse series below (a tradeoff/decimation pass may keep 0 or 1 point).
    if points <= 1.0 {
        return 0.0;
    }
    if points / cells <= 0.1 {
        // Sparse regime: lambda = sum_{i>=2} (-1)^i * C(points, i) * cells^(1-i).
        let mut u = points - 1.0;
        let mut v = 2.0;
        let mut t = (points * u) / (2.0 * cells);
        let mut lambda = t;

        let mut i = 3;
        while (t / lambda).abs() > f64::EPSILON && i < 64 {
            u -= 1.0;
            v += 1.0;
            t = -t * u / (cells * v);
            lambda += t;
            i += 1;
        }
        debug_assert!((t / lambda).abs() <= f64::EPSILON);
        lambda
    } else {
        // Dense regime: lambda = points - cells + cells * (1 - 1/cells)^points, with the
        // power evaluated as exp(-points · neg_log) where neg_log = -log(1 - 1/cells) is
        // computed by its Maclaurin series for numerical stability when cells is large.
        let mut t = 1.0 / cells;
        let mut neg_log = t;
        for i in 2..10 {
            t *= 1.0 / cells;
            neg_log += t / i as f64;
        }
        (points - cells) + cells * f64::exp(-(points * neg_log))
    }
}

/// Returns the variance of the number of collisions of `points` balls thrown
/// into `cells` bins, that is, the variance of the number of empty bins,
/// *k*(*k* − 1)*b* + *ka* − *k*²*a*², where *a* = (1 − 1/*k*)*ⁿ* and *b* = (1 −
/// 2/*k*)*ⁿ*.
///
/// # Implementation Details
///
/// To avoid cancellation, we compute *k*(*a* − *b*) + *k*²(*b* − *a*²) using
/// *b*/*a* = (1 − 1/(*k* − 1))*ⁿ* and *b*/*a*² = (1 − 1/(*k* − 1)²)*ⁿ*, through
/// `ln_1p` and `exp_m1`.
pub fn collision_variance(points: f64, cells: f64) -> f64 {
    if points <= 1.0 {
        return 0.0;
    }
    let a = (points * (-1.0 / cells).ln_1p()).exp();
    let km1 = cells - 1.0;
    // a − b = −a · expm1(n · ln(1 − 1/(k − 1)))
    let a_minus_b = -a * (points * (-1.0 / km1).ln_1p()).exp_m1();
    // b − a² = a² · expm1(n · ln(1 − 1/(k − 1)²))
    let b_minus_a2 = a * a * (points * (-1.0 / (km1 * km1)).ln_1p()).exp_m1();
    (cells * a_minus_b + cells * cells * b_minus_a2).max(0.0)
}

/// A TestU01-style two-sided adjusted *p*-value, kept in a form that survives the
/// approach to 1.
///
/// A left-tail anomaly (too few collisions) has *p*-value `1 − p_left` with
/// `p_left = Pr[X ≤ coll]` astronomically small. Storing that as a single `f64`
/// would round it to exactly `1.0` as soon as `p_left` drops below the machine
/// epsilon (~1.1·10⁻¹⁶), discarding the tail. We therefore keep `p_left` itself
/// in [`PValue::NearOne`] and defer the lossy `1 − ε` subtraction to formatting,
/// where it is only performed in the plain (non-pretty) rendering.
#[derive(Clone, Copy, Debug)]
pub enum PValue {
    /// The *p*-value, stored directly (near 0 or in the middle of the range; also
    /// carries `f64::NAN` when CDFLIB reports an error).
    Direct(f64),
    /// A near-1 *p*-value equal to `1 − eps`, with `eps = p_left` retained at full
    /// precision so that pretty mode can print it as `1 − eps`.
    NearOne(f64),
}

/// Returns Pr[*X* ≤ *c*] and Pr[*X* ≥ *c*], where *c* = `coll`, for a binomial
/// distribution with `n` trials (possibly not an integer) and success
/// probability `q`; `q_c` = 1 − `q` is passed separately to avoid cancellation.
///
/// The tails are *I*₁₋_q(*n* − *c*, *c* + 1) and *I*_q(*c*, *n* − *c* + 1), where
/// *I* is the regularized incomplete beta function.
fn binomial_tails(coll: f64, n: f64, q: f64, q_c: f64) -> Option<(f64, f64)> {
    let p_left = if n - coll <= 0.0 {
        1.0
    } else {
        try_beta_inc(n - coll, coll + 1.0, q_c, q).ok()?.0
    };
    let p_right = if coll == 0.0 {
        1.0
    } else if n - coll + 1.0 <= 0.0 {
        0.0
    } else {
        try_beta_inc(coll, n - coll + 1.0, q, q_c).ok()?.0
    };
    Some((p_left, p_right))
}

/// Returns Pr[*X* ≤ *c*] and Pr[*X* ≥ *c*], where *c* = `coll`, for the
/// approximating distribution of `null` (see [`Null::kind`]), or `None` if
/// `coll` is not a nonnegative integer or CDFLIB reports an error.
///
/// The binomial distribution has success probability *q* = 1 − σ²/λ and λ/*q*
/// trials; the normal tails are, as in TestU01, Φ((*c* − λ)/σ) and 1 − Φ((*c* −
/// 1 − λ)/σ).
pub fn null_tails(coll: f64, null: Null) -> Option<(f64, f64)> {
    if !coll.is_finite() || coll < 0.0 || coll.fract() != 0.0 {
        return None;
    }
    match null.kind() {
        NullKind::Poisson => {
            let tails = poisson_tails(coll, null.mean)?;
            Some((tails.p_left, tails.p_right))
        }
        NullKind::Binomial => {
            // q and 1 − q, computed without cancellation.
            let q = (null.mean - null.var) / null.mean;
            let q_c = null.var / null.mean;
            binomial_tails(coll, null.mean / q, q, q_c)
        }
        NullKind::Normal => {
            let normal = Normal::try_new(null.mean, null.var.sqrt()).ok()?;
            Some((normal.cdf(coll), normal.ccdf(coll - 1.0)))
        }
    }
}

/// Returns the TestU01-style two-sided adjusted *p*-value of `coll` under
/// `null` (see Chapter 3 of the long TestU01 guide), or
/// [`PValue::Direct`]`(f64::NAN)` if CDFLIB reports an error.
pub fn p_value(coll: f64, null: Null) -> PValue {
    let Some((p_left, p_right)) = null_tails(coll, null) else {
        return PValue::Direct(f64::NAN);
    };
    if p_right < p_left {
        // Right-tail anomaly (too many collisions): the p-value is the small
        // p_right, representable directly.
        PValue::Direct(p_right)
    } else if p_left < 0.5 {
        // Left-tail anomaly (too few collisions): the p-value is 1 − p_left.
        // Keep p_left (the small quantity) rather than the cancelled difference.
        PValue::NearOne(p_left)
    } else {
        PValue::Direct(0.5)
    }
}

/// Formats a *p*-value, optionally rendering values close to 1 as `1 − ε` for
/// readability (and at full precision, however small ε is).
pub fn format_p_value(p: PValue, pretty_p: bool) -> Cow<'static, str> {
    match p {
        PValue::Direct(v) => {
            if v == 0.0 {
                "0".into()
            } else if v == 1.0 {
                "1".into()
            } else {
                format!("{v:?}").into()
            }
        }
        // Value is 1 − eps. When eps underflowed to 0 the left tail is below the
        // f64 range, yet the count is still a left-tail anomaly: print a sentinel
        // just below 1 (`1 − 1e-307`, regardless of pretty mode) rather than a bare
        // "1", which would read as a perfect/degenerate value. Otherwise pretty
        // mode prints the tail exactly, and plain mode falls back to the decimal
        // 1 − eps (which may itself round to "1" for tiny eps).
        PValue::NearOne(eps) => {
            if eps == 0.0 {
                "1 − 1e-307".into()
            } else if pretty_p && eps < 1e-4 {
                format!("1 − {eps:?}").into()
            } else {
                let v = 1.0 - eps;
                if v == 1.0 {
                    "1".into()
                } else {
                    format!("{v:?}").into()
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // X = 0 against mean 700: p_left = e⁻⁷⁰⁰ is tiny but representable, and the
    // p-value must print as `1 − p_left`, not as 1.
    #[test]
    fn test_deep_left_tail_renders_as_one_minus_eps() {
        let s = format_p_value(p_value(0.0, Null::poisson(700.0)), true);
        assert!(
            s.starts_with("1 − ") && s.len() > "1 − ".len(),
            "deep left tail must render as `1 − eps`, got {s:?}"
        );
    }

    // Different deep left tails must print differently.
    #[test]
    fn test_distinct_deep_tails_render_distinctly() {
        let a = format_p_value(p_value(0.0, Null::poisson(700.0)), true);
        let b = format_p_value(p_value(0.0, Null::poisson(600.0)), true);
        assert_ne!(a, b, "different left-tail depths must format differently");
    }

    // When the left tail underflows (e⁻⁷⁵⁰), we print the sentinel `1 − 1e-307`.
    #[test]
    fn test_underflowed_left_tail_prints_sentinel() {
        assert_eq!(
            format_p_value(p_value(0.0, Null::poisson(750.0)), true),
            "1 − 1e-307"
        );
        assert_eq!(
            format_p_value(p_value(0.0, Null::poisson(750.0)), false),
            "1 − 1e-307"
        );
    }

    // A small right-tail p-value prints as a plain decimal.
    #[test]
    fn test_right_tail_small_pvalue_is_plain() {
        let s = format_p_value(p_value(700.0, Null::poisson(1.0)), false);
        assert!(
            !s.starts_with("1-") && s != "1",
            "right-tail anomaly must be a small p-value, got {s:?}"
        );
    }

    // For few cells the textbook formula has no cancellation problems.
    #[test]
    fn test_collision_variance_matches_textbook_formula() {
        for &(n, k) in &[
            (10.0f64, 7.0f64),
            (100.0, 64.0),
            (1000.0, 800.0),
            (5000.0, 4000.0),
        ] {
            let a: f64 = (1.0 - 1.0 / k).powf(n);
            let b: f64 = (1.0 - 2.0 / k).powf(n);
            let direct = k * (k - 1.0) * b + k * a - k * k * a * a;
            let stable = collision_variance(n, k);
            assert!(
                (stable - direct).abs() <= 1e-9 * direct,
                "n = {n}, k = {k}: {stable} vs {direct}"
            );
        }
    }

    // For large k the variance approaches k·e^−α·(1 − (1 + α)·e^−α), α = n/k.
    #[test]
    fn test_collision_variance_asymptotics() {
        let k = 2f64.powi(60);
        for &alpha in &[0.5f64, 1.0, 1.256431] {
            let asym = k * (-alpha).exp() * (1.0 - (1.0 + alpha) * (-alpha).exp());
            let v = collision_variance(alpha * k, k);
            assert!(
                (v - asym).abs() <= 1e-6 * asym,
                "alpha = {alpha}: {v} vs {asym}"
            );
        }
    }

    /// Returns the exact distribution of the number of collisions of `n` balls
    /// thrown into `k` bins as an offset and a vector of probabilities.
    ///
    /// After *i* balls and *c* collisions, *i* − *c* bins are occupied, so the
    /// next ball collides with probability (*i* − *c*)/*k*. Probabilities below
    /// 10⁻⁴⁰ are discarded.
    fn exact_collisions(n: usize, k: f64) -> (usize, Vec<f64>) {
        let (mut lo, mut p) = (0usize, vec![1.0f64]);
        for i in 1..n {
            let mut q = vec![0.0f64; p.len() + 1];
            for (j, &pj) in p.iter().enumerate() {
                let pc = (i - (lo + j)) as f64 / k;
                q[j] += pj * (1.0 - pc);
                q[j + 1] += pj * pc;
            }
            let first = q.iter().position(|&x| x > 1e-40).unwrap();
            let last = q.iter().rposition(|&x| x > 1e-40).unwrap();
            lo += first;
            p = q[first..=last].to_vec();
        }
        (lo, p)
    }

    /// Returns Pr[*C* ≤ *c*] and Pr[*C* ≥ *c*] for an exact distribution.
    fn exact_tails(lo: usize, p: &[f64], c: usize) -> (f64, f64) {
        let idx = c.saturating_sub(lo);
        let left: f64 = if c < lo {
            0.0
        } else {
            p[..=idx.min(p.len() - 1)].iter().rev().sum()
        };
        let right: f64 = if c < lo {
            1.0
        } else if idx >= p.len() {
            0.0
        } else {
            p[idx..].iter().rev().sum()
        };
        (left, right)
    }

    /// Returns the regime and the worst |log₁₀(approximate/exact)| for the
    /// tails at ±2, ±3, ±4, and ±6 standard deviations.
    fn worst_tail_error(n: usize, alpha: f64) -> (NullKind, f64) {
        let k = (n as f64 / alpha).round();
        let null = Null::collisions(n as f64, k);
        let (lo, p) = exact_collisions(n, k);
        // The exact standard deviation, even in the Poisson regime.
        let sd = collision_variance(n as f64, k).sqrt();
        let mut worst = 0.0f64;
        for z in [2.0, 3.0, 4.0, 6.0, -2.0, -3.0, -4.0, -6.0] {
            let x = null.mean + z * sd;
            if x < 0.0 {
                continue;
            }
            let (c, right_tail) = if z > 0.0 {
                (x.ceil(), true)
            } else {
                (x.floor(), false)
            };
            let (el, er) = exact_tails(lo, &p, c as usize);
            let (al, ar) = null_tails(c, null).unwrap();
            let (approx, exact) = if right_tail { (ar, er) } else { (al, el) };
            assert!(
                exact > 0.0 && approx > 0.0,
                "n={n} α={alpha} z={z}: {approx} vs {exact}"
            );
            worst = worst.max((approx / exact).log10().abs());
        }
        (null.kind(), worst)
    }

    // With 10⁴ points (the minimum) all tails out to six standard deviations
    // are within 0.1 decades of the exact ones; the worst cases (about 0.08) are
    // at the boundaries between regimes.
    #[test]
    fn test_null_tails_match_exact_distribution() {
        for &(alpha, expected) in &[
            (0.002, NullKind::Poisson),
            (0.005, NullKind::Poisson),
            (0.0076, NullKind::Binomial),
            (0.02, NullKind::Binomial),
            (0.05, NullKind::Binomial),
            (0.1, NullKind::Binomial),
            (0.2, NullKind::Binomial),
            (0.3, NullKind::Binomial),
            (0.44, NullKind::Binomial),
            (0.48, NullKind::Normal),
            (0.6, NullKind::Normal),
            (0.8, NullKind::Normal),
            (1.0, NullKind::Normal),
            (VARIANCE_MAXIMIZING_DENSITY, NullKind::Normal),
            (MAX_DENSITY, NullKind::Normal),
        ] {
            let (kind, worst) = worst_tail_error(10_000, alpha);
            assert_eq!(kind, expected, "α = {alpha}");
            assert!(
                worst <= 0.1,
                "α = {alpha} ({kind:?}): worst tail error {worst} decades"
            );
        }
    }

    // The error decreases with the number of points.
    #[test]
    fn test_null_tails_improve_with_points() {
        for &alpha in &[0.0076, 0.1, 0.44, 0.48, 1.0] {
            let (_, w1) = worst_tail_error(10_000, alpha);
            let (_, w3) = worst_tail_error(30_000, alpha);
            assert!(
                w3 <= 0.06 && w3 < w1,
                "α = {alpha}: {w3} (n = 3·10⁴) vs {w1} (n = 10⁴)"
            );
        }
    }

    // Regime boundaries.
    #[test]
    fn test_null_kind_boundaries() {
        let at = |ratio: f64| {
            Null {
                mean: 100.0,
                var: 100.0 * ratio,
            }
            .kind()
        };
        assert_eq!(at(1.0), NullKind::Poisson);
        assert_eq!(at(POISSON_VARIANCE_RATIO), NullKind::Poisson);
        assert_eq!(at(POISSON_VARIANCE_RATIO - 1e-9), NullKind::Binomial);
        assert_eq!(at(NORMAL_VARIANCE_RATIO), NullKind::Binomial);
        assert_eq!(at(NORMAL_VARIANCE_RATIO - 1e-9), NullKind::Normal);
        assert_eq!(Null::ZERO.kind(), NullKind::Poisson);
        assert_eq!(Null::poisson(1e-30).kind(), NullKind::Poisson);
        // Sparse collision tests have a Poisson null distribution.
        for &alpha in &[1e-12, 1e-6, 0.0005, 0.005] {
            let k = 2f64.powi(40);
            let null = Null::collisions(alpha * k, k);
            assert_eq!(null.var, null.mean, "α = {alpha}");
        }
    }

    // The p-values of the paper (Section 5 and Table 2).
    #[test]
    fn test_reported_poisson_p_values_unchanged() {
        let k32 = 2f64.powi(32);
        let k64 = 2f64.powi(64);
        let fmt = |c: f64, null: Null| format_p_value(p_value(c, null), true).into_owned();
        assert_eq!(
            fmt(11754.0, Null::collisions(1e7, k32)),
            "0.1310416600017843"
        );
        assert_eq!(fmt(0.0, Null::collisions(8e9, k64)), "0.8235510139931854");
        let mut reps = Null::ZERO;
        for _ in 0..16 {
            reps += Null::collisions(8e9, k64);
        }
        assert_eq!(fmt(0.0, reps), "1 − 8.828901494125378e-13");
        assert_eq!(
            fmt(0.0, Null::collisions(3.2e10, k64)),
            "1 − 8.828901577425957e-13"
        );
        assert_eq!(
            fmt(222.0, Null::collisions(6.4e10, k64)),
            "1.2959913978087028e-20"
        );
        assert_eq!(
            fmt(16.0, Null::collisions(6.4e10, k64)),
            "1 − 1.804698905648335e-29"
        );
        assert_eq!(
            fmt(3524.0, Null::collisions(2.56e11, k64)),
            "5.169498564896866e-292"
        );
    }

    // With an integer number of trials we must obtain the binomial tails.
    #[test]
    fn test_binomial_tails_integer_trials() {
        let (n, q) = (60u32, 0.3f64);
        let pmf: Vec<f64> = (0..=n)
            .map(|j| {
                let mut c = 1.0f64;
                for i in 0..j {
                    c = c * f64::from(n - i) / f64::from(i + 1);
                }
                c * q.powi(j as i32) * (1.0 - q).powi((n - j) as i32)
            })
            .collect();
        for c in 0..=n {
            let left: f64 = pmf[..=c as usize].iter().sum();
            let right: f64 = pmf[c as usize..].iter().sum();
            let (l, r) = binomial_tails(f64::from(c), f64::from(n), q, 1.0 - q).unwrap();
            assert!(
                (l - left).abs() <= 1e-13 + 1e-11 * left,
                "c = {c}: {l} vs {left}"
            );
            assert!(
                (r - right).abs() <= 1e-13 + 1e-11 * right,
                "c = {c}: {r} vs {right}"
            );
        }
    }

    // With a non-integer number of trials: closed forms at 0 and 1, and deep
    // tails against references computed by mpmath with 50 digits.
    #[test]
    fn test_binomial_tails_real_trials() {
        let (n, q) = (1000.5f64, 0.02f64);
        let rel = |a: f64, b: f64| ((a - b) / b).abs();
        let (l0, r0) = binomial_tails(0.0, n, q, 1.0 - q).unwrap();
        let none = (1.0 - q).powf(n); // Pr[X = 0] = (1 − q)^N
        assert!(rel(l0, none) < 1e-12 && r0 == 1.0, "{l0} vs {none}");
        let (_, r1) = binomial_tails(1.0, n, q, 1.0 - q).unwrap();
        assert!(rel(r1, 1.0 - none) < 1e-12, "{r1} vs {}", 1.0 - none);
        let (l2, _) = binomial_tails(2.0, n, q, 1.0 - q).unwrap();
        assert!(rel(l2, 3.826_337_882_388_679_5e-7) < 1e-11, "{l2}");
        let (_, r120) = binomial_tails(120.0, n, q, 1.0 - q).unwrap();
        assert!(rel(r120, 2.740_708_898_294_940_3e-54) < 1e-11, "{r120}");
        let (_, r200) = binomial_tails(200.0, n, q, 1.0 - q).unwrap();
        assert!(rel(r200, 1.225_611_788_297_090_8e-131) < 1e-11, "{r200}");
        // Beyond the support.
        assert_eq!(binomial_tails(1002.0, n, q, 1.0 - q).unwrap(), (1.0, 0.0));
    }

    // As the variance tends to the mean, the binomial distribution tends to the
    // Poisson distribution.
    #[test]
    fn test_binomial_tends_to_poisson() {
        let lambda = 50.0;
        for c in [20.0, 40.0, 50.0, 60.0, 90.0] {
            let (bl, br) = null_tails(
                c,
                Null {
                    mean: lambda,
                    var: lambda * (1.0 - 1e-7),
                },
            )
            .unwrap();
            let t = poisson_tails(c, lambda).unwrap();
            assert!(
                ((bl - t.p_left) / t.p_left).abs() < 1e-4,
                "c = {c}: {bl} vs {}",
                t.p_left
            );
            assert!(
                ((br - t.p_right) / t.p_right).abs() < 1e-4,
                "c = {c}: {br} vs {}",
                t.p_right
            );
        }
    }

    // In every regime Pr[X ≤ c] + Pr[X ≥ c + 1] = 1, and tails are monotone.
    #[test]
    fn test_null_tails_complementary_and_monotone() {
        for null in [
            Null::poisson(120.0),
            Null {
                mean: 120.0,
                var: 100.0,
            },
            Null {
                mean: 120.0,
                var: 40.0,
            },
        ] {
            let (mut prev_l, mut prev_r) = (0.0f64, 1.0f64);
            for c in 0..400u32 {
                let c = f64::from(c);
                let (l, r) = null_tails(c, null).unwrap();
                let (_, r1) = null_tails(c + 1.0, null).unwrap();
                assert!(
                    (l + r1 - 1.0).abs() < 1e-12,
                    "{:?} c = {c}: {l} + {r1}",
                    null.kind()
                );
                assert!(
                    l >= prev_l && r <= prev_r,
                    "{:?} c = {c}: not monotone",
                    null.kind()
                );
                (prev_l, prev_r) = (l, r);
            }
        }
    }

    // Sums and scaling do not change the regime.
    #[test]
    fn test_null_arithmetic_preserves_regime() {
        for null in [
            Null::poisson(3.0),
            Null::collisions(1e5, 1e6),
            Null::collisions(1.2e6, 1e6),
        ] {
            let mut sum = Null::ZERO;
            for _ in 0..16 {
                sum += null;
            }
            assert_eq!(sum.kind(), null.kind());
            assert_eq!((sum / 16.0).kind(), null.kind());
            assert_eq!((0.25 * null).kind(), null.kind());
        }
    }

    // Invalid counts, counts at the mean, and counts beyond the support.
    #[test]
    fn test_p_value_edge_cases() {
        let null = Null::collisions(1e5, 1e6);
        assert_eq!(null.kind(), NullKind::Binomial);
        assert!(null_tails(-1.0, null).is_none() && null_tails(2.5, null).is_none());
        for null in [
            Null::collisions(1e4, 1e7),
            null,
            Null::collisions(1.2e6, 1e6),
        ] {
            let p = match p_value(null.mean.round(), null) {
                PValue::Direct(p) => p,
                PValue::NearOne(eps) => 1.0 - eps,
            };
            assert!((0.4..=0.6).contains(&p), "{:?}: p = {p}", null.kind());
        }
        let null = Null {
            mean: 10.0,
            var: 9.0,
        }; // N = 100
        assert!(matches!(p_value(101.0, null), PValue::Direct(p) if p == 0.0));
    }
}
