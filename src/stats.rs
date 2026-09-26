/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

//! Statistical helpers: expected collision count, adjusted *p*-values, and formatting.

use std::borrow::Cow;

use std::ops::{Add, AddAssign, Div, Mul};

use cdflib::traits::{ContinuousCdf, DiscreteCdf};
use cdflib::{Normal, Poisson};

/// Variance-to-mean ratio of the number of collisions below which the count is
/// approximated by a normal distribution rather than by a Poisson one.
///
/// Note that this is *not* the criterion of TestU01, whose
/// `smultin_CreateCollisions` uses the Poisson approximation up to density
/// *n*/*k* = 1.0001 and the normal one only beyond. A Poisson distribution has
/// variance equal to its mean, but the variance of the number of collisions is
/// smaller than its mean by a factor of about 1 − 4α/3 when α = *n*/*k* is small,
/// and much smaller at larger densities (about 0.26 at α = 1). With TestU01's
/// threshold, a test at density close to 1 is thus judged against a null
/// distribution whose variance is almost four times too large: its *p*-values
/// cluster around 1/2, and the test hardly ever rejects. We therefore switch
/// to the normal approximation (with the exact mean and variance) as soon as
/// the Poisson variance overstates the true one by more than about 1%, which
/// happens at density ≈ 0.0075; at that point the expected number of
/// collisions is at least ≈ 37 (as tests have at least 10⁴ points), so the
/// normal approximation is already accurate.
pub const VARIANCE_RATIO_LIMIT: f64 = 0.99;

/// Densities below this value always use the Poisson approximation (the
/// variance-to-mean ratio is above 0.998 there), which avoids computing the
/// variance where [`collision_variance`] loses relative precision.
const POISSON_DENSITY: f64 = 0.001;

/// The density *n*/*k* that asymptotically maximizes the variance of the number
/// of collisions (the root of e^α = 1 + 2α); it is the default density of a
/// collision test.
pub const VARIANCE_MAXIMIZING_DENSITY: f64 = 1.256431;

/// The maximum density *n*/*k* accepted for the collision test: just above the
/// density ≈ 1.256431 that asymptotically maximizes the variance of the number
/// of collisions (the root of e^α = 1 + 2α).
pub const MAX_DENSITY: f64 = 1.26;

/// The null distribution of a collision (or birthday-spacings) count.
///
/// In the sparse regime the count is approximately Poisson, and only the mean
/// matters; when the variance-to-mean ratio falls below
/// [`VARIANCE_RATIO_LIMIT`] it is
/// approximately normal, with the exact mean and variance of the occupancy
/// problem. Null distributions of independent tests add up (means and
/// variances sum), and they can be scaled to obtain nominal per-pass shares.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Null {
    /// The expected count.
    pub mean: f64,
    /// The variance of the count (equal to the mean in the Poisson case).
    pub var: f64,
    /// Whether the normal approximation is used.
    pub normal: bool,
}

impl Null {
    /// The empty null distribution (the neutral element of addition).
    pub const ZERO: Null = Null {
        mean: 0.0,
        var: 0.0,
        normal: false,
    };

    /// A Poisson null distribution with the given mean.
    pub fn poisson(mean: f64) -> Self {
        Null {
            mean,
            var: mean,
            normal: false,
        }
    }

    /// The null distribution of the number of collisions of `points` balls
    /// thrown into `cells` bins: Poisson while the variance-to-mean ratio is at
    /// least [`VARIANCE_RATIO_LIMIT`], normal otherwise (see the documentation
    /// of [`VARIANCE_RATIO_LIMIT`] for why this differs from TestU01).
    pub fn collisions(points: f64, cells: f64) -> Self {
        let mean = expected_collisions(points, cells);
        if points / cells < POISSON_DENSITY {
            return Null::poisson(mean);
        }
        let var = collision_variance(points, cells);
        if var >= VARIANCE_RATIO_LIMIT * mean {
            Null::poisson(mean)
        } else {
            Null {
                mean,
                var,
                normal: true,
            }
        }
    }
}

impl Add for Null {
    type Output = Null;
    fn add(self, other: Null) -> Null {
        Null {
            mean: self.mean + other.mean,
            var: self.var + other.var,
            normal: self.normal || other.normal,
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
            normal: null.normal,
        }
    }
}

impl Div<f64> for Null {
    type Output = Null;
    fn div(self, d: f64) -> Null {
        Null {
            mean: self.mean / d,
            var: self.var / d,
            normal: self.normal,
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

/// Variance of the number of collisions when throwing `points` balls into
/// `cells` bins.
///
/// The number of collisions is `points` minus the number of occupied cells, so
/// its variance is that of the number of empty cells, that is,
/// *k*(*k* − 1)*b* + *ka* − *k*²*a*², where *a* = (1 − 1/*k*)*ⁿ* and *b* = (1
/// − 2/*k*)*ⁿ*. To avoid catastrophic cancellation when *k* is large, we
/// rewrite it as *k*(*a* − *b*) + *k*²(*b* − *a*²), and use *b*/*a* = (1 − 1/(*k*
/// − 1))*ⁿ* and *b*/*a*² = (1 − 1/(*k* − 1)²)*ⁿ*, evaluated through `ln_1p` and
/// `exp_m1`.
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

/// TestU01-style two-sided adjusted *p*-value of an observed count under the
/// given null distribution (cf. Chapter 3 of the long TestU01 guide). Returns
/// [`PValue::Direct`]`(f64::NAN)` if CDFLIB reports an error.
///
/// In the Poisson case the tails are `Pr[X ≤ coll]` and `Pr[X ≥ coll]`; in the
/// normal case they are, as in TestU01, Φ((coll − μ)/σ) and 1 − Φ((coll − 1 −
/// μ)/σ), the latter computed directly as an upper tail.
pub fn p_value(coll: f64, null: Null) -> PValue {
    let (p_left, p_right) = if null.normal {
        let Ok(normal) = Normal::try_new(null.mean, null.var.sqrt()) else {
            return PValue::Direct(f64::NAN);
        };
        (normal.cdf(coll), normal.ccdf(coll - 1.0))
    } else {
        let Some(tails) = poisson_tails(coll, null.mean) else {
            return PValue::Direct(f64::NAN);
        };
        (tails.p_left, tails.p_right)
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

    // A count deep in the left tail (here X = 0 against mean 700) has
    // p_left = e^-700 ≈ 9.9e-305, tiny but representable. Its two-sided p-value
    // is 1 - p_left, which must render in pretty mode as `1 - <that tiny eps>`, not
    // collapse to "1": computing `1 - p_left` in an f64 rounds to exactly 1.0 once
    // p_left < machine epsilon, destroying the tail. This pins the precision.
    #[test]
    fn test_deep_left_tail_renders_as_one_minus_eps() {
        let s = format_p_value(p_value(0.0, Null::poisson(700.0)), true);
        assert!(
            s.starts_with("1 − ") && s.len() > "1 − ".len(),
            "deep left tail must render as `1 − eps`, got {s:?}"
        );
    }

    // Two distinct deep-left-tail counts must produce distinct pretty strings:
    // the old `1 - p_left` collapse mapped both to "1".
    #[test]
    fn test_distinct_deep_tails_render_distinctly() {
        let a = format_p_value(p_value(0.0, Null::poisson(700.0)), true);
        let b = format_p_value(p_value(0.0, Null::poisson(600.0)), true);
        assert_ne!(a, b, "different left-tail depths must format differently");
    }

    // When the left tail underflows f64 to 0 (here e^-750 = 0), the p-value is 1
    // to machine precision but is still a left-tail anomaly: print the sentinel
    // `1 − 1e-307` (in both pretty and plain mode) rather than a bare "1".
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

    // A small right-tail p-value (too many collisions) is unaffected and prints
    // as a plain tiny decimal.
    #[test]
    fn test_right_tail_small_pvalue_is_plain() {
        let s = format_p_value(p_value(700.0, Null::poisson(1.0)), false);
        assert!(
            !s.starts_with("1-") && s != "1",
            "right-tail anomaly must be a small p-value, got {s:?}"
        );
    }

    // For a small number of cells the textbook occupancy variance
    // k(k − 1)(1 − 2/k)ⁿ + k(1 − 1/k)ⁿ − k²(1 − 1/k)²ⁿ can be evaluated directly
    // without cancellation problems; the stable rewriting must agree with it.
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

    // The null distribution switches from Poisson to normal when the
    // variance-to-mean ratio drops below VARIANCE_RATIO_LIMIT (density ≈ 0.0075),
    // and a count at the mean of a dense test is unremarkable.
    #[test]
    fn test_dense_regime_uses_normal_approximation() {
        let k = 1_000_000.0;
        assert!(!Null::collisions(0.005 * k, k).normal);
        assert!(Null::collisions(0.01 * k, k).normal);
        assert!(Null::collisions(1.0 * k, k).normal);
        let null = Null::collisions(1.2 * k, k);
        assert!(null.normal && null.var < null.mean);
        let p = match p_value(null.mean.round(), null) {
            PValue::Direct(p) => p,
            PValue::NearOne(eps) => 1.0 - eps,
        };
        assert!((0.4..=0.6).contains(&p), "p = {p}");
        // Ten standard deviations above the mean is a right-tail anomaly.
        let PValue::Direct(p) = p_value((null.mean + 10.0 * null.var.sqrt()).round(), null) else {
            panic!("expected a right-tail anomaly");
        };
        assert!(p < 1e-20, "p = {p}");
    }
}
