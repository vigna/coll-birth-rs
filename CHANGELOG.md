# Change Log

## [0.3.1] - 2026-10-01

### Fixed

- The normal approximation introduced in 0.3.0 underestimated right tails
  (by a factor of up to 50 at six standard deviations) when the expected
  number of collisions is small. Collision tests whose variance is between 54%
  and 99% of the mean (density up to ≈ 0.46) now use a binomial approximation
  with the same mean and variance; the run header reports the approximation
  used. Sparse tests are unaffected.

- `--pass` is now incompatible with `-d`: under decimation the null
  distribution depends on the total number of points kept, so the share of a
  single pass cannot be computed.

### Changed

- `Null` now holds only the mean and the variance; the approximation is chosen
  when computing tails (`Null::kind`, `stats::null_tails`).

## [0.3.0] - 2026-09-29

### New

- Collision tests whose number of collisions has a variance smaller than its
  mean by more than 1% (density above ≈ 0.0075) now compute _p_-values against
  a normal distribution with the exact mean and variance of the occupancy
  problem, rather than against a Poisson distribution. The run header appends
  `(normal approximation, variance: …)` in this regime.

- Collision tests now accept densities up to 1.26 (it was 1).

- In the dense regime, `--pass` also prints the unit's variance share as a
  third `var=` column; the units are then recombined with a normal _p_-value
  whose mean and variance are the sums of the shares.

### Changed

- When `m` is omitted, the collision test now uses ⌊1.256431 · _k_⌋ points on
  _k_ cells (the density maximizing the variance of the number of collisions)
  instead of _k_ points.

- The library API now describes null distributions with the `Null` type:
  `stats::p_value` takes a `Null` instead of a Poisson mean,
  `common::test_lambda` has been replaced by `common::test_null`, and the test
  runners return a `Null` instead of a summed Poisson mean.

- The `prng` module documents the `try_skip` contract: it must succeed for
  every offset or fail for every offset, as the parallel runners probe the
  capability with `try_skip(0)`.

- The run header now states how the points were generated: “sequentially”, or
  “using _k_ parallel generators (jump-ahead|pre-scan)” under `-P`. This
  replaces the old “parallel <test> test (k CPUs, jump-ahead)” phrasing and
  makes the effect of `RAYON_NUM_THREADS` visible in every mode's output. Only
  generation is described, since sorting and counting use the Rayon pool either
  way.

- Parallel runs now report the balls-into-bins headroom as a percentage after
  the memory figure, as sequential runs already did. The allocation itself is
  unchanged; it was simply not being reported.

### Fixed

- Collision tests at densities close to 1 were judged against a Poisson
  distribution whose variance is almost four times the true one (as happens in
  TestU01), so their _p_-values clustered around 1/2 and they hardly ever
  rejected.

## [0.2.3] - 2026-07-28

### Changed

- Changes to documentation.

## [0.2.2] - 2026-07-23

### Changed

- License is now MIT or Apache.

## [0.2.1] - 2026-07-14

### Changed

- The parallel birthday-spacings spacing filter now runs on the Rayon global
  pool instead of raw threads, so `RAYON_NUM_THREADS` governs every parallel
  phase as documented.

### Fixed

- Under `--pass`, per-repetition progress lines no longer print a p-value
  conditioned on a single unit's point count, which was statistically
  meaningless (spuriously small for collisions, spuriously close to one for
  birthday spacings); they now report raw counts only.

- `t` is now validated to be at most 128: larger values silently truncated
  the cell-count computation, defeating the 2¹²⁸-cells check.

- Omitting `m` with very large cell spaces (or passing an `m` whose point
  count overflows the address space) now produces a clean command-line or
  allocation error instead of a panic.

- The parallel checkpoint runner no longer panics when a checkpoint stage is
  smaller than the thread count (pre-scan generators only).

## [0.2.0] - 2026-07-01

### Changed

- Removed unused logging code and dependencies.

- `-P`/`--parallel` is now a boolean flag: it enables parallel generation on the
  Rayon global thread pool, and the number of threads for every phase
  (generation, sorting, counting) is controlled by `RAYON_NUM_THREADS`.

- Improved option names.

## [0.1.0] - 2026-07-01

### New

- First release.
