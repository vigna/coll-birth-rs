# Change Log

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
