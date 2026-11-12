/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

//! Pseudorandom number generators selected at build time via [Cargo features].
//!
//! Exactly one feature must be enabled when building the crate; each variant
//! exposes a single [`Prng`] type with a [`new(seed: u64) -> Self`][new]
//! constructor, a [`next_u64(&mut self) -> u64`][next_u64] step function, and a
//! [`try_skip(&mut self, n: u64) -> Result<(), ()>`][try_skip] method to skip
//! ahead by `n` steps.
//!
//! [`try_skip(n)`][try_skip] must have the same effect as `n` calls to
//! [`next_u64`][next_u64]. It must succeed for every `n`, or fail for every `n`
//! (including 0) without changing the state: the parallel runners
//! [`run_test_parallel`] and [`run_birthday_parallel`] probe the capability
//! with [`try_skip(0)`][try_skip].
//!
//! Generators outputting less than 64 bits must shift their outputs to the top
//! (e.g., 32-bit generators must return their output shifted to the left by
//! 32).
//!
//! The `MWCX` feature, which must be combined with an MWC feature, applies the
//! scrambler X to the output of the MWC generator: the output digit is xored
//! with the carry of the same state (as in [Thomas's MWC64X]), and the name of
//! the generator gets an `X` suffix.
//!
//! The `LACUNARY` feature, combined with an MWC feature (and possibly with
//! `MWCX`), returns, from each block of *k* + 2 consecutive outputs of an MWC
//! generator with lag *k*, the outputs of index 0, 1, and *k* + 1, so that a
//! three-dimensional test (which is the only one accepted) tests the lacunary
//! triples (*xₙ*, *xₙ*₊₁, *xₙ*₊*ₖ*₊₁), which lie in planes at a distance
//! similar to that of the successive triples of an MWC generator with lag one.
//!
//! The `ranlux_0`, …, `ranlux_4` features select [Lüscher's RANLUX] at the
//! given luxury level. When combined with a RANLUX feature, `LACUNARY` returns
//! the outputs involved in a sparse linear relation among those kept by the
//! generator, so that a test whose dimension is the weight of the relation
//! ([`LACUNARY_DIM`], the only one accepted) tests points lying on a family of
//! parallel hyperplanes (see the [RANLUX example]).
//!
//! [Cargo features]: https://doc.rust-lang.org/cargo/reference/features.html
//! [new]: Prng::new
//! [next_u64]: Prng::next_u64
//! [try_skip]: Prng::try_skip
//! [`run_test_parallel`]: crate::collision::run_test_parallel
//! [`run_birthday_parallel`]: crate::birthday::run_birthday_parallel
//! [Thomas's MWC64X]: https://web.archive.org/web/20240702142320/https://cas.ee.ic.ac.uk/people/dt10/research/rngs-gpu-mwc64x.html
//! [Lüscher's RANLUX]: https://luscher.web.cern.ch/luscher/ranlux/
//! [RANLUX example]: crate#example-ranlux

// When no PRNG feature is selected, the _prng marker (enabled by every PRNG
// feature in Cargo.toml) is off. Emit one clear error AND expose a placeholder Prng
// so the rest of the crate still type-checks.
#[cfg(not(feature = "_prng"))]
compile_error!(
    "no PRNG selected: enable exactly one PRNG feature, as in `--features splitmix` \
     (see the [features] table in Cargo.toml)"
);

#[cfg(not(feature = "_prng"))]
mod placeholder {
    #[derive(Clone, Copy)]
    pub struct Prng;
    impl Prng {
        pub const NAME: &str = "(no generator selected)";
        pub fn new(_seed: u64) -> Self {
            Self
        }
        #[inline(always)]
        pub fn next_u64(&mut self) -> u64 {
            0
        }
        pub fn try_skip(&mut self, _n: u64) -> Result<(), ()> {
            Err(())
        }
    }
}

#[cfg(not(feature = "_prng"))]
pub use placeholder::Prng;

#[cfg(all(feature = "MWCX", not(feature = "_mwc")))]
compile_error!("the MWCX feature applies only to MWC generators: combine it with an mwc_* feature");

#[cfg(all(feature = "LACUNARY", not(any(feature = "_mwc", feature = "_ranlux"))))]
compile_error!(
    "the LACUNARY feature applies only to MWC and RANLUX generators: combine it with an mwc_* or \
     a ranlux_* feature"
);

/// The only dimension accepted when the `LACUNARY` feature is enabled, as the
/// generator returns one lacunary tuple every `LACUNARY_DIM` outputs.
#[cfg(feature = "_ranlux")]
pub const LACUNARY_DIM: usize = ranlux::LEN;
/// The only dimension accepted when the `LACUNARY` feature is enabled, as the
/// generator returns one lacunary tuple every `LACUNARY_DIM` outputs.
#[cfg(not(feature = "_ranlux"))]
pub const LACUNARY_DIM: usize = 3;

// ----- Multiply-with-carry family -----------------------------------------------------

// The name of an MWC generator: the base name, followed by X if MWCX is enabled,
// by the multiplier, and, if LACUNARY is enabled, by the indices of the triples.
#[cfg(feature = "_mwc")]
macro_rules! mwc_name {
    ($base:literal, $a:expr, $lacunary:literal) => {
        match (cfg!(feature = "MWCX"), cfg!(feature = "LACUNARY")) {
            (false, false) => concat!($base, " (", stringify!($a), ")"),
            (true, false) => concat!($base, "X (", stringify!($a), ")"),
            (false, true) => concat!(
                $base,
                " (",
                stringify!($a),
                "), lacunary triples ",
                $lacunary
            ),
            (true, true) => concat!(
                $base,
                "X (",
                stringify!($a),
                "), lacunary triples ",
                $lacunary
            ),
        }
    };
}

#[cfg(any(
    feature = "mwc_128_64_0xff96d28c3f3329da",
    feature = "mwc_128_64_0xffebb71d94fcdaf9",
    feature = "mwc_128_64_0xff6da236274e252e"
))]
macro_rules! mwc_128_64 {
    ($a1:expr) => {
        const MWC_A1: u64 = $a1;

        #[derive(Clone, Copy)]
        pub struct Prng {
            x: u64,
            c: u64,
        }

        impl Prng {
            pub const NAME: &str = mwc_name!("MWC128", $a1, "(0, 1, 2)");
            /// The lag *k* of the generator.
            pub const LAG: u64 = 1;

            pub fn new(seed: u64) -> Self {
                Self { x: seed, c: 1 }
            }

            #[inline(always)]
            pub fn next_u64(&mut self) -> u64 {
                let result = if cfg!(feature = "MWCX") {
                    self.x ^ self.c
                } else {
                    self.x
                };
                let t = (MWC_A1 as u128)
                    .wrapping_mul(self.x as u128)
                    .wrapping_add(self.c as u128);
                self.x = t as u64;
                self.c = (t >> 64) as u64;
                result
            }

            pub fn try_skip(&mut self, n: u64) -> Result<(), ()> {
                use ::num::BigUint;
                let b = BigUint::from(1u128 << 64);
                let a = BigUint::from(MWC_A1);
                let m = &a * &b - BigUint::from(1u8); // a·b − 1
                let mu = a % &m; // b⁻¹ = a·b⁰ = a
                let factor = mu.modpow(&BigUint::from(n), &m);
                let s = BigUint::from(self.x) + BigUint::from(self.c) * &b;
                let s = (s * factor) % &m;
                let d = s.to_u64_digits();
                self.x = d.first().copied().unwrap_or(0);
                self.c = d.get(1).copied().unwrap_or(0);
                Ok(())
            }
        }
    };
}

#[cfg(any(
    feature = "mwc_192_64_0xffa04e67b3c95d86",
    feature = "mwc_192_64_0xff4e24dd418f5433"
))]
macro_rules! mwc_192_64 {
    ($a2:expr) => {
        const MWC_A2: u64 = $a2;

        #[derive(Clone, Copy)]
        pub struct Prng {
            x: u64,
            y: u64,
            c: u64,
        }

        impl Prng {
            pub const NAME: &str = mwc_name!("MWC192", $a2, "(0, 1, 3)");
            /// The lag *k* of the generator.
            pub const LAG: u64 = 2;

            pub fn new(seed: u64) -> Self {
                Self {
                    x: seed,
                    y: seed,
                    c: 1,
                }
            }

            #[inline(always)]
            pub fn next_u64(&mut self) -> u64 {
                let result = if cfg!(feature = "MWCX") {
                    self.y ^ self.c
                } else {
                    self.y
                };
                let t = (MWC_A2 as u128)
                    .wrapping_mul(self.x as u128)
                    .wrapping_add(self.c as u128);
                self.x = self.y;
                self.y = t as u64;
                self.c = (t >> 64) as u64;
                result
            }

            pub fn try_skip(&mut self, n: u64) -> Result<(), ()> {
                use ::num::BigUint;
                let b = BigUint::from(1u128 << 64);
                let a = BigUint::from(MWC_A2);
                let m = &a * &b * &b - BigUint::from(1u8); // a·b² − 1
                let mu = (&a * &b) % &m; // b⁻¹ = a·b
                let factor = mu.modpow(&BigUint::from(n), &m);
                let s = BigUint::from(self.x)
                    + BigUint::from(self.y) * &b
                    + BigUint::from(self.c) * &b * &b;
                let s = (s * factor) % &m;
                let d = s.to_u64_digits();
                self.x = d.first().copied().unwrap_or(0);
                self.y = d.get(1).copied().unwrap_or(0);
                self.c = d.get(2).copied().unwrap_or(0);
                Ok(())
            }
        }
    };
}

#[cfg(any(
    feature = "mwc_256_64_0xfff62cf2ccc0cdaf",
    feature = "mwc_256_64_0xff32e7e2bdcb0a33"
))]
macro_rules! mwc_256_64 {
    ($a3:expr) => {
        const MWC_A3: u64 = $a3;

        #[derive(Clone, Copy)]
        pub struct Prng {
            x: u64,
            y: u64,
            z: u64,
            c: u64,
        }

        impl Prng {
            pub const NAME: &str = mwc_name!("MWC256", $a3, "(0, 1, 4)");
            /// The lag *k* of the generator.
            pub const LAG: u64 = 3;

            pub fn new(seed: u64) -> Self {
                Self {
                    x: seed,
                    y: seed,
                    z: seed,
                    c: 1,
                }
            }

            #[inline(always)]
            pub fn next_u64(&mut self) -> u64 {
                let result = if cfg!(feature = "MWCX") {
                    self.z ^ self.c
                } else {
                    self.z
                };
                let t = (MWC_A3 as u128)
                    .wrapping_mul(self.x as u128)
                    .wrapping_add(self.c as u128);
                self.x = self.y;
                self.y = self.z;
                self.z = t as u64;
                self.c = (t >> 64) as u64;
                result
            }

            pub fn try_skip(&mut self, n: u64) -> Result<(), ()> {
                use ::num::BigUint;
                let b = BigUint::from(1u128 << 64);
                let a = BigUint::from(MWC_A3);
                let m = &a * &b * &b * &b - BigUint::from(1u8); // a·b³ − 1
                let mu = (&a * &b * &b) % &m; // b⁻¹ = a·b²
                let factor = mu.modpow(&BigUint::from(n), &m);
                let s = BigUint::from(self.x)
                    + BigUint::from(self.y) * &b
                    + BigUint::from(self.z) * &b * &b
                    + BigUint::from(self.c) * &b * &b * &b;
                let s = (s * factor) % &m;
                let d = s.to_u64_digits();
                self.x = d.first().copied().unwrap_or(0);
                self.y = d.get(1).copied().unwrap_or(0);
                self.z = d.get(2).copied().unwrap_or(0);
                self.c = d.get(3).copied().unwrap_or(0);
                Ok(())
            }
        }
    };
}

#[cfg(any(feature = "mwc_128_32_0xfffea2df", feature = "mwc_128_32_0xffcccc66"))]
macro_rules! mwc_128_32 {
    ($a3:expr) => {
        const MWC_A3: u32 = $a3;

        #[derive(Clone, Copy)]
        pub struct Prng {
            x: u32,
            y: u32,
            z: u32,
            c: u32,
        }

        impl Prng {
            pub const NAME: &str = mwc_name!("MWC128-32", $a3, "(0, 1, 4)");
            /// The lag *k* of the generator.
            pub const LAG: u64 = 3;

            pub fn new(seed: u64) -> Self {
                Self {
                    x: seed as u32,
                    y: (seed >> 32) as u32,
                    z: 0,
                    c: 1,
                }
            }

            #[inline(always)]
            pub fn next_u64(&mut self) -> u64 {
                let result = if cfg!(feature = "MWCX") {
                    ((self.z ^ self.c) as u64) << 32
                } else {
                    (self.z as u64) << 32
                };
                let t = (MWC_A3 as u64)
                    .wrapping_mul(self.x as u64)
                    .wrapping_add(self.c as u64);
                self.x = self.y;
                self.y = self.z;
                self.z = t as u32;
                self.c = (t >> 32) as u32;
                result
            }

            pub fn try_skip(&mut self, n: u64) -> Result<(), ()> {
                use ::num::BigUint;
                let b = BigUint::from(1u64 << 32);
                let a = BigUint::from(MWC_A3);
                let m = &a * &b * &b * &b - BigUint::from(1u8); // a·b³ − 1 = a·2⁹⁶ − 1
                let mu = (&a * &b * &b) % &m; // b⁻¹ = a·b²
                let factor = mu.modpow(&BigUint::from(n), &m);
                let s = BigUint::from(self.x)
                    + BigUint::from(self.y) * &b
                    + BigUint::from(self.z) * &b * &b
                    + BigUint::from(self.c) * &b * &b * &b;
                let s = (s * factor) % &m;
                let d = s.to_u32_digits();
                self.x = d.first().copied().unwrap_or(0);
                self.y = d.get(1).copied().unwrap_or(0);
                self.z = d.get(2).copied().unwrap_or(0);
                self.c = d.get(3).copied().unwrap_or(0);
                Ok(())
            }
        }
    };
}

// The MWC generators live in a private module, so that LACUNARY can wrap them.
#[cfg(feature = "_mwc")]
mod mwc {
    #[cfg(feature = "mwc_128_64_0xffebb71d94fcdaf9")]
    // Unique value quantile 0.855469, f₃ = 0.000609206
    mwc_128_64!(0xffebb71d94fcdaf9);

    #[cfg(feature = "mwc_128_64_0xff96d28c3f3329da")]
    // f₃ = 9.1743e-09
    mwc_128_64!(0xff96d28c3f3329da);

    #[cfg(feature = "mwc_192_64_0xffa04e67b3c95d86")]
    // Best harmonic score among first 1000 by quantile
    mwc_192_64!(0xffa04e67b3c95d86);

    #[cfg(feature = "mwc_256_64_0xfff62cf2ccc0cdaf")]
    // Unique value quantile 0.967285
    mwc_256_64!(0xfff62cf2ccc0cdaf);

    #[cfg(feature = "mwc_128_32_0xfffea2df")]
    // Unique value quantile 0.972656
    mwc_128_32!(0xfffea2df);

    // Multipliers suggested in "Scrambled Multiply-with-Carry Pseudorandom Number
    // Generators" (best weighted geometric mean of the figures of merit)

    #[cfg(feature = "mwc_128_64_0xff6da236274e252e")]
    mwc_128_64!(0xff6da236274e252e);

    #[cfg(feature = "mwc_192_64_0xff4e24dd418f5433")]
    mwc_192_64!(0xff4e24dd418f5433);

    #[cfg(feature = "mwc_256_64_0xff32e7e2bdcb0a33")]
    mwc_256_64!(0xff32e7e2bdcb0a33);

    #[cfg(feature = "mwc_128_32_0xffcccc66")]
    mwc_128_32!(0xffcccc66);
}

#[cfg(all(feature = "_mwc", not(feature = "LACUNARY")))]
pub use mwc::Prng;

// The lacunary triples of an MWC generator with lag *k*.
#[cfg(all(feature = "_mwc", feature = "LACUNARY"))]
mod lacunary {
    use super::mwc::Prng as Mwc;

    /// Returns, from each block of *k* + 2 consecutive outputs *x*ₙ, *x*ₙ₊₁, …,
    /// *x*ₙ₊ₖ₊₁ of an MWC generator with lag *k*, the triple *x*ₙ, *x*ₙ₊₁,
    /// *x*ₙ₊ₖ₊₁.
    ///
    /// A three-dimensional test, which builds each point from three
    /// consecutive outputs starting from the first one, thus tests the
    /// non-overlapping lacunary triples of the generator.
    #[derive(Clone, Copy)]
    pub struct Prng {
        mwc: Mwc,
        // The position in the current triple of the next output.
        pos: u64,
    }

    impl Prng {
        pub const NAME: &str = Mwc::NAME;

        pub fn new(seed: u64) -> Self {
            Self {
                mwc: Mwc::new(seed),
                pos: 0,
            }
        }

        #[inline(always)]
        pub fn next_u64(&mut self) -> u64 {
            if self.pos == 2 {
                // Discard *x*ₙ₊₂, …, *x*ₙ₊ₖ (nothing, if the lag is one)
                #[allow(clippy::reversed_empty_ranges)]
                for _ in 1..Mwc::LAG {
                    self.mwc.next_u64();
                }
                self.pos = 0;
            } else {
                self.pos += 1;
            }
            self.mwc.next_u64()
        }

        pub fn try_skip(&mut self, n: u64) -> Result<(), ()> {
            // Position `pos` of a triple is position `pos` of its block, and
            // position (`pos` + *n*) mod 3 of triple ⌊(`pos` + *n*) / 3⌋ is the
            // same position of block ⌊(`pos` + *n*) / 3⌋; as *k* + 2 ≥ 3, the
            // difference is nonnegative.
            let target = self.pos as u128 + n as u128;
            let steps = target / 3 * (Mwc::LAG as u128 + 2) + target % 3 - self.pos as u128;
            let (mut q, r) = (steps >> 64, steps as u64);
            while q > 0 {
                self.mwc.try_skip(u64::MAX)?;
                self.mwc.try_skip(1)?;
                q -= 1;
            }
            self.mwc.try_skip(r)?;
            self.pos = (target % 3) as u64;
            Ok(())
        }
    }

    #[cfg(test)]
    mod tests {
        use super::{Mwc, Prng};

        #[test]
        fn test_triples() {
            let mut lacunary = Prng::new(42);
            let mut mwc = Mwc::new(42);
            for _ in 0..1000 {
                let block: Vec<u64> = (0..Mwc::LAG + 2).map(|_| mwc.next_u64()).collect();
                assert_eq!(lacunary.next_u64(), block[0]);
                assert_eq!(lacunary.next_u64(), block[1]);
                assert_eq!(lacunary.next_u64(), block[Mwc::LAG as usize + 1]);
            }
        }

        #[test]
        fn test_skip() {
            for start in 0..7 {
                for n in [0, 1, 2, 3, 4, 5, 6, 7, 299, 300, 301, 1000] {
                    let mut skipped = Prng::new(42);
                    let mut stepped = Prng::new(42);
                    for _ in 0..start {
                        skipped.next_u64();
                        stepped.next_u64();
                    }
                    skipped
                        .try_skip(n)
                        .expect("try_skip failed on a skip-capable generator");
                    for _ in 0..n {
                        stepped.next_u64();
                    }
                    for _ in 0..10 {
                        assert_eq!(
                            skipped.next_u64(),
                            stepped.next_u64(),
                            "start {start}, n {n}"
                        );
                    }
                }
            }
        }
    }
}

#[cfg(all(feature = "_mwc", feature = "LACUNARY"))]
pub use lacunary::Prng;

// ----- RANLUX ---------------------------------------------------------------------

// The constants of a luxury level. The relation is a list of pairs (offset in a
// block of 2p subtract-with-borrow outputs, coefficient).
#[cfg(feature = "_ranlux")]
macro_rules! ranlux_level {
    ($level:literal, $p:literal, [$(($offset:literal, $coeff:literal)),+ $(,)?]) => {
        /// The number of subtract-with-borrow outputs per block of kept
        /// outputs.
        const P: usize = $p;
        /// A sparse relation among kept outputs, as pairs (offset,
        /// coefficient).
        pub(super) const RELATION: &[(usize, i64)] = &[$(($offset, $coeff)),+];
        pub(super) const NAME: &str = if cfg!(feature = "LACUNARY") {
            concat!(
                "RANLUX (luxury level ", $level, ", p = ", $p, "), lacunary tuples (",
                stringify!($($offset),+),
                ")"
            )
        } else {
            concat!("RANLUX (luxury level ", $level, ", p = ", $p, ")")
        };
    };
}

/// [Lüscher's RANLUX].
///
/// RANLUX decimates the subtract-with-borrow generator *x*ₙ = (*x*ₙ₋₁₀ −
/// *x*ₙ₋₂₄ − *c*ₙ₋₁) mod 2²⁴ of [Marsaglia and Zaman], where the borrow *c*ₙ is
/// 1 if *x*ₙ₋₁₀ − *x*ₙ₋₂₄ − *c*ₙ₋₁ < 0, and 0 otherwise: from each block of *p*
/// outputs it keeps the first 24, where *p*, which is determined by the luxury
/// level, is 24, 48, 97, 223, or 389.
///
/// The generator is an LCG in disguise: if *b* = 2²⁴ and *m* = *b*²⁴ − *b*¹⁰ +
/// 1, the state with outputs *x*ₙ₋₂₄, …, *x*ₙ₋₁ and carry *c*ₙ₋₁ corresponds to
///
/// *Z*ₙ = ∑ⱼ *b*ʲ *x*ₙ₋₂₄₊ⱼ − ∑ₖ *b*ᵏ *x*ₙ₋₁₀₊ₖ + *c*ₙ₋₁,
///
/// where 0 ≤ *j* < 24 and 0 ≤ *k* < 10, and *Z*ₙ₊₁ ≡ *b*⁻¹ *Z*ₙ (mod *m*), with
/// *x*ₙ = ⌊*b* *Z*ₙ₊₁ / *m*⌋. Thus, *x*ₙ / *b* ≈ *Z*ₙ₊₁ / *m*, and whenever ∑
/// *cᵢ* *b*⁻ⁱ ≡ 0 (mod *m*) the sum ∑ *cᵢ* *x*ₙ₊ᵢ / *b* is within ∑ |*cᵢ*| /
/// *b* of an integer, whatever the carries are. Decimation does not remove such
/// relations: it just hides those involving discarded outputs.
///
/// Since *z*²⁴ − *z*¹⁴ + 1 is a polynomial in *z*², there are such relations
/// with small coefficients among outputs with offsets of the same parity. The
/// relation of each level ([`RELATION`]) has been found by lattice reduction,
/// and its weight is 3, 5, 8, 16, and 23, respectively. The points of a test of
/// that dimension lie on a family of parallel hyperplanes, but a test can
/// detect them only if the number *u* of bits is large with respect to the
/// logarithm of the norm of the coefficients.
///
/// # Implementation Notes
///
/// The generator is seeded as [`std::ranlux24_base`], so that the output of
/// [`new(0)`][new] is that of a default-constructed instance, and reproduces
/// the value mandated by the C++ standard. Note that in the standard library
/// [`ranlux24`] keeps 23 outputs out of 223, whereas we keep 24 as in the
/// original RANLUX.
///
/// There is a schedule of [`LEN`] outputs for each period of [`PERIOD`]
/// subtract-with-borrow steps: the first 24 outputs of *p* or, with the
/// `LACUNARY` feature, the outputs of a relation in a block of 2*p*, so that
/// different tuples share no output. Offsets are subtract-with-borrow steps
/// from the start of such a block, which is aligned with the blocks of *p*
/// outputs, and they all fall in the first 24 outputs of a block of *p*.
///
/// [Lüscher's RANLUX]: https://luscher.web.cern.ch/luscher/ranlux/
/// [Marsaglia and Zaman]: https://doi.org/10.1214/aoap/1177005878
/// [`RELATION`]: ranlux::RELATION
/// [`std::ranlux24_base`]: https://en.cppreference.com/w/cpp/numeric/random/subtract_with_carry_engine.html
/// [new]: ranlux::Prng::new
/// [`ranlux24`]: https://en.cppreference.com/w/cpp/numeric/random/discard_block_engine.html
/// [`LEN`]: ranlux::LEN
/// [`PERIOD`]: ranlux::PERIOD
#[cfg(feature = "_ranlux")]
mod ranlux {
    use ::num::{BigUint, ToPrimitive};

    /// The number of bits of an output.
    const W: u32 = 24;
    /// The long lag.
    const R: usize = 24;
    /// The short lag.
    const S: usize = 10;
    /// The mask of an output.
    const MASK: u32 = (1 << W) - 1;

    #[cfg(feature = "ranlux_0")]
    ranlux_level!(0, 24, [(0, 1), (14, -1), (24, 1)]);
    #[cfg(feature = "ranlux_1")]
    ranlux_level!(1, 48, [(0, -1), (10, -1), (14, 1), (48, -1), (58, 1)]);
    #[cfg(feature = "ranlux_2")]
    ranlux_level!(
        2,
        97,
        [
            (8, 1),
            (18, 1),
            (22, -1),
            (98, 1),
            (104, 3),
            (108, -4),
            (114, -1),
            (118, 3),
        ]
    );
    #[cfg(feature = "ranlux_3")]
    ranlux_level!(
        3,
        223,
        [
            (1, -3),
            (3, 23),
            (5, 6),
            (7, 11),
            (9, 5),
            (13, 20),
            (15, 10),
            (17, -18),
            (21, -13),
            (23, 6),
            (223, 1),
            (225, -4),
            (233, -5),
            (241, 1),
            (243, 3),
            (245, 2),
        ]
    );
    #[cfg(feature = "ranlux_4")]
    ranlux_level!(
        4,
        389,
        [
            (1, 21),
            (3, 20),
            (5, -69),
            (7, 2),
            (9, -48),
            (11, 19),
            (13, 51),
            (15, -54),
            (17, 5),
            (19, 78),
            (21, 24),
            (23, 115),
            (389, -25),
            (391, -74),
            (393, 56),
            (395, -11),
            (399, 84),
            (401, 4),
            (403, -25),
            (405, -28),
            (407, 25),
            (409, -63),
            (411, 28),
        ]
    );

    /// The number of outputs for each period.
    #[cfg(not(feature = "LACUNARY"))]
    pub(super) const LEN: usize = R;
    /// The number of outputs for each period.
    #[cfg(feature = "LACUNARY")]
    pub(super) const LEN: usize = RELATION.len();

    /// The length of a period in subtract-with-borrow steps.
    #[cfg(not(feature = "LACUNARY"))]
    pub(super) const PERIOD: usize = P;
    /// The length of a period in subtract-with-borrow steps.
    #[cfg(feature = "LACUNARY")]
    pub(super) const PERIOD: usize = 2 * P;

    /// The position in a period of the output of index `i`.
    #[cfg(not(feature = "LACUNARY"))]
    #[inline(always)]
    const fn offset(i: usize) -> usize {
        i
    }
    /// The position in a period of the output of index `i`.
    #[cfg(feature = "LACUNARY")]
    #[inline(always)]
    const fn offset(i: usize) -> usize {
        RELATION[i].0
    }

    /// Returns the number of steps from the start of a period to the state in
    /// which output `i` of the period is next.
    #[inline(always)]
    const fn steps_to(i: usize) -> usize {
        if i == 0 { 0 } else { offset(i - 1) + 1 }
    }

    /// The subtract-with-borrow generator underlying RANLUX.
    #[derive(Clone, Copy)]
    struct Swb {
        /// The last 24 outputs, as a circular buffer.
        x: [u32; R],
        /// The position in `x` of the oldest output.
        i: usize,
        /// The carry.
        c: u32,
    }

    impl Swb {
        /// Creates a generator seeded as [`std::ranlux24_base`].
        ///
        /// [`std::ranlux24_base`]: https://en.cppreference.com/w/cpp/numeric/random/subtract_with_carry_engine.html
        fn new(seed: u64) -> Self {
            const LCG_A: u64 = 40014;
            const LCG_M: u64 = 2147483563;
            // The default seed of std::ranlux24_base.
            let mut e = if seed == 0 { 19780503 } else { seed % LCG_M };
            if e == 0 {
                e = 1;
            }
            let mut x = [0; R];
            for v in &mut x {
                e = e * LCG_A % LCG_M;
                *v = e as u32 & MASK;
            }
            let c = (x[R - 1] == 0) as u32;
            Self { x, i: 0, c }
        }

        /// Performs a step, returning the output.
        #[inline(always)]
        fn step(&mut self) -> u32 {
            let i = self.i;
            let j = if i < S { i + R - S } else { i - S };
            let t = self.x[j].wrapping_sub(self.x[i]).wrapping_sub(self.c);
            self.c = t >> 31;
            self.x[i] = t & MASK;
            self.i = if i == R - 1 { 0 } else { i + 1 };
            t & MASK
        }

        /// Skips `n` steps.
        ///
        /// See the [documentation of the module] for the correspondence with an
        /// LCG. The inverse correspondence recovers the outputs, from the last
        /// one, as the base-*b* digits of *Z* / *m*.
        ///
        /// The correspondence is not injective, but equivalent states yield the
        /// same outputs, as outputs depend only on *Z*: since a state reached
        /// by 24 steps or more is the one recovered from its *Z*, this method
        /// might change, with respect to stepping, just the representation of
        /// the initial state.
        ///
        /// [documentation of the module]: self
        fn skip(&mut self, n: u128) {
            let b = BigUint::from(1u32 << W);
            let m = b.pow(R as u32) - b.pow(S as u32) + 1u32;
            // b⁻¹ = m − (m − 1) / b
            let a = &m - (&m - 1u32) / &b;

            let digits = |x: &dyn Fn(usize) -> u32, len: usize| {
                (0..len).rev().fold(BigUint::ZERO, |acc, j| acc * &b + x(j))
            };
            let high = digits(&|j| self.x[(self.i + j) % R], R);
            let low = digits(&|j| self.x[(self.i + R - S + j) % R], S);
            let z = (high + self.c + &m - low) % &m;

            let z = (z * a.modpow(&BigUint::from(n), &m)) % &m;

            let mut r = z.clone();
            for j in (0..R).rev() {
                let t = r * &b;
                self.x[j] = (&t / &m).to_u32().expect("digits are smaller than b");
                r = t % &m;
            }
            self.i = 0;
            let high = digits(&|j| self.x[j], R);
            let low = digits(&|j| self.x[R - S + j], S);
            self.c = (z + low - high)
                .to_u32()
                .expect("the carry is either zero or one");
        }
    }

    #[derive(Clone, Copy)]
    pub struct Prng {
        swb: Swb,
        /// The index in the period of the next output.
        pos: usize,
    }

    impl Prng {
        pub const NAME: &str = NAME;

        pub fn new(seed: u64) -> Self {
            Self {
                swb: Swb::new(seed),
                pos: 0,
            }
        }

        #[inline(always)]
        pub fn next_u64(&mut self) -> u64 {
            for _ in steps_to(self.pos)..offset(self.pos) {
                self.swb.step();
            }
            let result = (self.swb.step() as u64) << (64 - W);
            self.pos += 1;
            if self.pos == LEN {
                for _ in steps_to(LEN)..PERIOD {
                    self.swb.step();
                }
                self.pos = 0;
            }
            result
        }

        pub fn try_skip(&mut self, n: u64) -> Result<(), ()> {
            let target = self.pos as u128 + n as u128;
            let (periods, pos) = (target / LEN as u128, (target % LEN as u128) as usize);
            // A period is at least as long as LEN steps, and steps_to() is
            // monotone in a period, so the difference is nonnegative.
            let steps = periods * PERIOD as u128 + steps_to(pos) as u128;
            self.swb.skip(steps - steps_to(self.pos) as u128);
            self.pos = pos;
            Ok(())
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_standard_value() {
            // The 10000th output of a default-constructed std::ranlux24_base.
            let mut swb = Swb::new(0);
            let mut last = 0;
            for _ in 0..10000 {
                last = swb.step();
            }
            assert_eq!(last, 7937952);
        }

        #[test]
        fn test_schedule() {
            let mut prng = Prng::new(42);
            let mut swb = Swb::new(42);
            for _ in 0..100 {
                let period: Vec<u32> = (0..PERIOD).map(|_| swb.step()).collect();
                for i in 0..LEN {
                    assert_eq!(prng.next_u64() >> 40, period[offset(i)] as u64);
                }
            }
        }

        #[test]
        fn test_skip_outputs() {
            for start in 0..2 * LEN {
                for n in [
                    0,
                    1,
                    2,
                    3,
                    LEN as u64 - 1,
                    LEN as u64,
                    LEN as u64 + 1,
                    1000,
                    100_001,
                ] {
                    let mut skipped = Prng::new(42);
                    let mut stepped = Prng::new(42);
                    for _ in 0..start {
                        skipped.next_u64();
                        stepped.next_u64();
                    }
                    skipped
                        .try_skip(n)
                        .expect("try_skip failed on a skip-capable generator");
                    for _ in 0..n {
                        stepped.next_u64();
                    }
                    for _ in 0..2 * LEN {
                        assert_eq!(
                            skipped.next_u64(),
                            stepped.next_u64(),
                            "start {start}, n {n}"
                        );
                    }
                }
            }
        }

        #[test]
        fn test_skip_state() {
            // A jump must restore the whole state, carry included, once the
            // state has lost the representation assigned by seeding.
            for seed in 1..50 {
                for n in [0, 1, 9, 10, 23, 24, 25, 1000, 12345] {
                    let mut skipped = Swb::new(seed);
                    for _ in 0..R {
                        skipped.step();
                    }
                    let mut stepped = skipped;
                    skipped.skip(n as u128);
                    for _ in 0..n {
                        stepped.step();
                    }
                    assert_eq!(skipped.c, stepped.c, "seed {seed}, n {n}");
                    for j in 0..R {
                        assert_eq!(
                            skipped.x[(skipped.i + j) % R],
                            stepped.x[(stepped.i + j) % R],
                            "seed {seed}, n {n}, j {j}"
                        );
                    }
                }
            }
        }

        #[test]
        fn test_skip_composes() {
            // Jumps longer than any feasible number of steps must compose.
            let (n, k) = (1u128 << 100, 12345);
            let mut once = Swb::new(42);
            once.skip(n + k);
            let mut twice = Swb::new(42);
            twice.skip(n);
            for _ in 0..k {
                twice.step();
            }
            for _ in 0..100 {
                assert_eq!(once.step(), twice.step());
            }
        }

        #[test]
        fn test_relation() {
            // The residue must be bounded by the sum of the absolute values of
            // the coefficients: for uniform outputs, the probability of this
            // event is at most ≈ 2⁻¹⁰ per sample.
            let l1: i64 = RELATION.iter().map(|&(_, c)| c.abs()).sum();
            let mut swb = Swb::new(42);
            for _ in 0..10000 {
                let block: Vec<u32> = (0..2 * P).map(|_| swb.step()).collect();
                let sum: i64 = RELATION
                    .iter()
                    .map(|&(o, c)| {
                        assert!(o % P < R, "offset {o} is not kept");
                        c * block[o] as i64
                    })
                    .sum();
                let residue = (sum + (1 << 23)).rem_euclid(1 << W) - (1 << 23);
                assert!(residue.abs() <= l1, "residue {residue}, L1 {l1}");
            }
        }
    }
}

#[cfg(feature = "_ranlux")]
pub use ranlux::Prng;

// ----- SplitMix -------------------------------------------------------------------

#[cfg(feature = "splitmix")]
#[derive(Clone, Copy)]
pub struct Prng {
    x: u64,
}

#[cfg(feature = "splitmix")]
impl Prng {
    pub const NAME: &str = "SplitMix";
    pub fn new(seed: u64) -> Self {
        Self { x: seed }
    }

    #[inline(always)]
    pub fn next_u64(&mut self) -> u64 {
        const PHI: u64 = 0x9e3779b97f4a7c15;
        let mut z = self.x;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        self.x = self.x.wrapping_add(PHI);
        z ^ (z >> 31)
    }

    pub fn try_skip(&mut self, n: u64) -> Result<(), ()> {
        const PHI: u64 = 0x9e3779b97f4a7c15;
        self.x = self.x.wrapping_add(n.wrapping_mul(PHI));
        Ok(())
    }
}

// ----- Sirius64 ------------------------------------------------------------------
//
// https://github.com/matteo65/Sirius64

#[cfg(feature = "sirius64")]
#[derive(Clone, Copy)]
pub struct Prng {
    x: u64,
}

#[cfg(feature = "sirius64")]
impl Prng {
    pub const NAME: &str = "Sirius64";
    pub fn new(seed: u64) -> Self {
        Self { x: seed }
    }

    #[inline(always)]
    pub fn next_u64(&mut self) -> u64 {
        const PHI: u64 = 0x9e3779b97f4a7c15;
        self.x = self.x.wrapping_add(PHI);
        let mut z = self.x;
        z = PHI.wrapping_mul(z ^ (z >> 17));
        z = z.rotate_left(32);
        PHI.wrapping_mul(self.x ^ z ^ (z >> 17))
    }

    pub fn try_skip(&mut self, n: u64) -> Result<(), ()> {
        const PHI: u64 = 0x9e3779b97f4a7c15;
        self.x = self.x.wrapping_add(n.wrapping_mul(PHI));
        Ok(())
    }
}

// ----- Trivial counters (sanity/baseline) ---------------------------------------------

#[cfg(feature = "incr")]
#[derive(Clone, Copy)]
pub struct Prng {
    x: u64,
}

#[cfg(feature = "incr")]
impl Prng {
    pub const NAME: &str = "incr (counter, x += 1)";
    pub fn new(seed: u64) -> Self {
        Self { x: seed }
    }

    #[inline(always)]
    pub fn next_u64(&mut self) -> u64 {
        self.x = self.x.wrapping_add(1);
        self.x
    }

    pub fn try_skip(&mut self, n: u64) -> Result<(), ()> {
        self.x = self.x.wrapping_add(n);
        Ok(())
    }
}

// ----- wyrand -------------------------------------------------------------------------

#[cfg(feature = "wyrand")]
#[derive(Clone, Copy)]
pub struct Prng {
    x: u64,
}

#[cfg(feature = "wyrand")]
impl Prng {
    pub const NAME: &str = "wyrand";
    pub fn new(seed: u64) -> Self {
        Self { x: seed }
    }

    #[inline(always)]
    pub fn next_u64(&mut self) -> u64 {
        let z = self.x;
        self.x = self.x.wrapping_add(0xa0761d6478bd642f);
        let t = ((z ^ 0xe7037ed1a0b428db) as u128).wrapping_mul(z as u128);
        ((t >> 64) ^ t) as u64
    }

    pub fn try_skip(&mut self, n: u64) -> Result<(), ()> {
        self.x = self.x.wrapping_add(n.wrapping_mul(0xa0761d6478bd642f));
        Ok(())
    }
}

// ----- wyrand variant (https://github.com/wangyi-fudan/wyhash/issues/130#issuecomment-4835746792) -----------

#[cfg(feature = "wyrand-reinerp")]
#[derive(Clone, Copy)]
pub struct Prng {
    x: u64,
}

#[cfg(feature = "wyrand-reinerp")]
impl Prng {
    pub const NAME: &str = "wyrand (reinerp)";
    pub fn new(seed: u64) -> Self {
        Self { x: seed }
    }

    #[inline(always)]
    pub fn next_u64(&mut self) -> u64 {
        let y = self.x;
        self.x = self.x.wrapping_add(0xa0761d6478bd642f);
        let t = ((y ^ 0xe7037ed1a0b428db) as u128).wrapping_mul(self.x as u128);
        ((t >> 64) ^ t) as u64
    }
    pub fn try_skip(&mut self, n: u64) -> Result<(), ()> {
        self.x = self.x.wrapping_add(n.wrapping_mul(0xa0761d6478bd642f));
        Ok(())
    }
}

// ----- Romu family --------------------------------------------------------------------

#[cfg(feature = "romuduo")]
#[derive(Clone, Copy)]
pub struct Prng {
    x: u64,
    y: u64,
}

#[cfg(feature = "romuduo")]
impl Prng {
    pub const NAME: &str = "RomuDuo";
    pub fn new(seed: u64) -> Self {
        // Romu requires a non-zero state; remap the degenerate all-zero seed.
        let seed = if seed == 0 { 0x9e3779b97f4a7c15 } else { seed };
        Self { x: seed, y: seed }
    }

    #[inline(always)]
    pub fn next_u64(&mut self) -> u64 {
        let xp = self.x;
        self.x = 15241094284759029579u64.wrapping_mul(self.y);
        self.y = u64::rotate_left(self.y, 36)
            .wrapping_add(u64::rotate_left(self.y, 15))
            .wrapping_sub(xp);
        xp
    }

    pub fn try_skip(&mut self, _n: u64) -> Result<(), ()> {
        Err(())
    }
}

#[cfg(feature = "romuduojr")]
#[derive(Clone, Copy)]
pub struct Prng {
    x: u64,
    y: u64,
}

#[cfg(feature = "romuduojr")]
impl Prng {
    pub const NAME: &str = "RomuDuoJr";
    pub fn new(seed: u64) -> Self {
        // Romu requires a non-zero state; remap the degenerate all-zero seed.
        let seed = if seed == 0 { 0x9e3779b97f4a7c15 } else { seed };
        Self { x: seed, y: seed }
    }

    #[inline(always)]
    pub fn next_u64(&mut self) -> u64 {
        let xp = self.x;
        self.x = 15241094284759029579u64.wrapping_mul(self.y);
        self.y = self.y.wrapping_sub(xp);
        self.y = u64::rotate_left(self.y, 27);
        xp
    }

    pub fn try_skip(&mut self, _n: u64) -> Result<(), ()> {
        Err(())
    }
}

#[cfg(feature = "romutrio")]
#[derive(Clone, Copy)]
pub struct Prng {
    x: u64,
    y: u64,
    z: u64,
}

#[cfg(feature = "romutrio")]
impl Prng {
    pub const NAME: &str = "RomuTrio";
    pub fn new(seed: u64) -> Self {
        Self {
            x: seed,
            y: 1,
            z: 1,
        }
    }

    #[inline(always)]
    pub fn next_u64(&mut self) -> u64 {
        let (xp, yp, zp) = (self.x, self.y, self.z);
        self.x = 15241094284759029579u64.wrapping_mul(zp);
        self.y = yp.wrapping_sub(xp);
        self.y = u64::rotate_left(self.y, 12);
        self.z = zp.wrapping_sub(yp);
        self.z = u64::rotate_left(self.z, 44);
        xp
    }

    pub fn try_skip(&mut self, _n: u64) -> Result<(), ()> {
        Err(())
    }
}

#[cfg(feature = "romutrio32")]
#[derive(Clone, Copy)]
pub struct Prng {
    x: u32,
    y: u32,
    z: u32,
}

#[cfg(feature = "romutrio32")]
impl Prng {
    pub const NAME: &str = "RomuTrio32";
    pub fn new(seed: u64) -> Self {
        // Romu requires a non-zero state; remap the degenerate all-zero seed.
        let seed = if seed == 0 { 0x9e3779b97f4a7c15 } else { seed };
        Self {
            x: seed as u32,
            y: (seed >> 32) as u32,
            z: 0,
        }
    }

    #[inline(always)]
    pub fn next_u64(&mut self) -> u64 {
        let (xp, yp, zp) = (self.x, self.y, self.z);
        self.x = 3323815723u32.wrapping_mul(zp);
        self.y = yp.wrapping_sub(xp);
        self.y = u32::rotate_left(self.y, 6);
        self.z = zp.wrapping_sub(yp);
        self.z = u32::rotate_left(self.z, 22);
        (xp as u64) << 32
    }

    pub fn try_skip(&mut self, _n: u64) -> Result<(), ()> {
        Err(())
    }
}

// ----- 32-bit LCGs --------------------------------------------------------------------

#[cfg(feature = "lcg_32_32_0xec65035")]
#[derive(Clone, Copy)]
pub struct Prng {
    x: u32,
}

#[cfg(feature = "lcg_32_32_0xec65035")]
impl Prng {
    pub const NAME: &str = "LCG32 (0xec65035)";
    pub fn new(seed: u64) -> Self {
        Self { x: seed as u32 }
    }

    #[inline(always)]
    pub fn next_u64(&mut self) -> u64 {
        self.x = self.x.wrapping_mul(0xec65035).wrapping_add(1);
        (self.x as u64) << 32
    }

    pub fn try_skip(&mut self, n: u64) -> Result<(), ()> {
        const A: u32 = 0xec65035;
        // Compose the affine map (mul, add) = (A, 1) with itself n times by
        // binary exponentiation; (am, ac) accumulates the result, (ai, ci) the
        // current 2ᵏ-th power. compose((m1,c1),(m2,c2)) = (m1*m2, m2*c1 + c2).
        let (mut am, mut ac) = (1u32, 0u32);
        let (mut ai, mut ci) = (A, 1u32);
        let mut k = n;
        while k > 0 {
            if k & 1 == 1 {
                ac = ac.wrapping_mul(ai).wrapping_add(ci);
                am = am.wrapping_mul(ai);
            }
            ci = ci.wrapping_mul(ai.wrapping_add(1));
            ai = ai.wrapping_mul(ai);
            k >>= 1;
        }
        self.x = am.wrapping_mul(self.x).wrapping_add(ac);
        Ok(())
    }
}

#[cfg(feature = "lcg_32_32_0x915f77f5")]
#[derive(Clone, Copy)]
pub struct Prng {
    x: u32,
}

#[cfg(feature = "lcg_32_32_0x915f77f5")]
impl Prng {
    pub const NAME: &str = "LCG32 (0x915f77f5)";
    pub fn new(seed: u64) -> Self {
        Self { x: seed as u32 }
    }

    #[inline(always)]
    pub fn next_u64(&mut self) -> u64 {
        self.x = self.x.wrapping_mul(0x915f77f5).wrapping_add(1);
        (self.x as u64) << 32
    }

    pub fn try_skip(&mut self, n: u64) -> Result<(), ()> {
        const A: u32 = 0x915f77f5;
        // Compose the affine map (mul, add) = (A, 1) with itself n times by
        // binary exponentiation; (am, ac) accumulates the result, (ai, ci) the
        // current 2ᵏ-th power. compose((m1,c1),(m2,c2)) = (m1*m2, m2*c1 + c2).
        let (mut am, mut ac) = (1u32, 0u32);
        let (mut ai, mut ci) = (A, 1u32);
        let mut k = n;
        while k > 0 {
            if k & 1 == 1 {
                ac = ac.wrapping_mul(ai).wrapping_add(ci);
                am = am.wrapping_mul(ai);
            }
            ci = ci.wrapping_mul(ai.wrapping_add(1));
            ai = ai.wrapping_mul(ai);
            k >>= 1;
        }
        self.x = am.wrapping_mul(self.x).wrapping_add(ac);
        Ok(())
    }
}

// ----- 64-bit LCGs --------------------------------------------------------------------

// Truncated 64-bit LCG x ↦ A·x + 1 over the full 64-bit word, parameterized by the
// multiplier A. The macro lets new multipliers be added with a single gated
// invocation (see below), exactly like the MWC families above.
#[cfg(feature = "lcg_64_64_0xa5b9ee81534fa94d")]
macro_rules! lcg64 {
    ($a:expr) => {
        const LCG64_A: u64 = $a;

        #[derive(Clone, Copy)]
        pub struct Prng {
            x: u64,
        }

        impl Prng {
            pub const NAME: &str = concat!("LCG64 (", stringify!($a), ")");

            pub fn new(seed: u64) -> Self {
                Self { x: seed }
            }

            #[inline(always)]
            pub fn next_u64(&mut self) -> u64 {
                // The 64-bit state already fills the word; the test reads the top
                // u bits, which are the high-quality bits of the LCG (the low bits
                // have short periods), so no left-justifying shift is needed.
                self.x = self.x.wrapping_mul(LCG64_A).wrapping_add(1);
                self.x
            }

            pub fn try_skip(&mut self, n: u64) -> Result<(), ()> {
                // Compose the affine map (mul, add) = (LCG64_A, 1) with itself n
                // times by binary exponentiation; (am, ac) accumulates the result,
                // (ai, ci) the current 2ᵏ-th power.
                // compose((m1,c1),(m2,c2)) = (m1*m2, m2*c1 + c2).
                let (mut am, mut ac) = (1u64, 0u64);
                let (mut ai, mut ci) = (LCG64_A, 1u64);
                let mut k = n;
                while k > 0 {
                    if k & 1 == 1 {
                        ac = ac.wrapping_mul(ai).wrapping_add(ci);
                        am = am.wrapping_mul(ai);
                    }
                    ci = ci.wrapping_mul(ai.wrapping_add(1));
                    ai = ai.wrapping_mul(ai);
                    k >>= 1;
                }
                self.x = am.wrapping_mul(self.x).wrapping_add(ac);
                Ok(())
            }
        }
    };
}

// f₂ = 0.977689 (figure of merit in two dimensions)
#[cfg(feature = "lcg_64_64_0xa5b9ee81534fa94d")]
lcg64!(0xa5b9ee81534fa94d);

// ----- Middle-square Weyl-sequence (Widynski) counter variant -------------------------

#[cfg(feature = "MSWS-CTR")]
#[derive(Clone, Copy)]
pub struct Prng {
    ctr: u64,
}

#[cfg(feature = "MSWS-CTR")]
impl Prng {
    pub const NAME: &str = "MSWS-CTR (middle-square Weyl sequence, counter-based)";
    pub fn new(seed: u64) -> Self {
        Self { ctr: seed }
    }

    #[inline(always)]
    pub fn next_u64(&mut self) -> u64 {
        self.ctr = self.ctr.wrapping_add(1);
        let x = self.ctr.wrapping_mul(0x9e3779b97f4a7c15);
        let y = x;
        let z = y.wrapping_add(0x9e3779b97f4a7c15);
        let mut x = x.wrapping_mul(x).wrapping_add(y);
        x = x.rotate_left(32); // round 1
        x = x.wrapping_mul(x).wrapping_add(z);
        x = x.rotate_left(32); // round 2
        x = x.wrapping_mul(x).wrapping_add(y);
        x = x.rotate_left(32); // round 3
        x = x.wrapping_mul(x).wrapping_add(z);
        let t = x;
        x = x.rotate_left(32); // round 4
        t ^ (x.wrapping_mul(x).wrapping_add(y) >> 32) // round 5
    }

    pub fn try_skip(&mut self, n: u64) -> Result<(), ()> {
        self.ctr = self.ctr.wrapping_add(n);
        Ok(())
    }
}

#[cfg(test)]
mod skip_tests {
    use super::*;

    // A single generic test that works for every feature-selected generator.
    // For skip-capable generators it checks that try_skip(n) lands on the same
    // state as n sequential next_u64() calls, and that two skips compose.
    // For generators without jump-ahead (try_skip returns Err), there is
    // nothing to verify and the test passes trivially.
    #[test]
    fn test_skip_matches_repeated_next() {
        let seed = 0x0123_4567_89ab_cdef;
        // Cheap-to-step values (sequential stepping must stay fast).
        for &n in &[0u64, 1, 2, 7, 1000, 100_000] {
            let mut a = Prng::new(seed);
            if a.try_skip(n).is_err() {
                return; // generator has no jump-ahead; nothing to check
            }
            let mut b = Prng::new(seed);
            for _ in 0..n {
                b.next_u64();
            }
            for k in 0..64 {
                assert_eq!(
                    a.next_u64(),
                    b.next_u64(),
                    "skip({n}) disagreed with stepping at output {k}"
                );
            }
        }
        // Composition: two successive skips must equal one combined skip. Trivial
        // for counter generators; exercises the doubling (LCG) and modpow (MWC)
        // jump-ahead with large exponents.
        let (x, y) = (1u64 << 40, (1u64 << 41) + 12_345);
        let skip_err = "try_skip failed on a skip-capable generator";
        let mut p = Prng::new(seed);
        p.try_skip(x).expect(skip_err);
        p.try_skip(y).expect(skip_err);
        let mut q = Prng::new(seed);
        q.try_skip(x + y).expect(skip_err);
        for k in 0..64 {
            assert_eq!(
                p.next_u64(),
                q.next_u64(),
                "skip composition failed at output {k}"
            );
        }
    }
}
