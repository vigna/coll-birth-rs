/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

//! Pseudorandom number generators selected at build time via Cargo features.
//!
//! Exactly one feature must be enabled when building the crate; each variant
//! exposes a single [`Prng`] type with a `new(seed: u64) -> Self` constructor,
//! a `next_u64(&mut self) -> u64` step function, and a `try_skip(&mut self, n:
//! u64) -> Result<(), ()>` method to skip ahead by `n` steps.
//!
//! `try_skip(n)` must have the same effect as `n` calls to `next_u64`. It must
//! succeed for every `n`, or fail for every `n` (including 0) without changing
//! the state: the parallel runners probe the capability with `try_skip(0)`.
//!
//! Generators outputting less than 64 bits must shift their outputs to the top
//! (e.g., 32-bit generators must return their output shifted to the left by 32).
//!
//! The `MWCX` feature, which must be combined with an MWC feature, applies the
//! scrambler X to the output of the MWC generator: the output digit is xored
//! with the carry of the same state (as in Thomas's MWC64X), and the name of
//! the generator gets an `X` suffix.
//!
//! The `LACUNARY` feature, which must be combined with an MWC feature (and
//! possibly with `MWCX`), returns, from each block of *k* + 2 consecutive
//! outputs of an MWC generator with lag *k*, the outputs of index 0, 1, and
//! *k* + 1, so that a three-dimensional test (which is the only one accepted)
//! tests the lacunary triples (*xₙ*, *xₙ*₊₁, *xₙ*₊*ₖ*₊₁), which lie in planes at
//! a distance similar to that of the successive triples of an MWC generator
//! with lag one.

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

#[cfg(all(feature = "LACUNARY", not(feature = "_mwc")))]
compile_error!(
    "the LACUNARY feature applies only to MWC generators: combine it with an mwc_* feature"
);

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
                // Discard *x*ₙ₊₂, …, *x*ₙ₊ₖ
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
        fn triples() {
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
        fn skip() {
            for start in 0..7 {
                for n in [0, 1, 2, 3, 4, 5, 6, 7, 299, 300, 301, 1000] {
                    let mut skipped = Prng::new(42);
                    let mut stepped = Prng::new(42);
                    for _ in 0..start {
                        skipped.next_u64();
                        stepped.next_u64();
                    }
                    skipped.try_skip(n).unwrap();
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
    fn test_skip_matches_repeated_next() -> anyhow::Result<()> {
        let seed = 0x0123_4567_89ab_cdef;
        // Cheap-to-step values (sequential stepping must stay fast).
        for &n in &[0u64, 1, 2, 7, 1000, 100_000] {
            let mut a = Prng::new(seed);
            if a.try_skip(n).is_err() {
                return Ok(()); // generator has no jump-ahead; nothing to check
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
        // jump-ahead with large exponents once those generators gain try_skip.
        let (x, y) = (1u64 << 40, (1u64 << 41) + 12_345);
        let skip_err = |()| anyhow::anyhow!("try_skip failed on a skip-capable generator");
        let mut p = Prng::new(seed);
        p.try_skip(x).map_err(skip_err)?;
        p.try_skip(y).map_err(skip_err)?;
        let mut q = Prng::new(seed);
        q.try_skip(x + y).map_err(skip_err)?;
        for k in 0..64 {
            assert_eq!(
                p.next_u64(),
                q.next_u64(),
                "skip composition failed at output {k}"
            );
        }
        Ok(())
    }
}
