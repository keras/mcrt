//! RT-4: Image comparison and perceptual diff metrics for render regression testing.
//!
//! ## Usage
//!
//! ```ignore
//! let a = image::open("current/diffuse_sphere.png")?;
//! let b = image::open("baseline/diffuse_sphere.png")?;
//! let result = regression::compare_images(&a, &b)?;
//! println!("MSE={:.6}  PSNR={:.2}dB  max_delta={:.4}", result.mse, result.psnr, result.max_delta);
//! if result.mse > threshold {
//!     regression::write_diff_image(&a, &b, "diff/diffuse_sphere_diff.png", 4.0)?;
//! }
//! ```
//!
//! ## Metric definitions
//!
//! Pixel values are normalised to **[0.0, 1.0]** before any arithmetic.  All
//! metrics are computed in the encoded (gamma/sRGB) space that the PNG carries:
//! this matches visual inspection and avoids requiring a colour-profile look-up
//! table.  Only the R, G, and B channels are compared; alpha is ignored.
//!
//! | Metric | Formula | Pass condition |
//! |--------|---------|----------------|
//! | MSE    | mean of (pa − pb)² over all pixels and channels | `mse ≤ threshold_mse` |
//! | PSNR   | 10 · log₁₀(1 / MSE) dB, `+∞` when MSE = 0 | `psnr ≥ threshold_psnr` |
//! | max delta | max |pa − pb| over all pixels and channels | informational |

#![allow(dead_code)] // Public API used from integration tests via lib crate; also compiled into bin

use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
use std::borrow::Cow;
use std::path::Path;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Metrics produced by [`compare_images`].
#[derive(Debug, Clone, PartialEq)]
pub struct DiffResult {
    /// Mean Squared Error per channel, in the normalised [0.0, 1.0] range.
    /// Averaged over all (pixel, channel) pairs.  Lower is better; 0.0 means
    /// the images are identical.
    pub mse: f64,

    /// Peak Signal-to-Noise Ratio in dB.  Higher is better.
    /// Defined as `10 · log₁₀(1 / mse)`.  `f64::INFINITY` when `mse == 0`.
    pub psnr: f64,

    /// Maximum absolute per-channel difference across all pixels and channels,
    /// in the normalised [0.0, 1.0] range.  Informational; not gated by a
    /// threshold in standard regression checks.
    pub max_delta: f64,
}

/// Returned when two images have incompatible (width, height) dimensions.
#[derive(Debug, Clone)]
pub struct DimensionMismatch {
    /// Dimensions of the first image (width × height).
    pub a: (u32, u32),
    /// Dimensions of the second image (width × height).
    pub b: (u32, u32),
}

impl std::fmt::Display for DimensionMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "dimension mismatch: image A is {}×{}, image B is {}×{}",
            self.a.0, self.a.1, self.b.0, self.b.1
        )
    }
}

impl std::error::Error for DimensionMismatch {}

/// Errors returned by [`write_diff_image`].
#[derive(Debug)]
pub enum RegressionError {
    /// The two input images have different dimensions.
    DimensionMismatch(DimensionMismatch),
    /// An encoding or I/O error while writing the diff image.
    Image(image::ImageError),
}

impl std::fmt::Display for RegressionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegressionError::DimensionMismatch(e) => write!(f, "{e}"),
            RegressionError::Image(e) => write!(f, "image error: {e}"),
        }
    }
}

impl std::error::Error for RegressionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            RegressionError::DimensionMismatch(e) => Some(e),
            RegressionError::Image(e) => Some(e),
        }
    }
}

impl From<image::ImageError> for RegressionError {
    fn from(e: image::ImageError) -> Self {
        RegressionError::Image(e)
    }
}

// ---------------------------------------------------------------------------
// Core comparison
// ---------------------------------------------------------------------------

/// Compare two images and return [`DiffResult`] containing MSE, PSNR, and
/// max delta.
///
/// Only the R, G, and B channels are compared; alpha is ignored.
/// Returns [`Err(DimensionMismatch)`] immediately when the images differ in
/// size — callers should treat this as a hard failure before checking thresholds.
pub fn compare_images(a: &DynamicImage, b: &DynamicImage) -> Result<DiffResult, DimensionMismatch> {
    let (wa, ha) = a.dimensions();
    let (wb, hb) = b.dimensions();
    if wa != wb || ha != hb {
        return Err(DimensionMismatch {
            a: (wa, ha),
            b: (wb, hb),
        });
    }

    // Zero-size images are trivially identical.
    if wa == 0 || ha == 0 {
        return Ok(DiffResult {
            mse: 0.0,
            psnr: f64::INFINITY,
            max_delta: 0.0,
        });
    }

    let a_rgba = match a.as_rgba8() {
        Some(b) => Cow::Borrowed(b),
        None => Cow::Owned(a.to_rgba8()),
    };
    let b_rgba = match b.as_rgba8() {
        Some(b) => Cow::Borrowed(b),
        None => Cow::Owned(b.to_rgba8()),
    };

    let mut sum_sq: f64 = 0.0;
    let mut max_delta: f64 = 0.0;
    const N_CHANNELS: usize = 3; // RGB only

    for y in 0..ha {
        for x in 0..wa {
            let pa = a_rgba.get_pixel(x, y);
            let pb = b_rgba.get_pixel(x, y);
            for c in 0..N_CHANNELS {
                let va = pa[c] as f64 / 255.0;
                let vb = pb[c] as f64 / 255.0;
                let diff = (va - vb).abs();
                sum_sq += diff * diff;
                if diff > max_delta {
                    max_delta = diff;
                }
            }
        }
    }

    let total_samples = (wa as f64) * (ha as f64) * N_CHANNELS as f64;
    let mse = sum_sq / total_samples;
    let psnr = if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (1.0_f64 / mse).log10()
    };

    Ok(DiffResult {
        mse,
        psnr,
        max_delta,
    })
}

// ---------------------------------------------------------------------------
// Diff image
// ---------------------------------------------------------------------------

/// Write a false-colour difference image to `path`.
///
/// Each output pixel maps the maximum per-channel absolute error at that pixel
/// to a heat-map palette so that even subtle differences are visible:
///
/// | Normalised error × `amplify` | Colour |
/// |------------------------------|--------|
/// | 0.00                         | black  |
/// | 0.25                         | blue   |
/// | 0.50                         | green  |
/// | 0.75                         | yellow |
/// | 1.00 (clamped)               | red    |
///
/// `amplify` scales errors before palette lookup.  A factor of `4.0` makes a
/// difference of 1/4 of the full range appear as maximum red — suitable for
/// catching the typical noise-level regressions in 64–256 SPP renders.  Pass
/// `1.0` for a non-amplified diff.
///
/// Returns `Err` with a [`DimensionMismatch`] description when the images
/// differ in size, or any I/O error from writing the PNG.
pub fn write_diff_image(
    a: &DynamicImage,
    b: &DynamicImage,
    path: &Path,
    amplify: f32,
) -> Result<(), RegressionError> {
    let (wa, ha) = a.dimensions();
    let (wb, hb) = b.dimensions();
    if wa != wb || ha != hb {
        return Err(RegressionError::DimensionMismatch(DimensionMismatch {
            a: (wa, ha),
            b: (wb, hb),
        }));
    }

    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .map_err(|e| RegressionError::Image(image::ImageError::IoError(e)))?;
    }

    let a_rgba = match a.as_rgba8() {
        Some(b) => Cow::Borrowed(b),
        None => Cow::Owned(a.to_rgba8()),
    };
    let b_rgba = match b.as_rgba8() {
        Some(b) => Cow::Borrowed(b),
        None => Cow::Owned(b.to_rgba8()),
    };
    let mut out: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(wa, ha);

    for y in 0..ha {
        for x in 0..wa {
            let pa = a_rgba.get_pixel(x, y);
            let pb = b_rgba.get_pixel(x, y);

            // Maximum per-channel absolute error, normalised to [0, 1].
            let mut max_err: f32 = 0.0;
            for c in 0..3usize {
                let diff = (pa[c] as f32 - pb[c] as f32).abs() / 255.0;
                if diff > max_err {
                    max_err = diff;
                }
            }

            let t = (max_err * amplify).clamp(0.0, 1.0);
            out.put_pixel(x, y, Rgb(heat_map(t)));
        }
    }

    out.save(path)?;
    Ok(())
}

/// Map a scalar `t ∈ [0, 1]` to an RGB heat-map colour.
///
/// Palette control points (linearly interpolated):
/// - 0.00 → black  (0, 0, 0)
/// - 0.25 → blue   (0, 0, 255)
/// - 0.50 → green  (0, 255, 0)
/// - 0.75 → yellow (255, 255, 0)
/// - 1.00 → red    (255, 0, 0)
#[inline]
fn heat_map(t: f32) -> [u8; 3] {
    let (r, g, b) = if t < 0.25 {
        let s = t * 4.0;
        (0.0_f32, 0.0_f32, s)
    } else if t < 0.5 {
        let s = (t - 0.25) * 4.0;
        (0.0, s, 1.0 - s)
    } else if t < 0.75 {
        let s = (t - 0.5) * 4.0;
        (s, 1.0, 0.0)
    } else {
        let s = (t - 0.75) * 4.0;
        (1.0, 1.0 - s, 0.0)
    };
    [
        (r * 255.0 + 0.5) as u8,
        (g * 255.0 + 0.5) as u8,
        (b * 255.0 + 0.5) as u8,
    ]
}

// ---------------------------------------------------------------------------
// Sidecar threshold loading
// ---------------------------------------------------------------------------

/// Per-scene regression thresholds loaded from the `[regression]` section of a
/// `.test.toml` sidecar file.
#[derive(Debug, Clone)]
pub struct SceneThresholds {
    /// Maximum allowed MSE (mean squared error, normalised 0–1 range).
    /// The test **fails** when `mse > threshold_mse`.
    pub threshold_mse: f64,
    /// Minimum required PSNR in dB.
    /// The test **fails** when `psnr < threshold_psnr` (and `psnr` is finite).
    pub threshold_psnr: f64,
}

impl Default for SceneThresholds {
    fn default() -> Self {
        SceneThresholds {
            threshold_mse: 0.002,
            threshold_psnr: 38.0,
        }
    }
}

#[derive(serde::Deserialize, Default)]
struct RegressionSection {
    threshold_mse: Option<f64>,
    threshold_psnr: Option<f64>,
}

#[derive(serde::Deserialize, Default)]
struct ThresholdSidecar {
    regression: Option<RegressionSection>,
}

/// Load regression thresholds from the `.test.toml` sidecar adjacent to
/// `scene_yaml`.
///
/// For `diffuse_sphere.yaml`, looks for `diffuse_sphere.test.toml` in the same
/// directory.  Falls back to [`SceneThresholds::default`] on any I/O or parse
/// error so missing sidecars are silently ignored.
pub fn load_thresholds(scene_yaml: &Path) -> SceneThresholds {
    let sidecar = scene_yaml.with_extension("").with_extension("test.toml");
    let text = match std::fs::read_to_string(&sidecar) {
        Ok(t) => t,
        Err(_) => return SceneThresholds::default(),
    };
    let parsed: ThresholdSidecar = toml::from_str(&text).unwrap_or_default();
    let reg = parsed.regression.unwrap_or_default();
    let defaults = SceneThresholds::default();
    SceneThresholds {
        threshold_mse: reg.threshold_mse.unwrap_or(defaults.threshold_mse),
        threshold_psnr: reg.threshold_psnr.unwrap_or(defaults.threshold_psnr),
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgba};

    /// Create a solid-colour RGBA image of the given size.
    fn solid(w: u32, h: u32, r: u8, g: u8, b: u8, a: u8) -> DynamicImage {
        let buf = ImageBuffer::from_fn(w, h, |_, _| Rgba([r, g, b, a]));
        DynamicImage::ImageRgba8(buf)
    }

    // ------------------------------------------------------------------
    // compare_images — correctness
    // ------------------------------------------------------------------

    #[test]
    fn identical_images_give_zero_mse() {
        let a = solid(4, 4, 128, 64, 255, 255);
        let b = solid(4, 4, 128, 64, 255, 255);
        let res = compare_images(&a, &b).unwrap();
        assert_eq!(res.mse, 0.0);
        assert_eq!(res.psnr, f64::INFINITY);
        assert_eq!(res.max_delta, 0.0);
    }

    #[test]
    fn black_vs_white_mse_is_one() {
        // All-black vs all-white: every channel diff = 255/255 = 1.0.
        // MSE = mean((1.0)²) = 1.0
        let a = solid(2, 2, 0, 0, 0, 255);
        let b = solid(2, 2, 255, 255, 255, 255);
        let res = compare_images(&a, &b).unwrap();
        assert!(
            (res.mse - 1.0).abs() < 1e-12,
            "expected MSE=1.0, got {}",
            res.mse
        );
        assert!(
            (res.psnr - 0.0).abs() < 1e-9,
            "expected PSNR=0 dB, got {}",
            res.psnr
        );
        assert!(
            (res.max_delta - 1.0).abs() < 1e-12,
            "expected max_delta=1.0, got {}",
            res.max_delta
        );
    }

    #[test]
    fn single_channel_one_lsb_off() {
        // One pixel differs by exactly 1 LSB on the red channel.
        // diff = 1/255; MSE = (1/255)^2 / 3  (3 channels, only R differs)
        let a = solid(1, 1, 128, 0, 0, 255);
        let b = solid(1, 1, 129, 0, 0, 255);

        let expected_diff = 1.0_f64 / 255.0;
        let expected_mse = (expected_diff * expected_diff) / 3.0;
        let expected_psnr = 10.0 * (1.0 / expected_mse).log10();

        let res = compare_images(&a, &b).unwrap();
        assert!(
            (res.mse - expected_mse).abs() < 1e-15,
            "MSE mismatch: got {}, expected {}",
            res.mse,
            expected_mse
        );
        assert!(
            (res.psnr - expected_psnr).abs() < 1e-6,
            "PSNR mismatch: got {}, expected {}",
            res.psnr,
            expected_psnr
        );
        assert!(
            (res.max_delta - expected_diff).abs() < 1e-15,
            "max_delta mismatch: got {}, expected {}",
            res.max_delta,
            expected_diff
        );
    }

    #[test]
    fn alpha_channel_is_ignored() {
        // Images differ only in alpha — MSE must be 0.
        let a = solid(2, 2, 100, 100, 100, 0);
        let b = solid(2, 2, 100, 100, 100, 255);
        let res = compare_images(&a, &b).unwrap();
        assert_eq!(res.mse, 0.0, "alpha differences should be ignored");
    }

    #[test]
    fn dimension_mismatch_returns_error() {
        let a = solid(4, 4, 0, 0, 0, 255);
        let b = solid(8, 4, 0, 0, 0, 255);
        let err = compare_images(&a, &b).unwrap_err();
        assert_eq!(err.a, (4, 4));
        assert_eq!(err.b, (8, 4));
    }

    // ------------------------------------------------------------------
    // heat_map — sanity checks
    // ------------------------------------------------------------------

    #[test]
    fn heat_map_endpoints() {
        assert_eq!(heat_map(0.0), [0, 0, 0], "0 should be black");
        assert_eq!(heat_map(1.0), [255, 0, 0], "1 should be red");
    }

    #[test]
    fn heat_map_control_points() {
        // At t=0.25 the palette should be blue (0, 0, 255).
        let [r, g, b] = heat_map(0.25);
        assert_eq!(r, 0);
        assert_eq!(g, 0);
        assert_eq!(b, 255);

        // At t=0.5 the palette should be green (0, 255, 0).
        let [r, g, b] = heat_map(0.5);
        assert_eq!(r, 0);
        assert_eq!(b, 0);
        assert_eq!(g, 255);

        // At t=0.75 the palette should be yellow (255, 255, 0).
        let [r, g, b] = heat_map(0.75);
        assert_eq!(b, 0);
        assert_eq!(r, 255);
        assert_eq!(g, 255);
    }

    // ------------------------------------------------------------------
    // write_diff_image — round-trip test
    // ------------------------------------------------------------------

    #[test]
    fn write_diff_image_creates_file() {
        let a = solid(8, 8, 0, 0, 0, 255);
        let b = solid(8, 8, 255, 255, 255, 255);
        let path = std::path::Path::new("target/test_diff_output.png");
        // Remove any stale file so existence really proves this run wrote it.
        let _ = std::fs::remove_file(path);
        write_diff_image(&a, &b, path, 1.0).expect("write_diff_image should succeed");
        assert!(path.exists(), "diff image should have been written");

        // The entire image should be fully-red (max error, not amplified past 1).
        let img = image::open(path).unwrap().to_rgb8();
        let px = img.get_pixel(0, 0);
        assert_eq!(px[0], 255, "red channel should be 255");
        assert_eq!(px[1], 0, "green channel should be 0");
        assert_eq!(px[2], 0, "blue channel should be 0");
    }

    #[test]
    fn write_diff_image_identical_is_black() {
        let a = solid(4, 4, 128, 128, 128, 255);
        let path = std::path::Path::new("target/test_diff_black.png");
        write_diff_image(&a, &a, path, 4.0).expect("write_diff_image should succeed");

        let img = image::open(path).unwrap().to_rgb8();
        let px = img.get_pixel(0, 0);
        assert_eq!(px[0], 0);
        assert_eq!(px[1], 0);
        assert_eq!(px[2], 0);
    }

    #[test]
    fn write_diff_image_dimension_mismatch_returns_error() {
        let a = solid(4, 4, 0, 0, 0, 255);
        let b = solid(8, 4, 0, 0, 0, 255);
        let path = std::path::Path::new("target/test_diff_should_not_exist.png");
        assert!(
            write_diff_image(&a, &b, path, 1.0).is_err(),
            "dimension mismatch should return an error"
        );
    }

    // ------------------------------------------------------------------
    // compare_images — edge cases
    // ------------------------------------------------------------------

    // ------------------------------------------------------------------
    // Real-scene sanity check (uses on-disk baselines; run explicitly)
    // ------------------------------------------------------------------

    /// Verify metrics against the on-disk baselines created by RT-3.
    ///
    /// Run with: `cargo test real_scene -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn real_scene_pair_metrics() {
        let base = "tmp/regression/baselines/0c186bc";
        let sphere_path = format!("{base}/diffuse_sphere.png");
        let box_path = format!("{base}/emissive_box.png");

        // Self-comparison must be exactly zero.
        let sphere = image::open(&sphere_path).expect("open diffuse_sphere baseline");
        let same = compare_images(&sphere, &sphere).unwrap();
        assert_eq!(same.mse, 0.0, "self-comparison MSE must be 0");
        assert_eq!(same.psnr, f64::INFINITY);

        // Cross-scene comparison must produce a non-trivial, finite MSE.
        let emissive = image::open(&box_path).expect("open emissive_box baseline");
        let cross = compare_images(&sphere, &emissive).unwrap();
        println!(
            "cross-scene: MSE={:.6}  PSNR={:.2} dB  max_delta={:.4}",
            cross.mse, cross.psnr, cross.max_delta
        );
        assert!(cross.mse > 0.0, "different scenes should not be identical");
        assert!(
            cross.psnr.is_finite(),
            "PSNR should be finite for differing images"
        );
        assert!(cross.max_delta > 0.0);

        // Write a diff image and confirm it exists.
        let diff_path = std::path::Path::new("target/test_real_scene_diff.png");
        write_diff_image(&sphere, &emissive, diff_path, 4.0)
            .expect("write_diff_image for real scene pair");
        assert!(diff_path.exists());
    }

    #[test]
    fn zero_size_image_gives_zero_mse() {
        // 0×0 images should not produce NaN — they are trivially identical.
        let buf: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::new(0, 0);
        let a = DynamicImage::ImageRgba8(buf.clone());
        let b = DynamicImage::ImageRgba8(buf);
        let res = compare_images(&a, &b).unwrap();
        assert_eq!(res.mse, 0.0, "zero-size MSE should be 0.0, not NaN");
        assert_eq!(res.psnr, f64::INFINITY);
        assert_eq!(res.max_delta, 0.0);
    }
}
