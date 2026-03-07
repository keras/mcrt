// texture.rs — Phase 11: CPU-side texture loading and environment map generation
//
// Provides pure-Rust helpers (no wgpu) for:
//   - RGBA8-Unorm albedo texture layers (loaded from PNG/JPG or procedural)
//   - RGBA32-Float environment map data (loaded from Radiance HDR or procedural gradient)
//
// All output is raw byte/f32 slices ready for gpu.rs to upload via
// queue.write_texture.  This module never imports wgpu so it remains
// independently unit-testable.


// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Side length (pixels) shared by all albedo texture array layers.
/// Must be a power of two; all loaded images are resampled to this size.
pub const TEXTURE_SIZE: u32 = 512;

/// Number of layers in the albedo texture array (GPU binding 10).
///
/// * Layer **0** is always a solid-white fill.  Materials that set
///   `tex_idx = 0` in `type_pad[1]` will therefore see pure white, so
///   multiplying by `albedo_fuzz.xyz` leaves the colour unchanged — the
///   equivalent of "no texture".
/// * Layers **1 .. MAX_TEXTURES − 1** are loaded from files or generated
///   procedurally as fallbacks.
///
/// **Must match** `ALBEDO_ARRAY_LAYERS` in `path_trace.wgsl`.
pub const MAX_TEXTURES: u32 = 4;

/// Width (pixels) of the equirectangular environment map.
pub const ENV_MAP_WIDTH: u32 = 1024;

/// Height (pixels) of the equirectangular environment map.
pub const ENV_MAP_HEIGHT: u32 = 512;

// ---------------------------------------------------------------------------
// Albedo texture helpers (RGBA8 Unorm)
// ---------------------------------------------------------------------------

/// Build an RGBA8 `size × size` checkerboard texture.
///
/// `color_a` / `color_b` are `[R, G, B, A]` byte values in Unorm [0, 255].
/// Tiles are `size / 8` pixels wide; if `size < 8`, each tile is 1 pixel.
pub fn build_checker_texture(size: u32, color_a: [u8; 4], color_b: [u8; 4]) -> Vec<u8> {
    let tile = (size / 8).max(1);
    let n = (size * size * 4) as usize;
    let mut data = Vec::with_capacity(n);
    for y in 0..size {
        for x in 0..size {
            let c = if (x / tile + y / tile).is_multiple_of(2) {
                color_a
            } else {
                color_b
            };
            data.extend_from_slice(&c);
        }
    }
    data
}

/// Try to load a PNG or JPEG file and return an RGBA8 buffer of exactly
/// `TEXTURE_SIZE × TEXTURE_SIZE` pixels (resampled with Lanczos3).
///
/// Returns `None` if the path does not exist, the format is unsupported,
/// or any other I/O error occurs.
pub fn try_load_rgba8(path: &str) -> Option<Vec<u8>> {
    let bytes = crate::platform::load_bytes(path).ok()?;
    let img = image::load_from_memory(&bytes).ok()?;
    let resized = img.resize_exact(
        TEXTURE_SIZE,
        TEXTURE_SIZE,
        image::imageops::FilterType::Lanczos3,
    );
    Some(resized.to_rgba8().into_raw())
}

/// Build the full `MAX_TEXTURES`-layer albedo array.
///
/// `paths` maps each layer index (1-based) to an optional PNG/JPEG file path:
/// * `paths[0]` is **ignored** — layer 0 is always pure white.
/// * `paths[1..MAX_TEXTURES]` are attempted in order; a `None` entry or a
///   failed load falls back to the magenta-black checkerboard pattern (the
///   classic "missing texture" indicator).
///
/// If `paths.len() < MAX_TEXTURES`, the remaining layers also use the
/// checker fallback.
pub fn build_albedo_layers(paths: &[Option<&str>]) -> Vec<Vec<u8>> {
    let pixels_per_layer = (TEXTURE_SIZE * TEXTURE_SIZE * 4) as usize;
    let mut layers: Vec<Vec<u8>> = Vec::with_capacity(MAX_TEXTURES as usize);

    // Layer 0: solid white — tex_idx=0 means "no texture / use solid colour".
    layers.push(vec![255u8; pixels_per_layer]);

    // Layers 1..MAX_TEXTURES-1: file load or checker fallback.
    for i in 1..MAX_TEXTURES as usize {
        let layer = paths
            .get(i)
            .copied()
            .flatten()
            .and_then(try_load_rgba8)
            .unwrap_or_else(|| {
                build_checker_texture(TEXTURE_SIZE, [200, 0, 200, 255], [30, 30, 30, 255])
            });
        layers.push(layer);
    }

    layers
}

// ---------------------------------------------------------------------------
// HDR environment map helpers (RGBA32 Float, interleaved f32)
// ---------------------------------------------------------------------------

/// Try to load a Radiance HDR (`.hdr`) image at `path`.
///
/// Returns the pixel data as a packed `Vec<f32>` in R, G, B, 1.0 layout
/// for `ENV_MAP_WIDTH × ENV_MAP_HEIGHT` pixels, or `None` on any error.
///
/// The image is rescaled to `ENV_MAP_WIDTH × ENV_MAP_HEIGHT` if needed.
///
/// **Requires** the `hdr` feature of the `image` crate (enabled in Cargo.toml).
pub fn try_load_hdr(path: &str) -> Option<Vec<f32>> {
    let bytes = crate::platform::load_bytes(path).ok()?;
    let img = image::load_from_memory(&bytes).ok()?;
    let resized = img.resize_exact(
        ENV_MAP_WIDTH,
        ENV_MAP_HEIGHT,
        // Nearest-neighbour avoids Lanczos negative lobes around extreme HDR
        // highlights (sun disk), which would otherwise be clamped to zero and
        // appear as a black halo around the sun.
        image::imageops::FilterType::Nearest,
    );
    // to_rgb32f() converts to a linear-float ImageBuffer<Rgb<f32>, Vec<f32>>.
    // into_raw() returns the flat [R, G, B, R, G, B, ...] pixel data.
    let raw = resized.to_rgb32f().into_raw();
    let mut out = Vec::with_capacity((ENV_MAP_WIDTH * ENV_MAP_HEIGHT * 4) as usize);
    for chunk in raw.chunks(3) {
        // Replace NaN / Inf (can arise from Lanczos resampling of extreme HDR
        // sun-disk pixels) with 0 so they don't propagate as black fireflies.
        out.push(chunk[0].clamp(0.0, 1e10));
        out.push(chunk[1].clamp(0.0, 1e10));
        out.push(chunk[2].clamp(0.0, 1e10));
        out.push(1.0_f32); // alpha pad — Rgba32Float needs 4 channels
    }
    Some(out)
}

/// Build a procedural RGBA32-Float equirectangular environment map.
///
/// The sky hemisphere blends from warm white at the horizon to sky-blue at
/// the zenith.  The ground hemisphere fades through warm amber to black.
///
/// **Coordinate convention**: row `y = 0` of the resulting image corresponds
/// to the zenith (top of the sphere), matching the `env_color()` function in
/// `path_trace.wgsl` which maps `v = high → iy = low`.
pub fn build_gradient_env_map() -> Vec<f32> {
    let n = (ENV_MAP_WIDTH * ENV_MAP_HEIGHT * 4) as usize;
    let mut data = Vec::with_capacity(n);

    for y in 0..ENV_MAP_HEIGHT {
        // v ∈ [0, 1]: 0 at the top of the image, 1 at the bottom.
        // elevation ∈ [+1, -1]: +1 = zenith, -1 = nadir.
        let v = (y as f32 + 0.5) / ENV_MAP_HEIGHT as f32;
        let elevation = 1.0 - 2.0 * v;

        for _x in 0..ENV_MAP_WIDTH {
            let (r, g, b) = gradient_sky_rgb(elevation);
            data.push(r);
            data.push(g);
            data.push(b);
            data.push(1.0_f32);
        }
    }

    data
}

/// Returns linear-light RGB sky colour for `elevation ∈ [-1, 1]`.
///
/// Positive = above horizon (sky), negative = below (ground).
fn gradient_sky_rgb(elevation: f32) -> (f32, f32, f32) {
    if elevation >= 0.0 {
        // Sky: warm white at the horizon → sky blue at the zenith.
        let t = elevation; // 0 at horizon, 1 at zenith
        let r = 1.0 - 0.5 * t;
        let g = 1.0 - 0.3 * t;
        let b = 1.0_f32;
        (r, g, b)
    } else {
        // Ground: warm amber near the horizon, black at the nadir.
        let t = (-elevation).min(1.0); // 0 at horizon, 1 at nadir
        let r = (1.0 - t) * 0.8;
        let g = (1.0 - t) * 0.4;
        let b = (1.0 - t) * 0.1;
        (r, g, b)
    }
}

/// Load or generate equirectangular environment-map data (RGBA32 Float).
///
/// Priority:
/// 1. `scene_path` — explicit HDR path from the scene YAML
/// 2. `textures/env.hdr` — well-known default location
/// 3. Procedural gradient sky
///
/// Returns `ENV_MAP_WIDTH × ENV_MAP_HEIGHT × 4` f32 values.
pub fn load_env_map_data(scene_path: Option<&str>) -> Vec<f32> {
    if let Some(p) = scene_path {
        if let Some(data) = try_load_hdr(p) {
            return data;
        }
        log::warn!(
            "env_map '{}' could not be loaded; falling back to default",
            p
        );
    }
    try_load_hdr("textures/env.hdr").unwrap_or_else(build_gradient_env_map)
}

// ---------------------------------------------------------------------------
// Combined entry point
// ---------------------------------------------------------------------------

/// Load or generate all Phase 11 textures.
///
/// Returns `(albedo_layers, env_map_rgba32f)`:
/// * `albedo_layers` — `MAX_TEXTURES` RGBA8 slices, each `TEXTURE_SIZE²×4` bytes.
/// * `env_map_rgba32f` — `ENV_MAP_WIDTH × ENV_MAP_HEIGHT × 4` f32 values (RGBA).
///
/// File paths currently attempted (relative to the working directory):
/// * Layer 2: `textures/earth.png` — falls back to checker if absent.
/// * Env map: `textures/env.hdr`   — falls back to procedural gradient if absent.
pub fn load_all_textures() -> (Vec<Vec<u8>>, Vec<f32>) {
    let albedo_paths: &[Option<&str>] = &[
        None,                       // layer 0: always white (managed internally)
        None,                       // layer 1: built explicitly below
        Some("textures/earth.png"), // layer 2: optional earth texture
        None,                       // layer 3: checker fallback
    ];

    let mut albedo_layers = build_albedo_layers(albedo_paths);

    // Layer 1: white/gray grid checker used by the ground sphere in scene.yaml.
    // Overwrite the magenta-black "missing texture" fallback with a clean pattern.
    albedo_layers[1] = build_checker_texture(
        TEXTURE_SIZE,
        [220, 220, 220, 255], // light gray
        [50, 50, 50, 255],    // dark gray
    );

    let env_map = try_load_hdr("textures/env.hdr").unwrap_or_else(build_gradient_env_map);

    (albedo_layers, env_map)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checker_texture_correct_size() {
        let data = build_checker_texture(TEXTURE_SIZE, [255, 0, 0, 255], [0, 255, 0, 255]);
        assert_eq!(data.len(), (TEXTURE_SIZE * TEXTURE_SIZE * 4) as usize);
    }

    #[test]
    fn checker_texture_small_alternates_colors() {
        // size=8 → tile=1: every pixel alternates.
        let a = [255u8, 0, 0, 255];
        let b = [0u8, 255, 0, 255];
        let data = build_checker_texture(8, a, b);
        // Pixel (0,0): (0/1 + 0/1) % 2 = 0 → color_a.
        assert_eq!(&data[0..4], &a);
        // Pixel (1,0): (1/1 + 0/1) % 2 = 1 → color_b.
        assert_eq!(&data[4..8], &b);
    }

    #[test]
    fn albedo_layer_count_matches_constant() {
        let layers = build_albedo_layers(&[None, None, None, None]);
        assert_eq!(layers.len(), MAX_TEXTURES as usize);
    }

    #[test]
    fn albedo_layer_0_is_white() {
        let layers = build_albedo_layers(&[None, None, None, None]);
        assert!(
            layers[0].iter().all(|&b| b == 255),
            "layer 0 must be solid white so tex_idx=0 acts as no-texture"
        );
    }

    #[test]
    fn albedo_layer_sizes_are_uniform() {
        let layers = build_albedo_layers(&[None, None, None, None]);
        let expected = (TEXTURE_SIZE * TEXTURE_SIZE * 4) as usize;
        for (i, layer) in layers.iter().enumerate() {
            assert_eq!(
                layer.len(),
                expected,
                "layer {i} has len {} expected {expected}",
                layer.len()
            );
        }
    }

    #[test]
    fn env_map_correct_size() {
        let data = build_gradient_env_map();
        assert_eq!(data.len(), (ENV_MAP_WIDTH * ENV_MAP_HEIGHT * 4) as usize);
    }

    #[test]
    fn env_map_values_finite_and_non_negative() {
        let data = build_gradient_env_map();
        for &v in &data {
            assert!(
                v.is_finite() && v >= 0.0,
                "env map contains invalid f32: {v}"
            );
        }
    }

    #[test]
    fn env_map_sky_brighter_than_ground() {
        // Row y=0 (zenith) should be brighter than row y=H-1 (nadir).
        let data = build_gradient_env_map();

        let lum = |base: usize| -> f32 {
            0.2126 * data[base] + 0.7152 * data[base + 1] + 0.0722 * data[base + 2]
        };

        let zenith_lum = lum(0);
        let nadir_base = (ENV_MAP_WIDTH * (ENV_MAP_HEIGHT - 1) * 4) as usize;
        let nadir_lum = lum(nadir_base);

        assert!(
            zenith_lum > nadir_lum,
            "zenith lum {zenith_lum:.3} should exceed nadir lum {nadir_lum:.3}"
        );
    }

    // M4: additional edge-case tests suggested by code review ----------------

    #[test]
    fn albedo_layer_0_is_white_even_with_path_given() {
        // paths[0] = Some(...) must be ignored; layer 0 is always white.
        let layers = build_albedo_layers(&[Some("this_path_does_not_exist.png"), None, None, None]);
        assert!(
            layers[0].iter().all(|&b| b == 255),
            "layer 0 must be solid white regardless of the path provided for index 0"
        );
    }

    #[test]
    fn albedo_layers_with_short_paths_fills_remaining_as_checker() {
        // When paths.len() < MAX_TEXTURES, remaining layers use the checker fallback.
        let layers = build_albedo_layers(&[None]); // only 1 entry, rest implicit
        assert_eq!(layers.len(), MAX_TEXTURES as usize);
    }

    #[test]
    fn checker_texture_size_less_than_8_uses_single_pixel_tiles() {
        // size=4 → tile = max(4/8, 1) = 1: every pixel alternates.
        let a = [255u8, 0, 0, 255];
        let b = [0u8, 0, 255, 255];
        let data = build_checker_texture(4, a, b);
        assert_eq!(data.len(), (4 * 4 * 4) as usize);
        // Pixel (0,0): (0/1 + 0/1) % 2 = 0 → color_a
        assert_eq!(&data[0..4], &a);
        // Pixel (1,0): (1 + 0) % 2 = 1 → color_b
        assert_eq!(&data[4..8], &b);
    }
}
