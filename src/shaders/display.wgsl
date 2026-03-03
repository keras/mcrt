// display.wgsl — full-screen triangle + HDR accumulation tone-map
//
// Phase 14 update:
//   - ACES filmic tone-mapping (Narkowicz 2016) replaces per-channel Reinhard.
//     ACES preserves colour saturation in bright regions and produces a
//     characteristic filmic shoulder that Reinhard lacks.
//   - Proper piecewise IEC 61966-2-1 sRGB transfer function replaces the
//     approximate pow(x, 1/2.2), eliminating mild shadow banding.
//
// Vertex stage: clip-space full-screen triangle (no vertex buffer).
// Fragment stage: textureLoad → exposure → ACES → sRGB → swapchain.

// ---- constants -------------------------------------------------------------

/// Linear exposure multiplier applied before the tone-mapping curve.
///
/// The Narkowicz ACES formula is calibrated for scene-referred linear values
/// roughly in [0, 1].  A bright accumulation (many emissive bounces) can
/// exceed 10×, saturating the curve and washing out colour.  Reduce EXPOSURE
/// (e.g. 0.5) to preserve highlights; increase it to brighten a dark scene.
const EXPOSURE: f32 = 1.0;

// ---- tone-mapping helpers --------------------------------------------------

/// ACES filmic tone-mapping curve (Narkowicz 2016).
///
/// Maps linear HDR radiance ∈ [0, ∞) to display-ready LDR ∈ [0, 1] with a
/// filmic toe and shoulder.  The curve is applied per-channel.
///
/// Note: per-channel application can shift hue on saturated highlights
/// (dominant channel saturates before weaker ones).  This is a known
/// limitation of the simplified curve vs the full ACES RRT+ODT.
///
/// Reference: https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
fn aces_filmic(x: vec3<f32>) -> vec3<f32> {
    let a: f32 = 2.51;
    let b: f32 = 0.03;
    let c: f32 = 2.43;
    let d: f32 = 0.59;
    let e: f32 = 0.14;
    // Denominator discriminant is negative → always positive; no divide-by-zero.
    return clamp(
        (x * (a * x + b)) / (x * (c * x + d) + e),
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
}

/// Piecewise IEC 61966-2-1 sRGB transfer function (linear light → encoded).
///
/// Below the linear threshold 0.0031308 the function is linear (×12.92);
/// above it a power law with exponent 1/2.4 is used.  Matches the standard
/// precisely and avoids the mild banding of the pow(x, 1/2.2) approximation.
fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let lo = 12.92 * c;
    let hi = 1.055 * pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
    // select(false_val, true_val, cond): returns lo (linear branch) where
    // c ≤ 0.0031308 and hi (power branch) elsewhere.
    return select(hi, lo, c <= vec3<f32>(0.0031308));
}

// ---- bindings ---------------------------------------------------------------

/// Rgba32Float accumulation (or denoised) texture written by the compute pass.
/// Read by pixel coordinate — no filtering sampler needed.
@group(0) @binding(0) var accum_tex: texture_2d<f32>;

// ---- vertex stage -----------------------------------------------------------

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    // One large triangle covering the whole clip space:
    //   vertex 0: (-1, -1)  bottom-left
    //   vertex 1: ( 3, -1)  far right (off-screen)
    //   vertex 2: (-1,  3)  far top   (off-screen)
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    return vec4<f32>(positions[vi], 0.0, 1.0);
}

// ---- fragment stage ---------------------------------------------------------

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let coord = vec2<i32>(i32(frag_coord.x), i32(frag_coord.y));
    let raw   = textureLoad(accum_tex, coord, 0).xyz;

    // 1. Exposure: scale radiance before tone-mapping so that very bright
    //    accumulations don't saturate the ACES shoulder into uniform white.
    // 2. ACES filmic tone-map: HDR linear radiance → [0, 1] with filmic look.
    let mapped = aces_filmic(raw * EXPOSURE);

    // 3. Piecewise sRGB transfer: linear light → display-encoded sRGB.
    let srgb = linear_to_srgb(mapped);

    return vec4<f32>(srgb, 1.0);
}
