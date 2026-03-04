// denoise.wgsl — Phase 14: edge-aware joint-bilateral spatial denoiser
//
// Reads the HDR accumulation texture and the G-buffer (first-hit normal xyz +
// linear depth w) written by the path-trace pass, then applies a joint
// bilateral filter to reduce Monte-Carlo noise while preserving geometric
// edges (silhouettes, sharp normal discontinuities, depth boundaries).
//
// Weight for neighbour q relative to centre pixel p:
//   w = exp(−‖p−q‖² / (2·σ_s²))        [spatial Gaussian]
//     × max(0, dot(n_p, n_q))^normal_pow [normal similarity]
//     × exp(−(d_p−d_q)² / (2·σ_d²))     [depth proximity]
//
// Sky pixels (depth == −1) are blended only with other sky pixels, which
// prevents sky colour from bleeding across foreground silhouettes.
//
// Dispatch: same 16×16 workgroup grid as the path tracer.

// ---- runtime-tunable parameters (TD-5: uniform buffer, binding 3) ---------

/// Denoiser knobs uploaded from the CPU each frame.
/// Mirrors `DenoiseParams` in gpu.rs (repr(C), std140, 16 bytes).
struct DenoiseParams {
    /// Half-kernel radius in pixels.  Total tap count = (2R+1)².
    radius:     i32,
    /// Spatial Gaussian sigma (pixels).
    sigma_s:    f32,
    /// Depth similarity sigma (world units).
    sigma_d:    f32,
    /// Exponent for the normal dot-product weight factor.
    normal_pow: f32,
}

// ---- bindings -------------------------------------------------------------

/// HDR accumulation texture (path-traced radiance, linear light, Rgba32Float).
/// Read via textureLoad (non-filterable).
@group(0) @binding(0) var accum_tex : texture_2d<f32>;

/// G-buffer written by the path-trace kernel.
///   .xyz = world-space surface normal at the first hit (unit length).
///   .w   = linear depth (> 0 on surface hit; −1 for sky / miss rays).
@group(0) @binding(1) var gbuf_tex  : texture_2d<f32>;

/// Denoised output (write-only; read by the display pass when denoising is on).
@group(0) @binding(2) var out_tex   : texture_storage_2d<rgba32float, write>;

/// Runtime-tunable denoiser parameters.
@group(0) @binding(3) var<uniform> dp: DenoiseParams;

// ---- denoiser kernel ------------------------------------------------------

@compute @workgroup_size(16, 16)
fn cs_denoise(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = vec2<i32>(textureDimensions(out_tex));
    let p    = vec2<i32>(gid.xy);
    if p.x >= dims.x || p.y >= dims.y { return; }

    // Centre pixel G-buffer data.
    let g_p    = textureLoad(gbuf_tex, p, 0);
    let n_p    = g_p.xyz;
    let d_p    = g_p.w;
    let on_sky = d_p < 0.0;

    // Pre-compute constant denominators for the Gaussian exponents.
    let inv2ss = 1.0 / (2.0 * dp.sigma_s * dp.sigma_s);
    let inv2ds = 1.0 / (2.0 * dp.sigma_d * dp.sigma_d);

    var color_acc:  vec3<f32> = vec3<f32>(0.0);
    var weight_acc: f32       = 0.0;

    for (var dy = -dp.radius; dy <= dp.radius; dy++) {
        for (var dx = -dp.radius; dx <= dp.radius; dx++) {
            // Clamp neighbour address to valid pixel range (border duplication).
            let q = clamp(p + vec2<i32>(dx, dy),
                          vec2<i32>(0),
                          dims - vec2<i32>(1));

            let g_q   = textureLoad(gbuf_tex, q, 0);
            let n_q   = g_q.xyz;
            let d_q   = g_q.w;
            let q_sky = d_q < 0.0;

            // Reject cross-category blends: sky ⟷ surface boundaries would
            // cause sky colour to leak across foreground edges.
            if on_sky != q_sky { continue; }

            // Spatial Gaussian weight (same for sky and surface pixels).
            let dist_sq = f32(dx * dx + dy * dy);
            let w_s     = exp(-dist_sq * inv2ss);

            // Normal similarity: cos(θ)^normal_pow — falls to ~0 for θ > 45°.
            // Only applied to surface pixels (sky has no meaningful normal).
            let w_n = select(1.0,
                             pow(max(0.0, dot(n_p, n_q)), dp.normal_pow),
                             !on_sky);

            // Depth proximity (relative formulation, surface pixels only).
            // Normalising by d_p makes the filter scale-invariant: a 10 % depth
            // difference is penalised equally whether objects are at d=1 or d=100.
            let d_diff_rel = (d_p - d_q) / max(d_p, 1e-4);
            let w_d    = select(1.0,
                                exp(-(d_diff_rel * d_diff_rel) * inv2ds),
                                !on_sky);

            let w = w_s * w_n * w_d;
            color_acc  += w * textureLoad(accum_tex, q, 0).xyz;
            weight_acc += w;
        }
    }

    // Normalise; fall back to the raw pixel value if all weights collapsed.
    // Invariant: weight_acc >= 1.0 because the centre tap (dx=dy=0) always
    // passes the sky-category check and contributes w_s = exp(0) = 1.0.
    // The select guard is retained for defensive correctness only.
    let result = select(textureLoad(accum_tex, p, 0).xyz,
                        color_acc / weight_acc,
                        weight_acc > 0.0);

    textureStore(out_tex, p, vec4<f32>(result, 1.0));
}
