// display.wgsl — full-screen triangle + HDR accumulation tone-map
//
// Vertex stage: generates a clip-space full-screen triangle from the vertex
// index alone (no vertex buffer required).
//
// Fragment stage: reads the Rgba32Float accumulation texture by integer pixel
// coordinate (no sampler needed), applies Reinhard tone-mapping and approximate
// sRGB gamma correction, then outputs the result to the swapchain.

// ---- bindings ---------------------------------------------------------------

/// Rgba32Float accumulation texture written by the compute shader each frame.
/// Sampled here by pixel coordinate — no filtering sampler is required.
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
    let raw = textureLoad(accum_tex, coord, 0).xyz;

    // Reinhard tone-mapping: maps [0, ∞) HDR radiance to [0, 1) LDR.
    // Applied per-channel so bright colours desaturate gracefully.
    let mapped = raw / (raw + vec3<f32>(1.0));

    // Approximate sRGB gamma correction (γ ≈ 2.2).
    // A proper piecewise sRGB transfer would be slightly more accurate but
    // the difference is imperceptible for path-traced output.
    let gamma = pow(max(mapped, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));

    return vec4<f32>(gamma, 1.0);
}
