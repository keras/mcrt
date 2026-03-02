// display.wgsl — full-screen triangle + texture sample
//
// Vertex stage: generates a clip-space full-screen triangle from the vertex
// index alone (no vertex buffer required). The three vertices cover all
// fragments of the render target when drawn with Draw(3, 1, 0, 0).
//
// Fragment stage: samples the display texture and returns its colour directly.

// ---- bindings ---------------------------------------------------------------

@group(0) @binding(0) var display_texture: texture_2d<f32>;
@group(0) @binding(1) var display_sampler: sampler;

// ---- vertex stage -----------------------------------------------------------

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    // One large triangle that covers the whole clip space:
    //   vertex 0: (-1, -1)  bottom-left
    //   vertex 1: ( 3, -1)  far right (off-screen)
    //   vertex 2: (-1,  3)  far top   (off-screen)
    // Any fragment inside the [-1, 1]^2 clip square is covered.
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );

    let pos = positions[vi];

    // Derive UV from clip position:
    //   clip x in [-1, 1]  →  u in [0, 1]
    //   clip y in [-1, 1]  →  v in [1, 0]  (flip Y: clip +y = top, texture +v = bottom)
    let uv = vec2<f32>(
        (pos.x + 1.0) * 0.5,
        (1.0 - pos.y) * 0.5,
    );

    var out: VertexOutput;
    out.clip_position = vec4<f32>(pos, 0.0, 1.0);
    out.uv = uv;
    return out;
}

// ---- fragment stage ---------------------------------------------------------

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(display_texture, display_sampler, in.uv);
}
