// path_trace.wgsl — Phase 3: GPU ray generation & sky gradient
//
// Each invocation handles one pixel.  The shader derives a pinhole camera
// ray, evaluates a sky-gradient background based on the ray's Y component,
// and stores the result into the output storage texture.  In later phases
// this function will be extended with sphere intersection, materials, and
// recursive path tracing.

// ---- camera uniform -------------------------------------------------------
// All vec4 fields; the .w component is unused padding to satisfy alignment.

struct Camera {
    origin:     vec4<f32>,  // .xyz = eye position in world space
    lower_left: vec4<f32>,  // .xyz = lower-left corner of the virtual screen
    horizontal: vec4<f32>,  // .xyz = full horizontal extent of the screen
    vertical:   vec4<f32>,  // .xyz = full vertical extent of the screen
}

// ---- bindings -------------------------------------------------------------

/// Output image written by this compute shader; sampled by the display pass.
@group(0) @binding(0) var output: texture_storage_2d<rgba8unorm, write>;

/// Camera parameters uploaded from the CPU every time the viewport changes.
@group(0) @binding(1) var<uniform> camera: Camera;

// ---- sky gradient helper --------------------------------------------------

/// Classic "Ray Tracing in One Weekend" sky: blends white (horizon) →
/// sky-blue (zenith) based on the normalised ray direction's Y component.
fn sky_color(ray_dir: vec3<f32>) -> vec3<f32> {
    let unit_dir = normalize(ray_dir);
    // Remap [-1, 1] → [0, 1] for the blend factor.
    let t = 0.5 * (unit_dir.y + 1.0);
    return mix(vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(0.5, 0.7, 1.0), t);
}

// ---- compute entry point --------------------------------------------------

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims  = textureDimensions(output);
    let coord = gid.xy;

    // Guard against threads that fall outside the texture for resolutions
    // that are not exact multiples of the workgroup tile size (8×8).
    if coord.x >= dims.x || coord.y >= dims.y {
        return;
    }

    // Normalised UV coordinates with a half-pixel centre offset.
    //   u ∈ [0, 1] : 0 = left edge,   1 = right edge
    //   v ∈ [0, 1] : 0 = bottom edge, 1 = top edge  (Y-up 3-D convention)
    let u = (f32(coord.x) + 0.5) / f32(dims.x);
    let v = 1.0 - (f32(coord.y) + 0.5) / f32(dims.y);

    // Pinhole camera ray direction (not normalised — sky_color normalises it).
    let ray_dir =
          camera.lower_left.xyz
        + u * camera.horizontal.xyz
        + v * camera.vertical.xyz
        - camera.origin.xyz;

    let color = sky_color(ray_dir);
    textureStore(output, coord, vec4<f32>(color, 1.0));
}
