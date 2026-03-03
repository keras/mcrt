// path_trace.wgsl — Phase 6: progressive HDR accumulation
//
// Each invocation shoots one diffuse path per pixel and blends the new sample
// into a running per-pixel average stored in the Rgba32Float accumulation
// texture.  The display pass tone-maps the average each frame.

// ---- shared primitives ----------------------------------------------------

/// A ray in world space.  `dir` must be unit length.
struct Ray {
    origin: vec3<f32>,
    dir:    vec3<f32>,
}

/// Record of the closest intersection found so far.
/// `t < 0` is the "no hit" sentinel.
struct HitRecord {
    t:          f32,
    point:      vec3<f32>,
    normal:     vec3<f32>,  // always faces the incident ray
    front_face: bool,
    mat_index:  u32,        // sphere's material slot (used in Phase 7)
}

// ---- camera uniform -------------------------------------------------------
// All fields are vec4; unused .w components serve as free padding.
// frame_count is a dedicated u32 field to avoid f32 precision loss beyond 2²⁴.

struct Camera {
    origin:     vec4<f32>,  // .xyz = eye position; .w unused
    lower_left: vec4<f32>,  // .xyz = lower-left corner of the virtual screen
    horizontal: vec4<f32>,  // .xyz = full horizontal extent of the screen
    vertical:   vec4<f32>,  // .xyz = full vertical extent of the screen
    // Monotonically increasing frame index.  Stored as u32 (not f32) so it
    // remains exact beyond 2²⁴ (~77 h at 60 fps with a f32 representation).
    frame_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// ---- sphere scene buffer --------------------------------------------------
// Each sphere packs centre + radius into one vec4 to keep the Rust-side layout
// identical without any manual padding.

struct Sphere {
    center_r:    vec4<f32>,  // .xyz = centre, .w = radius
    mat_and_pad: vec4<u32>,  // .x = material index, .yzw = unused
}

// Fixed-capacity scene; must match `MAX_SPHERES` in main.rs.
struct SceneData {
    sphere_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    spheres: array<Sphere, 8>,
}

// ---- bindings -------------------------------------------------------------

/// Accumulation write target: the compute shader writes the blended average here.
/// Ping-pongs with accum_read every frame.
@group(0) @binding(0) var accum_write: texture_storage_2d<rgba32float, write>;

/// Camera uniform; updated every frame with the current frame index for PRNG.
@group(0) @binding(1) var<uniform> camera: Camera;

/// Scene: static sphere list.
@group(0) @binding(2) var<uniform> scene: SceneData;

/// Accumulation read source: the previous frame's running average.
/// Ping-pongs with accum_write every frame.
@group(0) @binding(3) var accum_read: texture_2d<f32>;

// ---- PCG PRNG -------------------------------------------------------------
//
// PCG32/XSH-RR: one 32-bit state, 32-bit output.
// Reference: https://www.pcg-random.org/

/// Single PCG32 step; maps any seed to a well-distributed u32.
fn pcg_hash(v: u32) -> u32 {
    var s = v * 747796405u + 2891336453u;
    // XSH-RR output mixing.
    let rot = s >> 28u;
    s = ((s >> (rot + 4u)) ^ s) * 277803737u;
    return (s >> 22u) ^ s;
}

/// Advance `rng` by one PCG step and return a uniform f32 in [0, 1).
/// Uses the bit-cast trick: sets the f32 exponent to 127 (= 1.0) and fills
/// the 23-bit mantissa from the PRNG output → value ∈ [1, 2), subtract 1.
fn rand_f32(rng: ptr<function, u32>) -> f32 {
    *rng = pcg_hash(*rng);
    return bitcast<f32>((*rng >> 9u) | 0x3f800000u) - 1.0;
}

// ---- sky gradient ---------------------------------------------------------

/// Classic RTIOW sky: white at the horizon, sky-blue at the zenith.
fn sky_color(unit_dir: vec3<f32>) -> vec3<f32> {
    let t = 0.5 * (unit_dir.y + 1.0);
    return mix(vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(0.5, 0.7, 1.0), t);
}

// ---- ray-sphere intersection ----------------------------------------------

/// Tests ray `r` against `sphere` in the interval (t_min, t_max).
/// Returns a HitRecord with `t > t_min` on hit, `t = -1` on miss.
fn ray_sphere_hit(r: Ray, sphere: Sphere, t_min: f32, t_max: f32) -> HitRecord {
    let center = sphere.center_r.xyz;
    let radius = sphere.center_r.w;

    let oc    = r.origin - center;
    // a = dot(dir, dir).  For unit-length rays this is always 1.0, but we keep
    // the variable so the function remains correct if called with non-unit dirs.
    let a     = dot(r.dir, r.dir);
    let h     = dot(oc, r.dir);      // b/2 shortcut
    let c     = dot(oc, oc) - radius * radius;
    let disc  = h * h - a * c;

    var hit: HitRecord;
    hit.t = -1.0;
    if disc < 0.0 { return hit; }

    let sqrtd = sqrt(disc);

    // Take the nearest root in (t_min, t_max).
    var t = (-h - sqrtd) / a;
    if t <= t_min || t >= t_max {
        t = (-h + sqrtd) / a;
        if t <= t_min || t >= t_max { return hit; }
    }

    hit.t         = t;
    hit.point     = r.origin + t * r.dir;
    let outward_n = (hit.point - center) / radius;
    hit.front_face = dot(r.dir, outward_n) < 0.0;
    // Normal always faces the incident ray (required for refraction in Phase 7).
    hit.normal    = select(-outward_n, outward_n, hit.front_face);
    hit.mat_index = sphere.mat_and_pad.x;
    return hit;
}

// ---- scene traversal ------------------------------------------------------

/// Finds the closest sphere hit along ray `r` in (t_min, t_max).
fn scene_hit(r: Ray, t_min: f32, t_max: f32) -> HitRecord {
    var best: HitRecord;
    best.t = -1.0;
    var closest = t_max;

    // Cap to array size to guard against a corrupt sphere_count from the CPU.
    let count = min(scene.sphere_count, 8u);
    for (var i = 0u; i < count; i++) {
        let hit = ray_sphere_hit(r, scene.spheres[i], t_min, closest);
        // ray_sphere_hit returns t > t_min (1e-4) on a hit; the `> 0.0` check
        // is equivalent here because t_min > 0.  (If t_min were ever negative,
        // the check would need to be > t_min instead.)
        if hit.t > 0.0 {
            best    = hit;
            closest = hit.t;
        }
    }
    return best;
}

// ---- hemisphere sampling --------------------------------------------------

/// Uniform point on the unit sphere via spherical coordinates.
/// Uses two uniform samples — no rejection loop, no thread divergence.
///
///   cos_phi ∈ [−1, 1] gives uniform solid-angle coverage.
///   theta   ∈ [0, 2π) spans the azimuth.
fn random_unit_sphere(rng: ptr<function, u32>) -> vec3<f32> {
    let cos_phi = 1.0 - 2.0 * rand_f32(rng);             // uniform in [-1, 1]
    let sin_phi = sqrt(max(0.0, 1.0 - cos_phi * cos_phi));
    let theta   = 6.283185307179586 * rand_f32(rng);      // 2π
    return vec3<f32>(sin_phi * cos(theta), sin_phi * sin(theta), cos_phi);
}

/// Cosine-weighted Lambertian scatter direction.
/// Offsets the surface normal by a random unit-sphere vector, then normalises.
/// This produces a cosine distribution (pdf ∝ cos θ) without trig-heavy
/// explicit hemisphere parameterisation.
/// Guards against the degenerate case where the random vector nearly cancels
/// the normal, which would produce a near-zero direction.
fn cosine_scatter(rng: ptr<function, u32>, normal: vec3<f32>) -> vec3<f32> {
    let scatter = normal + random_unit_sphere(rng);
    if dot(scatter, scatter) < 1e-10 { return normal; }
    return normalize(scatter);
}

// ---- material albedos (Phase 5: hardcoded; replaced by buffer in Phase 7) -

fn albedo_for_material(mat_index: u32) -> vec3<f32> {
    switch mat_index {
        case 0u: { return vec3<f32>(0.5, 0.5, 0.5); } // ground: grey
        case 1u: { return vec3<f32>(0.7, 0.3, 0.3); } // centre: warm red
        case 2u: { return vec3<f32>(0.3, 0.7, 0.3); } // left:   green
        case 3u: { return vec3<f32>(0.3, 0.3, 0.7); } // right:  blue
        default: { return vec3<f32>(1.0, 0.0, 1.0); } // error:  magenta
    }
}

// ---- path tracing ---------------------------------------------------------

/// Maximum number of bounces per path.  Rays that exceed this depth return
/// black (fully absorbed), introducing a small energy-loss bias.
const MAX_BOUNCES: u32 = 8u;

/// Trace `initial_ray` through the scene, scattering Lambertian at each hit.
/// Returns the estimated radiance for this single-sample path.
///
/// `t_min = 1e-4` assumes world-scale geometry (sphere radii ~ 0.5–100 units).
/// For very small or large  scenes this value should be adjusted accordingly.
fn path_color(initial_ray: Ray, rng: ptr<function, u32>) -> vec3<f32> {
    var ray        = initial_ray;
    var throughput = vec3<f32>(1.0, 1.0, 1.0);

    for (var bounce = 0u; bounce < MAX_BOUNCES; bounce++) {
        let hit = scene_hit(ray, 1e-4, 1e9);

        if hit.t < 0.0 {
            // Ray escaped — multiply throughput by sky radiance and return.
            return throughput * sky_color(ray.dir);
        }

        // Lambertian BRDF: attenuate by albedo, scatter diffusely.
        throughput *= albedo_for_material(hit.mat_index);
        let scatter_dir = cosine_scatter(rng, hit.normal);
        ray = Ray(hit.point, scatter_dir);
    }

    // Path absorbed after MAX_BOUNCES with no sky contribution.
    return vec3<f32>(0.0, 0.0, 0.0);
}

// ---- compute entry point --------------------------------------------------

@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims  = textureDimensions(accum_write);
    let coord = gid.xy;

    // Discard out-of-bounds threads (non-multiple-of-16 resolutions).
    if coord.x >= dims.x || coord.y >= dims.y { return; }

    let frame_count = camera.frame_count;

    // Seed the PRNG uniquely per pixel per frame.
    // Use `coord.x + coord.y * dims.x` for the spatial component — this is a
    // bijection over [0, width*height) that avoids the coord.y << 16 trick
    // silently collapsing rows 0 and 65536 on 8K+ monitors.
    var rng = pcg_hash(pcg_hash(coord.x + coord.y * dims.x) ^ (frame_count * 2654435761u));
    // (2654435761 is the Knuth multiplicative hash constant, chosen so nearby
    // frames produce maximally different seeds.)

    // Sub-pixel jitter for AA and better convergence of the running average.
    let u = (f32(coord.x) + rand_f32(&rng)) / f32(dims.x);
    let v = 1.0 - (f32(coord.y) + rand_f32(&rng)) / f32(dims.y);

    let raw_dir =
          camera.lower_left.xyz
        + u * camera.horizontal.xyz
        + v * camera.vertical.xyz
        - camera.origin.xyz;
    let ray = Ray(camera.origin.xyz, normalize(raw_dir));

    let sample = path_color(ray, &rng);

    // Progressive accumulation — running average.
    //   weight  = 1 / (frame_count + 1)
    //   new_avg = mix(prev_avg, sample, weight)
    // When frame_count = 0 (first frame or after a resize reset) weight = 1.0,
    // so the previous texture value is discarded regardless of its contents.
    //
    // Cap at 65535 frames: prevents the denominator from wrapping to 0 at
    // u32::MAX which would produce +Inf weight and corrupt all pixels with NaN.
    let weight = 1.0 / f32(min(frame_count, 65535u) + 1u);
    let prev   = textureLoad(accum_read, coord, 0).xyz;
    let accum  = mix(prev, sample, weight);

    // Store raw HDR radiance — tone-mapping is applied in the display pass.
    textureStore(accum_write, coord, vec4<f32>(accum, 1.0));
}
