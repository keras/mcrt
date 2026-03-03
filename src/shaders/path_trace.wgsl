// path_trace.wgsl — Phase 8: interactive orbit camera + depth of field
//
// Each invocation shoots one full path per pixel, dispatching to the correct
// scatter function (Lambertian / metal / dielectric) based on the material
// buffer, and accumulates the HDR result into the ping-pong texture pair.
// Depth-of-field is implemented via the thin-lens model: rays originate from
// a random point on the aperture disk and converge on the focus plane.

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
    vertical:   vec4<f32>,  // .xyz = full vertical extent (on the focus plane)
    /// Camera right-basis scaled by lens radius — used to offset the ray origin
    /// across the aperture disk (depth of field).  Zero when aperture = 0.
    defocus_u:  vec4<f32>,
    /// Camera up-basis scaled by lens radius — paired with defocus_u above.
    defocus_v:  vec4<f32>,
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

// ---- material buffer -----------------------------------------------------

/// Per-material descriptor; layout must match `GpuMaterial` in main.rs.
///
/// mat_type:  0 = Lambertian diffuse
///            1 = metal  (specular reflect + fuzz perturbation)
///            2 = dielectric (glass: Snell refraction + Schlick reflection)
struct Material {
    /// .x = mat_type (u32), .yzw = unused padding.
    type_pad:    vec4<u32>,
    /// .xyz = albedo (diffuse colour / metal tint), .w = fuzz ∈ [0, 1].
    albedo_fuzz: vec4<f32>,
    /// .x = index of refraction (dielectric), .yzw = unused.
    ior_pad:     vec4<f32>,
}

/// Fixed-capacity material table; must match `MAX_MATERIALS` in main.rs.
struct MaterialData {
    mat_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    mats: array<Material, 8>,
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

/// Material descriptors for all spheres in the scene.
@group(0) @binding(4) var<uniform> materials: MaterialData;

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
    if dot(scatter, scatter) < 1e-6 { return normal; }
    return normalize(scatter);
}

/// Uniform point inside the unit disk via inverse-CDF polar sampling.
/// O(1), no rejection loop, no thread divergence.
///   r     = sqrt(uniform[0,1])  — inverse CDF of f(r) = 2r gives uniform area density
///   theta = uniform[0, 2π)      — uniform azimuth
fn random_in_unit_disk(rng: ptr<function, u32>) -> vec2<f32> {
    let r     = sqrt(rand_f32(rng));
    let theta = 6.283185307179586 * rand_f32(rng);
    return r * vec2<f32>(cos(theta), sin(theta));
}

// ---- reflect / refract helpers -------------------------------------------

/// Mirror reflection: `v` reflected about unit normal `n`.
fn reflect_vec(v: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    return v - 2.0 * dot(v, n) * n;
}

/// Snell's law refraction.
///   v         : unit incident direction
///   n         : unit surface normal pointing toward the incident medium
///   eta_ratio : η_incident / η_transmitted
fn refract_vec(v: vec3<f32>, n: vec3<f32>, eta_ratio: f32) -> vec3<f32> {
    let cos_theta = min(dot(-v, n), 1.0);
    let r_perp    = eta_ratio * (v + cos_theta * n);
    let r_para    = -sqrt(max(0.0, 1.0 - dot(r_perp, r_perp))) * n;
    return r_perp + r_para;
}

/// Schlick's approximation for Fresnel reflectance at a dielectric interface.
///   cos_theta : cosine of the angle of incidence
///   eta_ratio : η_incident / η_transmitted (same ratio used in refract_vec)
fn schlick(cos_theta: f32, eta_ratio: f32) -> f32 {
    var r0 = (1.0 - eta_ratio) / (1.0 + eta_ratio);
    r0     = r0 * r0;
    return r0 + (1.0 - r0) * pow(1.0 - cos_theta, 5.0);
}

// ---- scatter --------------------------------------------------------------

/// Result of a material scatter event.
struct ScatterResult {
    /// New ray direction (unit length) after scattering.
    direction:   vec3<f32>,
    /// Throughput multiplier (albedo / attenuation).
    attenuation: vec3<f32>,
    /// True when the ray is absorbed and the path must terminate.
    absorbed:    bool,
}

/// Lambertian (diffuse) scatter: cosine-weighted random hemisphere direction.
fn scatter_lambertian(mat: Material, hit: HitRecord,
                      rng: ptr<function, u32>) -> ScatterResult {
    var r: ScatterResult;
    r.direction   = cosine_scatter(rng, hit.normal);
    r.attenuation = mat.albedo_fuzz.xyz;
    r.absorbed    = false;
    return r;
}

/// Metal scatter: perfect mirror reflect + fuzz perturbation.
/// Absorbed when fuzz pushes the scattered ray below the surface.
fn scatter_metal(mat: Material, ray_in: Ray, hit: HitRecord,
                 rng: ptr<function, u32>) -> ScatterResult {
    var r: ScatterResult;
    let fuzz      = mat.albedo_fuzz.w;
    // Normalise incoming direction defensively against floating-point drift.
    let reflected = reflect_vec(normalize(ray_in.dir), hit.normal);
    // Perturb by a random unit-sphere vector scaled by fuzz; re-normalise.
    // Guard: when fuzz≈1 and the random vector is exactly anti-parallel to
    // reflected, the sum approaches zero and normalize would yield NaN.
    let perturbed = reflected + fuzz * random_unit_sphere(rng);
    r.direction   = normalize(select(reflected, perturbed, dot(perturbed, perturbed) >= 1e-6));
    r.attenuation = mat.albedo_fuzz.xyz;
    r.absorbed    = dot(r.direction, hit.normal) <= 0.0;
    return r;
}

/// Dielectric scatter: Snell refraction with probabilistic Schlick reflection.
/// Handles total internal reflection and the air↔glass direction swap.
fn scatter_dielectric(mat: Material, ray_in: Ray, hit: HitRecord,
                      rng: ptr<function, u32>) -> ScatterResult {
    var r: ScatterResult;
    r.absorbed    = false;
    r.attenuation = vec3<f32>(1.0); // glass transmits all wavelengths equally

    let ior = mat.ior_pad.x;
    // Air-to-glass (front face): η_ratio = 1/ior.
    // Glass-to-air (back face) : η_ratio = ior.
    let eta_ratio = select(ior, 1.0 / ior, hit.front_face);

    let cos_theta = min(dot(-ray_in.dir, hit.normal), 1.0);
    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));

    // Total internal reflection: refraction is geometrically impossible.
    let cannot_refract = eta_ratio * sin_theta > 1.0;

    // Schlick approximation: when exiting a dense medium (glass→air) and
    // refraction is possible, the formula should use the transmitted-side
    // cosine rather than the incident cosine, for physically correct Fresnel
    // at grazing angles.
    //   cos_theta_t = sqrt(1 - eta_ratio² · sin²θ_i)
    let schlick_cos = select(
        cos_theta,
        sqrt(max(0.0, 1.0 - eta_ratio * eta_ratio * (1.0 - cos_theta * cos_theta))),
        !hit.front_face && !cannot_refract);

    // Probabilistic Fresnel via Schlick: even when refraction is possible,
    // randomly choose to reflect with probability equal to reflectance.
    let reflectance = schlick(schlick_cos, eta_ratio);

    if cannot_refract || reflectance > rand_f32(rng) {
        r.direction = reflect_vec(ray_in.dir, hit.normal);
    } else {
        // Re-normalise to guard against floating-point drift near TIR boundary.
        r.direction = normalize(refract_vec(ray_in.dir, hit.normal, eta_ratio));
    }
    return r;
}

/// Dispatch to the correct scatter function based on material type.
fn scatter(mat: Material, ray_in: Ray, hit: HitRecord,
           rng: ptr<function, u32>) -> ScatterResult {
    switch mat.type_pad.x {
        case 1u: { return scatter_metal(mat, ray_in, hit, rng); }
        case 2u: { return scatter_dielectric(mat, ray_in, hit, rng); }
        default: { return scatter_lambertian(mat, hit, rng); }
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

        // Clamp to valid range; guards against corrupt sphere mat_index data.
        // Using 7u (MAX_MATERIALS - 1) instead of mat_count-1 avoids underflow
        // when mat_count is 0.
        let mat_idx = min(hit.mat_index, 7u);
        let mat     = materials.mats[mat_idx];

        let sr = scatter(mat, ray, hit, rng);
        if sr.absorbed { return vec3<f32>(0.0, 0.0, 0.0); }

        throughput *= sr.attenuation;
        ray = Ray(hit.point, sr.direction);
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

    // Focal point on the focus plane.  lower_left / horizontal / vertical are
    // already scaled by focus_dist in compute_camera() on the CPU side.
    let focal_point =
          camera.lower_left.xyz
        + u * camera.horizontal.xyz
        + v * camera.vertical.xyz;

    // Sample a random point on the thin-lens aperture disk (depth of field).
    // When aperture = 0 (pinhole), defocus_u/v are zero vectors → no offset.
    let lds        = random_in_unit_disk(&rng);
    let lens_offset = lds.x * camera.defocus_u.xyz
                    + lds.y * camera.defocus_v.xyz;
    let ray_origin  = camera.origin.xyz + lens_offset;
    let ray = Ray(ray_origin, normalize(focal_point - ray_origin));

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
