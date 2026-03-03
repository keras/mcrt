// path_trace.wgsl — Phase 9: BVH-accelerated path tracer
//
// Phase 9 changes over Phase 8:
//   - Scene spheres are now in a runtime-sized storage buffer (binding 2)
//     instead of a fixed-capacity SceneData uniform.  This removes MAX_SPHERES.
//   - A flat BVH node array is added on binding 5.  Each frame the compute
//     kernel traverses the BVH to find intersections instead of a O(N) loop.
//   - Ray-AABB slab test + iterative stack-based BVH traversal are implemented
//     below.  The rest of the pipeline (camera, materials, accumulation) is
//     unchanged from Phase 8.

// ---- module-level constants -----------------------------------------------

/// Full circle in radians (2π).  Used in spherical- and disk-sampling functions.
const TAU: f32 = 6.283185307179586;

/// Maximum material count; must match MAX_MATERIALS in material.rs.
const MAX_MATERIALS: u32 = 8u;

/// Last valid material index (MAX_MATERIALS − 1) for safe clamping.
const MAX_MAT_IDX: u32 = 7u;

/// Minimum ray distance — avoids self-intersection (shadow acne).
/// Assumes world-scale geometry with sphere radii ~0.5–100 units.
const T_MIN: f32 = 1e-4;

/// Maximum ray distance (effectively infinite for this scene scale).
const T_MAX: f32 = 1e9;

/// Frame accumulation cap.  Prevents u32 overflow in the weight denominator
/// (f32 can represent integers exactly up to 2²⁴ ≈ 16 M; we stop earlier).
const MAX_FRAME_COUNT: u32 = 65535u;

// PCG32 PRNG mixing constants (https://www.pcg-random.org/).
const PCG_MULT: u32 = 747796405u;
const PCG_INC:  u32 = 2891336453u;
const PCG_MIX:  u32 = 277803737u;

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

// ---- sphere storage buffer -----------------------------------------------
// Phase 9: the sphere list is a runtime-sized storage buffer, replacing the
// fixed-capacity SceneData uniform.  Leaf BVH nodes index into this array.

struct Sphere {
    center_r:    vec4<f32>,  // .xyz = centre, .w = radius
    mat_and_pad: vec4<u32>,  // .x = material index, .yzw = unused
}

// ---- BVH node buffer ------------------------------------------------------
// Flat BVH node layout (48 bytes, 3 × 16-byte chunks — vec4-aligned).
// Must match GpuBvhNode in bvh.rs.
//
//   prim_count == 0  →  internal node
//       left child  = this_index + 1      (pre-order invariant)
//       right child = right_or_offset
//   prim_count >  0  →  leaf node
//       sphere range = spheres[right_or_offset .. right_or_offset + prim_count]
struct BvhNode {
    aabb_min:        vec4<f32>,  // .xyz = AABB minimum corner, .w = unused
    aabb_max:        vec4<f32>,  // .xyz = AABB maximum corner, .w = unused
    right_or_offset: u32,        // internal: right child index; leaf: sphere start
    prim_count:      u32,        // 0 = internal node; > 0 = leaf (primitive count)
    _pad0:           u32,
    _pad1:           u32,
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

/// Fixed-capacity material table; must match MAX_MATERIALS in material.rs (and this file).
struct MaterialData {
    mat_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    mats: array<Material, MAX_MATERIALS>,
}

// ---- bindings -------------------------------------------------------------

/// Accumulation write target: the compute shader writes the blended average here.
/// Ping-pongs with accum_read every frame.
@group(0) @binding(0) var accum_write: texture_storage_2d<rgba32float, write>;

/// Camera uniform; updated every frame with the current frame index for PRNG.
@group(0) @binding(1) var<uniform> camera: Camera;

/// BVH-reordered sphere list.  Indexed by leaf-node primitive ranges.
@group(0) @binding(2) var<storage, read> spheres: array<Sphere>;

/// Accumulation read source: the previous frame's running average.
/// Ping-pongs with accum_write every frame.
@group(0) @binding(3) var accum_read: texture_2d<f32>;

/// Material descriptors for all spheres in the scene.
@group(0) @binding(4) var<uniform> materials: MaterialData;

/// Flat BVH node array.  Traversal always starts at index 0 (root).
@group(0) @binding(5) var<storage, read> bvh_nodes: array<BvhNode>;

// ---- PCG PRNG -------------------------------------------------------------
//
// PCG32/XSH-RR: one 32-bit state, 32-bit output.
// Reference: https://www.pcg-random.org/

/// Single PCG32 step; maps any seed to a well-distributed u32.
fn pcg_hash(v: u32) -> u32 {
    var s = v * PCG_MULT + PCG_INC;
    // XSH-RR output mixing.
    let rot = s >> 28u;
    s = ((s >> (rot + 4u)) ^ s) * PCG_MIX;
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

// ---- ray-AABB intersection (slab method) ----------------------------------

/// Test ray `r` against axis-aligned bounding box [bb_min, bb_max] within the
/// open interval (t_min, t_max).  Uses branchless min/max per axis so there is
/// no divergence for rays with negative direction components.
///
/// **NaN guard**: direction components with magnitude < 1e-7 are replaced by
/// a large finite reciprocal (±1e30) so `0 * ∞ = NaN` cannot occur on slab
/// boundaries.  This is the Kensler sign-safe variant.
fn ray_aabb_hit(r: Ray, bb_min: vec3<f32>, bb_max: vec3<f32>, t_min: f32, t_max: f32) -> bool {
    // Clamp near-zero direction components to avoid NaN when origin sits exactly
    // on a slab boundary (0 * ±Inf = NaN in IEEE 754, GPU-undefined in WGSL).
    let safe_d = select(r.dir, sign(r.dir + vec3(1e-30)) * vec3(1e-7), abs(r.dir) < vec3(1e-7));
    let inv_d = 1.0 / safe_d;
    let ta    = (bb_min - r.origin) * inv_d;
    let tb    = (bb_max - r.origin) * inv_d;
    // Narrow the interval [t0, t1] with each axis slab.
    let t0 = max(t_min, max(min(ta.x, tb.x), max(min(ta.y, tb.y), min(ta.z, tb.z))));
    let t1 = min(t_max, min(max(ta.x, tb.x), min(max(ta.y, tb.y), max(ta.z, tb.z))));
    return t1 > t0;
}

// ---- BVH traversal --------------------------------------------------------

/// Finds the closest sphere hit along ray `r` in (t_min, t_max) via BVH traversal.
///
/// Uses a fixed-size iterative stack (32 entries) to avoid recursion.
/// The pre-order layout guarantees that the left child always immediately
/// follows the parent: `left = node_idx + 1`.  The right child index is
/// stored explicitly in `node.right_or_offset`.
///
/// Right is pushed before left so the left sub-tree is popped and processed
/// first, matching both spatial proximity and memory locality.
fn scene_hit(r: Ray, t_min: f32, t_max: f32) -> HitRecord {
    var best: HitRecord;
    best.t     = -1.0;
    var closest = t_max;

    let node_count = i32(arrayLength(&bvh_nodes));
    if node_count == 0 { return best; }

    // Fixed traversal stack — large enough for trees with ≤ 512 leaf primitives.
    var stack: array<u32, 32>;
    var sp: i32 = 0;  // index of the top-of-stack element
    stack[0] = 0u;    // push root (index 0)

    while sp >= 0 {
        let node_idx = stack[u32(sp)];
        sp -= 1;

        let node = bvh_nodes[node_idx];

        // Reject immediately if this node's AABB is not hit.
        if !ray_aabb_hit(r, node.aabb_min.xyz, node.aabb_max.xyz, t_min, closest) {
            continue;
        }

        if node.prim_count > 0u {
            // Leaf: test every primitive in the range.
            let prim_end = node.right_or_offset + node.prim_count;
            for (var i = node.right_or_offset; i < prim_end; i++) {
                let hit = ray_sphere_hit(r, spheres[i], t_min, closest);
                if hit.t > 0.0 {
                    best    = hit;
                    closest = hit.t;
                }
            }
        } else {
            // Internal node: push right child first, then left.
            // Guard: stack capacity is 32 entries (indices 0..31).  For scenes with
            // ≤ 512 primitives the SAH tree depth is ≤ 9, so sp never exceeds ~18
            // during normal traversal.  The guard at sp < 30 ensures two safe pushes
            // (sp would reach at most 31 = last valid index).  If sp ≥ 30 the subtree
            // is silently skipped; this cannot happen for any scene registered in this
            // application, but a future scene with ≥32 nesting levels would need a
            // larger stack or an iterative builder.
            if sp < 30 {
                sp += 1;
                stack[u32(sp)] = node.right_or_offset;  // right child
                sp += 1;
                stack[u32(sp)] = node_idx + 1u;          // left child (pre-order)
            }
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
    let theta   = TAU * rand_f32(rng);                    // 2π
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
    let theta = TAU * rand_f32(rng);
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
        let hit = scene_hit(ray, T_MIN, T_MAX);

        if hit.t < 0.0 {
            // Ray escaped — multiply throughput by sky radiance and return.
            return throughput * sky_color(ray.dir);
        }

        // Clamp to valid range; guards against corrupt sphere mat_index data.
        // Using MAX_MAT_IDX instead of mat_count-1 avoids underflow when
        // mat_count is 0, and ties the value to the array-size constant.
        let mat_idx = min(hit.mat_index, MAX_MAT_IDX);
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
    // Cap at MAX_FRAME_COUNT frames: prevents the denominator from wrapping to
    // 0 at u32::MAX (which would produce +Inf weight, corrupting pixels with NaN).
    let weight = 1.0 / f32(min(frame_count, MAX_FRAME_COUNT) + 1u);
    let prev   = textureLoad(accum_read, coord, 0).xyz;
    let accum  = mix(prev, sample, weight);

    // Store raw HDR radiance — tone-mapping is applied in the display pass.
    textureStore(accum_write, coord, vec4<f32>(accum, 1.0));
}
