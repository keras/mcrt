// path_trace.wgsl — GPU path tracer: phases 9-14
//
// Changes from the original phases 1-8 shader:
//
//   Binding 2 (spheres): was `var<uniform> scene: SceneData` with a fixed
//   8-entry array; now `var<storage, read> spheres: array<Sphere>` — an
//   unbounded runtime-sized array matching the BGL declaration.  Sphere count
//   is obtained via `arrayLength(&spheres)`.
//
//   Bindings 5-13 (new): sphere BVH, mesh vertices, mesh triangles, mesh BVH,
//   albedo texture sampler, albedo texture array, HDR env map, emissive sphere
//   list, G-buffer write target.
//
//   Traversal: replaces the O(N) linear sphere loop with stack-based BVH
//   traversal for both spheres (binding 5) and triangle meshes (binding 8).
//
//   Triangle intersection: Möller–Trumbore with barycentric normal / UV
//   interpolation for smooth shading.
//
//   Materials: MAX_MATERIALS raised to 64. Emissive material type 3 handled —
//   terminates the path and returns the stored emission radiance.
//
//   Textures: albedo texture array (binding 10) looked up via the texture layer
//   stored in material.type_pad.y; binding 9 provides the linear sampler.
//
//   Environment map: HDR equirectangular map always sampled via textureLoad
//   (non-filterable Rgba32Float; binding 11).
//
//   G-buffer: written once per pixel at the primary hit (depth + world normal)
//   so the denoiser pass can read geometry data for bilateral filtering.

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const TAU: f32 = 6.283185307179586;
const PI:  f32 = 3.141592653589793;

/// Maximum material count — must match `MAX_MATERIALS` in `material.rs`.
const MAX_MATERIALS: u32 = 64u;

/// Highest valid material index (used for safe clamping).
const MAX_MAT_IDX: u32 = 63u;

/// Emissive material type — must match `MAT_EMISSIVE` in `material.rs`.
const MAT_EMISSIVE: u32 = 3u;

/// BVH traversal stack depth.  64 levels handles trees over ~2^64 primitives.
const BVH_STACK_SIZE: u32 = 64u;

/// Minimum ray distance — avoids self-intersection (shadow acne).
const T_MIN: f32 = 1e-4;

/// Maximum ray distance (effectively infinite for this scene scale).
const T_MAX: f32 = 1e9;

/// Frame accumulation cap prevents the weight denominator from overflowing.
const MAX_FRAME_COUNT: u32 = 65535u;

/// Maximum path bounces before the path is terminated.
const MAX_BOUNCES: u32 = 8u;

// PCG32 PRNG mixing constants (https://www.pcg-random.org/).
const PCG_MULT: u32 = 747796405u;
const PCG_INC:  u32 = 2891336453u;
const PCG_MIX:  u32 = 277803737u;

// ---------------------------------------------------------------------------
// Shared primitive structs
// ---------------------------------------------------------------------------

/// A ray in world space.  `dir` must be unit length.
struct Ray {
    origin: vec3<f32>,
    dir:    vec3<f32>,
}

/// Record of the closest intersection.  `t < 0` = no hit.
struct HitRecord {
    t:          f32,
    point:      vec3<f32>,
    normal:     vec3<f32>,   // always faces the incident ray
    uv:         vec2<f32>,   // interpolated texture coordinates
    front_face: bool,
    mat_index:  u32,
}

// ---------------------------------------------------------------------------
// Camera uniform — layout must match `CameraUniform` in `camera.rs`.
// ---------------------------------------------------------------------------

struct Camera {
    origin:      vec4<f32>,
    lower_left:  vec4<f32>,
    horizontal:  vec4<f32>,
    vertical:    vec4<f32>,
    defocus_u:   vec4<f32>,
    defocus_v:   vec4<f32>,
    frame_count: u32,
    _pad0: u32, _pad1: u32, _pad2: u32,
}

// ---------------------------------------------------------------------------
// Scene primitive structs — layouts match repr(C) types in Rust.
// ---------------------------------------------------------------------------

struct Sphere {
    center_r:    vec4<f32>,  // .xyz = centre, .w = radius
    mat_and_pad: vec4<u32>,  // .x = material index, .yzw unused
}

/// Flat BVH node (pre-order; root = 0).
/// Internal: left = node+1, right = right_or_offset.
/// Leaf: primitives at [right_or_offset, right_or_offset+prim_count).
struct BvhNode {
    aabb_min:        vec4<f32>,
    aabb_max:        vec4<f32>,
    right_or_offset: u32,
    prim_count:      u32,
    _pad0:           u32,
    _pad1:           u32,
}

struct ProbeGrid {
    // .xyz = origin, .w = spacing
    origin_spacing: vec4<f32>,
    // .xyz = dimensions, .w = total probes
    dims_count: vec4<u32>,
    // Debug visualization flags
    show_grid: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct Vertex {
    position: vec4<f32>,  // .xyz = world-space position
    normal:   vec4<f32>,  // .xyz = vertex normal
    uv:       vec4<f32>,  // .xy  = texture coordinates
}

/// Triangle with three separate u32 vertex indices to guarantee matching
/// storage layout with `GpuTriangle` repr(C) in Rust.
struct Triangle {
    v0:      u32,
    v1:      u32,
    v2:      u32,
    mat_idx: u32,
}

// ---------------------------------------------------------------------------
// Material structs — layout must match `GpuMaterial` / `GpuMaterialData`.
// ---------------------------------------------------------------------------

/// type_pad.x  : 0=Lambertian, 1=Metal, 2=Dielectric, 3=Emissive
/// type_pad.y  : albedo texture array layer (0 = solid colour)
/// albedo_fuzz : .xyz = albedo / emission colour, .w = fuzz (metal)
/// ior_pad.x   : IOR (dielectric) or emission strength (emissive)
struct Material {
    type_pad:    vec4<u32>,
    albedo_fuzz: vec4<f32>,
    ior_pad:     vec4<f32>,
}

/// Full material table.  Array literal 64 must match `MAX_MATERIALS` in Rust.
struct MaterialData {
    mat_count:  u32,
    n_emissive: u32,
    _pad1:      u32,
    _pad2:      u32,
    mats:       array<Material, 64>,
}

struct ScatterResult {
    direction:   vec3<f32>,
    attenuation: vec3<f32>,
    absorbed:    bool,
}

// ---------------------------------------------------------------------------
// Bindings — group 0, bindings 0-13
// ---------------------------------------------------------------------------

@group(0) @binding(0)  var         accum_write:      texture_storage_2d<rgba32float, write>;
@group(0) @binding(1)  var<uniform> camera:           Camera;
/// Dynamic sphere list — unbounded storage array; use arrayLength(&spheres).
@group(0) @binding(2)  var<storage, read> spheres:   array<Sphere>;
@group(0) @binding(3)  var         accum_read:       texture_2d<f32>;
@group(0) @binding(4)  var<uniform> materials:       MaterialData;
@group(0) @binding(5)  var<storage, read> sphere_bvh: array<BvhNode>;
@group(0) @binding(6)  var<storage, read> vertices:  array<Vertex>;
@group(0) @binding(7)  var<storage, read> triangles: array<Triangle>;
@group(0) @binding(8)  var<storage, read> mesh_bvh:  array<BvhNode>;
@group(0) @binding(9)  var         tex_sampler:      sampler;
@group(0) @binding(10) var         albedo_tex:       texture_2d_array<f32>;
/// Non-filterable HDR env map; sampled via textureLoad.
@group(0) @binding(11) var         env_map:          texture_2d<f32>;
@group(0) @binding(12) var<storage, read> emissive_spheres: array<Sphere>;
/// G-buffer: primary hit normal (.xyz) + linear depth (.w).
@group(0) @binding(13) var         gbuffer_write:    texture_storage_2d<rgba32float, write>;
@group(0) @binding(14) var<uniform> grid:             ProbeGrid;

// ---------------------------------------------------------------------------
// PCG32 PRNG
// ---------------------------------------------------------------------------

fn pcg_hash(v: u32) -> u32 {
    var s = v * PCG_MULT + PCG_INC;
    let rot = s >> 28u;
    s = ((s >> (rot + 4u)) ^ s) * PCG_MIX;
    return (s >> 22u) ^ s;
}

fn rand_f32(rng: ptr<function, u32>) -> f32 {
    *rng = pcg_hash(*rng);
    return bitcast<f32>((*rng >> 9u) | 0x3f800000u) - 1.0;
}

// ---------------------------------------------------------------------------
// Sampling helpers
// ---------------------------------------------------------------------------

fn random_unit_sphere(rng: ptr<function, u32>) -> vec3<f32> {
    let cos_phi = 1.0 - 2.0 * rand_f32(rng);
    let sin_phi = sqrt(max(0.0, 1.0 - cos_phi * cos_phi));
    let theta   = TAU * rand_f32(rng);
    return vec3<f32>(sin_phi * cos(theta), sin_phi * sin(theta), cos_phi);
}

fn cosine_scatter(rng: ptr<function, u32>, normal: vec3<f32>) -> vec3<f32> {
    let scatter = normal + random_unit_sphere(rng);
    if dot(scatter, scatter) < 1e-6 { return normal; }
    return normalize(scatter);
}

fn random_in_unit_disk(rng: ptr<function, u32>) -> vec2<f32> {
    let r     = sqrt(rand_f32(rng));
    let theta = TAU * rand_f32(rng);
    return r * vec2<f32>(cos(theta), sin(theta));
}

// ---------------------------------------------------------------------------
// Environment map (equirectangular lookup via textureLoad)
// ---------------------------------------------------------------------------

fn sample_env_map(dir: vec3<f32>) -> vec3<f32> {
    let phi   = atan2(dir.z, dir.x);            // [-π, π]
    let theta = asin(clamp(dir.y, -1.0, 1.0));  // [-π/2, π/2]
    let u     = phi / TAU + 0.5;               // [0, 1]
    let v     = 0.5 - theta / PI;              // [0, 1], y-up
    let dims  = textureDimensions(env_map);
    let ix    = u32(u * f32(dims.x)) % dims.x;
    let iy    = u32(v * f32(dims.y)) % dims.y;
    return textureLoad(env_map, vec2<u32>(ix, iy), 0).xyz;
}

// ---------------------------------------------------------------------------
// Ray-AABB slab test
// ---------------------------------------------------------------------------

fn ray_aabb_hit(r: Ray, aabb_min: vec3<f32>, aabb_max: vec3<f32>,
                t_min: f32, t_max: f32) -> bool {
    let inv_d  = 1.0 / r.dir;
    let t0     = (aabb_min - r.origin) * inv_d;
    let t1     = (aabb_max - r.origin) * inv_d;
    let tenter = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
    let texit  = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
    return texit > max(tenter, t_min) && tenter < t_max;
}

// ---------------------------------------------------------------------------
// Ray-sphere intersection
// ---------------------------------------------------------------------------

fn ray_sphere_hit(r: Ray, sphere: Sphere, t_min: f32, t_max: f32) -> HitRecord {
    let center = sphere.center_r.xyz;
    let radius = sphere.center_r.w;
    let oc     = r.origin - center;
    let a      = dot(r.dir, r.dir);
    let h      = dot(oc, r.dir);
    let c      = dot(oc, oc) - radius * radius;
    let disc   = h * h - a * c;

    var hit: HitRecord;
    hit.t = -1.0;
    if disc < 0.0 { return hit; }

    let sqrtd = sqrt(disc);
    var t = (-h - sqrtd) / a;
    if t <= t_min || t >= t_max {
        t = (-h + sqrtd) / a;
        if t <= t_min || t >= t_max { return hit; }
    }

    hit.t      = t;
    hit.point  = r.origin + t * r.dir;
    let outn   = (hit.point - center) / radius;
    hit.front_face = dot(r.dir, outn) < 0.0;
    hit.normal = select(-outn, outn, hit.front_face);
    hit.mat_index = sphere.mat_and_pad.x;
    // Spherical UV mapping: u = longitude [0,1], v = latitude [0,1].
    let phi   = atan2(outn.z, outn.x);             // [-π, π]
    let theta = asin(clamp(outn.y, -1.0, 1.0));    // [-π/2, π/2]
    hit.uv    = vec2<f32>(phi / TAU + 0.5, 0.5 - theta / PI);
    return hit;
}

// ---------------------------------------------------------------------------
// Ray-triangle intersection — Möller–Trumbore
// ---------------------------------------------------------------------------

/// Returns (t, u, v, 1.0) on hit with t ∈ (t_min, t_max), else (-1,0,0,0).
fn ray_triangle_uvt(r: Ray, v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>,
                    t_min: f32, t_max: f32) -> vec4<f32> {
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let h  = cross(r.dir, e2);
    let a  = dot(e1, h);
    if abs(a) < 1e-8 { return vec4<f32>(-1.0, 0.0, 0.0, 0.0); }
    let f  = 1.0 / a;
    let s  = r.origin - v0;
    let u  = f * dot(s, h);
    if u < 0.0 || u > 1.0 { return vec4<f32>(-1.0, 0.0, 0.0, 0.0); }
    let q  = cross(s, e1);
    let v  = f * dot(r.dir, q);
    if v < 0.0 || u + v > 1.0 { return vec4<f32>(-1.0, 0.0, 0.0, 0.0); }
    let t  = f * dot(e2, q);
    if t < t_min || t >= t_max { return vec4<f32>(-1.0, 0.0, 0.0, 0.0); }
    return vec4<f32>(t, u, v, 1.0);
}

// ---------------------------------------------------------------------------
// BVH traversal helpers — shared stack/AABB DFS logic
//
// WGSL does not support ptr<storage> parameters or generic functions, so we
// cannot write a single traversal helper that works with both sphere_bvh and
// mesh_bvh.  Instead, each BVH gets its own `*_next_leaf` helper that shares
// identical stack-management code.  Bug fixes to the DFS or AABB test only
// need to be applied to the two ~10-line helpers, not to both full hit
// functions.
//
// Both helpers:
//   • Pop nodes from the stack and AABB-cull with the caller's current t_max.
//   • On finding a leaf (prim_count > 0) they return the node index so the
//     caller can test its primitives.
//   • On finding an internal node they push right then left child (DFS order).
//   • Return 0xFFFFFFFFu when the stack is exhausted (traversal complete).
//
// Because t_max is passed by value the caller must re-pass its updated t_max
// after recording a closer hit — this ensures subsequent AABB tests cull more
// aggressively and mirrors the original in-loop t_max shrinkage.
// ---------------------------------------------------------------------------

/// Advance the sphere BVH stack to the next leaf node.
/// Returns the node index, or 0xFFFFFFFFu when traversal is complete.
fn sphere_bvh_next_leaf(
    r:     Ray,
    t_min: f32,
    t_max: f32,
    stack: ptr<function, array<u32, 64>>,
    top:   ptr<function, u32>,
) -> u32 {
    while *top > 0u {
        *top -= 1u;
        let ni   = (*stack)[*top];
        let node = sphere_bvh[ni];
        if !ray_aabb_hit(r, node.aabb_min.xyz, node.aabb_max.xyz, t_min, t_max) { continue; }
        if node.prim_count > 0u { return ni; }
        // Push right child first so left child (ni+1) is visited first (DFS).
        // Guard: BVH depth for N prims is ⌈log₂ N⌉ ≪ 64; overflow is unreachable.
        if *top + 2u <= BVH_STACK_SIZE {
            (*stack)[*top] = node.right_or_offset; *top += 1u;
            (*stack)[*top] = ni + 1u;              *top += 1u;
        }
    }
    return 0xFFFFFFFFu;
}

/// Advance the mesh BVH stack to the next leaf node.
/// Returns the node index, or 0xFFFFFFFFu when traversal is complete.
fn mesh_bvh_next_leaf(
    r:     Ray,
    t_min: f32,
    t_max: f32,
    stack: ptr<function, array<u32, 64>>,
    top:   ptr<function, u32>,
) -> u32 {
    while *top > 0u {
        *top -= 1u;
        let ni   = (*stack)[*top];
        let node = mesh_bvh[ni];
        if !ray_aabb_hit(r, node.aabb_min.xyz, node.aabb_max.xyz, t_min, t_max) { continue; }
        if node.prim_count > 0u { return ni; }
        if *top + 2u <= BVH_STACK_SIZE {
            (*stack)[*top] = node.right_or_offset; *top += 1u;
            (*stack)[*top] = ni + 1u;              *top += 1u;
        }
    }
    return 0xFFFFFFFFu;
}

// ---------------------------------------------------------------------------
// BVH traversal — sphere BVH (binding 5)
// ---------------------------------------------------------------------------

fn sphere_bvh_hit(r: Ray, t_min: f32, t_max_in: f32) -> HitRecord {
    var best: HitRecord;
    best.t = -1.0;
    if arrayLength(&sphere_bvh) == 0u { return best; }

    var t_max = t_max_in;
    var stack: array<u32, 64>;
    var top = 0u;
    stack[top] = 0u; top += 1u;

    loop {
        let leaf_ni = sphere_bvh_next_leaf(r, t_min, t_max, &stack, &top);
        if leaf_ni == 0xFFFFFFFFu { break; }
        let node = sphere_bvh[leaf_ni];
        for (var j = 0u; j < node.prim_count; j += 1u) {
            let hit = ray_sphere_hit(r, spheres[node.right_or_offset + j], t_min, t_max);
            if hit.t > 0.0 { best = hit; t_max = hit.t; }
        }
    }
    return best;
}

// ---------------------------------------------------------------------------
// BVH traversal — mesh BVH (bindings 6, 7, 8)
// ---------------------------------------------------------------------------

fn mesh_bvh_hit(r: Ray, t_min: f32, t_max_in: f32) -> HitRecord {
    var best: HitRecord;
    best.t = -1.0;
    if arrayLength(&mesh_bvh) == 0u { return best; }

    var t_max = t_max_in;
    var stack: array<u32, 64>;
    var top = 0u;
    stack[top] = 0u; top += 1u;

    loop {
        let leaf_ni = mesh_bvh_next_leaf(r, t_min, t_max, &stack, &top);
        if leaf_ni == 0xFFFFFFFFu { break; }
        let node = mesh_bvh[leaf_ni];
        for (var j = 0u; j < node.prim_count; j += 1u) {
            let tri = triangles[node.right_or_offset + j];
            let va  = vertices[tri.v0];
            let vb  = vertices[tri.v1];
            let vc  = vertices[tri.v2];
            // Bit 31 of mat_idx carries the backface-cull flag set by
            // scene.rs (BACKFACE_CULL_FLAG = 1u32 << 31).
            let backface_cull = (tri.mat_idx >> 31u) != 0u;
            let clean_mat_idx = tri.mat_idx & 0x7FFFFFFFu;
            let res = ray_triangle_uvt(
                r, va.position.xyz, vb.position.xyz, vc.position.xyz, t_min, t_max);
            if res.x > 0.0 {
                let t = res.x;
                let u = res.y;
                let v = res.z;
                let w = 1.0 - u - v;
                // Barycentric interpolation of normal and UV.
                let n = normalize(w * va.normal.xyz + u * vb.normal.xyz + v * vc.normal.xyz);
                let front = dot(r.dir, n) < 0.0;
                if backface_cull && !front { continue; }
                var hit: HitRecord;
                hit.t          = t;
                hit.point      = r.origin + t * r.dir;
                hit.normal     = select(-n, n, front);
                hit.front_face = front;
                hit.mat_index  = clean_mat_idx;
                hit.uv         = w * va.uv.xy + u * vb.uv.xy + v * vc.uv.xy;
                best = hit; t_max = t;
            }
        }
    }
    return best;
}

// ---------------------------------------------------------------------------
// Combined world hit
// ---------------------------------------------------------------------------

fn world_hit(r: Ray, t_min: f32, t_max: f32) -> HitRecord {
    let sh = sphere_bvh_hit(r, t_min, t_max);
    // Narrow the mesh BVH query with the sphere tmax for early exit.
    let mesh_tmax = select(t_max, sh.t, sh.t > 0.0);
    let mh = mesh_bvh_hit(r, t_min, mesh_tmax);
    // select() does not work on struct types in WGSL — use an explicit if.
    if mh.t > 0.0 { return mh; }
    return sh;
}

// ---------------------------------------------------------------------------
// Reflect / refract / Schlick helpers
// ---------------------------------------------------------------------------

fn reflect_vec(v: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    return v - 2.0 * dot(v, n) * n;
}

fn refract_vec(v: vec3<f32>, n: vec3<f32>, eta_ratio: f32) -> vec3<f32> {
    let cos_theta = min(dot(-v, n), 1.0);
    let r_perp    = eta_ratio * (v + cos_theta * n);
    let r_para    = -sqrt(max(0.0, 1.0 - dot(r_perp, r_perp))) * n;
    return r_perp + r_para;
}

fn schlick(cos_theta: f32, eta_ratio: f32) -> f32 {
    var r0 = (1.0 - eta_ratio) / (1.0 + eta_ratio);
    r0     = r0 * r0;
    return r0 + (1.0 - r0) * pow(1.0 - cos_theta, 5.0);
}

// ---------------------------------------------------------------------------
// Albedo lookup — solid colour or texture array sample (binding 9, 10)
// ---------------------------------------------------------------------------

fn material_albedo(mat: Material, uv: vec2<f32>) -> vec3<f32> {
    let layer = mat.type_pad.y;
    if layer == 0u { return mat.albedo_fuzz.xyz; }
    let n_layers  = u32(textureNumLayers(albedo_tex));
    // Guard against empty texture array (n_layers == 0 → subtraction wraps to u32::MAX).
    if n_layers == 0u { return mat.albedo_fuzz.xyz; }
    let safe      = min(layer, n_layers - 1u);
    let sampled   = textureSampleLevel(albedo_tex, tex_sampler, uv, i32(safe), 0.0);
    return sampled.xyz * mat.albedo_fuzz.xyz;
}

// ---------------------------------------------------------------------------
// Scatter functions
// ---------------------------------------------------------------------------

fn scatter_lambertian(mat: Material, hit: HitRecord,
                      rng: ptr<function, u32>) -> ScatterResult {
    var r: ScatterResult;
    r.direction   = cosine_scatter(rng, hit.normal);
    r.attenuation = material_albedo(mat, hit.uv);
    r.absorbed    = false;
    return r;
}

fn scatter_metal(mat: Material, ray_in: Ray, hit: HitRecord,
                 rng: ptr<function, u32>) -> ScatterResult {
    var r: ScatterResult;
    let fuzz      = mat.albedo_fuzz.w;
    let reflected = reflect_vec(normalize(ray_in.dir), hit.normal);
    let perturbed = reflected + fuzz * random_unit_sphere(rng);
    r.direction   = normalize(select(reflected, perturbed,
                                     dot(perturbed, perturbed) >= 1e-6));
    r.attenuation = material_albedo(mat, hit.uv);
    r.absorbed    = dot(r.direction, hit.normal) <= 0.0;
    return r;
}

fn scatter_dielectric(mat: Material, ray_in: Ray, hit: HitRecord,
                      rng: ptr<function, u32>) -> ScatterResult {
    var r: ScatterResult;
    r.absorbed    = false;
    r.attenuation = vec3<f32>(1.0);
    let ior       = mat.ior_pad.x;
    let eta_ratio = select(ior, 1.0 / ior, hit.front_face);
    let cos_theta = min(dot(-ray_in.dir, hit.normal), 1.0);
    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    let cannot_refract = eta_ratio * sin_theta > 1.0;
    let schlick_cos = select(
        cos_theta,
        sqrt(max(0.0, 1.0 - eta_ratio * eta_ratio * (1.0 - cos_theta * cos_theta))),
        !hit.front_face && !cannot_refract);
    let reflectance = schlick(schlick_cos, eta_ratio);
    if cannot_refract || reflectance > rand_f32(rng) {
        r.direction = reflect_vec(ray_in.dir, hit.normal);
    } else {
        r.direction = normalize(refract_vec(ray_in.dir, hit.normal, eta_ratio));
    }
    return r;
}

fn scatter(mat: Material, ray_in: Ray, hit: HitRecord,
           rng: ptr<function, u32>) -> ScatterResult {
    switch mat.type_pad.x {
        case 1u: { return scatter_metal(mat, ray_in, hit, rng); }
        case 2u: { return scatter_dielectric(mat, ray_in, hit, rng); }
        default: { return scatter_lambertian(mat, hit, rng); }
    }
}

// ---------------------------------------------------------------------------
// Path tracing
// ---------------------------------------------------------------------------

/// Trace `initial_ray`, write the primary-hit G-buffer, return radiance.
///
/// Emissive surfaces (type 3) terminate the path and return their stored
/// emission (`albedo_fuzz.xyz * ior_pad.x` = colour × strength multiplier).
fn path_color(initial_ray: Ray, coord: vec2<u32>,
              rng: ptr<function, u32>,
              primary_hit_t: ptr<function, f32>) -> vec3<f32> {
    var ray        = initial_ray;
    var throughput = vec3<f32>(1.0);
    var first_hit  = true;

    for (var bounce = 0u; bounce < MAX_BOUNCES; bounce += 1u) {
        let hit = world_hit(ray, T_MIN, T_MAX);

        if hit.t < 0.0 {
            if first_hit {
                *primary_hit_t = T_MAX;
                textureStore(gbuffer_write, coord, vec4<f32>(0.0, 0.0, 0.0, T_MAX));
            }
            return throughput * sample_env_map(ray.dir);
        }

        if first_hit {
            *primary_hit_t = hit.t;
            textureStore(gbuffer_write, coord, vec4<f32>(hit.normal, hit.t));
            first_hit = false;
        }

        let mat_idx = min(hit.mat_index, MAX_MAT_IDX);
        let mat     = materials.mats[mat_idx];

        // Emissive: return emission and end path.
        if mat.type_pad.x == MAT_EMISSIVE {
            return throughput * mat.albedo_fuzz.xyz * mat.ior_pad.x;
        }

        let sr = scatter(mat, ray, hit, rng);
        if sr.absorbed { return vec3<f32>(0.0); }

        throughput *= sr.attenuation;
        ray = Ray(hit.point, sr.direction);
    }

    return vec3<f32>(0.0);
}

// ---------------------------------------------------------------------------
// Compute entry point
// ---------------------------------------------------------------------------

@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims  = textureDimensions(accum_write);
    let coord = gid.xy;

    if coord.x >= dims.x || coord.y >= dims.y { return; }

    let frame_count = camera.frame_count;

    // Seed PRNG uniquely per pixel per frame.
    var rng = pcg_hash(pcg_hash(coord.x + coord.y * dims.x)
                       ^ (frame_count * 2654435761u));

    // Sub-pixel jitter for AA.
    let u = (f32(coord.x) + rand_f32(&rng)) / f32(dims.x);
    let v = 1.0 - (f32(coord.y) + rand_f32(&rng)) / f32(dims.y);

    let focal_point =
          camera.lower_left.xyz
        + u * camera.horizontal.xyz
        + v * camera.vertical.xyz;

    let lds         = random_in_unit_disk(&rng);
    let lens_offset = lds.x * camera.defocus_u.xyz
                    + lds.y * camera.defocus_v.xyz;
    let ray_origin  = camera.origin.xyz + lens_offset;
    let ray = Ray(ray_origin, normalize(focal_point - ray_origin));

    var primary_hit_t = T_MAX;
    var sample = path_color(ray, coord, &rng, &primary_hit_t);

    // ---- Phase IC-1: Debug visualization for probe grid ------------------
    if (grid.show_grid != 0u) {
        let probe_radius = 0.05;
        let origin = grid.origin_spacing.xyz;
        let spacing = grid.origin_spacing.w;
        let dms = vec3<f32>(grid.dims_count.xyz);

        // Fast raymarching of the grid's distance field
        var t = 0.0;
        for (var step = 0; step < 64; step = step + 1) {
            let p = ray.origin + t * ray.dir;

            // local pos in grid space
            let local_p = (p - origin) / spacing;

            // nearest grid node
            let node_local = clamp(round(local_p), vec3<f32>(0.0), dms - vec3<f32>(1.0));
            let node_world = origin + node_local * spacing;

            let d = distance(p, node_world) - probe_radius;
            if d < 0.001 {
                if t > T_MIN && t < primary_hit_t {
                    sample = vec3<f32>(1.0, 0.5, 0.0); // Orange probes
                }
                break;
            }
            t += d;
            if t > primary_hit_t || t > 100.0 {
                break; // occluded or escaped grid
            }
        }
    }

    // Progressive accumulation (running average).
    let weight = 1.0 / f32(min(frame_count, MAX_FRAME_COUNT) + 1u);
    let prev   = textureLoad(accum_read, coord, 0).xyz;
    let accum  = mix(prev, sample, weight);

    textureStore(accum_write, coord, vec4<f32>(accum, 1.0));
}
