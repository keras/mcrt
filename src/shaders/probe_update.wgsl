// probe_update.wgsl — Phase IC-2: Irradiance Probe Radiance Capture
//
// One workgroup per probe.  Each thread traces NUM_RAYS_PER_PROBE/WORKGROUP_SIZE
// rays uniformly distributed over the sphere (Fibonacci lattice + per-frame
// random rotation), accumulates L2 SH coefficients, and blends into the flat
// irradiance buffer using a hysteresis EMA.
//
// Binding layout (group 0) — probe update BGL defined in gpu_layout.rs:
//   0  probe_params   : uniform  ProbeUpdateParams
//   1  grid           : uniform  ProbeGrid
//   2  irradiance_buf : storage  read_write  array<f32>   (9 coeffs * 3 channels per probe)
//   3  spheres        : storage  read         array<Sphere>
//   4  sphere_bvh     : storage  read         array<BvhNode>
//   5  vertices       : storage  read         array<Vertex>
//   6  triangles      : storage  read         array<Triangle>
//   7  mesh_bvh       : storage  read         array<BvhNode>
//   8  materials      : uniform  MaterialData
//   9  emissives      : storage  read         array<Sphere>
//  10  env_map        : texture_2d<f32>       (HDR env map, non-filterable)

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const TAU: f32 = 6.283185307179586;
const PI:  f32 = 3.141592653589793;
const T_MIN: f32 = 1e-3;
const T_MAX: f32 = 1e9;
const MAX_MAT_IDX: u32 = 63u;
const MAT_EMISSIVE: u32 = 3u;
const BVH_STACK_SIZE: u32 = 64u;
const MAX_MATERIALS: u32 = 64u;

// One workgroup handles one probe.
// NUM_RAYS divide work across threads within the workgroup.
const WORKGROUP_SIZE: u32 = 64u;  // threads per workgroup (= rays traced per frame per probe)

// L2 SH coefficient count (9 real coefficients for RGB)
const SH_COEFF: u32 = 9u;
// Total f32 values per probe in irradiance buffer: 9 coeffs * 3 channels
const SH_F32_PER_PROBE: u32 = 27u;

// PCG32 PRNG constants
const PCG_MULT: u32 = 747796405u;
const PCG_INC:  u32 = 2891336453u;
const PCG_MIX:  u32 = 277803737u;

// ---------------------------------------------------------------------------
// Structs (must match Rust repr(C) types)
// ---------------------------------------------------------------------------

struct ProbeUpdateParams {
    frame_count:        u32,
    probes_per_frame:   u32,  // max probes to update this frame
    probe_offset:       u32,  // time-slice starting probe index
    hysteresis:         f32,  // 1.0 - alpha; typical 0.97
}

struct ProbeGrid {
    origin_spacing: vec4<f32>,  // .xyz = origin, .w = spacing
    dims_count:     vec4<u32>,  // .xyz = dimensions, .w = total probes
    show_grid:      u32,
    _pad0: u32, _pad1: u32, _pad2: u32,
}

struct Ray {
    origin: vec3<f32>,
    dir:    vec3<f32>,
}

struct HitRecord {
    t:          f32,
    point:      vec3<f32>,
    normal:     vec3<f32>,
    uv:         vec2<f32>,
    front_face: bool,
    mat_index:  u32,
}

struct Sphere {
    center_r:    vec4<f32>,
    mat_and_pad: vec4<u32>,
}

struct BvhNode {
    aabb_min:        vec4<f32>,
    aabb_max:        vec4<f32>,
    right_or_offset: u32,
    prim_count:      u32,
    _pad0: u32, _pad1: u32,
}

struct Vertex {
    position: vec4<f32>,
    normal:   vec4<f32>,
    uv:       vec4<f32>,
}

struct Triangle {
    v0:      u32,
    v1:      u32,
    v2:      u32,
    mat_idx: u32,
}

struct Material {
    type_pad:    vec4<u32>,
    albedo_fuzz: vec4<f32>,
    ior_pad:     vec4<f32>,
}

struct MaterialData {
    mat_count:  u32,
    n_emissive: u32,
    _pad1: u32, _pad2: u32,
    mats: array<Material, 64>,
}

// Partial SH accumulator for one thread (9 × RGB = 27 f32)
struct ShAccum {
    r: array<f32, 9>,
    g: array<f32, 9>,
    b: array<f32, 9>,
}

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

@group(0) @binding(0)  var<uniform>             probe_params:  ProbeUpdateParams;
@group(0) @binding(1)  var<uniform>             grid:          ProbeGrid;
@group(0) @binding(2)  var<storage, read_write> irradiance_buf: array<f32>;
@group(0) @binding(3)  var<storage, read>       spheres:       array<Sphere>;
@group(0) @binding(4)  var<storage, read>       sphere_bvh:    array<BvhNode>;
@group(0) @binding(5)  var<storage, read>       vertices:      array<Vertex>;
@group(0) @binding(6)  var<storage, read>       triangles:     array<Triangle>;
@group(0) @binding(7)  var<storage, read>       mesh_bvh:      array<BvhNode>;
@group(0) @binding(8)  var<uniform>             materials:     MaterialData;
@group(0) @binding(9)  var<storage, read>       emissive_spheres: array<Sphere>;
@group(0) @binding(10) var                      env_map:       texture_2d<f32>;

// ---------------------------------------------------------------------------
// Shared memory — SH accumulators reduced across the workgroup.
// 9 coefficients × 3 channels (RGB)  ×  64 threads  = 1728 f32 = 6912 bytes.
// Well within Vulkan/Metal's 16 KB workgroup memory limit.
// ---------------------------------------------------------------------------
var<workgroup> sh_r: array<f32, 576>; // 9 * 64
var<workgroup> sh_g: array<f32, 576>;
var<workgroup> sh_b: array<f32, 576>;

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
// Environment map
// ---------------------------------------------------------------------------
fn sample_env_map(dir: vec3<f32>) -> vec3<f32> {
    let phi   = atan2(dir.z, dir.x);
    let theta = asin(clamp(dir.y, -1.0, 1.0));
    let u     = phi / TAU + 0.5;
    let v     = 0.5 - theta / PI;
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
    let phi   = atan2(outn.z, outn.x);
    let theta = asin(clamp(outn.y, -1.0, 1.0));
    hit.uv    = vec2<f32>(phi / TAU + 0.5, 0.5 - theta / PI);
    return hit;
}

// ---------------------------------------------------------------------------
// Ray-triangle Möller–Trumbore
// ---------------------------------------------------------------------------
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
// BVH traversal helpers
// ---------------------------------------------------------------------------
fn sphere_bvh_next_leaf(r: Ray, t_min: f32, t_max: f32,
                        stack: ptr<function, array<u32, 64>>,
                        top: ptr<function, u32>) -> u32 {
    while *top > 0u {
        *top -= 1u;
        let ni   = (*stack)[*top];
        let node = sphere_bvh[ni];
        if !ray_aabb_hit(r, node.aabb_min.xyz, node.aabb_max.xyz, t_min, t_max) { continue; }
        if node.prim_count > 0u { return ni; }
        if *top + 2u <= BVH_STACK_SIZE {
            (*stack)[*top] = node.right_or_offset; *top += 1u;
            (*stack)[*top] = ni + 1u;              *top += 1u;
        }
    }
    return 0xFFFFFFFFu;
}

fn mesh_bvh_next_leaf(r: Ray, t_min: f32, t_max: f32,
                      stack: ptr<function, array<u32, 64>>,
                      top: ptr<function, u32>) -> u32 {
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

fn sphere_bvh_hit(r: Ray, t_min: f32, t_max_in: f32) -> HitRecord {
    var best: HitRecord; best.t = -1.0;
    if arrayLength(&sphere_bvh) == 0u { return best; }
    var t_max = t_max_in;
    var stack: array<u32, 64>; var top = 0u;
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

fn mesh_bvh_hit(r: Ray, t_min: f32, t_max_in: f32) -> HitRecord {
    var best: HitRecord; best.t = -1.0;
    if arrayLength(&mesh_bvh) == 0u { return best; }
    var t_max = t_max_in;
    var stack: array<u32, 64>; var top = 0u;
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
            let backface_cull = (tri.mat_idx >> 31u) != 0u;
            let clean_mat_idx = tri.mat_idx & 0x7FFFFFFFu;
            let res = ray_triangle_uvt(r, va.position.xyz, vb.position.xyz, vc.position.xyz, t_min, t_max);
            if res.x > 0.0 {
                let t = res.x; let u = res.y; let v = res.z; let w = 1.0 - u - v;
                let n = normalize(w * va.normal.xyz + u * vb.normal.xyz + v * vc.normal.xyz);
                let front = dot(r.dir, n) < 0.0;
                if backface_cull && !front { continue; }
                var hit: HitRecord;
                hit.t = t; hit.point = r.origin + t * r.dir;
                hit.normal = select(-n, n, front); hit.front_face = front;
                hit.mat_index = clean_mat_idx;
                hit.uv = w * va.uv.xy + u * vb.uv.xy + v * vc.uv.xy;
                best = hit; t_max = t;
            }
        }
    }
    return best;
}

fn world_hit(r: Ray, t_min: f32, t_max: f32) -> HitRecord {
    let sh = sphere_bvh_hit(r, t_min, t_max);
    let mesh_tmax = select(t_max, sh.t, sh.t > 0.0);
    let mh = mesh_bvh_hit(r, t_min, mesh_tmax);
    if mh.t > 0.0 { return mh; }
    return sh;
}

// ---------------------------------------------------------------------------
// Fibonacci sphere directions (deterministic for ray index i, total N)
// ---------------------------------------------------------------------------
fn fibonacci_dir(i: u32, n: u32) -> vec3<f32> {
    let golden = 1.6180339887498948482;
    let theta = acos(clamp(1.0 - 2.0 * (f32(i) + 0.5) / f32(n), -1.0, 1.0));
    let phi   = TAU * f32(i) / golden;
    return vec3<f32>(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
}

// ---------------------------------------------------------------------------
// Random rotation: rotate `v` around axis `axis` by `angle` radians.
// Rodrigues' rotation formula.
// ---------------------------------------------------------------------------
fn rotate_around_axis(v: vec3<f32>, axis: vec3<f32>, angle: f32) -> vec3<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return v * c + cross(axis, v) * s + axis * dot(axis, v) * (1.0 - c);
}

// ---------------------------------------------------------------------------
// L2 Spherical Harmonics projection — real SH basis (Ramamoorthi & Hanrahan)
// Returns 9 SH coefficients for direction `d` (unit vector).
// Conventions: Y_l^m using the first 3 bands (l=0,1,2).
// ---------------------------------------------------------------------------
fn sh_basis(d: vec3<f32>) -> array<f32, 9> {
    // Band 0
    let y00 =  0.28209479177; // 1/(2√π)
    // Band 1
    let y1n1 =  0.48860251190 * d.y; // √(3/4π) y
    let y10  =  0.48860251190 * d.z; // √(3/4π) z
    let y11  =  0.48860251190 * d.x; // √(3/4π) x
    // Band 2
    let y2n2 =  1.09254843059 * d.x * d.y;            // √(15/4π) xy
    let y2n1 =  1.09254843059 * d.y * d.z;            // √(15/4π) yz
    let y20  =  0.31539156525 * (3.0 * d.z * d.z - 1.0); // √(5/4π)(3z²-1)/2
    let y21  =  1.09254843059 * d.x * d.z;            // √(15/4π) xz
    let y22  =  0.54627421529 * (d.x * d.x - d.y * d.y); // √(15/16π)(x²-y²)

    var b: array<f32, 9>;
    b[0] = y00; b[1] = y1n1; b[2] = y10; b[3] = y11;
    b[4] = y2n2; b[5] = y2n1; b[6] = y20; b[7] = y21; b[8] = y22;
    return b;
}

// ---------------------------------------------------------------------------
// Direct lighting (single emissive sphere NEE) at a hit surface.
// Returns the direct radiance contribution.
// ---------------------------------------------------------------------------
fn direct_light(hit: HitRecord, rng: ptr<function, u32>) -> vec3<f32> {
    let n_emissives = arrayLength(&emissive_spheres);
    if n_emissives == 0u { return vec3<f32>(0.0); }

    // Pick a random emissive sphere.
    *rng = pcg_hash(*rng);
    let em_idx = *rng % n_emissives;
    let em = emissive_spheres[em_idx];
    let em_center = em.center_r.xyz;
    let em_radius = em.center_r.w;

    // Sample a point on the emissive sphere.
    let to_center = em_center - hit.point;
    let dist = length(to_center);
    if dist < em_radius { return vec3<f32>(0.0); } // inside emitter

    // Solid angle sampling.
    let cos_theta_max = sqrt(max(0.0, 1.0 - (em_radius / dist) * (em_radius / dist)));
    let phi   = TAU * rand_f32(rng);
    let cos_t = 1.0 - rand_f32(rng) * (1.0 - cos_theta_max);
    let sin_t = sqrt(max(0.0, 1.0 - cos_t * cos_t));

    // Build local frame around to_center.
    let w = normalize(to_center);
    var u_axis: vec3<f32>;
    if abs(w.x) > 0.9 {
        u_axis = normalize(cross(w, vec3<f32>(0.0, 1.0, 0.0)));
    } else {
        u_axis = normalize(cross(w, vec3<f32>(1.0, 0.0, 0.0)));
    }
    let v_axis = cross(w, u_axis);
    let light_dir = normalize(sin_t * cos(phi) * u_axis + sin_t * sin(phi) * v_axis + cos_t * w);

    // Shadow ray.
    let shadow_ray = Ray(hit.point + hit.normal * T_MIN * 10.0, light_dir);
    let shadow_hit = world_hit(shadow_ray, T_MIN, dist - em_radius);
    if shadow_hit.t > 0.0 { return vec3<f32>(0.0); }

    // Emissive material.
    let em_mat_idx = min(em.mat_and_pad.x, MAX_MAT_IDX);
    let em_mat = materials.mats[em_mat_idx];
    let emission = em_mat.albedo_fuzz.xyz * em_mat.ior_pad.x;

    // Solid angle PDF and Lambert factor.
    let n_dot_l = max(0.0, dot(hit.normal, light_dir));
    let solid_angle = TAU * (1.0 - cos_theta_max);
    let weight = f32(n_emissives); // picking 1 from n with uniform probability

    return emission * n_dot_l * solid_angle * weight / PI;
}

// ---------------------------------------------------------------------------
// Trace a single probe ray and return its radiance.
// Uses 1-bounce path: direct NEE at first diffuse hit + emissive term.
// ---------------------------------------------------------------------------
fn trace_probe_ray(probe_pos: vec3<f32>, dir: vec3<f32>,
                   rng: ptr<function, u32>) -> vec3<f32> {
    let r = Ray(probe_pos, dir);
    let hit = world_hit(r, T_MIN, T_MAX);

    if hit.t < 0.0 {
        return sample_env_map(dir);
    }

    let mat_idx = min(hit.mat_index, MAX_MAT_IDX);
    let mat = materials.mats[mat_idx];

    if mat.type_pad.x == MAT_EMISSIVE {
        return mat.albedo_fuzz.xyz * mat.ior_pad.x;
    }

    // Diffuse/metal/dielectric — compute direct lighting (NEE) at hit point.
    let albedo = mat.albedo_fuzz.xyz;
    let direct = direct_light(hit, rng);
    return albedo * direct / PI;
}

// ---------------------------------------------------------------------------
// Compute entry point
// ---------------------------------------------------------------------------

@compute @workgroup_size(64)
fn cs_probe_update(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id)  lid: vec3<u32>,
    @builtin(workgroup_id)         wid: vec3<u32>,
) {
    let local_idx   = lid.x;
    let probe_local = wid.x;  // which probe within this dispatch

    let probe_idx = probe_params.probe_offset + probe_local;
    if probe_idx >= grid.dims_count.w { return; }

    // Compute probe world-space position.
    let pz  = probe_idx / (grid.dims_count.x * grid.dims_count.y);
    let rem = probe_idx % (grid.dims_count.x * grid.dims_count.y);
    let py  = rem / grid.dims_count.x;
    let px  = rem % grid.dims_count.x;
    let probe_pos = grid.origin_spacing.xyz
                  + vec3<f32>(f32(px), f32(py), f32(pz)) * grid.origin_spacing.w;

    // Per-frame random rotation axis + angle to de-correlate the Fibonacci lattice.
    // Seeded per-probe per-frame so different probes use different rotations.
    var rotation_rng = pcg_hash(pcg_hash(probe_idx) ^ (probe_params.frame_count * 2654435761u));
    let rot_theta = acos(clamp(1.0 - 2.0 * rand_f32(&rotation_rng), -1.0, 1.0));
    let rot_phi   = TAU * rand_f32(&rotation_rng);
    let rot_axis  = vec3<f32>(
        sin(rot_theta) * cos(rot_phi),
        sin(rot_theta) * sin(rot_phi),
        cos(rot_theta)
    );
    let rot_angle = TAU * rand_f32(&rotation_rng);

    // Each thread traces the ray indexed by `local_idx`.
    let n_rays = WORKGROUP_SIZE;
    let ray_dir_base = fibonacci_dir(local_idx, n_rays);
    let ray_dir = rotate_around_axis(ray_dir_base, rot_axis, rot_angle);

    var thread_rng = pcg_hash(pcg_hash(probe_idx * n_rays + local_idx)
                              ^ (probe_params.frame_count * 6364136u));

    let radiance = trace_probe_ray(probe_pos, ray_dir, &thread_rng);

    // Project onto L2 SH basis weighted by 4π/N (Monte Carlo estimator for
    // uniform sphere sampling: pdf = 1/(4π), weight = 4π).
    let basis = sh_basis(ray_dir);
    let mc_weight = (4.0 * PI) / f32(n_rays);

    // Store this thread's contribution in workgroup shared memory.
    for (var coeff = 0u; coeff < SH_COEFF; coeff += 1u) {
        let w = mc_weight * basis[coeff];
        let base = coeff * WORKGROUP_SIZE + local_idx;
        sh_r[base] = radiance.r * w;
        sh_g[base] = radiance.g * w;
        sh_b[base] = radiance.b * w;
    }

    workgroupBarrier();

    // Thread 0 reduces all contributions for this probe.
    if local_idx == 0u {
        let buf_base = probe_idx * SH_F32_PER_PROBE;
        let hysteresis = probe_params.hysteresis;
        let alpha = 1.0 - hysteresis;

        for (var coeff = 0u; coeff < SH_COEFF; coeff += 1u) {
            var sum_r = 0.0; var sum_g = 0.0; var sum_b = 0.0;
            let row_base = coeff * WORKGROUP_SIZE;
            for (var t = 0u; t < WORKGROUP_SIZE; t += 1u) {
                sum_r += sh_r[row_base + t];
                sum_g += sh_g[row_base + t];
                sum_b += sh_b[row_base + t];
            }

            // EMA blend: SH_new = lerp(SH_old, sample, alpha)
            let base_r = buf_base + coeff * 3u;
            let old_r = irradiance_buf[base_r];
            let old_g = irradiance_buf[base_r + 1u];
            let old_b = irradiance_buf[base_r + 2u];

            // Asymmetric hysteresis: respond faster to decreasing luminance.
            let lum_new = 0.2126 * sum_r + 0.7152 * sum_g + 0.0722 * sum_b;
            let lum_old = 0.2126 * old_r + 0.7152 * old_g + 0.0722 * old_b;
            let eff_alpha = select(alpha, min(alpha * 4.0, 0.25), lum_new < lum_old * 0.5);

            irradiance_buf[base_r]      = mix(old_r, sum_r, eff_alpha);
            irradiance_buf[base_r + 1u] = mix(old_g, sum_g, eff_alpha);
            irradiance_buf[base_r + 2u] = mix(old_b, sum_b, eff_alpha);
        }
    }
}
