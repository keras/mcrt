// path_trace.wgsl — Phase 4: ray-sphere intersection & normal visualisation
//
// Each invocation handles one pixel.  A pinhole camera ray is generated and
// traced against the scene's sphere list.  Hits are coloured by surface
// normal; misses fall back to the sky gradient.

// ---- shared primitives ----------------------------------------------------

/// A ray in world space.  `dir` must be unit length.
struct Ray {
    origin: vec3<f32>,
    dir:    vec3<f32>,
}

/// Record of the closest intersection found so far.
/// `t < 0` is used as the "no hit" sentinel.
struct HitRecord {
    t:          f32,
    point:      vec3<f32>,
    normal:     vec3<f32>,  // always points against the incident ray
    front_face: bool,
}

// ---- camera uniform -------------------------------------------------------
// All vec4 fields; the .w component is unused padding to satisfy alignment.

struct Camera {
    origin:     vec4<f32>,  // .xyz = eye position in world space; .w = frame_count (Phase 5)
    lower_left: vec4<f32>,  // .xyz = lower-left corner of the virtual screen
    horizontal: vec4<f32>,  // .xyz = full horizontal extent of the screen
    vertical:   vec4<f32>,  // .xyz = full vertical extent of the screen
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

/// Output image written by this compute shader; sampled by the display pass.
@group(0) @binding(0) var output: texture_storage_2d<rgba8unorm, write>;

/// Camera parameters uploaded from the CPU every frame (orbit moves the eye).
@group(0) @binding(1) var<uniform> camera: Camera;

/// Scene: sphere list uploaded from the CPU.
@group(0) @binding(2) var<uniform> scene: SceneData;

// ---- sky gradient ---------------------------------------------------------

/// Classic RTIOW sky: white at the horizon, sky-blue at the zenith.
/// Expects `unit_dir` to already be normalised.
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

    hit.t     = t;
    hit.point = r.origin + t * r.dir;
    let outward_normal = (hit.point - center) / radius;
    hit.front_face = dot(r.dir, outward_normal) < 0.0;
    // Normal always faces the incident ray (needed for refraction in Phase 7).
    hit.normal = select(-outward_normal, outward_normal, hit.front_face);
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

// ---- compute entry point --------------------------------------------------

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims  = textureDimensions(output);
    let coord = gid.xy;

    // Guard: discard threads outside the texture (non-multiple-of-8 res).
    if coord.x >= dims.x || coord.y >= dims.y { return; }

    // Normalised UV with half-pixel centre offset.
    //   u ∈ [0, 1] : left → right
    //   v ∈ [0, 1] : bottom → top  (flip Y from texture convention to Y-up world)
    let u = (f32(coord.x) + 0.5) / f32(dims.x);
    let v = 1.0 - (f32(coord.y) + 0.5) / f32(dims.y);

    // Build a unit-direction ray from the pinhole camera.
    let raw_dir =
          camera.lower_left.xyz
        + u * camera.horizontal.xyz
        + v * camera.vertical.xyz
        - camera.origin.xyz;
    let ray = Ray(camera.origin.xyz, normalize(raw_dir));

    // Trace the scene; use a small t_min to avoid self-intersection.
    let hit = scene_hit(ray, 1e-4, 1e9);

    var color: vec3<f32>;
    if hit.t > 0.0 {
        // Normal-map visualisation: remap [-1, 1] → [0, 1] per channel.
        color = 0.5 * (hit.normal + vec3<f32>(1.0));
    } else {
        color = sky_color(ray.dir);
    }

    textureStore(output, coord, vec4<f32>(color, 1.0));
}
