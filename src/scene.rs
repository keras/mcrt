// scene.rs — Phase 4+: GPU sphere scene data types and scene builders
//
// All types here are bytemuck-compatible (Pod + Zeroable) so they can be
// uploaded directly to wgpu storage buffers without manual packing.
// No wgpu types are imported, keeping this module independently testable.
//
// Phase 9 change: the fixed-size GpuSceneData uniform was replaced by a
// dynamic Vec<GpuSphere> that is uploaded via a storage buffer.  This removes
// the MAX_SPHERES cap and enables BVH-accelerated scenes with arbitrary counts.
//
// Phase 12 change: `load_scene_from_yaml` allows scenes to be described by a
// plain YAML file (see assets/scene.yaml) instead of hard-coded Rust builders.

// ---------------------------------------------------------------------------
// GPU types
// ---------------------------------------------------------------------------

/// GPU-side sphere descriptor.
///
/// Packing centre + radius into one `[f32; 4]` ensures identical alignment
/// in Rust (`repr(C)`) and WGSL (`vec4<f32>`).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSphere {
    /// xyz = world-space centre, w = radius.
    pub center_r: [f32; 4],
    /// x = material index, yzw = unused padding.
    pub mat_and_pad: [u32; 4],
}

// ---------------------------------------------------------------------------
// Scene builders
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// YAML scene loader (Phase 12+)
// ---------------------------------------------------------------------------

/// Material shading models supported in the YAML scene format.
#[derive(serde::Deserialize, Clone, Debug)]
#[serde(rename_all = "lowercase")]
enum MaterialKind {
    Lambertian,
    Metal,
    Dielectric,
    /// Phase 13: emissive surface / area light.
    /// `albedo` is the emission colour; `emission_strength` is the luminance
    /// multiplier (defaults to 1.0).  Emissive materials do not scatter rays.
    Emissive,
}

/// Camera placement and lens settings parsed from the YAML `camera:` block.
///
/// All fields except `look_from` and `look_at` are optional:
/// - `vfov`       → 60.0°
/// - `aperture`   → 0.0 (pinhole)
/// - `focus_dist` → distance between `look_from` and `look_at`
#[derive(serde::Deserialize, Clone, Debug)]
struct CameraDesc {
    look_from: [f32; 3],
    look_at: [f32; 3],
    #[serde(default)]
    vfov: Option<f32>,
    #[serde(default)]
    aperture: Option<f32>,
    #[serde(default)]
    focus_dist: Option<f32>,
}

/// Camera settings returned by [`load_scene_from_yaml`] when the YAML
/// contains a `camera:` block.  `gpu.rs` converts this into a `CameraState`.
#[derive(Clone, Debug)]
pub struct SceneCameraSettings {
    /// Eye position in world space.
    pub look_from: [f32; 3],
    /// Point the camera looks toward.
    pub look_at: [f32; 3],
    /// Vertical field of view in degrees.
    pub vfov: f32,
    /// Lens aperture radius (0 = pinhole / no depth-of-field).
    pub aperture: f32,
    /// Distance to the focal plane.
    pub focus_dist: f32,
}

/// A material definition, used both in the top-level `materials:` palette and
/// when a material is inlined directly inside an object definition.
///
/// All fields except `type` are optional and fall back to sensible defaults:
/// - `albedo` → `[1.0, 1.0, 1.0]`
/// - `fuzz` → `0.0`  (metal roughness)
/// - `ior` → `1.5`   (dielectric index of refraction)
/// - `emission_strength` → `1.0`  (emissive luminance multiplier)
/// - `texture_layer` → `0` (no albedo texture)
#[derive(serde::Deserialize, Clone, Debug)]
struct MaterialDesc {
    /// Shading model: `lambertian`, `metal`, `dielectric`, or `emissive`.
    #[serde(rename = "type")]
    kind: MaterialKind,
    /// RGB albedo colour.  For `emissive` materials this is the emission colour.
    /// Defaults to `[1.0, 1.0, 1.0]`.
    #[serde(default)]
    albedo: Option<[f32; 3]>,
    /// Metal roughness ∈ [0, 1].  Defaults to `0.0` (mirror).
    #[serde(default)]
    fuzz: Option<f32>,
    /// Index of refraction for dielectrics.  Defaults to `1.5`.
    #[serde(default)]
    ior: Option<f32>,
    /// Phase 13: emission luminance multiplier for `emissive` materials.
    /// Higher values produce brighter lights.  Defaults to `1.0`.
    #[serde(default)]
    emission_strength: Option<f32>,
    /// Albedo texture array layer (Phase 11).  `0` = no texture / white.
    #[serde(default)]
    texture_layer: u32,
}

/// How a material is specified on an object — either a name referencing the
/// top-level `materials:` palette, or a full inline definition.
///
/// YAML examples:
/// ```yaml
/// material: glass            # named reference
/// material:                  # inline definition
///   type: metal
///   albedo: [0.8, 0.6, 0.2]
///   fuzz: 0.05
/// ```
#[derive(serde::Deserialize, Clone, Debug)]
#[serde(untagged)]
enum MaterialRef {
    /// Name of a material declared in the top-level `materials:` list.
    Named(String),
    /// Full material definition inlined inside the object.
    Inline(MaterialDesc),
}

/// A single object in the scene.  The `type` YAML field selects the variant.
#[derive(serde::Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum ObjectDesc {
    /// An analytic sphere.
    Sphere {
        /// World-space centre `[x, y, z]`.
        center: [f32; 3],
        /// Radius — must be greater than zero.
        radius: f32,
        /// Material reference: a name string or an inline definition.
        material: MaterialRef,
    },
    /// A triangle mesh loaded from an OBJ file.
    ///
    /// ```yaml
    /// - type: mesh
    ///   path: models/bunny.obj
    ///   material: gold
    ///   scale: 8.0
    ///   translate: [0.0, -0.267, 0.0]
    ///   rotate: [0.0, 45.0, 0.0]   # optional XYZ Euler degrees
    /// ```
    Mesh {
        /// Path to a Wavefront OBJ file (relative to the working directory).
        path: String,
        /// Material for all triangles in this mesh.
        material: MaterialRef,
        /// Uniform scale applied to every vertex position before upload.
        /// Defaults to 1.0 (no scaling).
        #[serde(default)]
        scale: Option<f32>,
        /// Translation `[x, y, z]` added to every vertex position after
        /// scaling.  Defaults to `[0, 0, 0]` (no translation).
        #[serde(default)]
        translate: Option<[f32; 3]>,
        /// Optional Euler rotation angles in **degrees** applied in XYZ order
        /// (first X, then Y, then Z; extrinsic ZYX equivalent).
        /// Rotation is applied around the mesh origin after scaling but before
        /// translation.  Defaults to `[0, 0, 0]` (no rotation).
        #[serde(default)]
        rotate: Option<[f32; 3]>,
    },
    /// A finite rectangular plane tessellated into two triangles and added to
    /// the mesh buffer.  The normal determines which side is "front" (outward).
    ///
    /// ```yaml
    /// - type: plane
    ///   center: [0.0, 0.0, -5.5]
    ///   normal: [0.0, 0.0,  1.0]
    ///   half_extents: [2.75, 2.75]
    ///   material: white
    ///   rotate: [0.0, 15.0, 0.0]   # optional XYZ Euler degrees
    /// ```
    Plane {
        /// World-space centre of the rectangle.
        center: [f32; 3],
        /// Outward-facing unit normal.  Two tangent axes are derived
        /// automatically via Gram–Schmidt so no extra orientation is needed.
        normal: [f32; 3],
        /// Half-extents `[half_width, half_height]` of the rectangle.
        /// The total patch is `2*half_width × 2*half_height`.
        half_extents: [f32; 2],
        /// Material reference: a name string or an inline definition.
        material: MaterialRef,
        /// Optional Euler rotation angles in **degrees** applied in XYZ order
        /// (first X, then Y, then Z; extrinsic ZYX equivalent).
        /// Rotates the plane normal (and the derived tangent frame) around the
        /// plane centre.  Defaults to `[0, 0, 0]` (no rotation).
        #[serde(default)]
        rotate: Option<[f32; 3]>,
        /// When `true` rays hitting the back face (i.e. `dot(ray_dir, normal) > 0`)
        /// produce no intersection.  Useful for wall planes that will never be
        /// seen from behind, cutting the number of triangles the BVH has to test.
        /// Defaults to `false` (double-sided).
        #[serde(default)]
        backface_culling: bool,
    },
    /// An axis-aligned box tessellated into 12 triangles (6 faces × 2 each).
    ///
    /// ```yaml
    /// - type: box
    ///   center: [-1.5, 0.7, -4.0]
    ///   half_extents: [0.7, 0.7, 0.7]
    ///   material: white
    /// ```
    #[serde(rename = "box")]
    BoxPrim {
        /// World-space centre of the box.
        center: [f32; 3],
        /// Half-extents along each axis `[hx, hy, hz]`; full dimensions are
        /// `2*hx × 2*hy × 2*hz`.
        half_extents: [f32; 3],
        /// Optional Euler rotation angles in **degrees** applied in XYZ order
        /// (first X, then Y, then Z; extrinsic ZYX equivalent).
        /// Rotation is applied around the box centre before placement.
        /// Defaults to `[0, 0, 0]` (identity — axis-aligned).
        #[serde(default)]
        rotate: Option<[f32; 3]>,
        /// Material applied to all six faces.
        material: MaterialRef,
    },
    // Future variants:
    // Torus { center, major_r, minor_r, material }
}

/// Top-level structure of a YAML scene file.
#[derive(serde::Deserialize)]
struct SceneFile {
    /// Optional camera settings.  When present, the renderer initialises its
    /// camera from these values instead of the built-in defaults.
    #[serde(default)]
    camera: Option<CameraDesc>,
    /// Named material palette.  Keys are material names; values are their
    /// definitions.  Insertion order is preserved by `IndexMap`, so GPU
    /// indices are deterministic.  Duplicate keys are a YAML parse error,
    /// which guarantees that each name maps to exactly one definition.
    #[serde(default)]
    materials: indexmap::IndexMap<String, MaterialDesc>,
    /// Ordered list of scene objects.
    objects: Vec<ObjectDesc>,
    /// Optional path (relative to the working directory) to an HDR equirectangular
    /// environment map.  When absent the renderer falls back to `textures/env.hdr`,
    /// then to the procedural gradient sky.
    #[serde(default)]
    env_map: Option<String>,
}

/// The result of loading a YAML scene file — ready for BVH construction and
/// GPU upload without any further processing.
pub struct LoadedScene {
    /// Sphere primitives for BVH construction and the sphere storage buffer.
    pub spheres: Vec<GpuSphere>,
    /// Material table for the GPU material uniform buffer.
    pub materials: crate::material::GpuMaterialData,
    /// Merged vertex array for all `mesh` objects; pass to
    /// [`crate::mesh::build_mesh_bvh`] together with [`Self::mesh_triangles`].
    /// Always non-empty (a stub vertex is appended when no mesh objects
    /// exist so wgpu buffer binding validation always passes).
    pub mesh_vertices: Vec<crate::mesh::GpuVertex>,
    /// Merged, offset-adjusted triangle index array for all `mesh` objects.
    /// Always non-empty (see [`Self::mesh_vertices`]).
    pub mesh_triangles: Vec<crate::mesh::GpuTriangle>,
    /// Camera placement & lens settings.  `Some` when the YAML includes a
    /// `camera:` block; `None` when the built-in defaults should be used.
    pub camera: Option<SceneCameraSettings>,
    /// Phase 13: flat list of emissive sphere primitives for direct light
    /// sampling (Next-Event Estimation).
    ///
    /// Contains only spheres whose material type is `emissive`.  The count is
    /// also stored in `materials.n_emissive` for the GPU shader.
    /// May be empty when the scene has no emissive spheres; `gpu.rs` appends a
    /// stub entry to satisfy wgpu's minimum-buffer-binding-size requirement.
    ///
    /// Note: emissive *mesh* triangles are not tracked here (Phase 13 scope).
    pub emissive_spheres: Vec<GpuSphere>,
    /// Optional path to an HDR equirectangular environment map, as declared in
    /// the scene YAML via `env_map: <path>`.  `None` means use the default
    /// fallback chain (`textures/env.hdr` → procedural gradient).
    pub env_map_path: Option<String>,
    /// Human-readable material names in GPU-index order (Phase 15).
    /// Palette materials use their YAML key; inline materials use `"inline-{i}"`.
    pub material_names: Vec<String>,
}

// Convert a YAML MaterialDesc into a GPU-ready GpuMaterial.
fn material_desc_to_gpu(desc: &MaterialDesc) -> crate::material::GpuMaterial {
    use crate::material::{MAT_DIELECTRIC, MAT_EMISSIVE, MAT_LAMBERTIAN, MAT_METAL};
    let albedo = desc.albedo.unwrap_or([1.0, 1.0, 1.0]);
    match desc.kind {
        MaterialKind::Lambertian => crate::material::GpuMaterial {
            type_pad: [MAT_LAMBERTIAN, desc.texture_layer, 0, 0],
            albedo_fuzz: [albedo[0], albedo[1], albedo[2], 0.0],
            ior_pad: [1.0, 0.0, 0.0, 0.0],
        },
        MaterialKind::Metal => crate::material::GpuMaterial {
            type_pad: [MAT_METAL, desc.texture_layer, 0, 0],
            albedo_fuzz: [albedo[0], albedo[1], albedo[2], desc.fuzz.unwrap_or(0.0)],
            ior_pad: [1.0, 0.0, 0.0, 0.0],
        },
        MaterialKind::Dielectric => crate::material::GpuMaterial {
            type_pad: [MAT_DIELECTRIC, 0, 0, 0],
            albedo_fuzz: [1.0, 1.0, 1.0, 0.0],
            ior_pad: [desc.ior.unwrap_or(1.5), 0.0, 0.0, 0.0],
        },
        // Emissive: albedo_fuzz.xyz = emission colour; ior_pad.x = strength.
        // These materials terminate paths and serve as area lights for NEE.
        MaterialKind::Emissive => crate::material::GpuMaterial {
            type_pad: [MAT_EMISSIVE, 0, 0, 0],
            albedo_fuzz: [albedo[0], albedo[1], albedo[2], 0.0],
            ior_pad: [desc.emission_strength.unwrap_or(1.0), 0.0, 0.0, 0.0],
        },
    }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Rotation helpers for BoxPrim
// ---------------------------------------------------------------------------

/// Convert degrees → radians.
#[inline]
fn to_rad(deg: f32) -> f32 {
    deg * std::f32::consts::PI / 180.0
}

/// Build a **row-major** 3×3 rotation matrix from Euler angles (degrees).
/// Convention: R = Rz(z) · Ry(y) · Rx(x)  — intrinsic XYZ / extrinsic ZYX.
/// The resulting matrix `m` is applied as `m * v` via `rot3(m, v)`.
fn euler_to_mat3(rx: f32, ry: f32, rz: f32) -> [[f32; 3]; 3] {
    let (sx, cx) = to_rad(rx).sin_cos();
    let (sy, cy) = to_rad(ry).sin_cos();
    let (sz, cz) = to_rad(rz).sin_cos();
    [
        [cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz],
        [cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz],
        [-sy, sx * cy, cx * cy],
    ]
}

/// Multiply a row-major 3×3 matrix by a column vector.
#[inline]
fn rot3(m: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

// Float-3 helpers for plane / box tessellation
// ---------------------------------------------------------------------------

#[inline]
fn plane_cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn plane_normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 1e-9 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 1.0, 0.0] // fallback; should not occur for valid normals
    }
}

/// Append one rectangular quad face (4 vertices + 2 CCW triangles) to the
/// shared mesh buffers.  The face is centred at `center` with outward
/// `normal`.  Two tangent axes are derived automatically via Gram–Schmidt.
/// `hw` and `hh` are the half-extents along those tangent axes.
fn push_quad_face(
    center: [f32; 3],
    normal: [f32; 3],
    hw: f32,
    hh: f32,
    mat_idx: u32,
    verts: &mut Vec<crate::mesh::GpuVertex>,
    tris: &mut Vec<crate::mesh::GpuTriangle>,
) {
    let [nx, ny, nz] = normal;
    // Gram–Schmidt: pick the axis-aligned helper with the smallest component
    // to avoid a near-parallel cross product.
    let abs = [nx.abs(), ny.abs(), nz.abs()];
    let helper: [f32; 3] = if abs[0] <= abs[1] && abs[0] <= abs[2] {
        [1.0, 0.0, 0.0]
    } else if abs[1] <= abs[2] {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    let t1 = plane_normalize3(plane_cross3(helper, [nx, ny, nz]));
    let t2 = plane_cross3([nx, ny, nz], t1);

    let [cx, cy, cz] = center;
    let corner = |s1: f32, s2: f32| -> [f32; 4] {
        [
            cx + t1[0] * s1 * hw + t2[0] * s2 * hh,
            cy + t1[1] * s1 * hw + t2[1] * s2 * hh,
            cz + t1[2] * s1 * hw + t2[2] * s2 * hh,
            1.0,
        ]
    };
    let p0 = corner(-1.0, -1.0);
    let p1 = corner(1.0, -1.0);
    let p2 = corner(1.0, 1.0);
    let p3 = corner(-1.0, 1.0);

    let n4 = [nx, ny, nz, 0.0];
    let base = verts.len() as u32;
    verts.push(crate::mesh::GpuVertex {
        position: p0,
        normal: n4,
        uv: [0.0, 0.0, 0.0, 0.0],
    });
    verts.push(crate::mesh::GpuVertex {
        position: p1,
        normal: n4,
        uv: [1.0, 0.0, 0.0, 0.0],
    });
    verts.push(crate::mesh::GpuVertex {
        position: p2,
        normal: n4,
        uv: [1.0, 1.0, 0.0, 0.0],
    });
    verts.push(crate::mesh::GpuVertex {
        position: p3,
        normal: n4,
        uv: [0.0, 1.0, 0.0, 0.0],
    });
    tris.push(crate::mesh::GpuTriangle {
        v: [base, base + 1, base + 2],
        mat_idx,
    });
    tris.push(crate::mesh::GpuTriangle {
        v: [base, base + 2, base + 3],
        mat_idx,
    });
}

/// Parse a scene from a YAML *string* and return data ready for GPU upload.
///
/// `source` is only used in panic messages (e.g. the original file path, or
/// `"<inline>"` for tests).  This is the real implementation; the public
/// [`load_scene_from_yaml`] is a thin wrapper that reads the file first.
pub(crate) fn load_scene_from_str(text: &str, source: &str) -> LoadedScene {
    use crate::material::{GpuMaterialData, MAX_MATERIALS};
    use bytemuck::Zeroable;
    use std::collections::HashMap;

    let scene: SceneFile = serde_yaml::from_str(text)
        .unwrap_or_else(|e| panic!("failed to parse scene file '{}': {}", source, e));

    // Build the material palette from the top-level `materials:` dict.
    // IndexMap preserves insertion order, so GPU indices are deterministic.
    // Duplicate keys are rejected at parse time by serde_yaml.
    let mut name_to_idx: HashMap<String, u32> = HashMap::new();
    let mut gpu_mats = Vec::<crate::material::GpuMaterial>::new();
    for (name, desc) in &scene.materials {
        let idx = gpu_mats.len() as u32;
        gpu_mats.push(material_desc_to_gpu(desc));
        name_to_idx.insert(name.clone(), idx);
    }

    // Helper: resolve a MaterialRef → GPU material index.
    // Inline materials are always appended as new entries (no deduplication).
    macro_rules! resolve_mat {
        ($mat:expr) => {{
            match $mat {
                MaterialRef::Named(name) => *name_to_idx
                    .get(name.as_str())
                    .unwrap_or_else(|| panic!("object references unknown material '{}'", name)),
                MaterialRef::Inline(desc) => {
                    let idx = gpu_mats.len() as u32;
                    gpu_mats.push(material_desc_to_gpu(desc));
                    idx
                }
            }
        }};
    }

    // Process objects — spheres and mesh files — accumulating GPU data.
    let mut spheres: Vec<GpuSphere> = Vec::new();
    let mut mesh_vertices: Vec<crate::mesh::GpuVertex> = Vec::new();
    let mut mesh_triangles: Vec<crate::mesh::GpuTriangle> = Vec::new();

    for obj in &scene.objects {
        match obj {
            ObjectDesc::Sphere {
                center,
                radius,
                material,
            } => {
                let mat_idx = resolve_mat!(material);
                spheres.push(GpuSphere {
                    center_r: [center[0], center[1], center[2], *radius],
                    mat_and_pad: [mat_idx, 0, 0, 0],
                });
            }

            ObjectDesc::Plane {
                center,
                normal,
                half_extents,
                material,
                rotate,
                backface_culling,
            } => {
                let base_idx = resolve_mat!(material);
                // Pack the backface-cull flag into bit 31 of mat_idx.
                // Actual material indices are always ≤ MAX_MATERIALS (64), so
                // bit 31 is permanently free for use as a flag.  The shader
                // masks the flag off before indexing the material table.
                const BACKFACE_CULL_FLAG: u32 = 1u32 << 31;
                let mat_idx = if *backface_culling {
                    base_idx | BACKFACE_CULL_FLAG
                } else {
                    base_idx
                };
                let [hw, hh] = *half_extents;
                // Apply optional rotation to the plane normal.  The tangent
                // axes are re-derived automatically inside push_quad_face via
                // Gram–Schmidt, so rotating the normal is sufficient to orient
                // the plane in any direction.
                let eff_normal = if let Some([rx, ry, rz]) = *rotate {
                    rot3(&euler_to_mat3(rx, ry, rz), *normal)
                } else {
                    *normal
                };
                push_quad_face(
                    *center,
                    eff_normal,
                    hw,
                    hh,
                    mat_idx,
                    &mut mesh_vertices,
                    &mut mesh_triangles,
                );
            }

            ObjectDesc::BoxPrim {
                center,
                half_extents,
                rotate,
                material,
            } => {
                let mat_idx = resolve_mat!(material);
                let [cx, cy, cz] = *center;
                let [hx, hy, hz] = *half_extents;

                // Build optional rotation matrix from Euler angles (degrees).
                // When rotate is None or [0,0,0] the identity is used and the
                // box is axis-aligned.
                let rot = rotate
                    .map(|[rx, ry, rz]| euler_to_mat3(rx, ry, rz))
                    .unwrap_or([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]);

                // Six faces defined as (outward local normal, 4 CCW corner offsets).
                // Corner offsets are in the box's local space (centred at origin);
                // they are rotated and then shifted by `center`.
                let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
                    // +X
                    (
                        [1., 0., 0.],
                        [[hx, -hy, -hz], [hx, -hy, hz], [hx, hy, hz], [hx, hy, -hz]],
                    ),
                    // -X
                    (
                        [-1., 0., 0.],
                        [
                            [-hx, -hy, hz],
                            [-hx, -hy, -hz],
                            [-hx, hy, -hz],
                            [-hx, hy, hz],
                        ],
                    ),
                    // +Y
                    (
                        [0., 1., 0.],
                        [[-hx, hy, -hz], [hx, hy, -hz], [hx, hy, hz], [-hx, hy, hz]],
                    ),
                    // -Y
                    (
                        [0., -1., 0.],
                        [
                            [hx, -hy, -hz],
                            [-hx, -hy, -hz],
                            [-hx, -hy, hz],
                            [hx, -hy, hz],
                        ],
                    ),
                    // +Z
                    (
                        [0., 0., 1.],
                        [[hx, -hy, hz], [-hx, -hy, hz], [-hx, hy, hz], [hx, hy, hz]],
                    ),
                    // -Z
                    (
                        [0., 0., -1.],
                        [
                            [-hx, -hy, -hz],
                            [hx, -hy, -hz],
                            [hx, hy, -hz],
                            [-hx, hy, -hz],
                        ],
                    ),
                ];

                let uvs = [[0., 0.], [1., 0.], [1., 1.], [0., 1.]];
                for (local_normal, local_corners) in &faces {
                    // Rotate normal and corner offsets into world space.
                    let rn = rot3(&rot, *local_normal);
                    let n4 = [rn[0], rn[1], rn[2], 0.0];
                    let base = mesh_vertices.len() as u32;
                    for (i, c) in local_corners.iter().enumerate() {
                        let rc = rot3(&rot, *c);
                        mesh_vertices.push(crate::mesh::GpuVertex {
                            position: [cx + rc[0], cy + rc[1], cz + rc[2], 1.0],
                            normal: n4,
                            uv: [uvs[i][0], uvs[i][1], 0.0, 0.0],
                        });
                    }
                    mesh_triangles.push(crate::mesh::GpuTriangle {
                        v: [base, base + 1, base + 2],
                        mat_idx,
                    });
                    mesh_triangles.push(crate::mesh::GpuTriangle {
                        v: [base, base + 2, base + 3],
                        mat_idx,
                    });
                }
            }

            ObjectDesc::Mesh {
                path: obj_path,
                material,
                scale,
                translate,
                rotate,
            } => {
                let mat_idx = resolve_mat!(material);
                let (mut verts, tris) = crate::mesh::load_obj(obj_path, mat_idx)
                    .unwrap_or_else(|e| panic!("failed to load mesh '{}': {}", obj_path, e));

                // Apply optional uniform scale + rotation + translation to vertex positions.
                let s = scale.unwrap_or(1.0);
                // Negative scale would flip handedness without inverting stored normals,
                // producing inside-out lighting.  Zero scale collapses geometry into a
                // degenerate point.  Both are almost certainly unintentional.
                assert!(
                    s > 0.0,
                    "mesh '{}' scale must be positive (got {s})",
                    obj_path
                );
                let t = translate.unwrap_or([0.0f32; 3]);
                // Build optional rotation matrix from Euler angles (degrees).
                // Applied after scaling but before translation — i.e. the mesh
                // rotates around its (scaled) local origin.
                let rot = rotate.map(|[rx, ry, rz]| euler_to_mat3(rx, ry, rz));
                if s != 1.0 || t != [0.0f32; 3] || rot.is_some() {
                    for v in &mut verts {
                        // Scale → rotate → translate position.
                        let mut pos = [v.position[0] * s, v.position[1] * s, v.position[2] * s];
                        if let Some(m) = &rot {
                            pos = rot3(m, pos);
                        }
                        v.position[0] = pos[0] + t[0];
                        v.position[1] = pos[1] + t[1];
                        v.position[2] = pos[2] + t[2];
                        // Rotate normals.  Uniform scale preserves direction;
                        // translation does not affect direction vectors.
                        if let Some(m) = &rot {
                            let rn = rot3(m, [v.normal[0], v.normal[1], v.normal[2]]);
                            v.normal[0] = rn[0];
                            v.normal[1] = rn[1];
                            v.normal[2] = rn[2];
                        }
                    }
                }

                // Merge into the shared vertex/triangle arrays, offsetting
                // triangle indices so they index into the combined vertex buffer.
                let vert_offset = mesh_vertices.len() as u32;
                mesh_vertices.extend_from_slice(&verts);
                for tri in &tris {
                    mesh_triangles.push(crate::mesh::GpuTriangle {
                        v: [
                            tri.v[0] + vert_offset,
                            tri.v[1] + vert_offset,
                            tri.v[2] + vert_offset,
                        ],
                        mat_idx: tri.mat_idx,
                    });
                }
            }
        }
    }

    // Ensure mesh buffers are never empty: a degenerate point at a distant
    // location satisfies wgpu's minimum buffer-binding-size requirement while
    // contributing zero visible geometry (the BVH AABB will never be hit by
    // any camera ray originating from a normal scene position).
    if mesh_vertices.is_empty() {
        mesh_vertices.push(crate::mesh::GpuVertex {
            position: [1.0e9, 1.0e9, 1.0e9, 1.0],
            normal: [0.0, 1.0, 0.0, 0.0],
            uv: [0.0, 0.0, 0.0, 0.0],
        });
        // A degenerate triangle (single repeated vertex) that Möller–Trumbore
        // will always reject (det ≈ 0).
        mesh_triangles.push(crate::mesh::GpuTriangle {
            v: [0, 0, 0],
            mat_idx: 0,
        });
    }

    // Pack into GpuMaterialData.
    assert!(
        gpu_mats.len() <= MAX_MATERIALS,
        "scene uses {} materials but MAX_MATERIALS = {}",
        gpu_mats.len(),
        MAX_MATERIALS
    );
    let mut mat_data = GpuMaterialData::zeroed();
    mat_data.mat_count = gpu_mats.len() as u32;
    mat_data.materials[..gpu_mats.len()].copy_from_slice(&gpu_mats);

    // Phase 13: build the flat list of emissive sphere primitives.
    // The GPU shader uses this list for Next-Event Estimation (NEE).
    // Only sphere objects are tracked; emissive mesh triangles are out of scope.
    let emissive_spheres: Vec<GpuSphere> = spheres
        .iter()
        .filter(|s| {
            let idx = s.mat_and_pad[0] as usize;
            idx < gpu_mats.len() && gpu_mats[idx].type_pad[0] == crate::material::MAT_EMISSIVE
        })
        .copied()
        .collect();
    mat_data.n_emissive = emissive_spheres.len() as u32;

    // Convert the optional camera block.
    let camera = scene.camera.as_ref().map(|c| {
        let look_from = c.look_from;
        let look_at = c.look_at;
        let dist = {
            let d = [
                look_from[0] - look_at[0],
                look_from[1] - look_at[1],
                look_from[2] - look_at[2],
            ];
            (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
        };
        SceneCameraSettings {
            look_from,
            look_at,
            vfov: c.vfov.unwrap_or(crate::camera::DEFAULT_VFOV),
            aperture: c.aperture.unwrap_or(0.0),
            focus_dist: c.focus_dist.unwrap_or(dist),
        }
    });

    LoadedScene {
        spheres,
        materials: mat_data,
        mesh_vertices,
        mesh_triangles,
        camera,
        emissive_spheres,
        env_map_path: scene.env_map,
        material_names: {
            // Palette entries are named; any inline materials added later get
            // an auto name.  name_to_idx has all palette indices; entries
            // beyond that range are inline.
            let palette_count = name_to_idx.len();
            let total = gpu_mats.len();
            // Build reverse map: index → name for palette materials.
            let mut names: Vec<String> = (0..total).map(|i| format!("inline-{i}")).collect();
            for (name, &idx) in &name_to_idx {
                names[idx as usize] = name.clone();
            }
            let _ = palette_count; // suppress warning
            names
        },
    }
}

/// Load a scene from a YAML file and return spheres and materials ready for
/// GPU upload.
///
/// The YAML must contain a top-level `objects:` sequence where every entry has
/// a `type` field (`sphere`, …) and a `material` field.  A `materials:` map
/// at the top of the file declares named materials that objects can reference
/// by name; objects may also define their material inline.  See
/// `assets/scene.yaml` for a complete example.
///
/// # Panics
/// Panics if the file cannot be read, the YAML is malformed, an object
/// references an unknown material name, or more than `MAX_MATERIALS` unique
/// materials are used.
pub fn load_scene_from_yaml(path: &str) -> LoadedScene {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("failed to read scene file '{}': {}", path, e));
    load_scene_from_str(&text, path)
}

/// Construct the original four-sphere demo scene.
///
/// Useful for unit tests and quick sanity checks.  For the Phase 9 BVH
/// demonstration use [`build_large_scene`]; for a data-driven scene prefer
/// [`load_scene_from_yaml`] with `assets/scene.yaml`.
///
/// Material indices reference [`crate::material::build_materials`]:
/// - 0 — Lambertian grey  (ground)
/// - 1 — Lambertian red   (centre)
/// - 2 — Metal gold, fuzz 0.1  (left)
/// - 3 — Dielectric glass IOR 1.5 (right)
#[allow(dead_code)] // retained as reference scene and for unit tests
pub fn build_scene() -> Vec<GpuSphere> {
    vec![
        // Ground plane represented as a large sphere.
        GpuSphere {
            center_r: [0.0, -100.5, 0.0, 100.0],
            mat_and_pad: [0, 0, 0, 0],
        },
        // Centre sphere.
        GpuSphere {
            center_r: [0.0, 0.0, 0.0, 0.5],
            mat_and_pad: [1, 0, 0, 0],
        },
        // Left sphere.
        GpuSphere {
            center_r: [-1.2, 0.0, 0.0, 0.5],
            mat_and_pad: [2, 0, 0, 0],
        },
        // Right sphere.
        GpuSphere {
            center_r: [1.2, 0.0, 0.0, 0.5],
            mat_and_pad: [3, 0, 0, 0],
        },
    ]
}

/// Construct a larger scene designed to exercise the Phase 9 BVH.
///
/// The scene contains:
/// - 1  ground sphere (radius 100, material 0 — grey Lambertian)
/// - 3  large accent spheres (metal, glass, red diffuse)
/// - 64 small spheres in an 8 × 8 grid at y = 0 (radius 0.3)
///
/// Total: **68 spheres** — enough to demonstrate BVH acceleration.
///
/// Material indices cycle deterministically through the four slots so the scene
/// contains a mix of all material types without randomness.
///
/// The equivalent data-driven scene is `assets/scene.yaml`, loaded by
/// [`load_scene_from_yaml`].  This builder is retained for unit tests and as
/// a reference implementation.
#[allow(dead_code)] // superseded by load_scene_from_yaml + assets/scene.yaml; kept for tests
pub fn build_large_scene() -> Vec<GpuSphere> {
    let mut spheres = Vec::with_capacity(68);

    // Ground plane.
    spheres.push(GpuSphere {
        center_r: [0.0, -100.5, 0.0, 100.0],
        mat_and_pad: [0, 0, 0, 0],
    });

    // Three large accent spheres (centre, left, right).
    spheres.push(GpuSphere {
        center_r: [0.0, 2.5, 0.0, 1.0],
        mat_and_pad: [3, 0, 0, 0],
    }); // glass
    spheres.push(GpuSphere {
        center_r: [-4.0, 1.0, 0.0, 1.0],
        mat_and_pad: [2, 0, 0, 0],
    }); // metal
    spheres.push(GpuSphere {
        center_r: [4.0, 1.0, 0.0, 1.0],
        mat_and_pad: [1, 0, 0, 0],
    }); // red

    // 8 × 8 grid of small spheres scattered across the ground plane.
    for row in 0..8i32 {
        for col in 0..8i32 {
            let x = -3.5 + col as f32;
            let z = -3.5 + row as f32;
            // Deterministic material cycling: produces an even distribution across
            // the four material slots without requiring a PRNG.
            let mat = ((row * 3 + col) % 4).unsigned_abs();
            spheres.push(GpuSphere {
                center_r: [x, 0.0, z, 0.3],
                mat_and_pad: [mat, 0, 0, 0],
            });
        }
    }

    spheres
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ----- build_scene tests -----------------------------------------------

    #[test]
    fn small_scene_has_four_spheres() {
        assert_eq!(build_scene().len(), 4);
    }

    #[test]
    fn small_scene_ground_radius_is_100() {
        assert_eq!(build_scene()[0].center_r[3], 100.0);
    }

    #[test]
    fn small_scene_material_indices_are_distinct() {
        let spheres = build_scene();
        let mut indices: Vec<u32> = spheres.iter().map(|s| s.mat_and_pad[0]).collect();
        let before = indices.len();
        indices.dedup();
        assert_eq!(
            indices.len(),
            before,
            "expected all material indices to be unique"
        );
    }

    #[test]
    fn small_scene_all_radii_positive() {
        for s in build_scene() {
            assert!(s.center_r[3] > 0.0, "non-positive radius {:?}", s.center_r);
        }
    }

    // ----- build_large_scene tests -----------------------------------------

    #[test]
    fn large_scene_sphere_count() {
        // 1 ground + 3 accent + 8*8 grid = 68.
        assert_eq!(build_large_scene().len(), 68);
    }

    #[test]
    fn large_scene_ground_radius_is_100() {
        assert_eq!(build_large_scene()[0].center_r[3], 100.0);
    }

    #[test]
    fn large_scene_all_radii_positive() {
        for s in build_large_scene() {
            assert!(s.center_r[3] > 0.0, "non-positive radius {:?}", s.center_r);
        }
    }

    #[test]
    fn large_scene_material_indices_valid() {
        for s in build_large_scene() {
            assert!(
                s.mat_and_pad[0] < 4,
                "material index {} out of range",
                s.mat_and_pad[0]
            );
        }
    }

    #[test]
    fn large_scene_contains_all_material_types() {
        let spheres = build_large_scene();
        let mut used = [false; 4];
        for s in &spheres {
            let idx = s.mat_and_pad[0] as usize;
            used[idx] = true;
        }
        assert!(
            used.iter().all(|&u| u),
            "not all material slots are used: {:?}",
            used
        );
    }

    #[test]
    fn gpu_sphere_is_32_bytes() {
        assert_eq!(std::mem::size_of::<GpuSphere>(), 32);
    }

    // ----- load_scene_from_yaml tests (Phase 12) ---------------------------
    //
    // All tests use self-contained inline YAML so they are independent of the
    // contents of `assets/scene.yaml` on disk.

    /// Minimal self-contained scene fixture used by multiple Phase 12/13 tests.
    ///
    /// Contains:
    ///   - 6 named materials (ground, glass, metal, red, bunny, light)
    ///   - 5 spheres (4 non-emissive + 1 emissive area light at [-2,4,0] r=0.5)
    ///   - 1 mesh object (models/bunny.obj, scale 8, translated onto ground)
    ///   - camera block (look_from=[4,3,8], look_at=[0,0.5,0], vfov=40, aperture=0.1)
    const FIXTURE_SCENE: &str = r#"
camera:
  look_from: [4.0, 3.0, 8.0]
  look_at:   [0.0, 0.5, 0.0]
  vfov:      40.0
  aperture:  0.1

materials:
  ground: { type: lambertian, albedo: [0.5, 0.5, 0.5] }
  glass:  { type: dielectric, ior: 1.5 }
  metal:  { type: metal,      albedo: [0.8, 0.6, 0.2], fuzz: 0.1 }
  red:    { type: lambertian, albedo: [0.7, 0.1, 0.1] }
  bunny:  { type: lambertian, albedo: [0.1, 0.1, 0.65] }
  light:  { type: emissive,   albedo: [1.0, 0.85, 0.7], emission_strength: 8.0 }

objects:
  - { type: sphere, center: [0.0, -100.0, 0.0], radius: 100.0, material: ground }
  - { type: sphere, center: [0.0,   2.5,  0.0], radius: 1.0,   material: glass  }
  - { type: sphere, center: [-4.0,  1.0,  0.0], radius: 1.0,   material: metal  }
  - { type: sphere, center: [4.0,   1.0,  0.0], radius: 1.0,   material: red    }
  - { type: sphere, center: [-2.0,  4.0,  0.0], radius: 0.5,   material: light  }
  - type: mesh
    path: models/bunny.obj
    material: bunny
    scale: 8.0
    translate: [0.5, -0.264, 0.0]
"#;

    #[test]
    fn yaml_scene_sphere_count() {
        let loaded = load_scene_from_str(FIXTURE_SCENE, "<fixture>");
        // FIXTURE_SCENE has 5 sphere objects (4 non-emissive + 1 light).
        assert_eq!(loaded.spheres.len(), 5);
    }

    #[test]
    fn yaml_scene_sphere_centers_are_correct() {
        let loaded = load_scene_from_str(FIXTURE_SCENE, "<fixture>");
        assert_eq!(loaded.spheres[0].center_r, [0.0_f32, -100.0, 0.0, 100.0]);
        assert_eq!(loaded.spheres[1].center_r, [0.0_f32, 2.5, 0.0, 1.0]);
        assert_eq!(loaded.spheres[2].center_r, [-4.0_f32, 1.0, 0.0, 1.0]);
        assert_eq!(loaded.spheres[3].center_r, [4.0_f32, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn yaml_scene_sphere_material_indices_in_range() {
        let loaded = load_scene_from_str(FIXTURE_SCENE, "<fixture>");
        let mat_count = loaded.materials.mat_count;
        for (i, s) in loaded.spheres.iter().enumerate() {
            assert!(
                s.mat_and_pad[0] < mat_count,
                "sphere {} has material index {} but palette only has {} entries",
                i,
                s.mat_and_pad[0],
                mat_count
            );
        }
    }

    #[test]
    fn yaml_scene_material_count() {
        let loaded = load_scene_from_str(FIXTURE_SCENE, "<fixture>");
        // FIXTURE_SCENE declares 6 named materials: ground, glass, metal, red, bunny, light.
        assert_eq!(loaded.materials.mat_count, 6);
    }

    // ----- Phase 13: emissive material tests -------------------------------

    #[test]
    fn yaml_scene_emissive_sphere_tracked() {
        // Use a lightweight mesh-free inline YAML so this test has no
        // dependency on models/bunny.obj (FIXTURE_SCENE includes a mesh).
        let yaml = r#"
materials:
  wall:  { type: lambertian, albedo: [0.8, 0.8, 0.8] }
  light: { type: emissive,   albedo: [1.0, 0.85, 0.7], emission_strength: 8.0 }
objects:
  - { type: sphere, center: [0.0, -100.0, 0.0], radius: 100.0, material: wall  }
  - { type: sphere, center: [-2.0,   4.0,  0.0], radius: 0.5,   material: light }
"#;
        let loaded = load_scene_from_str(yaml, "<inline>");
        assert_eq!(
            loaded.emissive_spheres.len(),
            1,
            "expected 1 emissive sphere, got {}",
            loaded.emissive_spheres.len()
        );
        assert_eq!(loaded.materials.n_emissive, 1);
        // Verify the emissive sphere's position and radius.
        assert_eq!(
            loaded.emissive_spheres[0].center_r,
            [-2.0_f32, 4.0, 0.0, 0.5]
        );
    }

    #[test]
    fn yaml_scene_no_emissive_when_none_declared() {
        let yaml = r#"
materials:
  wall: { type: lambertian, albedo: [0.8, 0.8, 0.8] }
objects:
  - { type: sphere, center: [0, 0, 0], radius: 1, material: wall }
"#;
        let loaded = load_scene_from_str(yaml, "<inline>");
        assert_eq!(
            loaded.emissive_spheres.len(),
            0,
            "expected no emissive spheres"
        );
        assert_eq!(loaded.materials.n_emissive, 0);
    }

    #[test]
    fn yaml_scene_emissive_material_gpu_type() {
        // The GPU material type field must be MAT_EMISSIVE (3) for emissive mats.
        let yaml = r#"
materials:
  sun: { type: emissive, albedo: [1.0, 0.9, 0.7], emission_strength: 10.0 }
objects:
  - { type: sphere, center: [0, 5, 0], radius: 1.0, material: sun }
"#;
        let loaded = load_scene_from_str(yaml, "<inline>");
        assert_eq!(loaded.materials.mat_count, 1);
        assert_eq!(
            loaded.materials.materials[0].type_pad[0],
            crate::material::MAT_EMISSIVE,
            "emissive material must have GPU type = MAT_EMISSIVE"
        );
        // Emission colour stored in albedo_fuzz.xyz, strength in ior_pad.x.
        assert!((loaded.materials.materials[0].albedo_fuzz[0] - 1.0).abs() < 1e-5);
        assert!((loaded.materials.materials[0].ior_pad[0] - 10.0).abs() < 1e-5);
    }

    #[test]
    fn yaml_scene_inline_material_is_appended() {
        // An inline material on an object must be appended as a new GPU slot on
        // top of any named materials.  1 named + 1 inline = mat_count 2.
        let yaml = r#"
materials:
  base: { type: lambertian, albedo: [0.5, 0.5, 0.5] }
objects:
  - type: sphere
    center: [0, 0, 0]
    radius: 1
    material: { type: metal, albedo: [0.8, 0.6, 0.2], fuzz: 0.1 }
"#;
        let loaded = load_scene_from_str(yaml, "<inline>");
        assert_eq!(
            loaded.materials.mat_count, 2,
            "expected 1 named + 1 inline = 2 materials, got {}",
            loaded.materials.mat_count
        );
    }

    #[test]
    fn yaml_scene_has_mesh_geometry() {
        // FIXTURE_SCENE includes models/bunny.obj; verify non-trivial geometry.
        let loaded = load_scene_from_str(FIXTURE_SCENE, "<fixture>");
        assert!(
            loaded.mesh_vertices.len() > 100,
            "expected >100 mesh vertices (bunny), got {}",
            loaded.mesh_vertices.len()
        );
        assert!(
            loaded.mesh_triangles.len() > 100,
            "expected >100 mesh triangles (bunny), got {}",
            loaded.mesh_triangles.len()
        );
    }

    #[test]
    fn yaml_scene_camera_is_loaded() {
        let loaded = load_scene_from_str(FIXTURE_SCENE, "<fixture>");
        let cam = loaded.camera.expect("fixture must have a camera block");
        assert_eq!(cam.look_from, [4.0_f32, 3.0, 8.0]);
        assert_eq!(cam.look_at, [0.0_f32, 0.5, 0.0]);
        assert!((cam.vfov - 40.0).abs() < 1e-4, "vfov should be 40°");
        assert!(
            (cam.aperture - 0.1).abs() < 1e-5,
            "expected aperture 0.1, got {}",
            cam.aperture
        );
    }

    #[test]
    fn yaml_plane_produces_two_triangles_and_correct_normal() {
        // A floor plane: center at origin, normal up, half_extents 1×1.
        // Expected: 4 vertices added to mesh buffer, 2 triangles.
        // The cross product of the two triangle edges must equal the plane normal.
        let yaml = r#"
materials:
  floor: { type: lambertian, albedo: [0.5, 0.5, 0.5] }
objects:
  - type: plane
    center: [0.0, 0.0, 0.0]
    normal: [0.0, 1.0, 0.0]
    half_extents: [1.0, 1.0]
    material: floor
"#;
        let loaded = load_scene_from_str(yaml, "<inline>");
        // 4 real vertices + 0 stub (mesh is non-empty)
        assert_eq!(
            loaded.mesh_vertices.len(),
            4,
            "plane must produce 4 vertices"
        );
        assert_eq!(
            loaded.mesh_triangles.len(),
            2,
            "plane must produce 2 triangles"
        );

        // Verify the geometric normal of the first triangle equals [0,1,0].
        let tri = &loaded.mesh_triangles[0];
        let v = |i: u32| loaded.mesh_vertices[i as usize].position;
        let p0 = v(tri.v[0]);
        let p1 = v(tri.v[1]);
        let p2 = v(tri.v[2]);
        let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
        let gn = plane_normalize3(plane_cross3(e1, e2));
        assert!(
            (gn[1] - 1.0).abs() < 1e-5,
            "triangle geometric normal should be ~[0,1,0], got {:?}",
            gn
        );

        // Verify stored per-vertex normal matches what we declared.
        for v in &loaded.mesh_vertices {
            assert!(
                (v.normal[1] - 1.0).abs() < 1e-5,
                "stored vertex normal should be [0,1,0], got {:?}",
                v.normal
            );
        }
    }

    #[test]
    fn yaml_plane_material_index_in_range() {
        let yaml = r#"
materials:
  wall: { type: lambertian, albedo: [0.8, 0.1, 0.1] }
objects:
  - type: plane
    center: [0.0, 2.0, -5.0]
    normal: [0.0, 0.0,  1.0]
    half_extents: [2.0, 2.0]
    material: wall
"#;
        let loaded = load_scene_from_str(yaml, "<inline>");
        let mat_count = loaded.materials.mat_count;
        for tri in &loaded.mesh_triangles {
            // Mask off the backface-cull flag before comparing against mat_count.
            let bare_idx = tri.mat_idx & !(1u32 << 31);
            assert!(
                bare_idx < mat_count,
                "plane triangle mat_idx {} out of range (mat_count={})",
                bare_idx,
                mat_count
            );
        }
    }

    #[test]
    fn yaml_plane_backface_culling_sets_flag_in_mat_idx() {
        // When backface_culling: true, bit 31 of every triangle's mat_idx must be
        // set.  When omitted (default false), bit 31 must be clear.
        const BACKFACE_CULL_FLAG: u32 = 1u32 << 31;
        let yaml = r#"
materials:
  wall: { type: lambertian, albedo: [0.7, 0.7, 0.7] }
objects:
  - type: plane
    center: [0.0, 0.0, 0.0]
    normal: [0.0, 1.0, 0.0]
    half_extents: [1.0, 1.0]
    material: wall
    backface_culling: true
  - type: plane
    center: [0.0, 5.0, 0.0]
    normal: [0.0, -1.0, 0.0]
    half_extents: [1.0, 1.0]
    material: wall
"#;
        let loaded = load_scene_from_str(yaml, "<inline>");
        // First 2 triangles belong to the culled plane; next 2 to the double-sided one.
        for tri in &loaded.mesh_triangles[..2] {
            assert_ne!(
                tri.mat_idx & BACKFACE_CULL_FLAG,
                0,
                "culled plane triangle should have bit 31 set, got {:#010x}",
                tri.mat_idx
            );
        }
        for tri in &loaded.mesh_triangles[2..] {
            assert_eq!(
                tri.mat_idx & BACKFACE_CULL_FLAG,
                0,
                "double-sided plane triangle should NOT have bit 31 set, got {:#010x}",
                tri.mat_idx
            );
        }
    }

    #[test]
    fn yaml_box_produces_six_faces() {
        // A unit box should produce 6 faces × 4 vertices = 24 vertices and
        // 6 × 2 = 12 triangles.
        let yaml = r#"
Materials:
  wall: { type: lambertian, albedo: [0.8, 0.8, 0.8] }
objects:
  - type: box
    center: [0.0, 0.5, 0.0]
    half_extents: [0.5, 0.5, 0.5]
    material: { type: lambertian }
"#;
        // Lowercase keys required; reconstruct with valid YAML.
        let yaml = r#"
materials:
  wall: { type: lambertian, albedo: [0.8, 0.8, 0.8] }
objects:
  - type: box
    center: [0.0, 0.5, 0.0]
    half_extents: [0.5, 0.5, 0.5]
    material: wall
"#;
        let loaded = load_scene_from_str(yaml, "<inline>");
        assert_eq!(
            loaded.mesh_vertices.len(),
            24,
            "box must produce 24 vertices (6 faces × 4)"
        );
        assert_eq!(
            loaded.mesh_triangles.len(),
            12,
            "box must produce 12 triangles (6 faces × 2)"
        );
        // Every triangle must reference a valid material slot.
        let mat_count = loaded.materials.mat_count;
        for tri in &loaded.mesh_triangles {
            assert!(
                tri.mat_idx < mat_count,
                "box triangle mat_idx {} out of range (mat_count={})",
                tri.mat_idx,
                mat_count
            );
        }
    }

    #[test]
    fn yaml_box_normals_are_axis_aligned() {
        // Each face of an axis-aligned box must have a per-vertex normal that
        // is one of the six axis-aligned unit vectors.
        let yaml = r#"
materials:
  m: { type: lambertian, albedo: [1.0, 1.0, 1.0] }
objects:
  - type: box
    center: [0.0, 0.0, 0.0]
    half_extents: [1.0, 2.0, 3.0]
    material: m
"#;
        let loaded = load_scene_from_str(yaml, "<inline>");
        let axis_normals: [[f32; 3]; 6] = [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ];
        for (i, v) in loaded.mesh_vertices.iter().enumerate() {
            let n = [v.normal[0], v.normal[1], v.normal[2]];
            let is_axis = axis_normals.iter().any(|&a| {
                (n[0] - a[0]).abs() < 1e-5
                    && (n[1] - a[1]).abs() < 1e-5
                    && (n[2] - a[2]).abs() < 1e-5
            });
            assert!(is_axis, "vertex {i} has non-axis-aligned normal {n:?}");
        }
    }

    #[test]
    fn yaml_box_non_uniform_scale_maps_extents_to_correct_axes() {
        // Regression test for the Gram-Schmidt tangent-swap bug: when hx, hy,
        // hz are all different, each face must span exactly 2·hN on its two
        // non-normal axes.  Before the fix, the ±Y and ±Z faces silently
        // swapped two of their half-extents, so stretching one axis deformed
        // unrelated faces.
        let yaml = r#"
materials:
  m: { type: lambertian, albedo: [1.0, 1.0, 1.0] }
objects:
  - type: box
    center: [0.0, 0.0, 0.0]
    half_extents: [1.0, 2.0, 3.0]
    material: m
"#;
        // hx=1, hy=2, hz=3 → box is 2×4×6 in X×Y×Z.
        let loaded = load_scene_from_str(yaml, "<inline>");
        let verts = &loaded.mesh_vertices;

        // For each face, collect the 4 vertices whose stored normal matches
        // and verify the span in each world axis.
        let check_face = |nx: f32, ny: f32, nz: f32, span_x: f32, span_y: f32, span_z: f32| {
            let face_verts: Vec<_> = verts
                .iter()
                .filter(|v| {
                    (v.normal[0] - nx).abs() < 1e-5
                        && (v.normal[1] - ny).abs() < 1e-5
                        && (v.normal[2] - nz).abs() < 1e-5
                })
                .collect();
            assert_eq!(
                face_verts.len(),
                4,
                "face ({nx},{ny},{nz}) must have 4 vertices"
            );
            let xs: Vec<f32> = face_verts.iter().map(|v| v.position[0]).collect();
            let ys: Vec<f32> = face_verts.iter().map(|v| v.position[1]).collect();
            let zs: Vec<f32> = face_verts.iter().map(|v| v.position[2]).collect();
            let span = |vals: &[f32]| {
                vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
                    - vals.iter().cloned().fold(f32::INFINITY, f32::min)
            };
            let tol = 1e-4;
            assert!(
                (span(&xs) - span_x).abs() < tol,
                "face ({nx},{ny},{nz}) X-span={} expected {span_x}",
                span(&xs)
            );
            assert!(
                (span(&ys) - span_y).abs() < tol,
                "face ({nx},{ny},{nz}) Y-span={} expected {span_y}",
                span(&ys)
            );
            assert!(
                (span(&zs) - span_z).abs() < tol,
                "face ({nx},{ny},{nz}) Z-span={} expected {span_z}",
                span(&zs)
            );
        };

        // ±X faces: fixed x, span Y=4, Z=6
        check_face(1., 0., 0., 0., 4., 6.);
        check_face(-1., 0., 0., 0., 4., 6.);
        // ±Y faces: fixed y, span X=2, Z=6
        check_face(0., 1., 0., 2., 0., 6.);
        check_face(0., -1., 0., 2., 0., 6.);
        // ±Z faces: fixed z, span X=2, Y=4
        check_face(0., 0., 1., 2., 4., 0.);
        check_face(0., 0., -1., 2., 4., 0.);
    }

    #[test]
    fn yaml_box_rotate_y_90_swaps_x_and_z() {
        // A 90° Y rotation maps local +X → world -Z and local +Z → world +X.
        // All 24 vertices of a unit box rotated 90° around Y must lie within
        // the correct axis ranges: X ∈ [-1, 1], Y ∈ [-1, 1], Z ∈ [-1, 1]
        // (still a unit cube, just reoriented).
        // More specifically the vertices that were on the local ±X faces
        // (x = ±1 in local space) should now have |z| ≈ 1, |x| ≈ 0.
        let yaml = r#"
materials:
  m: { type: lambertian, albedo: [1.0, 1.0, 1.0] }
objects:
  - type: box
    center: [0.0, 0.0, 0.0]
    half_extents: [1.0, 1.0, 1.0]
    rotate: [0, 90, 0]
    material: m
"#;
        let loaded = load_scene_from_str(yaml, "<inline>");
        let tol = 1e-5_f32;

        // After Ry(90°): local X -> world -Z, local Z -> world X.
        // All rotated normals must still be unit vectors.
        for (i, v) in loaded.mesh_vertices.iter().enumerate() {
            let n = &v.normal;
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!(
                (len - 1.0).abs() < tol,
                "vertex {i} normal not unit: len={len}"
            );
        }

        // Collect face normals; after Ry(90) they must all be axis-aligned.
        let axis_normals: [[f32; 3]; 6] = [
            [1., 0., 0.],
            [-1., 0., 0.],
            [0., 1., 0.],
            [0., -1., 0.],
            [0., 0., 1.],
            [0., 0., -1.],
        ];
        for (i, v) in loaded.mesh_vertices.iter().enumerate() {
            let n = [v.normal[0], v.normal[1], v.normal[2]];
            let ok = axis_normals.iter().any(|&a| {
                (n[0] - a[0]).abs() < tol && (n[1] - a[1]).abs() < tol && (n[2] - a[2]).abs() < tol
            });
            assert!(
                ok,
                "vertex {i} normal {n:?} is not axis-aligned after Ry(90)"
            );
        }

        // Original +X face (local normal [1,0,0]) should map to world [0,0,-1].
        // Find those 4 vertices: they should have z ≈ -1, x ≈ 0.
        let minus_z_face: Vec<_> = loaded
            .mesh_vertices
            .iter()
            .filter(|v| (v.normal[2] + 1.0).abs() < tol)
            .collect();
        assert_eq!(
            minus_z_face.len(),
            4,
            "rotated +X face should appear as -Z face"
        );
        for v in &minus_z_face {
            assert!(
                (v.position[0]).abs() < tol + 1.0 + tol, // x ∈ [-1, 1]
                "rotated -Z face vertex has unexpected x={}",
                v.position[0]
            );
            assert!(
                (v.position[2] + 1.0).abs() < tol,
                "rotated -Z face vertex z should be -1, got {}",
                v.position[2]
            );
        }
    }

    #[test]
    fn yaml_scene_camera_focus_dist_defaults_to_eye_distance() {
        // When focus_dist is omitted the loader should default it to the
        // Euclidean distance between look_from and look_at (here = 5.0).
        let yaml = r#"
camera:
  look_from: [0.0, 0.0, 5.0]
  look_at:   [0.0, 0.0, 0.0]
objects:
  - { type: sphere, center: [0, 0, 0], radius: 1, material: { type: lambertian } }
"#;
        let loaded = load_scene_from_str(yaml, "<inline>");
        let cam = loaded.camera.expect("inline yaml must have a camera block");
        assert!(
            (cam.focus_dist - 5.0).abs() < 1e-4,
            "expected focus_dist ≈ 5.0 (eye distance), got {}",
            cam.focus_dist
        );
    }

    #[test]
    fn yaml_scene_mesh_material_index_in_range() {
        // Every triangle in the loaded mesh must reference a valid material slot.
        let loaded = load_scene_from_str(FIXTURE_SCENE, "<fixture>");
        let mat_count = loaded.materials.mat_count;
        for (i, tri) in loaded.mesh_triangles.iter().enumerate() {
            assert!(
                tri.mat_idx < mat_count,
                "triangle {} has mat_idx {} but mat_count = {}",
                i,
                tri.mat_idx,
                mat_count
            );
        }
    }
}
