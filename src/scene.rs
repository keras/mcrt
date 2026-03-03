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
    look_from:  [f32; 3],
    look_at:    [f32; 3],
    #[serde(default)]
    vfov:       Option<f32>,
    #[serde(default)]
    aperture:   Option<f32>,
    #[serde(default)]
    focus_dist: Option<f32>,
}

/// Camera settings returned by [`load_scene_from_yaml`] when the YAML
/// contains a `camera:` block.  `gpu.rs` converts this into a `CameraState`.
#[derive(Clone, Debug)]
pub struct SceneCameraSettings {
    /// Eye position in world space.
    pub look_from:  [f32; 3],
    /// Point the camera looks toward.
    pub look_at:    [f32; 3],
    /// Vertical field of view in degrees.
    pub vfov:       f32,
    /// Lens aperture radius (0 = pinhole / no depth-of-field).
    pub aperture:   f32,
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
    },
    // Future variants:
    // Cube  { center, size, material }
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
}

// Convert a YAML MaterialDesc into a GPU-ready GpuMaterial.
fn material_desc_to_gpu(desc: &MaterialDesc) -> crate::material::GpuMaterial {
    use crate::material::{MAT_DIELECTRIC, MAT_EMISSIVE, MAT_LAMBERTIAN, MAT_METAL};
    let albedo = desc.albedo.unwrap_or([1.0, 1.0, 1.0]);
    match desc.kind {
        MaterialKind::Lambertian => crate::material::GpuMaterial {
            type_pad:    [MAT_LAMBERTIAN, desc.texture_layer, 0, 0],
            albedo_fuzz: [albedo[0], albedo[1], albedo[2], 0.0],
            ior_pad:     [1.0, 0.0, 0.0, 0.0],
        },
        MaterialKind::Metal => crate::material::GpuMaterial {
            type_pad:    [MAT_METAL, desc.texture_layer, 0, 0],
            albedo_fuzz: [albedo[0], albedo[1], albedo[2], desc.fuzz.unwrap_or(0.0)],
            ior_pad:     [1.0, 0.0, 0.0, 0.0],
        },
        MaterialKind::Dielectric => crate::material::GpuMaterial {
            type_pad:    [MAT_DIELECTRIC, 0, 0, 0],
            albedo_fuzz: [1.0, 1.0, 1.0, 0.0],
            ior_pad:     [desc.ior.unwrap_or(1.5), 0.0, 0.0, 0.0],
        },
        // Emissive: albedo_fuzz.xyz = emission colour; ior_pad.x = strength.
        // These materials terminate paths and serve as area lights for NEE.
        MaterialKind::Emissive => crate::material::GpuMaterial {
            type_pad:    [MAT_EMISSIVE, 0, 0, 0],
            albedo_fuzz: [albedo[0], albedo[1], albedo[2], 0.0],
            ior_pad:     [desc.emission_strength.unwrap_or(1.0), 0.0, 0.0, 0.0],
        },
    }
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
    let mut mesh_vertices:  Vec<crate::mesh::GpuVertex>   = Vec::new();
    let mut mesh_triangles: Vec<crate::mesh::GpuTriangle> = Vec::new();

    for obj in &scene.objects {
        match obj {
            ObjectDesc::Sphere { center, radius, material } => {
                let mat_idx = resolve_mat!(material);
                spheres.push(GpuSphere {
                    center_r:    [center[0], center[1], center[2], *radius],
                    mat_and_pad: [mat_idx, 0, 0, 0],
                });
            }

            ObjectDesc::Mesh { path: obj_path, material, scale, translate } => {
                let mat_idx = resolve_mat!(material);
                let (mut verts, tris) = crate::mesh::load_obj(obj_path, mat_idx)
                    .unwrap_or_else(|e| panic!("failed to load mesh '{}': {}", obj_path, e));

                // Apply optional uniform scale + translation to vertex positions.
                let s = scale.unwrap_or(1.0);
                // Negative scale would flip handedness without inverting stored normals,
                // producing inside-out lighting.  Zero scale collapses geometry into a
                // degenerate point.  Both are almost certainly unintentional.
                assert!(s > 0.0, "mesh '{}' scale must be positive (got {s})", obj_path);
                let t = translate.unwrap_or([0.0f32; 3]);
                if s != 1.0 || t != [0.0f32; 3] {
                    for v in &mut verts {
                        v.position[0] = v.position[0] * s + t[0];
                        v.position[1] = v.position[1] * s + t[1];
                        v.position[2] = v.position[2] * s + t[2];
                        // Normals are direction vectors; uniform scale preserves
                        // their direction so no normal adjustment is needed.
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
            normal:   [0.0, 1.0, 0.0, 0.0],
            uv:       [0.0, 0.0, 0.0, 0.0],
        });
        // A degenerate triangle (single repeated vertex) that Möller–Trumbore
        // will always reject (det ≈ 0).
        mesh_triangles.push(crate::mesh::GpuTriangle { v: [0, 0, 0], mat_idx: 0 });
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
            idx < gpu_mats.len()
                && gpu_mats[idx].type_pad[0] == crate::material::MAT_EMISSIVE
        })
        .copied()
        .collect();
    mat_data.n_emissive = emissive_spheres.len() as u32;

    // Convert the optional camera block.
    let camera = scene.camera.as_ref().map(|c| {
        let look_from = c.look_from;
        let look_at   = c.look_at;
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
            vfov:       c.vfov.unwrap_or(crate::camera::DEFAULT_VFOV),
            aperture:   c.aperture.unwrap_or(0.0),
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
        assert_eq!(loaded.spheres[1].center_r, [0.0_f32,   2.5,  0.0, 1.0]);
        assert_eq!(loaded.spheres[2].center_r, [-4.0_f32,  1.0,  0.0, 1.0]);
        assert_eq!(loaded.spheres[3].center_r, [4.0_f32,   1.0,  0.0, 1.0]);
    }

    #[test]
    fn yaml_scene_sphere_material_indices_in_range() {
        let loaded = load_scene_from_str(FIXTURE_SCENE, "<fixture>");
        let mat_count = loaded.materials.mat_count;
        for (i, s) in loaded.spheres.iter().enumerate() {
            assert!(
                s.mat_and_pad[0] < mat_count,
                "sphere {} has material index {} but palette only has {} entries",
                i, s.mat_and_pad[0], mat_count
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
        // FIXTURE_SCENE has exactly one emissive sphere (the `light` material).
        let loaded = load_scene_from_str(FIXTURE_SCENE, "<fixture>");
        assert_eq!(
            loaded.emissive_spheres.len(), 1,
            "expected 1 emissive sphere, got {}",
            loaded.emissive_spheres.len()
        );
        assert_eq!(loaded.materials.n_emissive, 1);
        // Verify the emissive sphere's position and radius match the fixture.
        assert_eq!(loaded.emissive_spheres[0].center_r, [-2.0_f32, 4.0, 0.0, 0.5]);
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
        assert_eq!(loaded.emissive_spheres.len(), 0, "expected no emissive spheres");
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
        assert_eq!(cam.look_at,   [0.0_f32, 0.5, 0.0]);
        assert!((cam.vfov - 40.0).abs() < 1e-4, "vfov should be 40°");
        assert!((cam.aperture - 0.1).abs() < 1e-5, "expected aperture 0.1, got {}", cam.aperture);
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
                i, tri.mat_idx, mat_count
            );
        }
    }
}
