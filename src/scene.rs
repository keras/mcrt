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
// YAML scene loader (Phase 12)
// ---------------------------------------------------------------------------

/// Per-sphere entry in the YAML scene file.
#[derive(serde::Deserialize)]
struct SphereDesc {
    /// World-space centre as `[x, y, z]`.
    center: [f32; 3],
    /// Sphere radius (must be positive).
    radius: f32,
    /// Index into the material array built by [`crate::material::build_materials`].
    material: u32,
}

/// Top-level structure of a YAML scene file.
#[derive(serde::Deserialize)]
struct SceneFile {
    spheres: Vec<SphereDesc>,
}

/// Load a scene from a YAML file and return the corresponding `Vec<GpuSphere>`.
///
/// The YAML must contain a top-level `spheres` sequence; each element must
/// provide `center`, `radius`, and `material` fields.  See `assets/scene.yaml`
/// for a complete example.
///
/// # Panics
/// Panics if the file cannot be read or the YAML is malformed.
pub fn load_scene_from_yaml(path: &str) -> Vec<GpuSphere> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("failed to read scene file '{}': {}", path, e));
    let scene: SceneFile = serde_yaml::from_str(&text)
        .unwrap_or_else(|e| panic!("failed to parse scene file '{}': {}", path, e));
    scene
        .spheres
        .iter()
        .map(|s| GpuSphere {
            center_r: [s.center[0], s.center[1], s.center[2], s.radius],
            mat_and_pad: [s.material, 0, 0, 0],
        })
        .collect()
}



/// Construct the original four-sphere demo scene.
///
/// Useful for unit tests and quick sanity checks.  For the Phase 9 BVH
/// demonstration, prefer [`build_large_scene`] which exercises the
/// acceleration structure with many more primitives.
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
pub fn build_large_scene() -> Vec<GpuSphere> {
    let mut spheres = Vec::with_capacity(68);

    // Ground plane.
    spheres.push(GpuSphere {
        center_r: [0.0, -100.5, 0.0, 100.0],
        mat_and_pad: [0, 0, 0, 0],
    });

    // Three large accent spheres (centre, left, right).
    spheres.push(GpuSphere {
        center_r: [0.0, 1.0, 0.0, 1.0],
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
}
