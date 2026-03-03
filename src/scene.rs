// scene.rs — Phase 4+: GPU sphere scene data types and static scene builder
//
// All types here are bytemuck-compatible (Pod + Zeroable) so they can be
// uploaded directly to wgpu uniform/storage buffers without manual packing.
// No wgpu types are imported, keeping this module independently testable.

use bytemuck::Zeroable;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of spheres in a [`GpuSceneData`] buffer.
///
/// **Must match** `MAX_SPHERES` in `path_trace.wgsl`.  When this value
/// changes, also update the WGSL shader constant.
pub const MAX_SPHERES: usize = 8;

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
    pub center_r:    [f32; 4],
    /// x = material index, yzw = unused padding.
    pub mat_and_pad: [u32; 4],
}

/// Full scene uploaded to the GPU.
///
/// Size = 16 + [`MAX_SPHERES`] × 32 = 272 bytes (well under the 64 KiB
/// uniform buffer guarantee — resize to storage buffer for Phase 9+).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSceneData {
    pub sphere_count: u32,
    pub _pad:         [u32; 3],
    pub spheres:      [GpuSphere; MAX_SPHERES],
}

// ---------------------------------------------------------------------------
// Scene builder
// ---------------------------------------------------------------------------

/// Construct the default scene: a large ground sphere plus three accent spheres.
///
/// Material indices reference the table built by [`crate::material::build_materials`]:
/// - 0 — Lambertian grey  (ground)
/// - 1 — Lambertian red   (centre sphere)
/// - 2 — Metal gold, fuzz=0.1  (left sphere)
/// - 3 — Dielectric glass, IOR=1.5 (right sphere)
pub fn build_scene() -> GpuSceneData {
    let raw: &[GpuSphere] = &[
        // Ground plane represented as a large sphere.
        GpuSphere { center_r: [0.0, -100.5, 0.0, 100.0], mat_and_pad: [0, 0, 0, 0] },
        // Centre sphere.
        GpuSphere { center_r: [0.0,    0.0, 0.0,   0.5], mat_and_pad: [1, 0, 0, 0] },
        // Left sphere.
        GpuSphere { center_r: [-1.2,   0.0, 0.0,   0.5], mat_and_pad: [2, 0, 0, 0] },
        // Right sphere.
        GpuSphere { center_r: [ 1.2,   0.0, 0.0,   0.5], mat_and_pad: [3, 0, 0, 0] },
    ];

    let mut data = GpuSceneData::zeroed();
    data.sphere_count = raw.len() as u32;
    data.spheres[..raw.len()].copy_from_slice(raw);
    data
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scene_sphere_count_is_four() {
        assert_eq!(build_scene().sphere_count, 4);
    }

    #[test]
    fn scene_sphere_count_within_capacity() {
        let s = build_scene();
        assert!((s.sphere_count as usize) <= MAX_SPHERES);
    }

    #[test]
    fn ground_sphere_has_large_radius() {
        let s = build_scene();
        assert_eq!(s.spheres[0].center_r[3], 100.0);
    }

    #[test]
    fn material_indices_are_distinct() {
        let s = build_scene();
        let n = s.sphere_count as usize;
        let indices: Vec<u32> = (0..n).map(|i| s.spheres[i].mat_and_pad[0]).collect();
        // Each sphere uses a unique material.
        let mut seen = indices.clone();
        seen.dedup();
        assert_eq!(seen.len(), n, "Expected all material indices to be unique");
    }

    #[test]
    fn all_sphere_radii_positive() {
        let s = build_scene();
        for i in 0..s.sphere_count as usize {
            assert!(s.spheres[i].center_r[3] > 0.0, "sphere {i} has non-positive radius");
        }
    }
}
