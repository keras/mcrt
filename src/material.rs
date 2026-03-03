// material.rs — Phase 7+: GPU material data types and static material builder
//
// All types are bytemuck-compatible (Pod + Zeroable) for direct GPU upload.
// No wgpu imports live here, keeping the module unit-testable without a device.

use bytemuck::Zeroable;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of materials in a [`GpuMaterialData`] buffer.
///
/// **Must match** `MAX_MATERIALS` in `path_trace.wgsl`.  When this value
/// changes, update the WGSL shader constant too.
pub const MAX_MATERIALS: usize = 8;

// ---------------------------------------------------------------------------
// GPU types
// ---------------------------------------------------------------------------

/// GPU-side material descriptor.
///
/// Each field uses vec4 packing to keep the Rust `repr(C)` layout identical
/// to the WGSL `struct Material` layout.
///
/// | Field          | Meaning                                                      |
/// |----------------|--------------------------------------------------------------|
/// | `type_pad[0]`  | Material type: 0=Lambertian, 1=metal, 2=dielectric           |
/// | `type_pad[1]`  | Albedo texture layer index (0 = white/no-texture; Phase 11)  |
/// | `albedo_fuzz[0..3]` | Albedo / metal tint colour                            |
/// | `albedo_fuzz[3]`    | Fuzz radius (metal: 0=mirror, 1=fully rough)          |
/// | `ior_pad[0]`   | Index of refraction (dielectric, e.g. 1.5 for glass)         |
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuMaterial {
    /// .x = material type (u32), .yzw = unused.
    pub type_pad:    [u32; 4],
    /// .xyz = albedo colour, .w = fuzz ∈ [0, 1].
    pub albedo_fuzz: [f32; 4],
    /// .x = index of refraction, .yzw = unused.
    pub ior_pad:     [f32; 4],
}

/// Full material table uploaded to the GPU.
///
/// Size = 16 + [`MAX_MATERIALS`] × 48 = 400 bytes.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuMaterialData {
    pub mat_count: u32,
    pub _pad:      [u32; 3],
    pub materials: [GpuMaterial; MAX_MATERIALS],
}

// ---------------------------------------------------------------------------
// Material type constants (for readability in build_materials)
// ---------------------------------------------------------------------------

const MAT_LAMBERTIAN:  u32 = 0;
const MAT_METAL:       u32 = 1;
const MAT_DIELECTRIC:  u32 = 2;

// ---------------------------------------------------------------------------
// Material builder
// ---------------------------------------------------------------------------

/// Build the material table matching the scene from [`crate::scene::build_scene`]:
///
/// | Index | Description               |
/// |-------|---------------------------|
/// |   0   | Lambertian grey (ground)  |
/// |   1   | Lambertian red (centre)   |
/// |   2   | Metal gold, fuzz = 0.1    |
/// |   3   | Dielectric glass, IOR 1.5 |
pub fn build_materials() -> GpuMaterialData {
    let raw: &[GpuMaterial] = &[
        // 0: Lambertian grey — ground (checker texture, layer 1)
        GpuMaterial {
            type_pad:    [MAT_LAMBERTIAN, 1, 0, 0], // type_pad[1]=1 → albedo layer 1
            albedo_fuzz: [0.5, 0.5, 0.5, 0.0],
            ior_pad:     [1.0, 0.0, 0.0, 0.0],
        },
        // 1: Lambertian red — centre sphere (no texture)
        GpuMaterial {
            type_pad:    [MAT_LAMBERTIAN, 0, 0, 0],
            albedo_fuzz: [0.7, 0.3, 0.3, 0.0],
            ior_pad:     [1.0, 0.0, 0.0, 0.0],
        },
        // 2: Metal gold, fuzz = 0.1 — left sphere
        GpuMaterial {
            type_pad:    [MAT_METAL, 0, 0, 0],
            albedo_fuzz: [0.8, 0.6, 0.2, 0.1],
            ior_pad:     [1.0, 0.0, 0.0, 0.0],
        },
        // 3: Dielectric glass, IOR = 1.5 — right sphere
        GpuMaterial {
            type_pad:    [MAT_DIELECTRIC, 0, 0, 0],
            albedo_fuzz: [1.0, 1.0, 1.0, 0.0],
            ior_pad:     [1.5, 0.0, 0.0, 0.0],
        },
    ];

    let mut data = GpuMaterialData::zeroed();
    data.mat_count = raw.len() as u32;
    data.materials[..raw.len()].copy_from_slice(raw);
    data
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn material_count_is_four() {
        assert_eq!(build_materials().mat_count, 4);
    }

    #[test]
    fn material_count_within_capacity() {
        let m = build_materials();
        assert!((m.mat_count as usize) <= MAX_MATERIALS);
    }

    #[test]
    fn glass_ior_is_1_5() {
        let m = build_materials();
        let ior = m.materials[3].ior_pad[0];
        assert!((ior - 1.5).abs() < 1e-6, "glass IOR should be 1.5, got {ior}");
    }

    #[test]
    fn glass_type_is_dielectric() {
        let m = build_materials();
        assert_eq!(m.materials[3].type_pad[0], MAT_DIELECTRIC);
    }

    #[test]
    fn metal_fuzz_is_positive() {
        let m = build_materials();
        assert!(m.materials[2].albedo_fuzz[3] > 0.0, "metal fuzz should be > 0");
    }

    #[test]
    fn lambertian_materials_have_zero_fuzz() {
        let m = build_materials();
        // Indices 0 and 1 are Lambertian.
        for i in [0usize, 1] {
            assert_eq!(m.materials[i].type_pad[0], MAT_LAMBERTIAN);
            assert_eq!(m.materials[i].albedo_fuzz[3], 0.0);
        }
    }
}
