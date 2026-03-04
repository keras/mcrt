// gpu_layout.rs — single source of truth for the compute bind group layout.
//
// Centralising the layout here means gpu.rs and headless.rs share exactly the
// same 14-binding descriptor, so any future binding additions (IC-2 probes,
// VW chunks, SDF params) only need to be made in one place.

use std::mem::size_of;

use wgpu::*;

use crate::bvh::GpuBvhNode;
use crate::camera::CameraUniform;
use crate::material::GpuMaterialData;
use crate::mesh::{GpuTriangle, GpuVertex};
use crate::scene::GpuSphere;

// ---------------------------------------------------------------------------
// Named binding-slot constants — use these everywhere instead of bare integers
// ---------------------------------------------------------------------------

/// Binding 0: write-only storage texture (accumulation write target).
pub const BINDING_ACCUM_WRITE: u32 = 0;
/// Binding 1: camera parameters uniform buffer.
pub const BINDING_CAMERA: u32 = 1;
/// Binding 2: sphere list storage buffer (BVH-reordered).
pub const BINDING_SPHERES: u32 = 2;
/// Binding 3: previous-frame accumulation texture (read via textureLoad).
pub const BINDING_ACCUM_READ: u32 = 3;
/// Binding 4: material descriptors uniform buffer.
pub const BINDING_MATERIALS: u32 = 4;
/// Binding 5: sphere BVH node array storage buffer.
pub const BINDING_SPHERE_BVH: u32 = 5;
/// Binding 6: mesh vertex buffer.
pub const BINDING_VERTICES: u32 = 6;
/// Binding 7: mesh triangle index buffer.
pub const BINDING_TRIANGLES: u32 = 7;
/// Binding 8: mesh BVH node array storage buffer.
pub const BINDING_MESH_BVH: u32 = 8;
/// Binding 9: linear sampler for albedo texture array.
pub const BINDING_TEX_SAMPLER: u32 = 9;
/// Binding 10: albedo 2D texture array (RGBA8 Unorm, MAX_TEXTURES layers).
pub const BINDING_ALBEDO_TEX: u32 = 10;
/// Binding 11: HDR environment map (RGBA32 Float, non-filterable).
pub const BINDING_ENV_MAP: u32 = 11;
/// Binding 12: emissive sphere list for NEE.
pub const BINDING_EMISSIVES: u32 = 12;
/// Binding 13: G-buffer write target (normal.xyz + depth.w).
pub const BINDING_GBUFFER: u32 = 13;

// IC-2+ reserved — add to make_compute_bgl() and create buffers when IC-2 lands.
/// Binding 14: irradiance probe grid metadata uniform buffer (IC-2+).
#[allow(dead_code)]
pub const BINDING_PROBE_GRID: u32 = 14;
/// Binding 15: irradiance probe SH coefficients storage buffer (IC-2+).
#[allow(dead_code)]
pub const BINDING_PROBE_IRRADIANCE: u32 = 15;

// ---------------------------------------------------------------------------
// Denoise bind group layout slot (group 0, separate from the compute BGL)
// ---------------------------------------------------------------------------

/// Binding 3 in the **denoise** bind group: DenoiseParams uniform buffer.
/// (Bindings 0-2 are accum texture, G-buffer, and the write-only output texture.)
pub const BINDING_DENOISE_PARAMS: u32 = 3;

// ---------------------------------------------------------------------------
// Layout factory
// ---------------------------------------------------------------------------

/// Create the compute bind group layout shared by the interactive and headless
/// renderers.  Both pipelines must use this function — never inline a copy.
pub fn make_compute_bgl(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("compute bgl"),
        entries: &[
            // 0: write-only storage texture (accum write target)
            BindGroupLayoutEntry {
                binding: BINDING_ACCUM_WRITE,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    format: TextureFormat::Rgba32Float,
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            },
            // 1: camera parameters uniform
            BindGroupLayoutEntry {
                binding: BINDING_CAMERA,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(size_of::<CameraUniform>() as u64),
                },
                count: None,
            },
            // 2: sphere list storage buffer (BVH-reordered; runtime-sized array)
            BindGroupLayoutEntry {
                binding: BINDING_SPHERES,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(size_of::<GpuSphere>() as u64),
                },
                count: None,
            },
            // 3: previous-frame accumulation (read via textureLoad)
            BindGroupLayoutEntry {
                binding: BINDING_ACCUM_READ,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: false },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            // 4: material descriptors uniform
            BindGroupLayoutEntry {
                binding: BINDING_MATERIALS,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(size_of::<GpuMaterialData>() as u64),
                },
                count: None,
            },
            // 5: sphere BVH node array storage buffer (runtime-sized array)
            BindGroupLayoutEntry {
                binding: BINDING_SPHERE_BVH,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(size_of::<GpuBvhNode>() as u64),
                },
                count: None,
            },
            // 6: mesh vertex buffer (position, normal, UV; runtime-sized array)
            BindGroupLayoutEntry {
                binding: BINDING_VERTICES,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(size_of::<GpuVertex>() as u64),
                },
                count: None,
            },
            // 7: mesh triangle index buffer (runtime-sized array)
            BindGroupLayoutEntry {
                binding: BINDING_TRIANGLES,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(size_of::<GpuTriangle>() as u64),
                },
                count: None,
            },
            // 8: mesh BVH node array (runtime-sized array)
            BindGroupLayoutEntry {
                binding: BINDING_MESH_BVH,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(size_of::<GpuBvhNode>() as u64),
                },
                count: None,
            },
            // 9: linear sampler for albedo texture array
            BindGroupLayoutEntry {
                binding: BINDING_TEX_SAMPLER,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            },
            // 10: albedo 2D texture array (RGBA8 Unorm, MAX_TEXTURES layers)
            BindGroupLayoutEntry {
                binding: BINDING_ALBEDO_TEX,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            // 11: HDR environment map (RGBA32 Float; non-filterable, textureLoad)
            BindGroupLayoutEntry {
                binding: BINDING_ENV_MAP,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: false },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            // 12: emissive sphere list for NEE
            BindGroupLayoutEntry {
                binding: BINDING_EMISSIVES,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(size_of::<GpuSphere>() as u64),
                },
                count: None,
            },
            // 13: G-buffer write target (normal.xyz + depth.w)
            BindGroupLayoutEntry {
                binding: BINDING_GBUFFER,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    format: TextureFormat::Rgba32Float,
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            },
        ],
    })
}
