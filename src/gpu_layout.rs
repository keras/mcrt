// gpu_layout.rs — single source of truth for the compute bind group layout.
//
// Centralising the layout here means gpu.rs and headless.rs share exactly the
// same 14-binding descriptor, so any future binding additions (IC-2 probes,
// VW chunks, SDF params) only need to be made in one place.

use std::mem::size_of;

use bytemuck::Zeroable;
use log::info;
use wgpu::util::DeviceExt;
use wgpu::*;

use crate::bvh::{GpuBvhNode, build_bvh};
use crate::camera::CameraUniform;
use crate::material::GpuMaterialData;
use crate::mesh::{GpuTriangle, GpuVertex, build_mesh_bvh};
use crate::scene::{GpuSphere, LoadedScene, STUB_SPHERE};
use crate::texture::{
    ENV_MAP_HEIGHT, ENV_MAP_WIDTH, MAX_TEXTURES, TEXTURE_SIZE, load_all_textures, load_env_map_data,
};

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

// ---------------------------------------------------------------------------
// SceneBuffers — scene-dependent GPU resources
// ---------------------------------------------------------------------------

/// GPU resources that are rebuilt whenever the scene changes (initial load
/// or hot-reload).
pub struct SceneBuffers {
    pub sphere_buffer: wgpu::Buffer,
    pub bvh_buffer: wgpu::Buffer,
    pub vertex_buffer: wgpu::Buffer,
    pub triangle_buffer: wgpu::Buffer,
    pub mesh_bvh_buffer: wgpu::Buffer,
    pub emissive_buffer: wgpu::Buffer,
    /// Keep the texture alive so the view's underlying allocation is not freed.
    pub env_map_tex: wgpu::Texture,
    pub env_map_view: wgpu::TextureView,
}

/// Build all scene-dependent GPU buffers and textures from a parsed scene.
///
/// Called once in `GpuState::new()` and again on every hot-reload in
/// `GpuState::reload_scene()`.  The albedo texture array is *not* rebuilt here
/// because it depends only on files on disk, not on the scene YAML.
pub fn build_scene_buffers(device: &Device, queue: &Queue, loaded: &LoadedScene) -> SceneBuffers {
    // ---- Sphere BVH --------------------------------------------------------
    let bvh_result = build_bvh(&loaded.spheres);
    // wgpu requires every storage-buffer binding to be at least as large as
    // the stride declared in the shader.  Pad with a stub sphere when empty.
    let sphere_data: Vec<GpuSphere> = if bvh_result.ordered_spheres.is_empty() {
        vec![STUB_SPHERE]
    } else {
        bvh_result.ordered_spheres.clone()
    };
    let sphere_buffer = device.create_buffer_init(&util::BufferInitDescriptor {
        label: Some("sphere storage"),
        contents: bytemuck::cast_slice(&sphere_data),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });
    let bvh_node_data: Vec<GpuBvhNode> = if bvh_result.nodes.is_empty() {
        vec![GpuBvhNode::zeroed()]
    } else {
        bvh_result.nodes.clone()
    };
    let bvh_buffer = device.create_buffer_init(&util::BufferInitDescriptor {
        label: Some("bvh node storage"),
        contents: bytemuck::cast_slice(&bvh_node_data),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });
    info!(
        "BVH built: {} spheres → {} nodes",
        bvh_result.ordered_spheres.len(),
        bvh_result.nodes.len()
    );

    // ---- Triangle mesh BVH -------------------------------------------------
    let mesh_result = build_mesh_bvh(&loaded.mesh_vertices, &loaded.mesh_triangles);
    info!(
        "Mesh BVH built: {} triangles, {} vertices → {} nodes",
        mesh_result.ordered_triangles.len(),
        mesh_result.vertices.len(),
        mesh_result.nodes.len()
    );
    let mesh_verts: Vec<GpuVertex> = if mesh_result.vertices.is_empty() {
        vec![GpuVertex::zeroed()]
    } else {
        mesh_result.vertices.clone()
    };
    let mesh_tris: Vec<GpuTriangle> = if mesh_result.ordered_triangles.is_empty() {
        vec![GpuTriangle::zeroed()]
    } else {
        mesh_result.ordered_triangles.clone()
    };
    let mesh_bvh_nodes: Vec<GpuBvhNode> = if mesh_result.nodes.is_empty() {
        vec![GpuBvhNode::zeroed()]
    } else {
        mesh_result.nodes.clone()
    };
    let vertex_buffer = device.create_buffer_init(&util::BufferInitDescriptor {
        label: Some("mesh vertex storage"),
        contents: bytemuck::cast_slice(&mesh_verts),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });
    let triangle_buffer = device.create_buffer_init(&util::BufferInitDescriptor {
        label: Some("mesh triangle storage"),
        contents: bytemuck::cast_slice(&mesh_tris),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });
    let mesh_bvh_buffer = device.create_buffer_init(&util::BufferInitDescriptor {
        label: Some("mesh bvh node storage"),
        contents: bytemuck::cast_slice(&mesh_bvh_nodes),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });

    // ---- Emissive sphere buffer --------------------------------------------
    // Always non-empty: the shader unconditionally indexes this array during NEE.
    let emissive_data: Vec<GpuSphere> = if loaded.emissive_spheres.is_empty() {
        vec![STUB_SPHERE]
    } else {
        loaded.emissive_spheres.clone()
    };
    let emissive_buffer = device.create_buffer_init(&util::BufferInitDescriptor {
        label: Some("emissive sphere storage"),
        contents: bytemuck::cast_slice(&emissive_data),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });

    // ---- Environment map texture -------------------------------------------
    let env_map_data = load_env_map_data(loaded.env_map_path.as_deref());
    let env_map_tex = device.create_texture(&TextureDescriptor {
        label: Some("env map"),
        size: Extent3d {
            width: ENV_MAP_WIDTH,
            height: ENV_MAP_HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba32Float,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        TexelCopyTextureInfo {
            texture: &env_map_tex,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        bytemuck::cast_slice(&env_map_data),
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(ENV_MAP_WIDTH * 4 * 4), // 4 channels × 4 bytes
            rows_per_image: Some(ENV_MAP_HEIGHT),
        },
        Extent3d {
            width: ENV_MAP_WIDTH,
            height: ENV_MAP_HEIGHT,
            depth_or_array_layers: 1,
        },
    );
    let env_map_view = env_map_tex.create_view(&TextureViewDescriptor::default());
    info!("Env map: {ENV_MAP_WIDTH}×{ENV_MAP_HEIGHT}");

    SceneBuffers {
        sphere_buffer,
        bvh_buffer,
        vertex_buffer,
        triangle_buffer,
        mesh_bvh_buffer,
        emissive_buffer,
        env_map_tex,
        env_map_view,
    }
}

// ---------------------------------------------------------------------------
// Texture helpers
// ---------------------------------------------------------------------------

/// Creates an `Rgba32Float` storage texture that can be both read
/// (`TEXTURE_BINDING`) and written (`STORAGE_BINDING`) by compute shaders.
///
/// Used for accumulation ping-pong buffers, the G-buffer, and the
/// denoiser output texture.
pub fn create_storage_texture(
    device: &Device,
    width: u32,
    height: u32,
    label: &str,
) -> (Texture, TextureView) {
    let texture = device.create_texture(&TextureDescriptor {
        label: Some(label),
        size: Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba32Float,
        // COPY_SRC enables GPU → CPU readback for the screenshot feature.
        usage: TextureUsages::TEXTURE_BINDING
            | TextureUsages::STORAGE_BINDING
            | TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&TextureViewDescriptor::default());
    (texture, view)
}

/// Convenience wrapper: creates an accumulation texture with the standard label.
#[inline]
pub fn create_accum_texture(device: &Device, width: u32, height: u32) -> (Texture, TextureView) {
    create_storage_texture(device, width, height, "accum texture")
}

/// Creates the albedo 2D texture array (RGBA8 sRGB, `MAX_TEXTURES` layers).
pub fn create_albedo_texture(device: &Device, queue: &Queue) -> (Texture, TextureView) {
    let (albedo_layers, _) = load_all_textures();
    let albedo_tex = device.create_texture(&TextureDescriptor {
        label: Some("albedo texture array"),
        size: Extent3d {
            width: TEXTURE_SIZE,
            height: TEXTURE_SIZE,
            depth_or_array_layers: MAX_TEXTURES,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8UnormSrgb,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (i, layer) in albedo_layers.iter().enumerate() {
        queue.write_texture(
            TexelCopyTextureInfo {
                texture: &albedo_tex,
                mip_level: 0,
                origin: Origin3d { x: 0, y: 0, z: i as u32 },
                aspect: TextureAspect::All,
            },
            layer,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(TEXTURE_SIZE * 4),
                rows_per_image: Some(TEXTURE_SIZE),
            },
            Extent3d {
                width: TEXTURE_SIZE,
                height: TEXTURE_SIZE,
                depth_or_array_layers: 1,
            },
        );
    }
    let view = albedo_tex.create_view(&TextureViewDescriptor {
        dimension: Some(TextureViewDimension::D2Array),
        ..Default::default()
    });
    (albedo_tex, view)
}

// ---------------------------------------------------------------------------
// Pipeline helper
// ---------------------------------------------------------------------------

/// Create the path-trace compute pipeline from the standard shader.
pub fn create_compute_pipeline(device: &Device, bgl: &BindGroupLayout) -> ComputePipeline {
    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("path trace shader"),
        source: ShaderSource::Wgsl(include_str!("shaders/path_trace.wgsl").into()),
    });
    let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("compute pipeline layout"),
        bind_group_layouts: &[bgl],
        ..Default::default()
    });
    device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("path trace pipeline"),
        layout: Some(&layout),
        module: &shader,
        entry_point: Some("cs_main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

// ---------------------------------------------------------------------------
// Compute bind group helper
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub fn make_compute_bind_groups(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    accum_views: &[wgpu::TextureView; 2],
    camera_buffer: &wgpu::Buffer,
    sphere_buffer: &wgpu::Buffer,
    material_buffer: &wgpu::Buffer,
    bvh_buffer: &wgpu::Buffer,
    vertex_buffer: &wgpu::Buffer,
    triangle_buffer: &wgpu::Buffer,
    mesh_bvh_buffer: &wgpu::Buffer,
    tex_sampler: &wgpu::Sampler,
    albedo_tex_view: &wgpu::TextureView,
    env_map_view: &wgpu::TextureView,
    emissive_buffer: &wgpu::Buffer,
    gbuffer_view: &wgpu::TextureView,
) -> [wgpu::BindGroup; 2] {
    let make = |write_view: &wgpu::TextureView, read_view: &wgpu::TextureView, label: &str| {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &[
                BindGroupEntry {
                    binding: BINDING_ACCUM_WRITE,
                    resource: BindingResource::TextureView(write_view),
                },
                BindGroupEntry {
                    binding: BINDING_CAMERA,
                    resource: camera_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: BINDING_SPHERES,
                    resource: sphere_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: BINDING_ACCUM_READ,
                    resource: BindingResource::TextureView(read_view),
                },
                BindGroupEntry {
                    binding: BINDING_MATERIALS,
                    resource: material_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: BINDING_SPHERE_BVH,
                    resource: bvh_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: BINDING_VERTICES,
                    resource: vertex_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: BINDING_TRIANGLES,
                    resource: triangle_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: BINDING_MESH_BVH,
                    resource: mesh_bvh_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: BINDING_TEX_SAMPLER,
                    resource: BindingResource::Sampler(tex_sampler),
                },
                BindGroupEntry {
                    binding: BINDING_ALBEDO_TEX,
                    resource: BindingResource::TextureView(albedo_tex_view),
                },
                BindGroupEntry {
                    binding: BINDING_ENV_MAP,
                    resource: BindingResource::TextureView(env_map_view),
                },
                BindGroupEntry {
                    binding: BINDING_EMISSIVES,
                    resource: emissive_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: BINDING_GBUFFER,
                    resource: BindingResource::TextureView(gbuffer_view),
                },
            ],
        })
    };
    [
        // compute_bind_groups[i]: write to accum[i], read accum[1-i]
        make(&accum_views[0], &accum_views[1], "compute bg 0"),
        make(&accum_views[1], &accum_views[0], "compute bg 1"),
    ]
}
