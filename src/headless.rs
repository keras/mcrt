// headless.rs — Phase RT-2: offline headless renderer
//
// Renders a scene without a window:
//   1. Creates a wgpu device/queue with no surface.
//   2. Loads the scene YAML and a `.test.toml` sidecar for default parameters.
//   3. Accumulates exactly `spp` samples using the same path-trace compute
//      shader that the interactive renderer uses.
//   4. Reads back the float32 accumulation texture, applies Reinhard tonemapping
//      followed by gamma-2.2 (matching the realtime display shader), and writes
//      an 8-bit sRGB PNG via the `image` crate.
//
// The render is deterministic: setting `camera.frame_count` to `0, 1, …,
// spp-1` on successive dispatches seeds the per-pixel PRNG identically each
// run, guaranteeing byte-identical output for the same (scene, spp) pair.

use std::mem::size_of;
use std::path::Path;

use bytemuck::{Zeroable, bytes_of};
use wgpu::util::DeviceExt;
use wgpu::*;

use crate::bvh::{GpuBvhNode, build_bvh};
use crate::camera::{CameraUniform, INIT_LOOK_AT, INIT_LOOK_FROM, compute_camera};
use crate::gpu_layout::{self, *};
use crate::material::GpuMaterialData;
use crate::mesh::{GpuTriangle, GpuVertex, build_mesh_bvh};
use crate::scene::{GpuSphere, load_scene_from_yaml};
use crate::texture::{
    ENV_MAP_HEIGHT, ENV_MAP_WIDTH, MAX_TEXTURES, TEXTURE_SIZE, load_all_textures, load_env_map_data,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Must match `@workgroup_size(16, 16)` in `path_trace.wgsl`.
const WORKGROUP_SIZE: u32 = 16;

/// Default render dimensions when neither CLI flags nor the sidecar provide values.
const DEFAULT_WIDTH: u32 = 512;
const DEFAULT_HEIGHT: u32 = 512;

/// Default samples-per-pixel when neither CLI nor sidecar provides a value.
const DEFAULT_SPP: u32 = 64;

// ---------------------------------------------------------------------------
// Sidecar parsing
// ---------------------------------------------------------------------------

/// Render parameters from the `[render]` section of a `.test.toml` sidecar.
#[derive(serde::Deserialize, Default)]
struct RenderSection {
    width: Option<u32>,
    height: Option<u32>,
    spp: Option<u32>,
}

/// Top-level structure of a `.test.toml` sidecar file.
///
/// Only the `[render]` table is used by the headless renderer.  The
/// `[regression]` table (thresholds) is parsed but not consumed here;
/// extra keys are silently ignored by serde.
#[derive(serde::Deserialize, Default)]
struct SidecarParams {
    render: Option<RenderSection>,
}

/// Attempt to load and parse the sidecar for `scene_path`.
///
/// `diffuse_sphere.yaml` → looks for `diffuse_sphere.test.toml` in the same
/// directory.  Returns `SidecarParams::default()` on any I/O or parse error
/// so missing sidecars are silently ignored.
fn load_sidecar(scene_path: &str) -> SidecarParams {
    // Strip the final extension (`.yaml`) then add `.test.toml`.
    let base: std::path::PathBuf = Path::new(scene_path).with_extension("");
    let sidecar = base.with_extension("test.toml");
    let Ok(text) = std::fs::read_to_string(&sidecar) else {
        return SidecarParams::default();
    };
    toml::from_str(&text).unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Tonemap
// ---------------------------------------------------------------------------

/// Reinhard tonemap + gamma-2.2, matching the realtime display shader
/// (`display.wgsl`).
///
/// The display shader does:
///   mapped = raw / (raw + 1.0)   // per-channel Reinhard: compresses [0,∞) → [0,1)
///   gamma  = pow(mapped, 1/2.2)  // approximate sRGB transfer
///
/// Applying the same pipeline here ensures headless PNG output is visually
/// identical to what the interactive renderer shows on screen.
#[inline]
fn tonemap_to_u8(v: f32) -> u8 {
    let v = v.max(0.0);
    let mapped = v / (v + 1.0); // Reinhard
    let gamma = mapped.powf(1.0 / 2.2).clamp(0.0, 1.0); // gamma-2.2
    (gamma * 255.0 + 0.5) as u8
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Run the headless offline renderer.
///
/// # Arguments
/// * `scene_path`  — path to the scene YAML file
/// * `output_path` — where to write the output PNG
/// * `width_arg`   — optional width override (beats sidecar, beats default)
/// * `height_arg`  — optional height override
/// * `spp_arg`     — optional SPP override
pub fn run(
    scene_path: String,
    output_path: String,
    width_arg: Option<u32>,
    height_arg: Option<u32>,
    spp_arg: Option<u32>,
) -> Result<(), Box<dyn std::error::Error>> {
    // --- resolve parameters ------------------------------------------------
    let sidecar = load_sidecar(&scene_path);
    let render = sidecar.render.unwrap_or_default();
    let width = width_arg.or(render.width).unwrap_or(DEFAULT_WIDTH).max(1);
    let height = height_arg
        .or(render.height)
        .unwrap_or(DEFAULT_HEIGHT)
        .max(1);
    let spp = spp_arg.or(render.spp).unwrap_or(DEFAULT_SPP).max(1);

    log::info!(
        "Headless render: {scene_path} → {output_path} \
         ({width}×{height}, {spp} spp)"
    );

    // --- wgpu: headless device (no surface) --------------------------------
    let instance = Instance::new(&InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
        power_preference: PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .map_err(|e| format!("no suitable GPU adapter: {e}"))?;

    log::info!("Adapter: {}", adapter.get_info().name);

    let (device, queue) = pollster::block_on(adapter.request_device(&DeviceDescriptor {
        label: Some("headless device"),
        ..Default::default()
    }))?;

    // --- scene loading -----------------------------------------------------
    let loaded = load_scene_from_yaml(&scene_path)?;
    let bvh_result = build_bvh(&loaded.spheres);
    let mesh_result = build_mesh_bvh(&loaded.mesh_vertices, &loaded.mesh_triangles);

    log::info!(
        "BVH: {} spheres → {} nodes; mesh: {} tris → {} nodes",
        bvh_result.ordered_spheres.len(),
        bvh_result.nodes.len(),
        mesh_result.ordered_triangles.len(),
        mesh_result.nodes.len(),
    );

    // Pad empty slices to avoid zero-byte storage bindings (wgpu validation).
    let sphere_data: Vec<GpuSphere> = if bvh_result.ordered_spheres.is_empty() {
        vec![GpuSphere {
            center_r: [1.0e9, 1.0e9, 1.0e9, 0.0],
            mat_and_pad: [0; 4],
        }]
    } else {
        bvh_result.ordered_spheres
    };
    let bvh_node_data: Vec<GpuBvhNode> = if bvh_result.nodes.is_empty() {
        vec![GpuBvhNode::zeroed()]
    } else {
        bvh_result.nodes
    };
    let mesh_vertices_data: Vec<GpuVertex> = if mesh_result.vertices.is_empty() {
        vec![GpuVertex::zeroed()]
    } else {
        mesh_result.vertices
    };
    let mesh_triangles_data: Vec<GpuTriangle> = if mesh_result.ordered_triangles.is_empty() {
        vec![GpuTriangle::zeroed()]
    } else {
        mesh_result.ordered_triangles
    };
    let mesh_bvh_data: Vec<GpuBvhNode> = if mesh_result.nodes.is_empty() {
        vec![GpuBvhNode::zeroed()]
    } else {
        mesh_result.nodes
    };
    let emissive_data: Vec<GpuSphere> = if loaded.emissive_spheres.is_empty() {
        vec![GpuSphere {
            center_r: [1.0e9, 1.0e9, 1.0e9, 0.0],
            mat_and_pad: [0; 4],
        }]
    } else {
        loaded.emissive_spheres
    };

    // --- GPU buffers -------------------------------------------------------
    let sphere_buffer = device.create_buffer_init(&util::BufferInitDescriptor {
        label: Some("spheres"),
        contents: bytemuck::cast_slice(&sphere_data),
        usage: BufferUsages::STORAGE,
    });
    let bvh_buffer = device.create_buffer_init(&util::BufferInitDescriptor {
        label: Some("sphere bvh"),
        contents: bytemuck::cast_slice(&bvh_node_data),
        usage: BufferUsages::STORAGE,
    });
    let vertex_buffer = device.create_buffer_init(&util::BufferInitDescriptor {
        label: Some("vertices"),
        contents: bytemuck::cast_slice(&mesh_vertices_data),
        usage: BufferUsages::STORAGE,
    });
    let triangle_buffer = device.create_buffer_init(&util::BufferInitDescriptor {
        label: Some("triangles"),
        contents: bytemuck::cast_slice(&mesh_triangles_data),
        usage: BufferUsages::STORAGE,
    });
    let mesh_bvh_buffer = device.create_buffer_init(&util::BufferInitDescriptor {
        label: Some("mesh bvh"),
        contents: bytemuck::cast_slice(&mesh_bvh_data),
        usage: BufferUsages::STORAGE,
    });
    let emissive_buffer = device.create_buffer_init(&util::BufferInitDescriptor {
        label: Some("emissives"),
        contents: bytemuck::cast_slice(&emissive_data),
        usage: BufferUsages::STORAGE,
    });
    let material_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("materials"),
        size: size_of::<GpuMaterialData>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&material_buffer, 0, bytes_of(&loaded.materials));

    let camera_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("camera"),
        size: size_of::<CameraUniform>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // --- textures ----------------------------------------------------------
    let (albedo_layers, _) = load_all_textures();
    let albedo_tex = device.create_texture(&TextureDescriptor {
        label: Some("albedo"),
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
                origin: Origin3d {
                    x: 0,
                    y: 0,
                    z: i as u32,
                },
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
    let albedo_tex_view = albedo_tex.create_view(&TextureViewDescriptor {
        dimension: Some(TextureViewDimension::D2Array),
        ..Default::default()
    });

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

    // Ping-pong accumulation textures: COPY_SRC enables GPU→CPU readback.
    let make_accum_tex = |label: &str| {
        let tex = device.create_texture(&TextureDescriptor {
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
            usage: TextureUsages::TEXTURE_BINDING
                | TextureUsages::STORAGE_BINDING
                | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = tex.create_view(&TextureViewDescriptor::default());
        (tex, view)
    };
    let (accum_tex0, accum_view0) = make_accum_tex("accum 0");
    let (accum_tex1, accum_view1) = make_accum_tex("accum 1");
    let accum_textures = [accum_tex0, accum_tex1];
    let accum_views = [accum_view0, accum_view1];

    // G-buffer texture: path_trace.wgsl always writes to binding 13.
    // In headless mode we never read it; we just need a valid storage target.
    let gbuffer_tex = device.create_texture(&TextureDescriptor {
        label: Some("gbuffer"),
        size: Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba32Float,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });
    let gbuffer_view = gbuffer_tex.create_view(&TextureViewDescriptor::default());

    let tex_sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("linear sampler"),
        address_mode_u: AddressMode::Repeat,
        address_mode_v: AddressMode::Repeat,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        ..Default::default()
    });

    // --- compute bind group layout (single source of truth in gpu_layout) ---
    let compute_bgl = gpu_layout::make_compute_bgl(&device);

    // --- compute pipeline --------------------------------------------------
    let compute_shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("path trace shader"),
        source: ShaderSource::Wgsl(include_str!("shaders/path_trace.wgsl").into()),
    });
    let compute_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("compute pipeline layout"),
        bind_group_layouts: &[&compute_bgl],
        ..Default::default()
    });
    let compute_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("path trace pipeline"),
        layout: Some(&compute_pipeline_layout),
        module: &compute_shader,
        entry_point: Some("cs_main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // --- compute bind groups (ping-pong) -----------------------------------
    // compute_bind_groups[i]: write to accum[i], read from accum[1-i].
    // This mirrors the layout in gpu.rs so the first dispatch (frame_count=0)
    // reads from a zero-initialised texture, ignoring prev correctly.
    let make_compute_bg =
        |write_view: &TextureView, read_view: &TextureView, label: &str| -> BindGroup {
            device.create_bind_group(&BindGroupDescriptor {
                label: Some(label),
                layout: &compute_bgl,
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
                        resource: BindingResource::Sampler(&tex_sampler),
                    },
                    BindGroupEntry {
                        binding: BINDING_ALBEDO_TEX,
                        resource: BindingResource::TextureView(&albedo_tex_view),
                    },
                    BindGroupEntry {
                        binding: BINDING_ENV_MAP,
                        resource: BindingResource::TextureView(&env_map_view),
                    },
                    BindGroupEntry {
                        binding: BINDING_EMISSIVES,
                        resource: emissive_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: BINDING_GBUFFER,
                        resource: BindingResource::TextureView(&gbuffer_view),
                    },
                ],
            })
        };

    let compute_bind_groups = [
        make_compute_bg(&accum_views[0], &accum_views[1], "compute bg 0"),
        make_compute_bg(&accum_views[1], &accum_views[0], "compute bg 1"),
    ];

    // --- initial camera uniform --------------------------------------------
    let (look_from, look_at, vfov, aperture, focus_dist) = if let Some(ref c) = loaded.camera {
        let lf = glam::Vec3::from(c.look_from);
        let la = glam::Vec3::from(c.look_at);
        (lf, la, c.vfov, c.aperture, c.focus_dist)
    } else {
        (
            INIT_LOOK_FROM,
            INIT_LOOK_AT,
            crate::camera::DEFAULT_VFOV,
            0.0_f32,
            (INIT_LOOK_FROM - INIT_LOOK_AT).length(),
        )
    };

    // --- SPP accumulation loop ---------------------------------------------
    // Pre-allocate a staging buffer with one CameraUniform per sample.
    // A single command encoder then copies each entry into `camera_buffer`
    // immediately before its compute dispatch.  This collapses `spp` separate
    // CPU–GPU submissions into one, cutting per-frame host overhead and
    // allowing the GPU to pipeline dispatches without round-tripping to the CPU.
    //
    // wgpu's resource tracker inserts the required TRANSFER_WRITE → SHADER_READ
    // barrier between each copy_buffer_to_buffer and the following compute pass
    // so each dispatch reads the correct camera_buffer contents.
    //
    // Implementation note: the design is intentionally a flat `run()` function
    // rather than a `HeadlessRenderer` struct (as the roadmap suggest) to keep
    // the code self-contained for Phase RT-2.  A struct refactor is deferred
    // to a later phase if shared-device tests or progressive callbacks are needed.
    let cam_stride = size_of::<CameraUniform>() as u64;
    let cameras: Vec<CameraUniform> = (0..spp)
        .map(|frame| {
            let mut cam = compute_camera(
                width, height, look_from, look_at, vfov, aperture, focus_dist,
            );
            cam.frame_count = frame;
            cam
        })
        .collect();
    let cam_staging = device.create_buffer_init(&util::BufferInitDescriptor {
        label: Some("camera staging"),
        contents: bytemuck::cast_slice(&cameras),
        usage: BufferUsages::COPY_SRC,
    });

    let wg_x = width.div_ceil(WORKGROUP_SIZE);
    let wg_y = height.div_ceil(WORKGROUP_SIZE);

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("headless accumulation encoder"),
    });
    for frame in 0..spp {
        let f_idx = (frame % 2) as usize;
        // Copy this frame's camera uniform into the active camera_buffer slot.
        encoder.copy_buffer_to_buffer(
            &cam_staging,
            frame as u64 * cam_stride,
            &camera_buffer,
            0,
            cam_stride,
        );
        // Dispatch path tracer (reads updated camera_buffer via binding 1).
        let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("path trace pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &compute_bind_groups[f_idx], &[]);
        cpass.dispatch_workgroups(wg_x, wg_y, 1);
    }
    queue.submit([encoder.finish()]);
    log::info!("{spp}/{spp} samples accumulated");

    // --- texture readback --------------------------------------------------
    // After `spp` dispatches the last write target is accum[(spp-1) % 2].
    let last_idx = ((spp - 1) % 2) as usize;
    let src_tex = &accum_textures[last_idx];

    // Row stride must be a multiple of 256 bytes for copy_texture_to_buffer.
    // Each Rgba32Float pixel = 16 bytes.
    let bytes_per_pixel: u32 = 16;
    let unpadded_row = width * bytes_per_pixel;
    let padded_row = unpadded_row.div_ceil(256) * 256;
    let buf_size = (padded_row * height) as u64;

    let staging = device.create_buffer(&BufferDescriptor {
        label: Some("readback staging"),
        size: buf_size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("readback encoder"),
    });
    enc.copy_texture_to_buffer(
        TexelCopyTextureInfo {
            texture: src_tex,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_row),
                rows_per_image: Some(height),
            },
        },
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([enc.finish()]);

    // Map the staging buffer synchronously and read the pixels.
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    rx.recv()??;

    let raw = slice.get_mapped_range();
    let mut pixels: Vec<u8> = Vec::with_capacity((width * height * 4) as usize);
    for row in 0..height as usize {
        let row_off = row * padded_row as usize;
        for col in 0..width as usize {
            let off = row_off + col * 16;
            let r = f32::from_le_bytes(raw[off..off + 4].try_into().unwrap());
            let g = f32::from_le_bytes(raw[off + 4..off + 8].try_into().unwrap());
            let b = f32::from_le_bytes(raw[off + 8..off + 12].try_into().unwrap());
            // Sanitise non-finite values (fireflies from rare bad paths).
            let r = if r.is_finite() { r } else { 0.0 };
            let g = if g.is_finite() { g } else { 0.0 };
            let b = if b.is_finite() { b } else { 0.0 };
            pixels.push(tonemap_to_u8(r));
            pixels.push(tonemap_to_u8(g));
            pixels.push(tonemap_to_u8(b));
            pixels.push(255);
        }
    }
    drop(raw);
    staging.unmap();

    // --- write PNG ---------------------------------------------------------
    // Ensure the output directory exists.
    if let Some(parent) = Path::new(&output_path).parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }

    let img = image::RgbaImage::from_raw(width, height, pixels)
        .ok_or("failed to construct image buffer")?;
    img.save(&output_path)?;

    log::info!("PNG written → '{output_path}'");
    Ok(())
}
