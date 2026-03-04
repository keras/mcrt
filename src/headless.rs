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

use bytemuck::bytes_of;
use wgpu::util::DeviceExt;
use wgpu::*;

use crate::camera::{CameraUniform, INIT_LOOK_AT, INIT_LOOK_FROM, compute_camera};
use crate::gpu_layout;
use crate::material::GpuMaterialData;
use crate::scene::load_scene_from_yaml;

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
    let scene = gpu_layout::build_scene_buffers(&device, &queue, &loaded, 2.0);

    // --- GPU buffers -------------------------------------------------------
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
    let (_albedo_tex, albedo_tex_view) = gpu_layout::create_albedo_texture(&device, &queue);

    // Ping-pong accumulation textures: COPY_SRC enables GPU→CPU readback.
    let (accum_tex0, accum_view0) = gpu_layout::create_accum_texture(&device, width, height);
    let (accum_tex1, accum_view1) = gpu_layout::create_accum_texture(&device, width, height);
    let accum_textures = [accum_tex0, accum_tex1];
    let accum_views = [accum_view0, accum_view1];

    // G-buffer texture: path_trace.wgsl always writes to binding 13.
    // In headless mode we never read it; we just need a valid storage target.
    let (_gbuffer_tex, gbuffer_view) =
        gpu_layout::create_storage_texture(&device, width, height, "gbuffer");

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
    let compute_pipeline = gpu_layout::create_compute_pipeline(&device, &compute_bgl);

    // --- compute bind groups (ping-pong) -----------------------------------
    // compute_bind_groups[i]: write to accum[i], read from accum[1-i].
    let compute_bind_groups = gpu_layout::make_compute_bind_groups(
        &device,
        &compute_bgl,
        &accum_views,
        &camera_buffer,
        &scene.sphere_buffer,
        &material_buffer,
        &scene.bvh_buffer,
        &scene.vertex_buffer,
        &scene.triangle_buffer,
        &scene.mesh_bvh_buffer,
        &tex_sampler,
        &albedo_tex_view,
        &scene.env_map_view,
        &scene.emissive_buffer,
        &gbuffer_view,
        &scene.probe_grid_buffer,
        &scene.probe_irradiance_buffer,
    );

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
