// gpu.rs — GPU device, pipelines, and the main render/resize loop
//
// Owns all wgpu resources, the orbit camera state, and the thin-lens
// path-tracing render loop.  Input handling is exposed via methods so
// that `app.rs` never touches internal fields directly.

use std::mem::size_of;
use std::sync::Arc;

use bytemuck::bytes_of;
use glam::Vec3;
use log::info;
use wgpu::util::DeviceExt;
use wgpu::{
    AddressMode, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor, DeviceDescriptor,
    Extent3d, FilterMode, FragmentState, InstanceDescriptor, LoadOp, MultisampleState, Operations,
    Origin3d, PipelineLayoutDescriptor, PowerPreference, PrimitiveState, RenderPassColorAttachment,
    RenderPassDescriptor, RenderPipelineDescriptor, RequestAdapterOptions, SamplerBindingType,
    SamplerDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages, StorageTextureAccess,
    StoreOp, SurfaceError, TexelCopyBufferLayout, TexelCopyTextureInfo, TextureAspect,
    TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType, TextureUsages,
    TextureViewDescriptor, TextureViewDimension, VertexState,
};
use winit::{
    dpi::PhysicalPosition,
    event::{ElementState, MouseButton, MouseScrollDelta},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

// Phase 15: egui immediate-mode GUI.
use egui_wgpu::ScreenDescriptor;

use crate::bvh::{GpuBvhNode, build_bvh};
use crate::camera::{CameraUniform, DEFAULT_VFOV, INIT_LOOK_AT, INIT_LOOK_FROM, compute_camera};
use crate::material::GpuMaterialData;
use crate::mesh::{GpuTriangle, GpuVertex, build_mesh_bvh};
use crate::scene::{GpuSphere, load_scene_from_yaml};
use crate::texture::{
    ENV_MAP_HEIGHT, ENV_MAP_WIDTH, MAX_TEXTURES, TEXTURE_SIZE, load_all_textures, load_env_map_data,
};
use notify::Watcher as _;

// ---------------------------------------------------------------------------
// Render / input constants
// ---------------------------------------------------------------------------

/// Workgroup tile size; must match `@workgroup_size(16, 16)` in path_trace.wgsl.
const WORKGROUP_SIZE: u32 = 16;

/// Camera exponential-smoothing factor per frame.
/// At 60 fps, error shrinks to < 1 % in ~30 frames (~0.5 s).
const CAM_LERP: f32 = 0.15;

/// Squared movement threshold for accumulation reset.
/// Below this the camera is considered stationary.
const CAM_MOVE_EPSILON_SQ: f32 = 1e-12;

/// Mouse drag sensitivity: radians per pixel.
const MOUSE_SENSITIVITY: f32 = 0.005;

/// Maximum camera pitch angle (±85°) in radians.
const MAX_PITCH_RAD: f32 = 1.483_529_86_f32; // 85° × π/180

/// Scroll zoom speed multiplier.
const SCROLL_SPEED: f32 = 0.3;

/// Minimum orbital distance in world units.
const MIN_ZOOM: f32 = 0.3;

/// Maximum orbital distance in world units.
const MAX_ZOOM: f32 = 50.0;

/// WASD pan speed: world units moved per event-loop tick.
const PAN_SPEED: f32 = 0.04;

/// Default window width in pixels.
pub const DEFAULT_WINDOW_WIDTH: u32 = 1280;

/// Default window height in pixels.
pub const DEFAULT_WINDOW_HEIGHT: u32 = 720;

// ---------------------------------------------------------------------------
// CameraState — orbit + DOF parameters
// ---------------------------------------------------------------------------

/// All camera orbit and lens parameters, both the smoothed current values and
/// the raw input targets that the current values lerp toward each frame.
pub struct CameraState {
    // Current (smoothed) values written to the GPU each frame.
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub look_at: Vec3,
    pub vfov: f32,
    pub aperture: f32,
    pub focus_dist: f32,

    // Raw input targets — the values yaw/pitch/… are interpolating toward.
    pub tgt_yaw: f32,
    pub tgt_pitch: f32,
    pub tgt_distance: f32,
    pub tgt_look_at: Vec3,
}

impl CameraState {
    /// Initialise from the look-from / look-at pair used in Phase 6.
    pub fn from_initial_position(look_from: Vec3, look_at: Vec3) -> Self {
        let delta = look_from - look_at;
        let distance = delta.length();
        let pitch = (delta.y / distance).asin();
        let yaw = f32::atan2(delta.x, delta.z);

        Self {
            yaw,
            pitch,
            distance,
            look_at,
            vfov: DEFAULT_VFOV,
            aperture: 0.0,
            focus_dist: distance,
            tgt_yaw: yaw,
            tgt_pitch: pitch,
            tgt_distance: distance,
            tgt_look_at: look_at,
        }
    }

    /// Eye position in world space derived from the current orbit parameters.
    pub fn look_from(&self) -> Vec3 {
        let cos_p = self.pitch.cos();
        self.look_at
            + Vec3::new(
                self.distance * self.yaw.sin() * cos_p,
                self.distance * self.pitch.sin(),
                self.distance * self.yaw.cos() * cos_p,
            )
    }
}

// ---------------------------------------------------------------------------
// InputState — mouse drag + WASD
// ---------------------------------------------------------------------------

/// Raw input state consumed by the event handlers.
pub struct InputState {
    pub drag_active: bool,
    pub last_cursor: Option<PhysicalPosition<f64>>,
    /// Held WASD keys: [W, A, S, D].
    pub wasd_held: [bool; 4],
}

impl Default for InputState {
    fn default() -> Self {
        Self {
            drag_active: false,
            last_cursor: None,
            wasd_held: [false; 4],
        }
    }
}

// ---------------------------------------------------------------------------
// Texture helper
// ---------------------------------------------------------------------------

/// Creates an `Rgba32Float` storage texture that can be both read
/// (`TEXTURE_BINDING`) and written (`STORAGE_BINDING`) by compute shaders.
///
/// Used for accumulation ping-pong buffers, the G-buffer, and the
/// denoiser output texture.
fn create_storage_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    label: &str,
) -> (wgpu::Texture, wgpu::TextureView) {
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
        // COPY_SRC enables GPU → CPU readback for the screenshot feature (Phase 15).
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
fn create_accum_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    create_storage_texture(device, width, height, "accum texture")
}

// ---------------------------------------------------------------------------
// GpuState — all GPU resources + camera + input sub-structs
// ---------------------------------------------------------------------------

/// All GPU resources, camera state, and input state for one window session.
///
/// Created inside `App::resumed()` and lives until the application exits.
/// Fields are kept private; all externally observable behaviour is exposed
/// through methods to preserve encapsulation.
pub struct GpuState {
    // `surface` must be declared before `window` — Rust drops field in
    // declaration order and Surface must outlive the raw window handle.
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    window: Arc<Window>,

    // Ping-pong HDR accumulation textures.
    accum_textures: [wgpu::Texture; 2],
    accum_views: [wgpu::TextureView; 2],

    // Display (tone-map) pass.
    bind_group_layout: wgpu::BindGroupLayout,
    display_bind_groups: [wgpu::BindGroup; 2],
    render_pipeline: wgpu::RenderPipeline,

    // Compute (path trace) pass.
    camera_buffer: wgpu::Buffer,
    compute_bind_group_layout: wgpu::BindGroupLayout,
    compute_bind_groups: [wgpu::BindGroup; 2],
    compute_pipeline: wgpu::ComputePipeline,
    /// BVH-reordered sphere list (storage buffer, binding 2).
    sphere_buffer: wgpu::Buffer,
    material_buffer: wgpu::Buffer,
    /// Flat sphere BVH node array (storage buffer, binding 5).
    bvh_buffer: wgpu::Buffer,
    // ---- Phase 10: triangle mesh resources --------------------------------
    /// Vertex positions / normals / UVs (storage buffer, binding 6).
    vertex_buffer: wgpu::Buffer,
    /// BVH-reordered triangle index triples (storage buffer, binding 7).
    triangle_buffer: wgpu::Buffer,
    /// Flat mesh BVH node array (storage buffer, binding 8).
    mesh_bvh_buffer: wgpu::Buffer,
    // ---- Phase 13: emissive sphere list ----------------------------------
    /// Emissive sphere list for NEE direct-light sampling (storage buffer, binding 12).
    /// Contains a stub sphere at [1e9,1e9,1e9] r=0 when there are no emissives.
    emissive_buffer: wgpu::Buffer,
    // ---- Phase 14: G-buffer and denoiser pass ----------------------------
    /// Primary-hit G-buffer: (.xyz = world-space normal, .w = linear depth).
    /// Written every frame by path_trace binding 13; read by the denoiser.
    /// Kept to prevent texture deallocation; accessed only through gbuffer_view.
    _gbuffer_tex: wgpu::Texture,
    gbuffer_view: wgpu::TextureView,
    /// Denoiser output texture.  Written by the denoiser pass; displayed in
    /// place of the raw accumulation texture when `denoise_enabled` is true.
    /// Kept to prevent texture deallocation; accessed only through denoise_output_view.
    _denoise_output_tex: wgpu::Texture,
    denoise_output_view: wgpu::TextureView,
    /// Bind group layout shared by both denoiser bind groups.
    denoise_bind_group_layout: wgpu::BindGroupLayout,
    /// Two denoiser bind groups.
    /// denoise_bind_groups[i] reads accum[i] — the slice *just written* by
    /// compute_bind_groups[i] — plus the shared G-buffer, and writes the
    /// denoised result to denoise_output_view.
    denoise_bind_groups: [wgpu::BindGroup; 2],
    /// The joint-bilateral denoiser compute pipeline.
    denoise_pipeline: wgpu::ComputePipeline,
    /// Display bind group that reads from the denoised output texture.
    /// Replaces `display_bind_groups[f_idx]` when `denoise_enabled` is true.
    display_bg_denoised: wgpu::BindGroup,
    /// When `true` the denoiser compute pass runs before the display pass.
    /// Toggle at runtime with the 'N' key.
    denoise_enabled: bool,
    // ---- Phase 11: texture resources ----------------------------------------
    /// Albedo texture 2D array (RGBA8 Unorm, MAX_TEXTURES layers; binding 10).
    /// Stored alongside the view so the Texture Arc is not dropped early.
    _albedo_tex: wgpu::Texture,
    albedo_tex_view: wgpu::TextureView,
    /// HDR equirectangular environment map (RGBA32 Float; binding 11).
    _env_map_tex: wgpu::Texture,
    env_map_view: wgpu::TextureView,
    /// Linear sampler shared by the albedo texture array (binding 9).
    tex_sampler: wgpu::Sampler,

    /// Monotonically increasing frame index; resets to 0 on camera move / resize.
    frame_count: u32,

    // Sub-structs for non-GPU state.
    camera: CameraState,
    input: InputState,
    // ---- Phase 12: hot-reload ------------------------------------------
    /// Path of the currently loaded scene file (for hot-reload).
    scene_path: String,
    /// File-system watcher that fires when the scene file changes on disk.
    /// Owned here only to keep the OS registration alive; never read after
    /// construction.
    _scene_watcher: Option<notify::RecommendedWatcher>,
    /// Receives events from `_scene_watcher`; polled with `try_recv` each frame.
    scene_change_rx: Option<std::sync::mpsc::Receiver<notify::Result<notify::Event>>>,

    // ---- Phase 15: egui UI -----------------------------------------------
    /// egui context — persistent across frames (holds font atlas, style, etc.).
    egui_ctx: egui::Context,
    /// egui ↔ winit platform bridge: translates OS events into egui input.
    egui_state: egui_winit::State,
    /// wgpu-side renderer for egui tessellated geometry.
    egui_renderer: egui_wgpu::Renderer,
    /// When `true` the egui side-panel is visible.  Toggle with the 'H' key.
    show_ui: bool,
    /// CPU copy of the material table — mutated by the UI and re-uploaded on change.
    material_data_cpu: GpuMaterialData,
    /// Human-readable material names (palette YAML keys; inline materials get "inline-N").
    /// Same index order as `material_data_cpu.materials`.
    material_names: Vec<String>,
    /// Timestamp of the previous frame — used for FPS estimation.
    last_frame_time: std::time::Instant,
    /// Exponentially-smoothed frames-per-second estimate (α = 0.05 per frame).
    fps_smooth: f32,
}

impl GpuState {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    pub fn new(window: Arc<Window>, scene_path: String) -> Self {
        let instance = wgpu::Instance::new(&InstanceDescriptor::default());

        let surface = instance
            .create_surface(Arc::clone(&window))
            .expect("failed to create surface");

        let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .expect("no suitable GPU adapter found");

        info!("Adapter: {}", adapter.get_info().name);

        let (device, queue) = pollster::block_on(adapter.request_device(&DeviceDescriptor {
            label: Some("mcrt device"),
            ..Default::default()
        }))
        .expect("failed to create device");

        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);
        let mut config = surface
            .get_default_config(&adapter, width, height)
            .expect("surface not supported by adapter");

        // Lock to vsync; swapchain drives at display refresh rate.
        config.present_mode = wgpu::PresentMode::Fifo;
        surface.configure(&device, &config);

        // ---- accumulation textures ----------------------------------------
        let (accum_tex0, accum_view0) = create_accum_texture(&device, config.width, config.height);
        let (accum_tex1, accum_view1) = create_accum_texture(&device, config.width, config.height);
        let accum_textures = [accum_tex0, accum_tex1];
        let accum_views = [accum_view0, accum_view1];

        // ---- display bind group layout ------------------------------------
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("display bgl"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: false },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });

        // ---- display shader + pipeline ------------------------------------
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("display shader"),
            source: ShaderSource::Wgsl(include_str!("shaders/display.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("display pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            ..Default::default()
        });
        let render_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("display pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // ---- camera buffer -------------------------------------------------
        let camera_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("camera buffer"),
            size: size_of::<CameraUniform>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ---- sphere list + BVH --------------------------------------------
        // Load the declarative scene from YAML, then build the BVH on the CPU
        // and upload both the reordered sphere list and the flat node array as
        // storage buffers.  The loaded scene also carries the material table,
        // the mesh geometry, and optional camera settings.
        let loaded = load_scene_from_yaml(&scene_path)
            .expect("failed to load initial scene");
        let bvh_result = build_bvh(&loaded.spheres);

        // wgpu requires every storage-buffer binding to be at least as large as
        // the stride declared in the shader (32 bytes = one GpuSphere).  When
        // the scene has no spheres the buffer would be zero bytes, causing a
        // validation error.  Pad with a stub sphere placed far outside the
        // scene so BVH traversal never hits it.
        let sphere_data: Vec<GpuSphere> = if bvh_result.ordered_spheres.is_empty() {
            vec![GpuSphere {
                center_r: [1.0e9, 1.0e9, 1.0e9, 0.0],
                mat_and_pad: [0; 4],
            }]
        } else {
            bvh_result.ordered_spheres.clone()
        };
        let sphere_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sphere storage"),
            contents: bytemuck::cast_slice(&sphere_data),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        // Sphere BVH nodes: also needs at least one entry when there are no
        // spheres (the shader indexes this array unconditionally during traversal).
        let bvh_node_data: Vec<GpuBvhNode> = if bvh_result.nodes.is_empty() {
            use bytemuck::Zeroable;
            vec![GpuBvhNode::zeroed()]
        } else {
            bvh_result.nodes.clone()
        };
        let bvh_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bvh node storage"),
            contents: bytemuck::cast_slice(&bvh_node_data),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        info!(
            "BVH built: {} spheres → {} nodes",
            bvh_result.ordered_spheres.len(),
            bvh_result.nodes.len()
        );

        // ---- triangle mesh (Phase 12: data-driven from YAML scene file) ----
        // Mesh geometry is fully declarative: `load_scene_from_yaml` above has
        // already parsed all `type: mesh` objects and merged their vertices and
        // triangles.  We just build the BVH here.
        let mesh_result = build_mesh_bvh(&loaded.mesh_vertices, &loaded.mesh_triangles);

        info!(
            "Mesh BVH built: {} triangles, {} vertices → {} nodes",
            mesh_result.ordered_triangles.len(),
            mesh_result.vertices.len(),
            mesh_result.nodes.len()
        );

        // ---- Phase 11: textures + environment map --------------------------
        let (albedo_layers, _) = load_all_textures();
        let env_map_data = load_env_map_data(loaded.env_map_path.as_deref());

        // RGBA8 Unorm 2D texture array — MAX_TEXTURES layers, TEXTURE_SIZE².
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
            format: TextureFormat::Rgba8UnormSrgb, // sRGB→linear decode on fetch
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Upload each layer by offsetting the z-origin (array slice index).
        for (layer_idx, layer_data) in albedo_layers.iter().enumerate() {
            queue.write_texture(
                TexelCopyTextureInfo {
                    texture: &albedo_tex,
                    mip_level: 0,
                    origin: Origin3d {
                        x: 0,
                        y: 0,
                        z: layer_idx as u32,
                    },
                    aspect: TextureAspect::All,
                },
                layer_data,
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

        info!("Albedo texture array: {MAX_TEXTURES} layers @ {TEXTURE_SIZE}×{TEXTURE_SIZE}");

        // RGBA32 Float equirectangular environment map.
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

        // ---- Phase 14: G-buffer + denoiser output textures ---------------
        // Both need TEXTURE_BINDING (read by denoise/display) and
        // STORAGE_BINDING (written by path_trace / denoise compute passes).
        let (_gbuffer_tex, gbuffer_view) =
            create_storage_texture(&device, config.width, config.height, "gbuffer");
        let (_denoise_output_tex, denoise_output_view) =
            create_storage_texture(&device, config.width, config.height, "denoise output");

        // Linear/clamp sampler for the albedo texture array.
        let tex_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("linear sampler"),
            address_mode_u: AddressMode::Repeat, // wrap for sphere/mesh UV
            address_mode_v: AddressMode::Repeat,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..Default::default()
        });

        // Mesh buffers: guard against zero-byte uploads (no-mesh scene).
        // wgpu requires every bound buffer to be at least as large as the
        // stride declared in the shader via min_binding_size.
        let mesh_vertices_data: Vec<GpuVertex> = if mesh_result.vertices.is_empty() {
            use bytemuck::Zeroable;
            vec![GpuVertex::zeroed()]
        } else {
            mesh_result.vertices.clone()
        };
        let mesh_triangles_data: Vec<GpuTriangle> = if mesh_result.ordered_triangles.is_empty() {
            use bytemuck::Zeroable;
            vec![GpuTriangle::zeroed()]
        } else {
            mesh_result.ordered_triangles.clone()
        };
        let mesh_bvh_data: Vec<GpuBvhNode> = if mesh_result.nodes.is_empty() {
            use bytemuck::Zeroable;
            vec![GpuBvhNode::zeroed()]
        } else {
            mesh_result.nodes.clone()
        };
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh vertex storage"),
            contents: bytemuck::cast_slice(&mesh_vertices_data),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        let triangle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh triangle storage"),
            contents: bytemuck::cast_slice(&mesh_triangles_data),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        let mesh_bvh_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh bvh node storage"),
            contents: bytemuck::cast_slice(&mesh_bvh_data),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        // ---- Phase 13: emissive sphere buffer (binding 12) ---------------
        // Always non-empty: pad with a stub sphere far away so the shader
        // can unconditionally index the array (no out-of-bounds).
        let emissive_data: Vec<GpuSphere> = if loaded.emissive_spheres.is_empty() {
            vec![GpuSphere {
                center_r: [1.0e9, 1.0e9, 1.0e9, 0.0],
                mat_and_pad: [0; 4],
            }]
        } else {
            loaded.emissive_spheres.clone()
        };
        let emissive_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("emissive sphere storage"),
            contents: bytemuck::cast_slice(&emissive_data),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        // ---- material buffer -----------------------------------------------
        // Use the material table parsed from the YAML scene file.
        let material_data = loaded.materials;
        // Keep a CPU copy for the Phase 15 UI material editor.
        let material_data_cpu = material_data;
        let material_names = loaded.material_names;
        let material_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("material buffer"),
            size: size_of::<GpuMaterialData>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&material_buffer, 0, bytes_of(&material_data));

        // ---- compute bind group layout ------------------------------------
        let compute_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("compute bgl"),
                entries: &[
                    // 0: write-only storage texture (accum write target)
                    BindGroupLayoutEntry {
                        binding:    0,
                        visibility: ShaderStages::COMPUTE,
                        ty:         BindingType::StorageTexture {
                            access:         StorageTextureAccess::WriteOnly,
                            format:         TextureFormat::Rgba32Float,
                            view_dimension: TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // 1: camera parameters uniform
                    BindGroupLayoutEntry {
                        binding:    1,
                        visibility: ShaderStages::COMPUTE,
                        ty:         BindingType::Buffer {
                            ty:                wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size:   wgpu::BufferSize::new(
                                size_of::<CameraUniform>() as u64
                            ),
                        },
                        count: None,
                    },
                    // 2: sphere list storage buffer (BVH-reordered; runtime-sized array)
                    BindGroupLayoutEntry {
                        binding:    2,
                        visibility: ShaderStages::COMPUTE,
                        ty:         BindingType::Buffer {
                            ty:                wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size:   wgpu::BufferSize::new(
                                size_of::<GpuSphere>() as u64
                            ),
                        },
                        count: None,
                    },
                    // 3: previous-frame accumulation (read via textureLoad)
                    BindGroupLayoutEntry {
                        binding:    3,
                        visibility: ShaderStages::COMPUTE,
                        ty:         BindingType::Texture {
                            sample_type:    TextureSampleType::Float { filterable: false },
                            view_dimension: TextureViewDimension::D2,
                            multisampled:   false,
                        },
                        count: None,
                    },
                    // 4: material descriptors uniform
                    BindGroupLayoutEntry {
                        binding:    4,
                        visibility: ShaderStages::COMPUTE,
                        ty:         BindingType::Buffer {
                            ty:                wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size:   wgpu::BufferSize::new(
                                size_of::<GpuMaterialData>() as u64
                            ),
                        },
                        count: None,
                    },
                    // 5: sphere BVH node array storage buffer (runtime-sized array)
                    BindGroupLayoutEntry {
                        binding:    5,
                        visibility: ShaderStages::COMPUTE,
                        ty:         BindingType::Buffer {
                            ty:                wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size:   wgpu::BufferSize::new(
                                size_of::<GpuBvhNode>() as u64
                            ),
                        },
                        count: None,
                    },
                    // 6: mesh vertex buffer (position, normal, UV; runtime-sized array)
                    BindGroupLayoutEntry {
                        binding:    6,
                        visibility: ShaderStages::COMPUTE,
                        ty:         BindingType::Buffer {
                            ty:                wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size:   wgpu::BufferSize::new(
                                size_of::<GpuVertex>() as u64
                            ),
                        },
                        count: None,
                    },
                    // 7: mesh triangle index buffer (runtime-sized array)
                    BindGroupLayoutEntry {
                        binding:    7,
                        visibility: ShaderStages::COMPUTE,
                        ty:         BindingType::Buffer {
                            ty:                wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size:   wgpu::BufferSize::new(
                                size_of::<GpuTriangle>() as u64
                            ),
                        },
                        count: None,
                    },
                    // 8: mesh BVH node array (runtime-sized array)
                    BindGroupLayoutEntry {
                        binding:    8,
                        visibility: ShaderStages::COMPUTE,
                        ty:         BindingType::Buffer {
                            ty:                wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size:   wgpu::BufferSize::new(
                                size_of::<GpuBvhNode>() as u64
                            ),
                        },
                        count: None,
                    },
                    // 9: linear sampler for albedo texture array
                    BindGroupLayoutEntry {
                        binding: 9,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Sampler(SamplerBindingType::Filtering),
                        count: None,
                    },
                    // 10: albedo 2D texture array (RGBA8 Unorm, MAX_TEXTURES layers)
                    BindGroupLayoutEntry {
                        binding: 10,
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
                        binding: 11,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: false },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // 12: emissive sphere list for NEE (Phase 13)
                    BindGroupLayoutEntry {
                        binding:    12,
                        visibility: ShaderStages::COMPUTE,
                        ty:         BindingType::Buffer {
                            ty:                wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size:   wgpu::BufferSize::new(
                                size_of::<GpuSphere>() as u64
                            ),
                        },
                        count: None,
                    },
                    // 13: G-buffer write target (Phase 14) — normal.xyz + depth.w
                    BindGroupLayoutEntry {
                        binding:    13,
                        visibility: ShaderStages::COMPUTE,
                        ty:         BindingType::StorageTexture {
                            access:         StorageTextureAccess::WriteOnly,
                            format:         TextureFormat::Rgba32Float,
                            view_dimension: TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });

        // ---- compute shader + pipeline ------------------------------------
        let compute_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("path trace shader"),
            source: ShaderSource::Wgsl(include_str!("shaders/path_trace.wgsl").into()),
        });
        let compute_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("compute pipeline layout"),
            bind_group_layouts: &[&compute_bind_group_layout],
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

        // ---- Phase 14: denoiser pipeline ----------------------------------
        let denoise_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("denoise bgl"),
                entries: &[
                    // 0: accum texture (HDR radiance, non-filterable Rgba32Float)
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: false },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // 1: G-buffer (normal + depth, non-filterable Rgba32Float)
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: false },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // 2: denoised output (write-only storage texture)
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::StorageTexture {
                            access: StorageTextureAccess::WriteOnly,
                            format: TextureFormat::Rgba32Float,
                            view_dimension: TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });
        let denoise_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("denoise shader"),
            source: ShaderSource::Wgsl(include_str!("shaders/denoise.wgsl").into()),
        });
        let denoise_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("denoise pipeline layout"),
            bind_group_layouts: &[&denoise_bind_group_layout],
            ..Default::default()
        });
        let denoise_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("denoise pipeline"),
            layout: Some(&denoise_pipeline_layout),
            module: &denoise_shader,
            entry_point: Some("cs_denoise"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ---- initial camera state -----------------------------------------
        // Use camera settings from the YAML scene file when present; otherwise
        // fall back to the built-in orbit-camera defaults.
        let camera = if let Some(ref c) = loaded.camera {
            let look_from = glam::Vec3::from(c.look_from);
            let look_at = glam::Vec3::from(c.look_at);
            let mut cs = CameraState::from_initial_position(look_from, look_at);
            cs.vfov = c.vfov;
            cs.aperture = c.aperture;
            cs.focus_dist = c.focus_dist;
            // Keep tgt_* in sync so the initial lerp is a no-op.
            cs.tgt_yaw = cs.yaw;
            cs.tgt_pitch = cs.pitch;
            cs.tgt_distance = cs.distance;
            cs.tgt_look_at = cs.look_at;
            cs
        } else {
            CameraState::from_initial_position(INIT_LOOK_FROM, INIT_LOOK_AT)
        };

        // ---- Phase 12: scene file watcher for hot-reload ------------------
        let scene_path_str = scene_path.as_str();
        let (sc_tx, sc_rx) = std::sync::mpsc::channel::<notify::Result<notify::Event>>();
        let _scene_watcher: Option<notify::RecommendedWatcher> =
            match notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
                let _ = sc_tx.send(res);
            }) {
                Ok(mut w) => {
                    if let Err(e) = w.watch(
                        std::path::Path::new(scene_path_str),
                        notify::RecursiveMode::NonRecursive,
                    ) {
                        log::warn!("scene hot-reload disabled (watch failed): {e}");
                        None
                    } else {
                        info!("watching '{}' for hot-reload", scene_path_str);
                        Some(w)
                    }
                }
                Err(e) => {
                    log::warn!("scene hot-reload disabled (watcher creation failed): {e}");
                    None
                }
            };
        let scene_change_rx = if _scene_watcher.is_some() {
            Some(sc_rx)
        } else {
            None
        };

        // ---- build bind groups from the already-created resources ---------
        // Use a temporary partial struct so rebuild_bind_groups can be called
        // as a regular (non-self) function before we finish constructing Self.
        let display_bind_groups =
            make_display_bind_groups(&device, &bind_group_layout, &accum_views);
        let compute_bind_groups = make_compute_bind_groups(
            &device,
            &compute_bind_group_layout,
            &accum_views,
            &camera_buffer,
            &sphere_buffer,
            &material_buffer,
            &bvh_buffer,
            &vertex_buffer,
            &triangle_buffer,
            &mesh_bvh_buffer,
            &tex_sampler,
            &albedo_tex_view,
            &env_map_view,
            &emissive_buffer,
            &gbuffer_view,
        );
        let denoise_bind_groups = make_denoise_bind_groups(
            &device,
            &denoise_bind_group_layout,
            &accum_views,
            &gbuffer_view,
            &denoise_output_view,
        );
        let display_bg_denoised =
            make_display_bg_denoised(&device, &bind_group_layout, &denoise_output_view);

        // ---- Phase 15: egui initialisation --------------------------------
        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &*window,
            Some(window.scale_factor() as f32),
            None, // theme — None = use egui default
            None, // max_texture_side — None = use device limit
        );
        let egui_renderer = egui_wgpu::Renderer::new(
            &device,
            config.format,
            egui_wgpu::RendererOptions::default(),
        );

        Self {
            surface,
            device,
            queue,
            config,
            window,
            accum_textures,
            accum_views,
            bind_group_layout,
            display_bind_groups,
            render_pipeline,
            camera_buffer,
            compute_bind_group_layout,
            compute_bind_groups,
            compute_pipeline,
            sphere_buffer,
            material_buffer,
            bvh_buffer,
            vertex_buffer,
            triangle_buffer,
            mesh_bvh_buffer,
            emissive_buffer,
            _gbuffer_tex,
            gbuffer_view,
            _denoise_output_tex,
            denoise_output_view,
            denoise_bind_group_layout,
            denoise_bind_groups,
            denoise_pipeline,
            display_bg_denoised,
            denoise_enabled: false,
            _albedo_tex: albedo_tex,
            albedo_tex_view,
            _env_map_tex: env_map_tex,
            env_map_view,
            tex_sampler,
            frame_count: 0,
            camera,
            input: InputState::default(),
            scene_path: scene_path_str.to_string(),
            _scene_watcher,
            scene_change_rx,
            egui_ctx,
            egui_state,
            egui_renderer,
            show_ui: true,
            material_data_cpu,
            material_names,
            last_frame_time: std::time::Instant::now(),
            fps_smooth: 60.0,
        }
    }

    // -----------------------------------------------------------------------
    // Resize
    // -----------------------------------------------------------------------

    /// Handle a surface resize: recreates textures and bind groups, resets accumulation.
    pub fn resize(&mut self, new_width: u32, new_height: u32) {
        if new_width == 0 || new_height == 0 {
            return; // wgpu panics on zero-size surfaces (e.g. when minimised)
        }
        self.config.width = new_width;
        self.config.height = new_height;
        self.surface.configure(&self.device, &self.config);

        let (tex0, view0) = create_accum_texture(&self.device, new_width, new_height);
        let (tex1, view1) = create_accum_texture(&self.device, new_width, new_height);
        self.accum_textures = [tex0, tex1];
        self.accum_views = [view0, view1];

        // Recreate G-buffer and denoiser output to match the new resolution.
        let (gb_tex, gb_view) =
            create_storage_texture(&self.device, new_width, new_height, "gbuffer");
        let (dn_tex, dn_view) =
            create_storage_texture(&self.device, new_width, new_height, "denoise output");
        self._gbuffer_tex = gb_tex;
        self.gbuffer_view = gb_view;
        self._denoise_output_tex = dn_tex;
        self.denoise_output_view = dn_view;

        self.rebuild_bind_groups();
        self.frame_count = 0;
    }

    // -----------------------------------------------------------------------
    // Bind group reconstruction (called by both new→resize and resize)
    // -----------------------------------------------------------------------

    /// Recreate both display and compute bind groups to reference the current
    /// `accum_views`.  Must be called whenever `accum_views` are replaced.
    fn rebuild_bind_groups(&mut self) {
        self.display_bind_groups =
            make_display_bind_groups(&self.device, &self.bind_group_layout, &self.accum_views);
        self.compute_bind_groups = make_compute_bind_groups(
            &self.device,
            &self.compute_bind_group_layout,
            &self.accum_views,
            &self.camera_buffer,
            &self.sphere_buffer,
            &self.material_buffer,
            &self.bvh_buffer,
            &self.vertex_buffer,
            &self.triangle_buffer,
            &self.mesh_bvh_buffer,
            &self.tex_sampler,
            &self.albedo_tex_view,
            &self.env_map_view,
            &self.emissive_buffer,
            &self.gbuffer_view,
        );
        self.denoise_bind_groups = make_denoise_bind_groups(
            &self.device,
            &self.denoise_bind_group_layout,
            &self.accum_views,
            &self.gbuffer_view,
            &self.denoise_output_view,
        );
        self.display_bg_denoised = make_display_bg_denoised(
            &self.device,
            &self.bind_group_layout,
            &self.denoise_output_view,
        );
    }

    // -----------------------------------------------------------------------
    // Scene hot-reload
    // -----------------------------------------------------------------------

    /// Returns `true` if the watched scene file has changed since the last call,
    /// and drains any additional queued events so no spurious second reload fires.
    fn poll_scene_changed(&self) -> bool {
        let changed = self
            .scene_change_rx
            .as_ref()
            .map(|rx| rx.try_recv().is_ok())
            .unwrap_or(false);
        if changed {
            if let Some(ref rx) = self.scene_change_rx {
                while rx.try_recv().is_ok() {}
            }
        }
        changed
    }

    /// Reload the scene file from disk: re-parse YAML, rebuild BVH and mesh,
    /// recreate scene GPU buffers, rebuild compute bind groups, reset
    /// accumulation.  The camera position is intentionally preserved so the
    /// user's viewpoint survives edits.
    fn reload_scene(&mut self) {
        let path = self.scene_path.clone();
        let loaded = match load_scene_from_yaml(&path) {
            Ok(s) => s,
            Err(e) => {
                log::error!("reload_scene: {e}");
                return;
            }
        };

        let bvh_result = build_bvh(&loaded.spheres);
        let sphere_data: Vec<GpuSphere> = if bvh_result.ordered_spheres.is_empty() {
            vec![GpuSphere {
                center_r: [1.0e9, 1.0e9, 1.0e9, 0.0],
                mat_and_pad: [0; 4],
            }]
        } else {
            bvh_result.ordered_spheres.clone()
        };
        self.sphere_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("sphere storage"),
                contents: bytemuck::cast_slice(&sphere_data),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            });
        let reload_bvh_nodes: Vec<GpuBvhNode> = {
            use bytemuck::Zeroable;
            if bvh_result.nodes.is_empty() {
                vec![GpuBvhNode::zeroed()]
            } else {
                bvh_result.nodes.clone()
            }
        };
        self.bvh_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("bvh node storage"),
                contents: bytemuck::cast_slice(&reload_bvh_nodes),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            });

        let mesh_result = build_mesh_bvh(&loaded.mesh_vertices, &loaded.mesh_triangles);
        use bytemuck::Zeroable;
        let reload_mesh_verts: Vec<GpuVertex> = if mesh_result.vertices.is_empty() {
            vec![GpuVertex::zeroed()]
        } else {
            mesh_result.vertices.clone()
        };
        let reload_mesh_tris: Vec<GpuTriangle> = if mesh_result.ordered_triangles.is_empty() {
            vec![GpuTriangle::zeroed()]
        } else {
            mesh_result.ordered_triangles.clone()
        };
        let reload_mesh_bvh: Vec<GpuBvhNode> = if mesh_result.nodes.is_empty() {
            vec![GpuBvhNode::zeroed()]
        } else {
            mesh_result.nodes.clone()
        };
        self.vertex_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mesh vertex storage"),
                contents: bytemuck::cast_slice(&reload_mesh_verts),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            });
        self.triangle_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mesh triangle storage"),
                contents: bytemuck::cast_slice(&reload_mesh_tris),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            });
        self.mesh_bvh_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mesh bvh node storage"),
                contents: bytemuck::cast_slice(&reload_mesh_bvh),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            });

        // Material buffer has a fixed GPU layout — just overwrite in place.
        self.queue
            .write_buffer(&self.material_buffer, 0, bytes_of(&loaded.materials));
        // Keep the CPU mirror and material names in sync so the UI panel
        // reflects the hot-reloaded values and slider edits don't re-upload stale data.
        self.material_data_cpu = loaded.materials;
        self.material_names = loaded.material_names;

        // Recreate emissive buffer (count may have changed after hot-reload).
        let emissive_data: Vec<GpuSphere> = if loaded.emissive_spheres.is_empty() {
            vec![GpuSphere {
                center_r: [1.0e9, 1.0e9, 1.0e9, 0.0],
                mat_and_pad: [0; 4],
            }]
        } else {
            loaded.emissive_spheres.clone()
        };
        self.emissive_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("emissive sphere storage"),
                contents: bytemuck::cast_slice(&emissive_data),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            });

        // Recreate the env map texture (the scene's env_map: field may have changed).
        let env_map_data = load_env_map_data(loaded.env_map_path.as_deref());
        let env_map_tex = self.device.create_texture(&TextureDescriptor {
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
        self.queue.write_texture(
            TexelCopyTextureInfo {
                texture: &env_map_tex,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            bytemuck::cast_slice(&env_map_data),
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(ENV_MAP_WIDTH * 4 * 4),
                rows_per_image: Some(ENV_MAP_HEIGHT),
            },
            Extent3d {
                width: ENV_MAP_WIDTH,
                height: ENV_MAP_HEIGHT,
                depth_or_array_layers: 1,
            },
        );
        self.env_map_view = env_map_tex.create_view(&TextureViewDescriptor::default());
        self._env_map_tex = env_map_tex;

        // G-buffer and denoised-output textures are resolution-dependent, not
        // scene-dependent.  They are not recreated here; resize() handles them.
        self.rebuild_bind_groups();
        self.frame_count = 0;
        info!(
            "hot-reload: {} spheres, {} triangles, {} materials from '{}'",
            loaded.spheres.len(),
            loaded.mesh_triangles.len(),
            loaded.materials.mat_count,
            path,
        );
        // The camera block in the YAML is intentionally NOT re-applied on hot-reload
        // so that the user's current viewpoint is preserved while editing the scene.
        // Camera changes (look_from / look_at / vfov / aperture) require a restart.
        if loaded.camera.is_some() {
            info!(
                "hot-reload: 'camera:' block changes require a restart to take effect (current viewpoint preserved)"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Render
    // -----------------------------------------------------------------------

    /// Advance the camera lerp, upload the camera uniform, then dispatch the
    /// compute path tracer and the display (tone-map) render pass.
    pub fn render(&mut self) -> Result<(), SurfaceError> {
        // ---- FPS estimation (Phase 15) ------------------------------------
        let now = std::time::Instant::now();
        let dt = now
            .duration_since(self.last_frame_time)
            .as_secs_f32()
            .max(1e-6);
        self.last_frame_time = now;
        // Exponential moving average: α = 0.05 (smooths over ~20 frames).
        let inst_fps = 1.0 / dt;
        self.fps_smooth = self.fps_smooth * 0.95 + inst_fps * 0.05;

        // ---- hot-reload: check if scene file has changed on disk ----------
        if self.poll_scene_changed() {
            self.reload_scene();
        }

        // ---- smooth-interpolate camera toward targets ---------------------
        let prev_look_from = self.camera.look_from();

        self.camera.yaw += (self.camera.tgt_yaw - self.camera.yaw) * CAM_LERP;
        self.camera.pitch += (self.camera.tgt_pitch - self.camera.pitch) * CAM_LERP;
        self.camera.distance += (self.camera.tgt_distance - self.camera.distance) * CAM_LERP;
        self.camera.look_at += (self.camera.tgt_look_at - self.camera.look_at) * CAM_LERP;
        // Keep focus_dist tracking orbit distance so scene stays sharp.
        self.camera.focus_dist = self.camera.distance;

        let look_from = self.camera.look_from();
        let look_at = self.camera.look_at;

        // Reset accumulation whenever the camera has moved at all.
        if (look_from - prev_look_from).length_squared() > CAM_MOVE_EPSILON_SQ {
            self.frame_count = 0;
        }

        let mut cam = compute_camera(
            self.config.width,
            self.config.height,
            look_from,
            look_at,
            self.camera.vfov,
            self.camera.aperture,
            self.camera.focus_dist,
        );

        // Derive ping-pong index before incrementing so the formula is a
        // simple modulo with no subtraction (no u32 underflow risk).
        let f_idx = (self.frame_count % 2) as usize;
        cam.frame_count = self.frame_count;
        self.frame_count = self.frame_count.wrapping_add(1);
        self.queue
            .write_buffer(&self.camera_buffer, 0, bytes_of(&cam));

        let frame = self.surface.get_current_texture()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("frame encoder"),
            });

        // ---- compute pass: path tracer ------------------------------------
        {
            let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("path trace pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, &self.compute_bind_groups[f_idx], &[]);
            // Ceil-divide: every pixel covered even for non-multiple-of-WORKGROUP_SIZE sizes.
            let wg_x = self.config.width.div_ceil(WORKGROUP_SIZE);
            let wg_y = self.config.height.div_ceil(WORKGROUP_SIZE);
            cpass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // ---- compute pass: joint-bilateral denoiser (optional, 'N' key) --
        // wgpu automatically inserts a pipeline barrier between the two compute
        // passes, ensuring the path-tracer's writes to accum[f_idx] and
        // gbuffer_write are visible to the denoiser's texture reads.
        if self.denoise_enabled {
            let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("denoise pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.denoise_pipeline);
            cpass.set_bind_group(0, &self.denoise_bind_groups[f_idx], &[]);
            let wg_x = self.config.width.div_ceil(WORKGROUP_SIZE);
            let wg_y = self.config.height.div_ceil(WORKGROUP_SIZE);
            cpass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // ---- render pass: tone-map (accum or denoised) to swapchain ------
        // When denoising is on the display pass reads from the denoised output
        // texture; otherwise it reads directly from the accumulation buffer.
        let display_bg: &wgpu::BindGroup = if self.denoise_enabled {
            &self.display_bg_denoised
        } else {
            &self.display_bind_groups[f_idx]
        };
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("display pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: Operations {
                        load: LoadOp::Clear(wgpu::Color::BLACK),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.render_pipeline);
            pass.set_bind_group(0, display_bg, &[]);
            pass.draw(0..3, 0..1); // full-screen triangle
        }

        // ---- egui UI pass (Phase 15) -------------------------------------
        // Runs after the display pass so the UI overlays the path-traced image.
        // LoadOp::Load preserves whatever the display pass wrote to the swapchain.
        if self.show_ui {
            let pixels_per_point = self.window.scale_factor() as f32;
            let screen_desc = ScreenDescriptor {
                size_in_pixels: [self.config.width, self.config.height],
                pixels_per_point,
            };

            let raw_input = self.egui_state.take_egui_input(&self.window);

            // Run the UI build; collect mutation flags as locals to avoid
            // overlapping borrows on `self`.
            let mut cam_changed = false;
            let mut mat_changed = false;
            let mut save_request = false;

            let full_output = self.egui_ctx.run(raw_input, |ctx| {
                let (cc, mc, sr) = build_ui_panels(
                    ctx,
                    &mut self.camera,
                    &mut self.denoise_enabled,
                    &mut self.material_data_cpu,
                    &self.material_names,
                    self.config.width,
                    self.config.height,
                    self.frame_count,
                    self.fps_smooth,
                );
                cam_changed = cc;
                mat_changed = mc;
                save_request = sr;
            });

            self.egui_state
                .handle_platform_output(&self.window, full_output.platform_output);

            if cam_changed {
                self.frame_count = 0;
            }
            if mat_changed {
                self.queue.write_buffer(
                    &self.material_buffer,
                    0,
                    bytes_of(&self.material_data_cpu),
                );
                self.frame_count = 0;
                // The camera uniform was already written this frame with a non-zero
                // frame_count.  Re-upload it with frame_count = 0 so the compute pass
                // treats this as the first accumulation sample and discards stale pixels
                // from the previous material, preventing a one-frame blend artifact.
                cam.frame_count = 0;
                self.queue
                    .write_buffer(&self.camera_buffer, 0, bytes_of(&cam));
            }
            if save_request {
                self.save_screenshot();
            }

            // Upload any new / changed egui font textures.
            for (id, delta) in &full_output.textures_delta.set {
                self.egui_renderer
                    .update_texture(&self.device, &self.queue, *id, delta);
            }
            for id in &full_output.textures_delta.free {
                self.egui_renderer.free_texture(id);
            }

            let clipped = self
                .egui_ctx
                .tessellate(full_output.shapes, pixels_per_point);
            self.egui_renderer.update_buffers(
                &self.device,
                &self.queue,
                &mut encoder,
                &clipped,
                &screen_desc,
            );

            // Render egui on top of the swapchain image.
            {
                let mut egui_pass = encoder
                    .begin_render_pass(&RenderPassDescriptor {
                        label: Some("egui pass"),
                        color_attachments: &[Some(RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            depth_slice: None,
                            ops: Operations {
                                load: LoadOp::Load, // preserve path tracer output
                                store: StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        occlusion_query_set: None,
                        timestamp_writes: None,
                    })
                    .forget_lifetime();
                self.egui_renderer
                    .render(&mut egui_pass, &clipped, &screen_desc);
            }
        }

        self.queue.submit([encoder.finish()]);
        frame.present();
        Ok(())
    }

    pub fn handle_mouse_button(&mut self, button: MouseButton, state: ElementState) {
        if button == MouseButton::Left {
            self.input.drag_active = state == ElementState::Pressed;
            if !self.input.drag_active {
                self.input.last_cursor = None;
            }
        }
    }

    pub fn handle_cursor_moved(&mut self, position: PhysicalPosition<f64>) {
        if self.input.drag_active {
            if let Some(last) = self.input.last_cursor {
                let dx = (position.x - last.x) as f32;
                let dy = (position.y - last.y) as f32;

                self.camera.tgt_yaw -= dx * MOUSE_SENSITIVITY;
                // Normalise tgt_yaw so the lerp always takes the short arc
                // around the Y axis, preventing a long-way-round spin after
                // rapid drags.
                let diff = self.camera.tgt_yaw - self.camera.yaw;
                if diff > std::f32::consts::PI {
                    self.camera.tgt_yaw -= std::f32::consts::TAU;
                } else if diff < -std::f32::consts::PI {
                    self.camera.tgt_yaw += std::f32::consts::TAU;
                }

                self.camera.tgt_pitch = (self.camera.tgt_pitch + dy * MOUSE_SENSITIVITY)
                    .clamp(-MAX_PITCH_RAD, MAX_PITCH_RAD);
            }
            self.input.last_cursor = Some(position);
        }
    }

    pub fn handle_mouse_wheel(&mut self, delta: MouseScrollDelta) {
        let scroll = match delta {
            MouseScrollDelta::LineDelta(_, y) => y,
            MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.01,
        };
        self.camera.tgt_distance =
            (self.camera.tgt_distance - scroll * SCROLL_SPEED).clamp(MIN_ZOOM, MAX_ZOOM);
    }

    pub fn handle_key(&mut self, key: PhysicalKey, pressed: bool) {
        match key {
            PhysicalKey::Code(KeyCode::KeyW) => self.input.wasd_held[0] = pressed,
            PhysicalKey::Code(KeyCode::KeyA) => self.input.wasd_held[1] = pressed,
            PhysicalKey::Code(KeyCode::KeyS) => self.input.wasd_held[2] = pressed,
            PhysicalKey::Code(KeyCode::KeyD) => self.input.wasd_held[3] = pressed,
            PhysicalKey::Code(KeyCode::KeyN) => {
                // Toggle the joint-bilateral denoiser on the key-down event only.
                if pressed {
                    self.denoise_enabled = !self.denoise_enabled;
                    info!(
                        "Denoiser: {}  (press N to toggle)",
                        if self.denoise_enabled { "ON" } else { "OFF" }
                    );
                }
            }
            PhysicalKey::Code(KeyCode::KeyH) => {
                // Toggle the egui UI panel (Phase 15).
                if pressed {
                    self.show_ui = !self.show_ui;
                    info!(
                        "UI: {}  (press H to toggle)",
                        if self.show_ui { "visible" } else { "hidden" }
                    );
                }
            }
            _ => {}
        }
    }

    /// Apply any held WASD keys as one pan step. Call once per event-loop tick.
    pub fn apply_movement(&mut self) {
        if self.input.wasd_held.iter().any(|&h| h) {
            // Forward/right ignore pitch so WASD feels like flat-plane movement.
            let forward = Vec3::new(-self.camera.yaw.sin(), 0.0, -self.camera.yaw.cos());
            let right = Vec3::new(self.camera.yaw.cos(), 0.0, -self.camera.yaw.sin());
            if self.input.wasd_held[0] {
                self.camera.tgt_look_at += forward * PAN_SPEED;
            } // W
            if self.input.wasd_held[1] {
                self.camera.tgt_look_at -= right * PAN_SPEED;
            } // A
            if self.input.wasd_held[2] {
                self.camera.tgt_look_at -= forward * PAN_SPEED;
            } // S
            if self.input.wasd_held[3] {
                self.camera.tgt_look_at += right * PAN_SPEED;
            } // D
        }
    }

    /// Ask the OS to schedule a redraw for this window.
    pub fn request_redraw(&self) {
        self.window.request_redraw();
    }

    /// Current surface width (used by App to re-issue correct resize).
    pub fn surface_width(&self) -> u32 {
        self.config.width
    }
    /// Current surface height (used by App to re-issue correct resize).
    pub fn surface_height(&self) -> u32 {
        self.config.height
    }

    // -----------------------------------------------------------------------
    // Phase 15: egui input bridge
    // -----------------------------------------------------------------------

    /// Forward a winit `WindowEvent` to egui's platform layer.
    ///
    /// Returns `true` if egui consumed the event; the caller should skip
    /// camera / orbit handling in that case so the UI receives exclusive input.
    pub fn handle_window_event_egui(&mut self, event: &winit::event::WindowEvent) -> bool {
        let consumed = self
            .egui_state
            .on_window_event(&self.window, event)
            .consumed;
        // If egui claims this event (e.g. a slider drag starts), release all held WASD
        // keys so the camera doesn't drift when the key-release event is swallowed.
        if consumed {
            self.input.wasd_held = [false; 4];
        }
        consumed
    }

    // -----------------------------------------------------------------------
    // Phase 15: screenshot saving
    // -----------------------------------------------------------------------

    /// Read back the current accumulation (or denoised) texture from the GPU,
    /// apply ACES tone-mapping and sRGB gamma, and write the result as a PNG.
    ///
    /// The file is named `render_<unix_seconds>.png` and placed in the current
    /// working directory.  The call blocks for one frame while the GPU finishes
    /// and the buffer is mapped — acceptable for an infrequent user action.
    fn save_screenshot(&self) {
        let width = self.config.width;
        let height = self.config.height;

        // Read from the denoised output when the denoiser is on; otherwise
        // from the last-written accumulation texture.
        let src_tex: &wgpu::Texture = if self.denoise_enabled {
            &self._denoise_output_tex
        } else {
            let last_written = self.frame_count.wrapping_sub(1) as usize % 2;
            &self.accum_textures[last_written]
        };

        // Row stride must be a multiple of 256 bytes for `copy_texture_to_buffer`.
        // Each Rgba32Float pixel = 16 bytes.
        let bytes_per_pixel: u32 = 16;
        let unpadded_row = width * bytes_per_pixel;
        let padded_row = unpadded_row.div_ceil(256) * 256;
        let buf_size = (padded_row * height) as u64;

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("screenshot staging"),
            size: buf_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut enc = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("screenshot encoder"),
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
        self.queue.submit([enc.finish()]);

        // Map synchronously: submit above, poll until done, then read.
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        match rx.recv() {
            Ok(Ok(())) => {} // mapping succeeded — proceed with readback
            _ => {
                log::error!("screenshot: GPU buffer mapping failed");
                return;
            }
        }

        let raw = slice.get_mapped_range();
        let mut pixels: Vec<u8> = Vec::with_capacity((width * height * 4) as usize);
        for row in 0..height as usize {
            let row_off = row * padded_row as usize;
            for col in 0..width as usize {
                let off = row_off + col * 16;
                // Each channel is a little-endian f32.
                let r = f32::from_le_bytes(raw[off..off + 4].try_into().unwrap());
                let g = f32::from_le_bytes(raw[off + 4..off + 8].try_into().unwrap());
                let b = f32::from_le_bytes(raw[off + 8..off + 12].try_into().unwrap());
                // Sanitise before tone-map.
                let r = if r.is_finite() { r } else { 0.0 };
                let g = if g.is_finite() { g } else { 0.0 };
                let b = if b.is_finite() { b } else { 0.0 };
                pixels.push(screenshot_to_u8(r));
                pixels.push(screenshot_to_u8(g));
                pixels.push(screenshot_to_u8(b));
                pixels.push(255);
            }
        }
        drop(raw);
        staging.unmap();

        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let filename = format!("render_{ts}.png");

        match image::RgbaImage::from_raw(width, height, pixels) {
            Some(img) => match img.save(&filename) {
                Ok(()) => info!("Screenshot saved → '{filename}'"),
                Err(e) => log::error!("Screenshot save failed: {e}"),
            },
            None => log::error!("Screenshot: image buffer construction failed"),
        }
    }
}

// ---------------------------------------------------------------------------
// Free functions creating bind groups
// (called both from GpuState::new and GpuState::rebuild_bind_groups)
// ---------------------------------------------------------------------------

fn make_display_bind_groups(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    accum_views: &[wgpu::TextureView; 2],
) -> [wgpu::BindGroup; 2] {
    let make = |view: &wgpu::TextureView, label: &str| {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(view),
            }],
        })
    };
    [
        make(&accum_views[0], "display bg 0"),
        make(&accum_views[1], "display bg 1"),
    ]
}

fn make_compute_bind_groups(
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
                    binding: 0,
                    resource: BindingResource::TextureView(write_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: camera_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: sphere_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(read_view),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: material_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: bvh_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: vertex_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: triangle_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 8,
                    resource: mesh_bvh_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 9,
                    resource: BindingResource::Sampler(tex_sampler),
                },
                BindGroupEntry {
                    binding: 10,
                    resource: BindingResource::TextureView(albedo_tex_view),
                },
                BindGroupEntry {
                    binding: 11,
                    resource: BindingResource::TextureView(env_map_view),
                },
                BindGroupEntry {
                    binding: 12,
                    resource: emissive_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 13,
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

/// Creates two denoiser bind groups (one per accum ping-pong frame).
///
/// Each group reads a different accumulation slice (the just-written one)
/// plus the shared G-buffer, and writes to the shared denoised output texture.
fn make_denoise_bind_groups(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    accum_views: &[wgpu::TextureView; 2],
    gbuffer_view: &wgpu::TextureView,
    denoise_output_view: &wgpu::TextureView,
) -> [wgpu::BindGroup; 2] {
    let make = |accum_view: &wgpu::TextureView, label: &str| {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(accum_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(gbuffer_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(denoise_output_view),
                },
            ],
        })
    };
    [
        make(&accum_views[0], "denoise bg 0"),
        make(&accum_views[1], "denoise bg 1"),
    ]
}

/// Creates the display bind group that reads from the denoised output texture.
///
/// The display bind group layout (`layout`) is the same one used for
/// `display_bind_groups`; it only requires a single `texture_2d<f32>` binding.
fn make_display_bg_denoised(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    denoise_output_view: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&BindGroupDescriptor {
        label: Some("display bg denoised"),
        layout,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: BindingResource::TextureView(denoise_output_view),
        }],
    })
}

// ---------------------------------------------------------------------------
// Phase 15: egui UI panel
// ---------------------------------------------------------------------------

/// Build the right-side egui panel for the current frame.
///
/// Returns `(camera_changed, materials_changed, save_screenshot_requested)`.
/// Returning `true` for either of the first two causes the accumulation buffer
/// to reset so the path tracer re-converges from the updated parameters.
///
/// This is a free function (not a method) so the borrow checker can split
/// `GpuState` fields; `egui::Context::run` holds an immutable borrow on
/// `egui_ctx` while the closure takes mutable borrows on other fields.
fn build_ui_panels(
    ctx: &egui::Context,
    camera: &mut CameraState,
    denoise_enabled: &mut bool,
    material_data: &mut GpuMaterialData,
    material_names: &[String],
    width: u32,
    height: u32,
    frame_count: u32,
    fps: f32,
) -> (bool, bool, bool) {
    use crate::material::{MAT_DIELECTRIC, MAT_EMISSIVE, MAT_LAMBERTIAN, MAT_METAL};

    let mut cam_changed = false;
    let mut mat_changed = false;
    let mut save_request = false;

    egui::SidePanel::right("mcrt_panel")
        .resizable(true)
        .min_width(240.0)
        .default_width(270.0)
        .show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                // ── Render Stats ──────────────────────────────────────────
                ui.add_space(4.0);
                ui.heading("Render Stats");
                ui.separator();
                egui::Grid::new("stats_grid")
                    .num_columns(2)
                    .spacing([8.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("Resolution:");
                        ui.label(format!("{width} × {height}"));
                        ui.end_row();
                        ui.label("Samples (SPP):");
                        ui.label(format!("{frame_count}"));
                        ui.end_row();
                        ui.label("FPS:");
                        ui.label(format!("{fps:.1}"));
                        ui.end_row();
                        ui.label("MPix/s:");
                        let mpx = fps * (width as f32 * height as f32) / 1_000_000.0;
                        ui.label(format!("{mpx:.1}"));
                        ui.end_row();
                    });

                ui.add_space(8.0);

                // ── Post-process ──────────────────────────────────────────
                ui.heading("Post-process");
                ui.separator();
                let was_denoise = *denoise_enabled;
                ui.checkbox(denoise_enabled, "Joint bilateral denoiser  (N)");
                if *denoise_enabled != was_denoise {
                    info!(
                        "Denoiser: {} (UI)",
                        if *denoise_enabled { "ON" } else { "OFF" }
                    );
                }
                ui.add_space(4.0);
                if ui.button("💾  Save PNG").clicked() {
                    save_request = true;
                }
                ui.add_space(8.0);

                // ── Camera ────────────────────────────────────────────────
                ui.heading("Camera");
                ui.separator();
                cam_changed |= ui
                    .add(egui::Slider::new(&mut camera.vfov, 10.0..=120.0).text("FOV (°)"))
                    .changed();
                ui.add_space(2.0);
                cam_changed |= ui
                    .add(
                        egui::Slider::new(&mut camera.aperture, 0.0..=2.0)
                            .step_by(0.01)
                            .text("Aperture"),
                    )
                    .changed();
                ui.add_space(2.0);
                cam_changed |= ui
                    .add(egui::Slider::new(&mut camera.focus_dist, 0.1..=50.0).text("Focus dist"))
                    .changed();
                ui.add_space(8.0);

                // ── Materials ─────────────────────────────────────────────
                ui.heading("Materials");
                ui.separator();
                let mat_count = material_data.mat_count as usize;
                for i in 0..mat_count {
                    let name = material_names
                        .get(i)
                        .map(String::as_str)
                        .unwrap_or("material");
                    let mat_type = material_data.materials[i].type_pad[0];
                    let type_label = match mat_type {
                        MAT_LAMBERTIAN => "Lambertian",
                        MAT_METAL => "Metal",
                        MAT_DIELECTRIC => "Dielectric",
                        MAT_EMISSIVE => "Emissive",
                        _ => "?",
                    };
                    egui::CollapsingHeader::new(format!("{i}: {name}  [{type_label}]"))
                        .id_salt(i)
                        .show(ui, |ui| {
                            let mat = &mut material_data.materials[i];
                            match mat_type {
                                MAT_LAMBERTIAN | MAT_METAL | MAT_EMISSIVE => {
                                    // Albedo / emission colour picker.
                                    let mut col = [
                                        mat.albedo_fuzz[0],
                                        mat.albedo_fuzz[1],
                                        mat.albedo_fuzz[2],
                                    ];
                                    ui.horizontal(|ui| {
                                        ui.label("Albedo:");
                                        if ui.color_edit_button_rgb(&mut col).changed() {
                                            mat.albedo_fuzz[0] = col[0];
                                            mat.albedo_fuzz[1] = col[1];
                                            mat.albedo_fuzz[2] = col[2];
                                            mat_changed = true;
                                        }
                                    });
                                    if mat_type == MAT_METAL {
                                        mat_changed |= ui
                                            .add(
                                                egui::Slider::new(
                                                    &mut mat.albedo_fuzz[3],
                                                    0.0..=1.0,
                                                )
                                                .text("Fuzz"),
                                            )
                                            .changed();
                                    }
                                    if mat_type == MAT_EMISSIVE {
                                        mat_changed |= ui
                                            .add(
                                                egui::Slider::new(
                                                    &mut mat.ior_pad[0],
                                                    0.1..=100_000.0,
                                                )
                                                .logarithmic(true)
                                                .text("Strength"),
                                            )
                                            .changed();
                                    }
                                }
                                MAT_DIELECTRIC => {
                                    mat_changed |= ui
                                        .add(
                                            egui::Slider::new(&mut mat.ior_pad[0], 1.0..=3.0)
                                                .step_by(0.01)
                                                .text("IOR"),
                                        )
                                        .changed();
                                }
                                _ => {
                                    ui.label("(no editable parameters)");
                                }
                            }
                        });
                }
                ui.add_space(4.0);
                ui.small("Press H to hide/show this panel");
            });
        });

    (cam_changed, mat_changed, save_request)
}

// ---------------------------------------------------------------------------
// Phase 15: screenshot tone-mapping helpers
// ---------------------------------------------------------------------------

/// ACES filmic tone-map for a single HDR channel (Narkowicz 2016).
#[inline]
fn aces_channel_cpu(x: f32) -> f32 {
    let x = x.max(0.0);
    const A: f32 = 2.51;
    const B: f32 = 0.03;
    const C: f32 = 2.43;
    const D: f32 = 0.59;
    const E: f32 = 0.14;
    ((x * (A * x + B)) / (x * (C * x + D) + E)).clamp(0.0, 1.0)
}

/// Linear → sRGB transfer function (IEC 61966-2-1) for a single channel.
#[inline]
fn linear_to_srgb_cpu(c: f32) -> f32 {
    if c <= 0.0 {
        return 0.0;
    }
    if c <= 0.003_130_8 {
        return (c * 12.92).min(1.0);
    }
    if c < 1.0 {
        return (1.055 * c.powf(1.0 / 2.4) - 0.055).clamp(0.0, 1.0);
    }
    1.0
}

/// Apply ACES tone-map + sRGB and quantise to `u8` for PNG output.
#[inline]
fn screenshot_to_u8(x: f32) -> u8 {
    (linear_to_srgb_cpu(aces_channel_cpu(x)) * 255.0 + 0.5) as u8
}
