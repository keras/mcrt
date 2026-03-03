use std::mem::size_of;
use std::sync::Arc;

use bytemuck::{bytes_of, Zeroable};
use glam::Vec3;
use log::info;
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingResource, BindingType, BufferDescriptor, BufferUsages, CommandEncoderDescriptor,
    ComputePassDescriptor, ComputePipelineDescriptor, DeviceDescriptor, Extent3d, FragmentState,
    InstanceDescriptor, LoadOp, MultisampleState, Operations, PipelineLayoutDescriptor,
    PowerPreference, PrimitiveState, RenderPassColorAttachment, RenderPassDescriptor,
    RenderPipelineDescriptor, RequestAdapterOptions, ShaderModuleDescriptor, ShaderSource,
    ShaderStages, StorageTextureAccess, StoreOp, SurfaceError, TextureDescriptor, TextureDimension,
    TextureFormat, TextureSampleType, TextureUsages, TextureViewDescriptor, TextureViewDimension,
    VertexState,
};
use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

// ---------------------------------------------------------------------------
// Helper: create an HDR accumulation texture and its default view
// ---------------------------------------------------------------------------

/// Creates an `Rgba32Float` texture used for HDR progressive accumulation.
///
/// Both `TEXTURE_BINDING` (for `textureLoad` in compute & display shaders)
/// and `STORAGE_BINDING` (for `textureStore` in the compute shader) are
/// required.  Two of these textures ping-pong every frame.
fn create_accum_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&TextureDescriptor {
        label: Some("accum texture"),
        size: Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba32Float,
        // TEXTURE_BINDING  — read by compute (accum_read) and display shaders.
        // STORAGE_BINDING  — written by compute shader (accum_write).
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&TextureViewDescriptor::default());
    (texture, view)
}

// ---------------------------------------------------------------------------
// Phase 3: camera uniform
// ---------------------------------------------------------------------------

/// GPU-side camera layout.  All fields use vec4 packing (16-byte aligned),
/// matching the WGSL `Camera` struct exactly.  Total size = 112 bytes.
///
/// Phase 8 adds `defocus_u` / `defocus_v` for the thin-lens depth-of-field
/// model:
///   defocus_u = cam_right × lens_radius
///   defocus_v = cam_up    × lens_radius
/// Both are zero for a pinhole camera (aperture = 0).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    /// Eye position in world space.  `.w` unused.
    origin:     [f32; 4],
    /// Lower-left corner of the virtual screen on the focus plane.  `.w` unused.
    lower_left: [f32; 4],
    /// Full horizontal extent of the screen on the focus plane.  `.w` unused.
    horizontal: [f32; 4],
    /// Full vertical extent of the screen on the focus plane.  `.w` unused.
    vertical:   [f32; 4],
    /// Camera right-basis scaled by lens radius.  Zero for pinhole camera.
    defocus_u:  [f32; 4],
    /// Camera up-basis   scaled by lens radius.  Zero for pinhole camera.
    defocus_v:  [f32; 4],
    /// Monotonically increasing frame index.  `u32` to stay exact beyond 2²⁴.
    frame_count: u32,
    _pad: [u32; 3],
}

/// Compute the camera uniform for a thin-lens camera with optional depth of field.
///
/// - `width` / `height`:  viewport dimensions (aspect ratio source)
/// - `look_from`:         eye position in world space
/// - `look_at`:           orbit target / aim point
/// - `vfov_deg`:          vertical field of view in degrees
/// - `aperture`:          lens diameter in world units (0 = pinhole, no blur)
/// - `focus_dist`:        distance from eye to the plane of sharp focus
///
/// The virtual screen (lower_left / horizontal / vertical) is placed on the
/// focus plane so all rays passing through the same screen point converge there,
/// regardless of aperture.
fn compute_camera(
    width: u32,
    height: u32,
    look_from: Vec3,
    look_at: Vec3,
    vfov_deg: f32,
    aperture: f32,
    focus_dist: f32,
) -> CameraUniform {
    let aspect   = width as f32 / height as f32;
    let vup      = Vec3::new(0.0, 1.0, 0.0);
    let vfov_rad = vfov_deg.to_radians();

    // Half-height at unit distance from the eye.
    let h              = (vfov_rad * 0.5).tan();
    let viewport_height = 2.0 * h;
    let viewport_width  = aspect * viewport_height;

    // Orthonormal camera basis (right-handed, Z points toward the viewer).
    let w = (look_from - look_at).normalize(); // backward
    // Guard: if vup is parallel to w, cross product is zero → NaN.
    debug_assert!(
        vup.cross(w).length_squared() > 1e-10,
        "camera vup is parallel to view direction (gimbal lock)"
    );
    let cam_u = vup.cross(w).normalize(); // right
    let cam_v = w.cross(cam_u);           // up (already unit: w⊥cam_u, both unit)

    // Scale screen extents by focus_dist so the virtual screen sits exactly
    // on the plane of sharp focus.
    let horizontal = focus_dist * viewport_width  * cam_u;
    let vertical   = focus_dist * viewport_height * cam_v;
    let lower_left = look_from - horizontal * 0.5 - vertical * 0.5 - focus_dist * w;

    // Lens disk basis vectors (zero for pinhole where aperture = 0).
    let lens_radius = aperture * 0.5;
    let defocus_u   = cam_u * lens_radius;
    let defocus_v   = cam_v * lens_radius;

    CameraUniform {
        origin:     look_from.extend(0.0).to_array(),
        lower_left: lower_left.extend(0.0).to_array(),
        horizontal: horizontal.extend(0.0).to_array(),
        vertical:   vertical.extend(0.0).to_array(),
        defocus_u:  defocus_u.extend(0.0).to_array(),
        defocus_v:  defocus_v.extend(0.0).to_array(),
        frame_count: 0, // caller sets this before uploading
        _pad: [0; 3],
    }
}

// ---------------------------------------------------------------------------
// Phase 4: scene data types
// ---------------------------------------------------------------------------

/// Maximum number of spheres that can be stored in a `GpuSceneData` buffer.
/// Must match `array<Sphere, 8>` in path_trace.wgsl.
const MAX_SPHERES: usize = 8;

/// Radius of the auto-orbit circle (world units).  Restored in Phase 8.
#[allow(dead_code)]
const ORBIT_RADIUS: f32 = 3.5;
/// Eye height above the world origin during the orbit.  Restored in Phase 8.
#[allow(dead_code)]
const ORBIT_HEIGHT: f32 = 1.0;

/// GPU-side sphere layout.  Packing centre + radius into one `[f32; 4]` ensures
/// identical alignment in Rust (`repr(C)`) and WGSL (`vec4<f32>`).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuSphere {
    /// xyz = world-space centre, w = radius.
    center_r: [f32; 4],
    /// x = material index, yzw = padding.
    mat_and_pad: [u32; 4],
}

/// Full scene uploaded to the GPU.  Size = 16 + 8 × 32 = 272 bytes.
/// Must stay ≤ the minimum guaranteed uniform buffer size (64 KiB).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuSceneData {
    sphere_count: u32,
    _pad: [u32; 3],
    spheres: [GpuSphere; MAX_SPHERES],
}

/// Construct the initial scene:
/// a large ground sphere and three smaller spheres clustered around the origin.
fn build_scene() -> GpuSceneData {
    let raw: &[GpuSphere] = &[
        // Ground plane as a huge sphere.
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
    ];

    let mut data = GpuSceneData::zeroed();
    data.sphere_count = raw.len() as u32;
    data.spheres[..raw.len()].copy_from_slice(raw);
    data
}

// ---------------------------------------------------------------------------
// Phase 7: material buffer
// ---------------------------------------------------------------------------

/// Maximum number of materials in a `GpuMaterialData` buffer.
/// Must match `array<Material, 8>` in path_trace.wgsl.
const MAX_MATERIALS: usize = 8;

/// GPU-side material descriptor.  Each field uses vec4 packing to keep the
/// Rust `repr(C)` layout identical to the WGSL `struct Material` layout.
///
/// - `type_pad[0]`     : material type  (0 = Lambertian, 1 = metal, 2 = dielectric)
/// - `albedo_fuzz[0..3]`: albedo colour  (diffuse / metal tint)
/// - `albedo_fuzz[3]`  : fuzz radius    (metal: 0 = mirror, 1 = fully rough)
/// - `ior_pad[0]`      : index of refraction (dielectric, e.g. 1.5 for glass)
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuMaterial {
    type_pad:    [u32; 4],   // .x = mat_type
    albedo_fuzz: [f32; 4],   // .xyz = albedo, .w = fuzz
    ior_pad:     [f32; 4],   // .x = IOR
}

/// Full material table uploaded to the GPU.  Size = 16 + 8 × 48 = 400 bytes.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuMaterialData {
    mat_count: u32,
    _pad: [u32; 3],
    materials: [GpuMaterial; MAX_MATERIALS],
}

/// Build the material table to match the scene defined in `build_scene()`:
///   0 — Lambertian grey  (ground)
///   1 — Lambertian red   (centre sphere)
///   2 — Metal gold, fuzz=0.1 (left sphere)
///   3 — Dielectric glass, IOR=1.5 (right sphere)
fn build_materials() -> GpuMaterialData {
    let raw: &[GpuMaterial] = &[
        GpuMaterial { type_pad: [0, 0, 0, 0], albedo_fuzz: [0.5, 0.5, 0.5, 0.0], ior_pad: [1.0, 0.0, 0.0, 0.0] },
        GpuMaterial { type_pad: [0, 0, 0, 0], albedo_fuzz: [0.7, 0.3, 0.3, 0.0], ior_pad: [1.0, 0.0, 0.0, 0.0] },
        GpuMaterial { type_pad: [1, 0, 0, 0], albedo_fuzz: [0.8, 0.6, 0.2, 0.1], ior_pad: [1.0, 0.0, 0.0, 0.0] },
        GpuMaterial { type_pad: [2, 0, 0, 0], albedo_fuzz: [1.0, 1.0, 1.0, 0.0], ior_pad: [1.5, 0.0, 0.0, 0.0] },
    ];
    let mut data = GpuMaterialData::zeroed();
    data.mat_count = raw.len() as u32;
    data.materials[..raw.len()].copy_from_slice(raw);
    data
}

// ---------------------------------------------------------------------------
// GPU state — created inside `resumed()`, lives for the rest of the session.
// ---------------------------------------------------------------------------

struct GpuState {
    // `surface` must be declared before `window` so it is dropped first.
    // Rust drops fields in declaration order; if `window` (the Arc) were
    // dropped before `surface`, the raw window handle the surface holds
    // could become a dangling pointer during `Surface` drop.
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    window: Arc<Window>,

    // Phase 6: ping-pong HDR accumulation textures
    /// The two `Rgba32Float` textures that alternate as write/read targets each frame.
    accum_textures: [wgpu::Texture; 2],
    /// Default views into `accum_textures`.
    accum_views: [wgpu::TextureView; 2],
    /// Kept so bind groups can be cheaply recreated on resize.
    bind_group_layout: wgpu::BindGroupLayout,
    /// `display_bind_groups[i]` reads from `accum_views[i]` — swapped every frame.
    display_bind_groups: [wgpu::BindGroup; 2],
    render_pipeline: wgpu::RenderPipeline,

    // Phase 3: compute (path tracer)
    camera_buffer: wgpu::Buffer,
    /// Kept for cheap bind-group recreation on resize.
    compute_bind_group_layout: wgpu::BindGroupLayout,
    /// `compute_bind_groups[i]` writes to `accum_views[i]`, reads `accum_views[1-i]`.
    compute_bind_groups: [wgpu::BindGroup; 2],
    compute_pipeline: wgpu::ComputePipeline,

    // Phase 4: sphere scene
    scene_buffer: wgpu::Buffer,
    // Phase 7: material descriptors
    material_buffer: wgpu::Buffer,
    /// Monotonically increasing frame index; resets to 0 on camera movement or resize.
    frame_count: u32,

    // ---- Phase 8: orbit camera state ----------------------------------------
    /// Current (smoothed) yaw: rotation around the Y axis, radians.
    cam_yaw:        f32,
    /// Current (smoothed) pitch: elevation angle, radians.  Clamped to ±85°.
    cam_pitch:      f32,
    /// Current (smoothed) orbital radius: metres from look_at to eye.
    cam_distance:   f32,
    /// Orbit target point (world space).
    cam_look_at:    Vec3,
    /// Vertical field of view in degrees (user-controllable via scroll FOV mode).
    cam_vfov:       f32,
    /// Lens aperture diameter in world units (0 = pinhole, no DoF blur).
    cam_aperture:   f32,
    /// Distance from eye to the plane of sharp focus.
    cam_focus_dist: f32,

    // Smooth-interpolation targets — user input writes here; render() lerps actual → target.
    tgt_yaw:        f32,
    tgt_pitch:      f32,
    tgt_distance:   f32,
    tgt_look_at:    Vec3,

    // Mouse drag state.
    drag_active:  bool,
    last_cursor:  Option<winit::dpi::PhysicalPosition<f64>>,
    /// Held WASD keys: index 0=W, 1=A, 2=S, 3=D.
    wasd_held:    [bool; 4],
}

impl GpuState {
    fn new(window: Arc<Window>) -> Self {
        // Instance is only needed for adapter/device creation; wgpu ref-counts
        // the adapter and device internally so the instance can be dropped here.
        let instance = wgpu::Instance::new(&InstanceDescriptor::default());

        // `Arc<Window>: 'static`, so the safe create_surface() API can hold
        // the Arc internally and guarantee the window outlives the surface.
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
        let mut config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .expect("surface not supported by adapter");

        // Lock to vsync so we don't spin the GPU harder than the display rate.
        config.present_mode = wgpu::PresentMode::Fifo;
        surface.configure(&device, &config);

        // ---- accumulation textures ----------------------------------------
        // Two Rgba32Float textures ping-pong every frame: the compute shader
        // writes to accum[f_idx] and reads from accum[1 - f_idx]; the display
        // pass reads from accum[f_idx] for tone-mapping.
        let (accum_tex0, accum_view0) = create_accum_texture(&device, config.width, config.height);
        let (accum_tex1, accum_view1) = create_accum_texture(&device, config.width, config.height);
        let accum_textures = [accum_tex0, accum_tex1];
        let accum_views = [accum_view0, accum_view1];

        // ---- display bind group layout ------------------------------------
        // Single entry: the Rgba32Float accumulation texture read via
        // textureLoad (no sampler needed, avoids FLOAT32_FILTERABLE feature).
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

        // ---- display bind groups ------------------------------------------
        // display_bind_groups[i] reads accum_views[i].
        let make_display_bg = |view: &wgpu::TextureView, label: &str| {
            device.create_bind_group(&BindGroupDescriptor {
                label: Some(label),
                layout: &bind_group_layout,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(view),
                }],
            })
        };
        let display_bind_groups = [
            make_display_bg(&accum_views[0], "display bg 0"),
            make_display_bg(&accum_views[1], "display bg 1"),
        ];

        // ---- shader --------------------------------------------------------
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("display shader"),
            source: ShaderSource::Wgsl(include_str!("shaders/display.wgsl").into()),
        });

        // ---- render pipeline -----------------------------------------------
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
                buffers: &[], // positions generated in the shader
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
                cull_mode: None, // single oversized triangle — culling unnecessary
                ..Default::default()
            },
            depth_stencil: None,
            multisample: MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // ---- camera buffer -------------------------------------------------
        // Sized to hold one CameraUniform; updated every frame for the orbit.
        // The initial contents are overwritten on the very first render() call
        // (elapsed ≈ 0 → identical position), so no explicit upload is needed.
        let camera_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("camera buffer"),
            size: size_of::<CameraUniform>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ---- scene buffer --------------------------------------------------
        let scene_data = build_scene();
        let scene_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("scene buffer"),
            size: size_of::<GpuSceneData>() as u64,
            // COPY_DST reserved for future dynamic scene updates (Phase 6+).
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&scene_buffer, 0, bytes_of(&scene_data));

        // ---- material buffer -----------------------------------------------
        let material_data = build_materials();
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
                    // binding 0: write-only storage texture (accumulation write target)
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::StorageTexture {
                            access: StorageTextureAccess::WriteOnly,
                            format: TextureFormat::Rgba32Float,
                            view_dimension: TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // binding 1: camera parameters uniform
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                size_of::<CameraUniform>() as u64
                            ),
                        },
                        count: None,
                    },
                    // binding 2: scene (spheres) uniform
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                size_of::<GpuSceneData>() as u64
                            ),
                        },
                        count: None,
                    },
                    // binding 3: previous frame's accumulation (read via textureLoad)
                    BindGroupLayoutEntry {
                        binding: 3,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: false },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // binding 4: material descriptors uniform
                    BindGroupLayoutEntry {
                        binding: 4,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                size_of::<GpuMaterialData>() as u64
                            ),
                        },
                        count: None,
                    },
                ],
            });

        // ---- compute bind groups ------------------------------------------
        // compute_bind_groups[i]: write to accum_views[i], read accum_views[1-i].
        let make_compute_bg =
            |write_view: &wgpu::TextureView, read_view: &wgpu::TextureView, label: &str| {
                device.create_bind_group(&BindGroupDescriptor {
                    label: Some(label),
                    layout: &compute_bind_group_layout,
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
                            resource: scene_buffer.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 3,
                            resource: BindingResource::TextureView(read_view),
                        },
                        BindGroupEntry {
                            binding: 4,
                            resource: material_buffer.as_entire_binding(),
                        },
                    ],
                })
            };
        let compute_bind_groups = [
            make_compute_bg(&accum_views[0], &accum_views[1], "compute bg 0"),
            make_compute_bg(&accum_views[1], &accum_views[0], "compute bg 1"),
        ];

        // ---- compute shader -----------------------------------------------
        let compute_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("path trace shader"),
            source: ShaderSource::Wgsl(include_str!("shaders/path_trace.wgsl").into()),
        });

        // ---- compute pipeline ---------------------------------------------
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

        // ---- Phase 8: orbit camera initial state -------------------------
        // Derive orbit parameters from the Phase 6 static camera position so
        // the scene looks identical on first launch.
        let init_look_from = Vec3::new(0.0, 1.0, 3.5);
        let init_look_at   = Vec3::ZERO;
        let delta          = init_look_from - init_look_at;
        let init_dist      = delta.length();
        let init_pitch     = (delta.y / init_dist).asin();
        let init_yaw       = f32::atan2(delta.x, delta.z);

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
            scene_buffer,
            material_buffer,
            frame_count: 0,
            // Phase 8: orbit camera
            cam_yaw:        init_yaw,
            cam_pitch:      init_pitch,
            cam_distance:   init_dist,
            cam_look_at:    init_look_at,
            cam_vfov:       60.0,
            cam_aperture:   0.0,          // pinhole — no DoF blur
            cam_focus_dist: init_dist,    // focus at orbit distance
            tgt_yaw:        init_yaw,
            tgt_pitch:      init_pitch,
            tgt_distance:   init_dist,
            tgt_look_at:    init_look_at,
            drag_active:    false,
            last_cursor:    None,
            wasd_held:      [false; 4],
        }
    }

    fn resize(&mut self, new_width: u32, new_height: u32) {
        // Guard: wgpu panics if either dimension is zero (e.g. when minimised).
        if new_width == 0 || new_height == 0 {
            return;
        }
        self.config.width = new_width;
        self.config.height = new_height;
        self.surface.configure(&self.device, &self.config);

        // Recreate both accumulation textures at the new size.
        let (tex0, view0) = create_accum_texture(&self.device, new_width, new_height);
        let (tex1, view1) = create_accum_texture(&self.device, new_width, new_height);
        self.accum_textures = [tex0, tex1];
        self.accum_views = [view0, view1];

        // Recreate display bind groups to reference the new views.
        self.display_bind_groups = [
            self.device.create_bind_group(&BindGroupDescriptor {
                label: Some("display bg 0"),
                layout: &self.bind_group_layout,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&self.accum_views[0]),
                }],
            }),
            self.device.create_bind_group(&BindGroupDescriptor {
                label: Some("display bg 1"),
                layout: &self.bind_group_layout,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&self.accum_views[1]),
                }],
            }),
        ];

        // Recreate compute bind groups to reference the new views.
        self.compute_bind_groups = [
            self.device.create_bind_group(&BindGroupDescriptor {
                label: Some("compute bg 0"),
                layout: &self.compute_bind_group_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(&self.accum_views[0]),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: self.camera_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: self.scene_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: BindingResource::TextureView(&self.accum_views[1]),
                    },
                    BindGroupEntry {
                        binding: 4,
                        resource: self.material_buffer.as_entire_binding(),
                    },
                ],
            }),
            self.device.create_bind_group(&BindGroupDescriptor {
                label: Some("compute bg 1"),
                layout: &self.compute_bind_group_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(&self.accum_views[1]),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: self.camera_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: self.scene_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: BindingResource::TextureView(&self.accum_views[0]),
                    },
                    BindGroupEntry {
                        binding: 4,
                        resource: self.material_buffer.as_entire_binding(),
                    },
                ],
            }),
        ];

        // Reset accumulation so the new-size buffer starts clean.
        self.frame_count = 0;
    }

    /// Returns the eye position in world space derived from the current orbit params.
    fn look_from(&self) -> Vec3 {
        let cos_p = self.cam_pitch.cos();
        self.cam_look_at
            + Vec3::new(
                self.cam_distance * self.cam_yaw.sin() * cos_p,
                self.cam_distance * self.cam_pitch.sin(),
                self.cam_distance * self.cam_yaw.cos() * cos_p,
            )
    }

    fn render(&mut self) -> Result<(), SurfaceError> {
        // ---- Phase 8: smooth interpolation toward camera targets ----------
        // Exponential ease: each frame covers `LERP` fraction of the remaining gap.
        // At 60 fps this settles to < 1 % error in ~30 frames (~0.5 s).
        const LERP: f32 = 0.15;
        let prev_look_from = self.look_from();

        self.cam_yaw      += (self.tgt_yaw      - self.cam_yaw)      * LERP;
        self.cam_pitch    += (self.tgt_pitch     - self.cam_pitch)    * LERP;
        self.cam_distance += (self.tgt_distance  - self.cam_distance) * LERP;
        self.cam_look_at  += (self.tgt_look_at   - self.cam_look_at)  * LERP;
        // Keep focus_dist tracking orbit distance so the scene stays sharp.
        self.cam_focus_dist = self.cam_distance;

        let look_from   = self.look_from();
        let look_at     = self.cam_look_at;

        // Reset accumulation whenever the camera has moved (even by the lerp step).
        if (look_from - prev_look_from).length_squared() > 1e-12 {
            self.frame_count = 0;
        }

        let mut cam = compute_camera(
            self.config.width,
            self.config.height,
            look_from,
            look_at,
            self.cam_vfov,
            self.cam_aperture,
            self.cam_focus_dist,
        );

        // Derive the ping-pong index *before* incrementing frame_count so the
        // formula is a simple modulo with no subtraction — avoids any risk of
        // u32 underflow and keeps the relationship obvious.
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

        // ---- compute pass: path tracer accumulates into accum[f_idx] ------
        {
            let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("path trace pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, &self.compute_bind_groups[f_idx], &[]);
            // Ceil-divide so every pixel is covered even for non-multiple-of-16 sizes.
            let wg_x = self.config.width.div_ceil(16);
            let wg_y = self.config.height.div_ceil(16);
            cpass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // ---- render pass: tone-map accum[f_idx] to the swap-chain surface --
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("display pass"),
                multiview_mask: None,
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
            pass.set_bind_group(0, &self.display_bind_groups[f_idx], &[]);
            pass.draw(0..3, 0..1); // full-screen triangle — no vertex buffer
        }

        self.queue.submit([encoder.finish()]);
        frame.present();

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Application — implements the winit 0.30 ApplicationHandler trait.
// ---------------------------------------------------------------------------

#[derive(Default)]
struct App {
    state: Option<GpuState>,
}

impl ApplicationHandler for App {
    /// Called when the OS has a surface ready for rendering.
    /// This is the correct and only place to create the window and wgpu state.
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            // Already initialised (can also fire on Android returning from
            // background — no-op here since we don't support suspend/resume).
            return;
        }

        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("mcrt — path tracer")
                        .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32)),
                )
                .expect("failed to create window"),
        );

        self.state = Some(GpuState::new(window));
        info!("GPU state initialised");
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId, // single window — no need to dispatch by ID
        event: WindowEvent,
    ) {
        let Some(state) = self.state.as_mut() else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => {
                info!("Close requested — exiting");
                event_loop.exit();
            }

            WindowEvent::Resized(size) => {
                state.resize(size.width, size.height);
            }

            WindowEvent::RedrawRequested => {
                match state.render() {
                    Ok(()) => {}
                    // Surface lost or outdated: reconfigure and skip this frame.
                    Err(SurfaceError::Lost | SurfaceError::Outdated) => {
                        let (w, h) = (state.config.width, state.config.height);
                        state.resize(w, h);
                    }
                    // GPU timed out: transient; just request the next frame.
                    Err(SurfaceError::Timeout) => {
                        state.window.request_redraw();
                    }
                    // Out of memory — not recoverable.
                    Err(SurfaceError::OutOfMemory) => {
                        log::error!("Out of GPU memory — exiting");
                        event_loop.exit();
                    }
                    Err(e) => log::warn!("Transient render error: {e:?}"),
                }
            }

            // ---- Phase 8: mouse drag → orbit yaw / pitch -----------------
            WindowEvent::MouseInput { button, state: btn_state, .. } => {
                if button == MouseButton::Left {
                    state.drag_active = btn_state == ElementState::Pressed;
                    if !state.drag_active {
                        state.last_cursor = None;
                    }
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                if state.drag_active {
                    if let Some(last) = state.last_cursor {
                        let dx = (position.x - last.x) as f32;
                        let dy = (position.y - last.y) as f32;
                        state.tgt_yaw   -= dx * 0.005;
                        // Normalise so the lerp always takes the short arc around the
                        // Y axis, preventing a "long-way-round" spin after fast drags.
                        let diff = state.tgt_yaw - state.cam_yaw;
                        if diff > std::f32::consts::PI {
                            state.tgt_yaw -= std::f32::consts::TAU;
                        } else if diff < -std::f32::consts::PI {
                            state.tgt_yaw += std::f32::consts::TAU;
                        }
                        state.tgt_pitch  = (state.tgt_pitch + dy * 0.005)
                            .clamp(-1.483, 1.483); // ±85°
                    }
                    state.last_cursor = Some(position);
                }
            }

            // Scroll: zoom by changing orbital distance (hold Shift for FOV tweak).
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p)   => p.y as f32 * 0.01,
                };
                state.tgt_distance = (state.tgt_distance - scroll * 0.3).clamp(0.3, 50.0);
            }

            // WASD: update held-key flags; translation is applied in about_to_wait.
            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;
                match event.physical_key {
                    PhysicalKey::Code(KeyCode::KeyW) => state.wasd_held[0] = pressed,
                    PhysicalKey::Code(KeyCode::KeyA) => state.wasd_held[1] = pressed,
                    PhysicalKey::Code(KeyCode::KeyS) => state.wasd_held[2] = pressed,
                    PhysicalKey::Code(KeyCode::KeyD) => state.wasd_held[3] = pressed,
                    _ => {}
                }
            }

            _ => {}
        }
    }

    /// Called after all pending events for a frame are processed.
    /// Continuous WASD translation is applied here (once per event-loop tick)
    /// so movement is smooth while a key is held.  The redraw request drives
    /// the render loop at display refresh rate.
    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = self.state.as_mut() {
            // Apply held WASD: translate the orbit target in the horizontal plane.
            if state.wasd_held.iter().any(|&h| h) {
                // Forward / right directions ignore pitch so WASD feels like
                // moving on a flat plane regardless of camera elevation.
                let forward = Vec3::new(-state.cam_yaw.sin(), 0.0, -state.cam_yaw.cos());
                let right   = Vec3::new( state.cam_yaw.cos(), 0.0, -state.cam_yaw.sin());
                let speed   = 0.04_f32;
                if state.wasd_held[0] { state.tgt_look_at += forward * speed; } // W
                if state.wasd_held[1] { state.tgt_look_at -= right   * speed; } // A
                if state.wasd_held[2] { state.tgt_look_at -= forward * speed; } // S
                if state.wasd_held[3] { state.tgt_look_at += right   * speed; } // D
            }
            state.window.request_redraw();
        }
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().expect("failed to create event loop");
    let mut app = App::default();
    event_loop.run_app(&mut app).expect("event loop error");
}
