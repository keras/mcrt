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
    dpi::PhysicalPosition,
    event::{ElementState, MouseButton, MouseScrollDelta},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

use crate::camera::{compute_camera, CameraUniform, DEFAULT_VFOV, INIT_LOOK_AT, INIT_LOOK_FROM};
use crate::material::{build_materials, GpuMaterialData};
use crate::scene::{build_scene, GpuSceneData};

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
    pub yaw:        f32,
    pub pitch:      f32,
    pub distance:   f32,
    pub look_at:    Vec3,
    pub vfov:       f32,
    pub aperture:   f32,
    pub focus_dist: f32,

    // Raw input targets — the values yaw/pitch/… are interpolating toward.
    pub tgt_yaw:      f32,
    pub tgt_pitch:    f32,
    pub tgt_distance: f32,
    pub tgt_look_at:  Vec3,
}

impl CameraState {
    /// Initialise from the look-from / look-at pair used in Phase 6.
    pub fn from_initial_position(look_from: Vec3, look_at: Vec3) -> Self {
        let delta    = look_from - look_at;
        let distance = delta.length();
        let pitch    = (delta.y / distance).asin();
        let yaw      = f32::atan2(delta.x, delta.z);

        Self {
            yaw,
            pitch,
            distance,
            look_at,
            vfov:       DEFAULT_VFOV,
            aperture:   0.0,
            focus_dist: distance,
            tgt_yaw:      yaw,
            tgt_pitch:    pitch,
            tgt_distance: distance,
            tgt_look_at:  look_at,
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
    pub drag_active:  bool,
    pub last_cursor:  Option<PhysicalPosition<f64>>,
    /// Held WASD keys: [W, A, S, D].
    pub wasd_held:    [bool; 4],
}

impl Default for InputState {
    fn default() -> Self {
        Self { drag_active: false, last_cursor: None, wasd_held: [false; 4] }
    }
}

// ---------------------------------------------------------------------------
// Texture helper
// ---------------------------------------------------------------------------

/// Creates an `Rgba32Float` texture used for HDR progressive accumulation.
///
/// Both `TEXTURE_BINDING` (read by compute & display shaders) and
/// `STORAGE_BINDING` (written by the compute shader) are required.
/// Two of these textures ping-pong every frame.
fn create_accum_texture(
    device: &wgpu::Device,
    width:  u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&TextureDescriptor {
        label: Some("accum texture"),
        size: Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count:    1,
        dimension:       TextureDimension::D2,
        format:          TextureFormat::Rgba32Float,
        usage:           TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
        view_formats:    &[],
    });
    let view = texture.create_view(&TextureViewDescriptor::default());
    (texture, view)
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
    surface:  wgpu::Surface<'static>,
    device:   wgpu::Device,
    queue:    wgpu::Queue,
    config:   wgpu::SurfaceConfiguration,
    window:   Arc<Window>,

    // Ping-pong HDR accumulation textures.
    accum_textures:   [wgpu::Texture;   2],
    accum_views:      [wgpu::TextureView; 2],

    // Display (tone-map) pass.
    bind_group_layout:   wgpu::BindGroupLayout,
    display_bind_groups: [wgpu::BindGroup; 2],
    render_pipeline:     wgpu::RenderPipeline,

    // Compute (path trace) pass.
    camera_buffer:              wgpu::Buffer,
    compute_bind_group_layout:  wgpu::BindGroupLayout,
    compute_bind_groups:        [wgpu::BindGroup; 2],
    compute_pipeline:           wgpu::ComputePipeline,
    scene_buffer:               wgpu::Buffer,
    material_buffer:            wgpu::Buffer,

    /// Monotonically increasing frame index; resets to 0 on camera move / resize.
    frame_count: u32,

    // Sub-structs for non-GPU state.
    camera: CameraState,
    input:  InputState,
}

impl GpuState {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    pub fn new(window: Arc<Window>) -> Self {
        let instance = wgpu::Instance::new(&InstanceDescriptor::default());

        let surface = instance
            .create_surface(Arc::clone(&window))
            .expect("failed to create surface");

        let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
            power_preference:       PowerPreference::HighPerformance,
            compatible_surface:     Some(&surface),
            force_fallback_adapter: false,
        }))
        .expect("no suitable GPU adapter found");

        info!("Adapter: {}", adapter.get_info().name);

        let (device, queue) = pollster::block_on(adapter.request_device(&DeviceDescriptor {
            label: Some("mcrt device"),
            ..Default::default()
        }))
        .expect("failed to create device");

        let size   = window.inner_size();
        let width  = size.width.max(1);
        let height = size.height.max(1);
        let mut config = surface
            .get_default_config(&adapter, width, height)
            .expect("surface not supported by adapter");

        // Lock to vsync; swapchain drives at display refresh rate.
        config.present_mode = wgpu::PresentMode::Fifo;
        surface.configure(&device, &config);

        // ---- accumulation textures ----------------------------------------
        let (accum_tex0, accum_view0) =
            create_accum_texture(&device, config.width, config.height);
        let (accum_tex1, accum_view1) =
            create_accum_texture(&device, config.width, config.height);
        let accum_textures = [accum_tex0, accum_tex1];
        let accum_views    = [accum_view0, accum_view1];

        // ---- display bind group layout ------------------------------------
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("display bgl"),
            entries: &[BindGroupLayoutEntry {
                binding:    0,
                visibility: ShaderStages::FRAGMENT,
                ty:         BindingType::Texture {
                    sample_type:   TextureSampleType::Float { filterable: false },
                    view_dimension: TextureViewDimension::D2,
                    multisampled:  false,
                },
                count: None,
            }],
        });

        // ---- display shader + pipeline ------------------------------------
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label:  Some("display shader"),
            source: ShaderSource::Wgsl(include_str!("shaders/display.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label:              Some("display pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            ..Default::default()
        });
        let render_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label:  Some("display pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module:              &shader,
                entry_point:         Some("vs_main"),
                buffers:             &[],
                compilation_options: Default::default(),
            },
            fragment: Some(FragmentState {
                module:              &shader,
                entry_point:         Some("fs_main"),
                targets:             &[Some(wgpu::ColorTargetState {
                    format:     config.format,
                    blend:      None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: PrimitiveState {
                topology:   wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode:  None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample:   MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // ---- camera buffer -------------------------------------------------
        let camera_buffer = device.create_buffer(&BufferDescriptor {
            label:              Some("camera buffer"),
            size:               size_of::<CameraUniform>() as u64,
            usage:              BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ---- scene buffer --------------------------------------------------
        let scene_data = build_scene();
        let scene_buffer = device.create_buffer(&BufferDescriptor {
            label:              Some("scene buffer"),
            size:               size_of::<GpuSceneData>() as u64,
            usage:              BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&scene_buffer, 0, bytes_of(&scene_data));

        // ---- material buffer -----------------------------------------------
        let material_data = build_materials();
        let material_buffer = device.create_buffer(&BufferDescriptor {
            label:              Some("material buffer"),
            size:               size_of::<GpuMaterialData>() as u64,
            usage:              BufferUsages::UNIFORM | BufferUsages::COPY_DST,
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
                    // 2: sphere scene uniform
                    BindGroupLayoutEntry {
                        binding:    2,
                        visibility: ShaderStages::COMPUTE,
                        ty:         BindingType::Buffer {
                            ty:                wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size:   wgpu::BufferSize::new(
                                size_of::<GpuSceneData>() as u64
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
                ],
            });

        // ---- compute shader + pipeline ------------------------------------
        let compute_shader = device.create_shader_module(ShaderModuleDescriptor {
            label:  Some("path trace shader"),
            source: ShaderSource::Wgsl(include_str!("shaders/path_trace.wgsl").into()),
        });
        let compute_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label:              Some("compute pipeline layout"),
            bind_group_layouts: &[&compute_bind_group_layout],
            ..Default::default()
        });
        let compute_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label:               Some("path trace pipeline"),
            layout:              Some(&compute_pipeline_layout),
            module:              &compute_shader,
            entry_point:         Some("cs_main"),
            compilation_options: Default::default(),
            cache:               None,
        });

        // ---- initial camera state -----------------------------------------
        let camera = CameraState::from_initial_position(INIT_LOOK_FROM, INIT_LOOK_AT);

        // ---- build bind groups from the already-created resources ---------
        // Use a temporary partial struct so rebuild_bind_groups can be called
        // as a regular (non-self) function before we finish constructing Self.
        let display_bind_groups = make_display_bind_groups(
            &device, &bind_group_layout, &accum_views,
        );
        let compute_bind_groups = make_compute_bind_groups(
            &device,
            &compute_bind_group_layout,
            &accum_views,
            &camera_buffer,
            &scene_buffer,
            &material_buffer,
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
            scene_buffer,
            material_buffer,
            frame_count: 0,
            camera,
            input: InputState::default(),
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
        self.config.width  = new_width;
        self.config.height = new_height;
        self.surface.configure(&self.device, &self.config);

        let (tex0, view0) = create_accum_texture(&self.device, new_width, new_height);
        let (tex1, view1) = create_accum_texture(&self.device, new_width, new_height);
        self.accum_textures = [tex0, tex1];
        self.accum_views    = [view0, view1];

        self.rebuild_bind_groups();
        self.frame_count = 0;
    }

    // -----------------------------------------------------------------------
    // Bind group reconstruction (called by both new→resize and resize)
    // -----------------------------------------------------------------------

    /// Recreate both display and compute bind groups to reference the current
    /// `accum_views`.  Must be called whenever `accum_views` are replaced.
    fn rebuild_bind_groups(&mut self) {
        self.display_bind_groups = make_display_bind_groups(
            &self.device, &self.bind_group_layout, &self.accum_views,
        );
        self.compute_bind_groups = make_compute_bind_groups(
            &self.device,
            &self.compute_bind_group_layout,
            &self.accum_views,
            &self.camera_buffer,
            &self.scene_buffer,
            &self.material_buffer,
        );
    }

    // -----------------------------------------------------------------------
    // Render
    // -----------------------------------------------------------------------

    /// Advance the camera lerp, upload the camera uniform, then dispatch the
    /// compute path tracer and the display (tone-map) render pass.
    pub fn render(&mut self) -> Result<(), SurfaceError> {
        // ---- smooth-interpolate camera toward targets ---------------------
        let prev_look_from = self.camera.look_from();

        self.camera.yaw      += (self.camera.tgt_yaw      - self.camera.yaw)      * CAM_LERP;
        self.camera.pitch    += (self.camera.tgt_pitch     - self.camera.pitch)    * CAM_LERP;
        self.camera.distance += (self.camera.tgt_distance  - self.camera.distance) * CAM_LERP;
        self.camera.look_at  += (self.camera.tgt_look_at   - self.camera.look_at)  * CAM_LERP;
        // Keep focus_dist tracking orbit distance so scene stays sharp.
        self.camera.focus_dist = self.camera.distance;

        let look_from = self.camera.look_from();
        let look_at   = self.camera.look_at;

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
        cam.frame_count  = self.frame_count;
        self.frame_count = self.frame_count.wrapping_add(1);
        self.queue.write_buffer(&self.camera_buffer, 0, bytes_of(&cam));

        let frame = self.surface.get_current_texture()?;
        let view  = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("frame encoder"),
        });

        // ---- compute pass: path tracer ------------------------------------
        {
            let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label:            Some("path trace pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, &self.compute_bind_groups[f_idx], &[]);
            // Ceil-divide: every pixel covered even for non-multiple-of-WORKGROUP_SIZE sizes.
            let wg_x = self.config.width.div_ceil(WORKGROUP_SIZE);
            let wg_y = self.config.height.div_ceil(WORKGROUP_SIZE);
            cpass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // ---- render pass: tone-map accum to swapchain --------------------
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label:         Some("display pass"),
                multiview_mask: None,
                color_attachments: &[Some(RenderPassColorAttachment {
                    view:           &view,
                    resolve_target: None,
                    depth_slice:    None,
                    ops:            Operations {
                        load:  LoadOp::Clear(wgpu::Color::BLACK),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set:      None,
                timestamp_writes:         None,
            });
            pass.set_pipeline(&self.render_pipeline);
            pass.set_bind_group(0, &self.display_bind_groups[f_idx], &[]);
            pass.draw(0..3, 0..1); // full-screen triangle
        }

        self.queue.submit([encoder.finish()]);
        frame.present();
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Input handling — called from App event handlers
    // -----------------------------------------------------------------------

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

                self.camera.tgt_yaw   -= dx * MOUSE_SENSITIVITY;
                // Normalise tgt_yaw so the lerp always takes the short arc
                // around the Y axis, preventing a long-way-round spin after
                // rapid drags.
                let diff = self.camera.tgt_yaw - self.camera.yaw;
                if diff > std::f32::consts::PI {
                    self.camera.tgt_yaw -= std::f32::consts::TAU;
                } else if diff < -std::f32::consts::PI {
                    self.camera.tgt_yaw += std::f32::consts::TAU;
                }

                self.camera.tgt_pitch =
                    (self.camera.tgt_pitch + dy * MOUSE_SENSITIVITY).clamp(-MAX_PITCH_RAD, MAX_PITCH_RAD);
            }
            self.input.last_cursor = Some(position);
        }
    }

    pub fn handle_mouse_wheel(&mut self, delta: MouseScrollDelta) {
        let scroll = match delta {
            MouseScrollDelta::LineDelta(_, y) => y,
            MouseScrollDelta::PixelDelta(p)   => p.y as f32 * 0.01,
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
            _ => {}
        }
    }

    /// Apply any held WASD keys as one pan step. Call once per event-loop tick.
    pub fn apply_movement(&mut self) {
        if self.input.wasd_held.iter().any(|&h| h) {
            // Forward/right ignore pitch so WASD feels like flat-plane movement.
            let forward = Vec3::new(-self.camera.yaw.sin(), 0.0, -self.camera.yaw.cos());
            let right   = Vec3::new( self.camera.yaw.cos(), 0.0, -self.camera.yaw.sin());
            if self.input.wasd_held[0] { self.camera.tgt_look_at += forward * PAN_SPEED; } // W
            if self.input.wasd_held[1] { self.camera.tgt_look_at -= right   * PAN_SPEED; } // A
            if self.input.wasd_held[2] { self.camera.tgt_look_at -= forward * PAN_SPEED; } // S
            if self.input.wasd_held[3] { self.camera.tgt_look_at += right   * PAN_SPEED; } // D
        }
    }

    /// Ask the OS to schedule a redraw for this window.
    pub fn request_redraw(&self) {
        self.window.request_redraw();
    }

    /// Current surface width (used by App to re-issue correct resize).
    pub fn surface_width(&self)  -> u32 { self.config.width  }
    /// Current surface height (used by App to re-issue correct resize).
    pub fn surface_height(&self) -> u32 { self.config.height }
}

// ---------------------------------------------------------------------------
// Free functions creating bind groups
// (called both from GpuState::new and GpuState::rebuild_bind_groups)
// ---------------------------------------------------------------------------

fn make_display_bind_groups(
    device:        &wgpu::Device,
    layout:        &wgpu::BindGroupLayout,
    accum_views:   &[wgpu::TextureView; 2],
) -> [wgpu::BindGroup; 2] {
    let make = |view: &wgpu::TextureView, label: &str| {
        device.create_bind_group(&BindGroupDescriptor {
            label:   Some(label),
            layout,
            entries: &[BindGroupEntry {
                binding:  0,
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
    device:          &wgpu::Device,
    layout:          &wgpu::BindGroupLayout,
    accum_views:     &[wgpu::TextureView; 2],
    camera_buffer:   &wgpu::Buffer,
    scene_buffer:    &wgpu::Buffer,
    material_buffer: &wgpu::Buffer,
) -> [wgpu::BindGroup; 2] {
    let make = |write_view: &wgpu::TextureView, read_view: &wgpu::TextureView, label: &str| {
        device.create_bind_group(&BindGroupDescriptor {
            label:   Some(label),
            layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: BindingResource::TextureView(write_view) },
                BindGroupEntry { binding: 1, resource: camera_buffer.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: scene_buffer.as_entire_binding() },
                BindGroupEntry { binding: 3, resource: BindingResource::TextureView(read_view) },
                BindGroupEntry { binding: 4, resource: material_buffer.as_entire_binding() },
            ],
        })
    };
    [
        // compute_bind_groups[i]: write to accum[i], read accum[1-i]
        make(&accum_views[0], &accum_views[1], "compute bg 0"),
        make(&accum_views[1], &accum_views[0], "compute bg 1"),
    ]
}
