use std::mem::size_of;
use std::sync::Arc;

use bytemuck;
use glam::Vec3;
use log::info;
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
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

// ---------------------------------------------------------------------------
// Helper: create the display texture and its default view
// ---------------------------------------------------------------------------

/// Creates an `Rgba8Unorm` texture that we write gradient pixels into from the
/// CPU (Phase 2) and will later be replaced by compute-shader writes (Phase 3+).
fn create_display_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&TextureDescriptor {
        label: Some("display texture"),
        size: Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        // TEXTURE_BINDING  — the fragment shader samples this texture.
        // COPY_DST         — the CPU fills it via queue.write_texture() (Phase 2).
        // STORAGE_BINDING  — the compute shader will write to it in Phase 3+.
        usage: TextureUsages::TEXTURE_BINDING
            | TextureUsages::COPY_DST
            | TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });

    let view = texture.create_view(&TextureViewDescriptor::default());
    (texture, view)
}

// ---------------------------------------------------------------------------
// Helper: fill the display texture with a CPU-generated gradient
// ---------------------------------------------------------------------------

/// Writes an RGBA gradient into `texture`:
///   R = x / width,  G = y / height,  B = 0.5,  A = 255.
/// Used in Phase 2 to verify the display pipeline end-to-end.
/// Retained for debugging; superseded by the compute shader in Phase 3+.
#[allow(dead_code)]
fn fill_gradient(queue: &wgpu::Queue, texture: &wgpu::Texture, width: u32, height: u32) {
    // Cast to usize before multiplying to avoid u32 overflow at large resolutions.
    let mut pixels: Vec<u8> = Vec::with_capacity(width as usize * height as usize * 4);
    for y in 0..height {
        for x in 0..width {
            let r = ((x as f32 / width as f32) * 255.0) as u8;
            let g = ((y as f32 / height as f32) * 255.0) as u8;
            pixels.extend_from_slice(&[r, g, 128, 255]);
        }
    }

    queue.write_texture(
        TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        &pixels,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * width),
            rows_per_image: None, // only meaningful for 3D/array textures
        },
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
}

// ---------------------------------------------------------------------------
// Phase 3: camera uniform
// ---------------------------------------------------------------------------

/// GPU-side camera layout.  All fields are vec4 so the struct is naturally
/// 16-byte aligned without any padding — safe to cast directly with bytemuck.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    /// Eye position in world space.  `.w` is unused padding.
    origin: [f32; 4],
    /// Lower-left corner of the virtual screen in world space.  `.w` unused.
    lower_left: [f32; 4],
    /// Full horizontal extent of the virtual screen (right − left).  `.w` unused.
    horizontal: [f32; 4],
    /// Full vertical extent of the virtual screen (top − bottom).  `.w` unused.
    vertical: [f32; 4],
}

/// Compute the camera uniform for a pinhole camera.
///
/// - `look_from`: eye position
/// - `look_at`:   focus point
/// - `vfov`:      vertical field of view in degrees
/// - `aspect`:    viewport width / height
fn compute_camera(width: u32, height: u32) -> CameraUniform {
    let aspect = width as f32 / height as f32;

    let look_from = Vec3::new(0.0, 1.0, 3.0);
    let look_at = Vec3::new(0.0, 0.0, 0.0);
    let vup = Vec3::new(0.0, 1.0, 0.0);
    let vfov_rad = 60.0_f32.to_radians();

    // Half-height of the near plane at unit distance from the eye.
    let h = (vfov_rad * 0.5).tan();
    let viewport_height = 2.0 * h;
    let viewport_width = aspect * viewport_height;

    // Orthonormal camera basis (right-handed, Z points toward the viewer).
    let w = (look_from - look_at).normalize(); // backward
    let u = vup.cross(w).normalize(); // right
    let v = w.cross(u); // up

    let horizontal = viewport_width * u;
    let vertical = viewport_height * v;
    // Lower-left corner of the virtual screen in world space.
    let lower_left = look_from - horizontal * 0.5 - vertical * 0.5 - w;

    CameraUniform {
        origin: look_from.extend(0.0).to_array(),
        lower_left: lower_left.extend(0.0).to_array(),
        horizontal: horizontal.extend(0.0).to_array(),
        vertical: vertical.extend(0.0).to_array(),
    }
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

    // Phase 2: display texture pipeline
    display_texture: wgpu::Texture,
    display_texture_view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    /// Kept so the bind group can be cheaply recreated on resize.
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    render_pipeline: wgpu::RenderPipeline,

    // Phase 3: compute (path tracer)
    camera_buffer: wgpu::Buffer,
    /// Kept for cheap bind-group recreation on resize.
    compute_bind_group_layout: wgpu::BindGroupLayout,
    compute_bind_group: wgpu::BindGroup,
    compute_pipeline: wgpu::ComputePipeline,
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

        // ---- display texture -----------------------------------------------
        let (display_texture, display_texture_view) =
            create_display_texture(&device, config.width, config.height);

        // ---- sampler -------------------------------------------------------
        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("display sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..Default::default()
        });

        // ---- bind group layout --------------------------------------------
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("display bgl"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // ---- bind group ----------------------------------------------------
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("display bg"),
            layout: &bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&display_texture_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&sampler),
                },
            ],
        });

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
        // Sized to hold one CameraUniform; updated whenever the viewport changes.
        let camera_uniform = compute_camera(config.width, config.height);
        let camera_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("camera buffer"),
            size: size_of::<CameraUniform>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&camera_buffer, 0, bytemuck::bytes_of(&camera_uniform));

        // ---- compute bind group layout ------------------------------------
        let compute_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("compute bgl"),
                entries: &[
                    // binding 0: write-only storage texture (the output image)
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::StorageTexture {
                            access: StorageTextureAccess::WriteOnly,
                            format: TextureFormat::Rgba8Unorm,
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
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // ---- compute bind group -------------------------------------------
        let compute_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("compute bg"),
            layout: &compute_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&display_texture_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: camera_buffer.as_entire_binding(),
                },
            ],
        });

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

        Self {
            surface,
            device,
            queue,
            config,
            window,
            display_texture,
            display_texture_view,
            sampler,
            bind_group_layout,
            bind_group,
            render_pipeline,
            camera_buffer,
            compute_bind_group_layout,
            compute_bind_group,
            compute_pipeline,
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

        // Recreate display texture at the new size.
        let (tex, view) = create_display_texture(&self.device, new_width, new_height);
        self.display_texture = tex;
        self.display_texture_view = view;

        // Bind group must reference the new texture view.
        self.bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("display bg"),
            layout: &self.bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&self.display_texture_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&self.sampler),
                },
            ],
        });

        // Update camera aspect ratio for the new viewport size.
        let camera_uniform = compute_camera(new_width, new_height);
        self.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&camera_uniform));

        // Compute bind group also references the (new) texture view.
        self.compute_bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("compute bg"),
            layout: &self.compute_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&self.display_texture_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: self.camera_buffer.as_entire_binding(),
                },
            ],
        });
    }

    fn render(&mut self) -> Result<(), SurfaceError> {
        let frame = self.surface.get_current_texture()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("frame encoder"),
            });

        // ---- compute pass: path tracer writes into the display texture ----
        {
            let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("path trace pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, &self.compute_bind_group, &[]);
            // Ceil-divide so every pixel is covered even for non-multiple-of-8 sizes.
            let wg_x = self.config.width.div_ceil(8);
            let wg_y = self.config.height.div_ceil(8);
            cpass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // ---- render pass: blit the texture to the swap-chain surface -------
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
            pass.set_bind_group(0, &self.bind_group, &[]);
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
                    // Out of memory — not recoverable.
                    Err(SurfaceError::OutOfMemory) => {
                        log::error!("Out of GPU memory — exiting");
                        event_loop.exit();
                    }
                    Err(e) => log::warn!("Transient render error: {e:?}"),
                }
            }

            _ => {}
        }
    }

    /// Called after all pending events for a frame are processed.
    /// Requesting a redraw here drives the render loop at display refresh rate
    /// without needing to set ControlFlow::Poll explicitly.
    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = self.state.as_ref() {
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
