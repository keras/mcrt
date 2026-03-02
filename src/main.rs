use std::sync::Arc;

use log::info;
use wgpu::{
    Color, CommandEncoderDescriptor, DeviceDescriptor, InstanceDescriptor, LoadOp, Operations,
    PowerPreference, RenderPassColorAttachment, RenderPassDescriptor, RequestAdapterOptions,
    StoreOp, SurfaceError,
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

// ---------------------------------------------------------------------------
// GPU state — created inside `resumed()`, lives for the rest of the session.
// ---------------------------------------------------------------------------

struct GpuState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
}

impl GpuState {
    fn new(window: Arc<Window>) -> Self {
        // The surface borrows from the Arc's heap allocation which lives as
        // long as the Arc itself. Casting to 'static is safe here because:
        //   1. `surface` and `window` (the Arc) are stored together in the
        //      same struct, so the Arc (and its heap data) is always alive.
        //   2. We never expose the surface outside this struct.
        let instance = wgpu::Instance::new(&InstanceDescriptor::default());

        // SAFETY: we hold an `Arc<Window>`; the surface is stored alongside
        // the Arc in the same struct, so the window data outlives the surface.
        let surface = unsafe {
            instance
                .create_surface_unsafe(
                    wgpu::SurfaceTargetUnsafe::from_window(window.as_ref()).unwrap(),
                )
                .expect("failed to create surface")
        };

        let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .expect("no suitable GPU adapter found");

        info!("Adapter: {}", adapter.get_info().name);

        let (device, queue) = pollster::block_on(adapter.request_device(
            &DeviceDescriptor {
                label: Some("mcrt device"),
                ..Default::default()
            },
        ))
        .expect("failed to create device");

        let size = window.inner_size();
        let config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .expect("surface not supported by adapter");

        surface.configure(&device, &config);

        Self {
            window,
            surface,
            device,
            queue,
            config,
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

        {
            let _pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("clear pass"),
                multiview_mask: None,
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            // Render pass ends here when `_pass` is dropped.
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
        _window_id: WindowId,
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
