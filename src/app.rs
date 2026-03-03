// app.rs — winit ApplicationHandler: window lifecycle + event dispatch
//
// `App` is the top-level application struct required by the winit 0.30
// event-loop API.  It owns an `Option<GpuState>` created on the first
// `resumed()` call and held until exit.  No GPU code lives here — all
// rendering and input logic is delegated to `GpuState` methods.

use std::sync::Arc;

use log::info;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, WindowEvent},
    event_loop::ActiveEventLoop,
    window::{Window, WindowId},
};

use crate::gpu::{GpuState, DEFAULT_WINDOW_HEIGHT, DEFAULT_WINDOW_WIDTH};

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

/// Top-level application state.  Created before the event loop starts.
pub struct App {
    /// Path of the YAML scene file to load; forwarded to [`GpuState::new`].
    scene_path: String,
    state: Option<GpuState>,
}

impl App {
    /// Create the application with the given scene file path.
    pub fn new(scene_path: String) -> Self {
        Self { scene_path, state: None }
    }
}

impl ApplicationHandler for App {
    // -----------------------------------------------------------------------
    // Window / surface lifecycle
    // -----------------------------------------------------------------------

    /// Called when the OS has a surface ready for rendering.
    ///
    /// This is the only correct place to create the window and the wgpu device.
    /// On desktop platforms it fires once at startup; on Android it can fire
    /// multiple times (foreground → background → foreground) — the existing
    /// `is_some()` guard makes the subsequent calls no-ops.
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return; // already initialised
        }

        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("mcrt — path tracer")
                        .with_inner_size(winit::dpi::LogicalSize::new(
                            DEFAULT_WINDOW_WIDTH,
                            DEFAULT_WINDOW_HEIGHT,
                        )),
                )
                .expect("failed to create window"),
        );

        self.state = Some(GpuState::new(window, self.scene_path.clone()));
        info!("GPU state initialised");
    }

    // -----------------------------------------------------------------------
    // Window events
    // -----------------------------------------------------------------------

    fn window_event(
        &mut self,
        event_loop:  &ActiveEventLoop,
        _window_id:  WindowId, // single window — no need to dispatch by ID
        event:       WindowEvent,
    ) {
        let Some(state) = self.state.as_mut() else { return };

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
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        let (w, h) = (state.surface_width(), state.surface_height());
                        state.resize(w, h);
                    }
                    // GPU timed out transiently: request next frame.
                    Err(wgpu::SurfaceError::Timeout) => {
                        state.request_redraw();
                    }
                    // Out of memory — unrecoverable.
                    Err(wgpu::SurfaceError::OutOfMemory) => {
                        log::error!("Out of GPU memory — exiting");
                        event_loop.exit();
                    }
                    Err(e) => log::warn!("Transient render error: {e:?}"),
                }
            }

            // ---- orbit interaction ----------------------------------------

            WindowEvent::MouseInput { button, state: btn_state, .. } => {
                state.handle_mouse_button(button, btn_state);
            }

            WindowEvent::CursorMoved { position, .. } => {
                state.handle_cursor_moved(position);
            }

            WindowEvent::MouseWheel { delta, .. } => {
                state.handle_mouse_wheel(delta);
            }

            WindowEvent::KeyboardInput { event, .. } => {
                state.handle_key(event.physical_key, event.state == ElementState::Pressed);
            }

            _ => {}
        }
    }

    // -----------------------------------------------------------------------
    // Per-tick update
    // -----------------------------------------------------------------------

    /// Called after all pending events for a frame are processed.
    ///
    /// Continuous WASD translation is applied here (once per event-loop tick)
    /// so movement is smooth as long as a key is held.  The redraw request
    /// drives the render loop at display refresh rate.
    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = self.state.as_mut() {
            state.apply_movement();
            state.request_redraw();
        }
    }
}
