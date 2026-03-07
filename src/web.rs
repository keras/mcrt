// web.rs — WASM browser entry point
//
// Registered as `#[wasm_bindgen(start)]` so the browser calls it immediately
// after the WASM module is instantiated.  Replaces `main()` / `App` for the
// web target:
//
//   1. Installs the panic hook and console logger.
//   2. Reads an optional `?scene=<name>` query parameter (defaults to
//      "cornell-box").
//   3. Spawns an async task (via wasm_bindgen_futures) that:
//        a. Pre-fetches the scene YAML via the browser Fetch API.
//        b. Awaits GPU adapter + device initialisation (WebGPU is async).
//        c. Stores the resulting `GpuState` and launches the winit event loop.
//
// No changes to `app.rs`, `gpu.rs` logic, or any shader are required — the
// same path-tracer runs unchanged; only the entry-point wiring differs.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use wasm_bindgen::prelude::*;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};
use winit::platform::web::EventLoopExtWebSys as _;
use winit::platform::web::WindowAttributesExtWebSys as _;

use crate::gpu::{GpuState, DEFAULT_WINDOW_HEIGHT, DEFAULT_WINDOW_WIDTH};

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Called by the browser as soon as the WASM binary is ready.
#[wasm_bindgen(start)]
pub fn web_main() {
    console_error_panic_hook::set_once();
    let _ = console_log::init_with_level(log::Level::Info);

    let scene_path = scene_from_query_string()
        .unwrap_or_else(|| "assets/cornell-box.yaml".to_string());

    let event_loop = EventLoop::new().expect("failed to create event loop");
    let app = WasmApp {
        state: Rc::new(RefCell::new(None)),
        window_slot: Rc::new(RefCell::new(None)),
        scene_path,
    };
    // `spawn_app` is non-blocking on WASM — it registers the handler with
    // requestAnimationFrame and returns immediately, rather than blocking the
    // JS event loop.
    event_loop.spawn_app(app);
}

// ---------------------------------------------------------------------------
// Application handler
// ---------------------------------------------------------------------------

/// Minimal winit `ApplicationHandler` for the browser.
///
/// On the first `resumed()` call it attaches to `<canvas id="canvas">` and
/// spawns an async task that initialises the GPU.  Subsequent event callbacks
/// delegate to `GpuState` methods — identical to `app.rs` on native.
struct WasmApp {
    /// The fully initialised GPU renderer; `None` until async init completes.
    state: Rc<RefCell<Option<GpuState>>>,
    /// The window (and canvas); `None` until `resumed()` creates it.
    window_slot: Rc<RefCell<Option<Arc<Window>>>>,
    /// YAML scene file path to load (e.g. `"assets/cornell-box.yaml"`).
    scene_path: String,
}

impl ApplicationHandler for WasmApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Guard: `resumed` can fire more than once on some platforms.
        if self.window_slot.borrow().is_some() {
            return;
        }

        // Attach to the <canvas id="canvas"> element declared in index.html.
        let canvas = web_sys::window()
            .and_then(|w| w.document())
            .and_then(|d| d.get_element_by_id("canvas"))
            .and_then(|e| e.dyn_into::<web_sys::HtmlCanvasElement>().ok())
            .expect("no <canvas id=\"canvas\"> found in the page");

        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("mcrt — path tracer")
                        .with_inner_size(winit::dpi::LogicalSize::new(
                            DEFAULT_WINDOW_WIDTH,
                            DEFAULT_WINDOW_HEIGHT,
                        ))
                        .with_canvas(Some(canvas)),
                )
                .expect("failed to create window"),
        );
        *self.window_slot.borrow_mut() = Some(Arc::clone(&window));

        // Launch async GPU init.  The closure captures `Rc`s so it cannot be
        // sent across threads — that's fine; WASM is single-threaded.
        let state_rc = Rc::clone(&self.state);
        let scene_path = self.scene_path.clone();
        wasm_bindgen_futures::spawn_local(async move {
            // Pre-fetch the scene YAML so `platform::load_bytes` can serve it
            // synchronously later when the scene is parsed inside `new_async`.
            match crate::platform::fetch_bytes(&scene_path).await {
                Ok(bytes) => crate::platform::cache_asset(&scene_path, bytes),
                Err(e) => log::warn!("scene pre-fetch failed ({e}); load will fail"),
            }

            let gpu = GpuState::new_async(window, scene_path).await;
            *state_rc.borrow_mut() = Some(gpu);
            log::info!("GPU state initialised");
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let mut borrow = self.state.borrow_mut();
        let Some(state) = borrow.as_mut() else { return };

        let egui_consumed = state.handle_window_event_egui(&event);

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                state.resize(size.width, size.height);
            }
            WindowEvent::RedrawRequested => {
                match state.render() {
                    Ok(()) => {}
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        let (w, h) = (state.surface_width(), state.surface_height());
                        state.resize(w, h);
                    }
                    Err(wgpu::SurfaceError::Timeout) => {
                        state.request_redraw();
                    }
                    Err(wgpu::SurfaceError::OutOfMemory) => {
                        log::error!("Out of GPU memory — exiting");
                        event_loop.exit();
                    }
                    Err(e) => log::warn!("Transient render error: {e:?}"),
                }
            }
            WindowEvent::MouseInput { button, state: btn_state, .. } if !egui_consumed => {
                state.handle_mouse_button(button, btn_state);
            }
            WindowEvent::CursorMoved { position, .. } if !egui_consumed => {
                state.handle_cursor_moved(position);
            }
            WindowEvent::MouseWheel { delta, .. } if !egui_consumed => {
                state.handle_mouse_wheel(delta);
            }
            WindowEvent::KeyboardInput { event, .. } if !egui_consumed => {
                state.handle_key(event.physical_key, event.state == ElementState::Pressed);
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        // Drive continuous rendering and WASD translation.
        if let Some(window) = self.window_slot.borrow().as_ref() {
            window.request_redraw();
        }
        if let Some(state) = self.state.borrow_mut().as_mut() {
            state.apply_movement();
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract `?scene=<name>` from the current URL and return the asset path.
///
/// Returns `None` when the query string is absent or does not contain `scene`.
fn scene_from_query_string() -> Option<String> {
    let location = web_sys::window()?.location();
    let search = location.search().ok()?;
    if search.is_empty() {
        return None;
    }
    let params = web_sys::UrlSearchParams::new_with_str(&search).ok()?;
    let name = params.get("scene")?;
    Some(format!("assets/{name}.yaml"))
}
