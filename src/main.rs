// main.rs — entry point and module tree
//
// The application is split into focused modules:
//
//   camera   — CameraUniform and compute_camera() (pure, testable, no wgpu)
//   scene    — GpuSphere, build_scene, and load_scene_from_yaml (Phase 12)
//   material — GpuMaterial / GpuMaterialData and build_materials()
//   bvh      — GpuBvhNode and build_bvh() (Phase 9 SAH acceleration structure)
//   mesh     — GpuVertex, GpuTriangle, build_mesh_bvh(), UV-sphere / OBJ loader
//              (Phase 10 triangle mesh support)
//   texture  — Albedo texture layers and HDR env map loading / generation
//              (Phase 11 textures & environment maps)
//   gpu      — GpuState: all wgpu resources, render loop, input methods
//   app      — App: winit ApplicationHandler, window lifecycle, event dispatch
//   headless — Phase RT-2: offline PNG renderer (no window, deterministic)

mod app;
mod bvh;
mod camera;
mod gpu;
mod headless;
mod material;
mod mesh;
mod scene;
mod texture;

use winit::event_loop::EventLoop;

/// Default scene file loaded when `--load-scene-yaml` is not supplied.
const DEFAULT_SCENE: &str = "assets/scene.yaml";

fn main() {
    env_logger::init();

    let args: Vec<String> = std::env::args().skip(1).collect();

    // -----------------------------------------------------------------------
    // Headless mode: --headless <scene.yaml> --output <out.png>
    //                           [--width W] [--height H] [--spp N]
    //
    // Detected before the EventLoop is created so no windowing system is
    // initialised at all.
    // -----------------------------------------------------------------------
    if args.contains(&"--headless".to_string()) {
        let scene = get_next_after(&args, "--headless").unwrap_or_else(|| {
            eprintln!("error: --headless requires a scene YAML path");
            std::process::exit(1);
        });
        let output = get_next_after(&args, "--output").unwrap_or_else(|| {
            eprintln!("error: --headless mode requires --output <path>");
            std::process::exit(1);
        });
        let width: Option<u32> = get_next_after(&args, "--width")
            .and_then(|s| s.parse().ok());
        let height: Option<u32> = get_next_after(&args, "--height")
            .and_then(|s| s.parse().ok());
        let spp: Option<u32> = get_next_after(&args, "--spp")
            .and_then(|s| s.parse().ok());

        if let Err(e) = headless::run(scene, output, width, height, spp) {
            eprintln!("headless render failed: {e}");
            std::process::exit(1);
        }
        return;
    }

    // -----------------------------------------------------------------------
    // Windowed (interactive) mode.
    // -----------------------------------------------------------------------
    let scene_path = get_next_after(&args, "--load-scene-yaml")
        .unwrap_or_else(|| DEFAULT_SCENE.to_string());

    let event_loop = EventLoop::new().expect("failed to create event loop");
    let mut app = app::App::new(scene_path);
    event_loop.run_app(&mut app).expect("event loop error");
}

/// Return the argument immediately following `flag` in `args`, or `None`.
fn get_next_after(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}
