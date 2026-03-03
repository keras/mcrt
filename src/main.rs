// main.rs — entry point and module tree
//
// The application is split into focused modules:
//
//   camera   — CameraUniform and compute_camera() (pure, testable, no wgpu)
//   scene    — GpuSphere and build_scene / build_large_scene
//   material — GpuMaterial / GpuMaterialData and build_materials()
//   bvh      — GpuBvhNode and build_bvh() (Phase 9 SAH acceleration structure)
//   gpu      — GpuState: all wgpu resources, render loop, input methods
//   app      — App: winit ApplicationHandler, window lifecycle, event dispatch

mod app;
mod bvh;
mod camera;
mod gpu;
mod material;
mod scene;

use winit::event_loop::EventLoop;

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().expect("failed to create event loop");
    let mut app = app::App::default();
    event_loop.run_app(&mut app).expect("event loop error");
}
