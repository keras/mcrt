//! mcrt — library surface.
//!
//! Exposes the `regression` module so that integration tests in `tests/` can
//! import `mcrt::regression::{compare_images, write_diff_image, load_thresholds, …}`
//! without duplicating the implementation.
//!
//! Only modules that carry no wgpu/winit dependency are re-exported here.
//! The rendering pipeline modules (`gpu`, `headless`, etc.) remain binary-only
//! to avoid pulling heavyweight GPU crates into the integration-test compile graph.

#[cfg(not(target_arch = "wasm32"))]
pub mod regression;
