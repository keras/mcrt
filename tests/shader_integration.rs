//! GPU shader integration tests.
//!
//! Two test tiers:
//!
//! 1. **Naga offline validation** (`naga_parses_*`) — parses and validates each
//!    WGSL source file using the naga compiler.  These tests run in any
//!    environment (no GPU required) and catch WGSL syntax/semantic errors that
//!    are invisible to `cargo build` because wgpu only compiles shaders at
//!    runtime.
//!
//! 2. **Headless wgpu pipeline creation** (`wgpu_*_pipeline_creation`) —
//!    requests a headless wgpu adapter, recreates the same bind-group layouts
//!    and compute pipelines used by `GpuState`, and asserts that the pipelines
//!    compile without a validation error.  These tests skip gracefully when no
//!    GPU adapter is available (e.g. CI without hardware/software rendering).

// ---------------------------------------------------------------------------
// Tier 1 — naga offline WGSL validation (no GPU required)
// ---------------------------------------------------------------------------

/// Parse and validate a WGSL source string.  Panics with a descriptive message
/// on any parse or validation error — which is all we need for test assertions.
fn validate_wgsl(label: &str, src: &str) {
    use naga::valid::{Capabilities, ValidationFlags, Validator};

    let module = naga::front::wgsl::parse_str(src)
        .unwrap_or_else(|e| panic!("WGSL parse error in {label}:\n{e}"));

    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::default());
    validator
        .validate(&module)
        .unwrap_or_else(|e| panic!("WGSL validation error in {label}:\n{e:#?}"));
}

#[test]
fn naga_parses_path_trace_shader() {
    validate_wgsl(
        "path_trace.wgsl",
        include_str!("../src/shaders/path_trace.wgsl"),
    );
}

#[test]
fn naga_parses_denoise_shader() {
    validate_wgsl(
        "denoise.wgsl",
        include_str!("../src/shaders/denoise.wgsl"),
    );
}

#[test]
fn naga_parses_display_shader() {
    validate_wgsl(
        "display.wgsl",
        include_str!("../src/shaders/display.wgsl"),
    );
}

#[test]
fn naga_parses_probe_update_shader() {
    validate_wgsl(
        "probe_update.wgsl",
        include_str!("../src/shaders/probe_update.wgsl"),
    );
}

// ---------------------------------------------------------------------------
// Tier 2 — headless wgpu pipeline creation
// ---------------------------------------------------------------------------

use wgpu::{
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, BufferBindingType,
    ComputePipelineDescriptor, DeviceDescriptor, Features, InstanceDescriptor, Limits,
    PipelineLayoutDescriptor, PowerPreference, RequestAdapterOptions, SamplerBindingType,
    ShaderModuleDescriptor, ShaderSource, ShaderStages, StorageTextureAccess, TextureFormat,
    TextureSampleType, TextureViewDimension,
};

/// Request a headless `(Device, Queue)` pair.
///
/// Returns `None` when no adapter is available so callers can skip gracefully.
fn headless_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(&InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
        power_preference: PowerPreference::None,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok()?;
    let (device, queue) = pollster::block_on(adapter.request_device(&DeviceDescriptor {
        label: Some("shader-test device"),
        required_features: Features::empty(),
        required_limits: Limits::default(),
        ..Default::default()
    }))
    .ok()?;
    Some((device, queue))
}

/// Recreates the 14-entry compute bind-group layout used by the path-trace pass.
///
/// Uses `min_binding_size: None` throughout so that the test does not need to
/// import the GPU-side struct definitions from the binary crate.  wgpu will
/// still validate that every binding's *type* (storage vs uniform, texture
/// format, read-only flag, etc.) matches the shader declarations.
fn make_path_trace_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("path trace bgl (test)"),
        entries: &[
            // 0: write-only storage texture — accumulation write target
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
            // 1: camera uniform
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 2: sphere list (BVH-reordered, read-only storage)
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 3: previous-frame accumulation texture (non-filterable read)
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
            // 4: material table uniform
            BindGroupLayoutEntry {
                binding: 4,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 5: sphere BVH nodes (read-only storage)
            BindGroupLayoutEntry {
                binding: 5,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 6: mesh vertices (position / normal / UV; read-only storage)
            BindGroupLayoutEntry {
                binding: 6,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 7: mesh triangle index triples (read-only storage)
            BindGroupLayoutEntry {
                binding: 7,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 8: mesh BVH node array (read-only storage)
            BindGroupLayoutEntry {
                binding: 8,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 9: linear/repeat sampler for the albedo texture array
            BindGroupLayoutEntry {
                binding: 9,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            },
            // 10: albedo 2D-array texture (RGBA8 Unorm, filterable)
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
            // 11: HDR equirectangular env map (Rgba32Float, non-filterable)
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
            // 12: emissive sphere list for NEE (read-only storage)
            BindGroupLayoutEntry {
                binding: 12,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 13: G-buffer write target (normal.xyz + linear depth.w)
            BindGroupLayoutEntry {
                binding: 13,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    format: TextureFormat::Rgba32Float,
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            },
            // 14: Probe grid uniform (IC-1)
            BindGroupLayoutEntry {
                binding: 14,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 15: Probe irradiance storage (IC-1)
            BindGroupLayoutEntry {
                binding: 15,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

/// Recreates the 4-entry bind-group layout used by the denoiser pass.
fn make_denoise_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("denoise bgl (test)"),
        entries: &[
            // 0: accumulated radiance texture (non-filterable)
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
            // 1: G-buffer (normal xyz + depth w, non-filterable)
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
            // 2: denoised output write target
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
            // 3: DenoiseParams uniform buffer (runtime-tunable knobs, TD-5)
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

/// Verifies that the path-trace shader compiles and the pipeline can be created
/// with the expected 14-binding layout.
///
/// Skips if no wgpu adapter is available (normal in headless CI environments).
#[test]
fn wgpu_path_trace_pipeline_creation() {
    let Some((device, _queue)) = headless_device() else {
        eprintln!("wgpu_path_trace_pipeline_creation: no GPU adapter — skipping");
        return;
    };

    let bgl = make_path_trace_bgl(&device);
    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("path_trace.wgsl (test)"),
        source: ShaderSource::Wgsl(include_str!("../src/shaders/path_trace.wgsl").into()),
    });
    let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("path trace layout (test)"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });
    let _pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("path trace pipeline (test)"),
        layout: Some(&layout),
        module: &shader,
        entry_point: Some("cs_main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Flush any deferred wgpu validation callbacks.
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
}

/// Verifies that the denoiser shader compiles with its 4-binding layout (TD-5 adds params uniform).
///
/// Skips if no wgpu adapter is available.
#[test]
fn wgpu_denoise_pipeline_creation() {
    let Some((device, _queue)) = headless_device() else {
        eprintln!("wgpu_denoise_pipeline_creation: no GPU adapter — skipping");
        return;
    };

    let bgl = make_denoise_bgl(&device);
    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("denoise.wgsl (test)"),
        source: ShaderSource::Wgsl(include_str!("../src/shaders/denoise.wgsl").into()),
    });
    let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("denoise layout (test)"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });
    let _pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("denoise pipeline (test)"),
        layout: Some(&layout),
        module: &shader,
        entry_point: Some("cs_denoise"),
        compilation_options: Default::default(),
        cache: None,
    });

    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
}
