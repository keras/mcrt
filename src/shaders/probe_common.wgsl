// probe_common.wgsl — Phase IC-1+: Irradiance probe structs and helpers
//
// Shared by path_trace.wgsl and eventually probe_update.wgsl.

struct Probe {
    // xyz = world-space centre; w = cascade index.
    position_cascade: vec4<f32>,
    // Index into flat radiance (SH) buffer.
    radiance_offset: u32,
    // Index into flat depth (octahedral) buffer.
    depth_offset: u32,
    // Bitmask: dirty / valid / invalid / needs-relocation.
    flags: u32,
    _pad: u32,
}

struct ProbeGrid {
    // .xyz = origin, .w = spacing
    origin_spacing: vec4<f32>,
    // .xyz = dimensions, .w = total probes
    dims_count: vec4<u32>,
    // Debug visualization flags
    show_grid: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// Global bindings (filled by Phase IC-1 build_scene_buffers)
@group(0) @binding(14) var<uniform> grid: ProbeGrid;
// binding 15 is used by the radiance buffer, but for IC-1 debug visualization
// we only need the grid uniform to draw the positions.

fn get_probe_pos(index: u32) -> vec3<f32> {
    let z = index / (grid.dims_count.x * grid.dims_count.y);
    let rem = index % (grid.dims_count.x * grid.dims_count.y);
    let y = rem / grid.dims_count.x;
    let x = rem % grid.dims_count.x;

    return grid.origin_spacing.xyz + vec3<f32>(f32(x), f32(y), f32(z)) * grid.origin_spacing.w;
}
