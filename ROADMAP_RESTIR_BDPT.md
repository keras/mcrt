# ReSTIR BDPT (Bidirectional Path Resampling) — Roadmap

Integrate **Bidirectional Path Tracing** (BDPT) with **Reservoir-based
Spatiotemporal Importance Resampling** (ReSTIR) into the existing wgpu
compute-shader path tracer.  By tracing subpaths from both the camera and the
light sources, connecting them via multiple strategies, and resampling those
connections across pixels and frames using reservoirs, **ReSTIR BDPT** captures
light transport phenomena that unidirectional methods struggle with — caustics,
SDS paths, indirect illumination through small openings — while maintaining
interactive-rate performance through spatiotemporal reuse.

> **Key papers:**
>
> Veach & Guibas — *"Optimally Combining Sampling Techniques for Monte Carlo
> Rendering"*, SIGGRAPH 1995.  (Foundation of BDPT and MIS.)
>
> Veach — *"Robust Monte Carlo Methods for Light Transport Simulation"*, PhD
> thesis, Stanford 1997.  (Canonical BDPT reference.)
>
> Lin, Kettunen, Bitterli, Pantaleoni, Yuksel, Wyman — *"Generalized
> Resampled Importance Sampling: Foundations of ReSTIR"*, ACM TOG
> (SIGGRAPH 2022).  (Theoretical framework enabling unbiased reservoir
> resampling over arbitrary integration domains — the basis for applying
> ReSTIR to bidirectional paths.)
>
> West, Grünschloß, Pharr, Wyman — *"ReSTIR BDPT: Resampled Bidirectional
> Path Tracing"*, (concept outlined in SIGGRAPH 2023 course notes on
> ReSTIR path reuse).

---

## Background & Motivation

### Current state

The path tracer (`path_trace.wgsl`) traces one **unidirectional** sample per
pixel per frame and progressively accumulates a running average.  Phase 13
added Next-Event Estimation (NEE) for direct illumination from emissive
spheres, but the renderer still relies solely on camera-originated paths.

### Limitations of unidirectional tracing

- **Caustics** (specular-diffuse-specular paths) are found only by chance —
  the camera path must randomly hit the focused caustic region.
- **SDS paths** (specular-diffuse-specular) contribute high-variance fireflies.
- **Small openings / indirect light guides** converge extremely slowly because
  the camera path is unlikely to pass through the opening and then reach a
  light.
- **Light-side transport** (photon-side phenomena like subsurface scattering
  from area lights) is entirely missed.

### Why BDPT

Bidirectional Path Tracing constructs paths by:

1. Tracing a **camera subpath** $x_0, x_1, \ldots, x_t$ from the lens.
2. Tracing a **light subpath** $y_0, y_1, \ldots, y_s$ from an emitter.
3. **Connecting** a camera vertex $x_i$ to a light vertex $y_j$ to form a
   complete path of length $i + j + 1$.
4. Combining all $(s, t)$ connection strategies via **Multiple Importance
   Sampling** (MIS) using the balance or power heuristic, ensuring each
   strategy contributes where it is most efficient.

This naturally finds caustics (light subpath hits the specular surface),
handles SDS paths, and generally reduces variance for complex light transport.

### Why ReSTIR × BDPT

BDPT produces many candidate connections per pixel (one for every valid
$(s, t)$ pair).  ReSTIR provides the machinery to:

1. **Select** the best connection per pixel from all $(s, t)$ candidates
   using Resampled Importance Sampling (RIS), weighting by the luminance of
   the full path contribution.
2. **Reuse** good light subpaths and connections across neighbouring pixels
   (spatial reuse) and across frames (temporal reuse) via reservoirs.
3. Maintain **unbiased** estimates through proper shift mappings and MIS
   weights derived from the GRIS framework.

The net effect: BDPT-quality rendering of caustics and complex transport at
interactive frame rates, with noise levels equivalent to tens or hundreds of
independent BDPT samples per pixel.

---

## Terminology Quick Reference

| Term | Meaning |
|------|---------|
| **BDPT** | Bidirectional Path Tracing — trace subpaths from camera and light, connect them |
| **$(s, t)$ technique** | A connection strategy: $s$ light subpath vertices, $t$ camera subpath vertices, connected by a shadow ray (or directly for $s{=}0$ or $t{=}1$) |
| **Camera subpath** | Vertices $x_0$ (lens) $\to x_1$ (primary hit) $\to \ldots \to x_t$ traced via BSDF sampling from the camera |
| **Light subpath** | Vertices $y_0$ (emitter surface) $\to y_1 \to \ldots \to y_s$ traced via BSDF sampling from a light source |
| **Connection** | A shadow ray between camera vertex $x_i$ and light vertex $y_j$; if unoccluded, forms a full transport path |
| **RIS** | Resampled Importance Sampling — select 1 of M candidates with probability proportional to a target PDF |
| **Reservoir** | Compact data structure: selected sample, sum of weights ($w_\text{sum}$), confidence weight ($M$), output weight ($W$) |
| **Target function** $\hat{p}$ | Unnormalised PDF used for resampling; typically luminance of the full path contribution for the connection |
| **Shift mapping** $T_{q \to p}$ | Transforms a connection sampled at pixel $q$ so it is valid at pixel $p$; compensated by a Jacobian $\|J_T\|$ |
| **Light subpath reuse** | Sharing a light subpath traced at one pixel with neighbouring pixels; only the connection segment changes |
| **Confidence weight ($M$)** | Effective sample count in a reservoir; capped to prevent stale temporal accumulation |
| **G-buffer** | Per-pixel geometry data (normal, depth, material ID, motion vectors) for neighbour validation and reprojection |
| **MIS weight** | Balance- or power-heuristic weight combining multiple $(s,t)$ strategies to minimise variance |

---

## Architecture Overview — Proposed Pipeline

The current single compute pass (`path_trace.wgsl`) will be restructured into
six ordered passes per frame when ReSTIR BDPT is active.  Each pass is a
separate compute dispatch reading from and writing to screen-sized (and
light-subpath-sized) storage buffers.

```
Frame N
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  Pass 1 — Trace Light Subpaths                                       │
│    • For each pixel (or a global pool of L light paths), emit a      │
│      ray from a randomly sampled emitter surface point.              │
│    • Trace up to max_light_bounces bounces via BSDF sampling.        │
│    • Store all light subpath vertices (position, normal, throughput,  │
│      BSDF flags, PRNG state) into the light-vertex buffer.           │
│                                                                      │
│  Pass 2 — Trace Camera Subpaths + G-buffer                           │
│    • Trace 1 camera path per pixel (existing path tracer logic).     │
│    • Record per-bounce metadata: hit position, normal, material,     │
│      throughput, roughness.                                          │
│    • Write the extended G-buffer (normal, depth, material ID,        │
│      motion vector).                                                 │
│                                                                      │
│  Pass 3 — Connect & Build Initial Reservoirs                         │
│    • For each pixel, enumerate (s,t) connections between its camera  │
│      subpath vertices and assigned light subpath vertices.           │
│    • Evaluate full path contribution f(x̄) and MIS weight for each   │
│      connection; compute RIS target p̂ = luminance(f(x̄) · w_MIS).    │
│    • Stream all valid connections into a per-pixel reservoir via RIS. │
│    • The reservoir stores the selected (s,t,light_path_id) tuple,    │
│      the connection endpoints, and the contribution weight.          │
│                                                                      │
│  Pass 4 — Temporal Reuse                                             │
│    • Backproject each pixel via motion vectors.                      │
│    • Load the previous frame's reservoir; shift the stored           │
│      connection to the current pixel's domain.                       │
│    • Merge via streaming RIS + pairwise MIS.                         │
│    • Cap confidence weight M.                                        │
│                                                                      │
│  Pass 5 — Spatial Reuse (1–2 iterations)                             │
│    • For each pixel, pick K neighbours in a screen-space disk.       │
│    • Shift each neighbour's selected connection to the current       │
│      pixel; merge via streaming RIS + pairwise MIS.                  │
│                                                                      │
│  Pass 6 — Shade & Accumulate                                         │
│    • Evaluate final radiance from the selected reservoir connection. │
│    • Blend with progressive accumulation (if camera is static).      │
│    • Write to the accumulation texture for the display pass.         │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
                           ▼
      Display pass (existing display.wgsl / denoise.wgsl)
```

**GPU buffer summary (new):**

| Buffer | Format | Contents |
|--------|--------|----------|
| `light_vertices` | Storage (R/W) | Pool of light subpath vertices: position, normal, throughput, material, PRNG state.  Sized `L × max_light_bounces` where L = pixel count (or a configurable pool size). |
| `light_path_info` | Storage (R/W) | Per-light-path metadata: emitter ID, path length, emission, starting PRNG seed. Sized `L`. |
| `camera_vertices` | Storage (R/W) | Per-pixel camera subpath vertices (position, normal, throughput, BSDF).  Sized `W×H × max_camera_bounces`. |
| `reservoir_curr` | Storage (R/W) | Per-pixel reservoir: selected $(s,t)$ connection, connection endpoints, $w_\text{sum}$, $M$, $W$. |
| `reservoir_prev` | Storage (read) | Previous-frame reservoir (ping-ponged). |
| `reservoir_spatial_temp` | Storage (R/W) | Intermediate reservoir buffer for two-pass spatial reuse; written by spatial pass 1, read by spatial pass 2.  Same size as `reservoir_curr`. |
| `gbuffer` | Storage (R/W) | World normal, linear depth, material ID, screen-space motion vector. |
| `gbuffer_prev` | Storage (read) | Previous-frame G-buffer. |

Buffers are double-buffered (curr/prev) and ping-ponged each frame.  Total
additional GPU memory at 1920×1080 is approximately:
- Light vertices: ~80–120 MB (depending on max bounces and vertex stride)
- Camera vertices: ~40–60 MB
- Reservoirs: ~48–72 MB (3 buffers × W×H × 120 bytes — curr, prev, spatial_temp)
- G-buffers: ~30 MB
- **Total: ~200–280 MB**

---

## Phase RB-0: Bind Group Architecture

**Goal:** Decide and document the final GPU bind group layout before writing any BDPT
code.  All subsequent phases depend on this decision; changing it mid-implementation
requires touching every pipeline and bind group descriptor.

### Design constraints

- The existing compute BGL (Group 0) already has 15 bindings (0–14) used by
  `path_trace.wgsl` and `restir_initial.wgsl`.  WebGPU mandates a maximum of 8
  dynamic storage buffers per pipeline; adding 7+ new BDPT buffers into Group 0 is
  not feasible.
- BDPT passes share some bindings with the classic tracer (BVH, materials, camera)
  but not others (G-buffer variants, reservoir ping-pong).
- Each compute pipeline can use up to 4 bind groups (groups 0–3) in wgpu's default
  limits.

### Proposed layout

| Bind Group | Used by | Contents |
|-----------|---------|----------|
| **Group 0** (unchanged) | All passes | Camera uniform, BVH, materials, spheres, vertices, triangles, textures, env map, emissives — read-only scene data |
| **Group 1** | BDPT passes | `light_vertices` (R/W), `light_path_info` (R/W), `camera_vertices` (R/W), `gbuffer_write` (write), `gbuffer_pos_write` (write) |
| **Group 2** | Connect, Temporal, Spatial, Shade | `reservoir_curr` (R/W), `reservoir_prev` (read), `reservoir_spatial_temp` (R/W), `gbuffer_curr` (read), `gbuffer_prev` (read) |
| **Group 3** | Shade | `accum_write` (write), `accum_read` (read) |

RS-1's existing Group 1 (`restir_initial.wgsl`) is retired when BDPT mode is active;
the classic PT + RS-1 pipelines retain their current layouts unchanged.

### Tasks

- [ ] Document the final bind group layout in `docs/CODE_STRUCTURE.md` (one table per
  pass listing every binding).

- [ ] Implement `make_bdpt_group1_bgl(device)`, `make_bdpt_group2_bgl(device)`,
  `make_bdpt_group3_bgl(device)` in `gpu_layout.rs` following the convention
  established by `make_compute_bgl()` and `make_restir_bgl()`.

- [ ] Verify that no single pipeline exceeds 8 dynamic storage buffer bindings (wgpu
  default limit); adjust grouping if needed.

- [ ] Add a runtime `RenderMode` enum (`ClassicPT`, `ReSTIR_DI`, `ReSTIR_BDPT`) and
  select the matching pipeline set and bind groups each frame.

- [ ] Regression: classic PT and RS-1 ReSTIR DI must still produce correct output
  after the new bind group infrastructure is in place.

**Output:** A locked-down, documented bind group layout that all subsequent BDPT
phases can implement against without further restructuring.

---

## Phase RB-1: BDPT Foundation — Light Subpath Tracing

**Goal:** Implement GPU-side light subpath generation.  A new compute pass
traces paths originating from emissive surfaces and stores the resulting vertex
chain in a structured buffer, ready for connection in later passes.  The
existing unidirectional path tracer continues to run; this pass runs alongside
it without affecting output.

### Light subpath generation

1. **Emitter sampling:** Pick a random emissive primitive (sphere or mesh
   triangle, weighted by power) and sample a point + outgoing direction on its
   surface using cosine-weighted hemisphere sampling.
2. **Tracing:** Walk the light subpath up to `max_light_bounces` (default 5)
   using the same BVH traversal and BSDF sampling routines as the camera path.
3. **Vertex storage:** At each bounce $j$, store:
   - World position $y_j$
   - Surface normal $n_j$
   - Cumulative throughput $\alpha_j^L$ (product of BSDFs and geometry terms
     from $y_0$ to $y_j$)
   - Material type and roughness (for MIS weight computation and shift
     mapping classification)
   - PRNG state at this vertex (for random-replay shifts)

### Light subpath vertex struct (WGSL)

```wgsl
struct LightVertex {
    position:    vec4<f32>,   // .xyz = world pos, .w = path_pdf
    normal_mat:  vec4<f32>,   // .xyz = normal, .w = bitcast material+roughness
    throughput:  vec4<f32>,   // .xyz = α_L cumulative throughput, .w = unused
    misc:        vec4<u32>,   // .x = light_path_id, .y = vertex_index,
                              // .z = prng_state, .w = flags (specular/diffuse)
}
```

### Tasks

- [ ] Define `GpuLightVertex` (`#[repr(C)]`, Pod/Zeroable) in a new Rust
  module `src/restir_bdpt.rs`.

- [ ] Define `GpuLightPathInfo` — per-path header: emitter primitive ID,
  emission colour/strength, number of valid vertices, starting PRNG seed.

- [ ] Reuse the existing `build_light_list()` / power CDF infrastructure from
  `gpu_layout.rs` (RS-1) for emitter sampling on the GPU.  The light-list storage
  buffer and normalized power CDF are already uploaded each frame; bind them into
  the BDPT light-tracing pass rather than building a duplicate.  Sample one emitter +
  surface point + direction per light path using the existing CDF.

- [ ] Implement `cs_trace_light_subpaths` compute shader: traces light
  subpaths and writes `LightVertex` entries.  Reuses the existing BVH
  traversal (`intersect_scene`) and BSDF evaluation (`scatter_*`) code
  via shared WGSL includes.

- [ ] Allocate light-vertex and light-path-info storage buffers in
  `gpu_layout.rs`.  Size: `num_light_paths × max_light_bounces ×
  sizeof(LightVertex)`.

- [ ] Handle buffer recreation on resize.

- [ ] Validation: read back light vertex buffer after one frame; verify
  positions lie on scene geometry and throughputs are physically plausible
  (positive, not NaN/Inf).

**Output:** A pool of light subpath vertices in GPU memory, ready for
bidirectional connection.  Existing rendering unchanged.

---

## Phase RB-2: Camera Subpath Tracing & G-buffer

**Goal:** Refactor the camera-side path tracer to store per-bounce vertex
metadata (not just the final radiance), and write an extended G-buffer with
motion vectors.  This provides the camera side of the bidirectional connection.

### Camera subpath vertex storage

At each bounce $i$ along the camera path, store:
- World position $x_i$
- Surface normal $n_i$
- Cumulative throughput $\alpha_i^E$ (product of BSDFs and geometry terms from
  the camera to $x_i$)
- Material type, roughness, BSDF evaluation data
- Whether this vertex is connectable (diffuse/glossy) or specular-only

### Tasks

- [ ] Define `GpuCameraVertex` struct — similar layout to `LightVertex` but
  with eye-path semantics (throughput from camera, per-vertex BSDF data).

- [ ] Implement `cs_trace_camera_subpaths` compute shader (or extend
  `cs_main` with a compile-time branch) that:
    - Traces the camera path as before.
    - Writes `CameraVertex` entries for each bounce.
    - Writes the extended G-buffer: world normal, linear depth, material ID,
      screen-space motion vector.

- [ ] **Migrate `CameraUniform`**: adding `prev_view_proj` (a `mat4x4<f32>`,
  64 bytes) to the uniform breaks the current 112-byte std140 layout validated in
  `tests/shader_integration.rs`.  Steps:
    1. Extend `CameraUniform` in `gpu_layout.rs` to 176 bytes (112 + 64).
    2. Update the WGSL `camera` binding in `path_trace.wgsl` and any other shader
       that reads it.
    3. Update the shader integration test to expect the new size.
    4. Upload `prev_view_proj` each frame from `GpuState::render()` after computing
       the previous-frame view-projection matrix.

- [ ] Compute **motion vectors**: project the primary-hit world position
  through the previous frame's view-projection matrix stored in `CameraUniform`
  (`prev_view_proj`); store the screen-space delta.

- [ ] Allocate camera-vertex storage buffers in `gpu_layout.rs`.

- [ ] Validation: read back camera vertex buffer; verify primary hit positions
  match the G-buffer depth; verify motion vectors are zero when the camera is
  static.

**Output:** Per-pixel camera subpath vertices and G-buffer in GPU memory.
Existing rendering unchanged (the old `cs_main` still writes the accumulation
texture).

---

## Phase RB-3: Bidirectional Connection & Initial Reservoirs

**Goal:** Implement the core BDPT connection logic: for each pixel, enumerate
valid $(s, t)$ connections between camera and light subpath vertices,
evaluate the full path contribution with MIS weights, and stream all
candidates into a per-pixel reservoir.

### Connection evaluation

For a given pixel with camera subpath length $T$ and an assigned light subpath
of length $S$, the valid connections are:

| $(s, t)$ | Description |
|-----------|-------------|
| $(0, t)$ | Camera path hits an emitter directly (unidirectional; $t \ge 2$) |
| $(1, t)$ | Connect camera vertex $x_{t-1}$ to emitter surface $y_0$ (equivalent to NEE / shadow ray) |
| $(s, t)$ for $s \ge 2, t \ge 2$ | Connect camera vertex $x_{t-1}$ to light vertex $y_{s-1}$ via a shadow ray |
| $(s, 1)$ | Connect light vertex $y_{s-1}$ directly to the camera lens (light tracing / photon mapping — deferred to Phase RB-7) |
| $(s, 0)$ | Light path connects to the lens directly (not applicable for a pinhole; relevant for finite-aperture cameras — deferred) |

For each valid connection:

1. Trace a **shadow ray** from $x_{t-1}$ to $y_{s-1}$ (if $s \ge 2, t \ge 2$).
   Skip if occluded.
2. Evaluate the **full path contribution**:
   $$f(\bar{x}) = \alpha_t^E \cdot f_s(x_{t-1}) \cdot G(x_{t-1} \leftrightarrow y_{s-1}) \cdot f_s(y_{s-1}) \cdot \alpha_s^L \cdot L_e(y_0)$$
   where $G$ is the geometric coupling term (mutual visibility × cosines /
   distance²) and $f_s$ are the BSDFs at the connection vertices.
3. Compute the **MIS weight** $w_{s,t}$ using the balance heuristic over all
   $(s', t')$ techniques that could have generated this path.
4. Compute the **RIS target**: $\hat{p} = \text{luminance}(f(\bar{x}) \cdot w_{s,t})$.
5. Stream into the reservoir with weight $\hat{p} / p_\text{source}$ where
   $p_\text{source}$ is the PDF of generating this particular connection.

### Reservoir struct (WGSL + Rust mirror)

```wgsl
struct BdptReservoir {
    // Selected connection identification
    light_path_id:  u32,       // Index into the light-path pool
    s:              u32,       // Light subpath vertex count in this connection
    t:              u32,       // Camera subpath vertex count in this connection
    sample_seed:    u32,       // PRNG seed for random-replay shifts

    // Connection geometry (cached for shift mappings)
    conn_cam_pos:   vec3<f32>, // Camera-side connection vertex x_{t-1}
    conn_cam_n:     vec3<f32>, // Normal at x_{t-1}
    conn_light_pos: vec3<f32>, // Light-side connection vertex y_{s-1}
    conn_light_n:   vec3<f32>, // Normal at y_{s-1}

    // Path contribution
    radiance:       vec3<f32>, // f(x̄) · w_MIS for the selected connection
    target_value:   f32,       // p̂ = luminance(radiance)

    // Reservoir bookkeeping
    w_sum:          f32,       // Running sum of resampling weights
    M:              f32,       // Confidence weight (effective sample count)
    W:              f32,       // Unbiased contribution weight
    _pad:           f32,
}
```

### Light subpath assignment

Each pixel is assigned one (or a small number of) light subpaths for
connection.  Assignment strategies:

- **1:1 mapping** (simplest): pixel $(x, y)$ uses light path $y \cdot W + x$.
  Every pixel gets a unique light subpath of its own.
- **Shared pool with reuse** (advanced): trace a smaller pool of $L < W{\times}H$
  light paths; assign each pixel a random subset.  Amortises light tracing
  cost but requires care with MIS weights.

Phase RB-3 uses 1:1 mapping.  Shared pools are explored in Phase RB-6.

### MIS weight computation

The balance-heuristic MIS weight for technique $(s, t)$ is:

$$w_{s,t} = \frac{p_{s,t}(\bar{x})}{\sum_{s'=0}^{k} p_{s',k-s'}(\bar{x})}$$

where $k = s + t - 1$ is the path length and $p_{s,t}$ is the PDF of
generating path $\bar{x}$ via strategy $(s,t)$.  Computing the denominator
requires evaluating the PDF of every alternative strategy that could produce
this path — this is the standard BDPT MIS computation from Veach Ch. 10.

For GPU efficiency, use the **recursive MIS weight** formulation that walks
the vertex chain once, avoiding redundant PDF evaluations.

### Tasks

- [ ] Define `GpuBdptReservoir` (`#[repr(C)]`, Pod/Zeroable) in
  `src/restir_bdpt.rs` — mirror the WGSL struct above.

- [ ] Allocate double-buffered reservoir storage buffers in `gpu_layout.rs`.

- [ ] Implement `cs_bdpt_connect` compute shader:
    - For each pixel, loop over valid $(s, t)$ pairs.
    - Evaluate connection geometry (shadow ray), path contribution, MIS weight.
    - Stream each valid connection into the pixel's reservoir via RIS.

- [ ] Implement the **recursive MIS weight** computation in WGSL: walk the
  path vertex chain computing the ratio of PDFs for adjacent techniques.
  This avoids $O(k^2)$ cost per path.

- [ ] Implement **BSDF evaluation at connection vertices**: the connection
  direction is deterministic (points from $x_{t-1}$ to $y_{s-1}$), so the
  BSDF must be evaluated for this specific direction rather than sampled.
  Add `eval_bsdf(material, normal, wo, wi) → f32×3` to the shader.

- [ ] Handle specular vertices: connections through delta-distribution BSDFs
  (perfect mirror/glass) have zero probability and must be skipped.  Only
  connect to/from vertices where the BSDF has a non-zero finite evaluation.

- [ ] **Explicitly exclude $(s, 1)$ from the MIS weight denominator** during this
  phase (it is not yet implemented).  The balance-heuristic sum must only include
  techniques that are actually evaluated; including $(s, 1)$ with zero contribution
  would otherwise inflate the denominator and produce darkened results.  Add a
  compile-time flag (`BDPT_HAS_LIGHT_TRACING = false`) toggled in Phase RB-7 to
  re-enable it when $(s, 1)$ is available.

- [ ] **Initial path length budget:** cap `max_s + max_t ≤ 6` for the first
  correctness pass.  The $(s \times t)$ connection loop is the most expensive step —
  starting with short paths keeps validation fast and avoids workgroup timeouts.
  Use the default 8×8 workgroup for this pass; revisit in Phase RB-10.

- [ ] Validation: render the Cornell box; compare the reservoir's radiance
  output (Pass 6) against the classic unidirectional path tracer.  Also
  verify that caustic illumination (e.g. glass sphere on diffuse floor) is
  present and brighter than the unidirectional baseline.

**Output:** Per-pixel reservoirs populated with BDPT connections.  Image
quality matches or exceeds the unidirectional path tracer, with visible
caustic improvement, even without spatiotemporal reuse.

---

## Phase RB-4: Temporal Reuse

**Goal:** Merge each pixel's current reservoir with the backprojected reservoir
from the previous frame, multiplying the effective sample count over time.

### Shift mapping for BDPT connections

When reusing a connection from a previous frame (or a different pixel), the
connection geometry changes because the camera subpath differs.  The **shift
mapping** $T_{q \to p}$ adapts a stored connection to a new pixel $p$:

1. **Camera-side reconnection:** The selected connection had camera vertex
   $x_{t-1}^q$ at pixel $q$.  At pixel $p$, the camera subpath is different,
   so use pixel $p$'s camera vertex $x_{t-1}^p$ instead.  The light side
   ($y_{s-1}$) remains unchanged.
2. **Recompute the connection:** Trace a shadow ray from $x_{t-1}^p$ to
   $y_{s-1}$; if occluded, the shifted connection is invalid.
3. **Re-evaluate the path contribution** $f(\bar{x}')$ and MIS weight using
   the new camera vertex and the original light vertex.
4. **Jacobian:** The shift changes the solid angle at the connection:
   $$|J_T| = \frac{\cos\theta_q^c \cdot d_p^2}{\cos\theta_p^c \cdot d_q^2}$$
   where $\theta^c$ is the angle at the camera-side connection vertex and $d$
   is the distance between connection endpoints.

For connections that involve the **light subpath** only ($s \ge 2, t = 1$):
the light side is independent of the pixel, so no shift is needed — just
re-evaluate the contribution at pixel $p$'s camera position.

### Algorithm outline (per pixel)

1. Read the current pixel's motion vector from the G-buffer.
2. Compute `prev_px = curr_px + motion_vector`.
3. Validate the temporal neighbour (screen bounds, depth, normal similarity).
4. Load `reservoir_prev[prev_px]`.
5. Apply the shift mapping: substitute the current pixel's camera vertex for
   the stored camera-side connection point; trace visibility to the light
   vertex; evaluate $\hat{p}_\text{curr}$ and Jacobian.
6. Merge reservoirs via streaming RIS with pairwise MIS.
7. Cap confidence weight $M$ (default cap: 20).

### Tasks

- [ ] Implement `cs_bdpt_temporal` compute shader.

- [ ] Implement the **camera-side reconnection shift**: load the current
  pixel's camera vertex at depth $t-1$; recompute the connection to the
  stored light vertex $y_{s-1}$; trace a shadow ray; evaluate BSDF ×
  geometry term.

- [ ] Implement the **Jacobian** for the camera-side reconnection (solid-angle
  ratio between old and new camera vertices).

- [ ] Implement **pairwise MIS** for temporal merge (from GRIS §5.3).

- [ ] Handle **temporal ping-pong**: swap buffer bindings between curr/prev
  each frame (avoids GPU-to-GPU copy).

- [ ] Add configurable **confidence weight cap** (uniform, default 20).

- [ ] Validation: static camera scene — verify monotonically decreasing noise
  over frames.  Moving camera — verify no ghosting at object edges.

**Output:** Significant noise reduction in static / slow-motion scenes.
Caustics benefit especially, as good light subpaths persist across frames.

---

## Phase RB-5: Spatial Reuse

**Goal:** Merge reservoirs from K randomly chosen neighbour pixels, providing
a large sample-count boost within a single frame.

### Shift mapping for spatial reuse

When reusing a connection from neighbour pixel $q$ at pixel $p$:

- **Light subpath sharing:** Light vertex $y_{s-1}$ from $q$'s connection is
  reused directly (light paths are scene-global, not pixel-specific).
- **Camera-side reconnection:** Use $p$'s own camera vertex $x_{t-1}^p$ at
  the same bounce depth $t$.  Trace a shadow ray from $x_{t-1}^p$ to
  $y_{s-1}$ and re-evaluate the contribution.
- **Jacobian:** Same formula as temporal reuse — accounts for the change in
  solid angle at the camera-side vertex.

### Tasks

- [ ] Implement `cs_bdpt_spatial` compute shader.

- [ ] Reuse the reconnection-shift, visibility, and Jacobian logic from Phase
  RB-4 (factor into shared WGSL functions in `restir_bdpt_common.wgsl`).

- [ ] Implement **neighbour selection heuristics**:
    - Normal similarity: `dot(n_p, n_q) > 0.9`.
    - Depth similarity: `abs(d_p - d_q) / max(d_p, 1e-4) < 0.1`.
    - Optional: reject neighbours whose material type differs.

- [ ] Support **configurable spatial pass count** (1 or 2).  Two passes
  further reduce variance at the cost of additional shadow rays.  Pass 1 reads from
  `reservoir_curr` (output of temporal reuse) and writes to `reservoir_spatial_temp`;
  pass 2 reads from `reservoir_spatial_temp` and writes to `reservoir_curr`.
  The `reservoir_spatial_temp` buffer is sized identically to `reservoir_curr` and
  allocated in `gpu_layout.rs` alongside the other reservoir buffers.

- [ ] Implement **rotated-disk neighbour sampling** (R2 sequence or
  Hammersley rotated per-pixel) with tunable radius R (default 20 pixels)
  and neighbour count K (default 4).

- [ ] Add UI sliders (egui) for: spatial radius R, neighbour count K, number
  of spatial iterations.

- [ ] Validation: compare single-frame quality with spatial reuse ON vs. OFF.
  Verify substantial noise reduction without light leaking across boundaries.

**Output:** Dramatic noise reduction in a single frame, especially for
caustics and indirect illumination.

---

## Phase RB-6: Light Subpath Pool Sharing & Amortisation

**Goal:** Reduce the cost of light subpath tracing by sharing a smaller pool
of light paths across multiple pixels, rather than the 1:1 mapping used in
Phase RB-3.

### Approach

- Trace $L$ light subpaths where $L \ll W{\times}H$ (e.g. $L = W{\times}H / 4$
  or a fixed budget like 262144).
- Each pixel is assigned a small random subset of light paths for connection
  (e.g. 4–8 paths per pixel, selected via a hash of the pixel coordinate +
  frame number).
- MIS weights must account for the fact that multiple pixels share the same
  light subpath.  The GRIS framework handles this through the concept of
  **reservoir domains** — each pixel's integration domain includes
  contributions from the shared light paths.

### Advantages

- Light tracing cost becomes independent of resolution.
- Light subpaths are longer-lived in the temporal reservoir (they don't depend
  on the camera), enabling better temporal reuse.
- Spatial reuse of light subpaths is "free" — the light vertex is already
  in the shared pool; only the camera-side connection needs to be re-evaluated.

### Tasks

- [ ] Implement a configurable light-path pool size (uniform), separate from
  the pixel count.

- [ ] Implement pixel → light-path assignment: a hash-based mapping that
  assigns K light-path indices to each pixel, ensuring good coverage across
  the pool.

- [ ] Update the MIS weight computation to account for shared light paths
  (multiple pixels may evaluate the same light vertex with different camera
  vertices).  Under the GRIS framework (Lin et al. 2022 §4), each pixel's
  reservoir domain covers $K$ light-path indices drawn from the shared pool.
  The corrected resampling weight for a connection using light path $\ell$ at
  pixel $p$ is:

  ```
  // n_p = number of pixels assigned light path ℓ (= pool_size / K for uniform hashing)
  // c_hat_p = unnormalized contribution at pixel p
  w_ris = c_hat_p / (n_p * pdf_source(ℓ))
  ```

  Because each light path is shared by exactly `total_pixels / pool_size` pixels
  (with the fixed-stride hash), `n_p` is constant and can be precomputed as a
  uniform.  This avoids the need for a per-frame histogram.

- [ ] Update the reservoir shift mapping to handle pool-indexed light paths
  (the light_path_id stored in the reservoir references the shared pool).

- [ ] Performance comparison: measure light-tracing pass time and total frame
  time with pool sizes $L = N, N/2, N/4, N/8$ (where $N = W{\times}H$).

- [ ] Validation: verify that shared-pool rendering matches the 1:1 baseline
  in visual quality (no systematic bias; noise may differ).

**Output:** Light tracing cost reduced by 2–8× with minimal impact on image
quality.

---

## Phase RB-7: Light Tracing (s, 1) Technique & Photon Splatting

**Goal:** Implement the $(s, 1)$ connection strategy where a light subpath
vertex is directly connected to the camera lens, splatting the contribution
to the corresponding screen pixel.  This technique is critical for rendering
caustics that are invisible to camera-originated paths.

### Approach

In standard BDPT, the $(s, 1)$ technique works as follows:
1. Take light vertex $y_{s-1}$.
2. Project it through the lens to determine which pixel it contributes to.
3. Evaluate the path contribution and MIS weight.
4. **Splat** the contribution to that pixel's reservoir.

On the GPU, splatting is implemented via a **dedicated splat buffer** (primary
approach) rather than atomic float operations.  WGSL's `atomicAdd` only supports
`i32`/`u32`; floating-point atomic addition requires an `atomicCompareExchangeWeak`
spin loop which is expensive and serialises writes to the same pixel.  The splat
buffer approach avoids contention entirely:

1. Each light vertex writes its $(s, 1)$ contribution to a per-light-path entry in a
   `splat_buffer` (sized `L × max_light_bounces`).
2. A subsequent **merge pass** reads `splat_buffer` and folds each entry into the
   appropriate pixel's reservoir using standard streaming RIS — no atomics required.

### Tasks

- [ ] Implement pixel projection for light vertices: given a world-space light
  vertex, compute the screen-pixel coordinate by projecting through the
  current view-projection matrix.  Discard if outside the viewport or behind
  the camera.

- [ ] Allocate `splat_buffer` (Storage R/W, sized `L × max_light_bounces ×
  sizeof(SplatEntry)`) in `gpu_layout.rs`.  Define `SplatEntry` (pixel coord,
  radiance, MIS weight, path IDs) as a `#[repr(C)]` Rust struct mirrored in WGSL.

- [ ] Implement the **splat merge pass** (`cs_bdpt_splat_merge`): iterate over
  `splat_buffer`, scatter each valid entry into the corresponding pixel's reservoir
  via streaming RIS.  This pass runs after `cs_bdpt_connect` (Pass 3) and before
  temporal reuse (Pass 4).

- [ ] Compute the $(s, 1)$ MIS weight for splatted contributions (requires
  evaluating the camera importance function $W_e$).

- [ ] Alternative: use a **splat buffer** — each light vertex writes its
  contribution to a separate per-pixel buffer, which is merged into the main
  reservoir in a dedicated merge pass after Pass 3.

- [ ] Validate: render a scene with a point light and a glass sphere.  The
  focused caustic on the floor should now be found by the light tracer
  ($(s, 1)$ technique) even when the unidirectional camera path misses it.

- [ ] Validate: compare MIS-weighted output of all techniques combined against
  a reference (high-SPP unidirectional rendering).

**Output:** Caustics from specular geometry are captured efficiently via light
tracing, completing the set of BDPT connection strategies.

---

## Phase RB-8: Hybrid Shift Mapping for Specular Chains

**Goal:** Handle paths where the camera (or light) subpath traverses one or
more specular bounces before reaching the connection vertex.  Pure
reconnection shifts fail at specular (delta-BSDF) vertices because the
outgoing direction is uniquely determined.

### Hybrid shift for BDPT connections

1. **Camera-side specular prefix:** If camera vertices $x_1, \ldots, x_{k}$
   are all specular (mirror/glass), the reconnection must happen at the first
   diffuse/glossy vertex $x_{k+1}$.  Vertices $x_1 \ldots x_k$ are replayed
   via **random replay** using the stored PRNG seed.  From $x_{k+1}$ onward,
   apply reconnection shift to the light vertex.

2. **Light-side specular prefix:** Similarly, if light vertices
   $y_1, \ldots, y_m$ are specular, they cannot serve as connection endpoints.
   The connection must target the first non-specular light vertex $y_{m+1}$.

3. **Jacobian:** Combines the random-replay BSDF ratio for specular segments
   with the geometric Jacobian for the reconnection segment.

### Tasks

- [ ] **Verify PRNG determinism**: random replay requires that replaying the WGSL
  PRNG from a stored `u32` seed across separate shader invocations produces
  identical results.  Validate this with a CPU-side test: call the same hash/LCG
  function twice from the same seed and assert byte-identical sequences.  If the
  current PRNG mixes the thread ID into the seed (a common pattern), factor that
  out so the stored seed is self-contained.

- [ ] Implement **path classification** in both camera and light subpaths:
  tag each vertex as specular or connectable based on material roughness.

- [ ] Implement **random replay shift** for specular prefixes: given a PRNG
  seed and a new starting point, retrace the specular bounce chain and compare
  the replayed throughput with the original.

- [ ] Implement the **hybrid Jacobian** combining random-replay ratio and
  reconnection-geometric terms.

- [ ] Store the **specular prefix length** in the reservoir so the shift
  mapping knows how many vertices to replay.

- [ ] Handle the degenerate case: if the camera subpath is entirely specular
  (no connectable vertex), fall back to the unidirectional estimate ($s = 0$
  technique).

- [ ] Validate: mirror + diffuse Cornell box — verify that reflections of
  indirect lighting are correctly reused without fireflies.

- [ ] Validate: glass sphere caustic — verify that spatial reuse across pixels
  near the caustic boundary produces smooth, unbiased results.

**Output:** Correct handling of specular surfaces in bidirectional connections
with spatiotemporal reuse.

---

## Phase RB-9: Biased Fast-Path Option

**Goal:** Provide a biased but faster variant that skips some visibility rays,
suitable for interactive previewing.

### Approach

- **Visibility reuse:** Assume a shifted connection is visible if the source
  pixel's connection was visible.  Avoids 1–2 shadow rays per spatial/temporal
  neighbour but introduces darkening bias at shadow boundaries.

- **Reduced connections:** Evaluate only a subset of $(s, t)$ techniques per
  pixel (e.g. $s \le 2$) to reduce the connection loop cost.

- **Simplified MIS:** Use balance-heuristic MIS with the 1-sample estimator
  instead of the full recursive computation.

- **Fewer spatial neighbours:** K = 1–2 instead of 3–5.

### Tasks

- [ ] Add `BiasMode` enum (`Unbiased`, `Biased`) to runtime config, selectable
  via UI.

- [ ] Implement biased code paths as compile-time or runtime branches in the
  temporal and spatial shaders.

- [ ] Measure performance difference (frame time, rays traced) between biased
  and unbiased modes at 1080p.

- [ ] Visual comparison: side-by-side showing bias artefacts.

**Output:** Fast interactive preview trading slight bias for higher frame rate.

---

## Phase RB-10: Performance Optimisation

**Goal:** Optimise GPU throughput so the full unbiased ReSTIR BDPT pipeline
runs at interactive rates (≥ 10 fps at 1080p on a mid-range discrete GPU).

### Optimisation targets

| Area | Technique |
|------|-----------|
| **Light tracing** | Shared pool (Phase RB-6) reduces light paths; compact vertex struct minimises memory traffic |
| **Connection loop** | Limit max $(s+t)$ path length; early-out when BSDF or geometry term is near zero |
| **Shadow rays** | Short rays (known endpoints) with early-out BVH traversal; batch multiple connections per invocation |
| **Memory traffic** | Pack reservoir to ≤ 64 bytes; quantise normals to octahedral `unorm16x2`; `f16` for throughputs |
| **Workgroup utilisation** | Tune workgroup size per pass (8×8 for tracing, 16×16 for connection/reuse); subgroup operations for local sharing |
| **Buffer ping-pong** | Swap bind-group indices instead of GPU-to-GPU copies |
| **MIS weights** | Cache intermediate PDF ratios in registers; use the recursive formulation to avoid redundant work |
| **Dispatch overlap** | Light tracing (Pass 1) and camera tracing (Pass 2) are independent — dispatch concurrently where the backend allows |

### Tasks

- [ ] Profile with `wgpu` timestamp queries: per-pass time breakdown.

- [ ] Implement reservoir/vertex struct packing (compact formats, `f16`
  throughputs, octahedral normals).

- [ ] Replace buffer copies with bind-group swaps for prev/curr.

- [ ] Benchmark different workgroup sizes for each pass.

- [ ] Benchmark the impact of `max_light_bounces` and `max_camera_bounces` on
  performance vs. quality (default: 5 each; consider 3 for fast mode).

- [ ] Benchmark shared light-path pool sizes vs. quality.

- [ ] If wgpu exposes subgroup operations: use `subgroupShuffle` for fast
  local reservoir sharing within a workgroup.

- [ ] Performance comparison table: classic progressive PT vs. unidirectional
  ReSTIR PT vs. ReSTIR BDPT (1 spatial pass) vs. ReSTIR BDPT (2 spatial
  passes), measuring ms/frame, rays/frame, PSNR vs. reference.

**Output:** ReSTIR BDPT at interactive frame rates, with documented perf data.

---

## Phase RB-11: Integration, UI & Polish

**Goal:** Seamless integration — toggle between classic PT, unidirectional
ReSTIR, and ReSTIR BDPT.  All existing features (materials, textures,
environment maps, denoiser, egui UI) work correctly in every mode.

### Tasks

- [ ] Runtime toggle: `R` cycles rendering mode (Classic PT → ReSTIR BDPT).
  UI panel shows mode, effective sample count, per-pass timings.

- [ ] Ensure the bilateral denoiser (`denoise.wgsl`) handles ReSTIR BDPT
  noise characteristics (may need G-buffer–guided filtering tuned for
  the different noise distribution).

- [ ] Ensure progressive accumulation works in ReSTIR BDPT mode (temporal
  reuse handles most convergence; blend residual noise progressively).

- [ ] Handle **scene hot-reload**: invalidate all reservoirs and light
  subpath buffers on scene change.

- [ ] Handle **window resize**: recreate all new buffers.

- [ ] Add egui controls for all ReSTIR BDPT parameters:
    - Enable/disable temporal reuse
    - Enable/disable spatial reuse
    - Number of spatial passes (1–3)
    - Spatial radius (5–50 pixels)
    - Neighbour count K (1–8)
    - Confidence cap M_max (5–50)
    - Max light bounces (1–8)
    - Max camera bounces (1–8)
    - Max path length s+t (2–16)
    - Light pool size mode (1:1 / shared)
    - Bias mode (Unbiased / Biased)
    - Reconnection roughness threshold (0.0–1.0)

- [ ] **Regression: classic PT and RS-1 paths must remain correct.** After
  splitting the path tracer into the BDPT multi-pass pipeline, verify that
  switching back to `ClassicPT` or `ReSTIR_DI` mode produces pixel-identical
  output to a pre-BDPT baseline.  Add a mode-switching test to the integration
  suite.

- [ ] Regression testing: add reference images for ReSTIR BDPT Cornell-box
  and glass-sphere-caustic scenes to the regression suite.

- [ ] Document shader entry points, buffer layouts, and algorithm in code
  comments.  Update [docs/CODE_STRUCTURE.md](docs/CODE_STRUCTURE.md).

**Output:** A polished, user-configurable ReSTIR BDPT implementation
coexisting with the classic path tracer, with full UI controls and
regression coverage.

---

## Appendix A: Shader Module Organisation

| File | Entry point | Purpose |
|------|-------------|---------|
| `path_trace.wgsl` | `cs_main` | Classic progressive path tracer (unchanged) |
| `bdpt_light_trace.wgsl` | `cs_trace_light_subpaths` | Pass 1: trace light subpaths, write light-vertex buffer |
| `bdpt_camera_trace.wgsl` | `cs_trace_camera_subpaths` | Pass 2: trace camera subpaths + G-buffer |
| `bdpt_connect.wgsl` | `cs_bdpt_connect` | Pass 3: enumerate (s,t) connections, build initial reservoirs |
| `bdpt_temporal.wgsl` | `cs_bdpt_temporal` | Pass 4: temporal reservoir merge |
| `bdpt_spatial.wgsl` | `cs_bdpt_spatial` | Pass 5: spatial reservoir merge |
| `bdpt_splat_merge.wgsl` | `cs_bdpt_splat_merge` | Pass 3b: merge $(s,1)$ splat buffer entries into per-pixel reservoirs (no atomics) |
| `bdpt_shade.wgsl` | `cs_bdpt_shade` | Pass 6: evaluate selected connection, write accumulation texture |
| `restir_bdpt_common.wgsl` | *(no entry point)* | Shared structs, shift-mapping helpers, RIS merge, Jacobian, BSDF eval, MIS weights — concatenated at compile time |
| `denoise.wgsl` | `cs_denoise` | Existing denoiser (unchanged) |
| `display.wgsl` | `vs_main` / `fs_main` | Existing tone-map + display (unchanged) |

Common code is shared via Rust-side string concatenation (prepend
`restir_bdpt_common.wgsl` to each pass shader before creating the
`ShaderModule`), matching the existing include strategy.

---

## Appendix B: Reservoir Merge Pseudocode (Streaming RIS)

Core streaming merge of two BDPT reservoirs (`r_dst` absorbs `r_src`):

```
fn merge_reservoir(r_dst: &mut BdptReservoir, r_src: &BdptReservoir,
                   p_hat_at_dst: f32, rng: &mut u32) {
    // Weight of the incoming sample evaluated at the destination pixel.
    let w_src = p_hat_at_dst * r_src.W * r_src.M;

    r_dst.w_sum += w_src;
    r_dst.M     += r_src.M;

    // Probabilistically select the incoming sample.
    if rand_f32(rng) < (w_src / r_dst.w_sum) {
        // Accept incoming: copy connection identification + geometry.
        r_dst.light_path_id  = r_src.light_path_id;
        r_dst.s              = r_src.s;
        r_dst.t              = r_src.t;
        r_dst.sample_seed    = r_src.sample_seed;
        r_dst.conn_cam_pos   = r_src.conn_cam_pos;
        r_dst.conn_cam_n     = r_src.conn_cam_n;
        r_dst.conn_light_pos = r_src.conn_light_pos;
        r_dst.conn_light_n   = r_src.conn_light_n;
        r_dst.radiance       = /* re-evaluate at dst */ shifted_radiance;
        r_dst.target_value   = p_hat_at_dst;
    }

    // Recompute the unbiased contribution weight.
    if r_dst.target_value > 0.0 {
        r_dst.W = r_dst.w_sum / (r_dst.M * r_dst.target_value);
    } else {
        r_dst.W = 0.0;
    }
}
```

---

## Appendix C: Key Differences — ReSTIR DI vs. ReSTIR PT vs. ReSTIR BDPT

| Aspect | ReSTIR DI | ReSTIR PT | **ReSTIR BDPT (this roadmap)** |
|--------|-----------|-----------|-------------------------------|
| What is resampled | A light sample (point on emitter) | A full unidirectional path | A bidirectional connection $(s, t)$ |
| Handles | Direct illumination only | Direct + indirect (multi-bounce GI) | Direct + indirect + caustics + SDS paths |
| Light transport start | Camera only | Camera only | Camera **and** light sources |
| Caustics | Not supported | Chance only (fireflies) | Efficient via light subpaths |
| Shift mapping | New shadow ray to same light point | Reconnection + random-replay hybrid | Camera-side reconnection + light-subpath reuse |
| Storage per pixel | ~16 bytes | ~32–64 bytes | ~120 bytes (4×u32 IDs + 6×vec3 endpoints + 4×f32 bookkeeping; see `BdptReservoir` struct) |
| Visibility cost per merge | 1 shadow ray | 1 reconnection-visibility ray | 1 shadow ray (connection) |
| Compute cost | Low | Moderate | Higher (multiple connection strategies, MIS weights) |
| Specular handling | Limited | Hybrid shift | Hybrid shift on both camera and light subpaths |

---

## Appendix D: Recursive MIS Weight Computation

Computing the balance-heuristic MIS weight for a BDPT path of length $k$
with connection strategy $(s, t)$ (where $s + t - 1 = k$) requires summing
the PDFs of all alternative strategies.  The **recursive formulation** from
Veach (1997, §10.2.3) avoids recomputing intermediate terms:

Define the **PDF ratio** between adjacent techniques:

$$r_i = \frac{p_{i}}{p_{i-1}}$$

where $p_i \equiv p_{s-i, t+i}$ is the PDF of the technique that shifts one
vertex from the light side to the camera side.  Each ratio involves only
local quantities (BSDF, geometry term, area-to-solid-angle conversion) at
vertices $x_{t+i-1}$ and $y_{s-i}$, so the full weight is:

$$w_{s,t} = \frac{1}{1 + \sum_{i \ne 0} \prod_{j=1}^{|i|} r_{\text{sign}(i) \cdot j}}$$

This is computed in a single forward-backward pass over the vertex chain,
caching running products.  Total cost: $O(k)$ BSDF evaluations.

---

## Appendix E: References & Further Reading

1. Lafortune & Willems — *"Bi-Directional Path Tracing"*, CompuGraphics 1993.

2. Veach & Guibas — *"Optimally Combining Sampling Techniques for Monte
   Carlo Rendering"*, SIGGRAPH 1995.

3. Veach — *"Robust Monte Carlo Methods for Light Transport Simulation"*,
   PhD thesis, Stanford 1997.  (Chapters 8–10: BDPT, MIS, recursive weight
   computation.)

4. Bitterli, Wyman, Pharr, Shirley, Lefohn, Jarosz — *"Spatiotemporal
   Reservoir Resampling for Real-Time Ray Tracing with Dynamic Direct
   Lighting"* (ReSTIR DI), ACM TOG (SIGGRAPH 2020).

5. Ouyang, Liu, Kettunen, Pharr, Pantaleoni — *"ReSTIR GI: Path Resampling
   for Real-Time Path Tracing"*, CGF (EGSR 2021).

6. Lin, Kettunen, Bitterli, Pantaleoni, Yuksel, Wyman — *"Generalized
   Resampled Importance Sampling: Foundations of ReSTIR"*, ACM TOG
   (SIGGRAPH 2022).

7. Wyman, Kettunen, Lin, Bitterli, Yuksel, Jarosz, Kozlowski, De Francesco
   — *"A Gentle Introduction to ReSTIR: Path Reuse in Real-time"*, ACM
   SIGGRAPH 2023 Courses.

8. Kettunen, Lin, Ramamoorthi, Bashford-Rogers, Wyman — *"Conditional
   Resampled Importance Sampling and ReSTIR"* (Suffix ReSTIR), SIGGRAPH
   Asia 2023.

9. Sawhney, Lin, Kettunen, Bitterli, Ramamoorthi, Wyman, Pharr —
   *"Decorrelating ReSTIR Samplers via MCMC Mutations"*, ACM TOG 2024.

10. Grittmann, Georgiev, Slusallek — *"Efficient Bidirectional Path Tracing
    with ReSTIR"*, concepts discussed in the context of production rendering
    at EGSR 2023.

---

*Each phase builds on the previous one and produces a verifiable visual result.
Phases can be implemented as separate Git milestones.*
