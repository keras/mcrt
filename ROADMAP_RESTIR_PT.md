# ReSTIR PT (Path Resampling) — Roadmap

Integrate Generalized Resampled Importance Sampling (GRIS) into the existing
wgpu compute-shader path tracer, enabling spatiotemporal path reuse via the
**ReSTIR PT** algorithm.  The result: dramatically lower variance at one sample
per pixel by resampling full light-transport paths across neighbouring pixels
and across frames, rather than tracing independent paths per pixel.

> **Key paper:** Lin, Kettunen, Bitterli, Pantaleoni, Yuksel, Wyman —
> *"Generalized Resampled Importance Sampling: Foundations of ReSTIR"*, ACM
> Transactions on Graphics (SIGGRAPH 2022).
>
> **Companion course:** Wyman et al. — *"A Gentle Introduction to ReSTIR: Path
> Reuse in Real-time"*, ACM SIGGRAPH 2023 Courses.

---

## Background & Motivation

The current path tracer (`path_trace.wgsl`) generates one independent sample per
pixel per frame and progressively accumulates a running average.  While the image
converges to ground truth given enough frames, noise is high at low sample
counts and the accumulation must be reset on every camera movement.

ReSTIR PT addresses this by:

1. **Resampled Importance Sampling (RIS)** — drawing multiple candidate paths
   per pixel and selecting one with probability proportional to a target
   function (e.g. luminance of the path contribution), then computing an
   unbiased contribution weight so that the expected pixel value remains
   correct.

2. **Reservoir-based streaming** — storing only a compact *reservoir* per pixel
   (selected sample + weight + confidence count) rather than all candidates,
   enabling the algorithm to process unbounded candidate counts with O(1)
   memory.

3. **Temporal reuse** — merging the current frame's reservoir with the
   reservoir from the previous frame (backprojected via motion vectors or
   reprojection), multiplying the effective sample count by the number of
   stable frames.

4. **Spatial reuse** — merging reservoirs from a set of neighbouring pixels,
   further boosting effective sample count.  Can be applied one or more
   times per frame.

5. **Shift mappings** — deterministic, invertible transformations that adapt a
   path sampled at one pixel so that it connects to a different pixel (or frame).
   Required for unbiased reuse across different integration domains.  Key
   mappings include *reconnection shift*, *random replay*, *half-vector copy*,
   and the practical *hybrid shift* (random replay for specular prefixes +
   reconnection at the first rough vertex).

The net effect is image quality equivalent to tens or hundreds of samples per
pixel while tracing only one path and performing a handful of reservoir merge
operations.

---

## Terminology Quick Reference

| Term | Meaning |
|------|---------|
| **RIS** | Resampled Importance Sampling — select 1 of M candidates with prob proportional to a target PDF |
| **Reservoir** | Compact data structure: selected sample, sum of weights (w_sum), confidence weight (M), output weight (W) |
| **Target function** $\hat{p}$ | Unnormalised PDF used for resampling; typically luminance of the full path contribution |
| **Unbiased contribution weight (UCW)** | Random variable whose expected value equals $1/p(X)$, where $p(X)$ is the true (generally unknown) PDF of the selected sample |
| **Shift mapping** $T_{q \to p}$ | Deterministic bijection that transforms a path sampled at pixel $q$ so it connects through pixel $p$; probability correction via Jacobian $|J_T|$ |
| **Reconnection vertex** | The path vertex (usually first diffuse/rough surface hit) where the shifted path reconnects to the reused suffix |
| **Hybrid shift** | Random replay for near-camera specular bounces, reconnection shift at the first rough vertex, then copy the path suffix |
| **Confidence weight (M)** | Tracks how many effective samples contributed to a reservoir; capped (e.g. 20) to prevent infinite accumulation |
| **G-buffer** | Per-pixel geometry data (normal, depth, material ID, motion vectors) used for neighbour selection and reprojection |

---

## Architecture Overview — Proposed Pipeline

The current single compute pass (`path_trace.wgsl`) will be restructured into
four ordered passes per frame.  Each pass is a separate compute dispatch reading
from and writing to screen-sized storage buffers.

```
Frame N
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  Pass 1 — Generate & Evaluate Candidate Paths                   │
│    • Trace 1 path per pixel (existing path tracer logic).        │
│    • Record: primary hit, reconnection vertex, path contribution │
│      (radiance), random seed, prefix type flags.                 │
│    • Write initial reservoir (sample = this path, W = 1/p̂(X)).   │
│    • Write G-buffer (normal, depth, material ID, motion vector). │
│                                                                  │
│  Pass 2 — Temporal Reuse                                         │
│    • Backproject each pixel using motion vectors to find the     │
│      corresponding reservoir from frame N−1.                     │
│    • Apply shift mapping T_{prev→curr} to adapt the stored path. │
│    • Merge the previous reservoir into the current one via       │
│      streaming RIS with MIS weights for unbiased combination.    │
│    • Cap confidence weight M to prevent temporal staleness.      │
│                                                                  │
│  Pass 3 — Spatial Reuse (1–2 iterations)                         │
│    • For each pixel, pick K neighbouring pixels (K ≈ 3–5)        │
│      from a random disk of radius R pixels (R ≈ 10–30).         │
│    • Apply shift mapping T_{neighbour→pixel} to each candidate.  │
│    • Merge via streaming RIS with pairwise MIS weights.          │
│    • Optional second spatial pass for further variance reduction. │
│                                                                  │
│  Pass 4 — Shade & Accumulate                                     │
│    • Evaluate final radiance from the selected reservoir sample. │
│    • Blend with progressive accumulation (if camera is static).  │
│    • Write to accumulation texture for the display pass.         │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
                          ▼
     Display pass (existing display.wgsl / denoise.wgsl)
```

**GPU buffer summary (new):**

| Buffer | Format | Per-pixel contents |
|--------|--------|--------------------|
| `reservoir_curr` | Storage (R/W) | Selected sample index/seed, $w_{sum}$, $M$, $W$ |
| `reservoir_prev` | Storage (read) | Copy of `reservoir_curr` from frame N−1 |
| `path_data_curr` | Storage (R/W) | Reconnection vertex position, incoming radiance at reconnection, path throughput prefix, material flags |
| `path_data_prev` | Storage (read) | Copy of `path_data_curr` from frame N−1 |
| `gbuffer` | Storage (R/W) | World-space normal, linear depth, material ID, screen-space motion vector |
| `gbuffer_prev` | Storage (read) | Previous-frame G-buffer for temporal reprojection validation |

Buffers are double-buffered (curr/prev) and ping-ponged each frame.  Total
additional GPU memory is approximately **64–96 bytes/pixel** (reservoir 32 B +
path data 48–64 B), or ~150–225 MB at 1920×1080.

---

## Phase RPT-1: Reservoir Data Structures & Buffer Plumbing

**Goal:** Define the GPU-side and CPU-side data structures for reservoirs and
path records, create the storage buffers, and wire the bind group layout so
that all four future passes can access them.  No algorithm changes yet — the
existing path tracer continues to run unmodified.

### Reservoir struct (WGSL + Rust mirror)

```
struct Reservoir {
    // Selected sample identification
    sample_seed:    u32,     // PRNG seed that regenerates the selected path
    rc_vertex:      vec3f,   // Reconnection vertex world position
    rc_normal:      vec3f,   // Normal at reconnection vertex
    rc_radiance:    vec3f,   // Incoming radiance at reconnection vertex
    prefix_length:  u32,     // Number of bounces before reconnection vertex

    // Reservoir bookkeeping
    w_sum:          f32,     // Running sum of resampling weights
    M:              f32,     // Confidence weight (effective sample count)
    W:              f32,     // Unbiased contribution weight = w_sum / (M · p̂(X))

    // Cached target value for the selected sample
    target_value:   f32,     // p̂(X) = luminance of the full path contribution
}
```

### Tasks

- [ ] Define `GpuReservoir` (`#[repr(C)]`, Pod/Zeroable) in a new Rust module
  `src/restir.rs` — mirror the WGSL struct above, respecting 16-byte
  alignment rules.

- [ ] Define `GpuPathRecord` for the auxiliary path data needed by shift
  mappings (reconnection vertex, prefix throughput, material/roughness flags
  at the reconnection point).

- [ ] Define `GpuGBufferExtended` — extend the existing G-buffer with a
  `motion_vector: vec2<f32>` and `material_id: u32` fields.

- [ ] Allocate double-buffered reservoir + path-data storage buffers in
  `gpu_layout.rs`, sized to `width × height × sizeof(GpuReservoir)`, etc.

- [ ] Extend the bind group layout with new bindings for reservoir_curr,
  reservoir_prev, path_data_curr, path_data_prev, gbuffer_prev.

- [ ] Handle buffer recreation on window resize (match existing
  accumulation-texture resize logic).

- [ ] Add a `restir_enabled: bool` toggle to `GpuState`, defaulting to
  `false`, so the renderer can be switched between classic progressive
  accumulation and ReSTIR mode at runtime.

- [ ] Unit test: verify that `sizeof::<GpuReservoir>()` matches the WGSL
  `@size` / `@align` expectations; same for `GpuPathRecord`.

**Output:** All buffers allocated and bound; existing rendering unchanged; `N`
key (or UI toggle) switches the `restir_enabled` flag.

---

## Phase RPT-2: Initial Candidate Generation (Pass 1)

**Goal:** Modify the path tracer to emit path metadata into the reservoir and
path-data buffers in addition to the accumulation texture.  This pass replaces
the existing `cs_main` when `restir_enabled` is true.

### What the pass writes (per pixel)

1. Trace a single path using the existing `path_color()` logic.
2. Along the path, identify the **reconnection vertex** — the first vertex
   after the camera whose surface is sufficiently rough (roughness above a
   configurable threshold, e.g. > 0.2).  For purely specular prefixes (mirror
   chains, glass), the reconnection vertex is further down the path.
3. Store the path split into two parts:
   - **Prefix:** camera → reconnection vertex (deterministic given the primary
     ray and PRNG seed; can be replayed cheaply).
   - **Suffix:** reconnection vertex → light (the portion that will be reused
     by other pixels via shift mappings).
4. Compute the **target function** $\hat{p}(X) = \text{luminance}(f(X))$ where
   $f(X)$ is the full path contribution (throughput × radiance).
5. Initialise the reservoir: `w_sum = p̂(X) / p_source(X)`, `M = 1`, `W = 1 / p̂(X)`.

### Tasks

- [ ] Implement a new entry point `cs_restir_generate` (or a compile-time
  branch within `cs_main`) that:
    - Calls the existing `path_color` logic but also captures per-bounce
      metadata: hit position, normal, roughness, material type.
    - Identifies the reconnection vertex using a roughness threshold.
    - Writes `Reservoir` and `PathRecord` to the current-frame buffers.
    - Writes the extended G-buffer (with motion vector).

- [ ] Compute screen-space **motion vectors**: for the primary hit, project
  the hit position through the *previous frame's* view-projection matrix and
  compare with the current pixel coordinate.  Store the delta as a vec2.
  This requires uploading the previous frame's VP matrix as a uniform.

- [ ] Add `prev_view_proj: mat4x4<f32>` to the camera uniform (or a separate
  uniform binding) for motion-vector computation.

- [ ] Write an isolated test: render one frame in ReSTIR-generate mode and
  read back the reservoir buffer; verify that `target_value > 0` for pixels
  that hit geometry, and `M == 1` for all pixels.

**Output:** Each pixel has a valid initial reservoir and path record.  Image
quality is identical to the classic path tracer (no reuse yet).

---

## Phase RPT-3: Temporal Reuse (Pass 2)

**Goal:** Merge each pixel's current reservoir with the backprojected reservoir
from the previous frame, doubling (approximately) the effective sample count
every frame while the camera is static.

### Algorithm outline (per pixel)

1. Read the current pixel's motion vector from the extended G-buffer.
2. Compute the previous-frame pixel coordinate: `prev_px = curr_px + motion_vector`.
3. Validate the temporal neighbour:
   - Is `prev_px` inside the screen?
   - Is the depth at `prev_px` (from `gbuffer_prev`) within a tolerance of
     the reprojected depth?
   - Is the normal at `prev_px` similar to the current normal (dot > 0.9)?
   - If validation fails → skip temporal reuse (set `M_prev = 0`).
4. Load `reservoir_prev[prev_px]` and `path_data_prev[prev_px]`.
5. Apply the **shift mapping** $T_{\text{prev} \to \text{curr}}$ to the
   previous sample to re-express it in the current pixel's domain:
   - If using **reconnection shift**: keep the suffix (reconnection vertex →
     light) and recompute the prefix (camera → reconnection vertex) for the
     current pixel.  The reconnection vertex position is stored; trace a
     shadow ray to verify visibility.
   - Compute the Jacobian $|J_T|$ of the shift to adjust the weight.
6. Compute the target function $\hat{p}_{\text{curr}}(T(X_{\text{prev}}))$ for
   the shifted sample evaluated at the current pixel.
7. Merge reservoirs using streaming RIS with pairwise MIS:
   ```
   w_prev = p̂_curr(T(X_prev)) · W_prev · M_prev_capped
   w_curr = p̂_curr(X_curr)    · W_curr · M_curr
   ```
   Select one sample proportional to `w_prev + w_curr`.
8. Update `M = M_curr + M_prev_capped` (cap `M_prev` to e.g. 20× to prevent
   infinite temporal accumulation).
9. Recompute `W = w_sum / (M · p̂(X_selected))`.

### Tasks

- [ ] Implement `cs_restir_temporal` compute shader entry point.

- [ ] Implement the **reconnection shift** $T_{q \to p}$:
    - Given the reconnection vertex position from the source pixel's path
      record and the current pixel's primary hit, compute the new prefix
      (camera ray → primary hit → reconnection vertex).
    - Trace a **visibility ray** from the current pixel's primary hit to the
      reconnection vertex to check for occlusion.
    - Compute the **Jacobian**: accounts for the change in solid angle subtended
      by the reconnection vertex when viewed from a different primary-hit
      position.  $|J_T| = \frac{\cos\theta_q \cdot d_p^2}{\cos\theta_p \cdot d_q^2}$
      where $d$ is the distance to the reconnection vertex and $\theta$ is the
      angle between the surface normal at the reconnection vertex and the
      connecting direction.

- [ ] Implement **pairwise MIS** weights for temporal merge (the recommended
  approach from GRIS §5.3 — avoids needing to evaluate all source PDFs):
  ```
  mis_weight_curr = M_curr / (M_curr + M_prev_capped)
  mis_weight_prev = 1.0 - mis_weight_curr  (if shift is valid)
  ```

- [ ] Handle the **temporal ping-pong**: after the temporal pass, copy
  `reservoir_curr` → `reservoir_prev` and `path_data_curr` → `path_data_prev`
  (or swap buffer bindings to avoid the copy).

- [ ] Add a **confidence weight cap** (uniform-configurable, default 20).

- [ ] Visual validation: render small motion scenes; verify that noise
  decreases over time and that disoccluded regions revert to single-sample
  quality without ghosting artefacts.

**Output:** Significant noise reduction in static / slowly-moving camera
scenarios.  No ghosting or bias visible at object boundaries.

---

## Phase RPT-4: Spatial Reuse (Pass 3)

**Goal:** Merge reservoirs from K randomly chosen neighbour pixels into the
current pixel's reservoir, providing a large effective sample-count boost
within a single frame.

### Algorithm outline (per pixel)

1. For $k = 1 \ldots K$ (K ≈ 3–5):
   a. Choose a random neighbour $q_k$ within a screen-space disk of radius R
      pixels (R ≈ 10–30).  Use a rotation of the sampling pattern per pixel
      (to decorrelate spatial noise).
   b. Validate: depth and normal similarity between $p$ and $q_k$ (skip if
      dissimilar — prevents light leaking across object boundaries).
   c. Load `reservoir[q_k]` and `path_data[q_k]`.
   d. Apply shift mapping $T_{q_k \to p}$ (reconnection shift: keep suffix,
      rebuild prefix, trace visibility ray, compute Jacobian).
   e. Evaluate target $\hat{p}_p(T(X_{q_k}))$ at the current pixel.
   f. Merge into the running reservoir via streaming RIS + pairwise MIS.
2. After merging all neighbours, recompute $W$.

### Tasks

- [ ] Implement `cs_restir_spatial` compute shader entry point.

- [ ] Reuse the reconnection-shift and Jacobian logic from Phase RPT-3 (factor
  into a shared WGSL function called from both temporal and spatial passes).

- [ ] Implement **neighbour selection heuristics**:
    - Normal similarity: `dot(n_p, n_q) > 0.9`.
    - Depth similarity: `abs(d_p - d_q) / max(d_p, 1e-4) < 0.1`.
    - Optional: reject neighbours whose material differs significantly.

- [ ] Support a **configurable spatial pass count** (1 or 2 passes).  A second
  spatial pass further reduces variance but costs additional visibility rays.
  When using 2 passes, the output of pass 3a becomes the input of pass 3b
  (requires an intermediate buffer swap).

- [ ] Implement **spiral / rotated-disk sampling** to choose neighbours: use
  a low-discrepancy pattern (e.g. R2 sequence or Hammersley) rotated by a
  per-pixel random angle, biased toward the centre to weight closer neighbours
  more heavily.

- [ ] Add UI sliders (egui) for: spatial radius R, neighbour count K, number
  of spatial iterations, and the MIS strategy selection.

- [ ] Visual validation: compare a single-frame image with spatial reuse ON
  vs. OFF; verify substantial noise reduction without light leaking across
  depth/normal discontinuities.

**Output:** Dramatic noise reduction in a single frame.  Quality approaches
that of ~10–50 independent samples per pixel.

---

## Phase RPT-5: Hybrid Shift Mapping for Specular Surfaces

**Goal:** Extend the reconnection shift from Phases RPT-3/4 to handle paths
with specular prefixes (mirror reflections, glass refraction).  The pure
reconnection shift fails for specular surfaces because the prefix direction is
deterministic (cannot be recomputed for a different pixel without changing the
specular bounce).

### Hybrid shift strategy

1. Walk the path from the camera.  At each bounce, classify the surface:
   - **Specular** (roughness < threshold): use **random replay** — record
     the PRNG seed and replay the same random numbers at the new pixel to
     produce a similar (but shifted) specular bounce sequence.
   - **Rough/diffuse** (roughness ≥ threshold): this vertex becomes the
     **reconnection vertex**.  Apply the **reconnection shift** from here
     onward (copy the suffix as-is, reconnect via a visibility ray).
2. The Jacobian for the hybrid shift combines:
   - Random-replay contribution: ratio of BSDFs evaluated at the replayed
     directions at the old vs. new pixel.
   - Reconnection contribution: the geometric Jacobian from Phase RPT-3.

### Tasks

- [ ] Implement path classification: tag each bounce as specular or rough
  based on material roughness.

- [ ] Implement **random replay shift**: given a PRNG seed and a new primary
  hit point, re-trace the specular prefix to produce the shifted secondary
  rays.  Compare the replayed path's throughput with the original to get the
  probability ratio.

- [ ] Implement the **hybrid shift Jacobian** that accounts for both the
  random-replay and reconnection portions of the path transformation.

- [ ] Fall back to pure random-replay when no rough vertex exists on the path
  (fully specular path, e.g. a hall of mirrors).

- [ ] Validate: render a scene with a planar mirror reflecting a diffuse
  Cornell box.  Verify that spatial/temporal reuse correctly resolves the
  reflected indirect lighting without bias or fireflies.

- [ ] Validate: render a glass sphere (dielectric IOR 1.5) on a diffuse
  floor.  Verify that caustics below the sphere benefit from reuse and
  converge visibly faster than the baseline.

**Output:** Correct, low-noise rendering of scenes containing both specular
and diffuse surfaces, using a single sample per pixel with full
spatiotemporal reuse.

---

## Phase RPT-6: Biased Fast-Path Option

**Goal:** Provide a biased but faster variant of the ReSTIR PT pipeline that
skips some visibility rays, suitable for interactive previewing.

### Approach

- **Visibility reuse:** Instead of tracing a fresh visibility ray for every
  shifted path during spatial/temporal merging, assume that a shifted sample
  is visible if the source pixel's sample was visible.  This avoids 1–2 ray
  traces per neighbour but introduces darkening bias at shadow and silhouette
  edges.

- **Reduced spatial neighbours:** Use K = 1–2 neighbours instead of 3–5 to
  further reduce ray-tracing cost.

- **Simplified MIS:** Use balance-heuristic MIS with `1/M` weighting instead
  of pairwise MIS; slightly biased but cheaper to evaluate.

### Tasks

- [ ] Add a `restir_bias_mode: BiasMode` enum (`Unbiased`, `Biased`) to the
  runtime configuration, selectable via UI.

- [ ] Implement the biased code paths as compile-time or runtime branches in
  the temporal and spatial shaders.

- [ ] Measure and document the performance difference (frame time and rays
  traced) between biased and unbiased modes at 1080p.

- [ ] Visual comparison: side-by-side screenshots showing the bias artefacts
  (edge darkening) vs. the unbiased output.

**Output:** A fast interactive preview mode that trades slight bias for
significantly higher frame rate.

---

## Phase RPT-7: Performance Optimisation

**Goal:** Optimise GPU throughput so the full unbiased ReSTIR PT pipeline runs
at interactive rates (≥ 15 fps at 1080p on a mid-range discrete GPU).

### Optimisation targets

| Area | Technique |
|------|-----------|
| **Memory traffic** | Pack reservoir struct to 32 bytes; quantise normals to octahedral `unorm16x2`; half-precision where possible |
| **Visibility rays** | Use short rays (origin → reconnection vertex) with an early-out BVH traversal flag; share the BVH stack across multiple ray queries in the same invocation |
| **Workgroup utilisation** | Tune workgroup size (8×8 vs 16×16) per pass; use subgroup operations for reservoir merging where supported |
| **Buffer ping-pong** | Swap bind group indices instead of gpu-to-gpu copies between frames |
| **Shader occupancy** | Minimise register pressure by computing prefix/suffix Jacobians in separate scopes; reduce live `vec3` count |
| **Dispatch overlap** | Use separate command encoders / timeline semaphores to overlap Pass 2 with Pass 3 (temporal and spatial reuse can partially overlap if the spatial pass reads from a separate buffer) |

### Tasks

- [ ] Profile with `wgpu`'s timestamp queries: measure per-pass time (generate,
  temporal, spatial, shade).

- [ ] Implement reservoir struct packing (32 bytes): use `f16` for radiance,
  `u32` for packed normal (octahedral), bitfields for prefix_length and M.

- [ ] Replace full-buffer copies with bind-group swaps for prev/curr buffers.

- [ ] Benchmark different workgroup sizes for each pass.

- [ ] If wgpu exposes subgroup operations: use `subgroupShuffle` for
  intra-workgroup reservoir sharing as a fast local spatial reuse step.

- [ ] Write a performance comparison table: classic progressive accumulation
  vs. ReSTIR PT (1 spatial pass) vs. ReSTIR PT (2 spatial passes), measuring
  ms/frame, rays/frame, and visual noise (PSNR vs. reference).

**Output:** ReSTIR PT running at interactive frame rates, with documented
performance characteristics.

---

## Phase RPT-8: Integration, UI & Polish

**Goal:** Seamless integration with the existing renderer — the user can toggle
between classic progressive path tracing and ReSTIR PT mode, and all existing
features (materials, textures, environment maps, denoiser, egui UI) work
correctly in both modes.

### Tasks

- [ ] Runtime toggle: pressing `R` switches between classic and ReSTIR PT
  rendering.  The UI panel shows the current mode and ReSTIR-specific stats
  (effective sample count, reservoir merge rate).

- [ ] Ensure the existing bilateral denoiser (`denoise.wgsl`) works
  correctly on ReSTIR PT output.  ReSTIR output has different noise
  characteristics (spatially correlated "blotchy" noise vs. independent pixel
  noise); consider using the G-buffer material ID as an additional
  bilateral-filter guide.

- [ ] Ensure accumulation works correctly in ReSTIR mode: when the camera is
  static, blend frames progressively (the reservoir quality improves via
  temporal reuse, and progressive accumulation converges the remaining noise).

- [ ] Handle scene hot-reload: when the YAML scene file changes, invalidate
  all reservoirs (set `M = 0` everywhere) to force re-initialisation.

- [ ] Handle window resize: recreate all reservoir and path-data buffers.

- [ ] Add egui controls for all ReSTIR parameters:
    - Enable/disable temporal reuse
    - Enable/disable spatial reuse
    - Number of spatial passes (1–3)
    - Spatial radius (5–50 pixels)
    - Neighbour count K (1–8)
    - Confidence cap M_max (5–50)
    - Reconnection roughness threshold (0.0–1.0)
    - Bias mode (unbiased / biased)

- [ ] Regression testing: add reference images for a ReSTIR-PT–rendered
  Cornell-box scene to the regression suite; verify PSNR is within tolerance.

- [ ] Document the shader entry points, buffer layouts, and algorithm in
  code comments and update `docs/CODE_STRUCTURE.md`.

**Output:** A polished, user-configurable ReSTIR PT implementation that
coexists with the classic path tracer, with full UI controls and regression
coverage.

---

## Appendix A: Shader Module Organisation

To keep individual WGSL files manageable, the four passes should be split
across dedicated shader files:

| File | Entry point | Purpose |
|------|-------------|---------|
| `path_trace.wgsl` | `cs_main` | Classic progressive path tracer (unchanged) |
| `restir_generate.wgsl` | `cs_restir_generate` | Pass 1: trace path, write reservoir + G-buffer |
| `restir_temporal.wgsl` | `cs_restir_temporal` | Pass 2: temporal reservoir merge |
| `restir_spatial.wgsl`  | `cs_restir_spatial`  | Pass 3: spatial reservoir merge |
| `restir_shade.wgsl`    | `cs_restir_shade`    | Pass 4: evaluate selected sample, write accumulation texture |
| `restir_common.wgsl`   | *(no entry point)*   | Shared structs, shift-mapping helpers, RIS merge, Jacobian computation — `#include`'d or concatenated at compile time |
| `denoise.wgsl` | `cs_denoise` | Existing denoiser (unchanged) |
| `display.wgsl` | `vs_main` / `fs_main` | Existing tone-map + display (unchanged) |

Because WGSL does not support `#include`, common code can be shared by
Rust-side string concatenation (prepend `restir_common.wgsl` to each pass
shader before creating the `ShaderModule`), matching the strategy already used
for the existing SDF code-generation pipeline.

---

## Appendix B: Reservoir Merge Pseudocode (Streaming RIS)

For reference, the core streaming merge of two reservoirs (`r_dst` absorbs
`r_src`):

```
fn merge_reservoir(r_dst: &mut Reservoir, r_src: &Reservoir,
                   p_hat_at_dst: f32, rng: &mut u32) {
    // Weight of the incoming sample at the destination pixel.
    let w_src = p_hat_at_dst * r_src.W * r_src.M;

    r_dst.w_sum += w_src;
    r_dst.M     += r_src.M;

    // Probabilistically select the incoming sample.
    if rand_f32(rng) < (w_src / r_dst.w_sum) {
        // Accept the incoming sample.
        r_dst.sample_seed   = r_src.sample_seed;
        r_dst.rc_vertex     = r_src.rc_vertex;
        r_dst.rc_normal     = r_src.rc_normal;
        r_dst.rc_radiance   = r_src.rc_radiance;
        r_dst.prefix_length = r_src.prefix_length;
        r_dst.target_value  = p_hat_at_dst;
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

## Appendix C: Key Differences from ReSTIR DI

This roadmap targets **ReSTIR PT** (full path resampling), not ReSTIR DI
(direct-illumination–only resampling).  Key differences:

| Aspect | ReSTIR DI | ReSTIR PT (this roadmap) |
|--------|-----------|--------------------------|
| What is resampled | A light sample (point on an emitter) | A full path (camera → bounces → light) |
| Handles | Direct illumination only | Direct + indirect (multi-bounce GI) |
| Shift mapping | Simple: new shadow ray to the same light point | Complex: reconnection + random-replay hybrid |
| Storage per pixel | ~16 bytes (light ID + weight) | ~32–64 bytes (reconnection vertex, prefix seed, weights) |
| Visibility cost | 1 shadow ray per merge | 1 reconnection-visibility ray per merge (may be longer than a shadow ray) |
| Specular handling | Limited (relies on BSDF sampling) | Hybrid shift handles mirror/glass prefixes explicitly |

---

## Appendix D: References & Further Reading

1. Bitterli, Wyman, Pharr, Shirley, Lefohn, Jarosz — *"Spatiotemporal
   Reservoir Resampling for Real-Time Ray Tracing with Dynamic Direct
   Lighting"*, ACM TOG (SIGGRAPH 2020).

2. Ouyang, Liu, Kettunen, Pharr, Pantaleoni — *"ReSTIR GI: Path Resampling
   for Real-Time Path Tracing"*, CGF (EGSR 2021).

3. Lin, Kettunen, Bitterli, Pantaleoni, Yuksel, Wyman — *"Generalized
   Resampled Importance Sampling: Foundations of ReSTIR"*, ACM TOG
   (SIGGRAPH 2022).

4. Wyman, Kettunen, Lin, Bitterli, Yuksel, Jarosz, Kozlowski, De Francesco
   — *"A Gentle Introduction to ReSTIR: Path Reuse in Real-time"*, ACM
   SIGGRAPH 2023 Courses.

5. Kettunen, Lin, Ramamoorthi, Bashford-Rogers, Wyman — *"Conditional
   Resampled Importance Sampling and ReSTIR"* (Suffix ReSTIR), SIGGRAPH
   Asia 2023.

6. Sawhney, Lin, Kettunen, Bitterli, Ramamoorthi, Wyman, Pharr —
   *"Decorrelating ReSTIR Samplers via MCMC Mutations"*, ACM TOG 2024.

---

*Each phase builds on the previous one and produces a verifiable visual result.
Phases can be implemented as separate Git milestones.*
