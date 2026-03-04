****# Path Tracer — Rust + wgpu Roadmap

A progressive, GPU-accelerated path tracer built with Rust and wgpu compute shaders.

---

## Phase 1: Project Scaffold & Window

**Goal:** Minimal wgpu application that opens a window and clears to a solid color.

- [x] Initialize Cargo project with dependencies
- [x] Create a window with event loop
- [x] Initialize wgpu adapter, device, queue, and surface
- [x] Configure surface with preferred format
- [x] Render loop that clears the surface to a solid color
- [x] Handle resize and close events

**Output:** A colored window that stays open and responds to resize.

---

## Phase 2: Full-Screen Quad & Texture Display

**Goal:** Render a texture to the screen via a full-screen triangle/quad so we can later display compute shader output.

- [x] Create a storage texture (RGBA32Float or RGBA8Unorm) matching window size
- [x] Write a vertex shader that generates a full-screen triangle (no vertex buffer)
- [x] Write a fragment shader that samples the storage texture
- [x] Set up the render pipeline (vertex + fragment shaders, bind group with texture + sampler)
- [x] Recreate texture on window resize
- [x] Verify by filling the texture with a gradient from the CPU side

**Output:** A gradient displayed on screen, sourced from a GPU texture.

---

## Phase 3: Compute Shader — Ray Generation

**Goal:** Replace the CPU-filled texture with a compute shader that generates rays and writes a basic sky gradient.

- [x] Write a compute shader (`path_trace.wgsl`) that:
  - Derives pixel coordinates from `global_invocation_id`
  - Computes a camera ray (origin + direction) per pixel using a simple pinhole model
  - Writes a sky-gradient color based on ray direction to the output texture
- [x] Create the compute pipeline and bind group (output texture as `storage` binding)
- [x] Dispatch the compute shader each frame before the render pass
- [x] Pass camera parameters (origin, look-at, FOV, aspect ratio) via a uniform buffer

**Output:** A sky gradient rendered entirely on the GPU.

---

## Phase 4: Sphere Intersection

**Goal:** Trace rays against hardcoded spheres in the compute shader.

- [x] Implement ray-sphere intersection in WGSL
- [x] Hardcode a small scene (2–3 spheres + ground plane) directly in the shader
- [x] Color pixels by surface normal on hit (normal mapping visualization)
- [x] Introduce a `Scene` uniform/storage buffer to pass sphere data from CPU
  - Struct: center (vec3f), radius (f32), material index (u32)
- [x] Verify correct intersection by orbiting the camera

**Output:** Shaded spheres visible on screen with normal-based coloring.

---

## Phase 5: Basic Path Tracing (Diffuse)

**Goal:** Implement a simple unbiased path tracer with diffuse (Lambertian) materials.

- [x] Implement a PRNG in WGSL (PCG or similar) seeded per-pixel per-frame
- [x] On hit, scatter a new ray in a random direction on the hemisphere (cosine-weighted)
- [x] Recurse (loop) up to a max bounce depth (e.g., 8)
- [x] Accumulate color: multiply albedo at each bounce, add sky contribution on miss
- [x] Single sample per pixel per frame — no accumulation yet

**Output:** Noisy but correct diffuse global illumination (single sample).

---

## Phase 6: Progressive Accumulation

**Goal:** Accumulate samples across frames for a converging image.

- [x] Add a second texture (accumulation buffer) that persists across frames
      — two `Rgba32Float` textures ping-pong every frame (`TEXTURE_BINDING | STORAGE_BINDING`)
- [x] Maintain a frame counter uniform; blend new sample with running average
      — `mix(prev, sample, 1/(frame+1))`; cap at 65 535 frames to prevent `+Inf` weight
- [x] Reset accumulation when camera moves or scene changes
      — `frame_count = 0` in `resize()`; static camera for Phase 6 (orbit re-enabled in Phase 8)
- [x] Display the accumulated result via the render pass
      — `textureLoad` + Reinhard tone-map + γ 2.2 in `display.wgsl` (no sampler, no `FLOAT32_FILTERABLE` requirement)

**Output:** Image that progressively converges to a clean render while the camera is still.

---

## Phase 7: Materials — Metal & Dielectric

**Goal:** Support reflective and refractive materials alongside diffuse.

- [x] Define a material struct: type (diffuse/metal/dielectric), albedo, fuzz, IOR
- [x] Pass materials via a storage buffer, index from sphere hit data
- [x] Implement metal reflection with optional fuzz (perturbed reflection vector)
- [x] Implement dielectric refraction (Snell's law + Schlick's approximation)
- [x] Randomly choose reflect vs refract for dielectrics

**Output:** A scene with diffuse, metallic, and glass spheres.

---

## Phase 8: Camera Controls

**Goal:** Interactive camera with mouse/keyboard input.

- [x] Implement an orbit camera (yaw, pitch, distance) controlled by mouse drag
- [x] Keyboard WASD for translation, scroll for zoom / FOV adjustment
- [x] Defocus blur (depth of field): thin lens model with aperture radius + focus distance
- [x] Upload updated camera uniform each frame; reset accumulation on change
- [x] Smooth interpolation for camera movement

**Output:** Freely navigable scene with optional depth-of-field effect.

---

## Phase 9: BVH Acceleration Structure

**Goal:** Accelerate ray-scene intersection for larger scenes.

- [x] Build a Bounding Volume Hierarchy (BVH) on the CPU using SAH (Surface Area Heuristic)
- [x] Flatten the BVH tree into a GPU-friendly array of nodes
  - Node struct: AABB min/max, left/right child or primitive range
- [x] Upload the flattened BVH + primitive list as storage buffers
- [x] Implement stackless or small-stack BVH traversal in WGSL
- [x] Ray-AABB intersection test (slab method)

**Output:** Scenes with hundreds of spheres at interactive rates.

---

## Phase 10: Triangle Meshes

**Goal:** Support triangle geometry in addition to spheres.

- [x] Define vertex buffer (position, normal, UV) and index buffer formats
- [x] Implement ray-triangle intersection (Möller–Trumbore)
- [x] Integrate triangles into the BVH as leaf primitives
- [x] Load a simple OBJ/glTF mesh on the CPU and upload buffers
- [x] Interpolate vertex normals for smooth shading

**Output:** A triangle mesh (e.g., Stanford Bunny) rendered with path tracing.

---

## Phase 11: Textures & Environment Maps

**Goal:** Add image-based textures and HDR environment lighting.

- [x] Load image textures (PNG/JPG) on the CPU, upload as GPU textures
- [x] Sample albedo textures in the shader using UV coordinates from hit data
- [x] Load an HDR equirectangular environment map
- [x] Sample the environment map on ray miss instead of the procedural sky
- [x] Optional: importance-sample the environment map for reduced noise

**Output:** Textured objects lit by an HDR environment.

---

## Phase 12: Declarative Scene Definitions

**Goal:** Replace hard-coded scene builders with data-driven YAML scene files.

- [x] Define a human-readable YAML scene format (spheres with centre, radius, material)
- [x] Add `serde` + `serde_yaml` dependencies
- [x] Implement `load_scene_from_yaml` in `scene.rs` that parses the YAML and produces `Vec<GpuSphere>`
- [x] Write the existing `build_large_scene` demo scene as `assets/scene.yaml`
- [x] Load `assets/scene.yaml` in `gpu.rs` instead of calling `build_large_scene`
- [x] Extend the format to cover triangle meshes, camera, and material overrides
- [x] Hot-reload scene file on disk change without restarting the renderer

**Output:** A running renderer whose scene is fully described by a plain YAML file that can be edited without recompiling.

---

## Phase 13: Emissive Materials & Light Sampling

**Goal:** Support emissive surfaces and reduce noise with next-event estimation.

- [x] Add emissive material type (emission color + strength)
- [x] Implement direct light sampling (next-event estimation / shadow rays)
- [x] Multiple Importance Sampling (MIS) to combine BSDF sampling and light sampling
- [x] Maintain a list of emissive primitives for direct sampling

**Output:** Scenes with area lights and significantly reduced noise.

---

## Phase 14: Denoising & Tone Mapping

**Goal:** Post-process the raw path-traced output for display quality.

- [x] Implement a simple tone-mapping pass (ACES / Reinhard) in a post-process shader
      — ACES filmic (Narkowicz 2016) with configurable `EXPOSURE` constant replaces per-channel Reinhard
- [x] Gamma correction (linear → sRGB)
      — Proper piecewise IEC 61966-2-1 sRGB transfer (1/2.4 exponent, correct linear threshold) replaces pow(1/2.2)
- [x] Optional: implement a simple spatial denoiser (e.g., edge-aware blur using normals + depth as guide)
      — Joint bilateral filter (7×7, spatial + normal + relative-depth weights) in `denoise.wgsl`
  - [x] Add toggle to enable/disable denoising for comparison — press 'N' at runtime
- [x] Output auxiliary buffers (albedo, normal, depth) for denoiser guidance
      — G-buffer (world-space normal + Euclidean depth) written by path_trace binding 13 each frame

**Output:** Clean, displayable images even at low sample counts.

---

## Phase 15: UI & Scene Editing

**Goal:** Add a GUI for tweaking scene and render parameters at runtime.

- [x] Integrate `egui` via `egui-wgpu` for immediate-mode GUI
- [x] Expose material parameters (albedo, roughness, IOR) in the UI
- [x] Camera settings panel (FOV, aperture, focus distance)
- [x] Render stats display (samples/sec, total samples, resolution)
- [x] Save rendered image to PNG

**Output:** A usable interactive path tracer with parameter tweaking.

---

> See [docs/CODE_STRUCTURE.md](docs/CODE_STRUCTURE.md) for detailed code structure guidelines, best practices, and the code review process.

---

## Architecture Overview

```
┌──────────────────────────────────────────────┐
│                   CPU (Rust)                  │
│                                              │
│  Scene definition ──► BVH build ──► Upload   │
│  Camera state     ──► Uniform buffer update  │
│  Input handling   ──► Accumulation reset     │
│  GUI (egui)       ──► Parameter tweaking     │
└──────────────────┬───────────────────────────┘
                   │ wgpu
┌──────────────────▼───────────────────────────┐
│                  GPU (WGSL)                   │
│                                              │
│  Compute pass:  path_trace.wgsl              │
│    ├─ Ray generation (camera)                │
│    ├─ BVH traversal + intersection           │
│    ├─ Material evaluation + scattering       │
│    ├─ Accumulation blend                     │
│    └─ Write to storage texture               │
│                                              │
│  Render pass:   display.wgsl                 │
│    ├─ Full-screen triangle                   │
│    ├─ Tone mapping + gamma correction        │
│    └─ Present to surface                     │
└──────────────────────────────────────────────┘
```

---

*Each phase builds on the previous one and produces a verifiable visual result. Phases can be implemented as separate Git milestones.*
