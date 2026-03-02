# Path Tracer — Rust + wgpu Roadmap

A progressive, GPU-accelerated path tracer built with Rust and wgpu compute shaders.

---

## Phase 1: Project Scaffold & Window

**Goal:** Minimal wgpu application that opens a window and clears to a solid color.

- [ ] Initialize Cargo project with dependencies (`wgpu`, `winit`, `pollster`, `bytemuck`, `glam`)
- [ ] Create a window with `winit` event loop
- [ ] Initialize wgpu adapter, device, queue, and surface
- [ ] Configure surface with preferred format
- [ ] Render loop that clears the surface to a solid color
- [ ] Handle resize and close events

**Output:** A colored window that stays open and responds to resize.

---

## Phase 2: Full-Screen Quad & Texture Display

**Goal:** Render a texture to the screen via a full-screen triangle/quad so we can later display compute shader output.

- [ ] Create a storage texture (RGBA32Float or RGBA8Unorm) matching window size
- [ ] Write a vertex shader that generates a full-screen triangle (no vertex buffer)
- [ ] Write a fragment shader that samples the storage texture
- [ ] Set up the render pipeline (vertex + fragment shaders, bind group with texture + sampler)
- [ ] Recreate texture on window resize
- [ ] Verify by filling the texture with a gradient from the CPU side

**Output:** A gradient displayed on screen, sourced from a GPU texture.

---

## Phase 3: Compute Shader — Ray Generation

**Goal:** Replace the CPU-filled texture with a compute shader that generates rays and writes a basic sky gradient.

- [ ] Write a compute shader (`path_trace.wgsl`) that:
  - Derives pixel coordinates from `global_invocation_id`
  - Computes a camera ray (origin + direction) per pixel using a simple pinhole model
  - Writes a sky-gradient color based on ray direction to the output texture
- [ ] Create the compute pipeline and bind group (output texture as `storage` binding)
- [ ] Dispatch the compute shader each frame before the render pass
- [ ] Pass camera parameters (origin, look-at, FOV, aspect ratio) via a uniform buffer

**Output:** A sky gradient rendered entirely on the GPU.

---

## Phase 4: Sphere Intersection

**Goal:** Trace rays against hardcoded spheres in the compute shader.

- [ ] Implement ray-sphere intersection in WGSL
- [ ] Hardcode a small scene (2–3 spheres + ground plane) directly in the shader
- [ ] Color pixels by surface normal on hit (normal mapping visualization)
- [ ] Introduce a `Scene` uniform/storage buffer to pass sphere data from CPU
  - Struct: center (vec3f), radius (f32), material index (u32)
- [ ] Verify correct intersection by orbiting the camera

**Output:** Shaded spheres visible on screen with normal-based coloring.

---

## Phase 5: Basic Path Tracing (Diffuse)

**Goal:** Implement a simple unbiased path tracer with diffuse (Lambertian) materials.

- [ ] Implement a PRNG in WGSL (PCG or similar) seeded per-pixel per-frame
- [ ] On hit, scatter a new ray in a random direction on the hemisphere (cosine-weighted)
- [ ] Recurse (loop) up to a max bounce depth (e.g., 8)
- [ ] Accumulate color: multiply albedo at each bounce, add sky contribution on miss
- [ ] Single sample per pixel per frame — no accumulation yet

**Output:** Noisy but correct diffuse global illumination (single sample).

---

## Phase 6: Progressive Accumulation

**Goal:** Accumulate samples across frames for a converging image.

- [ ] Add a second texture (accumulation buffer) that persists across frames
- [ ] Maintain a frame counter uniform; blend new sample with running average
  - `accumulated = (accumulated * (frame - 1) + new_sample) / frame`
- [ ] Reset accumulation when camera moves or scene changes
- [ ] Display the accumulated result via the render pass

**Output:** Image that progressively converges to a clean render while the camera is still.

---

## Phase 7: Materials — Metal & Dielectric

**Goal:** Support reflective and refractive materials alongside diffuse.

- [ ] Define a material struct: type (diffuse/metal/dielectric), albedo, fuzz, IOR
- [ ] Pass materials via a storage buffer, index from sphere hit data
- [ ] Implement metal reflection with optional fuzz (perturbed reflection vector)
- [ ] Implement dielectric refraction (Snell's law + Schlick's approximation)
- [ ] Randomly choose reflect vs refract for dielectrics

**Output:** A scene with diffuse, metallic, and glass spheres.

---

## Phase 8: Camera Controls

**Goal:** Interactive camera with mouse/keyboard input.

- [ ] Implement an orbit camera (yaw, pitch, distance) controlled by mouse drag
- [ ] Keyboard WASD for translation, scroll for zoom / FOV adjustment
- [ ] Defocus blur (depth of field): thin lens model with aperture radius + focus distance
- [ ] Upload updated camera uniform each frame; reset accumulation on change
- [ ] Smooth interpolation for camera movement

**Output:** Freely navigable scene with optional depth-of-field effect.

---

## Phase 9: BVH Acceleration Structure

**Goal:** Accelerate ray-scene intersection for larger scenes.

- [ ] Build a Bounding Volume Hierarchy (BVH) on the CPU using SAH (Surface Area Heuristic)
- [ ] Flatten the BVH tree into a GPU-friendly array of nodes
  - Node struct: AABB min/max, left/right child or primitive range
- [ ] Upload the flattened BVH + primitive list as storage buffers
- [ ] Implement stackless or small-stack BVH traversal in WGSL
- [ ] Ray-AABB intersection test (slab method)

**Output:** Scenes with hundreds of spheres at interactive rates.

---

## Phase 10: Triangle Meshes

**Goal:** Support triangle geometry in addition to spheres.

- [ ] Define vertex buffer (position, normal, UV) and index buffer formats
- [ ] Implement ray-triangle intersection (Möller–Trumbore)
- [ ] Integrate triangles into the BVH as leaf primitives
- [ ] Load a simple OBJ/glTF mesh on the CPU and upload buffers
- [ ] Interpolate vertex normals for smooth shading

**Output:** A triangle mesh (e.g., Stanford Bunny) rendered with path tracing.

---

## Phase 11: Textures & Environment Maps

**Goal:** Add image-based textures and HDR environment lighting.

- [ ] Load image textures (PNG/JPG) on the CPU, upload as GPU textures
- [ ] Sample albedo textures in the shader using UV coordinates from hit data
- [ ] Load an HDR equirectangular environment map
- [ ] Sample the environment map on ray miss instead of the procedural sky
- [ ] Optional: importance-sample the environment map for reduced noise

**Output:** Textured objects lit by an HDR environment.

---

## Phase 12: Emissive Materials & Light Sampling

**Goal:** Support emissive surfaces and reduce noise with next-event estimation.

- [ ] Add emissive material type (emission color + strength)
- [ ] Implement direct light sampling (next-event estimation / shadow rays)
- [ ] Multiple Importance Sampling (MIS) to combine BSDF sampling and light sampling
- [ ] Maintain a list of emissive primitives for direct sampling

**Output:** Scenes with area lights and significantly reduced noise.

---

## Phase 13: Denoising & Tone Mapping

**Goal:** Post-process the raw path-traced output for display quality.

- [ ] Implement a simple tone-mapping pass (ACES / Reinhard) in a post-process shader
- [ ] Gamma correction (linear → sRGB)
- [ ] Optional: implement a simple spatial denoiser (e.g., edge-aware blur using normals + depth as guide)
- [ ] Output auxiliary buffers (albedo, normal, depth) for denoiser guidance

**Output:** Clean, displayable images even at low sample counts.

---

## Phase 14: UI & Scene Editing

**Goal:** Add a GUI for tweaking scene and render parameters at runtime.

- [ ] Integrate `egui` via `egui-wgpu` for immediate-mode GUI
- [ ] Expose material parameters (albedo, roughness, IOR) in the UI
- [ ] Camera settings panel (FOV, aperture, focus distance)
- [ ] Render stats display (samples/sec, total samples, resolution)
- [ ] Save rendered image to PNG

**Output:** A usable interactive path tracer with parameter tweaking.

---

## Dependency Summary

```
wgpu        — GPU abstraction (WebGPU API)
winit       — Window creation and input handling
pollster    — Block on async wgpu initialization
bytemuck    — Safe casting of structs to byte slices for GPU upload
glam        — Math library (vec3, mat4, etc.) for camera and scene on CPU
image       — Image loading (textures, environment maps) and PNG export
egui + egui-wgpu — Immediate-mode GUI (Phase 14)
tobj / gltf — Mesh loading (Phase 10)
```

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
