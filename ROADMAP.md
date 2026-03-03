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

## Code Structure & Best Practices

### Architecture Principles

**Separation of Concerns:**
- Entry point handles only window events, frame timing, and high-level orchestration
- Isolate wgpu resource management (device, pipelines, buffers, textures) in dedicated modules
- Keep domain logic (camera, scene) independent of GPU APIs where possible
- Scene data structures should be separate from GPU upload logic

**GPU Data Alignment:**
- Use `#[repr(C)]` for all GPU-bound structs
- Derive `bytemuck::Pod` and `bytemuck::Zeroable` for safe casting
- Respect WGSL alignment rules:
  - `vec3` is 16-byte aligned (pad to `vec4` or use explicit padding)
  - Struct fields must align to their natural size
- Use `wgpu::util::DeviceExt::create_buffer_init` for small one-time uploads

**Shader Loading:**
- Embed shaders at compile time: `include_str!("../shaders/path_trace.wgsl")`
- Use `ShaderModuleDescriptor` with source as a string
- For hot-reload during development, optionally read from disk and watch for changes

**Error Handling:**
- Propagate wgpu errors with `Result` and `?` where possible
- Use `pollster::block_on` or `env_logger` for async wgpu initialization
- Log pipeline validation errors clearly; wgpu provides detailed messages

**Performance:**
- Minimize CPU-GPU synchronization; avoid `queue.submit()` + immediate read-back
- Reuse buffers and textures across frames (reallocate only on resize)
- Use storage buffers (not uniforms) for large arrays (BVH nodes, primitives)
- Keep compute workgroup size at 8×8 or 16×16 for typical GPUs

### WGSL Conventions

**Naming:**
- `snake_case` for functions and variables (WGSL standard)
- `PascalCase` for struct types
- `SCREAMING_CASE` for constants

**Modularity:**
- Group related functions (e.g., `ray_sphere_intersect`, `ray_triangle_intersect`)
- Use comments to delimit sections: `// === Ray Intersection ===`
- Keep the main kernel function (`@compute fn path_trace(...)`) concise

**Shared Code:**
- Define shared structs (Ray, Hit, Material) consistently between CPU and GPU
- Use `const` for compile-time values (max bounces, workgroup size)
- Avoid excessive branching; prefer branchless math where possible (GPU performance)

**Debugging:**
- Output debug colors (e.g., visualize normals as RGB) during development
- Use `storageBarrier()` when needed between dependent compute dispatches
- Test shaders incrementally—verify each phase before moving on

### Code Quality Expectations

**Correctness:**
- Compile without warnings (`cargo build --release` should be clean)
- All GPU data structures must respect WGSL alignment rules—test on multiple platforms
- Validate shader compilation errors provide actionable feedback
- No unsafe code unless absolutely necessary; if used, document safety invariants

**Readability:**
- Use descriptive names for functions, variables, and types
- Add doc comments (`///`) for public APIs and non-obvious algorithms
- Keep functions focused and short; extract complex logic into helper functions
- Use `rustfmt` for consistent formatting; use `clippy` for idiomatic Rust

**Maintainability:**
- Avoid "magic numbers"—use named constants for important values (max bounces, workgroup sizes)
- Group related functionality into modules; avoid monolithic files
- Refactor when adding features exposes poor abstractions
- Keep shader code modular with clear section boundaries (comments)

**Performance Awareness:**
- Profile before optimizing—don't prematurely sacrifice clarity for speed
- Document performance-critical code paths and assumptions
- Minimize allocations in hot loops (frame updates, input handling)
- Balance GPU parallelism with memory bandwidth (consider workgroup sizes, cache coherency)

**Testing Strategy:**
- Visual validation: each phase should produce verifiable output
- Unit test pure functions (BVH construction, ray-sphere intersection math on CPU)
- Regression testing: save reference images for comparison after refactors
- Platform testing: verify on at least two GPU vendors (NVIDIA, AMD, or Intel)

### Modularity & Long-Term Maintainability

**Interface Design:**
- Define clear, minimal public APIs for each module—hide implementation details
- Use traits for abstractions that may have multiple implementations (e.g., `Primitive` trait for spheres, triangles, meshes)
- Keep interfaces stable; add new methods rather than changing signatures when possible
- Document preconditions, postconditions, and invariants for public functions

**Dependency Management:**
- Minimize coupling—modules should depend on abstractions, not concrete types
- Avoid circular dependencies; restructure if they emerge
- Use dependency injection: pass resources (device, queue) explicitly rather than storing globally
- Consider builder patterns for complex configuration (camera setup, render pipeline creation)

**Configuration Over Hardcoding:**
- Externalize constants that might change (scene files, render settings, shader parameters)
- Use configuration structs passed at initialization rather than scattered literals
- Support runtime reconfiguration where feasible (resolution, quality presets)
- Document default values and valid ranges

**Composability:**
- Design small, single-purpose components that can be combined
- Prefer pure functions where possible—deterministic, easier to test and reason about
- Use composition over inheritance (Rust's trait system encourages this)
- Each module should be independently testable without full system setup

**State Management:**
- Minimize shared mutable state; prefer message passing or explicit ownership transfer
- Keep GPU resource lifetime tied to clear ownership (avoid `Rc<RefCell<>>` for wgpu resources)
- Document state transitions (e.g., when accumulation resets, when buffers are recreated)
- Use type states to enforce valid state transitions at compile time where appropriate

**Evolution Strategy:**
- Design for extension: anticipate adding new material types, primitive types, post-effects
- Use enums with non-exhaustive patterns where appropriate (`#[non_exhaustive]`)
- Version breaking changes thoughtfully; document migration paths in commit messages
- Keep experimental features behind feature flags (`#[cfg(feature = "experimental")]`)

**Documentation Practices:**
- Each module should have a top-level doc comment explaining its purpose and usage
- Document *why* decisions were made, not just *what* the code does
- Keep a CHANGELOG or update commit messages with rationale for architectural changes
- Use examples in doc comments for complex APIs

### Git Workflow

- **Commit per phase:** Each ROADMAP phase = one or more commits
- **Tag milestones:** `git tag phase-3-ray-generation` after completing a phase
- **Branch for experiments:** Use feature branches for risky changes (e.g., BVH refactor)
- **Keep commits atomic:** One logical change per commit (e.g., "Add sphere intersection")

### Code Review Process

**Review Guidelines:**
- Review code incrementally—each phase completion triggers a review checkpoint
- Focus on correctness first, then readability, then performance
- Verify visual output matches expectations for the phase (screenshots/recordings)
- Check that code compiles without warnings and passes `clippy` lints

**Review Checklist:**

*Correctness & Safety:*
- [ ] GPU data structures use `#[repr(C)]` and bytemuck derives
- [ ] WGSL alignment rules are respected (vec3 padding, struct alignment)
- [ ] Buffer sizes match shader bindings (group/binding numbers are consistent)
- [ ] No unsafe code without clear safety documentation
- [ ] Error handling is present for fallible operations

*Architecture & Design:*
- [ ] Changes follow separation of concerns (domain logic vs GPU API)
- [ ] New abstractions have clear, documented interfaces
- [ ] Dependencies are minimal and explicit
- [ ] State management is clear and ownership is obvious

*GPU-Specific Concerns:*
- [ ] Workgroup sizes are appropriate for target GPUs (multiples of 32/64)
- [ ] No unnecessary CPU-GPU synchronization points
- [ ] Buffers/textures are reused across frames when possible
- [ ] Shader code avoids excessive branching in hot paths

*Code Quality:*
- [ ] Names are descriptive and follow Rust conventions
- [ ] Complex algorithms have explanatory comments
- [ ] Public APIs have doc comments with usage examples
- [ ] No "magic numbers"—important constants are named

*Testing & Validation:*
- [ ] Phase output visually matches expected result
- [ ] Pure functions have unit tests where appropriate
- [ ] Changes have been tested on resize/window events
- [ ] Accumulation resets correctly when scene/camera changes

**Feedback Culture:**
- Be specific: "This buffer allocation happens every frame" not "This is slow"
- Suggest alternatives: "Consider caching this computation" with code example
- Ask questions: "Why did you choose this approach?" to understand intent
- Acknowledge good patterns: Call out clever solutions or clean abstractions
- Prioritize feedback: Critical (correctness) > Important (maintainability) > Nice-to-have (style)

**Reviewer Responsibilities:**
- Test the branch locally and verify visual output
- Run `cargo clippy` and `cargo fmt --check` before approving
- If suggesting major changes, explain the reasoning and long-term benefit
- Approve when changes meet quality bar, even if minor improvements remain (open follow-up issues)

**Author Responsibilities:**
- Provide context in PR description: what changed, why, what was tested
- Self-review before requesting review (walk through your own diff)
- Respond to all feedback even if not making suggested changes
- Update based on feedback promptly; re-request review when ready

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
