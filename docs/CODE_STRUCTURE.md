# Code Structure & Best Practices

## Architecture Principles

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

## WGSL Conventions

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

## Code Quality Expectations

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

## Modularity & Long-Term Maintainability

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

## Git Workflow

- **Commit per phase:** Each ROADMAP phase = one or more commits
- **Tag milestones:** `git tag phase-3-ray-generation` after completing a phase
- **Branch for experiments:** Use feature branches for risky changes (e.g., BVH refactor)
- **Keep commits atomic:** One logical change per commit (e.g., "Add sphere intersection")

## Code Review Process

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
