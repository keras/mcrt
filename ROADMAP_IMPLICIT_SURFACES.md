# Implicit Surfaces via Runtime WGSL Code Generation — Roadmap

Add a new `type: implicit` object to the path tracer whose surface is defined by
a Signed Distance Function (SDF).  SDF expressions are written in YAML scene
files using a declarative DSL — no Rust recompilation required.  The DSL
supports raw math expressions, built-in SDF primitives, CSG combinators, domain
transforms, and fractal presets.  At load time the DSL is parsed into an AST,
compiled to a WGSL function, and spliced into the path tracer shader before
pipeline creation.

> **Strategy:** Runtime WGSL code generation (Strategy A).  The generated SDF
> function is native shader code, fully optimised by the GPU driver.  Numeric
> parameters that change at runtime (radius, blend factor, power) are uploaded
> as uniforms so they can be adjusted without pipeline recompilation.

---

## Phase SDF-1: Expression Parser & WGSL Emitter

**Goal:** Parse a scalar math expression string into an AST and emit valid WGSL
code that evaluates it as a distance function.

### Expression Language

The expression operates on an implicit input point `(x, y, z)` and must
evaluate to a single `f32` — the signed distance.  Supported constructs:

| Construct | Example | WGSL output |
|-----------|---------|-------------|
| Variables | `x`, `y`, `z` | `p.x`, `p.y`, `p.z` |
| Literals | `1.5`, `0.0` | `1.5`, `0.0` |
| Arithmetic | `x^2 + y^2` | `(p.x * p.x) + (p.y * p.y)` |
| Unary minus | `-x` | `(-p.x)` |
| Functions | `sqrt(x^2+y^2)` | `sqrt((p.x*p.x)+(p.y*p.y))` |
| Named params | `r`, `k` | `params.x`, `params.y` (uniform) |
| Constants | `pi`, `tau` | `3.14159265...`, `6.28318530...` |

Supported functions (map 1:1 to WGSL built-ins): `abs`, `sqrt`, `sin`, `cos`,
`tan`, `asin`, `acos`, `atan`, `atan2`, `exp`, `log`, `pow`, `min`, `max`,
`clamp`, `floor`, `ceil`, `round`, `fract`, `sign`, `length`, `dot`,
`smoothstep`, `mix`.

### Tasks

- [ ] Add `fasteval` (or `exmex`) crate as a dependency, **or** write a
  custom recursive-descent parser (~300-400 lines) for the expression grammar
  above.  A custom parser is preferred if SDF-specific operators (smooth_min,
  domain ops) will be first-class syntax rather than function calls.

- [ ] Define `SdfExpr` AST enum:
  ```rust
  enum SdfExpr {
      Literal(f32),
      Var(SdfVar),           // X, Y, Z, or named param
      Unary(UnaryOp, Box<SdfExpr>),
      Binary(BinaryOp, Box<SdfExpr>, Box<SdfExpr>),
      Call(SdfFunc, Vec<SdfExpr>),
  }
  ```

- [ ] Implement `fn emit_wgsl(expr: &SdfExpr) -> String` that walks the AST
  and emits a parenthesised WGSL expression string with correct operator
  precedence.

- [ ] Implement `fn parse_expr(input: &str) -> Result<SdfExpr, SdfParseError>`
  with clear error messages (line/column, expected token).

- [ ] Unit tests:
  - `"x^2 + y^2 + z^2 - 1.0"` → sphere SDF
  - `"sqrt(x^2 + y^2 + z^2) - r"` → sphere with param `r`
  - `"max(abs(x), max(abs(y), abs(z))) - 1.0"` → box SDF
  - Operator precedence: `"x + y * z"` → `(p.x + (p.y * p.z))`
  - Error cases: unclosed parens, unknown function, division by zero guard

**Output:** A Rust module `sdf_expr.rs` that can parse `"x^2+y^2+z^2-1"` and
produce `"((p.x*p.x)+(p.y*p.y))+(p.z*p.z)-(1.0)"`.

---

## Phase SDF-2: SDF Node Tree & Declarative DSL

**Goal:** Extend the expression system to a full SDF scene graph with
primitives, CSG combinators, domain transforms, and fractal presets — all
expressible in YAML.

### SDF Node Types

```rust
enum SdfNode {
    // --- Primitives ---
    Sphere   { radius: f32 },
    Box      { half_extents: [f32; 3] },
    Torus    { major_r: f32, minor_r: f32 },
    Cylinder { radius: f32, height: f32 },
    Plane    { normal: [f32; 3], offset: f32 },

    // --- Raw expression (Phase SDF-1) ---
    Expr     { expr: String },

    // --- CSG combinators ---
    Union        { children: Vec<SdfNode> },
    Intersection { children: Vec<SdfNode> },
    Subtraction  { a: Box<SdfNode>, b: Box<SdfNode> },
    SmoothUnion  { k: f32, children: Vec<SdfNode> },
    SmoothSub    { k: f32, a: Box<SdfNode>, b: Box<SdfNode> },

    // --- Domain transforms ---
    Translate { offset: [f32; 3], child: Box<SdfNode> },
    Scale     { factor: f32,      child: Box<SdfNode> },  // uniform only
    RotateY   { degrees: f32,     child: Box<SdfNode> },
    Symmetry  { axes: [bool; 3],  child: Box<SdfNode> },  // abs(p.x), etc.
    Repeat    { spacing: [f32; 3], child: Box<SdfNode> },
    Twist     { rate: f32,        child: Box<SdfNode> },
    Bend      { rate: f32,        child: Box<SdfNode> },
    Round     { radius: f32,      child: Box<SdfNode> },
    Onion     { thickness: f32,   child: Box<SdfNode> },

    // --- Fractal presets ---
    Mandelbulb  { power: f32, iterations: u32 },
    Menger      { iterations: u32 },
    Sierpinski  { iterations: u32, scale: f32 },
    JuliaQuat   { c: [f32; 4], iterations: u32 },
}
```

### YAML DSL Format

```yaml
objects:
  # Simplest form: raw expression
  - type: implicit
    material: glass
    bounds_radius: 1.5
    center: [0, 1, 0]
    sdf:
      expr: "sqrt(x*x + y*y + z*z) - 1.0"

  # Built-in primitive
  - type: implicit
    material: gold
    bounds_radius: 2.0
    center: [3, 1, 0]
    sdf:
      torus: { major_r: 1.0, minor_r: 0.3 }

  # CSG composition
  - type: implicit
    material: red
    bounds_radius: 2.5
    center: [-3, 1, 0]
    sdf:
      smooth_union:
        k: 0.3
        children:
          - sphere: { radius: 1.0 }
          - translate:
              offset: [1.2, 0, 0]
              child:
                box: { half_extents: [0.7, 0.7, 0.7] }

  # Fractal
  - type: implicit
    material: white
    bounds_radius: 1.3
    center: [0, 2, -3]
    sdf:
      mandelbulb: { power: 8.0, iterations: 12 }

  # Domain ops: repeated twisted torus
  - type: implicit
    material: green
    bounds_radius: 10.0
    center: [0, 0, 0]
    sdf:
      repeat:
        spacing: [4.0, 4.0, 4.0]
        child:
          twist:
            rate: 0.5
            child:
              torus: { major_r: 0.8, minor_r: 0.2 }
```

### Tasks

- [ ] Define `SdfNode` enum with `serde::Deserialize` (tag = first key,
  similar to how `ObjectDesc` works).

- [ ] Implement `fn compile_sdf_node(node: &SdfNode) -> String` that
  recursively emits WGSL code.  Each node type maps to a known pattern:

  | Node | WGSL pattern |
  |------|-------------|
  | `Sphere { r }` | `length(p) - r` |
  | `Box { h }` | Inigo Quilez exact box SDF |
  | `Union [a, b]` | `min(sdf_a(p), sdf_b(p))` |
  | `SmoothUnion { k, a, b }` | Polynomial smooth min |
  | `Translate { offset, child }` | `child(p - offset)` |
  | `Symmetry { [true,_,_], child }` | `child(vec3(abs(p.x), p.y, p.z))` |
  | `Repeat { s, child }` | `child(p - s * round(p / s))` |
  | `Mandelbulb { power, iter }` | Full mandelbulb distance estimator |

- [ ] For each fractal preset, write a self-contained WGSL helper function
  (e.g., `fn _sdf_mandelbulb_N(p: vec3<f32>, power: f32, iter: u32) -> f32`)
  emitted once and called by reference.  The `N` suffix is a unique ID per
  implicit object to avoid name collisions when multiple fractals coexist.

- [ ] Implement bounding-box estimation for each node type:
  - Sphere: `center ± radius`
  - Box: `center ± half_extents`
  - Union: AABB union of children
  - Translate: shift child AABB
  - Fractal presets: known bounds (Mandelbulb ≈ 1.2, Menger ≈ 1.0, etc.)
  - Raw `expr`: user-supplied `bounds_radius` (required field)

- [ ] Add `ObjectDesc::Implicit` variant to scene.rs:
  ```rust
  Implicit {
      center: [f32; 3],
      bounds_radius: f32,
      material: MaterialRef,
      sdf: SdfNode,
      #[serde(default)]
      transform: TransformDesc,
      #[serde(default)]
      params: Option<[f32; 4]>,  // runtime-adjustable parameters
  }
  ```

- [ ] Extend `load_scene_from_yaml` to parse `type: implicit` objects.

- [ ] Unit tests: parse each YAML example above and verify the emitted WGSL
  compiles (validate with `naga` in dev-dependencies, already available).

**Output:** `scene.yaml` can describe arbitrary SDF objects; the loader
produces WGSL source code for each implicit object's distance function.

---

## Phase SDF-3: Shader Template & Pipeline Recompilation

**Goal:** Integrate the generated SDF functions into the path tracer shader
and rebuild the GPU pipeline at scene load time.

### Shader Template Architecture

The current shader is loaded via `include_str!("shaders/path_trace.wgsl")`.
This phase replaces the static include with a template that has an injection
point for generated SDF code:

```
┌──────────────────────────────────────────────┐
│  path_trace.wgsl (static template)           │
│                                              │
│  ... existing bindings, structs, BVH ...     │
│                                              │
│  // @@SDF_FUNCTIONS@@                        │  ← injection marker
│                                              │
│  fn sdf_scene(p, obj_idx) -> f32 { ... }     │  ← generated dispatch
│  fn sdf_march(ray, t_min, t_max) -> Hit      │  ← sphere tracing loop
│  fn sdf_normal(p, obj_idx) -> vec3f           │  ← tetrahedron gradient
│                                              │
│  fn world_hit(r, t_min, t_max) -> Hit {      │
│      ... existing sphere + mesh BVH ...      │
│      let sdf_h = sdf_march(r, t_min, best);  │  ← NEW
│      ...                                     │
│  }                                           │
└──────────────────────────────────────────────┘
```

The `// @@SDF_FUNCTIONS@@` marker is replaced at runtime with the generated
per-object SDF functions.  Everything else in the shader remains static.

### Generated Code Structure

For a scene with two implicit objects (a mandelbulb and a smooth-union), the
injected code looks like:

```wgsl
// --- Generated SDF functions (2 objects) ---

fn _sdf_mandelbulb_0(p: vec3<f32>) -> f32 {
    // ... mandelbulb distance estimator ...
}

fn _sdf_smooth_union_1(p: vec3<f32>) -> f32 {
    let a = length(p) - 1.0;
    let b = sdf_box(p - vec3<f32>(1.2, 0.0, 0.0), vec3<f32>(0.7, 0.7, 0.7));
    return smin_poly(a, b, 0.3);
}

// --- SDF shared helpers ---

fn sdf_box(p: vec3<f32>, b: vec3<f32>) -> f32 { ... }
fn smin_poly(a: f32, b: f32, k: f32) -> f32 { ... }
```

### GPU Data Layout

```rust
/// Per-implicit-object descriptor uploaded to a storage buffer.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSdfObject {
    /// AABB min (xyz) + material index (w, bitcast to u32).
    pub aabb_min_mat: [f32; 4],
    /// AABB max (xyz) + sdf_object_index (w, bitcast to u32).
    pub aabb_max_idx: [f32; 4],
    /// User-adjustable parameters (radius, power, k, etc.).
    pub params: [f32; 4],
}
```

### Tasks

- [ ] Add `// @@SDF_FUNCTIONS@@` marker to `path_trace.wgsl`.

- [ ] Write `fn assemble_shader(template: &str, sdf_objects: &[SdfCompiled]) -> String`
  that replaces the marker with the generated code.

- [ ] Write the static WGSL scaffolding (added to the template, not generated):
  - `fn sdf_scene(p: vec3<f32>, obj_idx: u32) -> f32` — a `switch` over
    `obj_idx` dispatching to `_sdf_*_N()` functions.
  - `fn sdf_march(r: Ray, t_min: f32, t_max: f32) -> HitRecord` — sphere
    tracing loop with configurable `MAX_SDF_STEPS` (128), `SDF_EPSILON`
    (0.0001), and per-object AABB entry/exit clipping.
  - `fn sdf_normal_at(p: vec3<f32>, obj_idx: u32) -> vec3<f32>` — tetrahedron
    gradient technique (4 SDF evaluations).

- [ ] Add `sdf_objects: array<SdfObject>` storage buffer binding
  (new binding slot, or reuse an existing padding slot).

- [ ] In `gpu.rs`, change shader loading:
  ```rust
  // Before:
  let source = include_str!("shaders/path_trace.wgsl");
  // After:
  let template = include_str!("shaders/path_trace.wgsl");
  let source = sdf_compiler::assemble_shader(template, &compiled_sdfs);
  ```

- [ ] Pipeline recompilation: after `assemble_shader`, call
  `device.create_shader_module()` + `device.create_compute_pipeline()` as
  usual.  This happens once at scene load (and on hot-reload).

- [ ] Integrate into `world_hit()`:
  ```wgsl
  fn world_hit(r: Ray, t_min: f32, t_max: f32) -> HitRecord {
      let sh = sphere_bvh_hit(r, t_min, t_max);
      let mesh_tmax = select(t_max, sh.t, sh.t > 0.0);
      let mh = mesh_bvh_hit(r, t_min, mesh_tmax);
      var best = select(sh, mh, mh.t > 0.0);
      let sdf_tmax = select(t_max, best.t, best.t > 0.0);
      let sdf_h = sdf_march(r, t_min, sdf_tmax);    // NEW
      if sdf_h.t > 0.0 { best = sdf_h; }             // NEW
      return best;
  }
  ```

- [ ] Validate the assembled shader string with `naga` before passing to wgpu
  (catch codegen bugs early with a clear error message rather than a wgpu
  validation panic).

- [ ] Integration test: load a scene with one `type: implicit` sphere
  (`"x*x+y*y+z*z-1.0"`), verify the shader compiles, and compare the rendered
  result against an analytic sphere at the same position.

**Output:** Implicit SDF objects render alongside spheres and meshes in the
path tracer.  The sphere equation `x^2+y^2+z^2=r^2` works as a drop-in
replacement for the analytic sphere.

---

## Phase SDF-4: Sphere Tracing Core & Normal Estimation

**Goal:** Implement robust sphere tracing and normal estimation in WGSL,
handling edge cases (AABB clipping, adaptive epsilon, over-relaxation).

### Sphere Tracing Algorithm

```wgsl
fn sdf_march(r: Ray, t_min: f32, t_max_in: f32) -> HitRecord {
    var best: HitRecord;
    best.t = -1.0;
    let n_objs = arrayLength(&sdf_objects);
    if n_objs == 0u { return best; }

    for (var oi = 0u; oi < n_objs; oi++) {
        let obj = sdf_objects[oi];
        let aabb_min = obj.aabb_min_mat.xyz;
        let aabb_max = obj.aabb_max_idx.xyz;
        let t_max = select(t_max_in, best.t, best.t > 0.0);

        // Clip ray to object AABB — skip entirely on miss.
        let t_aabb = ray_aabb_interval(r, aabb_min, aabb_max, t_min, t_max);
        if t_aabb.x > t_aabb.y { continue; }

        var t = t_aabb.x;
        for (var step = 0u; step < MAX_SDF_STEPS; step++) {
            let p = r.origin + t * r.dir;
            let d = sdf_scene(p, oi);
            // Adaptive epsilon: scale with distance to reduce aliasing.
            let eps = max(SDF_EPSILON, SDF_EPSILON * t);
            if d < eps {
                best.t          = t;
                best.point      = p;
                best.normal     = sdf_normal_at(p, oi);
                best.front_face = dot(r.dir, best.normal) < 0.0;
                best.normal     = select(-best.normal, best.normal, best.front_face);
                best.mat_index  = bitcast<u32>(obj.aabb_min_mat.w);
                best.uv         = sdf_uv(best.normal);  // spherical UV from normal
                break;
            }
            t += d;
            if t > t_aabb.y { break; }
        }
    }
    return best;
}
```

### Normal Estimation — Tetrahedron Technique (4 evals)

```wgsl
fn sdf_normal_at(p: vec3<f32>, obj_idx: u32) -> vec3<f32> {
    let e = max(SDF_NORMAL_EPS, SDF_NORMAL_EPS * length(p));
    let k = vec2<f32>(1.0, -1.0);
    return normalize(
        k.xyy * sdf_scene(p + k.xyy * e, obj_idx) +
        k.yyx * sdf_scene(p + k.yyx * e, obj_idx) +
        k.yxy * sdf_scene(p + k.yxy * e, obj_idx) +
        k.xxx * sdf_scene(p + k.xxx * e, obj_idx)
    );
}
```

### Tasks

- [ ] Implement `ray_aabb_interval()` returning `vec2<f32>(t_enter, t_exit)`.

- [ ] Implement `sdf_march()` with per-object AABB clipping as shown above.

- [ ] Implement `sdf_normal_at()` using the tetrahedron technique.

- [ ] Implement `sdf_uv()` — spherical UV mapping from the hit normal
  (reuse the same phi/theta mapping as the analytic sphere).

- [ ] Add tunable constants:
  - `MAX_SDF_STEPS`: 128 (simple), 256 (fractals)
  - `SDF_EPSILON`: 0.0001
  - `SDF_NORMAL_EPS`: 0.0005

- [ ] Optional: implement over-relaxation (omega = 1.2-1.6) with fallback
  detection to reduce step count by ~30-50%.

- [ ] Test correctness:
  - Sphere SDF matches analytic sphere intersection within epsilon.
  - Box SDF produces correct normals on each face.
  - Torus SDF self-shadows correctly.
  - Fractal normals are smooth and produce correct lighting.

**Output:** Robust sphere tracing that handles all SDF node types with correct
normals, adaptive precision, and AABB-bounded marching.

---

## Phase SDF-5: Fractal Presets

**Goal:** Implement high-quality distance estimators for the four fractal
presets, usable as single YAML keys.

### Fractal Implementations

| Fractal | Key Parameters | Bounding Radius | Notes |
|---------|---------------|-----------------|-------|
| **Mandelbulb** | `power` (8.0), `iterations` (8-15) | ~1.2 | Spherical coords, Hubbard-Douady DE; use trig identity optimisation for power 8 |
| **Menger sponge** | `iterations` (3-5) | 1.0 (unit cube) | Iterative cross subtraction via folding |
| **Sierpinski tetrahedron** | `iterations` (8-15), `scale` (2.0) | ~1.8 | Vertex folding + scale |
| **Quaternion Julia** | `c: [f32; 4]`, `iterations` (8-12) | ~1.5-2.0 | Quaternion multiplication, Hubbard-Douady DE |

### Tasks

- [ ] Implement each fractal as a WGSL helper function emitted by the
  compiler when the corresponding `SdfNode` variant is encountered.

- [ ] For the Mandelbulb, implement both the general trigonometric version
  and the optimised polynomial version (power 8 only, avoids
  `acos`/`atan2`/`sin`/`cos`).  Select the optimised path when
  `power == 8.0`.

- [ ] Expose fractal-specific parameters via the `params` uniform so they
  can be adjusted in the egui panel without shader recompilation:
  - Mandelbulb: `params.x = power`, `params.y = iterations`
  - Menger: `params.x = iterations`
  - Sierpinski: `params.x = iterations`, `params.y = scale`
  - Julia: `params = c` (all four components)

- [ ] Write an `assets/scene_fractals.yaml` demo scene featuring all four
  fractal types with appropriate materials and camera position.

- [ ] Test: verify each fractal renders without artifacts (missed surfaces,
  banding, noise in normals).  Larger `SDF_NORMAL_EPS` (0.0005-0.001) may
  be needed for fractals whose distance estimators are approximate.

**Output:** Four fractal types available as single-key YAML declarations,
rendering correctly in the path tracer.

---

## Phase SDF-6: egui Integration & Hot-Reload

**Goal:** Expose SDF parameters in the UI and support hot-reload of SDF
expressions when the scene YAML changes.

### Tasks

- [ ] Add an **"Implicit Objects"** panel to the egui sidebar listing each
  SDF object with its type and editable parameters.

- [ ] For fractal objects, add sliders:
  - Power (Mandelbulb): 2.0 – 16.0
  - Iterations: 1 – 20
  - Scale (Sierpinski): 1.5 – 3.0
  - Julia c components: -2.0 – 2.0 each

- [ ] Parameter changes update the `params` uniform buffer only — no
  pipeline recompilation (instant feedback).

- [ ] For structural changes (editing the `sdf:` tree in YAML), the existing
  `notify`-based hot-reload triggers a full re-parse → re-compile →
  pipeline rebuild.  Debounce recompilation by 200ms after the last file
  change event.

- [ ] Add a text field in egui for entering raw SDF expressions.  On
  submit: parse → validate (naga) → recompile pipeline.  Show parse/
  validation errors inline.

- [ ] Display per-frame SDF stats in the render stats panel:
  - Number of implicit objects
  - Whether the pipeline was recently recompiled

- [ ] Optional: add an "SDF expression" text input that allows previewing
  an expression without editing the YAML file.

**Output:** Interactive fractal parameter tweaking via sliders; structural
SDF changes hot-reload from YAML.

---

## Phase SDF-7: Validation & Performance

**Goal:** Ensure correctness, acceptable performance, and regression safety.

### Correctness Tests

- [ ] **Sphere equivalence**: render `type: implicit` sphere
  (`"x*x+y*y+z*z-1.0"`) next to `type: sphere` with identical radius,
  center, and material.  The pixel difference should be below a threshold
  (accounting for sphere-tracing epsilon vs analytic precision).

- [ ] **CSG sanity**: `subtraction(sphere(1.0), sphere(0.8))` should
  produce a hollow shell.  Verify normals point inward on the inner surface.

- [ ] **Normal consistency**: for each SDF primitive, compare the numerical
  gradient normal against the known analytic normal.  Max angular error
  < 2 degrees.

- [ ] **Shader compilation**: integration test that parses every example
  from this roadmap's YAML snippets, assembles the shader, and validates
  with naga.

### Performance

- [ ] Benchmark: measure frame time with 0, 1, 4, and 8 SDF objects
  (simple primitives) vs. equivalent analytic/mesh objects.

- [ ] Benchmark: Mandelbulb at iterations 8, 12, 16 — document frame time
  and identify the iteration count sweet spot.

- [ ] Profile: use wgpu timestamp queries to measure SDF marching time
  as a fraction of total path-trace dispatch time.

- [ ] If SDF marching dominates frame time, consider:
  - Reducing `MAX_SDF_STEPS` for non-fractal objects (32-64 is sufficient).
  - Per-object step count as an `SdfNode` field.
  - Over-relaxation (Phase SDF-4 optional task).

**Output:** Documented performance characteristics; regression tests in CI.

---

## Optional Extension: Hybrid Parameter Strategy (Strategy C)

> This extension separates *expression structure* (which triggers shader
> recompilation) from *numeric parameters* (which are uniforms updated
> per-frame).  It is not required for a functional system but improves the
> interactive editing experience.

### Concept

Every numeric literal in an SDF expression can optionally be promoted to a
named parameter stored in a uniform buffer.  The expression
`"smooth_union(k, sphere(r1), box(r2, r2, r2))"` compiles to WGSL that
reads `params.x` for `k`, `params.y` for `r1`, `params.z` for `r2`.

The YAML format gains a `params` mapping:

```yaml
- type: implicit
  material: gold
  sdf:
    smooth_union:
      k: $blend   # reference to named param
      children:
        - sphere: { radius: $r1 }
        - box: { half_extents: [$r2, $r2, $r2] }
  params:
    blend: 0.3
    r1: 1.0
    r2: 0.7
```

### Tasks

- [ ] Extend the parser to recognise `$name` tokens as parameter references.
- [ ] During WGSL emission, replace `$name` with `sdf_params[obj_idx].field`
  lookups into a per-object parameter buffer.
- [ ] Upload a `Vec<[f32; 16]>` (up to 16 params per object) as a storage
  buffer alongside `GpuSdfObject`.
- [ ] In egui, auto-generate sliders for each named parameter with
  sensible ranges (inferred from the initial value ± 10x, or user-specified
  `param_range` in YAML).
- [ ] Slider changes write directly to the parameter buffer — no pipeline
  recompilation, no accumulation reset needed (the image reconverges
  naturally).

---

## File Plan

| File | Change |
|------|--------|
| `src/sdf_expr.rs` | **New.** Expression parser, AST, WGSL emitter. |
| `src/sdf_compiler.rs` | **New.** SdfNode tree → WGSL code generation, shader assembly, AABB estimation. |
| `src/scene.rs` | Add `ObjectDesc::Implicit`, `SdfNode` deserialization, `GpuSdfObject`. |
| `src/gpu.rs` | Shader template assembly, SDF storage buffer, pipeline rebuild logic. |
| `src/shaders/path_trace.wgsl` | Add `@@SDF_FUNCTIONS@@` marker, `sdf_march`, `sdf_normal_at`, `sdf_scene`, `sdf_uv`, SDF shared helpers. Extend `world_hit`. |
| `src/app.rs` | egui panel for SDF parameters. |
| `src/main.rs` | Add `mod sdf_expr; mod sdf_compiler;` |
| `assets/scene_fractals.yaml` | **New.** Demo scene with all fractal types. |
| `tests/sdf_integration.rs` | **New.** Shader compilation + naga validation tests. |

---

## Architecture Diagram

```
  scene.yaml                    Rust (CPU)                         GPU (WGSL)
 ┌────────────┐
 │ type:      │    parse
 │   implicit │ ──────────► SdfNode AST
 │ sdf:       │                 │
 │   mandel.. │                 │ compile_sdf_node()
 │            │                 ▼
 └────────────┘          WGSL fn string ──► assemble_shader() ──► path_trace.wgsl
                                                                   (with injected
                                  GpuSdfObject ──► storage buf      SDF functions)
                                  params       ──► uniform buf          │
                                                                        ▼
                                                                  sdf_march()
                                                                  ├─ AABB clip
                                                                  ├─ sphere trace
                                                                  ├─ sdf_scene()
                                                                  │    └─ switch(obj_idx)
                                                                  │         ├─ _sdf_0()
                                                                  │         └─ _sdf_1()
                                                                  └─ sdf_normal_at()
                                                                        │
                                                                        ▼
                                                                  world_hit()
                                                                  (unified with
                                                                   sphere + mesh BVH)
```

---

## Milestone Summary

| Phase | Milestone | Key Deliverable |
|-------|-----------|-----------------|
| SDF-1 | Expression Parser & WGSL Emitter | Parse `"x^2+y^2+z^2-1"` → valid WGSL |
| SDF-2 | SDF Node Tree & Declarative DSL | Full SDF scene graph in YAML |
| SDF-3 | Shader Template & Pipeline Recompilation | SDF objects render in path tracer |
| SDF-4 | Sphere Tracing Core & Normal Estimation | Robust marching with AABB clipping |
| SDF-5 | Fractal Presets | Mandelbulb, Menger, Sierpinski, Julia |
| SDF-6 | egui Integration & Hot-Reload | Interactive parameter sliders, live YAML reload |
| SDF-7 | Validation & Performance | Tests, benchmarks, documented perf |

Phases SDF-1 through SDF-4 form the **minimum viable feature**.  SDF-5 adds
the fractal showcase.  SDF-6 and SDF-7 are polish and production-readiness.

---

*Each phase builds on the previous one and produces a verifiable result.
Phases can be implemented as separate Git milestones.*
