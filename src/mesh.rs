// mesh.rs — Phase 10: Triangle mesh types, BVH builder, UV-sphere generator, OBJ loader
//
// Defines GPU-friendly vertex and triangle types, a SAH BVH builder over
// triangle primitives, a procedural UV-sphere mesh generator, and a minimal
// OBJ parser.  The mesh BVH reuses `GpuBvhNode` from bvh.rs — the layout
// contract is identical (root at 0, left child = node+1, right = right_or_offset,
// prim_count > 0 for leaves) but primitives are triangles rather than spheres.
//
// No wgpu types are present; this module is independently testable.

use crate::bvh::{Aabb, GpuBvhNode, BVH_LEAF_MAX, SAH_BINS};
use bytemuck::Zeroable;

// ---------------------------------------------------------------------------
// GPU types (must match WGSL structs in path_trace.wgsl)
// ---------------------------------------------------------------------------

/// A single vertex uploaded to the GPU.
///
/// **Size:** 48 bytes (3 × 16-byte vec4 — naturally aligned in WGSL).
///
/// All fields use 4-component vectors for WGSL alignment; `.w` components are
/// padding except where noted.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuVertex {
    /// World-space position.  `.w` is kept at `1.0` by convention (homogeneous
    /// coordinate), but the shader reads only `.xyz`; `.w` is unused padding.
    pub position: [f32; 4],
    /// Vertex normal (unit length).  `.w` is unused padding (always 0.0).
    pub normal: [f32; 4],
    /// Texture coordinates.  `.xy` = UV, `.zw` = unused padding.
    pub uv: [f32; 4],
}

/// A triangle defined by three indices into the vertex buffer plus a material slot.
///
/// **Size:** 16 bytes (one 16-byte chunk — vec4<u32>-aligned in WGSL).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuTriangle {
    /// Indices into the `GpuVertex` buffer.
    pub v: [u32; 3],
    /// Material slot index; same table as sphere materials.
    pub mat_idx: u32,
}

/// Output of a successful mesh BVH build.
pub struct MeshBvhResult {
    /// Triangles reordered so that each leaf's primitive range is contiguous.
    pub ordered_triangles: Vec<GpuTriangle>,
    /// Flat BVH node array in pre-order (root = index 0).
    /// Shares the `GpuBvhNode` layout with the sphere BVH.
    pub nodes: Vec<GpuBvhNode>,
    /// Vertices are **not** reordered — the triangle indices still reference
    /// the original vertex buffer, so they must be uploaded together.
    pub vertices: Vec<GpuVertex>,
}

// ---------------------------------------------------------------------------
// Internal BVH build types
// ---------------------------------------------------------------------------

struct TriPrimInfo {
    bounds:   Aabb,
    centroid: [f32; 3],
    tri_idx:  usize, // index into the original (unordered) triangle slice
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Build a flat BVH over `tris` using SAH with binned partitioning.
///
/// `verts` is required only to compute per-triangle AABBs; it is not reordered.
/// The returned [`MeshBvhResult`] carries `ordered_triangles` (a reordering of
/// `tris`) plus the flat pre-order `nodes` array.
///
/// An empty triangle list produces an empty result; callers must guard against
/// a zero-length node buffer before dispatching the GPU shader.
pub fn build_mesh_bvh(verts: &[GpuVertex], tris: &[GpuTriangle]) -> MeshBvhResult {
    if tris.is_empty() {
        return MeshBvhResult {
            ordered_triangles: Vec::new(),
            nodes: Vec::new(),
            vertices: verts.to_vec(),
        };
    }

    // Catch construction bugs early: every index must be in-bounds for `verts`.
    debug_assert!(
        tris.iter()
            .all(|t| t.v.iter().all(|&i| (i as usize) < verts.len())),
        "build_mesh_bvh: triangle vertex index out of bounds (verts.len()={})",
        verts.len()
    );

    let mut prim_infos: Vec<TriPrimInfo> = tris
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let bounds = triangle_aabb(verts, t);
            let centroid = bounds.centroid();
            TriPrimInfo { bounds, centroid, tri_idx: i }
        })
        .collect();

    let mut nodes = Vec::with_capacity(2 * tris.len());
    let mut ordered = Vec::with_capacity(tris.len());

    build_recursive(&mut prim_infos, &mut nodes, &mut ordered, tris);

    MeshBvhResult {
        ordered_triangles: ordered,
        nodes,
        vertices: verts.to_vec(),
    }
}

// ---------------------------------------------------------------------------
// Recursive SAH build
// ---------------------------------------------------------------------------

fn build_recursive(
    prims:   &mut [TriPrimInfo],
    nodes:   &mut Vec<GpuBvhNode>,
    ordered: &mut Vec<GpuTriangle>,
    all:     &[GpuTriangle],
) -> usize {
    let node_idx = nodes.len();
    nodes.push(GpuBvhNode::zeroed()); // placeholder

    let bounds = prims.iter().fold(Aabb::empty(), |acc, p| acc.union(&p.bounds));

    if prims.len() <= BVH_LEAF_MAX {
        nodes[node_idx] = make_leaf(bounds, ordered, prims, all);
        return node_idx;
    }

    let centroid_bounds = prims.iter().fold(Aabb::empty(), |acc, p| {
        acc.union(&Aabb::point(p.centroid))
    });

    match find_best_split(prims, &centroid_bounds, &bounds) {
        Some((axis, split_idx)) => {
            prims.sort_unstable_by(|a, b| {
                a.centroid[axis]
                    .partial_cmp(&b.centroid[axis])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let left_child = build_recursive(&mut prims[..split_idx], nodes, ordered, all);
            debug_assert_eq!(
                left_child,
                node_idx + 1,
                "pre-order invariant: left child must immediately follow parent"
            );
            let right_child = nodes.len();
            build_recursive(&mut prims[split_idx..], nodes, ordered, all);

            let (aabb_min, aabb_max) = bounds.to_gpu_pair();
            nodes[node_idx] = GpuBvhNode {
                aabb_min,
                aabb_max,
                right_or_offset: right_child as u32,
                prim_count: 0,
                _pad: [0; 2],
            };
        }
        None => {
            nodes[node_idx] = make_leaf(bounds, ordered, prims, all);
        }
    }

    node_idx
}

fn make_leaf(
    bounds:  Aabb,
    ordered: &mut Vec<GpuTriangle>,
    prims:   &[TriPrimInfo],
    all:     &[GpuTriangle],
) -> GpuBvhNode {
    let prim_start = ordered.len() as u32;
    for p in prims {
        ordered.push(all[p.tri_idx]);
    }
    let (aabb_min, aabb_max) = bounds.to_gpu_pair();
    GpuBvhNode {
        aabb_min,
        aabb_max,
        right_or_offset: prim_start,
        prim_count: prims.len() as u32,
        _pad: [0; 2],
    }
}

/// Binned SAH split for triangles.  Identical algorithm to the sphere version.
fn find_best_split(
    prims:           &[TriPrimInfo],
    centroid_bounds: &Aabb,
    node_bounds:     &Aabb,
) -> Option<(usize, usize)> {
    let parent_half_area = node_bounds.half_area().max(1e-8);
    let leaf_cost = prims.len() as f32;

    let mut best_cost = leaf_cost;
    let mut best_axis = 0usize;
    let mut best_split = 0usize;
    let mut found = false;

    for axis in 0..3usize {
        let axis_len = centroid_bounds.max[axis] - centroid_bounds.min[axis];
        if axis_len < 1e-8 {
            continue;
        }

        let mut bin_bounds = [Aabb::empty(); SAH_BINS];
        let mut bin_counts = [0u32; SAH_BINS];

        for p in prims {
            let t = (p.centroid[axis] - centroid_bounds.min[axis]) / axis_len;
            let b = ((t * SAH_BINS as f32) as usize).min(SAH_BINS - 1);
            bin_bounds[b] = bin_bounds[b].union(&p.bounds);
            bin_counts[b] += 1;
        }

        let mut left_bounds = [Aabb::empty(); SAH_BINS];
        let mut left_counts = [0u32; SAH_BINS];
        left_bounds[0] = bin_bounds[0];
        left_counts[0] = bin_counts[0];
        for i in 1..SAH_BINS {
            left_bounds[i] = left_bounds[i - 1].union(&bin_bounds[i]);
            left_counts[i] = left_counts[i - 1] + bin_counts[i];
        }

        let mut right_bounds = Aabb::empty();
        let mut right_count = 0u32;

        for split in (0..SAH_BINS - 1).rev() {
            right_bounds = right_bounds.union(&bin_bounds[split + 1]);
            right_count += bin_counts[split + 1];

            let l_count = left_counts[split];
            let r_count = right_count;
            if l_count == 0 || r_count == 0 {
                continue;
            }

            let cost = (left_bounds[split].half_area() * l_count as f32
                + right_bounds.half_area() * r_count as f32)
                / parent_half_area;

            if cost < best_cost {
                best_cost = cost;
                best_axis = axis;
                best_split = l_count as usize;
                found = true;
            }
        }
    }

    if found && best_split > 0 && best_split < prims.len() {
        Some((best_axis, best_split))
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// AABB helper for triangles
// ---------------------------------------------------------------------------

/// Compute the tight AABB around a triangle given the vertex buffer.
pub fn triangle_aabb(verts: &[GpuVertex], tri: &GpuTriangle) -> Aabb {
    let p0 = verts[tri.v[0] as usize].position;
    let p1 = verts[tri.v[1] as usize].position;
    let p2 = verts[tri.v[2] as usize].position;
    Aabb::from_points_3(&[p0, p1, p2])
}

// ---------------------------------------------------------------------------
// Procedural mesh generators
// ---------------------------------------------------------------------------

/// Build a UV sphere mesh centred at `center` with the given `radius`.
///
/// Generates `stacks` latitude bands and `slices` longitude segments, producing
/// `stacks × slices × 2` triangles.  All vertices carry smooth outward-pointing
/// normals, so even a coarse mesh (e.g. 12 × 24) looks round under path tracing.
///
/// Material index `mat_idx` is assigned to every triangle.
#[allow(dead_code)] // used in tests and available as public API for future scenes
pub fn build_uv_sphere_mesh(
    center:  [f32; 3],
    radius:  f32,
    stacks:  u32,
    slices:  u32,
    mat_idx: u32,
) -> (Vec<GpuVertex>, Vec<GpuTriangle>) {
    use std::f32::consts::PI;
    let stacks = stacks.max(2);
    let slices = slices.max(3);

    let mut vertices  = Vec::new();
    let mut triangles = Vec::new();

    // Build a (stacks+1) × (slices+1) grid of vertices.
    // Stack 0 is the south pole, stack `stacks` is the north pole.
    for i in 0..=stacks {
        let phi = PI * (i as f32 / stacks as f32); // 0 (south) → π (north)
        let sin_phi = phi.sin();
        let cos_phi = phi.cos();

        for j in 0..=slices {
            let theta = 2.0 * PI * (j as f32 / slices as f32);
            let sin_t = theta.sin();
            let cos_t = theta.cos();

            // Unit sphere normal (same as normalised position offset).
            let nx = sin_phi * cos_t;
            let ny = cos_phi;
            let nz = sin_phi * sin_t;

            vertices.push(GpuVertex {
                position: [
                    center[0] + radius * nx,
                    center[1] + radius * ny,
                    center[2] + radius * nz,
                    1.0,
                ],
                normal: [nx, ny, nz, 0.0],
                uv: [j as f32 / slices as f32, i as f32 / stacks as f32, 0.0, 0.0],
            });
        }
    }

    let ring = slices + 1; // vertices per latitude ring (including seam duplicate)

    // Emit two triangles per quad.
    for i in 0..stacks {
        for j in 0..slices {
            let v00 = i * ring + j;
            let v10 = (i + 1) * ring + j;
            let v01 = i * ring + (j + 1);
            let v11 = (i + 1) * ring + (j + 1);

            // Skip cap triangles that would degenerate at the poles.
            if i > 0 {
                triangles.push(GpuTriangle { v: [v00, v10, v01], mat_idx });
            }
            if i + 1 < stacks {
                triangles.push(GpuTriangle { v: [v01, v10, v11], mat_idx });
            }
        }
    }

    (vertices, triangles)
}

/// Build a torus mesh centred at `center` with major radius `R` and tube radius `r`.
///
/// Generates `major_seg` segments around the main ring and `minor_seg` around the tube.
/// The torus lies in the XZ plane by default.
///
/// Material index `mat_idx` is assigned to every triangle.
#[allow(dead_code)] // used in tests and available as public API for future scenes
pub fn build_torus_mesh(
    center:    [f32; 3],
    major_r:   f32,
    minor_r:   f32,
    major_seg: u32,
    minor_seg: u32,
    mat_idx:   u32,
) -> (Vec<GpuVertex>, Vec<GpuTriangle>) {
    use std::f32::consts::PI;
    let major_seg = major_seg.max(3);
    let minor_seg = minor_seg.max(3);

    let mut vertices  = Vec::new();
    let mut triangles = Vec::new();

    for i in 0..=major_seg {
        let phi = 2.0 * PI * (i as f32 / major_seg as f32);
        let cos_phi = phi.cos();
        let sin_phi = phi.sin();

        for j in 0..=minor_seg {
            let theta = 2.0 * PI * (j as f32 / minor_seg as f32);
            let cos_t = theta.cos();
            let sin_t = theta.sin();

            // Position on the torus surface.
            let px = (major_r + minor_r * cos_t) * cos_phi + center[0];
            let py = minor_r * sin_t + center[1];
            let pz = (major_r + minor_r * cos_t) * sin_phi + center[2];

            // Outward normal (points away from the tube centre-line).
            let nx = cos_t * cos_phi;
            let ny = sin_t;
            let nz = cos_t * sin_phi;

            vertices.push(GpuVertex {
                position: [px, py, pz, 1.0],
                normal:   [nx, ny, nz, 0.0],
                uv: [
                    i as f32 / major_seg as f32,
                    j as f32 / minor_seg as f32,
                    0.0,
                    0.0,
                ],
            });
        }
    }

    let ring = minor_seg + 1;

    for i in 0..major_seg {
        for j in 0..minor_seg {
            let v00 = i * ring + j;
            let v10 = (i + 1) * ring + j;
            let v01 = i * ring + (j + 1);
            let v11 = (i + 1) * ring + (j + 1);

            triangles.push(GpuTriangle { v: [v00, v10, v01], mat_idx });
            triangles.push(GpuTriangle { v: [v01, v10, v11], mat_idx });
        }
    }

    (vertices, triangles)
}

// ---------------------------------------------------------------------------
// Minimal OBJ loader
// ---------------------------------------------------------------------------

/// Parse a Wavefront OBJ file at `path` into GPU vertices and triangles.
///
/// Supported directives: `v`, `vn`, `vt`, `f`.
/// Faces may use `v`, `v/vt`, `v//vn`, or `v/vt/vn` formats; polygons with
/// more than three vertices are fan-triangulated.  Negative OBJ indices (relative
/// from the end of the current list) are handled correctly.
///
/// All triangles are assigned `mat_idx`.  If the file is missing or unparseable,
/// an `Err` string is returned; callers may fall back to the procedural mesh.
///
/// # Normals
/// When the OBJ supplies per-vertex normals they are used directly.  If a face
/// vertex has no normal reference the face's geometric normal is computed and
/// applied to all three of its vertices (flat shading for that triangle).
#[allow(dead_code)] // retained as public API; called when an OBJ path is provided
pub fn load_obj(path: &str, mat_idx: u32) -> Result<(Vec<GpuVertex>, Vec<GpuTriangle>), String> {
    let content =
        std::fs::read_to_string(path).map_err(|e| format!("cannot read '{path}': {e}"))?;

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals:   Vec<[f32; 3]> = Vec::new();
    let mut uvs:       Vec<[f32; 2]> = Vec::new();
    let mut vertices:  Vec<GpuVertex> = Vec::new();
    let mut triangles: Vec<GpuTriangle> = Vec::new();

    // Map (pos_idx, uv_idx, norm_idx) → GpuVertex index for deduplication.
    // Sentinel u32::MAX means "absent" for optional attributes.
    let mut vert_cache: std::collections::HashMap<(u32, u32, u32), u32> =
        std::collections::HashMap::new();

    /// Resolve an OBJ 1-based (or negative) index to a 0-based usize.
    fn resolve(raw: &str, len: usize) -> Option<u32> {
        let i: i64 = raw.parse().ok()?;
        let idx = if i < 0 { len as i64 + i } else { i - 1 };
        if idx >= 0 && (idx as usize) < len {
            Some(idx as u32)
        } else {
            None
        }
    }

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_ascii_whitespace();
        let tag = match parts.next() {
            Some(t) => t,
            None    => continue,
        };
        let rest: Vec<&str> = parts.collect();

        match tag {
            "v" if rest.len() >= 3 => {
                let x: f32 = rest[0].parse().unwrap_or(0.0);
                let y: f32 = rest[1].parse().unwrap_or(0.0);
                let z: f32 = rest[2].parse().unwrap_or(0.0);
                positions.push([x, y, z]);
            }
            "vn" if rest.len() >= 3 => {
                let x: f32 = rest[0].parse().unwrap_or(0.0);
                let y: f32 = rest[1].parse().unwrap_or(0.0);
                let z: f32 = rest[2].parse().unwrap_or(0.0);
                normals.push(normalize3([x, y, z]));
            }
            "vt" if rest.len() >= 2 => {
                let u: f32 = rest[0].parse().unwrap_or(0.0);
                let v: f32 = rest[1].parse().unwrap_or(0.0);
                uvs.push([u, v]);
            }
            "f" if rest.len() >= 3 => {
                // Gather unique vertex references for this face.
                let mut face_verts: Vec<u32> = Vec::with_capacity(rest.len());
                for token in &rest {
                    let parts: Vec<&str> = token.split('/').collect();
                    let pi = parts.first()
                        .and_then(|s| resolve(s, positions.len()))
                        .unwrap_or(0);
                    let ti = parts.get(1)
                        .and_then(|s| if s.is_empty() { None } else { resolve(s, uvs.len()) })
                        .unwrap_or(u32::MAX);
                    let ni = parts.get(2)
                        .and_then(|s| resolve(s, normals.len()))
                        .unwrap_or(u32::MAX);

                    // Vertices that lack an explicit normal reference (ni == u32::MAX)
                    // are NOT deduplicated across faces: different faces sharing the
                    // same position+UV but with no stored normal need independent
                    // GpuVertex entries so that each face's geometric normal can be
                    // patched without overwriting another face's result.
                    let vert_idx = if ni != u32::MAX {
                        let key = (pi, ti, ni);
                        *vert_cache.entry(key).or_insert_with(|| {
                            let pos    = positions.get(pi as usize).copied().unwrap_or([0.0; 3]);
                            let normal = normals.get(ni as usize).copied().unwrap_or([0.0, 1.0, 0.0]);
                            let uv     = if ti != u32::MAX { uvs.get(ti as usize).copied().unwrap_or([0.0; 2]) } else { [0.0; 2] };
                            let idx    = vertices.len() as u32;
                            vertices.push(GpuVertex {
                                position: [pos[0], pos[1], pos[2], 1.0],
                                normal:   [normal[0], normal[1], normal[2], 0.0],
                                uv:       [uv[0], uv[1], 0.0, 0.0],
                            });
                            idx
                        })
                    } else {
                        // No explicit normal: always create a fresh vertex entry.
                        // The geometric normal will be patched below after all three
                        // vertex indices for this triangle are known.
                        let pos = positions.get(pi as usize).copied().unwrap_or([0.0; 3]);
                        let uv  = if ti != u32::MAX { uvs.get(ti as usize).copied().unwrap_or([0.0; 2]) } else { [0.0; 2] };
                        let idx = vertices.len() as u32;
                        vertices.push(GpuVertex {
                            position: [pos[0], pos[1], pos[2], 1.0],
                            normal:   [0.0, 1.0, 0.0, 0.0], // placeholder; patched below
                            uv:       [uv[0], uv[1], 0.0, 0.0],
                        });
                        idx
                    };
                    face_verts.push(vert_idx);
                }

                // Fan triangulation.
                for i in 1..face_verts.len().saturating_sub(1) {
                    let v0 = face_verts[0];
                    let v1 = face_verts[i];
                    let v2 = face_verts[i + 1];

                    // If any vertex was created without an explicit normal (ni ==
                    // u32::MAX), patch it with this triangle's geometric normal.
                    let keys = [rest[0], rest[i], rest.get(i + 1).copied().unwrap_or("")];
                    for (vi, token) in [(v0, keys[0]), (v1, keys[1]), (v2, keys[2])] {
                        let has_norm = token.split('/').nth(2)
                            .map(|s| !s.is_empty() && s.parse::<i64>().is_ok())
                            .unwrap_or(false);
                        if !has_norm {
                            // Compute geometric normal and patch vertex in-place.
                            let p0 = vertices[v0 as usize].position;
                            let p1 = vertices[v1 as usize].position;
                            let p2 = vertices[v2 as usize].position;
                            let e1 = sub3(p1, p0);
                            let e2 = sub3(p2, p0);
                            let gn = normalize3(cross3(e1, e2));
                            vertices[vi as usize].normal =
                                [gn[0], gn[1], gn[2], 0.0];
                        }
                    }

                    triangles.push(GpuTriangle { v: [v0, v1, v2], mat_idx });
                }
            }
            _ => {}
        }
    }

    if vertices.is_empty() || triangles.is_empty() {
        return Err(format!("'{path}': no geometry found after parsing"));
    }
    Ok((vertices, triangles))
}

// ---------------------------------------------------------------------------
// Small float helpers (no dependency on glam in this module)
// Used by load_obj; allow(dead_code) suppresses warnings when load_obj
// is not called in the current build path.
// ---------------------------------------------------------------------------

#[allow(dead_code)]
#[inline]
fn sub3(a: [f32; 4], b: [f32; 4]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[allow(dead_code)]
#[inline]
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[allow(dead_code)]
#[inline]
fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-8 {
        return [0.0, 1.0, 0.0];
    }
    [v[0] / len, v[1] / len, v[2] / len]
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- size checks -------------------------------------------------------

    #[test]
    fn gpu_vertex_is_48_bytes() {
        assert_eq!(std::mem::size_of::<GpuVertex>(), 48);
    }

    #[test]
    fn gpu_triangle_is_16_bytes() {
        assert_eq!(std::mem::size_of::<GpuTriangle>(), 16);
    }

    // ---- uv-sphere sanity --------------------------------------------------

    #[test]
    fn uv_sphere_vertex_count() {
        let (verts, tris) = build_uv_sphere_mesh([0.0; 3], 1.0, 12, 24, 0);
        // (stacks+1) × (slices+1) vertices
        assert_eq!(verts.len(), 13 * 25);
        // 2 × (stacks-1) × slices triangles (poles capped off)
        assert_eq!(tris.len(), 2 * 11 * 24);
    }

    #[test]
    fn uv_sphere_normals_are_unit_length() {
        let (verts, _) = build_uv_sphere_mesh([0.0; 3], 1.0, 8, 16, 0);
        for v in &verts {
            let n = v.normal;
            let len_sq = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
            assert!((len_sq - 1.0).abs() < 1e-5, "normal not unit length: {len_sq}");
        }
    }

    #[test]
    fn uv_sphere_positions_on_surface() {
        let center = [1.0, 2.0, 3.0];
        let radius = 2.5_f32;
        let (verts, _) = build_uv_sphere_mesh(center, radius, 6, 12, 0);
        for v in &verts {
            let dx = v.position[0] - center[0];
            let dy = v.position[1] - center[1];
            let dz = v.position[2] - center[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            assert!((dist - radius).abs() < 1e-4, "vertex not on sphere surface: dist={dist}");
        }
    }

    // ---- torus sanity ------------------------------------------------------

    #[test]
    fn torus_vertex_count() {
        let (verts, tris) = build_torus_mesh([0.0; 3], 2.0, 0.5, 16, 8, 0);
        assert_eq!(verts.len(), (16 + 1) * (8 + 1));
        assert_eq!(tris.len(), 2 * 16 * 8);
    }

    // ---- BVH over mesh -----------------------------------------------------

    #[test]
    fn empty_mesh_bvh_is_empty() {
        let result = build_mesh_bvh(&[], &[]);
        assert!(result.nodes.is_empty());
        assert!(result.ordered_triangles.is_empty());
    }

    #[test]
    fn mesh_bvh_covers_all_triangles() {
        let (verts, tris) = build_uv_sphere_mesh([0.0; 3], 1.0, 8, 16, 0);
        let result = build_mesh_bvh(&verts, &tris);
        assert_eq!(result.ordered_triangles.len(), tris.len());
        assert!(!result.nodes.is_empty());
        // Node count bound: ≤ 2n − 1.
        assert!(result.nodes.len() <= 2 * tris.len() - 1);
    }

    #[test]
    fn mesh_bvh_leaf_ranges_in_bounds() {
        let (verts, tris) = build_uv_sphere_mesh([0.0; 3], 1.0, 6, 12, 0);
        let result = build_mesh_bvh(&verts, &tris);
        let n_tris = result.ordered_triangles.len();
        for (i, node) in result.nodes.iter().enumerate() {
            if node.prim_count > 0 {
                let end = (node.right_or_offset + node.prim_count) as usize;
                assert!(
                    end <= n_tris,
                    "leaf {i}: right_or_offset+prim_count={end} > ordered_triangles.len()={n_tris}"
                );
            }
        }
    }

    #[test]
    fn mesh_bvh_internal_children_in_bounds() {
        let (verts, tris) = build_torus_mesh([0.0; 3], 2.0, 0.5, 12, 8, 0);
        let result = build_mesh_bvh(&verts, &tris);
        let n = result.nodes.len();
        for (i, node) in result.nodes.iter().enumerate() {
            if node.prim_count == 0 {
                assert!(
                    (i + 1) < n,
                    "internal node {i}: left child {}", i + 1
                );
                assert!(
                    (node.right_or_offset as usize) < n,
                    "internal node {i}: right child {} out of bounds (n={n})",
                    node.right_or_offset
                );
            }
        }
    }

    #[test]
    fn mesh_bvh_traversal_matches_brute_force() {
        // Build a small UV sphere mesh and verify the mesh BVH gives the same
        // closest-hit triangle index as a brute-force O(N) scan.
        let (verts, tris) = build_uv_sphere_mesh([0.0; 3], 1.0, 6, 12, 0);
        let result = build_mesh_bvh(&verts, &tris);

        let rays: &[([f32; 3], [f32; 3])] = &[
            ([0.0, 5.0,  0.0], [0.0, -1.0,  0.0]),
            ([0.0, 0.0,  5.0], [0.0,  0.0, -1.0]),
            ([2.0, 2.0,  2.0], [-1.0, -1.0, -1.0]),
        ];

        for &(origin, dir) in rays {
            let len = (dir[0]*dir[0]+dir[1]*dir[1]+dir[2]*dir[2]).sqrt();
            let d = [dir[0]/len, dir[1]/len, dir[2]/len];

            let bf_t  = brute_force_tri_hit(&result.vertices, &result.ordered_triangles, origin, d, 1e-4, 1e9);
            let bvh_t = bvh_tri_traverse(&result.nodes, &result.vertices, &result.ordered_triangles, origin, d, 1e-4, 1e9);

            match (bf_t, bvh_t) {
                (None, None) => {}
                (Some(a), Some(b)) => assert!((a-b).abs() < 1e-3, "bf={a} bvh={b}"),
                (Some(a), None)    => panic!("brute-force hit t={a} but BVH missed, ray {origin:?} {d:?}"),
                (None, Some(b))    => panic!("BVH hit t={b} but brute-force missed, ray {origin:?} {d:?}"),
            }
        }
    }

    // ---- test helpers -------------------------------------------------------

    fn brute_force_tri_hit(
        verts: &[GpuVertex],
        tris:  &[GpuTriangle],
        o:     [f32; 3],
        d:     [f32; 3],
        t_min: f32,
        t_max: f32,
    ) -> Option<f32> {
        let mut best = t_max;
        let mut found = false;
        for tri in tris {
            if let Some(t) = moller_trumbore(verts, tri, o, d, t_min, best) {
                best = t;
                found = true;
            }
        }
        if found { Some(best) } else { None }
    }

    fn bvh_tri_traverse(
        nodes: &[crate::bvh::GpuBvhNode],
        verts: &[GpuVertex],
        tris:  &[GpuTriangle],
        o:     [f32; 3],
        d:     [f32; 3],
        t_min: f32,
        t_max: f32,
    ) -> Option<f32> {
        use crate::bvh::GpuBvhNode;
        if nodes.is_empty() { return None; }
        let mut best = t_max;
        let mut found = false;
        let mut stack = [0u32; 32];
        let mut sp: i32 = 0;
        stack[0] = 0;
        while sp >= 0 {
            let idx = stack[sp as usize] as usize;
            sp -= 1;
            let node: &GpuBvhNode = &nodes[idx];
            if !cpu_aabb_hit(o, d, node.aabb_min, node.aabb_max, t_min, best) {
                continue;
            }
            if node.prim_count > 0 {
                let start = node.right_or_offset as usize;
                let end   = start + node.prim_count as usize;
                for tri in &tris[start..end] {
                    if let Some(t) = moller_trumbore(verts, tri, o, d, t_min, best) {
                        best = t;
                        found = true;
                    }
                }
            } else if sp < 30 {
                sp += 1; stack[sp as usize] = node.right_or_offset;
                sp += 1; stack[sp as usize] = (idx + 1) as u32;
            }
        }
        if found { Some(best) } else { None }
    }

    fn cpu_aabb_hit(
        o:     [f32; 3],
        d:     [f32; 3],
        bb_min: [f32; 4],
        bb_max: [f32; 4],
        t_min: f32,
        t_max: f32,
    ) -> bool {
        let mut t0 = t_min;
        let mut t1 = t_max;
        for i in 0..3 {
            let inv_d = if d[i].abs() < 1e-7 {
                if d[i] >= 0.0 { 1e30_f32 } else { -1e30_f32 }
            } else {
                1.0 / d[i]
            };
            let ta = (bb_min[i] - o[i]) * inv_d;
            let tb = (bb_max[i] - o[i]) * inv_d;
            t0 = t0.max(ta.min(tb));
            t1 = t1.min(ta.max(tb));
            if t1 <= t0 { return false; }
        }
        true
    }

    /// CPU Möller-Trumbore for the test traversal.
    fn moller_trumbore(
        verts: &[GpuVertex],
        tri:   &GpuTriangle,
        o:     [f32; 3],
        d:     [f32; 3],
        t_min: f32,
        t_max: f32,
    ) -> Option<f32> {
        let p0 = &verts[tri.v[0] as usize].position;
        let p1 = &verts[tri.v[1] as usize].position;
        let p2 = &verts[tri.v[2] as usize].position;

        let e1 = [p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]];
        let e2 = [p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2]];

        let h = cross3(d, e2);
        let a = e1[0]*h[0]+e1[1]*h[1]+e1[2]*h[2]; // dot(e1, h)
        if a.abs() < 1e-8 { return None; }
        let f  = 1.0 / a;
        let s  = [o[0]-p0[0], o[1]-p0[1], o[2]-p0[2]];
        let u  = f * (s[0]*h[0]+s[1]*h[1]+s[2]*h[2]);
        if u < 0.0 || u > 1.0 { return None; }
        let q  = cross3(s, e1);
        let v  = f * (d[0]*q[0]+d[1]*q[1]+d[2]*q[2]);
        if v < 0.0 || u + v > 1.0 { return None; }
        let t  = f * (e2[0]*q[0]+e2[1]*q[1]+e2[2]*q[2]);
        if t > t_min && t < t_max { Some(t) } else { None }
    }
}
