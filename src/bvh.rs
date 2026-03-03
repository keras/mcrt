// bvh.rs — Phase 9: CPU SAH BVH builder and GPU node types
//
// Builds a Bounding Volume Hierarchy (BVH) from a sphere list on the CPU
// using the Surface Area Heuristic (SAH) with binned partitioning.  The
// output is a flat GpuBvhNode array in pre-order layout, ready for GPU
// traversal, plus a reordered GpuSphere list for contiguous leaf access.
//
// Layout contract (shared with path_trace.wgsl):
//   - Root node is always at index 0.
//   - Internal node (prim_count == 0):
//       left child  = node_index + 1   (immediately follows in pre-order)
//       right child = right_or_offset
//   - Leaf node (prim_count > 0):
//       primitives  = ordered_spheres[right_or_offset .. +prim_count]
//
// No wgpu types are present; this module is independently testable.

use bytemuck::Zeroable;

use crate::scene::GpuSphere;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum primitives in a leaf before a SAH split is *attempted*.
///
/// Keeping it small (≤ 4) gives tighter inner nodes and faster average traversal.
/// **Not a hard upper bound:** when all centroids coincide the SAH returns `None`
/// and the builder falls back to a leaf regardless of count, so degenerate scenes
/// may produce leaves larger than this value.
pub(crate) const BVH_LEAF_MAX: usize = 4;

/// Number of SAH candidate buckets per axis.
///
/// 8 bins strike a good balance: near-optimal split quality without noticeably
/// increasing build time.
pub(crate) const SAH_BINS: usize = 8;

// ---------------------------------------------------------------------------
// GPU types (shared with WGSL, must obey WGSL alignment rules)
// ---------------------------------------------------------------------------

/// A single node in the flattened BVH, uploaded to the GPU as a storage buffer.
///
/// **Size:** 48 bytes (3 × 16-byte chunks — naturally vec4-aligned in WGSL).
///
/// **Internal node** (`prim_count == 0`):
/// - Left child  is at `(current_index + 1)` in the flat array (pre-order).
/// - Right child is at `right_or_offset`.
///
/// **Leaf node** (`prim_count > 0`):
/// - Sphere range is `spheres[right_or_offset .. right_or_offset + prim_count]`.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuBvhNode {
    /// AABB minimum corner.  `.w` is unused padding.
    pub aabb_min: [f32; 4],
    /// AABB maximum corner.  `.w` is unused padding.
    pub aabb_max: [f32; 4],
    /// Internal: right-child index in the flat node array.
    /// Leaf: start index into the ordered-sphere list.
    pub right_or_offset: u32,
    /// `0` = internal node; `> 0` = leaf (number of primitives in this leaf).
    pub prim_count: u32,
    /// Explicit padding to reach 48 bytes total.  Not part of the public API.
    pub(crate) _pad: [u32; 2],
}

/// Output of a successful BVH build.
pub struct BvhBuildResult {
    /// Spheres reordered so that each leaf's primitives form a contiguous slice.
    pub ordered_spheres: Vec<GpuSphere>,
    /// Flat BVH node array in pre-order (root = index 0).
    pub nodes: Vec<GpuBvhNode>,
}

// ---------------------------------------------------------------------------
// CPU-only AABB helper
// ---------------------------------------------------------------------------

/// Axis-aligned bounding box used during BVH construction on the CPU.
///
/// Exposed as `pub(crate)` so [`crate::mesh`] can reuse it for triangle AABBs
/// without duplicating the helper methods.
#[derive(Clone, Copy)]
pub(crate) struct Aabb {
    pub(crate) min: [f32; 3],
    pub(crate) max: [f32; 3],
}

impl Aabb {
    /// An AABB that is the identity element for union (grows to any finite box).
    ///
    /// Uses `±INFINITY` rather than `f32::MIN/MAX` so that NaN inputs to [`union`]
    /// propagate visibly instead of silently producing a finite-looking result.
    #[inline]
    pub(crate) fn empty() -> Self {
        Self { min: [f32::INFINITY; 3], max: [f32::NEG_INFINITY; 3] }
    }

    /// A degenerate AABB that contains exactly one point (zero volume).
    #[inline]
    pub(crate) fn point(p: [f32; 3]) -> Self {
        Self { min: p, max: p }
    }

    /// Tight AABB enclosing three `[f32; 4]` points (`.xyz` used, `.w` ignored).
    ///
    /// Intended for computing triangle AABBs from the GPU vertex buffer where
    /// each position is stored as a `[f32; 4]`.
    #[inline]
    pub(crate) fn from_points_3(pts: &[[f32; 4]; 3]) -> Self {
        let mut aabb = Aabb::empty();
        for p in pts {
            aabb.min[0] = aabb.min[0].min(p[0]);
            aabb.min[1] = aabb.min[1].min(p[1]);
            aabb.min[2] = aabb.min[2].min(p[2]);
            aabb.max[0] = aabb.max[0].max(p[0]);
            aabb.max[1] = aabb.max[1].max(p[1]);
            aabb.max[2] = aabb.max[2].max(p[2]);
        }
        aabb
    }

    /// Tight AABB around a sphere with the given centre and radius.
    #[inline]
    pub(crate) fn from_sphere(center: [f32; 3], radius: f32) -> Self {
        debug_assert!(radius.is_finite() && radius >= 0.0, "sphere radius must be finite and non-negative, got {radius}");
        Self {
            min: [center[0] - radius, center[1] - radius, center[2] - radius],
            max: [center[0] + radius, center[1] + radius, center[2] + radius],
        }
    }

    /// Smallest AABB that contains both `self` and `other`.
    #[inline]
    pub(crate) fn union(self, other: &Aabb) -> Aabb {
        Aabb {
            min: [
                self.min[0].min(other.min[0]),
                self.min[1].min(other.min[1]),
                self.min[2].min(other.min[2]),
            ],
            max: [
                self.max[0].max(other.max[0]),
                self.max[1].max(other.max[1]),
                self.max[2].max(other.max[2]),
            ],
        }
    }

    /// Half-open surface area of the AABB (full SA = 2 × this).
    ///
    /// We always compare SA values in the same SAH expression so the factor of 2
    /// cancels; using half-area keeps the numbers smaller.
    #[inline]
    pub(crate) fn half_area(&self) -> f32 {
        let d = [
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        ];
        d[0] * d[1] + d[1] * d[2] + d[0] * d[2]
    }

    /// Geometric centroid of the AABB.
    #[inline]
    pub(crate) fn centroid(&self) -> [f32; 3] {
        [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        ]
    }

    /// Convert to the GPU-friendly `[f32; 4]` pair used in `GpuBvhNode`.
    #[inline]
    pub(crate) fn to_gpu_pair(&self) -> ([f32; 4], [f32; 4]) {
        (
            [self.min[0], self.min[1], self.min[2], 0.0],
            [self.max[0], self.max[1], self.max[2], 0.0],
        )
    }
}

// ---------------------------------------------------------------------------
// Per-primitive metadata used only during the build
// ---------------------------------------------------------------------------

struct PrimInfo {
    bounds:     Aabb,
    centroid:   [f32; 3],
    sphere_idx: usize, // index into the original (unordered) sphere slice
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Build a flat BVH over `spheres` using SAH with binned partitioning.
///
/// The returned `BvhBuildResult` contains:
/// - `ordered_spheres`: `spheres` reordered so that leaf primitive ranges are
///   contiguous.  The original `spheres` slice is **not** modified.
/// - `nodes`: flat pre-order node array; root is at index 0.
///
/// An empty input produces an empty result (root = no nodes); callers must
/// guard against a zero-length node buffer before dispatching the GPU shader.
pub fn build_bvh(spheres: &[GpuSphere]) -> BvhBuildResult {
    if spheres.is_empty() {
        return BvhBuildResult {
            ordered_spheres: Vec::new(),
            nodes:           Vec::new(),
        };
    }

    let mut prim_infos: Vec<PrimInfo> = spheres
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let center = [s.center_r[0], s.center_r[1], s.center_r[2]];
            let radius = s.center_r[3];
            let bounds = Aabb::from_sphere(center, radius);
            PrimInfo { centroid: bounds.centroid(), bounds, sphere_idx: i }
        })
        .collect();

    let mut nodes           = Vec::with_capacity(2 * spheres.len());
    let mut ordered_spheres = Vec::with_capacity(spheres.len());

    build_recursive(&mut prim_infos, &mut nodes, &mut ordered_spheres, spheres);

    BvhBuildResult { ordered_spheres, nodes }
}

// ---------------------------------------------------------------------------
// Recursive build — pre-order node allocation
// ---------------------------------------------------------------------------

/// Recursively partition `prims` into a BVH sub-tree.
///
/// Reserves `nodes[node_idx]` as a placeholder before recursing, so both
/// children can write their indices back into `nodes[node_idx]` after the
/// recursion returns.  This guarantees:
///   - `node_idx + 1` is always the left child's index (pre-order invariant).
///   - `right_or_offset` is set to the right child's index after the left
///     sub-tree has been fully inserted.
fn build_recursive(
    prims:      &mut [PrimInfo],
    nodes:      &mut Vec<GpuBvhNode>,
    ordered:    &mut Vec<GpuSphere>,
    all:        &[GpuSphere],
) -> usize {
    let node_idx = nodes.len();
    nodes.push(GpuBvhNode::zeroed()); // placeholder; overwritten below

    // Bounding box of all primitives in this sub-tree.
    let bounds = prims.iter().fold(Aabb::empty(), |acc, p| acc.union(&p.bounds));

    // ----- Leaf creation (forced for small groups) -------------------------
    if prims.len() <= BVH_LEAF_MAX {
        nodes[node_idx] = make_leaf(bounds, ordered, prims, all);
        return node_idx;
    }

    // ----- Try SAH split ---------------------------------------------------
    let centroid_bounds =
        prims.iter().fold(Aabb::empty(), |acc, p| {
            acc.union(&Aabb { min: p.centroid, max: p.centroid })
        });

    match find_best_split(&prims, &centroid_bounds, &bounds) {
        Some((axis, split_idx)) => {
            // Sort prims by centroid along the chosen axis, then split.
            prims.sort_unstable_by(|a, b| {
                a.centroid[axis]
                    .partial_cmp(&b.centroid[axis])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Left child is always at node_idx + 1 (pre-order invariant).
            let left_child = build_recursive(&mut prims[..split_idx], nodes, ordered, all);
            debug_assert_eq!(left_child, node_idx + 1, "pre-order invariant: left child must immediately follow parent");
            // Right child follows the entire left sub-tree.
            let right_child = nodes.len();
            build_recursive(&mut prims[split_idx..], nodes, ordered, all);

            let (aabb_min, aabb_max) = bounds.to_gpu_pair();
            nodes[node_idx] = GpuBvhNode {
                aabb_min,
                aabb_max,
                right_or_offset: right_child as u32,
                prim_count:      0, // internal node
                _pad:            [0; 2],
            };
        }
        None => {
            // No useful split found (degenerate centroid distribution) — leaf.
            nodes[node_idx] = make_leaf(bounds, ordered, prims, all);
        }
    }

    node_idx
}

/// Construct a `GpuBvhNode` leaf and append this range's spheres to `ordered`.
fn make_leaf(
    bounds:  Aabb,
    ordered: &mut Vec<GpuSphere>,
    prims:   &[PrimInfo],
    all:     &[GpuSphere],
) -> GpuBvhNode {
    let prim_start = ordered.len() as u32;
    for p in prims {
        ordered.push(all[p.sphere_idx]);
    }
    let (aabb_min, aabb_max) = bounds.to_gpu_pair();
    GpuBvhNode {
        aabb_min,
        aabb_max,
        right_or_offset: prim_start,
        prim_count:      prims.len() as u32,
        _pad:            [0; 2],
    }
}

// ---------------------------------------------------------------------------
// SAH binned split selection
// ---------------------------------------------------------------------------

/// Return `(axis, split_index)` for the best SAH split found via binned SAH,
/// or `None` when all axes are degenerate (all centroids coincide).
///
/// The returned `split_index` is the count of elements that belong to the left
/// child after sorting `prims` by `centroid[axis]`.  It is guaranteed to be in
/// `[1, prims.len() - 1]`.
///
/// **Cost model**: $C_{split} = \frac{SA_L}{SA_P} \cdot N_L + \frac{SA_R}{SA_P} \cdot N_R$
/// with traversal cost $C_t = 0$ and intersection cost $C_i = 1$.  This
/// approximation is suitable for small-to-medium scenes (≤ a few thousand
/// primitives).  For production use, $C_t ≈ 1, C_i ≈ 4$ gives better trees
/// by making splits cheaper relative to intersections.
fn find_best_split(
    prims:           &[PrimInfo],
    centroid_bounds: &Aabb,
    node_bounds:     &Aabb,
) -> Option<(usize, usize)> {
    let parent_half_area = node_bounds.half_area().max(1e-8);
    let leaf_cost        = prims.len() as f32;

    let mut best_cost  = leaf_cost; // only accept strictly better splits
    let mut best_axis  = 0usize;
    let mut best_split = 0usize;
    let mut found      = false;

    for axis in 0..3usize {
        let axis_len = centroid_bounds.max[axis] - centroid_bounds.min[axis];
        if axis_len < 1e-8 {
            continue; // degenerate axis — all centroids coincide along this axis
        }

        // ---- Bin primitives ---------------------------------------------------
        let mut bin_bounds = [Aabb::empty(); SAH_BINS];
        let mut bin_counts = [0u32; SAH_BINS];

        for p in prims {
            let t = (p.centroid[axis] - centroid_bounds.min[axis]) / axis_len;
            // Clamp to [0, SAH_BINS - 1] so the boundary prim lands in the last bin.
            let b = ((t * SAH_BINS as f32) as usize).min(SAH_BINS - 1);
            bin_bounds[b] = bin_bounds[b].union(&p.bounds);
            bin_counts[b] += 1;
        }

        // ---- Left-prefix sweep -----------------------------------------------
        let mut left_bounds = [Aabb::empty(); SAH_BINS];
        let mut left_counts = [0u32; SAH_BINS];
        left_bounds[0] = bin_bounds[0];
        left_counts[0] = bin_counts[0];
        for i in 1..SAH_BINS {
            left_bounds[i] = left_bounds[i - 1].union(&bin_bounds[i]);
            left_counts[i] = left_counts[i - 1] + bin_counts[i];
        }

        // ---- Right-suffix sweep + SAH evaluation ----------------------------
        let mut right_bounds = Aabb::empty();
        let mut right_count  = 0u32;

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
                best_cost  = cost;
                best_axis  = axis;
                // `left_counts[split]` is the number of prims in left bins 0..=split.
                // After sorting by this axis, the split is at that index.
                best_split = l_count as usize;
                found      = true;
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
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::build_materials;

    /// Helper: create a sphere at (cx, 0, cz) with given radius and material.
    fn sphere(cx: f32, cz: f32, r: f32, mat: u32) -> GpuSphere {
        GpuSphere { center_r: [cx, 0.0, cz, r], mat_and_pad: [mat, 0, 0, 0] }
    }

    #[test]
    fn empty_scene_produces_empty_result() {
        let result = build_bvh(&[]);
        assert!(result.nodes.is_empty());
        assert!(result.ordered_spheres.is_empty());
    }

    #[test]
    fn single_sphere_produces_one_leaf() {
        let s = sphere(0.0, 0.0, 0.5, 0);
        let result = build_bvh(&[s]);
        assert_eq!(result.nodes.len(), 1, "single sphere should produce exactly 1 node");
        assert_eq!(result.nodes[0].prim_count, 1);
        assert_eq!(result.nodes[0].right_or_offset, 0);
        assert_eq!(result.ordered_spheres.len(), 1);
    }

    #[test]
    fn four_spheres_produce_one_leaf() {
        // 4 primitives == BVH_LEAF_MAX → leaf created unconditionally (SAH not consulted).
        let spheres: Vec<GpuSphere> = (0..4)
            .map(|i| sphere(i as f32 * 2.0, 0.0, 0.5, 0))
            .collect();
        let result = build_bvh(&spheres);
        assert_eq!(result.nodes.len(), 1, "exactly 4 prims ≤ BVH_LEAF_MAX should collapse to 1 leaf");
        assert_eq!(result.nodes[0].prim_count, 4);
        assert_eq!(result.ordered_spheres.len(), spheres.len());
    }

    #[test]
    fn large_scene_node_count_bounded() {
        let spheres: Vec<GpuSphere> = (0..64)
            .map(|i| sphere((i % 8) as f32, (i / 8) as f32, 0.3, 0))
            .collect();
        let result = build_bvh(&spheres);
        // Upper bound: 2N-1 internal nodes.
        assert!(result.nodes.len() <= 2 * spheres.len() - 1);
        assert_eq!(result.ordered_spheres.len(), spheres.len());
    }

    #[test]
    fn ordered_spheres_contain_all_originals() {
        let spheres: Vec<GpuSphere> = (0..20)
            .map(|i| sphere(i as f32, 0.0, 0.4, i % 4))
            .collect();
        let result = build_bvh(&spheres);
        assert_eq!(result.ordered_spheres.len(), spheres.len());
        // Every original sphere must appear exactly once in ordered_spheres.
        for s in &spheres {
            let found = result
                .ordered_spheres
                .iter()
                .filter(|o| o.center_r == s.center_r && o.mat_and_pad == s.mat_and_pad)
                .count();
            assert_eq!(found, 1, "sphere {:?} should appear exactly once", s.center_r);
        }
    }

    #[test]
    fn root_aabb_contains_all_spheres() {
        let spheres: Vec<GpuSphere> = (0..16)
            .map(|i| sphere(i as f32, 0.0, 0.5, 0))
            .collect();
        let result = build_bvh(&spheres);
        let root = &result.nodes[0];
        // Every sphere centre must lie inside the root AABB (expanded by radius).
        for s in &spheres {
            let cx = s.center_r[0];
            let r  = s.center_r[3];
            assert!(
                cx - r >= root.aabb_min[0] - 1e-4 && cx + r <= root.aabb_max[0] + 1e-4,
                "sphere at x={cx} r={r} not enclosed by root AABB [{}, {}]",
                root.aabb_min[0],
                root.aabb_max[0]
            );
        }
    }

    #[test]
    fn leaf_nodes_have_positive_prim_count() {
        let spheres: Vec<GpuSphere> = (0..32)
            .map(|i| sphere(i as f32, 0.0, 0.4, 0))
            .collect();
        let result = build_bvh(&spheres);
        for (i, node) in result.nodes.iter().enumerate() {
            if node.prim_count > 0 {
                let end = (node.right_or_offset + node.prim_count) as usize;
                assert!(
                    end <= result.ordered_spheres.len(),
                    "leaf {i} right_or_offset+prim_count={end} exceeds ordered_spheres.len()={}",
                    result.ordered_spheres.len()
                );
            }
        }
    }

    #[test]
    fn internal_nodes_have_valid_right_child() {
        let spheres: Vec<GpuSphere> = (0..32)
            .map(|i| sphere(i as f32 * 0.5, 0.0, 0.2, 0))
            .collect();
        let result = build_bvh(&spheres);
        let n = result.nodes.len();
        for (i, node) in result.nodes.iter().enumerate() {
            if node.prim_count == 0 {
                // left child = i + 1, right child = right_or_offset
                assert!(
                    (node.right_or_offset as usize) < n,
                    "internal node {i} right_or_offset={} out of bounds (node_count={n})",
                    node.right_or_offset
                );
                assert!(
                    (i + 1) < n,
                    "internal node {i} would place left child at {}, but only {n} nodes",
                    i + 1
                );
            }
        }
    }

    #[test]
    fn gpu_bvh_node_is_48_bytes() {
        assert_eq!(std::mem::size_of::<GpuBvhNode>(), 48);
    }

    #[test]
    fn coincident_spheres_do_not_panic() {
        // All centroids identical → SAH finds no useful split; must fall back to leaf.
        let spheres: Vec<GpuSphere> =
            (0..8).map(|i| sphere(0.0, 0.0, 0.5, i % 4)).collect();
        let result = build_bvh(&spheres);
        assert!(!result.nodes.is_empty());
        assert_eq!(result.ordered_spheres.len(), spheres.len());
    }

    /// Verify there are no broken references by doing a mock traversal
    /// (pushes all internal nodes' children and checks bounds).
    #[test]
    fn no_orphan_nodes_in_tree() {
        let spheres: Vec<GpuSphere> = (0..48)
            .map(|i| sphere((i % 8) as f32, (i / 8) as f32, 0.3, 0))
            .collect();
        let result = build_bvh(&spheres);
        let n = result.nodes.len();

        let mut visited = vec![false; n];
        let mut stack   = vec![0usize]; // start at root
        while let Some(idx) = stack.pop() {
            if visited[idx] { continue; }
            visited[idx] = true;
            let node = &result.nodes[idx];
            if node.prim_count == 0 {
                // Internal: push left (idx+1) and right (right_or_offset).
                let left  = idx + 1;
                let right = node.right_or_offset as usize;
                assert!(left < n, "left child {left} out of bounds");
                assert!(right < n, "right child {right} out of bounds");
                stack.push(left);
                stack.push(right);
            }
        }
        // Every node reachable from the root must have been visited.
        for (i, v) in visited.iter().enumerate() {
            assert!(v, "node {i} unreachable from root — orphan node detected");
        }
    }

    #[test]
    fn bvh_traversal_matches_brute_force() {
        // Build BVH; then for several test rays verify the closest-hit sphere index
        // found by BVH traversal equals the one found by a linear O(N) search.
        // This is the critical end-to-end correctness test for the whole module.
        let n = 32usize;
        let spheres: Vec<GpuSphere> = (0..n)
            .map(|i| sphere(i as f32 * 0.6 - 9.0, 0.0, 0.25, 0))
            .collect();
        let result = build_bvh(&spheres);

        // A handful of rays from different angles.
        let rays: &[([f32; 3], [f32; 3])] = &[
            ([0.0, 5.0, 0.0], [0.0, -1.0, 0.0]),  // straight down
            ([-10.0, 0.0, 5.0], [1.0, 0.0, 0.0]), // along X
            ([0.0, 0.0, 5.0], [0.0, 0.0, -1.0]),  // along -Z
            ([5.0, 2.0, 0.0], [-0.5, -0.5, 0.0]), // diagonal
        ];

        for &(origin, dir) in rays {
            let len = (dir[0]*dir[0]+dir[1]*dir[1]+dir[2]*dir[2]).sqrt();
            let dir_n = [dir[0]/len, dir[1]/len, dir[2]/len];

            // Brute-force: find closest intersection across ordered_spheres.
            let bf_hit = brute_force_hit(&result.ordered_spheres, origin, dir_n, 1e-4, 1e9);
            // BVH traversal: CPU mirror of the WGSL iterative stack algorithm.
            let bvh_hit = cpu_bvh_traverse(&result.nodes, &result.ordered_spheres, origin, dir_n, 1e-4, 1e9);

            match (bf_hit, bvh_hit) {
                (None, None) => {}
                (Some(bf_t), Some(bvh_t)) => {
                    assert!(
                        (bf_t - bvh_t).abs() < 1e-4,
                        "ray from {:?} dir {:?}: brute-force t={bf_t:.4} vs BVH t={bvh_t:.4}",
                        origin, dir
                    );
                }
                (Some(bf_t), None) => panic!(
                    "ray from {:?}: brute-force hit t={bf_t:.4} but BVH missed",
                    origin
                ),
                (None, Some(bvh_t)) => panic!(
                    "ray from {:?}: BVH hit t={bvh_t:.4} but brute-force missed",
                    origin
                ),
            }
        }
    }

    /// CPU mirror of the WGSL iterative BVH traversal (pre-order left-child=i+1 layout).
    fn cpu_bvh_traverse(
        nodes:   &[GpuBvhNode],
        spheres: &[GpuSphere],
        origin:  [f32; 3],
        dir:     [f32; 3],
        t_min:   f32,
        t_max:   f32,
    ) -> Option<f32> {
        if nodes.is_empty() { return None; }
        let mut best_t = t_max;
        let mut found  = false;
        let mut stack  = [0u32; 32];
        let mut sp: i32 = 0;
        stack[0] = 0;
        while sp >= 0 {
            let idx  = stack[sp as usize] as usize;
            sp -= 1;
            let node = &nodes[idx];
            if !cpu_aabb_hit(origin, dir, node.aabb_min, node.aabb_max, t_min, best_t) {
                continue;
            }
            if node.prim_count > 0 {
                let start = node.right_or_offset as usize;
                let end   = start + node.prim_count as usize;
                for s in &spheres[start..end] {
                    if let Some(t) = cpu_sphere_hit(origin, dir, s, t_min, best_t) {
                        best_t = t;
                        found  = true;
                    }
                }
            } else if sp < 30 {
                sp += 1; stack[sp as usize] = node.right_or_offset;
                sp += 1; stack[sp as usize] = (idx + 1) as u32;
            }
        }
        if found { Some(best_t) } else { None }
    }

    fn brute_force_hit(spheres: &[GpuSphere], origin: [f32; 3], dir: [f32; 3], t_min: f32, t_max: f32) -> Option<f32> {
        let mut best = t_max;
        let mut found = false;
        for s in spheres {
            if let Some(t) = cpu_sphere_hit(origin, dir, s, t_min, best) {
                best = t; found = true;
            }
        }
        if found { Some(best) } else { None }
    }

    fn cpu_aabb_hit(origin: [f32; 3], dir: [f32; 3], bb_min: [f32; 4], bb_max: [f32; 4], t_min: f32, t_max: f32) -> bool {
        let mut t0 = t_min;
        let mut t1 = t_max;
        for i in 0..3 {
            let inv_d = if dir[i].abs() < 1e-7 { if dir[i] >= 0.0 { 1e30_f32 } else { -1e30_f32 } } else { 1.0 / dir[i] };
            let ta = (bb_min[i] - origin[i]) * inv_d;
            let tb = (bb_max[i] - origin[i]) * inv_d;
            t0 = t0.max(ta.min(tb));
            t1 = t1.min(ta.max(tb));
            if t1 <= t0 { return false; }
        }
        true
    }

    fn cpu_sphere_hit(origin: [f32; 3], dir: [f32; 3], s: &GpuSphere, t_min: f32, t_max: f32) -> Option<f32> {
        let cx = s.center_r[0]; let cy = s.center_r[1]; let cz = s.center_r[2]; let r = s.center_r[3];
        let ocx = origin[0]-cx; let ocy = origin[1]-cy; let ocz = origin[2]-cz;
        let a = dir[0]*dir[0]+dir[1]*dir[1]+dir[2]*dir[2];
        let h = ocx*dir[0]+ocy*dir[1]+ocz*dir[2];
        let c = ocx*ocx+ocy*ocy+ocz*ocz - r*r;
        let disc = h*h - a*c;
        if disc < 0.0 { return None; }
        let sqrtd = disc.sqrt();
        let t = (-h - sqrtd) / a;
        if t > t_min && t < t_max { return Some(t); }
        let t = (-h + sqrtd) / a;
        if t > t_min && t < t_max { return Some(t); }
        None
    }

    #[test]
    fn bvh_preserves_sphere_count_with_varied_materials() {
        let mats = build_materials();
        let mat_count = mats.mat_count as u32;
        let spheres: Vec<GpuSphere> = (0..16)
            .map(|i| sphere(i as f32, 0.0, 0.4, i % mat_count))
            .collect();
        let result = build_bvh(&spheres);
        assert_eq!(result.ordered_spheres.len(), spheres.len());
    }
}
