// camera.rs — Phase 3+: camera uniform and projection helpers
//
// Owns the CPU-side representation of the thin-lens camera and the pure
// function that converts (orbit params → CameraUniform) for GPU upload.
// No wgpu types live here, making the geometry logic unit-testable without
// a GPU device.

use glam::Vec3;

// ---------------------------------------------------------------------------
// Camera constants
// ---------------------------------------------------------------------------

/// Default vertical field of view in degrees.
pub const DEFAULT_VFOV: f32 = 60.0;

/// Initial eye position in world space (from Phase 6 static camera).
pub const INIT_LOOK_FROM: Vec3 = Vec3::new(0.0, 1.0, 3.5);

/// Initial orbit target in world space.
pub const INIT_LOOK_AT: Vec3 = Vec3::ZERO;

// ---------------------------------------------------------------------------
// CameraUniform — GPU struct
// ---------------------------------------------------------------------------

/// GPU-side camera layout.  All fields use vec4 packing (16-byte aligned),
/// matching the WGSL `Camera` struct exactly.  Total size = 112 bytes.
///
/// Phase 8 adds `defocus_u` / `defocus_v` for the thin-lens depth-of-field
/// model:
///   defocus_u = cam_right × lens_radius
///   defocus_v = cam_up    × lens_radius
/// Both are zero for a pinhole camera (aperture = 0).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    /// Eye position in world space.  `.w` unused.
    pub origin: [f32; 4],
    /// Lower-left corner of the virtual screen on the focus plane.  `.w` unused.
    pub lower_left: [f32; 4],
    /// Full horizontal extent of the screen on the focus plane.  `.w` unused.
    pub horizontal: [f32; 4],
    /// Full vertical extent of the screen on the focus plane.  `.w` unused.
    pub vertical: [f32; 4],
    /// Camera right-basis scaled by lens radius.  Zero for pinhole camera.
    pub defocus_u: [f32; 4],
    /// Camera up-basis scaled by lens radius.   Zero for pinhole camera.
    pub defocus_v: [f32; 4],
    /// Monotonically increasing frame index.  `u32` to stay exact beyond 2²⁴.
    pub frame_count: u32,
    pub _pad: [u32; 3],
}

// ---------------------------------------------------------------------------
// compute_camera — pure projection function
// ---------------------------------------------------------------------------

/// Compute the camera uniform for a thin-lens camera with optional depth of field.
///
/// - `width` / `height`:  viewport dimensions (aspect ratio source)
/// - `look_from`:         eye position in world space
/// - `look_at`:           orbit target / aim point
/// - `vfov_deg`:          vertical field of view in degrees
/// - `aperture`:          lens diameter in world units (0 = pinhole, no blur)
/// - `focus_dist`:        distance from eye to the plane of sharp focus
///
/// The virtual screen is placed on the focus plane so all rays passing through
/// the same screen point converge there regardless of aperture.
///
/// # Gimbal-lock safety
/// When `look_from ≈ look_at` along the world Y axis (straight up/down), the
/// world-up vector `(0,1,0)` would be parallel to the view direction, making
/// the cross product zero. This function detects that case and falls back to
/// a Z-up vector so the camera remains numerically stable (no NaN propagation).
pub fn compute_camera(
    width: u32,
    height: u32,
    look_from: Vec3,
    look_at: Vec3,
    vfov_deg: f32,
    aperture: f32,
    focus_dist: f32,
) -> CameraUniform {
    let aspect = width as f32 / height as f32;
    let vup = Vec3::new(0.0, 1.0, 0.0);
    let vfov_rad = vfov_deg.to_radians();
    let h = (vfov_rad * 0.5).tan();
    let viewport_height = 2.0 * h;
    let viewport_width = aspect * viewport_height;

    // Orthonormal camera basis (right-handed, Z points toward the viewer).
    let w = (look_from - look_at).normalize(); // backward

    // Gimbal-lock guard: if vup is parallel to w, fall back to Z-up so the
    // cross product remains non-zero in release builds (no silent NaN).
    let vup_safe = if vup.cross(w).length_squared() > 1e-10 {
        vup
    } else {
        Vec3::new(0.0, 0.0, 1.0)
    };

    let cam_u = vup_safe.cross(w).normalize(); // right
    let cam_v = w.cross(cam_u); // up (unit: w⊥cam_u, both unit)

    // Scale screen extents by focus_dist so the virtual screen sits exactly
    // on the plane of sharp focus.
    let horizontal = focus_dist * viewport_width * cam_u;
    let vertical = focus_dist * viewport_height * cam_v;
    let lower_left = look_from - horizontal * 0.5 - vertical * 0.5 - focus_dist * w;

    // Lens disk basis vectors (zero for pinhole where aperture = 0).
    let lens_radius = aperture * 0.5;
    let defocus_u = cam_u * lens_radius;
    let defocus_v = cam_v * lens_radius;

    CameraUniform {
        origin: look_from.extend(0.0).to_array(),
        lower_left: lower_left.extend(0.0).to_array(),
        horizontal: horizontal.extend(0.0).to_array(),
        vertical: vertical.extend(0.0).to_array(),
        defocus_u: defocus_u.extend(0.0).to_array(),
        defocus_v: defocus_v.extend(0.0).to_array(),
        frame_count: 0, // caller sets this before uploading
        _pad: [0; 3],
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Pinhole camera should produce zero-length defocus basis vectors.
    #[test]
    fn pinhole_has_no_defocus() {
        let cam = compute_camera(
            800,
            600,
            Vec3::new(0.0, 0.0, 3.0),
            Vec3::ZERO,
            60.0,
            0.0, // aperture = 0 → pinhole
            3.0,
        );
        assert_eq!(cam.defocus_u, [0.0, 0.0, 0.0, 0.0]);
        assert_eq!(cam.defocus_v, [0.0, 0.0, 0.0, 0.0]);
    }

    /// Origin should equal look_from.
    #[test]
    fn origin_equals_look_from() {
        let look_from = Vec3::new(1.0, 2.0, 3.0);
        let cam = compute_camera(800, 600, look_from, Vec3::ZERO, 60.0, 0.0, 3.0);
        assert!((cam.origin[0] - 1.0).abs() < 1e-5);
        assert!((cam.origin[1] - 2.0).abs() < 1e-5);
        assert!((cam.origin[2] - 3.0).abs() < 1e-5);
    }

    /// frame_count is initialised to 0 by compute_camera (caller fills it in).
    #[test]
    fn frame_count_initialised_zero() {
        let cam = compute_camera(
            800,
            600,
            Vec3::new(0.0, 0.0, 3.0),
            Vec3::ZERO,
            60.0,
            0.0,
            3.0,
        );
        assert_eq!(cam.frame_count, 0);
    }

    /// Gimbal-lock: camera looking straight up must not produce NaN in any field.
    #[test]
    fn gimbal_lock_no_nan() {
        // eye directly above target → view direction is (0,1,0) = world-up
        let cam = compute_camera(
            800,
            600,
            Vec3::new(0.0, 5.0, 0.0),
            Vec3::ZERO,
            60.0,
            0.0,
            5.0,
        );
        let buf = [cam];
        let all_fields: &[f32] = bytemuck::cast_slice(&buf);
        for &v in all_fields {
            assert!(
                !v.is_nan(),
                "NaN detected in camera uniform during gimbal-lock test"
            );
        }
    }

    /// Thin-lens aperture should produce non-zero defocus vectors.
    #[test]
    fn thin_lens_has_defocus() {
        let cam = compute_camera(
            800,
            600,
            Vec3::new(0.0, 0.0, 3.0),
            Vec3::ZERO,
            60.0,
            0.5, // aperture > 0
            3.0,
        );
        let du_len = cam.defocus_u[0]
            .hypot(cam.defocus_u[1])
            .hypot(cam.defocus_u[2]);
        let dv_len = cam.defocus_v[0]
            .hypot(cam.defocus_v[1])
            .hypot(cam.defocus_v[2]);
        assert!(du_len > 1e-5, "defocus_u should be non-zero");
        assert!(dv_len > 1e-5, "defocus_v should be non-zero");
    }
}
