//! RT-5: Regression test runner.
//!
//! Compares rendered images from the current working copy against a captured
//! baseline, failing when any scene exceeds its configured MSE/PSNR thresholds.
//!
//! # Required environment variables
//!
//! | Variable                   | Description                                                      |
//! |----------------------------|------------------------------------------------------------------|
//! | `MCRT_REGRESSION_BASELINE` | Directory containing baseline PNGs (`<scene>.png`) + manifest    |
//! | `MCRT_REGRESSION_CURRENT`  | Directory for current-render PNGs (created automatically if absent) |
//!
//! When **either** variable is absent the entire test file is **skipped** — it
//! does not count as a failure.  This allows `cargo test` to run cleanly on
//! developer machines that have not captured a baseline.
//!
//! # Optional environment variables
//!
//! | Variable                   | Default                    | Description                              |
//! |----------------------------|----------------------------|------------------------------------------|
//! | `MCRT_REGRESSION_FORCE`    | `0`                        | Set to `1` to re-render even if current PNGs are fresh  |
//! | `MCRT_REGRESSION_DIFF_DIR` | `tmp/regression/diff`      | Directory for diff PNG output on failure |
//!
//! # Typical workflow
//!
//! ```sh
//! # 1. Capture a baseline from the previous commit:
//! scripts/regress_baseline.sh --from HEAD^
//!
//! # 2. Run the full regression suite (renders current images, then compares):
//! MCRT_REGRESSION_BASELINE=tmp/regression/baselines/<sha> \
//! MCRT_REGRESSION_CURRENT=tmp/regression/current \
//!   cargo test regression -- --nocapture
//!
//! # 3. Force a clean re-render:
//! MCRT_REGRESSION_FORCE=1 \
//! MCRT_REGRESSION_BASELINE=tmp/regression/baselines/<sha> \
//! MCRT_REGRESSION_CURRENT=tmp/regression/current \
//!   cargo test regression -- --nocapture
//! ```

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use mcrt::regression::{
    DiffResult, SceneThresholds, compare_images, load_thresholds, write_diff_image,
};

/// Directory where test scene files live.
const SCENES_DIR: &str = "tests/assets/scenes";
/// Default directory for diff PNG output.
const DEFAULT_DIFF_DIR: &str = "tmp/regression/diff";

// ---------------------------------------------------------------------------
// Sidecar render-param parsing (local to this test)
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize, Default)]
struct RenderSection {
    width: Option<u32>,
    height: Option<u32>,
    spp: Option<u32>,
}

#[derive(serde::Deserialize, Default)]
struct SidecarFile {
    render: Option<RenderSection>,
}

fn load_render_params(scene_yaml: &Path) -> RenderSection {
    let sidecar = scene_yaml.with_extension("").with_extension("test.toml");
    let text = std::fs::read_to_string(&sidecar).unwrap_or_default();
    let parsed: SidecarFile = toml::from_str(&text).unwrap_or_default();
    parsed.render.unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Scene freshness check
// ---------------------------------------------------------------------------

/// Return `true` when `current_png` already exists **and** its mtime is
/// strictly newer than `scene_yaml`, its `.test.toml` sidecar, **and** the
/// test binary.
///
/// Including the sidecar mtime ensures that changes to `spp`, `width`, or
/// `height` in the sidecar always trigger a re-render.
///
/// When the binary mtime cannot be determined (e.g. unsupported filesystem),
/// `SystemTime::now()` is used so the check conservatively triggers a re-render.
fn is_up_to_date(current_png: &Path, scene_yaml: &Path, binary: &Path) -> bool {
    let mtime = |p: &Path| -> Option<SystemTime> {
        std::fs::metadata(p).ok().and_then(|m| m.modified().ok())
    };
    let Some(png_mtime) = mtime(current_png) else {
        return false; // PNG does not exist
    };
    let sidecar = scene_yaml.with_extension("").with_extension("test.toml");
    let yaml_mtime = mtime(scene_yaml).unwrap_or(SystemTime::UNIX_EPOCH);
    let sidecar_mtime = mtime(&sidecar).unwrap_or(SystemTime::UNIX_EPOCH);
    let bin_mtime = mtime(binary).unwrap_or_else(SystemTime::now);
    png_mtime > yaml_mtime && png_mtime > sidecar_mtime && png_mtime > bin_mtime
}

// ---------------------------------------------------------------------------
// Renderer invocation
// ---------------------------------------------------------------------------

/// Invoke the `mcrt --headless` binary to render `scene_yaml` to `output`.
///
/// Returns `Ok(())` on zero exit, or `Err(message)` on non-zero exit or if the
/// binary cannot be launched.  The binary's stdout/stderr are inherited so
/// progress output is visible with `-- --nocapture`.
fn render_scene(
    binary: &Path,
    scene_yaml: &Path,
    output: &Path,
    render: &RenderSection,
) -> Result<(), String> {
    let mut cmd = std::process::Command::new(binary);
    cmd.arg("--headless")
        .arg(scene_yaml)
        .arg("--output")
        .arg(output);
    if let Some(w) = render.width {
        cmd.arg("--width").arg(w.to_string());
    }
    if let Some(h) = render.height {
        cmd.arg("--height").arg(h.to_string());
    }
    if let Some(s) = render.spp {
        cmd.arg("--spp").arg(s.to_string());
    }
    let status = cmd
        .status()
        .map_err(|e| format!("failed to launch `{}`: {e}", binary.display()))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("renderer exited with status {status}"))
    }
}

// ---------------------------------------------------------------------------
// Scene discovery
// ---------------------------------------------------------------------------

struct SceneEntry {
    stem: String,
    yaml_path: PathBuf,
}

fn discover_scenes() -> Vec<SceneEntry> {
    let mut scenes: Vec<SceneEntry> = std::fs::read_dir(SCENES_DIR)
        .unwrap_or_else(|e| panic!("cannot open scenes directory `{SCENES_DIR}`: {e}"))
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            // Keep only *.test.toml files.
            if path.extension()? != "toml" {
                return None;
            }
            let stem_with_test = path.file_stem()?.to_str()?.to_owned();
            if !stem_with_test.ends_with(".test") {
                return None;
            }
            // Use strip_suffix to remove exactly one ".test" occurrence;
            // trim_end_matches would be greedy and strip multiple suffixes.
            let stem = stem_with_test.strip_suffix(".test")?.to_owned();
            let yaml_path = PathBuf::from(SCENES_DIR).join(format!("{stem}.yaml"));
            yaml_path.exists().then_some(SceneEntry { stem, yaml_path })
        })
        .collect();
    scenes.sort_by(|a, b| a.stem.cmp(&b.stem));
    scenes
}

// ---------------------------------------------------------------------------
// Per-scene result
// ---------------------------------------------------------------------------

enum SceneResult {
    Pass {
        result: DiffResult,
    },
    Fail {
        result: DiffResult,
        reason: String,
    },
    /// Baseline PNG absent — schema-change skip.
    SkipNoBaseline,
    /// Current PNG absent and no render was attempted (e.g. cached path skipped).
    SkipNoRender(String),
    /// Renderer subprocess exited non-zero — always a hard failure.
    RenderFailed(String),
    /// Images have incompatible dimensions.
    DimMismatch(String),
    /// PNG could not be opened.
    IoError(String),
}

struct SummaryRow {
    stem: String,
    result: SceneResult,
}

// ---------------------------------------------------------------------------
// The test
// ---------------------------------------------------------------------------

#[test]
fn regression_suite() {
    // -----------------------------------------------------------------------
    // 1. Env-var gate — skip the whole test when vars are absent.
    // -----------------------------------------------------------------------
    let baseline_str = match std::env::var("MCRT_REGRESSION_BASELINE") {
        Ok(v) => v,
        Err(_) => {
            println!(
                "SKIP: MCRT_REGRESSION_BASELINE not set — \
                 run `scripts/regress_baseline.sh` first, then set the variable."
            );
            return;
        }
    };
    let current_str = match std::env::var("MCRT_REGRESSION_CURRENT") {
        Ok(v) => v,
        Err(_) => {
            println!("SKIP: MCRT_REGRESSION_CURRENT not set.");
            return;
        }
    };
    let force = std::env::var("MCRT_REGRESSION_FORCE")
        .map(|v| v.trim() == "1")
        .unwrap_or(false);
    let diff_dir =
        std::env::var("MCRT_REGRESSION_DIFF_DIR").unwrap_or_else(|_| DEFAULT_DIFF_DIR.to_string());

    let baseline_dir = PathBuf::from(&baseline_str);
    let current_dir = PathBuf::from(&current_str);
    let diff_path = PathBuf::from(&diff_dir);

    assert!(
        baseline_dir.is_dir(),
        "MCRT_REGRESSION_BASELINE `{baseline_str}` is not an existing directory"
    );
    std::fs::create_dir_all(&current_dir)
        .expect("failed to create MCRT_REGRESSION_CURRENT directory");

    let binary = PathBuf::from(env!("CARGO_BIN_EXE_mcrt"));
    let scenes = discover_scenes();
    assert!(!scenes.is_empty(), "no scenes found in `{SCENES_DIR}`");

    // -----------------------------------------------------------------------
    // 2. Render step — produce current PNGs, reusing fresh ones.
    // -----------------------------------------------------------------------
    println!("\n=== Render step (baseline: {baseline_str}) ===\n");

    let mut render_errors: std::collections::HashMap<String, String> = Default::default();
    let render_step_start = std::time::Instant::now();

    for scene in &scenes {
        let current_png = current_dir.join(format!("{}.png", scene.stem));
        let render = load_render_params(&scene.yaml_path);

        if !force && is_up_to_date(&current_png, &scene.yaml_path, &binary) {
            println!("[CACHED ] {}", scene.stem);
            continue;
        }

        print!("[RENDER ] {} … ", scene.stem);
        // Flush before the (potentially long) subprocess prints its own output.
        use std::io::Write as _;
        let _ = std::io::stdout().flush();

        let t = std::time::Instant::now();
        match render_scene(&binary, &scene.yaml_path, &current_png, &render) {
            Ok(()) => println!("ok ({:.2?})", t.elapsed()),
            Err(msg) => {
                println!("FAILED — {msg} ({:.2?})", t.elapsed());
                render_errors.insert(scene.stem.clone(), msg);
            }
        }
    }
    println!("\nRender step total: {:.2?}", render_step_start.elapsed());

    // -----------------------------------------------------------------------
    // 3. Compare step.
    // -----------------------------------------------------------------------
    println!("\n=== Compare step ===\n");

    let mut rows: Vec<SummaryRow> = Vec::new();

    for scene in &scenes {
        let baseline_png = baseline_dir.join(format!("{}.png", scene.stem));
        let current_png = current_dir.join(format!("{}.png", scene.stem));
        let thresholds = load_thresholds(&scene.yaml_path);

        // (a) No baseline — schema change or scene added after baseline was captured.
        if !baseline_png.exists() {
            println!("[SKIP   ] {} — no baseline PNG", scene.stem);
            rows.push(SummaryRow {
                stem: scene.stem.clone(),
                result: SceneResult::SkipNoBaseline,
            });
            continue;
        }

        // (b) No current render — either the render subprocess failed (hard
        // failure) or the PNG was never produced for another reason (skip).
        if !current_png.exists() {
            if let Some(err) = render_errors.get(&scene.stem) {
                println!("[FAIL   ] {} — render failed: {err}", scene.stem);
                rows.push(SummaryRow {
                    stem: scene.stem.clone(),
                    result: SceneResult::RenderFailed(err.clone()),
                });
            } else {
                println!(
                    "[SKIP   ] {} — current PNG absent (render not attempted?)",
                    scene.stem
                );
                rows.push(SummaryRow {
                    stem: scene.stem.clone(),
                    result: SceneResult::SkipNoRender(format!(
                        "`{}` absent",
                        current_png.display()
                    )),
                });
            }
            continue;
        }

        // (c) Load images.
        let img_baseline = match image::open(&baseline_png) {
            Ok(i) => i,
            Err(e) => {
                println!("[ERROR  ] {} — cannot open baseline: {e}", scene.stem);
                rows.push(SummaryRow {
                    stem: scene.stem.clone(),
                    result: SceneResult::IoError(format!("baseline: {e}")),
                });
                continue;
            }
        };
        let img_current = match image::open(&current_png) {
            Ok(i) => i,
            Err(e) => {
                println!("[ERROR  ] {} — cannot open current: {e}", scene.stem);
                rows.push(SummaryRow {
                    stem: scene.stem.clone(),
                    result: SceneResult::IoError(format!("current: {e}")),
                });
                continue;
            }
        };

        // (d) Compute diff metrics.
        let diff = match compare_images(&img_baseline, &img_current) {
            Ok(d) => d,
            Err(e) => {
                println!("[FAIL   ] {} — dimension mismatch: {e}", scene.stem);
                rows.push(SummaryRow {
                    stem: scene.stem.clone(),
                    result: SceneResult::DimMismatch(e.to_string()),
                });
                continue;
            }
        };

        // (e) Gate against thresholds.
        let mse_fail = diff.mse > thresholds.threshold_mse;
        let psnr_fail = diff.psnr.is_finite() && diff.psnr < thresholds.threshold_psnr;

        if mse_fail || psnr_fail {
            // Write a diff image to aid visual debugging.
            let _ = std::fs::create_dir_all(&diff_path);
            let diff_png = diff_path.join(format!("{}_diff.png", scene.stem));
            match write_diff_image(&img_baseline, &img_current, &diff_png, 4.0) {
                Ok(()) => println!("[FAIL   ] {} — diff → {}", scene.stem, diff_png.display()),
                Err(e) => println!("[FAIL   ] {} (diff write error: {e})", scene.stem),
            }
            let reason = build_fail_reason(&diff, &thresholds, mse_fail, psnr_fail);
            rows.push(SummaryRow {
                stem: scene.stem.clone(),
                result: SceneResult::Fail {
                    result: diff,
                    reason,
                },
            });
        } else {
            println!("[PASS   ] {}", scene.stem);
            rows.push(SummaryRow {
                stem: scene.stem.clone(),
                result: SceneResult::Pass { result: diff },
            });
        }
    }

    // -----------------------------------------------------------------------
    // 4. Print Markdown summary table.
    // -----------------------------------------------------------------------
    print_summary_table(&rows, &baseline_str);

    // -----------------------------------------------------------------------
    // 5. Fail the test if any scene regressed.
    // -----------------------------------------------------------------------
    let failures: Vec<&str> = rows
        .iter()
        .filter_map(|r| match &r.result {
            // RenderFailed: renderer exited non-zero (e.g. no GPU adapter).
            // IoError: corrupted or unreadable PNG.
            // Both must not silently pass the suite.
            SceneResult::Fail { .. }
            | SceneResult::RenderFailed(_)
            | SceneResult::DimMismatch(_)
            | SceneResult::IoError(_) => Some(r.stem.as_str()),
            _ => None,
        })
        .collect();

    if !failures.is_empty() {
        panic!(
            "\n{} scene(s) failed regression: {}\n\
             See diff images in `{diff_dir}` and the summary table above.",
            failures.len(),
            failures.join(", ")
        );
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn build_fail_reason(
    diff: &DiffResult,
    thresholds: &SceneThresholds,
    mse_fail: bool,
    psnr_fail: bool,
) -> String {
    let mut parts = Vec::new();
    if mse_fail {
        parts.push(format!(
            "MSE {:.6} > {:.6}",
            diff.mse, thresholds.threshold_mse
        ));
    }
    if psnr_fail {
        parts.push(format!(
            "PSNR {:.2} < {:.2} dB",
            diff.psnr, thresholds.threshold_psnr
        ));
    }
    parts.join("; ")
}

fn print_summary_table(rows: &[SummaryRow], baseline: &str) {
    println!("\n## Regression summary — baseline `{baseline}`\n");
    println!(
        "| {:<22} | {:>9} | {:>9} | {:>8} | Status |",
        "Scene", "MSE", "PSNR (dB)", "Max Δ"
    );
    println!(
        "|{:-<24}|{:-<11}|{:-<11}|{:-<10}|{:-<40}|",
        "", "", "", "", ""
    );
    for row in rows {
        let (mse_s, psnr_s, delta_s, status_s) = match &row.result {
            SceneResult::Pass { result } => (
                format!("{:.6}", result.mse),
                format!("{:.2}", result.psnr),
                format!("{:.4}", result.max_delta),
                "✓ PASS".to_string(),
            ),
            SceneResult::Fail { result, reason } => (
                format!("{:.6}", result.mse),
                format!("{:.2}", result.psnr),
                format!("{:.4}", result.max_delta),
                format!("✗ FAIL — {reason}"),
            ),
            SceneResult::SkipNoBaseline => (
                "—".into(),
                "—".into(),
                "—".into(),
                "⚠ SKIP — no baseline".to_string(),
            ),
            SceneResult::SkipNoRender(msg) => (
                "—".into(),
                "—".into(),
                "—".into(),
                format!("⚠ SKIP — {msg}"),
            ),
            SceneResult::RenderFailed(msg) => (
                "—".into(),
                "—".into(),
                "—".into(),
                format!("✗ RENDER FAILED — {msg}"),
            ),
            SceneResult::DimMismatch(msg) => (
                "—".into(),
                "—".into(),
                "—".into(),
                format!("✗ DIM MISMATCH — {msg}"),
            ),
            SceneResult::IoError(msg) => (
                "—".into(),
                "—".into(),
                "—".into(),
                format!("✗ IO ERROR — {msg}"),
            ),
        };
        println!(
            "| {:<22} | {:>9} | {:>9} | {:>8} | {status_s} |",
            row.stem, mse_s, psnr_s, delta_s
        );
    }
    println!();
}
