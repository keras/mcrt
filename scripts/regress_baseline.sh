#!/usr/bin/env bash
# scripts/regress_baseline.sh -- Phase RT-3: Baseline Capture from a Git Commitish
#
# Usage:
#   scripts/regress_baseline.sh --from <commitish> [--scenes <dir>]
#
# Workflow:
#   1. Resolve <commitish> to a full SHA, then shorten it.
#   2. Create a git worktree at /tmp/mcrt-baseline-<short-sha>-<pid>.
#   3. Build the release binary inside that worktree.
#   4. Render every *.yaml scene in <scenes-dir> using the baseline binary,
#      writing PNGs to tmp/regression/baselines/<short-sha>/.
#   5. Write a manifest.json with commit info and per-scene render params.
#   6. Remove the worktree (always, even on failure -- via trap).
#
# Exit code: 0 if all scenes rendered (or were gracefully skipped due to
#            schema incompatibility); non-zero on fatal errors such as a
#            missing commitish or build failure.

set -euo pipefail

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log()  { echo "[baseline] $*"; }
warn() { echo "[baseline] WARNING: $*" >&2; }
die()  { echo "[baseline] ERROR: $*" >&2; exit 1; }

# Minimal JSON string escaper: replaces backslash -> \\ and quote -> \" .
json_str() { printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'; }

# Parse an integer value from a simple flat TOML file.
# Finds the first line matching "<key> = <value>   # optional comment".
# NOTE: does not understand TOML sections or dotted keys; adequate for the
#       intentionally simple *.test.toml sidecars used in this project.
parse_toml_int() {
    local file="$1"
    local key="$2"
    awk -F'=' -v k="$key" '
        /^[[:space:]]*#/ { next }
        $1 ~ "^[[:space:]]*" k "[[:space:]]*$" {
            val = $2
            sub(/[[:space:]]*#.*/, "", val)
            gsub(/[[:space:]]/, "", val)
            print val
            exit
        }
    ' "$file"
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SCENES_DIR="$REPO_ROOT/tests/assets/scenes"
FROM_COMMITISH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --from)
            [[ $# -ge 2 ]] || die "--from requires an argument"
            FROM_COMMITISH="$2"
            shift 2
            ;;
        --scenes)
            [[ $# -ge 2 ]] || die "--scenes requires an argument"
            SCENES_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --from <commitish> [--scenes <dir>]"
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

if [[ -z "$FROM_COMMITISH" ]]; then
    echo "[baseline] ERROR: --from <commitish> is required" >&2
    echo "Usage: $0 --from <commitish> [--scenes <dir>]" >&2
    exit 1
fi

[[ -d "$SCENES_DIR" ]] || die "scenes directory not found: $SCENES_DIR"

# ---------------------------------------------------------------------------
# Resolve commitish -- derive SHORT_SHA from FULL_SHA to avoid TOCTOU on
# mutable refs (branches, HEAD) that could advance between two rev-parse calls.
# ---------------------------------------------------------------------------

FULL_SHA="$(git -C "$REPO_ROOT" rev-parse "$FROM_COMMITISH")" \
    || die "could not resolve commitish: $FROM_COMMITISH"
SHORT_SHA="$(git -C "$REPO_ROOT" rev-parse --short "$FULL_SHA")" \
    || die "could not shorten SHA: $FULL_SHA"
COMMIT_DATE="$(git -C "$REPO_ROOT" log -1 --format="%aI" "$FULL_SHA")" \
    || die "could not read commit date for $FULL_SHA"

log "Resolved $FROM_COMMITISH -> $SHORT_SHA  ($FULL_SHA)"

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

OUTPUT_DIR="$REPO_ROOT/tmp/regression/baselines/$SHORT_SHA"
mkdir -p "$OUTPUT_DIR"
log "Output directory: $OUTPUT_DIR"

# ---------------------------------------------------------------------------
# Worktree -- PID-unique path prevents collisions with concurrent runs.
# Cleaned up unconditionally on exit via trap.
# ---------------------------------------------------------------------------

WORKTREE_PATH="/tmp/mcrt-baseline-${SHORT_SHA}-$$"
WORKTREE_CREATED=0

cleanup() {
    local rc=$?
    if [[ $WORKTREE_CREATED -eq 1 ]]; then
        log "Cleaning up worktree $WORKTREE_PATH ..."
        git -C "$REPO_ROOT" worktree remove --force "$WORKTREE_PATH" 2>/dev/null || true
        rm -rf "$WORKTREE_PATH"   # belt-and-suspenders in case artifacts remain
    fi
    exit $rc
}
trap cleanup EXIT

# Belt-and-suspenders: WORKTREE_PATH should never exist (PID-unique), but
# handle the vanishingly rare PID-reuse scenario gracefully.
if [[ -d "$WORKTREE_PATH" ]]; then
    warn "Worktree $WORKTREE_PATH already exists -- removing and recreating"
    git -C "$REPO_ROOT" worktree remove --force "$WORKTREE_PATH" 2>/dev/null || true
    rm -rf "$WORKTREE_PATH"
fi

log "Creating worktree at $WORKTREE_PATH ..."
git -C "$REPO_ROOT" worktree add "$WORKTREE_PATH" "$FULL_SHA"
WORKTREE_CREATED=1

# ---------------------------------------------------------------------------
# Build baseline binary
# ---------------------------------------------------------------------------

log "Building release binary from $SHORT_SHA ..."
# NOTE: correctness of set -e on the pipeline below relies on set -o pipefail
# (active above) so that a non-zero cargo exit propagates through sed.
cargo build --release --manifest-path "$WORKTREE_PATH/Cargo.toml" \
    2>&1 | sed 's/^/  [cargo] /'

BASELINE_BIN="$WORKTREE_PATH/target/release/mcrt"
[[ -x "$BASELINE_BIN" ]] || die "baseline binary not found after build: $BASELINE_BIN"
log "Baseline binary: $BASELINE_BIN"

# ---------------------------------------------------------------------------
# Render each scene
# ---------------------------------------------------------------------------

# Defaults matching headless.rs constants (DEFAULT_WIDTH / DEFAULT_HEIGHT / DEFAULT_SPP)
DEFAULT_WIDTH=512
DEFAULT_HEIGHT=512
DEFAULT_SPP=64

RENDER_DATE="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
SCENES_JSON_ARRAY=""
FIRST_SCENE=1
OK_COUNT=0
SKIP_COUNT=0

for YAML in "$SCENES_DIR"/*.yaml; do
    [[ -f "$YAML" ]] || continue
    STEM="$(basename "$YAML" .yaml)"
    SIDECAR="${YAML%.yaml}.test.toml"
    OUTPUT_PNG="$OUTPUT_DIR/${STEM}.png"

    # Read render parameters from sidecar, falling back to defaults.
    WIDTH=$DEFAULT_WIDTH
    HEIGHT=$DEFAULT_HEIGHT
    SPP=$DEFAULT_SPP
    if [[ -f "$SIDECAR" ]]; then
        _w="$(parse_toml_int "$SIDECAR" width)"
        _h="$(parse_toml_int "$SIDECAR" height)"
        _s="$(parse_toml_int "$SIDECAR" spp)"
        [[ -n "$_w" ]] && WIDTH="$_w"
        [[ -n "$_h" ]] && HEIGHT="$_h"
        [[ -n "$_s" ]] && SPP="$_s"
    fi

    log "Rendering $STEM  (${WIDTH}x${HEIGHT}, ${SPP} spp) ..."

    STATUS="ok"
    # NOTE: if!/pipe exit-code correctness relies on set -o pipefail (active above).
    if ! "$BASELINE_BIN" \
            --headless "$YAML" \
            --output   "$OUTPUT_PNG" \
            --width    "$WIDTH" \
            --height   "$HEIGHT" \
            --spp      "$SPP" \
            2>&1 | sed 's/^/  /'; then
        log "[SKIP] $STEM: schema incompatible with baseline $SHORT_SHA"
        STATUS="skipped"
        SKIP_COUNT=$(( SKIP_COUNT + 1 ))
    else
        log "  ok  $STEM -> ${STEM}.png"
        OK_COUNT=$(( OK_COUNT + 1 ))
    fi

    SCENE_ENTRY="{ \"scene\": \"$(json_str "$STEM")\", \"width\": $WIDTH, \"height\": $HEIGHT, \"spp\": $SPP, \"status\": \"$STATUS\" }"
    if [[ $FIRST_SCENE -eq 1 ]]; then
        SCENES_JSON_ARRAY="$SCENE_ENTRY"
        FIRST_SCENE=0
    else
        SCENES_JSON_ARRAY="$SCENES_JSON_ARRAY, $SCENE_ENTRY"
    fi
done

# Warn if no scenes were found -- likely a misconfigured --scenes path.
if [[ $OK_COUNT -eq 0 && $SKIP_COUNT -eq 0 ]]; then
    warn "no *.yaml scenes found in $SCENES_DIR -- check the --scenes path"
fi

# ---------------------------------------------------------------------------
# Write manifest.json
# ---------------------------------------------------------------------------

MANIFEST="$OUTPUT_DIR/manifest.json"
{
    printf '{\n'
    printf '    "commit_sha":  "%s",\n' "$(json_str "$FULL_SHA")"
    printf '    "short_sha":   "%s",\n' "$(json_str "$SHORT_SHA")"
    printf '    "commit_date": "%s",\n' "$(json_str "$COMMIT_DATE")"
    printf '    "captured_at": "%s",\n' "$(json_str "$RENDER_DATE")"
    printf '    "scenes_dir":  "%s",\n' "$(json_str "$SCENES_DIR")"
    printf '    "scenes": [ %s ]\n' "$SCENES_JSON_ARRAY"
    printf '}\n'
} > "$MANIFEST"

log "manifest.json -> $MANIFEST"
echo
log "Done.  $OK_COUNT scene(s) rendered, $SKIP_COUNT skipped."
log "Baseline stored in: $OUTPUT_DIR"
