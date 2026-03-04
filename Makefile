# Makefile — helper targets for the mcrt render regression testing framework.
#
# All generated artefacts land in tmp/regression/ which is excluded from git.
# Only source files (scene YAMLs, sidecar TOMLs, generator scripts) are
# committed.  Run `make help` for a brief description of each target.

.PHONY: help gen-test-assets regress-baseline clean-regression

# Default target: print help.
help:
	@echo "Targets:"
	@echo "  gen-test-assets   Generate all synthetic test assets (HDR skymaps, etc.)"
	@echo "  regress-baseline  Capture regression baseline from a git commitish."
	@echo "                    Usage: make regress-baseline FROM=HEAD^"
	@echo "  clean-regression  Remove all generated artefacts in tmp/regression/"

# ---------------------------------------------------------------------------
# gen-test-assets — create the synthetic HDR environment map used by regression
# tests that exercise the env-map sampling code path.
#
# Requires uv (https://github.com/astral-sh/uv) to be on PATH.
# uv automatically resolves and caches the Python environment declared in the
# PEP 723 inline script metadata; no manual venv activation is needed.
# ---------------------------------------------------------------------------
gen-test-assets:
	@echo "→ Generating synthetic test HDR skymap …"
	@mkdir -p tmp/regression/skyboxes
	uv run tests/assets/gen_test_skymap.py \
		--width 512 --height 256 \
		--sky-horizon-nit 0.2 --sky-zenith-nit 0.5 \
		--output tmp/regression/skyboxes/test_sky.hdr
	@echo "✓ Test assets generated in tmp/regression/"

# ---------------------------------------------------------------------------
# regress-baseline — check out <FROM>, build it, render all regression scenes,
# and store results in tmp/regression/baselines/<short-sha>/.
#
# Usage:  make regress-baseline FROM=HEAD^
#         make regress-baseline FROM=a1b2c3d
# ---------------------------------------------------------------------------
regress-baseline:
	@test -n "$(FROM)" || (echo "error: FROM is required — e.g. make regress-baseline FROM=HEAD^"; exit 1)
	bash scripts/regress_baseline.sh --from $(FROM)

# ---------------------------------------------------------------------------
# clean-regression — wipe all generated regression artefacts.
# The tmp/ directory is git-ignored, so this is safe to run at any time.
# ---------------------------------------------------------------------------
clean-regression:
	@echo "→ Removing tmp/regression/ …"
	rm -rf tmp/regression/
	@echo "✓ Done."
