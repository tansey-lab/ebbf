#!/bin/bash
set -euo pipefail

# Run EBBF pipeline integration test on Xenium_Prime_Mouse_Ileum dataset
# Usage: ./scripts/run_mouse_ileum.sh [zarr_path]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default path - override with first argument if provided
ZARR_PATH="${1:-/path/to/Xenium_Prime_Mouse_Ileum.zarr}"

# Output directories
OUTDIR="${PROJECT_DIR}/results"
WORKDIR="${PROJECT_DIR}/work"

# Create samplesheet
SAMPLESHEET="${PROJECT_DIR}/samplesheets/mouse_ileum.csv"
mkdir -p "$(dirname "$SAMPLESHEET")"

cat > "$SAMPLESHEET" << EOF
sample,zarr_dir
mouse_ileum,${ZARR_PATH}
EOF

echo "Created samplesheet: $SAMPLESHEET"
echo "Using zarr: $ZARR_PATH"
echo "Output directory: $OUTDIR"

# Run Nextflow pipeline with test profile (minimal training: 1 epoch, 100 samples)
cd "${PROJECT_DIR}/nextflow"

nextflow run main.nf \
    -profile test \
    --input "$SAMPLESHEET" \
    --outdir "$OUTDIR" \
    --experiment_name mouse_ileum \
    -work-dir "$WORKDIR" \
    -resume

echo "Pipeline complete. Results in: $OUTDIR"
