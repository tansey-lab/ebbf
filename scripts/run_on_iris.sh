#!/bin/bash
set -euo pipefail

usage() {
    echo "Usage: $0 <input_zarr_dir> <output_dir> [conda_env_path]"
    echo ""
    echo "Arguments:"
    echo "  input_zarr_dir   Path to SpatialData zarr directory"
    echo "  output_dir       Path to output directory"
    echo "  conda_env_path   Path to conda environment (default: /home/quinnj2/.conda/ebbf)"
    exit 1
}

if [[ $# -lt 2 ]]; then
    usage
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"
CONDA_ENV="${3:-/home/quinnj2/.conda/ebbf}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(dirname "$SCRIPT_DIR")/nextflow"

# Create temporary samplesheet
SAMPLE_NAME=$(basename "$INPUT_DIR" | sed 's/\.zarr$//')
SAMPLESHEET=$(mktemp --suffix=.csv)
echo "sample,zarr_dir" > "$SAMPLESHEET"
echo "${SAMPLE_NAME},${INPUT_DIR}" >> "$SAMPLESHEET"

WANDB_ARGS=""
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    WANDB_ARGS="--wandb_enabled true --wandb_api_key $WANDB_API_KEY"
fi

echo "Running EBB&Flow pipeline on iris"
echo "  Input:  $INPUT_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Conda:  $CONDA_ENV"
echo "  Samplesheet: $SAMPLESHEET"
[[ -n "$WANDB_ARGS" ]] && echo "  W&B:    enabled"

nextflow -log "$OUTPUT_DIR/.nextflow.log" run "$PIPELINE_DIR" \
    -profile mskcc_iris,conda \
    --input "$SAMPLESHEET" \
    --outdir "$OUTPUT_DIR" \
    --conda_env_path "$CONDA_ENV" \
    $WANDB_ARGS \
    -resume

rm -f "$SAMPLESHEET"
