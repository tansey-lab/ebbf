## EBB&Flow

**Empirical Bayes Bootstrapping & Flow Matching** for spatial transcriptomics transcript-to-cell assignment.

EBB&Flow uses flow matching to assign transcripts to cells in spatial transcriptomics data (e.g., Xenium, CosMx, MERFISH).

## Installation

```bash
pip install -e .
```

## Pipeline Overview

```
Xenium Data  →  Preprocess  →  Train  →  Infer  →  Post-process  →  Assignments
```

### Supported Input Formats

EBB&Flow supports two input formats for Xenium spatial transcriptomics data:

#### 1. Xenium Ranger Output Directory

Direct output from the 10x Genomics Xenium instrument:

```
xenium_output/
├── transcripts.parquet           # Transcript locations
├── nucleus_boundaries.parquet    # Nucleus polygons (required)
├── cell_boundaries.parquet       # Cell polygons (optional)
└── ...
```

**Required columns in `transcripts.parquet`:**
| Column | Type | Description |
|--------|------|-------------|
| `x_location` | float | X coordinate |
| `y_location` | float | Y coordinate |
| `feature_name` | string | Gene/feature name |
| `cell_id` | string | Cell ID (can be "UNASSIGNED") |
| `overlaps_nucleus` | int | 1 if transcript overlaps a nucleus |
| `is_gene` | bool | True for gene transcripts |

**Required columns in `nucleus_boundaries.parquet` / `cell_boundaries.parquet`:**
| Column | Type | Description |
|--------|------|-------------|
| `cell_id` | string | Cell ID |
| `vertex_x` | float | X coordinate of polygon vertex |
| `vertex_y` | float | Y coordinate of polygon vertex |

#### 2. SpatialData Zarr Directory

A [SpatialData](https://spatialdata.scverse.org/) zarr directory (e.g., converted via `spatialdata-io`):

```
my_data.zarr/
├── points/
│   └── transcripts/
│       └── points.parquet        # Transcript locations
├── shapes/
│   ├── nucleus_boundaries/
│   │   └── shapes.parquet        # Nucleus polygons (required)
│   └── cell_boundaries/
│       └── shapes.parquet        # Cell polygons (optional)
└── ...
```

**Required columns in `points.parquet`:**
| Column | Type | Description |
|--------|------|-------------|
| `x` | float | X coordinate |
| `y` | float | Y coordinate |
| `feature_name` | string | Gene/feature name |
| `cell_id` | string | Cell ID (can be "UNASSIGNED") |
| `overlaps_nucleus` | int | 1 if transcript overlaps a nucleus |
| `is_gene` | bool | True for gene transcripts |

**Required columns in `shapes.parquet`:**
| Column | Type | Description |
|--------|------|-------------|
| `geometry` | Polygon | Shapely polygon geometry |
| (index) | string | Cell ID matching `cell_id` in transcripts |

> **Note:** The input format is automatically detected based on directory structure.

### Output

The final output is a parquet file with transcript assignments:

| Column                 | Description                        |
| ---------------------- | ---------------------------------- |
| `transcript_id`        | Original transcript ID             |
| `feature_name`         | Gene/feature name                  |
| `cell_id`              | Original cell assignment           |
| `x`, `y`               | Flow-adjusted position             |
| `centroid_distance`    | Distance to assigned cell centroid |
| `mahalanobis_distance` | Mahalanobis distance to cell       |
| `ef_cluster_id`        | EBB&Flow cluster assignment        |

## Usage

### 1. Preprocess

Convert a SpatialData zarr directory into the preprocessed format:

```bash
ebbf-preprocess \
    --zarr_dir /path/to/data.zarr \
    --output /path/to/output/stdata.npz \
    --box_width 400 \
    --max_dist 40
```

**Arguments:**

- `--zarr_dir`: Path to SpatialData zarr directory
- `--output`: Output path for preprocessed `.npz` file
- `--box_width`: Spatial box width for sampling (default: 400)
- `--max_dist`: Maximum distance for histogram binning (default: 40)

### 2. Train

Train the flow matching model:

```bash
ebbf-train --config configs/example.yaml --dataset /path/to/output/stdata.npz
```

### 3. Infer

Run inference to assign transcripts:

```bash
ebbf-infer \
    --config /path/to/config.yaml \
    --weights /path/to/chkpoint.pt \
    --dataset /path/to/stdata.npz \
    --zarr_dir /path/to/spatialdata.zarr \
    --n_workers 4 \
    --worker_id 0
```

For parallel processing, run multiple workers with different `--worker_id` values.

### 4. Post-process

Aggregate and finalize assignments:

```bash
ebbf-postprocess \
    inference_output/*.parquet \
    --dataset /path/to/stdata.npz \
    --output ef.parquet \
    --anndata_output ef.h5ad \
    --cluster_distance 1.5 \
    --original_transcripts /path/to/transcripts.parquet
```

**Optional: Generate boundary comparison visualizations:**

```bash
ebbf-postprocess \
    inference_output/*.parquet \
    --dataset /path/to/stdata.npz \
    --original_transcripts /path/to/transcripts.parquet \
    --visualize \
    --xenium_dir /path/to/xenium_output \
    --viz_output_dir /path/to/viz_output \
    --viz_patch_size 500
```

This generates:

- `boundaries_patch_XXX.pdf` - Full tissue visualization split into patches
- `boundaries_zoom_*.pdf` - Zoomed views of center regions
- `cell_boundaries_combined.parquet` - Combined GeoParquet with both segmentations

The `--xenium_dir` argument accepts either Xenium Ranger output or SpatialData zarr directories.

## Nextflow pipeline

The nf-core pipeline under `nextflow/` runs preprocess → train → infer → postprocess by default. To **reuse existing inference outputs** and start at postprocessing, pass **`--inference_results`** with a directory that contains `part-*.parquet` shards (same naming as the infer step: `part-0000.parquet`, `part-0001.parquet`, …).

When `--inference_results` is set, training and inference are skipped; `model_weights`, `resume_training`, and `n_infer_workers` are not used.

**Shard layout:**

- **Multiple samples:** For each samplesheet row, shards must live under `<inference_results>/<sample_id>/part-*.parquet`, where `<sample_id>` matches the `sample` column in the CSV.
- **Single sample:** If that path is empty, the pipeline falls back to a flat directory: `<inference_results>/part-*.parquet` (only when the samplesheet has one row).

You still provide `--input` (samplesheet with `zarr_dir` per sample) and the usual preprocessing outputs (run preprocess, or use `--preprocess_dir` with `stdata.npz` and `transcripts.parquet`).

Example:

```bash
nextflow run nextflow/main.nf \
  --input samplesheet.csv \
  --outdir results \
  --inference_results /path/to/inference_shards \
  -profile docker
```

## Integration test (mouse ileum)

To run the local integration test on the sample mouse ileum dataset:

```bash
source .venv/bin/activate
bash scripts/run_mouse_ileum.sh /Users/quinnj2/Code/ebbf/Xenium_Prime_Mouse_Ileum
```

If Nextflow reports a session lock error (`Unable to acquire lock on session ...`), identify and stop
the process holding the lock, then rerun:

```bash
lsof /Users/quinnj2/Code/ebbf/nextflow/.nextflow/cache/<session-id>/db/LOCK
kill <PID>
```

## Configuration

See `configs/example.yaml` for a full configuration example. Key settings:

```yaml
data:
  patch_size: 100 # Spatial patch size
  max_tx: 32768 # Max transcripts per batch

model:
  img_resolution: 128 # Internal resolution
  model_channels: 128 # Model width

train:
  batch_size: 64
  n_epochs: 350
```

## License

See LICENSE.txt
