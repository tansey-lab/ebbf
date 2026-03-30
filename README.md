## EBB&Flow

**Empirical Bayes Bootstrapping & Flow Matching** for spatial transcriptomics transcript-to-cell assignment.

EBB&Flow uses flow matching to assign transcripts to cells in spatial transcriptomics data (e.g., Xenium, CosMx, MERFISH).

## Installation

```bash
pip install -e .
```

## Pipeline Overview

```
SpatialData Zarr  →  Preprocess  →  Train  →  Infer  →  Post-process  →  Assignments
```

### Input: SpatialData Zarr Directory

EBB&Flow expects a [SpatialData](https://spatialdata.scverse.org/) zarr directory with the following structure:

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
    --model_directory /path/to/trained_model \
    --tma_directory /path/to/output \
    --cluster_distance 1.5
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
