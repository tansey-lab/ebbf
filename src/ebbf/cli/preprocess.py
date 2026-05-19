import argparse
import logging
import os

import numpy as np
import polars as pl

from ebbf.data.st_dataset import process_zarr

logger = logging.getLogger(__name__)


def plot_nuclear_ellipses(
    nuc_xy: np.ndarray, nuc_ellipse_mat: np.ndarray, output_path: str, n_grid: int = 64
):
    """Plot nuclear ellipses as a diagnostic figure.

    nuc_xy: (N, 2) centroids
    nuc_ellipse_mat: (N, 2, 2) ellipse matrices S
    """
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from shapely.geometry import Polygon

    c = np.asarray(nuc_xy)
    S = np.asarray(nuc_ellipse_mat)

    t = np.linspace(0.0, 2 * np.pi, n_grid, endpoint=False)
    unit = np.stack([np.cos(t), np.sin(t)], axis=1)
    ellipse_pts = c[:, np.newaxis, :] + np.einsum("bij, tj -> bti", S, unit)

    polygons = [Polygon(ellipse_pts[i]) for i in range(ellipse_pts.shape[0])]
    gdf = gpd.GeoDataFrame(geometry=polygons)

    fig, ax = plt.subplots(figsize=(12, 12))
    gdf.boundary.plot(ax=ax, linewidth=0.3, color="steelblue")
    ax.set_aspect("equal")
    ax.set_title(f"Nuclear ellipses (n={len(polygons)})")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved nuclear ellipse diagnostic plot to %s", output_path)


def write_summary_report(stdata, output_path: str):
    """Write an HTML report with summary statistics for an STDataset."""
    import html

    n_nucs = int(stdata.nuc_xy.shape[0])
    n_genes = int(stdata.genes.shape[0])
    n_boxes = (
        int(stdata.boxes.shape[0]) if hasattr(stdata, "boxes") and stdata.boxes is not None else 0
    )

    tx_counts = np.asarray(stdata.nuc_total_tx_counts)
    cell_counts = np.asarray(stdata.nuc_cell_counts)
    counts_per_entry = np.asarray(stdata.nuc_countmat_counts)

    xy = np.asarray(stdata.nuc_xy)
    x_min, y_min = xy.min(axis=0).tolist()
    x_max, y_max = xy.max(axis=0).tolist()

    S = np.asarray(stdata.nuc_ellipse_mat)
    ellipse_areas = np.pi * np.abs(np.linalg.det(S))

    def stats(a):
        a = np.asarray(a)
        if a.size == 0:
            return {
                "n": 0,
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "p25": float("nan"),
                "median": float("nan"),
                "p75": float("nan"),
                "max": float("nan"),
            }
        return {
            "n": int(a.size),
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
            "min": float(np.min(a)),
            "p25": float(np.percentile(a, 25)),
            "median": float(np.median(a)),
            "p75": float(np.percentile(a, 75)),
            "max": float(np.max(a)),
        }

    sections = {
        "Transcript counts per nucleus": stats(tx_counts),
        "Cells per nucleus": stats(cell_counts),
        "Nonzero gene counts (per nucleus×gene entry)": stats(counts_per_entry),
        "Nuclear ellipse area": stats(ellipse_areas),
    }

    def fmt(v):
        if isinstance(v, float):
            if not np.isfinite(v):
                return "nan"
            return f"{v:.3g}"
        return str(v)

    def stats_table(name, s):
        cols = ["n", "mean", "std", "min", "p25", "median", "p75", "max"]
        head = "".join(f"<th>{c}</th>" for c in cols)
        row = "".join(f"<td>{fmt(s[c])}</td>" for c in cols)
        return (
            f"<h3>{html.escape(name)}</h3>"
            f"<table><thead><tr>{head}</tr></thead>"
            f"<tbody><tr>{row}</tr></tbody></table>"
        )

    overview_rows = [
        ("Number of nuclei", f"{n_nucs:,}"),
        ("Number of genes", f"{n_genes:,}"),
        ("Number of boxes", f"{n_boxes:,}"),
        ("Total transcripts assigned to nuclei", f"{int(tx_counts.sum()):,}"),
        ("Nonzero countmat entries", f"{counts_per_entry.size:,}"),
        ("X range", f"[{x_min:.2f}, {x_max:.2f}]"),
        ("Y range", f"[{y_min:.2f}, {y_max:.2f}]"),
    ]
    overview = (
        "<table>"
        + "".join(
            f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>" for k, v in overview_rows
        )
        + "</table>"
    )

    body_stats = "\n".join(stats_table(k, v) for k, v in sections.items())

    plot_rel = "nuclear_ellipses.pdf"
    plot_link = f'<p><a href="{plot_rel}">Nuclear ellipses (PDF)</a></p>'

    html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>STDataset summary</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 2em; max-width: 900px; }}
table {{ border-collapse: collapse; margin: 0.5em 0 1.5em; }}
th, td {{ border: 1px solid #ccc; padding: 4px 10px; text-align: right; }}
th {{ background: #f4f4f4; }}
h1, h2, h3 {{ color: #222; }}
</style></head>
<body>
<h1>STDataset summary</h1>
<h2>Overview</h2>
{overview}
<h2>Distributions</h2>
{body_stats}
<h2>Diagnostic plots</h2>
{plot_link}
</body></html>
"""
    with open(output_path, "w") as f:
        f.write(html_doc)
    logger.info("Saved summary report to %s", output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess spatial transcriptomics data for EBB&Flow"
    )
    parser.add_argument(
        "--zarr_dir",
        type=str,
        required=True,
        help="Path to input directory (SpatialData zarr or Xenium output)",
    )
    parser.add_argument("--output", type=str, required=True, help="Output path for stdata.npz file")
    parser.add_argument(
        "--transcripts_output",
        type=str,
        default="transcripts.parquet",
        help="Output path for filtered transcripts parquet (default: transcripts.parquet)",
    )
    parser.add_argument(
        "--input_format",
        type=str,
        choices=["spatialdata", "xenium"],
        default=None,
        help="Input format: 'spatialdata' or 'xenium'. Auto-detected if not specified.",
    )
    parser.add_argument(
        "--box_width",
        type=float,
        default=400,
        help="Box width for spatial sampling (default: 400)",
    )
    parser.add_argument(
        "--ellipse_plot_n_grid",
        type=int,
        default=64,
        help="Number of boundary points per ellipse for the diagnostic plot (default: 64)",
    )
    args = parser.parse_args()

    stdata, points_folder, column_mapping = process_zarr(
        zarr_dir=args.zarr_dir,
        box_width=args.box_width,
        input_format=args.input_format,
    )
    stdata.save(args.output)
    logger.info("Saved STDataset to %s", args.output)

    plot_dir = os.path.join(os.path.dirname(os.path.abspath(args.output)), "plots")
    plot_nuclear_ellipses(
        nuc_xy=stdata.nuc_xy,
        nuc_ellipse_mat=stdata.nuc_ellipse_mat,
        output_path=os.path.join(plot_dir, "nuclear_ellipses.pdf"),
        n_grid=args.ellipse_plot_n_grid,
    )
    write_summary_report(stdata, os.path.join(plot_dir, "report.html"))

    # Write out original transcripts with normalized column names for postprocessing
    x_col = column_mapping["x"]
    y_col = column_mapping["y"]
    df = pl.scan_parquet(points_folder)
    df = df.filter(pl.col("is_gene"))
    df = df.select(
        [
            pl.col("transcript_id"),
            pl.col(x_col).alias("x"),
            pl.col(y_col).alias("y"),
            pl.col("overlaps_nucleus"),
        ]
    )
    df.sink_parquet(args.transcripts_output)
    logger.info("Saved transcripts to %s", args.transcripts_output)
