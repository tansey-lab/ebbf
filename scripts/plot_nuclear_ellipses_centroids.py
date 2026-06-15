import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import anndata as ad
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from shapely import from_wkb

ADATA_PATH = Path("ebbf.unfiltered.h5ad")
NUCLEAR_ELLIPSES_PATH = Path("../preprocess/plots/nuclear_ellipses.parquet")
OUTPUT_PATH = Path("nuclear_ellipses_ebbf_centroids.pdf")

POINT_COLORS = {
    "original": "#4c78a8",
    "denovo": "#f58518",
    "split": "#54a24b",
}


def main() -> None:
    adata = ad.read_h5ad(ADATA_PATH)
    obs = adata.obs.copy()

    ellipse_ids = set(obs.index.astype(str))
    ellipse_ids.update(obs["parent_cell_id"].dropna().astype(str))

    ellipse_df = pd.read_parquet(NUCLEAR_ELLIPSES_PATH, columns=["cell_id", "geometry"])
    ellipse_df = ellipse_df.loc[ellipse_df["cell_id"].isin(ellipse_ids)].copy()
    ellipse_df["geometry"] = from_wkb(ellipse_df["geometry"].to_numpy())
    ellipses = gpd.GeoDataFrame(ellipse_df, geometry="geometry", crs=None)
    centroids = gpd.GeoDataFrame(
        obs,
        geometry=gpd.points_from_xy(obs["x_centroid"], obs["y_centroid"]),
        crs=None,
    )

    fig, ax = plt.subplots(figsize=(13, 11), constrained_layout=True)

    ellipses.boundary.plot(
        ax=ax,
        color="#8a9099",
        linewidth=0.18,
        alpha=0.45,
        zorder=1,
    )

    for cell_type, color in POINT_COLORS.items():
        layer = centroids.loc[centroids["cell_category"].eq(cell_type)]
        layer.plot(
            ax=ax,
            marker="o",
            markersize=0.9 if cell_type == "original" else 1.6,
            color=color,
            edgecolor="none",
            linewidth=0,
            alpha=0.72,
            zorder=2 if cell_type == "original" else 3,
        )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.4,
            markersize=7,
            label=cell_type,
        )
        for cell_type, color in POINT_COLORS.items()
    ]
    ax.legend(
        handles=handles,
        title="Cell type",
        loc="upper right",
        frameon=True,
        framealpha=0.95,
    )

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_axis_off()
    ax.set_title(
        "Nuclear ellipses with EBBF cell centroids",
        fontsize=14,
        pad=12,
    )

    fig.savefig(OUTPUT_PATH, format="pdf", dpi=300)

    print(f"Wrote {OUTPUT_PATH.resolve()}")
    print(f"Nuclear ellipses plotted from {NUCLEAR_ELLIPSES_PATH}: {len(ellipses):,}")
    print(f"EBBF centroids plotted: {len(centroids):,}")
    print(centroids["cell_category"].value_counts().to_string())


if __name__ == "__main__":
    main()
