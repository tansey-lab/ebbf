import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import anndata as ad
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from shapely import from_wkb

ADATA_PATH = Path("ebbf.unfiltered.h5ad")
BOUNDARIES_PATH = Path("cell_boundaries_ebbf.parquet")
OUTPUT_PATH = Path("unfiltered_denovo_split_transcripts.pdf")


def main() -> None:
    adata = ad.read_h5ad(ADATA_PATH)
    obs = adata.obs.copy()
    obs["ef_cluster_id"] = obs["ef_cluster_id"].astype("Int64").astype(str)

    boundaries = pd.read_parquet(BOUNDARIES_PATH)
    boundaries["ef_cluster_id"] = boundaries["ef_cluster_id"].astype(str)
    boundaries["geometry"] = from_wkb(boundaries["geometry"].to_numpy())
    boundaries_gdf = gpd.GeoDataFrame(boundaries, geometry="geometry", crs=None)

    original_obs = obs.loc[obs["cell_category"].eq("original")].copy()
    original_shapes = boundaries_gdf.merge(
        original_obs[["ef_cluster_id"]],
        on="ef_cluster_id",
        how="inner",
    )

    denovo_split = obs.loc[obs["cell_category"].isin(["denovo", "split"])].copy()
    denovo_split_gdf = gpd.GeoDataFrame(
        denovo_split,
        geometry=gpd.points_from_xy(denovo_split["x_centroid"], denovo_split["y_centroid"]),
        crs=None,
    )

    denovo_split_gdf["log1p_n_tx"] = np.log1p(denovo_split_gdf["n_tx"])

    fig, (ax, hist_ax) = plt.subplots(
        2,
        1,
        figsize=(13, 14),
        gridspec_kw={"height_ratios": [5.0, 1.15]},
        constrained_layout=True,
    )

    original_shapes.plot(
        ax=ax,
        color="#c7cbd1",
        edgecolor="#9aa0a8",
        linewidth=0.12,
        alpha=0.75,
        zorder=1,
    )

    norm = Normalize(
        vmin=float(denovo_split_gdf["log1p_n_tx"].min()),
        vmax=float(denovo_split_gdf["log1p_n_tx"].max()),
    )
    denovo_gdf = denovo_split_gdf.loc[denovo_split_gdf["cell_category"].eq("denovo")]
    split_gdf = denovo_split_gdf.loc[denovo_split_gdf["cell_category"].eq("split")]

    denovo_gdf.plot(
        ax=ax,
        column="log1p_n_tx",
        cmap="viridis",
        norm=norm,
        marker="o",
        markersize=3,
        edgecolor="black",
        linewidth=0.03,
        alpha=0.82,
        legend=True,
        legend_kwds={"label": "log1p(number of transcripts)", "shrink": 0.65},
        zorder=2,
    )
    split_gdf.plot(
        ax=ax,
        column="log1p_n_tx",
        cmap="viridis",
        norm=norm,
        marker="*",
        markersize=18,
        edgecolor="black",
        linewidth=0.08,
        alpha=0.9,
        legend=False,
        zorder=3,
    )

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_axis_off()
    ax.set_title(
        "Unfiltered EBBF: de novo/split cells colored by log transcript count",
        fontsize=14,
        pad=12,
    )

    hist_ax.hist(
        denovo_split_gdf["n_tx"],
        bins=80,
        color="#4c78a8",
        edgecolor="white",
        linewidth=0.35,
    )
    hist_ax.set_xlabel("Number of transcripts (n_tx)")
    hist_ax.set_ylabel("Cells")
    hist_ax.set_title("n_tx distribution for de novo/split cells", fontsize=11, pad=8)
    hist_ax.spines["top"].set_visible(False)
    hist_ax.spines["right"].set_visible(False)
    hist_ax.grid(axis="y", color="#d8dde3", linewidth=0.6, alpha=0.8)

    fig.savefig(OUTPUT_PATH, format="pdf", dpi=300)

    print(f"Wrote {OUTPUT_PATH.resolve()}")
    print(f"Original shapes plotted: {len(original_shapes):,} of {len(original_obs):,}")
    print(f"de novo/split markers plotted: {len(denovo_split_gdf):,}")


if __name__ == "__main__":
    main()
