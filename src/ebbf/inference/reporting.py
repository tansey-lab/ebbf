"""
Reporting pipeline step: generates HTML reports comparing EBBF vs default segmentation.

Reports generated:
1. Boundary visualizations (PDF patches comparing segment boundaries)
2. Distribution comparison report (transcript count distributions)
3. Diff analysis report (UMAP/clustering of per-cell transcript changes)
4. Choropleth clusters report (interactive HTML map of diff clusters on cell boundaries)
"""

import base64
import json
import logging
import warnings
from io import BytesIO
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def _fig_to_base64(fig, dpi=150):
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _load_xenium_default(xenium_dir):
    """Load the default Xenium AnnData table via spatialdata_io."""
    import spatialdata_io

    sd = spatialdata_io.xenium(xenium_dir, cells_as_circles=True)
    return sd


def _compute_diff_adata(adata_ebbf, adata_default):
    """
    Compute a cell x gene diff matrix (EBBF - Default) and run
    PCA, UMAP, and Leiden clustering on it.

    Returns the diff AnnData object with raw_diff layer and clustering results.
    """
    import scanpy as sc
    from anndata import AnnData

    # EBBF original cells only
    ebbf_orig = adata_ebbf[adata_ebbf.obs["cell_category"] == "original"].copy()

    # Match on cell_id
    adata_default = adata_default.copy()
    adata_default.obs.index = adata_default.obs["cell_id"]
    shared_ids = sorted(set(adata_default.obs_names) & set(ebbf_orig.obs_names))
    genes = sorted(set(adata_default.var_names) & set(ebbf_orig.var_names))

    ad_d = adata_default[shared_ids, genes]
    ad_e = ebbf_orig[shared_ids, genes]
    logger.info(f"Diff analysis: {len(shared_ids):,} shared cells, {len(genes)} genes")

    # Compute diff matrix
    X_d = ad_d.X.toarray() if issparse(ad_d.X) else np.array(ad_d.X)
    X_e = ad_e.X.toarray() if issparse(ad_e.X) else np.array(ad_e.X)
    X_diff = X_e.astype(np.float32) - X_d.astype(np.float32)

    # Build AnnData for the diff matrix
    adata_diff = AnnData(X=csr_matrix(X_diff), obs=ad_e.obs.copy(), var=ad_e[:, genes].var.copy())
    adata_diff.obs_names = shared_ids
    adata_diff.var_names = genes

    # Metadata
    adata_diff.obs["default_total"] = X_d.sum(axis=1)
    adata_diff.obs["ebbf_total"] = X_e.sum(axis=1)
    adata_diff.obs["diff_total"] = X_diff.sum(axis=1)
    adata_diff.obs["n_genes_gained"] = (X_diff > 0).sum(axis=1)
    adata_diff.obs["n_genes_lost"] = (X_diff < 0).sum(axis=1)
    adata_diff.obs["abs_diff_total"] = np.abs(X_diff).sum(axis=1)

    # Spatial coords
    if "spatial" in ad_e.obsm:
        adata_diff.obsm["spatial"] = ad_e.obsm["spatial"]

    # Keep raw diffs
    adata_diff.layers["raw_diff"] = adata_diff.X.copy()

    # PCA, UMAP, clustering
    logger.info("Running PCA, UMAP, clustering on diff matrix...")
    sc.pp.scale(adata_diff, max_value=10)
    n_comps = min(30, adata_diff.n_obs, adata_diff.n_vars) - 1
    if n_comps < 2:
        logger.warning(
            f"Too few cells/genes for PCA ({adata_diff.n_obs} cells, {adata_diff.n_vars} genes). "
            "Skipping PCA/UMAP/clustering on diff matrix."
        )
        return adata_diff, X_diff, genes, shared_ids
    logger.info(f"  Using n_comps={n_comps} (cells={adata_diff.n_obs}, genes={adata_diff.n_vars})")
    sc.tl.pca(adata_diff, n_comps=n_comps)
    n_neighbors = min(15, adata_diff.n_obs - 1)
    sc.pp.neighbors(adata_diff, n_neighbors=n_neighbors, n_pcs=n_comps)
    sc.tl.umap(adata_diff)

    for res in [0.25, 0.5, 1.0]:
        sc.tl.leiden(
            adata_diff,
            resolution=res,
            key_added=f"leiden_{res}",
            flavor="igraph",
            n_iterations=2,
        )
        logger.info(f"  Leiden res={res}: {adata_diff.obs[f'leiden_{res}'].nunique()} clusters")

    sc.tl.rank_genes_groups(adata_diff, groupby="leiden_0.5", method="wilcoxon")
    if adata_diff.obs["leiden_0.5"].nunique() > 2:
        sc.tl.dendrogram(adata_diff, groupby="leiden_0.5")

    return adata_diff, X_diff, genes, shared_ids


def generate_distribution_report(adata_ebbf, adata_default, output_dir):
    """
    Generate the distribution comparison HTML report.

    Compares transcript count distributions between Default and EBBF segmentations.
    """
    import scanpy as sc

    output_path = Path(output_dir) / "distribution_report.html"
    logger.info("Generating distribution report...")

    sc.pp.filter_cells(adata_default, min_counts=1)

    # Split EBBF by category
    adata_ebbf_filt = adata_ebbf[adata_ebbf.obs["cell_category"] != "denovo"].copy()
    adata_denovo = adata_ebbf[adata_ebbf.obs["cell_category"] == "denovo"].copy()
    sc.pp.filter_cells(adata_ebbf_filt, min_counts=1)
    sc.pp.filter_cells(adata_denovo, min_counts=1)

    logger.info(
        f"Default: {adata_default.n_obs:,}, EBBF: {adata_ebbf_filt.n_obs:,}, "
        f"De novo: {adata_denovo.n_obs:,}"
    )

    def get_stats(ad):
        counts = np.array(ad.X.sum(axis=1)).flatten()
        genes_detected = np.array((ad.X > 0).sum(axis=1)).flatten()
        return counts, genes_detected

    d_counts, d_genes = get_stats(adata_default)
    e_counts, e_genes = get_stats(adata_ebbf_filt)
    dn_counts, dn_genes = get_stats(adata_denovo)

    images = {}

    # 1. Total counts distributions
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].hist(
        d_counts,
        bins=100,
        alpha=0.5,
        density=True,
        color="#1f77b4",
        label=f"Default (n={len(d_counts):,})",
    )
    axes[0, 0].hist(
        e_counts,
        bins=100,
        alpha=0.5,
        density=True,
        color="#ff7f0e",
        label=f"EBBF (n={len(e_counts):,})",
    )
    axes[0, 0].set_xlabel("Total Transcripts per Cell")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].set_title("Total Transcript Distribution")
    axes[0, 0].legend()
    axes[0, 0].set_xlim(0, np.percentile(np.concatenate([d_counts, e_counts]), 99))

    axes[0, 1].ecdf(d_counts, label="Default", color="#1f77b4")
    axes[0, 1].ecdf(e_counts, label="EBBF", color="#ff7f0e")
    axes[0, 1].set_xlabel("Total Transcripts per Cell")
    axes[0, 1].set_ylabel("Cumulative Fraction")
    axes[0, 1].set_title("CDF of Total Transcripts")
    axes[0, 1].legend()
    axes[0, 1].set_xlim(0, np.percentile(np.concatenate([d_counts, e_counts]), 99))

    axes[1, 0].hist(d_genes, bins=80, alpha=0.5, density=True, color="#1f77b4", label="Default")
    axes[1, 0].hist(e_genes, bins=80, alpha=0.5, density=True, color="#ff7f0e", label="EBBF")
    axes[1, 0].set_xlabel("Genes Detected per Cell")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].set_title("Genes per Cell Distribution")
    axes[1, 0].legend()

    data_box = [d_counts, e_counts]
    bp = axes[1, 1].boxplot(
        data_box,
        labels=["Default", "EBBF"],
        widths=0.5,
        patch_artist=True,
        showfliers=False,
    )
    bp["boxes"][0].set_facecolor("#1f77b4")
    bp["boxes"][0].set_alpha(0.5)
    bp["boxes"][1].set_facecolor("#ff7f0e")
    bp["boxes"][1].set_alpha(0.5)
    axes[1, 1].set_ylabel("Total Transcripts per Cell")
    axes[1, 1].set_title("Transcript Count Comparison")

    plt.suptitle("Default vs EBBF (original+split)", fontsize=15, y=1.01)
    plt.tight_layout()
    images["dist_main"] = _fig_to_base64(fig)

    # 2. De novo cell distributions
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].hist(dn_counts, bins=100, color="#d62728", alpha=0.7, edgecolor="none")
    axes[0].set_xlabel("Total Transcripts per Cell")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"De Novo Cells — Transcript Distribution (n={len(dn_counts):,})")
    axes[0].set_xlim(0, min(np.percentile(dn_counts, 99), 100))

    axes[1].hist(dn_genes, bins=60, color="#d62728", alpha=0.7, edgecolor="none")
    axes[1].set_xlabel("Genes Detected per Cell")
    axes[1].set_ylabel("Count")
    axes[1].set_title("De Novo Cells — Genes per Cell")

    axes[2].ecdf(d_counts, label="Default", color="#1f77b4")
    axes[2].ecdf(e_counts, label="EBBF", color="#ff7f0e")
    axes[2].ecdf(dn_counts, label="De novo", color="#d62728")
    axes[2].set_xlabel("Total Transcripts per Cell")
    axes[2].set_ylabel("Cumulative Fraction")
    axes[2].set_title("CDF — All Three Populations")
    axes[2].legend()
    axes[2].set_xlim(0, np.percentile(np.concatenate([d_counts, e_counts]), 95))
    plt.tight_layout()
    images["dist_denovo"] = _fig_to_base64(fig)

    # 3. Per-gene comparison
    shared_genes = sorted(set(adata_default.var_names) & set(adata_ebbf_filt.var_names))
    d_gene_totals = np.array(adata_default[:, shared_genes].X.sum(axis=0)).flatten()
    e_gene_totals = np.array(adata_ebbf_filt[:, shared_genes].X.sum(axis=0)).flatten()
    dn_gene_totals = np.array(adata_denovo[:, shared_genes].X.sum(axis=0)).flatten()

    d_gene_mean = d_gene_totals / adata_default.n_obs
    e_gene_mean = e_gene_totals / adata_ebbf_filt.n_obs
    dn_gene_mean = dn_gene_totals / adata_denovo.n_obs

    gene_df = pd.DataFrame(
        {
            "gene": shared_genes,
            "default_total": d_gene_totals,
            "ebbf_total": e_gene_totals,
            "denovo_total": dn_gene_totals,
            "default_mean": d_gene_mean,
            "ebbf_mean": e_gene_mean,
            "denovo_mean": dn_gene_mean,
            "diff_total": e_gene_totals - d_gene_totals,
            "diff_mean": e_gene_mean - d_gene_mean,
            "pct_change": (e_gene_totals - d_gene_totals) / np.maximum(d_gene_totals, 1) * 100,
            "fold_change": e_gene_mean / np.maximum(d_gene_mean, 1e-6),
        }
    )

    d_detect = (
        np.array((adata_default[:, shared_genes].X > 0).sum(axis=0)).flatten() / adata_default.n_obs
    )
    e_detect = (
        np.array((adata_ebbf_filt[:, shared_genes].X > 0).sum(axis=0)).flatten()
        / adata_ebbf_filt.n_obs
    )
    gene_df["default_detect"] = d_detect
    gene_df["ebbf_detect"] = e_detect
    gene_df["detect_change"] = e_detect - d_detect

    N_TOP = 30

    # 4. Top changed genes bar chart
    top_changed = gene_df.reindex(gene_df["diff_total"].abs().nlargest(N_TOP * 2).index)
    top_changed = top_changed.sort_values("diff_total", ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(8, len(top_changed) * 0.28)))
    y = np.arange(len(top_changed))
    colors = ["#2ca02c" if d >= 0 else "#d62728" for d in top_changed["diff_total"]]
    ax.barh(y, top_changed["diff_total"].values, color=colors, alpha=0.8, height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(top_changed["gene"].values, fontsize=8)
    ax.set_xlabel("Change in Total Transcripts (EBBF − Default)")
    ax.set_title(f"Top {len(top_changed)} Genes by Absolute Change in Intracellular Transcripts")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i, (_, row) in enumerate(top_changed.iterrows()):
        offset = top_changed["diff_total"].abs().max() * 0.02
        xpos = row["diff_total"] + (offset if row["diff_total"] >= 0 else -offset)
        ha = "left" if row["diff_total"] >= 0 else "right"
        ax.text(
            xpos,
            i,
            f"{row['pct_change']:+.0f}% ({row['diff_total']:+,.0f})",
            fontsize=7,
            va="center",
            ha=ha,
        )
    plt.tight_layout()
    images["gene_bars"] = _fig_to_base64(fig)

    # 5. Per-gene mean scatter
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ax = axes[0]
    ax.scatter(
        gene_df["default_mean"],
        gene_df["ebbf_mean"],
        s=15,
        alpha=0.5,
        c="#2c3e50",
        edgecolors="none",
    )
    lim = max(gene_df["default_mean"].max(), gene_df["ebbf_mean"].max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, linewidth=1)
    ax.set_xlabel("Default — Mean Transcripts/Cell")
    ax.set_ylabel("EBBF — Mean Transcripts/Cell")
    ax.set_title("Per-Gene Mean Expression")

    for _, row in gene_df.nlargest(10, "diff_mean").iterrows():
        ax.annotate(
            row["gene"],
            (row["default_mean"], row["ebbf_mean"]),
            fontsize=7,
            alpha=0.8,
            xytext=(4, 4),
            textcoords="offset points",
        )
    for _, row in gene_df.nsmallest(3, "diff_mean").iterrows():
        ax.annotate(
            row["gene"],
            (row["default_mean"], row["ebbf_mean"]),
            fontsize=7,
            alpha=0.8,
            xytext=(4, -8),
            textcoords="offset points",
        )

    ax = axes[1]
    ax.scatter(
        gene_df["default_detect"] * 100,
        gene_df["ebbf_detect"] * 100,
        s=15,
        alpha=0.5,
        c="#8e44ad",
        edgecolors="none",
    )
    ax.plot([0, 105], [0, 105], "k--", alpha=0.3, linewidth=1)
    ax.set_xlabel("Default — % Cells Detecting Gene")
    ax.set_ylabel("EBBF — % Cells Detecting Gene")
    ax.set_title("Per-Gene Detection Rate")

    for _, row in gene_df.nlargest(8, "detect_change").iterrows():
        ax.annotate(
            row["gene"],
            (row["default_detect"] * 100, row["ebbf_detect"] * 100),
            fontsize=7,
            alpha=0.8,
            xytext=(4, 4),
            textcoords="offset points",
        )
    plt.tight_layout()
    images["gene_scatter"] = _fig_to_base64(fig)

    # 6. Fold change volcano-style plot
    fig, ax = plt.subplots(figsize=(10, 7))
    fc = np.log2(gene_df["fold_change"].clip(lower=0.01))
    total = gene_df["default_total"] + gene_df["ebbf_total"]
    log_total = np.log10(total + 1)

    scatter = ax.scatter(
        fc,
        log_total,
        s=15,
        alpha=0.5,
        c=fc,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        edgecolors="none",
    )
    ax.axvline(0, color="black", linewidth=0.5, alpha=0.3)
    ax.set_xlabel("Log2 Fold Change (EBBF / Default, per-cell mean)")
    ax.set_ylabel("Log10 Total Transcripts")
    ax.set_title("Per-Gene Fold Change: EBBF vs Default")
    plt.colorbar(scatter, ax=ax, label="Log2 FC", shrink=0.8)

    for _, row in gene_df.nlargest(12, "diff_mean").iterrows():
        ax.annotate(
            row["gene"],
            (
                np.log2(max(row["fold_change"], 0.01)),
                np.log10(row["default_total"] + row["ebbf_total"] + 1),
            ),
            fontsize=7,
            alpha=0.8,
            xytext=(5, 3),
            textcoords="offset points",
        )
    for _, row in gene_df.nsmallest(5, "diff_mean").iterrows():
        ax.annotate(
            row["gene"],
            (
                np.log2(max(row["fold_change"], 0.01)),
                np.log10(row["default_total"] + row["ebbf_total"] + 1),
            ),
            fontsize=7,
            alpha=0.8,
            xytext=(5, -8),
            textcoords="offset points",
        )
    plt.tight_layout()
    images["gene_volcano"] = _fig_to_base64(fig)

    # 7. De novo gene composition
    fig, ax = plt.subplots(figsize=(14, 7))
    top_dn = gene_df.nlargest(40, "denovo_total")
    y = np.arange(len(top_dn))
    ax.barh(y, top_dn["denovo_total"].values, color="#d62728", alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(top_dn["gene"].values, fontsize=8)
    ax.set_xlabel("Total Transcripts in De Novo Cells")
    ax.set_title(f"Top 40 Genes in De Novo Cells (n={adata_denovo.n_obs:,})")
    ax.invert_yaxis()
    for i, (_, row) in enumerate(top_dn.iterrows()):
        pct = row["denovo_total"] / dn_gene_totals.sum() * 100
        ax.text(
            row["denovo_total"] + dn_gene_totals.max() * 0.01,
            i,
            f"{pct:.1f}%",
            fontsize=7,
            va="center",
        )
    plt.tight_layout()
    images["denovo_genes"] = _fig_to_base64(fig)

    # Build summary stats table
    stats_rows = []
    for label, counts, genes_det, n in [
        ("Default", d_counts, d_genes, adata_default.n_obs),
        ("EBBF (orig+split)", e_counts, e_genes, adata_ebbf_filt.n_obs),
        ("EBBF (de novo)", dn_counts, dn_genes, adata_denovo.n_obs),
    ]:
        stats_rows.append(
            {
                "Dataset": label,
                "Cells": f"{n:,}",
                "Total Tx": f"{counts.sum():,.0f}",
                "Mean Tx/cell": f"{counts.mean():.1f}",
                "Median Tx/cell": f"{np.median(counts):.0f}",
                "Std Tx/cell": f"{counts.std():.1f}",
                "Min": f"{counts.min():.0f}",
                "Max": f"{counts.max():.0f}",
                "Mean genes/cell": f"{genes_det.mean():.1f}",
                "Median genes/cell": f"{np.median(genes_det):.0f}",
            }
        )

    gene_table = gene_df.sort_values("diff_total", ascending=False).copy()
    gene_table = gene_table[
        [
            "gene",
            "default_total",
            "ebbf_total",
            "denovo_total",
            "diff_total",
            "pct_change",
            "default_mean",
            "ebbf_mean",
            "denovo_mean",
            "default_detect",
            "ebbf_detect",
            "detect_change",
        ]
    ]

    def stats_table_html():
        html = "<table><thead><tr>"
        for col in stats_rows[0]:
            html += f"<th>{col}</th>"
        html += "</tr></thead><tbody>"
        for row in stats_rows:
            html += "<tr>" + "".join(f"<td>{v}</td>" for v in row.values()) + "</tr>"
        html += "</tbody></table>"
        return html

    def gene_table_html(df):
        html = '<table class="gene-table"><thead><tr>'
        headers = [
            "Gene",
            "Default Total",
            "EBBF Total",
            "De Novo Total",
            "Diff",
            "% Change",
            "Default Mean/Cell",
            "EBBF Mean/Cell",
            "De Novo Mean/Cell",
            "Default Detect%",
            "EBBF Detect%",
            "Detect Change",
        ]
        for h in headers:
            html += f"<th>{h}</th>"
        html += "</tr></thead><tbody>"
        for _, row in df.iterrows():
            diff_class = (
                "pos" if row["diff_total"] > 0 else ("neg" if row["diff_total"] < 0 else "")
            )
            html += "<tr>"
            html += f"<td><b>{row['gene']}</b></td>"
            html += f"<td>{row['default_total']:,.0f}</td>"
            html += f"<td>{row['ebbf_total']:,.0f}</td>"
            html += f"<td>{row['denovo_total']:,.0f}</td>"
            html += f'<td class="{diff_class}">{row["diff_total"]:+,.0f}</td>'
            html += f'<td class="{diff_class}">{row["pct_change"]:+.1f}%</td>'
            html += f"<td>{row['default_mean']:.2f}</td>"
            html += f"<td>{row['ebbf_mean']:.2f}</td>"
            html += f"<td>{row['denovo_mean']:.2f}</td>"
            html += f"<td>{row['default_detect'] * 100:.1f}%</td>"
            html += f"<td>{row['ebbf_detect'] * 100:.1f}%</td>"
            html += f'<td class="{diff_class}">{row["detect_change"] * 100:+.1f}%</td>'
            html += "</tr>"
        html += "</tbody></table>"
        return html

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Distribution Comparison — Default vs EBBF</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
         max-width: 1500px; margin: 0 auto; padding: 20px; background: #fafafa; color: #333; }}
  h1 {{ border-bottom: 3px solid #2c3e50; padding-bottom: 10px; }}
  h2 {{ color: #1a5276; margin-top: 40px; border-left: 4px solid #1a5276; padding-left: 12px; }}
  .section {{ background: white; border-radius: 8px; padding: 24px; margin: 20px 0;
              box-shadow: 0 2px 6px rgba(0,0,0,0.08); }}
  img {{ max-width: 100%; height: auto; }}
  table {{ border-collapse: collapse; font-size: 12px; width: 100%; }}
  th {{ background: #2c3e50; color: white; padding: 8px 10px; text-align: center;
       position: sticky; top: 0; z-index: 1; cursor: pointer; }}
  td {{ border: 1px solid #ddd; padding: 5px 8px; text-align: center; }}
  tbody tr:nth-child(even) {{ background: #f8f9fa; }}
  tbody tr:hover {{ background: #e8f4f8; }}
  .gene-table {{ font-size: 11px; }}
  .gene-table td:first-child {{ text-align: left; }}
  .pos {{ color: #27ae60; font-weight: bold; }}
  .neg {{ color: #e74c3c; font-weight: bold; }}
  .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
              gap: 12px; margin-bottom: 20px; }}
  .stat-card {{ background: #ecf0f1; border-radius: 6px; padding: 15px; text-align: center; }}
  .stat-card .value {{ font-size: 28px; font-weight: bold; color: #2c3e50; }}
  .stat-card .label {{ font-size: 12px; color: #777; margin-top: 4px; }}
</style>
</head><body>

<h1>Transcript Distribution Comparison</h1>
<p><b>Comparison:</b> Default segmentation vs EBBF resegmentation (original+split cells).
De novo cells ({adata_denovo.n_obs:,}) shown separately.</p>

<div class="section">
<h2>Summary Statistics</h2>
<div class="summary">
  <div class="stat-card"><div class="value">{adata_default.n_obs:,}</div><div class="label">Default Cells</div></div>
  <div class="stat-card"><div class="value">{d_counts.sum():,.0f}</div><div class="label">Default Total Tx</div></div>
  <div class="stat-card"><div class="value">{adata_ebbf_filt.n_obs:,}</div><div class="label">EBBF Cells</div></div>
  <div class="stat-card"><div class="value">{e_counts.sum():,.0f}</div><div class="label">EBBF Total Tx</div></div>
  <div class="stat-card"><div class="value">{adata_denovo.n_obs:,}</div><div class="label">De Novo Cells</div></div>
  <div class="stat-card"><div class="value">{dn_counts.sum():,.0f}</div><div class="label">De Novo Total Tx</div></div>
</div>
{stats_table_html()}
</div>

<div class="section">
<h2>Cell-Level Transcript Distributions</h2>
<p>Comparison of total transcripts per cell and genes detected per cell between Default and EBBF (original+split) segmentations.</p>
<img src="data:image/png;base64,{images["dist_main"]}">
</div>

<div class="section">
<h2>De Novo Cell Distributions</h2>
<p>Distribution of transcripts in the {adata_denovo.n_obs:,} de novo cells created by EBBF resegmentation.
These cells have substantially fewer transcripts (median {np.median(dn_counts):.0f}) compared to
original cells (Default median {np.median(d_counts):.0f}, EBBF median {np.median(e_counts):.0f}).</p>
<img src="data:image/png;base64,{images["dist_denovo"]}">
</div>

<div class="section">
<h2>Per-Gene Comparison — Scatter Plots</h2>
<p>Left: mean transcripts per cell for each gene. Right: detection rate (% of cells with &gt;0 counts).
Points above the diagonal are higher in EBBF.</p>
<img src="data:image/png;base64,{images["gene_scatter"]}">
</div>

<div class="section">
<h2>Per-Gene Fold Change</h2>
<p>Log2 fold change of per-cell mean expression (EBBF / Default) vs total abundance.
Red = increased in EBBF, blue = decreased.</p>
<img src="data:image/png;base64,{images["gene_volcano"]}">
</div>

<div class="section">
<h2>Per-Gene Change in Intracellular Transcripts (EBBF − Default)</h2>
<p>Green = more transcripts assigned in EBBF, red = fewer. Annotated with absolute change and % change.</p>
<img src="data:image/png;base64,{images["gene_bars"]}">
</div>

<div class="section">
<h2>Top Genes in De Novo Cells</h2>
<p>Which genes make up the de novo cell transcriptomes.</p>
<img src="data:image/png;base64,{images["denovo_genes"]}">
</div>

<div class="section">
<h2>Full Gene-by-Gene Table</h2>
<p>All {len(gene_table)} genes sorted by total transcript change (EBBF vs Default).
Includes de novo cell counts and detection rates.</p>
<div style="overflow-x:auto; max-height:800px; overflow-y:auto;">
{gene_table_html(gene_table)}
</div>
</div>

</body></html>"""

    output_path.write_text(html)
    logger.info(f"Saved {output_path}")


def generate_diff_report(adata_diff, X_diff, genes, shared_ids, output_dir):
    """
    Generate the diff analysis HTML report from a pre-computed diff AnnData.
    """
    import scanpy as sc

    output_path = Path(output_dir) / "diff_analysis.html"
    logger.info("Generating diff analysis report...")

    images = {}
    n_clusters = adata_diff.obs["leiden_0.5"].nunique()

    # 1. UMAP overview
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    sc.pl.umap(
        adata_diff,
        color="diff_total",
        ax=axes[0],
        show=False,
        frameon=True,
        title="Net Transcript Gain/Loss",
        cmap="RdBu_r",
        vcenter=0,
        size=4,
        alpha=0.5,
    )
    sc.pl.umap(
        adata_diff,
        color="abs_diff_total",
        ax=axes[1],
        show=False,
        frameon=True,
        title="Absolute Change (magnitude)",
        size=4,
        alpha=0.5,
    )
    sc.pl.umap(
        adata_diff,
        color="default_total",
        ax=axes[2],
        show=False,
        frameon=True,
        title="Original Total Counts",
        size=4,
        alpha=0.5,
    )
    plt.tight_layout()
    images["umap_overview"] = _fig_to_base64(fig)

    # 2. UMAP clusters
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    for i, res in enumerate([0.25, 0.5, 1.0]):
        key = f"leiden_{res}"
        n = adata_diff.obs[key].nunique()
        sc.pl.umap(
            adata_diff,
            color=key,
            ax=axes[i],
            show=False,
            frameon=True,
            title=f"Leiden res={res} ({n} clusters)",
            legend_loc="on data",
            legend_fontsize=7,
            size=4,
            alpha=0.5,
        )
    plt.tight_layout()
    images["umap_clusters"] = _fig_to_base64(fig)

    # 3. Dotplot and heatmap
    dp = sc.pl.rank_genes_groups_dotplot(
        adata_diff,
        n_genes=5,
        groupby="leiden_0.5",
        show=False,
        return_fig=True,
    )
    dp.make_figure()
    images["dotplot"] = _fig_to_base64(plt.gcf())
    plt.close("all")

    sc.pl.rank_genes_groups_heatmap(
        adata_diff,
        n_genes=5,
        groupby="leiden_0.5",
        show=False,
        show_gene_labels=True,
        figsize=(14, 8),
    )
    images["heatmap"] = _fig_to_base64(plt.gcf())
    plt.close("all")

    # 4. Per-cluster gene heatmap
    fig, ax = plt.subplots(figsize=(14, 6))
    key_genes = [
        "IGKC",
        "JCHAIN",
        "XBP1",
        "IRF4",
        "TNFRSF17",
        "SDC1",
        "CCR2",
        "SPARC",
        "LUM",
        "FN1",
        "CTSD",
        "APOE",
        "CD8A",
        "GZMA",
        "ACTB",
    ]
    key_genes = [g for g in key_genes if g in genes]
    # Fall back to top marker genes if none of the hardcoded genes are present
    if not key_genes and "rank_genes_groups" in adata_diff.uns:
        result = adata_diff.uns["rank_genes_groups"]
        seen = set()
        for g in result["names"].dtype.names:
            for name in result["names"][g][:5]:
                if name not in seen:
                    key_genes.append(name)
                    seen.add(name)
        key_genes = key_genes[:15]

    raw_diff = adata_diff.layers["raw_diff"]
    if issparse(raw_diff):
        raw_diff = raw_diff.toarray()

    clusters = sorted(adata_diff.obs["leiden_0.5"].unique(), key=int)

    if not key_genes:
        logger.warning("No key genes available for cluster gene heatmap, skipping.")
        plt.close(fig)
    else:
        cluster_means = []
        for cl in clusters:
            mask = (adata_diff.obs["leiden_0.5"] == cl).values
            means = raw_diff[mask][:, [genes.index(g) for g in key_genes]].mean(axis=0)
            cluster_means.append(means)

        cm = np.array(cluster_means)
        abs_cm = np.abs(cm)
        vbound = np.percentile(abs_cm, 95) if abs_cm.size > 0 else 1.0
        im = ax.imshow(
            cm.T,
            cmap="RdBu_r",
            aspect="auto",
            vmin=-vbound,
            vmax=vbound,
        )
        ax.set_xticks(range(len(clusters)))
        ax.set_xticklabels(
            [f"Cluster {c}\n(n={(adata_diff.obs['leiden_0.5'] == c).sum():,})" for c in clusters],
            fontsize=8,
        )
        ax.set_yticks(range(len(key_genes)))
        ax.set_yticklabels(key_genes, fontsize=9)
        ax.set_title("Mean Transcript Gain/Loss per Cluster (key genes)", fontsize=13)
        plt.colorbar(im, ax=ax, label="Mean diff (EBBF − Default)", shrink=0.8)
        plt.tight_layout()
        images["cluster_gene_heatmap"] = _fig_to_base64(fig)

    # 5. Per-cluster stats
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    cluster_data = [
        adata_diff.obs.loc[adata_diff.obs["leiden_0.5"] == c, "diff_total"].values for c in clusters
    ]
    bp = axes[0].boxplot(cluster_data, labels=clusters, showfliers=False, patch_artist=True)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(plt.cm.tab10(i / max(len(clusters) - 1, 1)))
        patch.set_alpha(0.6)
    axes[0].set_xlabel("Cluster")
    axes[0].set_ylabel("Net Transcript Change")
    axes[0].set_title("Net Gain/Loss Distribution per Cluster")
    axes[0].axhline(0, color="black", linewidth=0.5, linestyle="--")

    sizes = [len(d) for d in cluster_data]
    colors = [plt.cm.tab10(i / max(len(clusters) - 1, 1)) for i in range(len(clusters))]
    axes[1].bar(range(len(clusters)), sizes, color=colors, alpha=0.7)
    axes[1].set_xticks(range(len(clusters)))
    axes[1].set_xticklabels(clusters)
    axes[1].set_xlabel("Cluster")
    axes[1].set_ylabel("Cells")
    axes[1].set_title("Cluster Sizes")
    for i, s in enumerate(sizes):
        axes[1].text(i, s + 50, f"{s:,}", ha="center", fontsize=8)
    plt.tight_layout()
    images["cluster_stats"] = _fig_to_base64(fig)

    # 6. Spatial plot
    spatial_section = ""
    if adata_diff.obsm.get("spatial") is not None:
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        coords = adata_diff.obsm["spatial"]

        sc_obj = axes[0].scatter(
            coords[:, 0],
            coords[:, 1],
            s=1,
            alpha=0.4,
            c=adata_diff.obs["diff_total"],
            cmap="RdBu_r",
            vmin=-np.percentile(np.abs(adata_diff.obs["diff_total"]), 95),
            vmax=np.percentile(np.abs(adata_diff.obs["diff_total"]), 95),
            rasterized=True,
        )
        plt.colorbar(sc_obj, ax=axes[0], label="Net diff", shrink=0.7)
        axes[0].set_title("Spatial — Net Transcript Change")
        axes[0].set_aspect("equal")
        axes[0].invert_yaxis()

        for cl in clusters:
            mask = (adata_diff.obs["leiden_0.5"] == cl).values
            axes[1].scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=1,
                alpha=0.4,
                label=f"{cl} ({mask.sum():,})",
                rasterized=True,
            )
        axes[1].set_title("Spatial — Diff Clusters (res=0.5)")
        axes[1].set_aspect("equal")
        axes[1].invert_yaxis()
        axes[1].legend(markerscale=6, fontsize=8, loc="upper right")
        plt.tight_layout()
        images["spatial"] = _fig_to_base64(fig)

        spatial_section = f"""
<div class="section">
<h2>Spatial Distribution of Diff Clusters</h2>
<p>Left: net transcript change mapped onto tissue coordinates. Right: diff clusters in tissue space.</p>
<img src="data:image/png;base64,{images["spatial"]}">
</div>"""

    # 7. Marker gene table
    result = adata_diff.uns["rank_genes_groups"]
    groups = result["names"].dtype.names
    n_marker_genes = min(15, len(result["names"][groups[0]])) if groups else 0
    marker_rows = []
    for i in range(n_marker_genes):
        row = {}
        for g in groups:
            row[f"{g}_gene"] = result["names"][g][i]
            row[f"{g}_score"] = f"{result['scores'][g][i]:.1f}"
        marker_rows.append(row)

    def marker_table_html():
        html = '<table class="marker-table"><thead><tr>'
        for g in groups:
            html += f'<th colspan="2">Cluster {g}</th>'
        html += "</tr><tr>"
        for _ in groups:
            html += "<th>Gene</th><th>Score</th>"
        html += "</tr></thead><tbody>"
        for row in marker_rows:
            html += "<tr>"
            for g in groups:
                html += f"<td><b>{row[f'{g}_gene']}</b></td><td>{row[f'{g}_score']}</td>"
            html += "</tr>"
        html += "</tbody></table>"
        return html

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Transcript Diff Analysis</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
         max-width: 1500px; margin: 0 auto; padding: 20px; background: #fafafa; color: #333; }}
  h1 {{ border-bottom: 3px solid #2c3e50; padding-bottom: 10px; }}
  h2 {{ color: #1a5276; margin-top: 40px; border-left: 4px solid #1a5276; padding-left: 12px; }}
  .section {{ background: white; border-radius: 8px; padding: 24px; margin: 20px 0;
              box-shadow: 0 2px 6px rgba(0,0,0,0.08); }}
  img {{ max-width: 100%; height: auto; }}
  table {{ border-collapse: collapse; font-size: 12px; width: 100%; }}
  th {{ background: #2c3e50; color: white; padding: 8px; text-align: center; }}
  td {{ border: 1px solid #ddd; padding: 5px 8px; text-align: center; }}
  tbody tr:nth-child(even) {{ background: #f8f9fa; }}
  .marker-table {{ font-size: 11px; }}
  .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
              gap: 12px; }}
  .stat-card {{ background: #ecf0f1; border-radius: 6px; padding: 15px; text-align: center; }}
  .stat-card .value {{ font-size: 28px; font-weight: bold; color: #2c3e50; }}
  .stat-card .label {{ font-size: 12px; color: #777; margin-top: 4px; }}
</style>
</head><body>

<h1>Transcript Gain/Loss Diff Analysis</h1>
<p>Each cell's entry is (EBBF count − Default count) per gene. UMAP and clustering performed on
this diff matrix to find patterns of transcript reassignment across {len(shared_ids):,} shared cells.</p>

<div class="section">
<h2>Overview</h2>
<div class="summary">
  <div class="stat-card"><div class="value">{len(shared_ids):,}</div><div class="label">Shared Cells</div></div>
  <div class="stat-card"><div class="value">{len(genes)}</div><div class="label">Genes</div></div>
  <div class="stat-card"><div class="value">{(X_diff.sum(axis=1) > 0).sum():,}</div><div class="label">Cells with Net Gain</div></div>
  <div class="stat-card"><div class="value">{(X_diff.sum(axis=1) < 0).sum():,}</div><div class="label">Cells with Net Loss</div></div>
  <div class="stat-card"><div class="value">{n_clusters}</div><div class="label">Diff Clusters (res=0.5)</div></div>
  <div class="stat-card"><div class="value">{X_diff.sum(axis=1).mean():+.1f}</div><div class="label">Mean Net Change/Cell</div></div>
</div>
</div>

<div class="section">
<h2>UMAP of Diff Matrix</h2>
<p>Cells embedded by their pattern of transcript gain/loss. Colored by net change, magnitude, and original count.</p>
<img src="data:image/png;base64,{images["umap_overview"]}">
</div>

<div class="section">
<h2>Diff Clusters</h2>
<p>Leiden clustering of the diff matrix at multiple resolutions. Clusters represent groups of cells
with similar patterns of transcript reassignment.</p>
<img src="data:image/png;base64,{images["umap_clusters"]}">
</div>

<div class="section">
<h2>Cluster Characterization</h2>
<p>What genes define each diff cluster? This heatmap shows mean gain/loss per gene per cluster.</p>
{'<img src="data:image/png;base64,' + images["cluster_gene_heatmap"] + '">' if "cluster_gene_heatmap" in images else "<p><em>Gene heatmap not available (no matching key genes).</em></p>"}
<br>
<img src="data:image/png;base64,{images["cluster_stats"]}">
</div>

{spatial_section}

<div class="section">
<h2>Marker Genes of Diff Clusters (Dotplot)</h2>
<p>Top 5 genes whose diff values best distinguish each cluster (Wilcoxon test on the diff matrix).</p>
<img src="data:image/png;base64,{images["dotplot"]}">
</div>

<div class="section">
<h2>Marker Genes Heatmap</h2>
<img src="data:image/png;base64,{images["heatmap"]}">
</div>

<div class="section">
<h2>Top 15 Marker Genes per Diff Cluster</h2>
<div style="overflow-x:auto;">{marker_table_html()}</div>
</div>

</body></html>"""

    output_path.write_text(html)
    logger.info(f"Saved {output_path}")


def generate_choropleth_report(adata_diff, gdf_boundaries, output_dir):
    """
    Generate an interactive HTML/JS choropleth of diff clusters on cell boundaries.
    """
    output_path = Path(output_dir) / "choropleth_clusters.html"
    logger.info("Generating choropleth clusters report...")

    cell_ids = list(adata_diff.obs_names)
    n_cells = len(cell_ids)
    genes = list(adata_diff.var_names)

    clusters = adata_diff.obs["leiden_0.5"].values.astype(str).tolist()
    unique_clusters = sorted(set(clusters), key=int)

    raw_diff = adata_diff.layers["raw_diff"]
    if issparse(raw_diff):
        raw_diff = raw_diff.toarray()

    # Extract polygon coordinates
    logger.info("Extracting polygons...")
    polys = []
    valid_mask = []
    for cid in cell_ids:
        if cid in gdf_boundaries.index:
            coords = list(gdf_boundaries.loc[cid, "geometry"].exterior.coords)
            polys.append([[round(x, 2), round(y, 2)] for x, y in coords])
            valid_mask.append(True)
        else:
            polys.append([[0, 0]])
            valid_mask.append(False)

    all_x = [c[0] for i, p in enumerate(polys) if valid_mask[i] for c in p]
    all_y = [c[1] for i, p in enumerate(polys) if valid_mask[i] for c in p]
    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)

    # Build sparse diff data for gene mode
    logger.info("Building sparse diff data...")
    gene_diffs = {}
    for gi, gene in enumerate(genes):
        col = raw_diff[:, gi]
        nonzero = {int(i): int(col[i]) for i in np.where(col != 0)[0]}
        if nonzero:
            gene_diffs[gene] = nonzero

    total_diff = raw_diff.sum(axis=1)
    gene_diffs["__TOTAL__"] = {
        int(i): int(total_diff[i]) for i in range(n_cells) if total_diff[i] != 0
    }

    gene_abs_mean = np.abs(raw_diff).mean(axis=0)
    gene_order = ["__TOTAL__"] + [genes[i] for i in np.argsort(-gene_abs_mean)]

    cluster_info = {}
    for cl in unique_clusters:
        mask = np.array(clusters) == cl
        top_genes = []
        for gi in np.argsort(-raw_diff[mask].mean(axis=0))[:5]:
            mean_val = raw_diff[mask, gi].mean()
            top_genes.append(f"{genes[gi]}: {mean_val:+.1f}")
        cluster_info[cl] = {
            "n": int(mask.sum()),
            "mean_diff": float(total_diff[mask].mean()),
            "top": top_genes,
        }

    poly_json = json.dumps(polys)
    diff_json = json.dumps(gene_diffs)
    logger.info(
        f"Polygon data: {len(poly_json) / 1e6:.1f} MB, Diff data: {len(diff_json) / 1e6:.1f} MB"
    )

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Diff Cluster Choropleth</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
         background: #1a1a2e; color: #eee; overflow: hidden; }}
  #controls {{ position: fixed; top: 0; left: 0; right: 0; z-index: 10;
               background: rgba(26,26,46,0.95); padding: 10px 20px;
               display: flex; align-items: center; gap: 15px; flex-wrap: wrap;
               border-bottom: 1px solid #333; }}
  #controls label {{ font-size: 13px; color: #aaa; }}
  #controls select, #controls input {{ font-size: 13px; padding: 4px 8px;
               background: #2a2a4a; color: #eee; border: 1px solid #444; border-radius: 4px; }}
  .btn {{ padding: 5px 12px; background: #3a3a6a; color: #eee; border: 1px solid #555;
          border-radius: 4px; cursor: pointer; font-size: 12px; }}
  .btn.active {{ background: #5a5aaa; border-color: #88f; }}
  .btn:hover {{ background: #4a4a8a; }}
  #stats {{ font-size: 12px; color: #ccc; margin-left: auto; }}
  canvas {{ display: block; cursor: grab; }}
  canvas:active {{ cursor: grabbing; }}
  #tooltip {{ position: fixed; pointer-events: none; background: rgba(0,0,0,0.9);
              color: #fff; padding: 8px 12px; border-radius: 4px; font-size: 12px;
              display: none; z-index: 20; border: 1px solid #555; max-width: 300px; }}
  #legend {{ position: fixed; bottom: 20px; right: 20px; background: rgba(26,26,46,0.9);
             padding: 12px; border-radius: 6px; border: 1px solid #444; z-index: 10;
             font-size: 11px; max-height: 80vh; overflow-y: auto; }}
  #cell-panel {{ position: fixed; top: 50px; right: 0; width: 320px; height: calc(100vh - 50px);
                 background: rgba(20,20,40,0.97); border-left: 1px solid #444; z-index: 15;
                 display: none; flex-direction: column; overflow: hidden; }}
  #cell-panel.open {{ display: flex; }}
  #cell-panel-header {{ padding: 12px 15px; border-bottom: 1px solid #444;
                        display: flex; justify-content: space-between; align-items: center; }}
  #cell-panel-header h3 {{ font-size: 14px; margin: 0; }}
  #cell-panel-close {{ background: none; border: none; color: #aaa; font-size: 18px;
                       cursor: pointer; padding: 2px 6px; }}
  #cell-panel-close:hover {{ color: #fff; }}
  #cell-panel-body {{ overflow-y: auto; flex: 1; padding: 0; }}
  #cell-panel-summary {{ padding: 10px 15px; font-size: 12px; color: #bbb;
                         border-bottom: 1px solid #333; }}
  #cell-panel-table {{ width: 100%; border-collapse: collapse; font-size: 11px; }}
  #cell-panel-table th {{ background: #2a2a4a; color: #ccc; padding: 6px 8px;
                          text-align: left; position: sticky; top: 0; z-index: 1; }}
  #cell-panel-table td {{ padding: 4px 8px; border-bottom: 1px solid #2a2a3a; }}
  #cell-panel-table tr:hover {{ background: rgba(255,255,255,0.05); }}
  .diff-pos {{ color: #2ecc71; }}
  .diff-neg {{ color: #e74c3c; }}
  .diff-zero {{ color: #555; }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; margin: 3px 0; cursor: pointer; }}
  .legend-item:hover {{ opacity: 0.8; }}
  .legend-swatch {{ width: 14px; height: 14px; border-radius: 3px; flex-shrink: 0; }}
  #legend canvas {{ border-radius: 3px; }}
  #legend-labels {{ display: flex; justify-content: space-between; font-size: 11px;
                    color: #aaa; margin-top: 4px; }}
</style>
</head><body>

<div id="controls">
  <button class="btn active" id="btn-cluster" onclick="setMode('cluster')">Clusters</button>
  <button class="btn" id="btn-gene" onclick="setMode('gene')">Gene Diff</button>
  <select id="gene-select" style="display:none;"></select>
  <label style="display:none;" id="scale-wrap">Scale:
    <input type="range" id="scale-slider" min="1" max="100" value="20" style="width:100px;">
    <span id="scale-label">20</span>
  </label>
  <span style="font-size:12px;color:#666;">Scroll=zoom, drag=pan</span>
  <span id="stats"></span>
</div>

<canvas id="map"></canvas>
<div id="tooltip"></div>
<div id="legend"></div>

<div id="cell-panel">
  <div id="cell-panel-header">
    <h3 id="cell-panel-title">Cell Detail</h3>
    <button id="cell-panel-close" onclick="closePanel()">&times;</button>
  </div>
  <div id="cell-panel-summary"></div>
  <div id="cell-panel-body">
    <table id="cell-panel-table">
      <thead><tr><th>Gene</th><th>Default</th><th>EBBF</th><th>Diff</th></tr></thead>
      <tbody id="cell-panel-tbody"></tbody>
    </table>
  </div>
</div>

<script>
const POLYS = {poly_json};
const DIFFS = {diff_json};
const GENE_ORDER = {json.dumps(gene_order)};
const CELL_IDS = {json.dumps(cell_ids)};
const CLUSTERS = {json.dumps(clusters)};
const UNIQUE_CLUSTERS = {json.dumps(unique_clusters)};
const CLUSTER_INFO = {json.dumps(cluster_info)};
const VALID = {json.dumps(valid_mask)};
const GENES = {json.dumps(genes)};
const BOUNDS = {{ xmin: {xmin}, xmax: {xmax}, ymin: {ymin}, ymax: {ymax} }};
const N = POLYS.length;

let mode = "cluster";
let currentGene = "__TOTAL__";
let currentDiffs = new Int32Array(N);
let maxScale = 20;
let transform = {{ x: 0, y: 0, scale: 1 }};
let isDragging = false, dragStart = {{ x: 0, y: 0 }};
let hoverCell = -1;
let selectedCell = -1;
let highlightCluster = null;
let dragMoved = false;

const canvas = document.getElementById("map");
const ctx = canvas.getContext("2d");
const tooltip = document.getElementById("tooltip");
const statsEl = document.getElementById("stats");
const legendEl = document.getElementById("legend");

const TAB20 = [
  "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f",
  "#bcbd22","#17becf","#aec7e8","#ffbb78","#98df8a","#ff9896","#c5b0d5","#c49c94",
  "#f7b6d2","#c7c7c7","#dbdb8d","#9edae5","#393b79","#5254a3","#6b6ecf","#9c9ede"
];
const clusterColor = {{}};
UNIQUE_CLUSTERS.forEach((c, i) => {{ clusterColor[c] = TAB20[i % TAB20.length]; }});

const select = document.getElementById("gene-select");
GENE_ORDER.forEach(g => {{
  const opt = document.createElement("option");
  opt.value = g;
  opt.textContent = g === "__TOTAL__" ? "TOTAL (all genes)" : g;
  select.appendChild(opt);
}});

function setMode(m) {{
  mode = m;
  document.getElementById("btn-cluster").classList.toggle("active", m === "cluster");
  document.getElementById("btn-gene").classList.toggle("active", m === "gene");
  document.getElementById("gene-select").style.display = m === "gene" ? "" : "none";
  document.getElementById("scale-wrap").style.display = m === "gene" ? "" : "none";
  highlightCluster = null;
  if (m === "gene") loadGene(select.value);
  updateLegend();
  render();
}}

function loadGene(gene) {{
  currentGene = gene;
  const sparse = DIFFS[gene] || {{}};
  currentDiffs = new Int32Array(N);
  for (const [idx, val] of Object.entries(sparse)) currentDiffs[parseInt(idx)] = val;
  const absVals = [];
  for (let i = 0; i < N; i++) if (currentDiffs[i] !== 0) absVals.push(Math.abs(currentDiffs[i]));
  absVals.sort((a, b) => a - b);
  maxScale = absVals.length > 0 ? Math.max(Math.ceil(absVals[Math.floor(absVals.length * 0.95)]), 1) : 1;
  document.getElementById("scale-slider").value = maxScale;
  document.getElementById("scale-label").textContent = maxScale;
  let gained = 0, lost = 0, nCh = 0;
  for (let i = 0; i < N; i++) {{
    if (currentDiffs[i] > 0) {{ gained += currentDiffs[i]; nCh++; }}
    else if (currentDiffs[i] < 0) {{ lost += currentDiffs[i]; nCh++; }}
  }}
  statsEl.textContent = `${{nCh.toLocaleString()}} changed | +${{gained.toLocaleString()}} / ${{lost.toLocaleString()}}`;
  updateLegend();
  render();
}}

function diffColor(val) {{
  if (val === 0) return "rgba(40,40,60,0.6)";
  const t = Math.min(Math.abs(val) / maxScale, 1);
  if (val > 0) return `rgba(${{30+Math.round((1-t)*30)}},${{80+Math.round(t*175)}},${{30+Math.round((1-t)*30)}},0.8)`;
  return `rgba(${{80+Math.round(t*175)}},${{30+Math.round((1-t)*30)}},${{30+Math.round((1-t)*30)}},0.8)`;
}}

function updateLegend() {{
  legendEl.innerHTML = "";
  if (mode === "cluster") {{
    UNIQUE_CLUSTERS.forEach(c => {{
      const info = CLUSTER_INFO[c];
      const div = document.createElement("div");
      div.className = "legend-item";
      div.innerHTML = `<div class="legend-swatch" style="background:${{clusterColor[c]}}"></div>` +
        `<span><b>${{c}}</b> (n=${{info.n.toLocaleString()}}, mean ${{info.mean_diff > 0 ? "+" : ""}}${{info.mean_diff.toFixed(0)}})</span>`;
      div.onclick = () => {{ highlightCluster = highlightCluster === c ? null : c; render(); }};
      legendEl.appendChild(div);
    }});
  }} else {{
    const lc = document.createElement("canvas");
    lc.width = 200; lc.height = 15;
    const lctx = lc.getContext("2d");
    for (let i = 0; i < 200; i++) {{
      lctx.fillStyle = diffColor(((i/200)*2-1)*maxScale);
      lctx.fillRect(i, 0, 1, 15);
    }}
    legendEl.appendChild(lc);
    const labels = document.createElement("div");
    labels.id = "legend-labels";
    labels.innerHTML = `<span>-${{maxScale}}</span><span>0</span><span>+${{maxScale}}</span>`;
    legendEl.appendChild(labels);
  }}
}}

function resize() {{
  canvas.width = window.innerWidth; canvas.height = window.innerHeight;
  const w = canvas.width, h = canvas.height - 50;
  const dW = BOUNDS.xmax - BOUNDS.xmin, dH = BOUNDS.ymax - BOUNDS.ymin;
  const s = Math.min(w / dW, h / dH) * 0.9;
  transform.scale = s;
  transform.x = (w - dW * s) / 2 - BOUNDS.xmin * s;
  transform.y = (h - dH * s) / 2 - BOUNDS.ymin * s + 50;
  render();
}}

function render() {{
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  const s = transform.scale, tx = transform.x, ty = transform.y;
  const vxmin = -tx/s - 20, vxmax = (w-tx)/s + 20;
  const vymin = -ty/s - 20, vymax = (h-ty)/s + 20;

  for (let i = 0; i < N; i++) {{
    if (!VALID[i]) continue;
    const poly = POLYS[i];
    const x0 = poly[0][0], y0 = poly[0][1];
    if (x0 < vxmin || x0 > vxmax || y0 < vymin || y0 > vymax) continue;

    ctx.beginPath();
    ctx.moveTo(poly[0][0]*s+tx, poly[0][1]*s+ty);
    for (let j = 1; j < poly.length; j++) ctx.lineTo(poly[j][0]*s+tx, poly[j][1]*s+ty);
    ctx.closePath();

    if (mode === "cluster") {{
      const cl = CLUSTERS[i];
      const alpha = (highlightCluster && highlightCluster !== cl) ? 0.1 : 0.7;
      ctx.fillStyle = clusterColor[cl];
      ctx.globalAlpha = alpha;
      ctx.fill();
      ctx.globalAlpha = 1;
    }} else {{
      ctx.fillStyle = diffColor(currentDiffs[i]);
      ctx.fill();
    }}

    if (i === selectedCell) {{
      ctx.strokeStyle = "#ffff00"; ctx.lineWidth = 3; ctx.stroke();
    }} else if (s > 2) {{
      ctx.strokeStyle = i === hoverCell ? "#fff" : "rgba(100,100,120,0.2)";
      ctx.lineWidth = i === hoverCell ? 2 : 0.3;
      ctx.stroke();
    }}
  }}
}}

function pointInPoly(x, y, poly) {{
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {{
    const xi = poly[i][0], yi = poly[i][1], xj = poly[j][0], yj = poly[j][1];
    if ((yi > y) !== (yj > y) && x < (xj-xi)*(y-yi)/(yj-yi)+xi) inside = !inside;
  }}
  return inside;
}}

canvas.addEventListener("wheel", e => {{
  e.preventDefault();
  const f = e.deltaY > 0 ? 0.85 : 1.18;
  transform.x = e.clientX - (e.clientX - transform.x) * f;
  transform.y = e.clientY - (e.clientY - transform.y) * f;
  transform.scale *= f;
  render();
}});
canvas.addEventListener("mousedown", e => {{
  isDragging = true; dragMoved = false;
  dragStart.x = e.clientX - transform.x; dragStart.y = e.clientY - transform.y;
}});
canvas.addEventListener("mousemove", e => {{
  if (isDragging) {{
    dragMoved = true;
    transform.x = e.clientX - dragStart.x; transform.y = e.clientY - dragStart.y;
    render(); return;
  }}
  const s = transform.scale, tx = transform.x, ty = transform.y;
  const dx = (e.clientX - tx)/s, dy = (e.clientY - ty)/s;
  let hit = -1;
  for (let i = N - 1; i >= 0; i--) {{ if (VALID[i] && pointInPoly(dx, dy, POLYS[i])) {{ hit = i; break; }} }}
  if (hit !== hoverCell) {{ hoverCell = hit; render(); }}
  if (hit >= 0) {{
    const cl = CLUSTERS[hit];
    const info = CLUSTER_INFO[cl];
    let html = `<b>${{CELL_IDS[hit]}}</b><br>Cluster: <b>${{cl}}</b> (n=${{info.n.toLocaleString()}})`;
    if (mode === "gene") {{
      const v = currentDiffs[hit]; const sign = v > 0 ? "+" : "";
      const gl = currentGene === "__TOTAL__" ? "Total" : currentGene;
      html += `<br>${{gl}}: <b>${{sign}}${{v}}</b>`;
    }}
    html += `<br>Top gains: ${{info.top.slice(0,3).join(", ")}}`;
    html += `<br><i style="color:#888;">Click for full detail</i>`;
    tooltip.style.display = "block";
    tooltip.style.left = (e.clientX+12)+"px"; tooltip.style.top = (e.clientY+12)+"px";
    tooltip.innerHTML = html;
  }} else tooltip.style.display = "none";
}});
canvas.addEventListener("mouseup", e => {{
  if (!dragMoved && hoverCell >= 0) showCellPanel(hoverCell);
  isDragging = false; dragMoved = false;
}});
canvas.addEventListener("mouseleave", () => {{
  isDragging=false; dragMoved=false; tooltip.style.display="none"; hoverCell=-1; render();
}});

function getCellDiffs(cellIdx) {{
  const result = [];
  for (const gene of GENES) {{
    const sparse = DIFFS[gene];
    const val = sparse ? (sparse[cellIdx] || 0) : 0;
    result.push(val);
  }}
  return result;
}}

function showCellPanel(cellIdx) {{
  selectedCell = cellIdx;
  const panel = document.getElementById("cell-panel");
  const cl = CLUSTERS[cellIdx];
  const info = CLUSTER_INFO[cl];

  document.getElementById("cell-panel-title").textContent = CELL_IDS[cellIdx];

  const diffs = getCellDiffs(cellIdx);
  let totalGain = 0, totalLoss = 0, nGained = 0, nLost = 0;
  diffs.forEach(d => {{
    if (d > 0) {{ totalGain += d; nGained++; }}
    else if (d < 0) {{ totalLoss += d; nLost++; }}
  }});
  const netDiff = totalGain + totalLoss;

  document.getElementById("cell-panel-summary").innerHTML =
    `<b>Cluster:</b> ${{cl}}<br>` +
    `<b>Net change:</b> <span class="${{netDiff >= 0 ? 'diff-pos' : 'diff-neg'}}">${{netDiff > 0 ? '+' : ''}}${{netDiff}}</span><br>` +
    `<b>Genes gained:</b> ${{nGained}} (+${{totalGain}}) &nbsp; <b>Lost:</b> ${{nLost}} (${{totalLoss}})`;

  const rows = GENES.map((g, i) => ({{ gene: g, diff: diffs[i] }}));
  rows.sort((a, b) => Math.abs(b.diff) - Math.abs(a.diff));

  const tbody = document.getElementById("cell-panel-tbody");
  tbody.innerHTML = "";
  rows.forEach(r => {{
    if (r.diff === 0) return;
    const tr = document.createElement("tr");
    const cls = r.diff > 0 ? "diff-pos" : (r.diff < 0 ? "diff-neg" : "diff-zero");
    const sign = r.diff > 0 ? "+" : "";
    tr.innerHTML = `<td>${{r.gene}}</td><td></td><td></td><td class="${{cls}}"><b>${{sign}}${{r.diff}}</b></td>`;
    tbody.appendChild(tr);
  }});

  const nZero = rows.filter(r => r.diff === 0).length;
  if (nZero > 0) {{
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="4" style="color:#555;text-align:center;"><i>${{nZero}} genes unchanged</i></td>`;
    tbody.appendChild(tr);
  }}

  panel.classList.add("open");
  render();
}}

function closePanel() {{
  document.getElementById("cell-panel").classList.remove("open");
  selectedCell = -1;
  render();
}}
select.addEventListener("change", () => {{ loadGene(select.value); }});
document.getElementById("scale-slider").addEventListener("input", e => {{
  maxScale = parseInt(e.target.value);
  document.getElementById("scale-label").textContent = maxScale;
  updateLegend(); render();
}});
window.addEventListener("resize", resize);
resize();
updateLegend();
</script>
</body></html>"""

    output_path.write_text(html)
    fsize = output_path.stat().st_size / 1e6
    logger.info(f"Saved {output_path} ({fsize:.1f} MB)")


def generate_morphology_report(adata_ebbf, adata_default, xenium_dir, output_dir):
    """
    Generate a morphology comparison HTML report.

    Compares geometric quality metrics of cell (and nucleus) boundaries between
    the default Xenium segmentation and the subset of EBBF cells classified as
    "original" (those that kept the default boundary, i.e. were not split or
    created de-novo).  The population-level shift in distributions reveals
    whether EBBF preferentially reassigns cells with particular shapes.

    Metrics computed per cell:
      cell_area, cell_perimeter, eccentricity, sphericity, solidity,
      convexity, circularity, compactness (and nucleus equivalents).
    """
    from ebbf.data.formats import has_cell_boundaries, load_cell_boundaries, load_nucleus_boundaries
    from ebbf.morphology import calculate_morphological_statistics

    output_path = Path(output_dir) / "morphology_report.html"
    logger.info("Generating morphology report...")

    if not has_cell_boundaries(xenium_dir):
        logger.warning("  No cell boundaries found in %s — skipping morphology report", xenium_dir)
        return

    logger.info("  Loading cell boundaries from %s...", xenium_dir)
    cell_boundaries = load_cell_boundaries(xenium_dir)

    nucleus_boundaries = None
    try:
        nucleus_boundaries = load_nucleus_boundaries(xenium_dir)
    except Exception:
        logger.info("  No nucleus boundaries found — nucleus metrics will be skipped")

    # --- default segmentation (before) ---
    logger.info("  Computing morphological statistics for default segmentation...")
    try:
        stats_before = calculate_morphological_statistics(
            adata_default, cell_boundaries, nucleus_boundaries
        )
    except Exception as e:
        logger.warning(f"  Failed to compute default morphology stats: {e}")
        return

    # --- EBBF original cells (after) ---
    # Original cells share the same polygon boundaries as the default segmentation.
    # Split / de-novo cells do not have stored polygon boundaries so they are
    # excluded here; their absence is itself informative at the population level.
    logger.info("  Computing morphological statistics for EBBF original cells...")
    adata_ebbf_orig = adata_ebbf[adata_ebbf.obs["cell_category"] == "original"].copy()
    try:
        stats_after = calculate_morphological_statistics(
            adata_ebbf_orig, cell_boundaries, nucleus_boundaries
        )
    except Exception as e:
        logger.warning(f"  Failed to compute EBBF morphology stats: {e}")
        return

    n_before = len(stats_before)
    n_after = len(stats_after)
    n_split = int((adata_ebbf.obs["cell_category"] == "split").sum())
    n_denovo = int((adata_ebbf.obs["cell_category"] == "denovo").sum())
    logger.info(
        f"  Default: {n_before:,} cells | EBBF original: {n_after:,} | "
        f"split: {n_split:,} | de novo: {n_denovo:,}"
    )

    # ------------------------------------------------------------------ #
    # Save raw tables as parquet                                          #
    # ------------------------------------------------------------------ #
    morph_default_path = Path(output_dir) / "morphology_stats_default.parquet"
    morph_ebbf_path = Path(output_dir) / "morphology_stats_ebbf_original.parquet"

    stats_before.rename_axis("cell_id").reset_index().to_parquet(morph_default_path, index=False)
    logger.info(f"  Saved {morph_default_path}")

    stats_after.rename_axis("cell_id").reset_index().to_parquet(morph_ebbf_path, index=False)
    logger.info(f"  Saved {morph_ebbf_path}")

    # ------------------------------------------------------------------ #
    # Build figures                                                        #
    # ------------------------------------------------------------------ #
    images = {}

    BLUE = "#1f77b4"
    ORANGE = "#ff7f0e"
    BINS = 80

    cell_metrics = [
        ("cell_area", "Cell Area (µm²)", False),
        ("cell_perimeter", "Cell Perimeter (µm)", False),
        ("cell_eccentricity", "Eccentricity\n(minor/major axis, 1 = circle)", True),
        ("cell_sphericity", "Sphericity\n(inner/outer radius, 1 = circle)", True),
        ("cell_solidity", "Solidity\n(area / convex hull area, 1 = convex)", True),
        ("cell_convexity", "Convexity\n(hull perim / perim, 1 = convex)", True),
        ("cell_circularity", "Circularity\n(4π·area / hull perim², 1 = circle)", True),
        ("cell_compactness", "Compactness\n(4π·area / perim², 1 = circle)", True),
    ]
    if "cn_ratio" in stats_before.columns:
        cell_metrics.insert(0, ("cn_ratio", "Cell/Nucleus Area Ratio", False))

    nucleus_metrics = [
        ("nucleus_area", "Nucleus Area (µm²)", False),
        ("nucleus_eccentricity", "Nucleus Eccentricity", True),
        ("nucleus_sphericity", "Nucleus Sphericity", True),
        ("nucleus_solidity", "Nucleus Solidity", True),
        ("nucleus_convexity", "Nucleus Convexity", True),
        ("nucleus_circularity", "Nucleus Circularity", True),
        ("nucleus_compactness", "Nucleus Compactness", True),
    ]
    # Only include nucleus metrics if they were computed
    nucleus_metrics = [
        (col, lbl, bounded) for col, lbl, bounded in nucleus_metrics if col in stats_before.columns
    ]

    def _plot_metric_grid(metrics, title, key):
        # Only plot metrics present in both DataFrames
        metrics = [
            (col, lbl, bounded)
            for col, lbl, bounded in metrics
            if col in stats_before.columns and col in stats_after.columns
        ]
        if not metrics:
            return
        ncols = 3
        nrows = (len(metrics) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.8))
        axes = np.array(axes).flatten()

        for ax, (col, label, bounded) in zip(axes, metrics):
            b = stats_before[col].dropna().values
            a = stats_after[col].dropna().values
            if bounded:
                xlim = (0.0, 1.0)
                bins = np.linspace(0, 1, BINS + 1)
            else:
                upper = np.percentile(np.concatenate([b, a]), 99)
                xlim = (0.0, upper)
                bins = np.linspace(0, upper, BINS + 1)

            ax.hist(
                b, bins=bins, alpha=0.55, density=True, color=BLUE, label=f"Default (n={len(b):,})"
            )
            ax.hist(
                a,
                bins=bins,
                alpha=0.55,
                density=True,
                color=ORANGE,
                label=f"EBBF original (n={len(a):,})",
            )
            ax.set_xlim(xlim)
            ax.set_xlabel(label, fontsize=8)
            ax.set_ylabel("Density", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=7)

        # Hide unused axes
        for ax in axes[len(metrics) :]:
            ax.set_visible(False)

        plt.suptitle(title, fontsize=13, y=1.01)
        plt.tight_layout()
        images[key] = _fig_to_base64(fig)

    _plot_metric_grid(cell_metrics, "Cell Shape Metrics — Default vs EBBF Original", "cell_metrics")
    if nucleus_metrics:
        _plot_metric_grid(
            nucleus_metrics, "Nucleus Shape Metrics — Default vs EBBF Original", "nucleus_metrics"
        )

    # Per-metric ECDFs for cell shape quality metrics (bounded 0-1)
    shape_metrics = [
        (col, lbl)
        for col, lbl, bounded in cell_metrics
        if bounded and col in stats_before.columns and col in stats_after.columns
    ]
    if shape_metrics:
        ncols = 3
        nrows = (len(shape_metrics) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.5))
        axes = np.array(axes).flatten()
        for ax, (col, label) in zip(axes, shape_metrics):
            b = np.sort(stats_before[col].dropna().values)
            a = np.sort(stats_after[col].dropna().values)
            ax.plot(b, np.linspace(0, 1, len(b)), color=BLUE, label="Default", linewidth=1.5)
            ax.plot(
                a, np.linspace(0, 1, len(a)), color=ORANGE, label="EBBF original", linewidth=1.5
            )
            ax.set_xlim(0, 1)
            ax.set_xlabel(label, fontsize=8)
            ax.set_ylabel("Cumulative Fraction", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=7)
        for ax in axes[len(shape_metrics) :]:
            ax.set_visible(False)
        plt.suptitle("ECDF of Cell Shape Quality Metrics", fontsize=13, y=1.01)
        plt.tight_layout()
        images["cell_ecdfs"] = _fig_to_base64(fig)

    # ------------------------------------------------------------------ #
    # Summary statistics table                                            #
    # ------------------------------------------------------------------ #
    all_metrics = [(col, lbl) for col, lbl, _ in cell_metrics + nucleus_metrics]
    summary_rows = []
    for col, label in all_metrics:
        if col not in stats_before.columns:
            continue
        b = stats_before[col].dropna()
        a = stats_after[col].dropna() if col in stats_after.columns else pd.Series([], dtype=float)
        summary_rows.append(
            {
                "Metric": label.replace("\n", " "),
                "Default — Mean": f"{b.mean():.3f}",
                "Default — Median": f"{b.median():.3f}",
                "Default — Std": f"{b.std():.3f}",
                "EBBF Orig — Mean": f"{a.mean():.3f}" if len(a) else "—",
                "EBBF Orig — Median": f"{a.median():.3f}" if len(a) else "—",
                "EBBF Orig — Std": f"{a.std():.3f}" if len(a) else "—",
            }
        )

    def summary_table_html():
        if not summary_rows:
            return ""
        cols = list(summary_rows[0].keys())
        html = "<table><thead><tr>"
        for c in cols:
            html += f"<th>{c}</th>"
        html += "</tr></thead><tbody>"
        for row in summary_rows:
            html += "<tr>" + "".join(f"<td>{v}</td>" for v in row.values()) + "</tr>"
        html += "</tbody></table>"
        return html

    nucleus_section = ""
    if "nucleus_metrics" in images:
        nucleus_section = f"""
<div class="section">
<h2>Nucleus Shape Metrics</h2>
<p>Same comparison for nucleus boundary polygons. Nucleus boundaries are unchanged
by EBBF; differences here reflect population composition (which cells are "original").</p>
<img src="data:image/png;base64,{images["nucleus_metrics"]}">
</div>"""

    ecdf_section = ""
    if "cell_ecdfs" in images:
        ecdf_section = f"""
<div class="section">
<h2>ECDF of Shape Quality Metrics</h2>
<p>Cumulative distribution functions for bounded (0–1) shape quality metrics.
A rightward shift indicates more "regular" (circular / convex) cells in that population.</p>
<img src="data:image/png;base64,{images["cell_ecdfs"]}">
</div>"""

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Morphology Report — Default vs EBBF</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
         max-width: 1500px; margin: 0 auto; padding: 20px; background: #fafafa; color: #333; }}
  h1 {{ border-bottom: 3px solid #2c3e50; padding-bottom: 10px; }}
  h2 {{ color: #1a5276; margin-top: 40px; border-left: 4px solid #1a5276; padding-left: 12px; }}
  .section {{ background: white; border-radius: 8px; padding: 24px; margin: 20px 0;
              box-shadow: 0 2px 6px rgba(0,0,0,0.08); }}
  img {{ max-width: 100%; height: auto; }}
  table {{ border-collapse: collapse; font-size: 12px; width: 100%; }}
  th {{ background: #2c3e50; color: white; padding: 8px 10px; text-align: center; }}
  td {{ border: 1px solid #ddd; padding: 5px 8px; text-align: center; }}
  tbody tr:nth-child(even) {{ background: #f8f9fa; }}
  tbody tr:hover {{ background: #e8f4f8; }}
  td:first-child {{ text-align: left; font-weight: bold; }}
  .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
              gap: 12px; margin-bottom: 20px; }}
  .stat-card {{ background: #ecf0f1; border-radius: 6px; padding: 15px; text-align: center; }}
  .stat-card .value {{ font-size: 28px; font-weight: bold; color: #2c3e50; }}
  .stat-card .label {{ font-size: 12px; color: #777; margin-top: 4px; }}
</style>
</head><body>

<h1>Cell Morphology Report</h1>
<p><b>Comparison:</b> Default Xenium segmentation (all cells with polygon boundaries) vs
EBBF "original" cells (cells retained without splitting in the EBBF resegmentation).
Split and de-novo cells are excluded because they do not have pre-existing polygon boundaries
stored in the Xenium output.  The population-level distribution shift reveals whether EBBF
preferentially reassigns or splits cells of particular shapes.</p>

<div class="section">
<h2>Overview</h2>
<div class="summary">
  <div class="stat-card"><div class="value">{n_before:,}</div><div class="label">Default Cells</div></div>
  <div class="stat-card"><div class="value">{n_after:,}</div><div class="label">EBBF Original Cells</div></div>
  <div class="stat-card"><div class="value">{n_split:,}</div><div class="label">EBBF Split Cells</div></div>
  <div class="stat-card"><div class="value">{n_denovo:,}</div><div class="label">EBBF De Novo Cells</div></div>
  <div class="stat-card"><div class="value">{n_before - n_after:,}</div><div class="label">Cells Split / Reassigned</div></div>
</div>
</div>

<div class="section">
<h2>Cell Shape Metrics</h2>
<p>Overlaid histograms for each geometric quality metric. <span style="color:#1f77b4;font-weight:bold;">Blue</span> = all default cells,
<span style="color:#ff7f0e;font-weight:bold;">Orange</span> = EBBF original cells (subset).
Shifts between distributions indicate which cell morphologies are enriched or depleted
after EBBF resegmentation.</p>
<img src="data:image/png;base64,{images["cell_metrics"]}">
</div>

{ecdf_section}

{nucleus_section}

<div class="section">
<h2>Summary Statistics</h2>
{summary_table_html()}
</div>

</body></html>"""

    output_path.write_text(html)
    logger.info(f"Saved {output_path}")


def reporting(
    anndata_fname: str,
    xenium_dir: str,
    output_dir: str = ".",
    ef_tx_fname: str | None = None,
    original_tx_fname: str | None = None,
    viz_patch_size: float = 500.0,
):
    """
    Generate all reports comparing EBBF vs default segmentation.

    Args:
        anndata_fname: Path to the EBBF AnnData h5ad file (from post_process)
        xenium_dir: Path to Xenium output directory
        output_dir: Directory for output report files
        ef_tx_fname: Path to EBBF transcript parquet file (for boundary visualization)
        original_tx_fname: Path to original transcripts parquet file (for boundary visualization)
        viz_patch_size: Size of each boundary visualization patch in microns
    """
    import scanpy as sc

    from ebbf.inference.visualization import visualize_boundaries

    warnings.filterwarnings("ignore")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading EBBF AnnData...")
    adata_ebbf = sc.read_h5ad(anndata_fname)

    logger.info("Loading Xenium data...")
    sd = _load_xenium_default(xenium_dir)
    adata_default = sd.tables["table"].copy()

    # Step 1: Boundary visualizations
    if ef_tx_fname is not None and original_tx_fname is not None:
        logger.info("Step 1: generating boundary visualizations...")
        visualize_boundaries(
            ef_tx_fname=ef_tx_fname,
            original_tx_fname=original_tx_fname,
            output_dir=output_dir,
            xenium_dir=xenium_dir,
            patch_size=viz_patch_size,
        )
    else:
        logger.info(
            "Step 1: skipping boundary visualization (requires --ef_tx and --original_transcripts)"
        )

    # Step 2: Distribution report
    logger.info("Step 2: generating distribution report...")
    generate_distribution_report(adata_ebbf, adata_default, output_dir)

    # Step 3: Diff analysis (compute + report)
    logger.info("Step 3: generating diff analysis report...")
    adata_diff, X_diff, genes, shared_ids = _compute_diff_adata(adata_ebbf, adata_default)

    # Save intermediate diff AnnData
    diff_path = Path(output_dir) / "adata_diff.h5ad"
    adata_diff.write_h5ad(diff_path)
    logger.info(f"Saved {diff_path}")

    has_clustering = "leiden_0.5" in adata_diff.obs
    if has_clustering:
        generate_diff_report(adata_diff, X_diff, genes, shared_ids, output_dir)
    else:
        logger.warning("Step 3: skipping diff report (too few cells/genes for clustering)")

    # Step 4: Choropleth clusters
    if has_clustering:
        logger.info("Step 4: generating choropleth clusters report...")
        gdf_boundaries = sd.shapes["cell_boundaries"]
        generate_choropleth_report(adata_diff, gdf_boundaries, output_dir)
    else:
        logger.warning("Step 4: skipping choropleth report (too few cells/genes for clustering)")

    # Step 5: Morphology report
    logger.info("Step 5: generating morphology report...")
    generate_morphology_report(adata_ebbf, adata_default, xenium_dir, output_dir)

    logger.info("DONE! All reports generated.")
