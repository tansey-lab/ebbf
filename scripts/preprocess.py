import os
import pandas as pd
import geopandas as gpd
import shapely
import polars as pl
import json
from ebbf.utils import compute_polygons
from ebbf.data.tma import process_tma


if __name__=="__main__":
    cancer = 'coad'

    tma_folder = "/data1/tanseyw/projects/tosh/nuc_diffusion/data/spatch/{}".format(cancer)
    os.makedirs(tma_folder, exist_ok=True)

    xenium_folder = "/data1/tanseyw/projects/spatial_data/spatch/{}/xenium/".format(cancer)


    tx_folder = os.path.join(tma_folder, "spatialdata.zarr/points/transcripts/")
    nucs_folder = os.path.join(tma_folder, "spatialdata.zarr/shapes/nucleus_boundaries")
    cell_folder = os.path.join(tma_folder, "spatialdata.zarr/shapes/cell_boundaries")

    os.makedirs(tx_folder, exist_ok=True)
    os.makedirs(nucs_folder, exist_ok=True)
    os.makedirs(cell_folder, exist_ok=True)

    points_fname = os.path.join(tx_folder, "points.parquet")
    nucs_fname = os.path.join(nucs_folder, "shapes.parquet")
    cell_fname = os.path.join(cell_folder, "shapes.parquet")
        
    ndf = pd.read_csv(os.path.join(xenium_folder, "segmentation_mask/nucleus_boundaries.csv"))
    nuc_df = compute_polygons(ndf)
    nuc_df.to_parquet(nucs_fname)

    cdf = pd.read_csv(os.path.join(xenium_folder, "segmentation_mask/cell_boundaries.csv"))
    cell_df = compute_polygons(cdf)
    cell_df.to_parquet(cell_fname)

    tx_df = pl.scan_parquet(os.path.join(xenium_folder, "transcripts/transcripts.parquet"))
    tx_df = tx_df.rename({"x_location": "x", "y_location": "y", "z_location": "z"})
    tx_df.sink_parquet(points_fname)

    ## build a gene panel
    features = tx_df.unique(subset=["feature_name", "codeword_category", "is_gene"]).collect().to_pandas().sort_values(by=['is_gene','feature_name'])
    new_gene_panel_targets = []
    for fname, codeword in zip(features['feature_name'], features['codeword_category']):
        if codeword=='predesigned_gene':
            new_gene_panel_targets.append({'type': {'data': {'name':fname}, 'descriptor':'gene'}})
        else:
            new_gene_panel_targets.append({'type': {'data': {'name':fname}, 'descriptor':codeword}})
    gene_panel = {'payload':{'targets':new_gene_panel_targets}}

    with open(os.path.join(tma_folder, "gene_panel.json"), 'w') as io:
        json.dump(gene_panel, io)


    ## No cell types
    n_nucs = nuc_df.shape[0]
    ctype_df = pd.DataFrame({"cell_type_assignment":["Unknown"]*n_nucs}, index=nuc_df.index.values)
    ctype_df.to_parquet(os.path.join(tma_folder, "modulator_celltype_assignments.parquet"))

    ## write out the bounding box
    tma_shapes_fname = os.path.join(tma_folder, "tma_shapes.parquet")
    minx,miny,_,_ = nuc_df.bounds.min()
    _,_,maxx,maxy = nuc_df.bounds.max()
    bbox = shapely.box(minx-50,miny-50,maxx+50,maxy+50)
    tma_shapes = gpd.GeoDataFrame({"Donor Block ID":[1]}, geometry=[bbox])
    tma_shapes.to_parquet(tma_shapes_fname)

    tma = process_tma(tma_folder=tma_folder, tma_shapes_fname=tma_shapes_fname)
    tma.save(os.path.join(tma_folder, "tma.npz"))
