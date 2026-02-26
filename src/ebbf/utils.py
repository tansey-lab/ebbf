from typing import Optional

import numpy as np
import pyarrow.dataset as ds
from tqdm import tqdm
from ebbf.data.tma import EllipseLookup
from scipy.sparse import csr_array, coo_array
import pandas as pd
import geopandas as gpd
import shapely

def batch_generator(dataset:ds.Dataset, cols:list[str], batch_size:int):
    n_cols = len(cols)
    batches = dataset.to_batches(
        batch_size=batch_size,
        columns=cols)

    curr_datas = []
    curr_size = 0
    
    for batch in batches:
        data = [np.array(batch[col]) for col in cols]
        curr_datas.append(data)
        curr_size += data[0].shape[0]
        
        if curr_size >= batch_size:
            data = tuple(np.concatenate([x[i] for x in curr_datas]) for i in range(n_cols))
            curr_datas = []
            curr_size = 0
            yield data

    if curr_size > 0:
        data = tuple(np.concatenate([x[i] for x in curr_datas]) for i in range(n_cols))
        yield data

def create_ellipse_nuc_x_gene_array(
        tx_dataset:ds.Dataset, 
        x_col:str, 
        y_col:str, 
        gene_idx_col:str,
        drop_non_zeros:bool,
        n_genes:int,
        nuc_xy:np.ndarray,
        nuc_ellipse_mat:np.ndarray, 
        batch_size:int = 10_000_000):
    
    n_nucs = nuc_xy.shape[0]
    nuc_lookup = EllipseLookup(S=nuc_ellipse_mat, c=nuc_xy)

    batches = batch_generator(tx_dataset, cols=[x_col, y_col, gene_idx_col], batch_size=batch_size)
    total_counts = csr_array((n_nucs, n_genes))
    n_rows = tx_dataset.count_rows()
    pbar = tqdm(total=n_rows)
    for x, y, gene_idx in batches:
        xy = np.stack((x,y), axis=-1)

        res = nuc_lookup.lookup(xy, drop_zeros=False)
        cell_idx = res["closest_ellipse_cell_idx"]
        ell_dist = res["closest_ellipse_distance"]
        
        if drop_non_zeros:
            mask = ell_dist<1e-7
            cell_idx = cell_idx[mask]
            gene_idx = gene_idx[mask]
        
        local_counts = coo_array((np.ones_like(cell_idx), (cell_idx, gene_idx)), shape=(n_nucs, n_genes)).tocsr()
        total_counts = total_counts + local_counts
        pbar.update(xy.shape[0])

    return(total_counts)

def create_cell_x_gene_array(
        tx_dataset:ds.Dataset, 
        gene_id_col:str,
        cell_id_col:str,
        cell_id2idx:Optional[dict[str,int]]=None,
        gene_id2idx:Optional[dict[str,int]]=None,
        n_cells:Optional[int]=None,
        n_genes:Optional[int]=None,
        batch_size:int = 10_000_000):
    
    if cell_id2idx is None:
        cell_id2idx_map = np.vectorize(lambda x:x)
    else:
        cell_id2idx_map = np.vectorize(lambda x:cell_id2idx.get(x, -1))
        n_cells = len(cell_id2idx)

    if gene_id2idx is None:
        gene_id2idx_map = np.vectorize(lambda x:x)
    else:
        gene_id2idx_map = np.vectorize(lambda x:gene_id2idx.get(x, -1))
        n_genes = len(gene_id2idx)
    
    batches = batch_generator(tx_dataset, cols=[cell_id_col, gene_id_col], batch_size=batch_size)
    total_counts = csr_array((n_cells, n_genes))
    n_rows = tx_dataset.count_rows()
    pbar = tqdm(total=n_rows)
    for cell_id, gene_id in batches:
        cell_idx = cell_id2idx_map(cell_id)
        gene_idx = gene_id2idx_map(gene_id)
        mask = (cell_idx>=0)&(gene_idx>=0)
        cell_idx = cell_idx[mask]
        gene_idx = gene_idx[mask]

        local_counts = coo_array((np.ones_like(cell_idx), (cell_idx, gene_idx)), shape=(n_cells, n_genes)).tocsr()
        total_counts = total_counts + local_counts
        pbar.update(cell_idx.shape[0])
    return(total_counts)

def vectorized_means_numpy(X:np.ndarray, labels:np.ndarray):
    K = labels.max() + 1  # number of unique labels, or you can pass it explicitly
    sums = np.zeros((K, X.shape[1]), dtype=X.dtype)
    np.add.at(sums, labels, X)
    counts = np.bincount(labels, minlength=K)
    X_means = np.divide(sums, counts[:, None], out=np.zeros_like(sums), where=counts[:, None] != 0)
    return(X_means)

def compute_polygons(
        vertex_df:pd.DataFrame, 
        cell_id_col:str="cell_id", 
        x_col:str="vertex_x", 
        y_col:str="vertex_y"):
    groups = vertex_df.groupby(cell_id_col).groups
    res = []
    for cell_id, idx in tqdm(groups.items()):
        sub_df = vertex_df.iloc[idx]
        res.append({"cell_id":cell_id, "geometry":shapely.Polygon(sub_df[[x_col, y_col]])})
    poly_df = gpd.GeoDataFrame(res)
    poly_df.set_index("cell_id", inplace=True)
    return(poly_df)


