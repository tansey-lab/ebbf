from typing import Optional
import os

import numpy as np
import pandas as pd
import geopandas as gpd
import polars as pl
import json
import pyarrow.dataset as ds
import shapely

from tqdm import tqdm
from itertools import product

import concurrent.futures as cf


from ebbf.data.datasets import NuclearPatchDataset, ComboDataset
from ebbf.data.utils import approximate_ellipse

from scipy.sparse import csr_array, coo_array
from scipy.spatial import KDTree

class EllipseLookup():
    def __init__(self, S:np.ndarray, c:np.ndarray):
        self.S = S
        self.Q = np.linalg.inv(np.matmul(S, S.mT))
        self.c = c
        
        ## Build ellipse lookup structure
        n_grid = 50
        
        t = np.linspace(0.0, 2*np.pi, n_grid, endpoint=False)
        unit = np.stack([np.cos(t), np.sin(t)], axis=1)
        ellipse_pts = c[:,np.newaxis,:] + np.einsum('bij, tj -> bti', S, unit)
        nuc_idx = np.repeat(np.arange(c.shape[0])[:,np.newaxis], n_grid, axis=1)
        
        ellipse_pts_flat = ellipse_pts.reshape((-1,2))
        self.nuc_idx_flat = nuc_idx.reshape((-1,))
        
        self.nuc_ellipse_lookup = KDTree(ellipse_pts_flat)
        
        ## Also build just regular nuc centroid lookup structure
        self.centroid_lookup = KDTree(c)

    def lookup(self, xy:np.ndarray, drop_zeros:bool=True):
        n = xy.shape[0]

        ## Check to see if we lie inside of any ellipses
        centroid_dist, centroid_idx = self.centroid_lookup.query(xy, k=5)
        
        d = xy[:,np.newaxis,:]-self.c[centroid_idx]
        
        ellipse_condition = np.einsum('...i,...ij,...j->...' , d, self.Q[centroid_idx], d) <= 1
        inside_ellipse_mask1 = ellipse_condition.any(axis=-1)
        inside_ellipse_idx1 = centroid_idx[np.arange(n), np.argmax(ellipse_condition, axis=-1)]

        ## Filter out the points that are inside the ellipse
        # xy = xy[~inside_ellipse_mask1]

        ## Get closest ellipse boundary
        _, ellipse_pt_idx = self.nuc_ellipse_lookup.query(xy, k=1)
        ellipse_pt_nuc_idx = self.nuc_idx_flat[ellipse_pt_idx]
        
        diff_closest_nuc_centroid = xy - self.c[ellipse_pt_nuc_idx]
        dist_to_closest_nuc_centroid = np.linalg.norm(diff_closest_nuc_centroid, axis=-1)
        mahalanobis_distance_to_closest_nuc_centroid = np.sqrt(np.einsum('...i,...ij,...j->...' , diff_closest_nuc_centroid, self.Q[ellipse_pt_nuc_idx], diff_closest_nuc_centroid))

        ## We may have missed some points that are inside an ellipse
        ## We'll just drop these
        inside_ellipse_mask2 = mahalanobis_distance_to_closest_nuc_centroid <= 1.0
        # ellipse_pt_nuc_idx = ellipse_pt_nuc_idx[~inside_ellipse_mask2]
        # dist_to_closest_nuc_centroid = dist_to_closest_nuc_centroid[~inside_ellipse_mask2]
        # mahalanobis_distance_to_closest_nuc_centroid = mahalanobis_distance_to_closest_nuc_centroid[~inside_ellipse_mask2]
        ellipse_dist = dist_to_closest_nuc_centroid*(1 - (1./np.clip(mahalanobis_distance_to_closest_nuc_centroid, a_min=1e-3, a_max=None)))

        ## Overwrite these points
        ellipse_pt_nuc_idx[inside_ellipse_mask1] = inside_ellipse_idx1[inside_ellipse_mask1]
        ellipse_dist[inside_ellipse_mask1|inside_ellipse_mask2] = 0.0

        if drop_zeros:
            mask = ellipse_dist!=0
            ellipse_pt_nuc_idx = ellipse_pt_nuc_idx[mask]
            ellipse_dist = ellipse_dist[mask]

        result = {
            "closest_ellipse_cell_idx":ellipse_pt_nuc_idx, 
            "closest_ellipse_distance":ellipse_dist, 
            "nuc_centroid_idx":centroid_idx[:,0], 
            "nuc_centroid_distance":centroid_dist[:,0]}
        
        return(result)

    def ellipse_lookup(self, xy:np.ndarray):
        n = xy.shape[0]

        ## Check to see if we lie inside of any ellipses
        _, centroid_idx = self.centroid_lookup.query(xy, k=5)
        
        d = xy[:,np.newaxis,:]-self.c[centroid_idx]
        
        ellipse_condition = np.einsum('...i,...ij,...j->...' , d, self.Q[centroid_idx], d) <= 1
        inside_ellipse_mask1 = ellipse_condition.any(axis=-1)
        inside_ellipse_idx1 = centroid_idx[np.arange(n), np.argmax(ellipse_condition, axis=-1)]

        ## Get closest ellipse boundary
        _, ellipse_pt_idx = self.nuc_ellipse_lookup.query(xy, k=1)
        ellipse_pt_nuc_idx = self.nuc_idx_flat[ellipse_pt_idx]
        
        diff_closest_nuc_centroid = xy - self.c[ellipse_pt_nuc_idx]
        dist_to_closest_nuc_centroid = np.linalg.norm(diff_closest_nuc_centroid, axis=-1)
        mahalanobis_distance_to_closest_nuc_centroid = np.sqrt(np.einsum('...i,...ij,...j->...' , diff_closest_nuc_centroid, self.Q[ellipse_pt_nuc_idx], diff_closest_nuc_centroid))

        ## We may have missed some points that are inside an ellipse
        ## We'll just drop these
        inside_ellipse_mask2 = mahalanobis_distance_to_closest_nuc_centroid <= 1.0
        ellipse_dist = dist_to_closest_nuc_centroid*(1 - (1./np.clip(mahalanobis_distance_to_closest_nuc_centroid, a_min=1e-3, a_max=None)))

        ## Overwrite these points
        ellipse_pt_nuc_idx[inside_ellipse_mask1] = inside_ellipse_idx1[inside_ellipse_mask1]
        ellipse_dist[inside_ellipse_mask1|inside_ellipse_mask2] = 0.0

        ## Compute distance to the centroid
        centroid_distance = np.linalg.norm(xy - self.c[ellipse_pt_nuc_idx], axis=-1) 

        result = {
            "ellipse_idx":ellipse_pt_nuc_idx, 
            "ellipse_distance":ellipse_dist, 
            "centroid_distance":centroid_distance}
        
        return(result)
    
# --- Globals set once per worker ---
_worker_lookup = None
_gene2idx = None
_cell_id2idx = None
_n_nucs = None
_n_genes = None
_n_bins = None
_max_dist = None


def _init_worker(S,c, gene2idx, cell_id2idx, n_nucs, n_genes, n_bins, max_dist):
    """Initializer, runs once per worker process."""
    global _worker_lookup, _gene2idx, _cell_id2idx, _n_nucs, _n_genes, _n_bins, _max_dist
    _worker_lookup = EllipseLookup(S,c)   # your custom class
    _gene2idx = gene2idx
    _cell_id2idx = cell_id2idx
    _n_nucs = n_nucs
    _n_genes = n_genes
    _n_bins = n_bins
    _max_dist = max_dist


def _assign_to_bin(vals):
    vals_clip = np.clip(vals, a_min=0, a_max=(_max_dist - 1e-5))
    return np.floor(vals_clip * _n_bins / _max_dist).astype(int)


def process_batch(batch):
    """Executed for each batch inside worker processes."""
    global _worker_lookup, _gene2idx, _cell_id2idx, _n_nucs, _n_genes

    ###########################
    ### Nuclear gene counts ###
    ###########################
    genes = np.array(batch['feature_name'], dtype=object)
    cell_id = np.array(batch['cell_id'], dtype=object)

    ## Filter to nuclear tx
    overlaps_nucleus = np.array(batch['overlaps_nucleus']) == 1
    nuc_genes = genes[overlaps_nucleus].astype(str)
    nuc_cell_ids = cell_id[overlaps_nucleus]

    ## Filter out weirdness
    mask = nuc_genes!='None'

    if not np.any(mask):
        print("EMPTY")
    
    nuclear_tx_gene_idx = np.vectorize(_gene2idx.get)(nuc_genes[mask])
    nuclear_tx_nuc_idx = np.vectorize(_cell_id2idx.get)(nuc_cell_ids[mask])

    local_nuc_gene_counts = coo_array(
        (np.ones_like(nuclear_tx_gene_idx),
         (nuclear_tx_nuc_idx, nuclear_tx_gene_idx)),
        shape=(_n_nucs, _n_genes)
    )

    ###########################
    ### Histogram distances ###
    ###########################
    xy = np.array([batch['x'], batch['y']]).T

    res = _worker_lookup.lookup(xy, drop_zeros=True)  # expects dict with 'closest_ellipse_distance', etc.
    ellipse_bins = _assign_to_bin(res['closest_ellipse_distance'])
    centroid_bins = _assign_to_bin(res['nuc_centroid_distance'])

    local_ellipse_hist = coo_array(
        (np.ones_like(ellipse_bins),
         (res['closest_ellipse_cell_idx'], ellipse_bins)),
        shape=(_n_nucs, _n_bins)
    )
    local_centroid_hist = coo_array(
        (np.ones_like(centroid_bins),
         (res['nuc_centroid_idx'], centroid_bins)),
        shape=(_n_nucs, _n_bins)
    )

    ### Where are we ###
    num = xy.shape[0]

    return {
        "local_nuc_gene_counts": local_nuc_gene_counts.tocsr(),
        "local_ellipse_hist": local_ellipse_hist.tocsr(),
        "local_centroid_hist": local_centroid_hist.tocsr(),
        "num":num
    }

def _compute_histograms(
        points_dataset:ds.Dataset,
        S:np.ndarray,
        c:np.ndarray,
        cell_id2idx:dict[str,int],
        gene2idx:dict[str,int], 
        n_bins:int=100, 
        max_dist:int=40,
        batch_size:int = 10_000_000,
        num_workers:Optional[int]=None):
    
    n_rows = points_dataset.count_rows(filter=(ds.field('is_gene')))

    if num_workers is None:
        num_workers = len(os.sched_getaffinity(0))//2
    
    n_nucs = len(cell_id2idx)
    n_genes = len(gene2idx)

    # Create a generator for streaming batches
    def batch_generator():
        batches = points_dataset.to_batches(
            batch_size=batch_size,
            columns=['x','y','feature_name', 'is_gene', 'cell_id', 'overlaps_nucleus'], 
            filter=(ds.field('is_gene')))
        for batch in batches:
            yield batch


    # --- Main driver ---
    nuc_gene_counts = csr_array((n_nucs, n_genes), dtype=np.int64)
    nuc_ellipsoid_hist = csr_array((n_nucs, n_bins), dtype=np.int64)
    nuc_centroid_hist = csr_array((n_nucs, n_bins), dtype=np.int64)

    with cf.ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_worker,
        initargs=(S,c, gene2idx, cell_id2idx, n_nucs, n_genes, n_bins, max_dist)
    ) as executor:
        
        # map yields results in submission order, but tqdm tracks progress smoothly
        pbar = tqdm(total=n_rows)
        for res in executor.map(process_batch, batch_generator(), chunksize=1):
            nuc_gene_counts = nuc_gene_counts + res['local_nuc_gene_counts']
            nuc_ellipsoid_hist = nuc_ellipsoid_hist + res['local_ellipse_hist']
            nuc_centroid_hist = nuc_centroid_hist + res['local_centroid_hist']
            pbar.update(res["num"])

    results = {
        "nuc_gene_counts":nuc_gene_counts,
        "nuc_ellipsoid_hist":nuc_ellipsoid_hist, 
        "nuc_centroid_hist": nuc_centroid_hist}
    
    return(results)

def _compute_boxes(tma_shapes:gpd.GeoDataFrame, box_width:float):
    
    all_boxes = []
    for punch in tma_shapes.geometry.values:
        min_x, min_y, max_x, max_y = punch.bounds
        
        L_x = max_x - min_x
        L_y = max_y - min_y
        
        
        n_x = int(L_x//box_width + 1)
        n_y = int(L_y//box_width + 1)
        
        a_x = min_x + np.arange(n_x)*(L_x - box_width)/(n_x - 1)
        a_y = min_y + np.arange(n_y)*(L_y - box_width)/(n_y - 1)
        
        boxes = np.array([(xmin, ymin, xmin+box_width, ymin+box_width)  for xmin, ymin in product(a_x, a_y)])
        all_boxes.append(boxes)
    boxes = np.concatenate(all_boxes)
    return(boxes)

def _estimate_per_box_ctype_distance_histograms(
        nuc_hist:csr_array,
        nucs:gpd.GeoDataFrame,
        boxes:np.ndarray, 
        ctype2index:dict[str,int])->np.ndarray:
    
    n_bins = nuc_hist.shape[1]
    n_ctypes = len(ctype2index)
    n_boxes = boxes.shape[0]
    hist_array = np.zeros((n_boxes, (n_ctypes+1), n_bins), dtype=int)

    for i, box in enumerate(tqdm(boxes)):
        box_mask = ((nucs['x']>box[0])&(nucs['y']>box[1])&(nucs['x']<box[2])&(nucs['y']<box[3])).values

        nuc_hist_box = nuc_hist[box_mask]
        nucs_box = nucs[box_mask]

        for ctype, cidx in ctype2index.items():
            ctype_mask = nucs_box['cell_type_assignment'].values == ctype
            if np.any(ctype_mask):
                hist_array[i, cidx,:] = nuc_hist_box[ctype_mask].sum(axis=0)
        
        hist_array[i,-1,:] = nuc_hist_box.sum(axis=0)

    return(hist_array)

def _compute_polygon_triangulation(polygons):
    all_triangles = shapely.constrained_delaunay_triangles(polygons)

    all_triangle_xy = []
    all_triangle_probs = []
    for triangles in tqdm(all_triangles): 
        triangle_xy = np.stack([ np.unique(np.stack(triangle.exterior.xy, axis=-1), axis=0) for triangle in triangles.geoms])
        all_triangle_xy.append(triangle_xy)
        triangle_areas = np.array([triangle.area for triangle in triangles.geoms])
        triangle_probs = triangle_areas/np.sum(triangle_areas)
        all_triangle_probs.append(triangle_probs)
    return(all_triangle_xy, all_triangle_probs)


def process_tma(
    tma_folder:str,
    tma_shapes_fname:str,
    max_dist:float=40, 
    n_bins:int=100, 
    box_width:float=400,
    nuc_density_per_um2:float=1e-4):

    num_workers = len(os.sched_getaffinity(0))

    gene_panel_fname = os.path.join(tma_folder, "gene_panel.json")
    celltype_assignments_fname = os.path.join(tma_folder, "modulator_celltype_assignments.parquet")
    points_folder = os.path.join(tma_folder, "spatialdata.zarr/points/transcripts/points.parquet")
    nucs_fname = os.path.join(tma_folder, "spatialdata.zarr/shapes/nucleus_boundaries/shapes.parquet")
    cells_fname = os.path.join(tma_folder, "spatialdata.zarr/shapes/cell_boundaries/shapes.parquet")

    ## Read gene panel
    ## OLD WAY
    # with open(gene_panel_fname, 'r') as io:
    #     gene_panel = json.load(io)
    # panel = np.unique([x['type']['data']['name'] for x in gene_panel['payload']['targets'] if x['type']['descriptor']=='gene']).astype(object)
    # gene2idx = {x:i for i,x in enumerate(panel)}
    ## New way
    df = pl.scan_parquet(points_folder)
    df = df.filter(pl.col("is_gene"))
    df = df.select(["feature_name"]).unique().collect().to_pandas()
    panel = np.unique(df['feature_name'].values)
    gene2idx = {x:i for i,x in enumerate(panel)}
    
    ## Read points/nucleus shapes
    points_dataset = ds.dataset(points_folder, format='parquet')
    nucs = gpd.read_parquet(nucs_fname)
    nuc_cell_id = nucs.index.values
    cell_id2idx = {x:i for i,x in enumerate(nuc_cell_id)}
    S, _, _, c = approximate_ellipse(nucs.geometry.values)

    ## Read cell shapes
    if os.path.isfile(cells_fname):
        print("Computing cell polygons")
        cells = gpd.read_parquet(cells_fname)
        polygons = shapely.make_valid(cells.loc[nuc_cell_id]['geometry'].values)
        cell_triangle_xy, cell_triangle_probs = _compute_polygon_triangulation(polygons)
        cell_triangle_num = np.array([p.shape[0] for p in cell_triangle_probs])
        cell_triangle_xy = np.concatenate(cell_triangle_xy, axis=0)
        cell_triangle_probs = np.concatenate(cell_triangle_probs, axis=0)
    else:
        cell_triangle_xy, cell_triangle_probs = None, None
        cell_triangle_num = None
    
    ## Read cell assignments and count them
    df = pl.scan_parquet(points_folder)
    df = df.filter((pl.col("cell_id")!="UNASSIGNED")&pl.col("is_gene"))
    cell_counts = df.group_by('cell_id').len().collect().to_pandas()
    nuc_cell_counts = np.zeros(nuc_cell_id.shape[0], dtype=np.int64)
    for cid, count in zip(cell_counts['cell_id'].values, cell_counts['len'].values):
        if cid in cell_id2idx:
            nuc_cell_counts[cell_id2idx[cid]] = count

    ## Read cell types
    ctypes = pd.read_parquet(celltype_assignments_fname)
    cell_id2ctypes = dict(zip(ctypes.index.values, ctypes['cell_type_assignment'].values))
    unique_cell_types = np.setdiff1d(np.unique(ctypes['cell_type_assignment'].astype(str)), ['nan'])
    ctype2index = {x:i for i,x in enumerate(unique_cell_types)}
    
    print("Computing histograms/count matrices")
    result = _compute_histograms(
        points_dataset, S, c, cell_id2idx, gene2idx, 
        n_bins=n_bins, max_dist=max_dist, num_workers=(num_workers//2))

    nuc_gene_counts:csr_array = result['nuc_gene_counts']

    ## Add spatial and cell type info to nucs
    nucs['x'] = c[:,0]
    nucs['y'] = c[:,1]
    nucs['cell_type_assignment'] = nucs.index.map(cell_id2ctypes).astype(str)

    ctype2index_plus_nan = ctype2index.copy()
    ctype2index_plus_nan['nan'] = -1
    nucs['cell_type_idx'] = nucs['cell_type_assignment'].map(ctype2index_plus_nan)

    ## Create boxes based on punch shapes
    tma_shapes = gpd.read_parquet(tma_shapes_fname)
    tma_shapes = tma_shapes[tma_shapes['Donor Block ID']!='KIDNEY MARKER']
    boxes = _compute_boxes(tma_shapes, box_width)

    ## Filter out the boxes that are too empty
    nuc_count_thresh = nuc_density_per_um2*(box_width**2)
    n_nucs_per_box = np.array([np.count_nonzero((nucs['x']>box[0])&(nucs['y']>box[1])&(nucs['x']<box[2])&(nucs['y']<box[3])) for box in boxes])
    boxes = boxes[n_nucs_per_box >= nuc_count_thresh]

    print("Estimating histogram per ctype/box")

    ## Centroid histograms
    centroid_hist_array = _estimate_per_box_ctype_distance_histograms(result['nuc_centroid_hist'], nucs, boxes, ctype2index)

    ## Ellipse histograms
    ellipse_hist_array = _estimate_per_box_ctype_distance_histograms(result['nuc_ellipsoid_hist'], nucs, boxes, ctype2index)

    ## Finalize stats
    nuc_total_tx_counts = nuc_gene_counts.sum(axis=1)
    nuc_gene_counts:coo_array = nuc_gene_counts.tocoo()
    tma = TMA(
        boxes=boxes, 
        centroid_hist_array=centroid_hist_array,
        ellipse_hist_array=ellipse_hist_array,
        max_dist=max_dist,
        nuc_cell_id=nuc_cell_id,
        nuc_total_tx_counts=nuc_total_tx_counts,
        nuc_ctype_idx=nucs['cell_type_idx'].values,
        nuc_cell_counts=nuc_cell_counts,
        nuc_xy=c,
        nuc_ellipse_mat=S,
        nuc_countmat_nuc_idx=nuc_gene_counts.row,
        nuc_countmat_gene_idx=nuc_gene_counts.col,
        nuc_countmat_counts=nuc_gene_counts.data, 
        genes=panel, 
        cell_types=unique_cell_types, 
        cell_triangle_num=cell_triangle_num,
        cell_triangle_xy=cell_triangle_xy, 
        cell_triangle_probs=cell_triangle_probs)
    
    return(tma)


class TMA:
    def __init__(
            self, 
            boxes:np.ndarray, 
            centroid_hist_array:np.ndarray,
            ellipse_hist_array:np.ndarray,
            max_dist:float,
            nuc_cell_id:np.ndarray,
            nuc_total_tx_counts:np.ndarray, 
            nuc_ctype_idx:np.ndarray,
            nuc_cell_counts:np.ndarray,
            nuc_xy:np.ndarray,
            nuc_ellipse_mat:np.ndarray,
            nuc_countmat_nuc_idx:np.ndarray, 
            nuc_countmat_gene_idx:np.ndarray, 
            nuc_countmat_counts:np.ndarray, 
            genes:np.ndarray, 
            cell_types:np.ndarray,
            cell_triangle_num:Optional[list[np.ndarray]]=None,
            cell_triangle_xy:Optional[list[np.ndarray]]=None, 
            cell_triangle_probs:Optional[list[np.ndarray]]=None, 
            **kwargs):
        
        self.boxes = boxes
        self.centroid_hist_array = centroid_hist_array
        self.ellipse_hist_array = ellipse_hist_array
        self.max_dist = max_dist
        self.nuc_cell_id = nuc_cell_id
        self.nuc_total_tx_counts = nuc_total_tx_counts
        self.nuc_ctype_idx = nuc_ctype_idx
        self.nuc_xy = nuc_xy
        self.nuc_ellipse_mat = nuc_ellipse_mat
        self.nuc_countmat_nuc_idx = nuc_countmat_nuc_idx
        self.nuc_countmat_gene_idx = nuc_countmat_gene_idx
        self.nuc_countmat_counts = nuc_countmat_counts
        self.genes = genes
        self.cell_types = cell_types
        self.nuc_cell_counts = nuc_cell_counts

        if cell_triangle_xy is None:
            self.cell_triangle_xy = np.array([])
            self.cell_triangle_probs = np.array([])
            self.cell_triangle_num = np.array([])
        else:
            self.cell_triangle_num = cell_triangle_num
            self.cell_triangle_xy = cell_triangle_xy
            self.cell_triangle_probs = cell_triangle_probs
            
        self.__dict__.update(kwargs)

    def save(self, fname:str):
        np.savez_compressed(fname, **self.__dict__)

    
    @classmethod
    def load(cls, fname:str):
        res = np.load(fname, allow_pickle=True)
        tma = cls(**res)
        tma.max_dist = res['max_dist'].item()
        return tma
    
    def get_nuc_dataset(
            self, 
            patch_size:float, 
            patch_padding:float, 
            dataset_size:int, 
            min_nucs:int, 
            use_ctypes:bool):
        
        n_nucs = self.nuc_xy.shape[0]
        n_genes = self.genes.shape[0]

        nuc2genes = csr_array((self.nuc_countmat_counts, (self.nuc_countmat_nuc_idx, self.nuc_countmat_gene_idx)), shape=(n_nucs, n_genes))
        nuc_total_sums = nuc2genes.sum(axis=1)[:,np.newaxis]
        nuc2gene_probs = (nuc2genes/nuc_total_sums).tocsr()

        nuc2ctypes = self.nuc_ctype_idx if use_ctypes else np.zeros_like(self.nuc_ctype_idx)
        dataset = NuclearPatchDataset(
            nuc2xy=self.nuc_xy, 
            nuc2total_gene_count=self.nuc_total_tx_counts, 
            nuc2gene_probs=nuc2gene_probs, 
            nuc2ctype=nuc2ctypes,
            nuc2cell_count=self.nuc_cell_counts,
            punch_bounds=self.boxes, 
            patch_size=patch_size, 
            patch_padding=patch_padding, 
            dataset_size=dataset_size, 
            min_nucs=min_nucs, 
            nuc2ellipse_mat=self.nuc_ellipse_mat, 
            cell_triangle_num=(self.cell_triangle_num if self.cell_triangle_num.shape[0]>0 else None), 
            cell_triangle_xy=(self.cell_triangle_xy if self.cell_triangle_xy.shape[0]>0 else None), 
            cell_triangle_probs=(self.cell_triangle_probs if self.cell_triangle_probs.shape[0]>0 else None))
        return(dataset)
    
    def get_combo_dataset(
            self,
            tx_data_fname:str,
            patch_size:float, 
            patch_padding:float, 
            dataset_size:int, 
            min_nucs:int,
            include_all_nucs_from_patch:bool,
            max_tx:Optional[int]=None):
        
        dataset = ComboDataset(
            tx_data_fname=tx_data_fname,
            nuc2xy=self.nuc_xy, ## n_nucs x 2
            nuc2ctype=self.nuc_ctype_idx, ## n_nucs
            genes=self.genes,
            box_bounds=self.boxes, ## n_punches x 4 (minx, miny, maxx, maxy) 
            patch_size=patch_size, 
            patch_padding=patch_padding, 
            dataset_size=dataset_size,
            include_all_nucs_from_patch=include_all_nucs_from_patch,
            min_nucs=min_nucs,
            max_tx=max_tx,
            nuc2ellipse_mat=self.nuc_ellipse_mat)
        
        return(dataset)

    