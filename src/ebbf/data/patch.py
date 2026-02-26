from typing import Optional

import numpy as np

import torch
import math
from copy import deepcopy

class NuclearBatch():
    def __init__(
            self,
            patch_center:torch.Tensor, ## B x 2
            nuc_idx:torch.Tensor, ## n_nucs
            nuc2xy:torch.Tensor, ## n_nucs x 2
            nuc2ctype:torch.Tensor, ## n_nucs
            nuc_seqlens:torch.Tensor, ## B
            nuc2total_gene_count:Optional[torch.Tensor]=None, ## n_nucs
            nuc2cell_count:Optional[torch.Tensor]=None, ## n_nucs
            nuc2genes_probs_seq:Optional[torch.Tensor]=None, ## n_sparse_genes
            nuc2genes_idx_seq:Optional[torch.Tensor]=None, ## n_sparse_genes
            nuc2genes_seqlens:Optional[torch.Tensor]=None, ## n_nucs
            punch_idx:Optional[torch.Tensor]=None, ## B
            nuc2ellipse_mat:Optional[torch.Tensor]=None, ## n_nucs x 2 x 2
            nuc2cell_triangle_xy:Optional[torch.Tensor]=None, ## n_triangles
            nuc2cell_triangle_probs:Optional[torch.Tensor]=None, ## n_triangles
            nuc2cell_triangle_seqlens:Optional[torch.Tensor]=None, ## n_nucs
            ):
        
        self.patch_center = patch_center
        self.punch_idx = punch_idx
        self.nuc_idx = nuc_idx
        self.nuc2xy = nuc2xy
        self.nuc2ctype = nuc2ctype
        self.nuc_seqlens = nuc_seqlens
        self.nuc2total_gene_count = nuc2total_gene_count
        self.nuc2genes_probs_seq = nuc2genes_probs_seq
        self.nuc2genes_idx_seq = nuc2genes_idx_seq
        self.nuc2genes_seqlens = nuc2genes_seqlens
        self.nuc2ellipse_mat = nuc2ellipse_mat
        self.nuc2cell_triangle_xy = nuc2cell_triangle_xy
        self.nuc2cell_triangle_probs = nuc2cell_triangle_probs
        self.nuc2cell_triangle_seqlens = nuc2cell_triangle_seqlens
        self.nuc2cell_count = nuc2cell_count

        
    def to(self, device:torch.device):
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.to(device))

    def has_ellipse_mat(self):
        return(self.nuc2ellipse_mat is not None)
    
    def has_cell_triangle(self):
        return(self.nuc2cell_triangle_xy is not None)
    
