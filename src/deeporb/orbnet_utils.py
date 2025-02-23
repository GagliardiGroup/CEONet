import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from cace.modules import NodeEncoder, NodeEmbedding
from cace.tools import torch_geometric
from typing import Optional, Dict, List, Callable, Tuple, Union
from deeporb.tensornet_utils import _aggregate_new, expand_to, find_distances, find_moment, _scatter_add, single_tensor_product

def fix_orbints(data : Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
    c_ptr = torch.hstack([torch.zeros(1,device=data["atomic_numbers"].device),data["c_ptr"]])
    c_ptr = torch.cumsum(c_ptr,0)[:-1]
    data["c_ptr"] = c_ptr

    l = 0
    while f"orbints_{l}" in data:
        c_idx = data[f"orbints_{l}"][:,:-1]
        atm_num = data[f"orbints_{l}"][:,-1]

        if len(data[f"orbdata_{l}_ptr"]) == 1:
            pass
        else:
            add = torch.hstack([torch.ones(x,device=atm_num.device)*y for x,y in 
                                zip(data[f"orbdata_{l}_ptr"],data["ptr"][:-1])]).int()
            atm_num += add
            add = torch.hstack([torch.ones(x,device=atm_num.device)*y for x,y in 
                                zip(data[f"orbdata_{l}_ptr"],c_ptr)]).int()
            c_idx += add[:,None]
        data[f"orbints_{l}"] = torch.hstack([c_idx,atm_num[:,None]])
        l += 1
    return data

# def calc_norm(alpha:torch.Tensor,l:int):
#     #see https://iodata.readthedocs.io/en/latest/basis.html
#     if l == 0:
#         return (2**(3/4)) * (torch.pi**(-3/4)) * (alpha**(3/4))
#     if l == 1:
#         return (2**(7/4)) * (torch.pi**(-3/4)) * (alpha**(5/4))

from functools import reduce
def dfac(n):
    if n in [-1,0]:
        return 1
    return reduce(int.__mul__, range(n, 0, -2))

def coef_norm(l_tuple):
    nx,ny,nz = l_tuple
    coef = 1
    coef *= dfac(2*nx - 1)
    coef *= dfac(2*ny - 1)
    coef *= dfac(2*nz - 1)
    coef = coef ** (-1/2)
    return coef

#pyscf convention, g orbs TBD
l_lists ={
    0: [(0,0,0)],
    1: [(1,0,0),(0,1,0),(0,0,1)],
    2: [(2,0,0),(1,1,0),(1,0,1),(0,2,0),(0,1,1),(0,0,2)],
    3: [(3,0,0),(2,1,0),(2,0,1),(1,2,0),(1,1,1),(1,0,2),(0,3,0),(0,2,1),(0,1,2),(0,0,3)],
}

#Now needs to return N x dim
def calc_norm(alpha:torch.Tensor,l:int):
    #see https://iodata.readthedocs.io/en/latest/basis.html
    norm = (2**((3+4*l)/4)) * (torch.pi**(-3/4)) * (alpha**((3+2*l)/4))
    return torch.stack([coef_norm(l_tuple)*norm for l_tuple in l_lists[l]],dim=-1)

import itertools
def reshape_ls(x,l):
    dim = [3]*l + [x.shape[0], x.shape[1]]
    out = torch.zeros(*dim,device=x.device)
    for i,(lx,ly,lz) in enumerate(l_lists[l]):
        idx = [0]*lx + [1]*ly + [2]*lz
        for p in itertools.permutations(idx):
            out[p] = x[...,i]
    out = out.movedim(-1,0).movedim(-1,0)
    return out


