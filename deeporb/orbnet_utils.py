import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from cace.cace.modules import NodeEncoder, NodeEmbedding
from cace.cace.tools import torch_geometric
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

def calc_norm(alpha:torch.Tensor,l:int):
    #see https://iodata.readthedocs.io/en/latest/basis.html
    if l == 0:
        return (2**(3/4)) * (torch.pi**(-3/4)) * (alpha**(3/4))
    if l == 1:
        return (2**(7/4)) * (torch.pi**(-3/4)) * (alpha**(5/4))