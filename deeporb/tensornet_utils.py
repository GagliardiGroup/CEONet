import torch
from typing import Optional, Dict, List, Callable, Tuple, Union

def expand_to(t     : torch.Tensor,
              n_dim : int,
              dim   : int=-1) -> torch.Tensor:
    """Expand dimension of the input tensor t at location 'dim' until the total dimention arrive 'n_dim'

    Args:
        t (torch.Tensor): Tensor to expand
        n_dim (int): target dimension
        dim (int, optional): location to insert axis. Defaults to -1.

    Returns:
        torch.Tensor: Expanded Tensor
    """
    while len(t.shape) < n_dim:
        t = torch.unsqueeze(t, dim=dim)
    return t


def multi_outer_product(v: torch.Tensor,
                        n: int) -> torch.Tensor:
    """Calculate 'n' times outer product of vector 'v'

    Args:
        v (torch.TensorType): vector or vectors ([n_dim] or [..., n_dim])
        n (int): outer prodcut times, will return [...] 1 if n = 0

    Returns:
        torch.Tensor: OvO
    """
    out = torch.ones_like(v[..., 0]) #very slick, cool!
    for _ in range(n):
        out = out[..., None] * expand_to(v, len(out.shape) + 1, dim=len(v.shape) - 1)
    return out

def find_distances(data  : Dict[str, torch.Tensor],) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    idx_i = data["edge_index"][0]
    idx_j = data["edge_index"][1]
    if 'rij' not in data:
        data['rij'] = data['positions'][idx_j] - data['positions'][idx_i]
    if 'dij' not in data:
        data['dij'] = torch.norm(data['rij'], dim=-1)
    if 'uij' not in data:
        data['uij'] = data['rij'] / data['dij'].unsqueeze(-1)
    return data['rij'], data['dij'], data['uij']


def find_moment(batch_data  : Dict[str, torch.Tensor],
                n_way       : int
                ) -> torch.Tensor:
    if 'moment' + str(n_way) not in batch_data:
        find_distances(batch_data)
        batch_data['moment' + str(n_way)] = multi_outer_product(batch_data['uij'], n_way)
    return batch_data['moment' + str(n_way)]

@torch.jit.script
def _scatter_add(x        : torch.Tensor, 
                 idx_i    : torch.Tensor, 
                 dim_size : Optional[int]=None, 
                 dim      : int = 0
                 ) -> torch.Tensor:
    shape = list(x.shape)
    if dim_size is None:
        dim_size = idx_i.max() + 1
    shape[dim] = dim_size
    tmp = torch.zeros(shape, dtype=x.dtype, device=x.device)
    y = tmp.index_add(dim, idx_i, x)
    return y

@torch.jit.script
def _aggregate_new(T1: torch.Tensor,
                   T2: torch.Tensor,
                   way1 : int,
                   way2 : int,
                   way3 : int,
                   ) -> torch.Tensor:
    #inputs are li, lr, lo
    coupling_way = (way1 + way2 - way3) // 2 #lc
    n_way = way1 + way2 - coupling_way + 2 #plus 2 is for E, C, so this is lo + lc (+ 2)
    output_tensor = expand_to(T1, n_way, dim=-1) * expand_to(T2, n_way, dim=2)
    # T1:  [n_edge, n_channel, n_dim, n_dim, ...,     1] 
    # T2:  [n_edge, n_channel,     1,     1, ..., n_dim]  
    # with (way1 + way2 - coupling_way) dim after n_channel
    # We should sum up (coupling_way) n_dim
    if coupling_way > 0: #definitely works for l=2, same logic as multi_outer_product
        sum_axis = [i for i in range(way1 - coupling_way + 2, way1 + 2)]
        output_tensor = torch.sum(output_tensor, dim=sum_axis)
    return output_tensor

def vector_to_skewtensor(vector):
    """Creates a skew-symmetric tensor from a vector."""
    batch_size = vector.size(0)
    zero = torch.zeros(batch_size, device=vector.device, dtype=vector.dtype)
    tensor = torch.stack(
        (
            zero,
            -vector[:, 2],
            vector[:, 1],
            vector[:, 2],
            zero,
            -vector[:, 0],
            -vector[:, 1],
            vector[:, 0],
            zero,
        ),
        dim=1,
    )
    tensor = tensor.view(-1, 3, 3)
    return tensor.squeeze(0)


def vector_to_symtensor(vector):
    """Creates a symmetric traceless tensor from the outer product of a vector with itself."""
    tensor = torch.matmul(vector.unsqueeze(-1), vector.unsqueeze(-2))
    I = (tensor.diagonal(offset=0, dim1=-1, dim2=-2)).mean(-1)[
        ..., None, None
    ] * torch.eye(3, 3, device=tensor.device, dtype=tensor.dtype)
    S = 0.5 * (tensor + tensor.transpose(-2, -1)) - I
    return S


def decompose_tensor(tensor):
    """Full tensor decomposition into irreducible components."""
    I = (tensor.diagonal(offset=0, dim1=-1, dim2=-2)).mean(-1)[
        ..., None, None
    ] * torch.eye(3, 3, device=tensor.device, dtype=tensor.dtype)
    A = 0.5 * (tensor - tensor.transpose(-2, -1))
    S = 0.5 * (tensor + tensor.transpose(-2, -1)) - I
    return I, A, S


def tensor_norm(tensor):
    """Computes Frobenius norm."""
    return (tensor**2).sum((-2, -1))
