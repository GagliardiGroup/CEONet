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

@torch.jit.script
def normalize_tensors(input_tensors : Dict[int, torch.Tensor], eps:float=1e-10) -> Dict[int, torch.Tensor]:
        input_tensors[0] = input_tensors[0] - input_tensors[0].mean(dim=-1)[:,None]
        rms = torch.sqrt(torch.mean(input_tensors[0] ** 2,dim=-1) + eps)

        output_tensors = torch.jit.annotate(Dict[int, torch.Tensor], {})
        output_tensors[0] = expand_to(1/rms,2) * input_tensors[0]
        for l in input_tensors.keys():
            if l == 0:
                continue
            input_tensor_ = input_tensors[l].reshape(input_tensors[l].shape[0], input_tensors[l].shape[1], -1)
            factor = 1/(torch.sum(input_tensor_ ** 2, dim=2) + 1)
            output_tensors[l] = expand_to(factor,l+2) * input_tensors[l]
        return output_tensors