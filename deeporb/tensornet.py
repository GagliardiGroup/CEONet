import torch
import torch.nn as nn
import lightning as L
from cace.cace.modules import NodeEncoder, NodeEmbedding
from cace.cace.tools import torch_geometric
from typing import Optional, Dict, List, Callable, Tuple, Union

from deeporb.tensornet_utils import find_distances

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

def tensor_message_passing(
    idx_i: torch.Tensor, idx_j: torch.Tensor, act: torch.Tensor, tensor: torch.Tensor, natoms:int,
) -> torch.Tensor:
    msg = act * tensor.index_select(0, idx_j)
    shape = (natoms, tensor.shape[1], tensor.shape[2], tensor.shape[3])
    tensor_m = torch.zeros(*shape, device=tensor.device, dtype=tensor.dtype)
    tensor_m = tensor_m.index_add(0, idx_i, msg)
    return tensor_m

class Interaction(nn.Module):
    def __init__(self,
                 nc,
                 n_rbf : int=8,
                ) -> None:
        super().__init__()
        self.nc = nc
        self.n_rbf = n_rbf
        self.act = nn.SiLU()

        #To compute tensor activations
        self.linears_scalar = nn.ModuleList()
        self.linears_scalar.append(
            nn.Linear(n_rbf, nc, bias=True)
        )
        self.linears_scalar.append(
            nn.Linear(nc, 2 * nc, bias=True)
        )
        self.linears_scalar.append(
            nn.Linear(2 * nc, 3 * nc, bias=True)
        )

        #Tensor mixing layers
        self.linears_tensor = nn.ModuleList()
        for _ in range(6):
            self.linears_tensor.append(
                nn.Linear(nc, nc, bias=False)
            )

    def forward(self,
                X: torch.Tensor,
                cij : torch.Tensor,
                rbf : torch.Tensor,
                idx_i : torch.Tensor,
                idx_j : torch.Tensor,
               ):
        act = self.act(self.linears_scalar[0](rbf))
        for linear_scalar in self.linears_scalar[1:]:
            act = self.act(linear_scalar(act))
        act = (act * cij.view(-1, 1)).reshape(
            rbf.shape[0], self.nc, 3
        )
        X = X / (tensor_norm(X) + 1)[..., None, None]
        I, A, S = decompose_tensor(X)
        #First mixing
        I = self.linears_tensor[0](I.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        A = self.linears_tensor[1](A.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        S = self.linears_tensor[2](S.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        Y = I + A + S

        #Message passing?
        Im = tensor_message_passing(
            idx_i, idx_j, act[..., 0, None, None], I, X.shape[0]
        )
        Am = tensor_message_passing(
            idx_i, idx_j, act[..., 1, None, None], A, X.shape[0]
        )
        Sm = tensor_message_passing(
            idx_i, idx_j, act[..., 2, None, None], S, X.shape[0]
        )
        msg = Im + Am + Sm

        #Weird thing for parity I don't understand
        A = torch.matmul(msg, Y)
        B = torch.matmul(Y, msg)
        I, A, S = decompose_tensor(A + B)

        normp1 = (tensor_norm(I + A + S) + 1)[..., None, None]
        I, A, S = I / normp1, A / normp1, S / normp1
        I = self.linears_tensor[3](I.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        A = self.linears_tensor[4](A.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        S = self.linears_tensor[5](S.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        dX = I + A + S
        X = X + dX + torch.matrix_power(dX, 2)
        return X

class TensorNet(nn.Module):
    def __init__(self,
                 nc    : int,
                 num_layers : int=2,
                 n_rbf : int=8,
                 cutoff : float=5.5,
                 zs : List[int]=[1,6,7,8,9],
                 ) -> None: 
        super().__init__()
        self.nc = nc
        self.n_rbf = n_rbf
        self.cutoff = cutoff
        self.num_layers = num_layers
        self.zs = zs
        self.nu = 3
        self.encoder = nn.Sequential(NodeEncoder(self.zs),NodeEmbedding(node_dim=len(self.zs),embedding_dim=nc,random_seed=34))
        self.zij_to_nc = nn.Linear(2*nc,nc,bias=True)

        from cace.cace.modules import BesselRBF, GaussianRBF, GaussianRBFCentered
        from cace.cace.modules import PolynomialCutoff
        # self.radial = BesselRBF(self.cutoff,self.n_rbf,trainable=True)
        self.radial = GaussianRBF(self.n_rbf,self.cutoff,trainable=True)
        self.cutoff_fn = PolynomialCutoff(self.cutoff)
        self.rbf_mixing_list = nn.ModuleList([nn.Linear(n_rbf,nc) for _ in range(3)])

        #For eq 9:
        self.init_norm = nn.LayerNorm(nc)
        self.act = nn.SiLU()
        self.linears_scalar = nn.ModuleList()
        self.linears_scalar.append(
            nn.Linear(nc, 2 * nc, bias=True)
        )
        self.linears_scalar.append(
            nn.Linear(2 * nc, 3 * nc, bias=True)
        )
        self.linears_tensor = nn.ModuleList() #channel mixing
        for _ in range(3):
            self.linears_tensor.append(
                nn.Linear(nc, nc, bias=False)
            )

        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(Interaction(self.nc,self.n_rbf))
            
        self.out_norm = nn.LayerNorm(3 * self.nc)
        self.linear = nn.Linear(3 * self.nc, self.nc)

    def _get_tensor_messages(self, zij: torch.Tensor, dij: torch.Tensor, cij: torch.Tensor, rbf: torch.Tensor,
                             uij: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cij = self.cutoff_fn(dij[:,None])
        rbf = self.radial(dij[:,None])
        C = cij.reshape(-1,1,1,1) * zij[...,None,None]
        eye = torch.eye(3, 3, device=C.device, dtype=C.dtype)[
            None, None, ...
        ]
        #N x C x 1 x 1 , N x C x 1 x 1, 1 x 1 x 3 x 3
        Iij = self.rbf_mixing_list[0](rbf)[..., None, None] * C * eye
        Aij = (
            self.rbf_mixing_list[1](rbf)[..., None, None]
            * C
            * vector_to_skewtensor(uij)[..., None, :, :]
        )
        Sij = (
            self.rbf_mixing_list[2](rbf)[..., None, None]
            * C
            * vector_to_symtensor(uij)[..., None, :, :]
        )
        return Iij, Aij, Sij
    
    def forward(self, data : Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        #Edge distances / rbf / cutoff
        _, _, _ = find_distances(data) #rij, dij, uij stored

        #Make Zij
        idx_i = data["edge_index"][0]
        idx_j = data["edge_index"][1]
        z = data["atomic_numbers"]
        zi = self.encoder(z[idx_i])
        zj = self.encoder(z[idx_j])
        zij = self.zij_to_nc(torch.hstack([zi,zj]))

        #Tensor messages
        dij = data["dij"]
        uij = data["uij"]
        cij = self.cutoff_fn(dij[:,None])
        rbf = self.radial(dij[:,None])
        Iij, Aij, Sij = self._get_tensor_messages(zij,dij,cij,rbf,uij)
        source = torch.zeros(
            z.shape[0], self.nc, 3, 3, device=z.device, dtype=zij.dtype
        )
        #index add -- accumulate onto i, so nice!
        I = source.index_add(dim=0, index=idx_i, source=Iij)
        A = source.index_add(dim=0, index=idx_i, source=Aij)
        S = source.index_add(dim=0, index=idx_i, source=Sij)

        norm = self.init_norm(tensor_norm(I + A + S)) #eq 9
        for linear_scalar in self.linears_scalar:
            norm = self.act(linear_scalar(norm))
        norm = norm.reshape(-1, self.nc, 3) #one set of C activations for each tensor

        I = (
            self.linears_tensor[0](I.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            * norm[..., 0, None, None]
        )
        A = (
            self.linears_tensor[1](A.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            * norm[..., 1, None, None]
        )
        S = (
            self.linears_tensor[2](S.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            * norm[..., 2, None, None]
        )
        X = I + A + S

        for layer in self.layers: #check rbf const?
            X = layer(X, cij, rbf, idx_i, idx_j)

        I, A, S = decompose_tensor(X)
        x = torch.cat((tensor_norm(I), tensor_norm(A), tensor_norm(S)), dim=-1)
        x = self.out_norm(x)
        x = self.act(self.linear((x)))
        
        try:
            batch = data["batch"]
        except:
            batch = torch.zeros_like(data["atomic_numbers"])
        output = {
            "positions": data["positions"],
            "cell": data["cell"],
            "batch": batch,
            "node_feats": x,
            "node_feats_A": A
            }
        return output