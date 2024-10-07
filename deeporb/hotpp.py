import torch
import torch.nn as nn
import lightning as L
from cace.cace.modules import NodeEncoder, NodeEmbedding
from cace.cace.tools import torch_geometric
from typing import Optional, Dict, List, Callable, Tuple, Union

from deeporb.tensornet_utils import multi_outer_product
from deeporb.tensornet_utils import _aggregate_new, expand_to, find_distances, find_moment, _scatter_add

class RBF(nn.Module):
    def __init__(self,n_rbf : int=8, cutoff : float=4.0) -> None:
        super().__init__()
        from cace.cace.modules import BesselRBF, GaussianRBF, GaussianRBFCentered
        from cace.cace.modules import PolynomialCutoff
        self.radial = BesselRBF(cutoff,n_rbf,trainable=True)
        self.cutoff_fn = PolynomialCutoff(cutoff)

    def forward(self, d : torch.Tensor) -> torch.Tensor:
        #d --> [E,]
        d = d.unsqueeze(-1)
        d = self.radial(d) * self.cutoff_fn(d)
        return d

class TensorLinearMixing(nn.Module):
    def __init__(self,
                 n_in : int,
                 n_out : int,
                 lomax : int,
                 ) -> None:
        super().__init__()
        self.linear_list = nn.ModuleList([
            nn.Linear(n_in, n_out, bias=(l==0)) for l in range(lomax + 1)
        ])

    def forward(self,
                input_tensors : Dict[int, torch.Tensor],
                ) -> Dict[int, torch.Tensor]:
        output_tensors = torch.jit.annotate(Dict[int, torch.Tensor], {})
        for l, linear in enumerate(self.linear_list):
            input_tensor = torch.transpose(input_tensors[l], 1, -1)
            output_tensor = linear(input_tensor)
            output_tensors[l] = torch.transpose(output_tensor, 1, -1)
        return output_tensors

class TensorActivation(nn.Module):
    def __init__(self,
                 nc : int,
                 lomax : int,
                 ) -> None:
        super().__init__()
        
        self.weight_list = nn.ParameterList([
            torch.ones(nc,requires_grad=True) for l in range(lomax)
        ])

        self.bias_list = nn.ParameterList([
            torch.zeros(nc,requires_grad=True) for l in range(lomax)
        ])

        self.scalar_act = nn.SiLU()
        self.tensor_act = nn.Sigmoid()
        # self.tensor_act = nn.SiLU()
        self.lomax = lomax
    
    def forward(self,
                input_tensors : Dict[int, torch.Tensor],
                ) -> Dict[int, torch.Tensor]:
        output_tensors = torch.jit.annotate(Dict[int, torch.Tensor], {})
        for l in range(self.lomax+1):
            if l == 0:
                output_tensors[l] = self.scalar_act(input_tensors[l])
            else:
                input_tensor_ = input_tensors[l].reshape(input_tensors[l].shape[0], input_tensors[l].shape[1], -1)
                norm = self.weight_list[l-1] * torch.sum(input_tensor_ ** 2, dim=2) + self.bias_list[l-1]
                factor = self.tensor_act(norm)
                output_tensors[l] = expand_to(factor,l+2) * input_tensors[l]
        return output_tensors

class TensorProductLayer(nn.Module):
    def __init__(self,
                 max_x_way      : int=2,
                 max_y_way      : int=2,
                 max_z_way      : int=2,
                 ) -> None:
        #lin, lr, lout
        super().__init__()
        self.combinations = []
        for x_way in range(max_x_way + 1):
            for y_way in range(max_y_way + 1):
                for z_way in range(abs(y_way - x_way), min(max_z_way, x_way + y_way) + 1, 2):
                    self.combinations.append((x_way, y_way, z_way))

    def forward(self,
                x : Dict[int, torch.Tensor],
                y : Dict[int, torch.Tensor],
                ) -> Dict[int, torch.Tensor]:
        output_tensors = torch.jit.annotate(Dict[int, torch.Tensor], {})
        for x_way, y_way, z_way in self.combinations:
            if x_way not in x or y_way not in y:
                continue
            output_tensor = _aggregate_new(x[x_way], y[y_way], x_way, y_way, z_way)
            if z_way not in output_tensors:
                output_tensors[z_way] = output_tensor
            else:
                output_tensors[z_way] += output_tensor
        return output_tensors

class MessagePassingLayer(nn.Module):
    def __init__(self,
                 nc    : int,
                 n_rbf : int=8,
                 lomax : int=2,
                 cutoff : float=4.0,
                 ) -> None:
        super().__init__()
        self.lomax = lomax
        self.rbf = RBF(n_rbf,cutoff=cutoff)
        self.rbf_mixing_list = nn.ModuleList([
            nn.Linear(n_rbf, nc, bias=True)
            for l in range(lomax + 1)
        ])
        self.tensor_product = TensorProductLayer()
        self.tensor_linear = TensorLinearMixing(nc,nc,lomax)
        self.message_processing = nn.Sequential(
            TensorLinearMixing(nc,nc,lomax),
            TensorActivation(nc,lomax),
        )

    def forward(self, data : Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        #Edge distances
        idx_i = data["edge_index"][0]
        idx_j = data["edge_index"][1]
        _, dij, _ = find_distances(data)
        rbf_ij = self.rbf(dij)

        #Define H
        x = torch.jit.annotate(Dict[int, torch.Tensor], {})
        for l in range(self.lomax + 1):
            x[l] = data["node_feats"][l][idx_j]
        x = self.tensor_linear(x) #for some reason they have this
        
        #Define U
        y = torch.jit.annotate(Dict[int, torch.Tensor], {})
        for l, rbf_mixing in enumerate(self.rbf_mixing_list):
            fn = rbf_mixing(rbf_ij)
            y[l] = find_moment(data, l).unsqueeze(1) * expand_to(fn, n_dim=l + 2)

        #Tensor H(j) x U(i,j)
        edge_messages = self.tensor_product(x,y)

        #Accumulate Message
        node_message = torch.jit.annotate(Dict[int, torch.Tensor], {})
        n_atoms = data["atomic_numbers"].shape[0]
        
        for l in edge_messages.keys():
            #accumulate onto i
            node_message[l] = _scatter_add(edge_messages[l], idx_i, dim_size=n_atoms)
        
        # Process Message
        node_message = self.message_processing(node_message)
        
        #Add to node_info
        for l in node_message.keys():
            data["node_feats"][l] = data["node_feats"][l] + node_message[l]

        return data

class HotPP(L.LightningModule):
    def __init__(self,
                 nc    : int,
                 layers : int=2,
                 # embedding_dim: int=3,
                 n_rbf : int=8,
                 lomax : int=2,
                 cutoff : float=4.0,
                 zs : List[int]=[1,6,7,8,9],
                 ) -> None:
        super().__init__()
        self.zs = zs
        self.nc = nc
        self.lomax = lomax
        self.encoder = nn.Sequential( #change later, a little big for embedding
                    NodeEncoder(self.zs),
                    NodeEmbedding(node_dim=len(self.zs),embedding_dim=nc,random_seed=34),
                    # nn.Linear(embedding_dim,nc,bias=True)
                )
        message_passing = [
            MessagePassingLayer(nc=nc,n_rbf=n_rbf,lomax=lomax,cutoff=cutoff) for _ in range(layers)
        ]
        self.message_passing = nn.Sequential(*message_passing)
        
    def forward(self, data : Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        #Encoding
        h0 = torch.jit.annotate(Dict[int, torch.Tensor], {})
        h0[0] = self.encoder(data["atomic_numbers"])
        h0[1] = torch.zeros(h0[0].shape[0],self.nc,3,device=h0[0].device)
        for l in range(2,self.lomax+1):
            #N x C x 3 x 3, etc.
            h0[l] = h0[l-1].unsqueeze(-2) * h0[l-1].unsqueeze(-1)
        data["node_feats"] = h0
        
        #Message Passing
        data = self.message_passing(data)
        
        try:
            batch = data["batch"]
        except:
            batch = torch.zeros_like(data["atomic_numbers"])
        output = {
            "positions": data["positions"],
            "cell": data["cell"],
            "batch": batch,
            "node_feats": data["node_feats"][0],
            "node_feats_A": data["node_feats"] #for equivariance testing
            } 
        
        return output