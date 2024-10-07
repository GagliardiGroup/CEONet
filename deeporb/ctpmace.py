import torch
import torch.nn as nn
from cace.cace.modules import NodeEncoder, NodeEmbedding
from cace.cace.tools import torch_geometric
from typing import Optional, Dict, List, Callable, Tuple, Union

from deeporb.tensornet_utils import multi_outer_product
from deeporb.tensornet_utils import _aggregate_new, expand_to, find_distances, find_moment, _scatter_add

class TensorLinearMixing(nn.Module):
    def __init__(self,
                 nc : int,
                 lomax : int,
                 ) -> None:
        super().__init__()
        self.linear_list = nn.ModuleList([ # no bias for any?
            nn.LazyLinear(nc, bias=False) for l in range(lomax + 1) #lazy for b mixing
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

class TensorProductLayer(nn.Module):
    def __init__(self) -> None:
        #lin, lr, lout
        super().__init__()
        
    def forward(self,
                x : torch.Tensor, #li
                y : torch.Tensor, #lr
                combination : Tuple[int], #(li,lr,lout)
                ) -> torch.Tensor:
        x_way, y_way, z_way = combination
        return _aggregate_new(x, y, x_way, y_way, z_way)

class MaceLayer(nn.Module):
    def __init__(self,
                 nc    : int,
                 lomax : int=2,
                 linmax : int=2,
                 n_rbf : int=8,
                 cutoff : float=5.5,
                 ) -> None: 
        #Implement for l=2 and see if it's okay before generalizing
        super().__init__()
        self.nc = nc
        self.lomax = lomax
        self.linmax = linmax
        self.n_rbf = n_rbf
        self.cutoff = cutoff
        self.nu = 3 #only nu=3 implemented
        
        #Learnable radial for each layer?
        from cace.cace.modules import BesselRBF, GaussianRBF, GaussianRBFCentered
        from cace.cace.modules import PolynomialCutoff
        self.radial = BesselRBF(self.cutoff,self.n_rbf,trainable=True)
        self.cutoff_fn = PolynomialCutoff(self.cutoff)
        self.tensor_product_layer = TensorProductLayer()

        #Form edge messages to construct A
        self.edge_linear_mixing = TensorLinearMixing(self.nc,self.linmax)
        self.edge_combinations = []
        for x_way in range(self.lomax + 1): #l(r), l(h), lo
            for y_way in range(self.linmax + 1):
                for z_way in range(abs(y_way - x_way), min(self.lomax, x_way + y_way) + 1, 2):
                    self.edge_combinations.append((x_way, y_way, z_way))        
        self.rbf_list = nn.ModuleList([nn.Linear(n_rbf,nc,bias=True) for c in self.edge_combinations])
        
        #Form B message
        self.A_mixing = TensorLinearMixing(self.nc,self.lomax)
        self.nu2_combinations = [] #excluding 0s?
        for x_way in range(1, self.lomax + 1): #l(r), l(h), lo
            for y_way in range(x_way,self.lomax + 1):
                for z_way in range(abs(y_way - x_way), min(self.lomax, x_way + y_way) + 1, 2):
                    self.nu2_combinations.append((x_way, y_way, z_way))
        self.nu3_combinations = []
        for combo in self.nu2_combinations:
            l1,l2,l3 = combo
            x_way = l3
            for y_way in range(1,self.lomax + 1):
                for z_way in range(abs(y_way - x_way), min(self.lomax, x_way + y_way) + 1, 2):
                    self.nu3_combinations.append((l1, l2, x_way, y_way, z_way))
        self.B_mixing = TensorLinearMixing(self.nc,self.lomax)

        #Update input H
        self.B_update_mixing = TensorLinearMixing(self.nc,self.lomax)
        self.H_update_mixing = TensorLinearMixing(self.nc,self.linmax)

    def forward(self, data : Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        #Edge distances
        idx_i = data["edge_index"][0]
        idx_j = data["edge_index"][1]
        _, dij, _ = find_distances(data)
        dij = dij[:,None]
        rij = self.radial(dij) * self.cutoff_fn(dij)

        #Define T(r)
        x = torch.jit.annotate(Dict[int, torch.Tensor], {})
        for l in range(self.lomax+1):
            x[l] = find_moment(data, l)

        #Gather edge hidden features
        h = data["node_feats"]
        h = self.edge_linear_mixing(h) #technically not needed for 1st layer
        y = torch.jit.annotate(Dict[int, torch.Tensor], {})
        for l in range(self.linmax+1):
            y[l] = h[l][idx_j]
        
        #Tensor products to compute A basis messages
        edge_messages = torch.jit.annotate(Dict[int, torch.Tensor], {})
        for i,combination in enumerate(self.edge_combinations):
            lx, ly, lz = combination
            xin = x[lx][:,None,...]
            rin = expand_to(self.rbf_list[i](rij),len(xin.shape),-1)
            xin = rin * xin
            yin = y[ly]
            output_tensor = self.tensor_product_layer(xin,yin,combination)
            # print(lx,ly,lz,output_tensor[0])
            if lz not in edge_messages:
                edge_messages[lz] = output_tensor
            else:
                edge_messages[lz] += output_tensor
        
        #Sum to form A basis
        node_feats_A = torch.jit.annotate(Dict[int, torch.Tensor], {})
        n_atoms = data["atomic_numbers"].shape[0]
        for l in range(self.lomax+1):
            node_feats_A[l] = _scatter_add(edge_messages[l], idx_i, dim_size=n_atoms)
                
        #B basis
        node_feats_A = self.A_mixing(node_feats_A) #mix A
        node_feats_B = torch.jit.annotate(Dict[int, torch.Tensor], {})
        node_feats_B[0] = node_feats_A[0] #N x C x 3 x 3... -- we want to hstack on C
                
        #nu2
        for combo in self.nu2_combinations:
            lx, ly, lz = combo
            xin = node_feats_A[lx]
            yin = node_feats_A[ly]
            out = self.tensor_product_layer(xin,yin,combo)
            if lz not in node_feats_B.keys():
                node_feats_B[lz] = out
            else:
                node_feats_B[lz] = torch.hstack([node_feats_B[lz],out])
                
        #nu3
        for combo in self.nu3_combinations:
            lx, ly, lz, la, lb = combo
            xin = node_feats_A[lx]
            yin = node_feats_A[ly]
            zin = node_feats_A[la]
            out = self.tensor_product_layer(xin,yin,(lx,ly,lz))
            out = self.tensor_product_layer(out,zin,(lz,la,lb))
            node_feats_B[lb] = torch.hstack([node_feats_B[lb],out])
        
        #Project down to number of channels
        node_feats_B = self.B_mixing(node_feats_B) #problem looks like it's here on 2nd loop?

        #Message update
        node_feats = torch.jit.annotate(Dict[int, torch.Tensor], {})
        update_B = self.B_update_mixing(node_feats_B) #problem definitely here
        update_H = self.H_update_mixing(data["node_feats"])
        
        for l in range(self.lomax+1):
            if l in data["node_feats"].keys():
                node_feats[l] = update_B[l] + update_H[l]
            else:
                node_feats[l] = update_B[l]
        data["node_feats"] = node_feats
        return data

class CTP_MACE(nn.Module):
    def __init__(self,
                 nc    : int,
                 lomax : int=2,
                 layers : int=2,
                 n_rbf : int=8,
                 cutoff : float=5.5,
                 zs : List[int]=[1,6,7,8,9],
                 ) -> None: 
        super().__init__()
        self.nc = nc
        self.lomax = lomax
        self.linmax = 0
        self.n_rbf = n_rbf
        self.cutoff = cutoff
        self.zs = zs
        self.nu = 3
        
        self.encoder = nn.Sequential(NodeEncoder(self.zs),NodeEmbedding(node_dim=len(self.zs),embedding_dim=nc,random_seed=34))        
        self.mace_layers = [MaceLayer(nc,lomax=self.lomax,linmax=self.linmax,n_rbf=self.n_rbf,cutoff=self.cutoff)]
        for _ in range(layers-1):
            self.mace_layers.append(MaceLayer(nc,lomax=self.lomax,linmax=self.lomax,n_rbf=self.n_rbf,cutoff=self.cutoff))
        self.mace_layers = nn.ModuleList(self.mace_layers)
    
    def forward(self, data : Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        #Make initial A basis from h0, only l == 0 for now
        natom = data["atomic_numbers"].shape[0]
        h0 = torch.jit.annotate(Dict[int, torch.Tensor], {})
        h0[0] = self.encoder(data["atomic_numbers"])
        data["node_feats"] = h0
        
        #hstack output features at each time step for representation
        representation = [h0[0]]
        for mace_layer in self.mace_layers:
            data = mace_layer(data)
            representation.append(data["node_feats"][0])
        representation = torch.hstack(representation)

        #Message passing
        try:
            batch = data["batch"]
        except:
            batch = torch.zeros_like(data["atomic_numbers"])
        output = {
            "positions": data["positions"],
            "cell": data["cell"],
            "batch": batch,
            "node_feats": representation,
            "node_feats_A": data["node_feats"] #for equivariance testing
            } 
        
        return output