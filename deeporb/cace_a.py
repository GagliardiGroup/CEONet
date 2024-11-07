import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Union, Sequence, Callable, Optional, Tuple, List

import itertools
from cace.cace.modules import Dense, ResidualBlock, build_mlp
from cace.cace.modules import NodeEncoder, NodeEmbedding, EdgeEncoder
from cace.cace.modules import AngularComponent, SharedRadialLinearTransform
from cace.cace.tools import torch_geometric
from cace.cace.tools import elementwise_multiply_3tensors, scatter_sum

#reverted
class CaceA(nn.Module):
    def __init__(
        self,
        zs=[1,6,7,8,9],
        n_atom_basis = 4,
        n_rbf = 8,
        n_radial_basis = 12,
        embed_receiver_nodes=True,
        cutoff=4.0,
        max_l=2,
        atom_embedding_random_seed = [34,34],
    ):
        super().__init__()
        self.zs = zs
        self.nz = len(zs)
        self.n_atom_basis = n_atom_basis
        self.cutoff = cutoff
        self.n_rbf = n_rbf
        self.n_radial_basis = n_radial_basis
        self.max_l = max_l

        self.node_onehot = NodeEncoder(self.zs)
        # sender node embedding
        self.node_embedding_sender = NodeEmbedding(
                         node_dim=self.nz, embedding_dim=self.n_atom_basis, random_seed=atom_embedding_random_seed[0]
                         )
        if embed_receiver_nodes:
            self.node_embedding_receiver = NodeEmbedding(
                         node_dim=self.nz, embedding_dim=self.n_atom_basis, random_seed=atom_embedding_random_seed[1]
                         )
        else:
            self.node_embedding_receiver = self.node_embedding_sender
        self.edge_coding = EdgeEncoder(directed=True)
        self.n_edge_channels = n_atom_basis**2

        from cace.cace.modules import BesselRBF, GaussianRBF, GaussianRBFCentered
        from cace.cace.modules import PolynomialCutoff
        # radial_basis = BesselRBF(cutoff=cutoff, n_rbf=n_rbf, trainable=True)
        radial_basis = GaussianRBFCentered(n_rbf=n_rbf, cutoff=cutoff, trainable=True)
        cutoff_fn = PolynomialCutoff(cutoff=cutoff, p=5)
        self.radial_basis = radial_basis
        
        self.cutoff_fn = cutoff_fn
        self.angular_basis = AngularComponent(self.max_l)
        # print("Angular list:",self.angular_basis.lxlylz_list)
        radial_transform = SharedRadialLinearTransform(
                                max_l=self.max_l,
                                radial_dim=self.n_rbf,
                                radial_embedding_dim=self.n_radial_basis,
                                channel_dim=self.n_edge_channels
                                )
        self.radial_transform = radial_transform

    def forward(self, data: Dict[str, torch.Tensor]):
        node_feats_list = []
        node_feats_A_list = []

        # Embeddings
        ## code each node/element in one-hot way
        node_one_hot = self.node_onehot(data['atomic_numbers'])
        ## embed to a different dimension
        node_embedded_sender = self.node_embedding_sender(node_one_hot)
        node_embedded_receiver = self.node_embedding_receiver(node_one_hot)
        ## get the edge type
        encoded_edges = self.edge_coding(edge_index=data["edge_index"],
                                         node_type=node_embedded_sender,
                                         node_type_2=node_embedded_receiver,
                                         data=data)

        # compute angular and radial terms
        # _, _, _ = find_distances(data)
        edge_lengths = data["dij"]
        edge_vectors = data["uij"]
        radial_component = self.radial_basis(edge_lengths[:,None]) 
        radial_cutoff = self.cutoff_fn(edge_lengths[:,None])
        angular_component = self.angular_basis(edge_vectors)

        # combine
        # 4-dimensional tensor: [n_edges, radial_dim, angular_dim, embedding_dim]
        edge_attri = elementwise_multiply_3tensors(
                      radial_component * radial_cutoff,
                      angular_component,
                      encoded_edges
        )

        # sum over edge features to each node
        # 4-dimensional tensor: [n_nodes, radial_dim, angular_dim, embedding_dim]
        n_nodes = data['positions'].shape[0]
        node_feat_A = scatter_sum(src=edge_attri, 
                                  index=data["edge_index"][1], 
                                  dim=0, 
                                  dim_size=n_nodes)

        # mix the different radial components
        #N x r x l x e^2 --> l x N x r x e^2
        #s x y z x2 xy xz y2 yz z2
        a_basis = self.radial_transform(node_feat_A).movedim(2,0)
        a_basis = a_basis.reshape(a_basis.shape[0],a_basis.shape[1],-1) #l x N x (r x e^2)
        adct = torch.jit.annotate(Dict[int, torch.Tensor], {})
        adct[0] = a_basis[0]
        
        for l in range(1,self.max_l+1):
            dim = [3]*l + [a_basis.shape[-2],a_basis.shape[-1]]
            adct[l] = torch.zeros(*dim,device=data["positions"].device)
        
        for i,(lx,ly,lz) in enumerate(self.angular_basis.lxlylz_list[1:]):
            l = lx + ly + lz
            idx = [0]*lx + [1]*ly + [2]*lz
            for p in itertools.permutations(idx):
                adct[l][p] = a_basis[i+1]
        for l in adct.keys():
            adct[l] = adct[l].movedim(-1,0).movedim(-1,0)
        return adct
