import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from cace.modules import NodeEncoder, NodeEmbedding
from cace.tools import torch_geometric
from typing import Optional, Dict, List, Callable, Tuple, Union

from deeporb.cace_a import CaceA
from deeporb.tensornet import TensorProductLayer, TensorLinearMixing, TensorFeedForward
from deeporb.tensornet_utils import expand_to, find_distances, find_moment, _scatter_add
from deeporb.tensornet_utils import irrep_tensors, normalize_tensors, single_tensor_product
from deeporb.orbnet_utils import fix_orbints, calc_norm, reshape_ls

#New model that can handle d orbitals & charged systems
class MessagePassingLayer(nn.Module):
    def __init__(self,
                 nc    : int,
                 n_rbf : int,
                 avg_neighbors = 3,
                 lomax : int=2,
                 linmax : int=2,
                 mix : bool=True,
                 norm_at_beginning : bool=True,
                 stacking : bool=False,
                 irrep_mixing : bool=False,
                 linear_messages : bool=True,
                 ) -> None:
        super().__init__()
        self.lomax = lomax
        self.linmax = linmax
        self.avg_neighbors = avg_neighbors
        self.nc = nc
        self.stacking = stacking
        self.norm_at_beginning = norm_at_beginning
        self.irrep_mixing = irrep_mixing
        self.norm_func = normalize_tensors
        self.linear_layer = TensorLinearMixing
        self.linear_messages = linear_messages
        
        #hi x hi
        self.tp_hi_hi = TensorProductLayer(nc,max_x_way=self.linmax,max_y_way=self.linmax,max_z_way=self.lomax,stacking=stacking)
        self.hi_hi_mix_hi_left = self.linear_layer(nc,linmax)
        self.hi_hi_mix_hi_right = self.linear_layer(nc,linmax)

        #hi x hj attention
        self.att_hi_hj = TensorProductLayer(nc,max_x_way=self.linmax,max_y_way=self.linmax,max_z_way=0,stacking=True)
        self.att_hi_mix = TensorLinearMixing(nc,linmax)
        self.att_hj_mix = TensorLinearMixing(nc,linmax)
        
        #hi x r x hj
        self.tp_right = TensorProductLayer(nc,max_x_way=self.lomax,max_y_way=self.linmax,max_z_way=self.lomax,stacking=stacking)
        self.tp_left = TensorProductLayer(nc,max_x_way=self.linmax,max_y_way=self.lomax,max_z_way=self.lomax,stacking=stacking)
        self.hi_r_hj_mix_hi = self.linear_layer(nc,linmax)
        self.hi_r_hj_mix_r_hj = self.linear_layer(nc,lomax)
        self.hi_r_hj_mix_hj = self.linear_layer(nc,linmax)

        if self.linear_messages:
            #hi x r
            self.tp_hi_r = TensorProductLayer(nc,max_x_way=self.linmax,max_y_way=self.lomax,max_z_way=self.lomax,stacking=stacking)
            self.hi_r_mix_hi = self.linear_layer(nc,linmax)
    
            #r x hj
            self.tp_r_hj = TensorProductLayer(nc,max_x_way=self.lomax,max_y_way=self.linmax,max_z_way=self.lomax,stacking=stacking)
            self.r_hj_mix_hj = self.linear_layer(nc,linmax)

        #combo count for attention & radial
        combo_count = {l:0 for l in range(self.lomax+1)}
        tensor_layers = [self.tp_left]
        if self.linear_messages:
            tensor_layers.append(self.tp_hi_r)
            tensor_layers.append(self.tp_r_hj)
        for m in tensor_layers:
            for l in range(self.lomax+1):
                if self.stacking:
                    combo_count[l] += len([t for t in m.combinations if t[-1] == l])
                else:
                    if len([t for t in m.combinations if t[-1] == l]) > 0:
                        combo_count[l] += 1        
        if irrep_mixing:
            combo_count[0] += combo_count[2] #for irrep mixing
            combo_count[2] += combo_count[2] #for irrep mixing
        self.combo_count = combo_count

        #Radial mixing
        #I think it needs to be linear (or at least non-biased?) to preserve cutoff
        self.rbf_mixing_list = nn.ModuleList([
            nn.Linear(n_rbf,combo_count[l]*nc,bias=False) for l in range(lomax+1)
        ])

        def build_mlp(nout):
            net = nn.Sequential(
                nn.LazyLinear(nout,bias=True),nn.SiLU(),
                nn.Linear(nout,nout,bias=True),nn.SiLU(),
                nn.Linear(nout,nout,bias=True)
            )
            return net

        #Attention
        self.attention_net_list = nn.ModuleList([
            build_mlp(combo_count[l]*nc) for l in range(lomax+1)
        ])

        self.mix = mix
        if self.mix:
            self.linear_mixing = TensorFeedForward(nc,lomax)

    def calc_hi_r_hj(self, hi : Dict[int, torch.Tensor], u : Dict[int, torch.Tensor], hj : Dict[int, torch.Tensor]):
        #Calc hi x hj attention
        att_feed = []
        h1 = self.att_hi_mix(hi)
        h2 = self.att_hj_mix(hj)
        att_feed.append(self.att_hi_hj(h1,h2)[0])

        #hi x r x hj
        h1 = self.hi_r_hj_mix_hi(hi)
        h2 = self.hi_r_hj_mix_hj(hj)
        edge_messages = self.tp_right(u,h2)
        if self.irrep_mixing:
            edge_messages = irrep_tensors(edge_messages)
        edge_messages = self.hi_r_hj_mix_r_hj(edge_messages)
        edge_messages = self.tp_left(h1,edge_messages)

        if self.linear_messages:
            #hi x r
            h1 = self.hi_r_mix_hi(hi)
            h1 = self.tp_hi_r(h1,u)
            for l in range(self.lomax+1):
                edge_messages[l] = torch.hstack([edge_messages[l],h1[l]])

            #r x hj
            h1 = self.r_hj_mix_hj(hj)
            h1 = self.tp_r_hj(u,h1)
            for l in range(self.lomax+1):
                edge_messages[l] = torch.hstack([edge_messages[l],h1[l]])

        att_feed.append(edge_messages[0])
        att_feed = torch.hstack(att_feed)
        return edge_messages, att_feed

    def calc_edge_messages(self, data : Dict[str, torch.Tensor]) -> Dict[int, torch.Tensor]:
        n_atoms = data["atomic_numbers"].shape[0]
        idx_i = data["edge_index"][0]
        idx_j = data["edge_index"][1]
        rbf_ij = data["rbf_ij"]

        #Define Hi
        hi = torch.jit.annotate(Dict[int, torch.Tensor], {})
        for l in range(self.linmax + 1):
            hi[l] = data["node_feats"][l][idx_i]

        #Define Hj
        hj = torch.jit.annotate(Dict[int, torch.Tensor], {})
        for l in range(self.linmax + 1):
            hj[l] = data["node_feats"][l][idx_j]

        #Define U
        u = torch.jit.annotate(Dict[int, torch.Tensor], {})
        for l, rbf_mixing in enumerate(self.rbf_mixing_list):
            ones = torch.ones(rbf_ij.shape[0],self.nc,device=rbf_ij.device)
            u[l] = find_moment(data, l).unsqueeze(1) * expand_to(ones,l+2)

        #Calc edge features
        edge_messages, att_feed = self.calc_hi_r_hj(hi,u,hj)
        if self.irrep_mixing:
            edge_messages = irrep_tensors(edge_messages)
            
        #Calculate & apply attention & radial mixing
        for l in range(self.lomax+1):
            attention = self.attention_net_list[l](att_feed)
            rbf_mixed = self.rbf_mixing_list[l](rbf_ij)
            edge_messages[l] = edge_messages[l] * expand_to(attention,l+2)
            edge_messages[l] = edge_messages[l] * expand_to(rbf_mixed,l+2)

        return edge_messages

    def calc_node_message_hi_hj(self, data : Dict[str, torch.Tensor]) -> Dict[int, torch.Tensor]:
        #Calc edge messages
        edge_messages = self.calc_edge_messages(data)

        #Aggregate messages onto idx_i
        n_atoms = data["atomic_numbers"].shape[0]
        idx_i = data["edge_index"][0]
        node_message_hi_hj = torch.jit.annotate(Dict[int, torch.Tensor], {})
        for l in range(self.lomax+1):
            node_message_hi_hj[l] = _scatter_add(edge_messages[l], idx_i, dim_size=n_atoms)
            node_message_hi_hj[l] = node_message_hi_hj[l]/self.avg_neighbors #normalize
        return node_message_hi_hj

    def self_interaction(self, h0 : Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        h1 = self.hi_hi_mix_hi_left(h0)
        h2 = self.hi_hi_mix_hi_right(h0)
        node_message_hi_hi = self.tp_hi_hi(h1,h2)
        if self.linear_messages:
            for l in range(self.linmax+1):
                node_message_hi_hi[l] = torch.hstack([node_message_hi_hi[l],h0[l]])
        return node_message_hi_hi

    def forward(self, data : Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        #Normalize
        if self.norm_at_beginning:
            data["node_feats"] = self.norm_func(data["node_feats"])
        
        #Self interaction
        node_message_hi_hi = self.self_interaction(data["node_feats"])

        #Edge interaction
        node_message_hi_hj = self.calc_node_message_hi_hj(data)

        #Combine hi_hi & hi_hj
        combined_message = torch.jit.annotate(Dict[int, torch.Tensor], {})
        for l in range(self.lomax+1):
            if l in node_message_hi_hi:
                combined_message[l] = torch.hstack([node_message_hi_hi[l],node_message_hi_hj[l]])
            else:
                combined_message[l] = node_message_hi_hj[l]
        if self.irrep_mixing:
            combined_message = irrep_tensors(combined_message)
        
        #Return if not mix:
        if not self.mix:
            data["node_feats"] = combined_message
            return data

        #Feed forward
        data["node_feats"] = self.linear_mixing(combined_message)
        
        return data

class CEONet(L.LightningModule):
    def __init__(self,
                 nc    : int,
                 layers : int=4,
                 n_rbf : int=8,
                 n_rsamples : int=8,
                 lomax : int=2,
                 linmax : int=1,
                 cutoff : float=4.0,
                 stacking : bool=False,
                 irrep_mixing : bool=False,
                 charge_embedding : bool=False,
                 ) -> None:
        super().__init__()
        zs = list(range(1,54))
        self.zs = zs
        self.nc = nc
        self.lomax = lomax
        self.linmax = linmax
        self.layers = layers
        self.irrep_mixing = irrep_mixing
        self.norm_func = normalize_tensors
        self.charge_embedding = charge_embedding
        
        #Orbital input radial functions
        rsamples = [nn.Parameter(torch.linspace(0,cutoff,n_rsamples)) for l in range(self.linmax+1)]
        self.rsamples = nn.ParameterList(rsamples)
        self.rsample_mixing_list = nn.ModuleList([
            nn.Linear(n_rsamples, nc, bias=False)
            for l in range(self.linmax + 1)
        ])

        #Radial functions
        from cace.modules import BesselRBF, GaussianRBF, GaussianRBFCentered
        from cace.modules import PolynomialCutoff
        self.radial = GaussianRBFCentered(n_rbf=n_rbf, cutoff=cutoff, start=1.0, trainable=True)
        self.cutoff_fn = PolynomialCutoff(cutoff)
        self.orbital_mp = MessagePassingLayer(nc=nc,n_rbf=n_rbf,lomax=self.lomax,linmax=self.linmax,
                                              stacking=stacking,irrep_mixing=irrep_mixing,
                                              linear_messages = False,
                                              mix=False,avg_neighbors=1,norm_at_beginning=False)

        #CACE representation
        n_atom_basis = 4
        n_radial_basis = 12
        self.representation = CaceA(
            zs = zs,
            n_atom_basis = n_atom_basis,
            n_rbf = n_rbf,
            n_radial_basis = n_radial_basis,
            embed_receiver_nodes = True,
            cutoff = cutoff,
            max_l = self.lomax
        )
        cace_size = n_radial_basis * (n_atom_basis**2)
        self.cace_size = cace_size

        #Mixing of CACE w/ orb features
        self.a_mix = TensorFeedForward(nc,lomax)
            
        #Message Passing
        self.mp_layers = nn.ModuleList([
            MessagePassingLayer(nc,n_rbf=n_rbf,lomax=self.lomax,linmax=self.lomax,
                                stacking=stacking,irrep_mixing=irrep_mixing,
                                linear_messages = True,
                                avg_neighbors=3) for _ in range(layers)
        ])

        #Layer normalization at end
        nu3_combos = []
        for l1 in range(1,self.lomax+1):
            for l2 in range(1,self.lomax+1):
                l3 = l1+l2
                if l3 <=self.lomax:
                    nu3_combos.append((l1,l3,l2))
        self.nu3_combos = nu3_combos
        num_b = 1 + self.lomax + len(self.nu3_combos)
        self.b_size = nc*num_b*(layers + 1)
        self.b_norm = nn.LayerNorm([self.b_size],bias=True)

        #Append charges at end
        charges = list(range(-10,11))
        if self.charge_embedding:
            self.charge_embedding_layer = nn.Sequential(
                NodeEncoder(charges),
                NodeEmbedding(node_dim=len(charges), embedding_dim=nc, random_seed=34)
            )

    def calc_rbf(self, data : Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        _, dij, _ = find_distances(data)
        data["rbf_ij"] = self.radial(dij[:,None]) * self.cutoff_fn(dij[:,None])
        return data

    def make_h0(self, data : Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        try:
            batch = data["batch"]
        except:
            data["batch"] = torch.zeros_like(data["atomic_numbers"])
    
        data = fix_orbints(data)
        
        natom = data["atomic_numbers"].shape[0]
        h0 = torch.jit.annotate(Dict[int, torch.Tensor], {})
        for l in range(self.linmax+1):
            c_idx = data[f"orbints_{l}"][:,:-1]
            atm_num = data[f"orbints_{l}"][:,-1]
            
            c = data["c"][c_idx.ravel()].reshape(c_idx.shape)
            alpha = data[f"orbfloats_{l}"][:,0]
            weight = data[f"orbfloats_{l}"][:,1]
    
            #Sample primitive orbitals
            norm = calc_norm(alpha,l)
            arg = alpha[:,None] * (-(self.rsamples[l]**2)[None,:]) #NP x rsamples
            prim = norm[:,None,:] * torch.exp(arg)[:,:,None] # NP x rsamples x dim
            prim = prim.movedim(1,-1)
            prim = self.rsample_mixing_list[l](prim) 
            prim = prim.movedim(-1,1) #NP x C x dim
    
            #Multiply by coeffs and sum
            orb = expand_to(weight,3,dim=-1) * prim #NP x C x dim
            out = orb * c[:,None,:] # NP x C x dim
            summed_out = torch.zeros(natom,out.shape[1],out.shape[2],device=out.device)
            summed_out = torch.index_add(summed_out,0,atm_num,out)
            
            #Reshape
            if l >= 2:
                summed_out = reshape_ls(summed_out,l)        
            h0[l] = summed_out.squeeze()
        return h0

    def make_b(self,dct):
        #nu = 1
        bfeats = [dct[0]]
    
        #nu = 2
        for l in range(1,self.lomax+1):
            bfeats.append(single_tensor_product(dct[l],dct[l],(l,l,0)))
    
        #nu = 3
        for l1,l2,l3 in self.nu3_combos:
            x = single_tensor_product(dct[l1],dct[l2],(l1,l2,l3))
            bfeats.append(single_tensor_product(x,dct[l3],(l3,l3,0)))
        
        bfeats = torch.hstack(bfeats)
        return bfeats

    def forward(self, data : Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        data = self.calc_rbf(data)

        #Make Orbital Features
        data["node_feats"] = self.make_h0(data)
        data = self.orbital_mp(data)
        data["node_feats"] = self.norm_func(data["node_feats"])

        #Get CACE features
        adct = self.representation(data)
        if self.irrep_mixing:
            adct = irrep_tensors(adct)
        adct = self.norm_func(adct)

        #Stack & Mix
        dct = torch.jit.annotate(Dict[int, torch.Tensor], {})
        for l in range(self.lomax+1):
            dct[l] = torch.hstack([data["node_feats"][l],adct[l]])
        mixed = self.a_mix(dct)
        
        #Make B basis
        bfeats = [self.make_b(mixed)]

        #Message passing layers
        data["node_feats"] = mixed
        for mp_layer in self.mp_layers:
            data = mp_layer(data)
            bfeats.append(self.make_b(data["node_feats"]))

        #Normalize b
        bfeats = torch.hstack(bfeats)
        bfeats = self.b_norm(bfeats)

        #Append charge embedding
        if self.charge_embedding:
            ce = self.charge_embedding_layer(data["charge"])
            bfeats = torch.hstack([bfeats,ce])

        data["node_feats"] = bfeats
        data["node_feats_A"] = mixed

        output = {
            "positions": data["positions"],
            "cell": data["cell"],
            "batch": data["batch"],
            "node_feats": data["node_feats"],
            "node_feats_A": data["node_feats_A"],
        }
        return output