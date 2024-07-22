import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import torch_geometric.nn as geonn
from cace.cace.tools import scatter_sum
from deeporb.data import OrbData

class CaceOrb(L.LightningModule):
    def __init__(self, lr=1e-3, cutoff=4.0):
        self.lr = lr
        self.cutoff = cutoff
        super().__init__()
        self.nnact = nn.ReLU()

        from deeporb.defaults import arg_to_default
        args = dict(arg_to_default)
        args["zs"] = [1,6,7,8,9]
        args["num_message_passing"] = 0
        args["cutoff"] = self.cutoff #default 4.0

        #Defaults that we can mess with to increase/decrease rep size
        # args["n_atom_basis"] = 3
        # args["n_radial_basis"] = 8
        # args["max_l"] = 3
        # args["max_nu"] = 3
        args["n_atom_basis"] = 3
        args["n_radial_basis"] = 8
        args["max_l"] = 3
        args["max_nu"] = 3
        
        ##representation
        from deeporb.cace import Cace
        from cace.cace.modules import PolynomialCutoff, BesselRBF, Atomwise
        cutoff_fn = PolynomialCutoff(cutoff=args["cutoff"], p=args["cutoff_fn_p"])
        radial_basis = BesselRBF(cutoff=args["cutoff"], n_rbf=args["n_rbf"], trainable=args["trainable_rbf"])
        self.representation = Cace(
        zs=args["zs"], n_atom_basis=args["n_atom_basis"], embed_receiver_nodes=args["embed_receiver_nodes"],
        cutoff=args["cutoff"], cutoff_fn=cutoff_fn, radial_basis=radial_basis,
        n_radial_basis=args["n_radial_basis"], max_l=args["max_l"], max_nu=args["max_nu"],
        num_message_passing=args["num_message_passing"])
        #With default args -- above gives 468

        #Message Passing
        gcn_dim = 468
        n_message_passing = 2
        self.invr0 = nn.Parameter((1.0 / args["cutoff"]) * (torch.rand(1) + 0.5)) #edge weights
        self.gat = geonn.GAT(in_channels=-1,hidden_channels=gcn_dim,num_layers=n_message_passing)
        
        #MLP Readouts (by occupancy):
        netlst = [nn.LazyLinear(256),
                  nn.ReLU(),
                  nn.LazyLinear(128),
                  nn.ReLU(),
                  nn.LazyLinear(64),
                  nn.ReLU(),
                  nn.LazyLinear(1)] #Same output net?
        self.elnet_2 = nn.Sequential(*netlst)
        netlst = [nn.LazyLinear(256),
                  nn.ReLU(),
                  nn.LazyLinear(128),
                  nn.ReLU(),
                  nn.LazyLinear(64),
                  nn.ReLU(),
                  nn.LazyLinear(1)]
        self.elnet_0 = nn.Sequential(*netlst)
    
    def forward(self,data,prnt=False):
        #Combine Features
        rep = self.representation(data)
        B = rep["node_feats"]
        B = B.reshape(B.shape[0], -1)
        O = data["orbdata"]
        X = torch.concat([B,O],axis=1)
        if prnt:
            print("Representation length:",X.shape[-1])

        #Message Passing
        E = data["edge_index"]
        edge_weights = torch.exp(-1.0 * self.invr0 * torch.squeeze(rep["edge_lengths"]))
        X = self.gat(X,E,edge_weights)

        #MLP Readout
        occ = data["occ"]
        ys = torch.zeros(X.shape[0]).to(X.device)
        for occnum in torch.unique(occ):
            if occnum == 2:
                net = self.elnet_2
            elif occnum == 0:
                net = self.elnet_0
            X_idx = torch.where(occ == occnum)[0]
            ys[X_idx] = torch.squeeze(net(X[X_idx,:]))

        y = scatter_sum(
            src=ys, 
            index=data["batch"], 
            dim=0)
        y = torch.squeeze(y, -1)
        return y

    def loss(self,yhat,y):
        return torch.nn.L1Loss()(yhat,y)

    def mae(self,yhat,y):
        return torch.nn.L1Loss()(yhat,y)

    def training_step(self,train_batch,batch_idx):
        y = train_batch["energy"]
        yhat = self.forward(train_batch)
        loss = self.loss(yhat,y)
        batch_size = len(y)
        self.log('train_mae_loss', loss, batch_size=batch_size)
        return loss

    def validation_step(self,val_batch,val_idx):
        y = val_batch["energy"]
        yhat = self.forward(val_batch)
        loss = self.loss(yhat,y)
        mae = self.mae(yhat,y)
        batch_size = len(y)
        # self.log('val_loss', loss, batch_size=batch_size)
        self.log('val_mae_loss', mae, batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        #self.parameters -- freebie from lightning, cool!
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

#Init models and data
CUTOFF = 4.0
data = OrbData(batch_size=128,cutoff=CUTOFF)
model = CaceOrb(cutoff=CUTOFF)

#Run test batch through for lazy layers
train_loader = data.train_dataloader()
for batch in train_loader:
    exdata = batch
    break
model.forward(exdata,prnt=True)

#Train
chkpt = None
trainer = L.Trainer()
trainer.fit(model,data,ckpt_path=chkpt)