import numpy as np
import torch
import torch.nn as nn
import logging

from cace.cace.tasks import GetLoss
from cace.cace.tools import Metrics

import cace.cace as cace
from cace.cace.representations import Cace
from cace.cace.modules import CosineCutoff, MollifierCutoff, PolynomialCutoff
from cace.cace.modules import BesselRBF, GaussianRBF, GaussianRBFCentered
import lightning as L

def get_cace(cutoff=4.0):
    radial_basis = BesselRBF(cutoff=cutoff, n_rbf=6, trainable=True)
    cutoff_fn = PolynomialCutoff(cutoff=cutoff)
    cace_representation = Cace(
        zs=[1,6,7,8,9],
        n_atom_basis=3,
        embed_receiver_nodes=True,
        cutoff=cutoff,
        cutoff_fn=cutoff_fn,
        radial_basis=radial_basis,
        n_radial_basis=12,
        max_l=2,
        max_nu=3,
        num_message_passing=0,
        # type_message_passing=['Bchi'], #Why only this?
        # args_message_passing={'Bchi': {'shared_channels': False, 'shared_l': False}},
        # device=device,
        # timeit=False
    )
    return cace_representation

def default_losses(e_weight=0.1,f_weight=1000):
    #TBD: Can use these with pass_epoch_to_loss = True
    # def scaled_weight(epoch=0):
    #     if epoch < (tot_epochs * 2 // 3):
    #         return 0.1
    #     else:
    #         return 1000

    # def const_weight(epoch=0):
    #     return 1000
    
    e_loss = GetLoss(
                target_name='energy',
                predict_name='pred_energy',
                loss_fn=torch.nn.MSELoss(),
                loss_weight=e_weight,
                )
    f_loss = GetLoss(
                target_name='forces',
                predict_name='pred_forces',
                loss_fn=torch.nn.MSELoss(),
                loss_weight=f_weight,
            )
    return [e_loss,f_loss]

def default_metrics():
    e_metric = Metrics(
                target_name='energy',
                predict_name='pred_energy',
                name='e_metric',
                per_atom=True,
            )
    f_metric = Metrics(
                target_name='forces',
                predict_name='pred_forces',
                name='f_metric',
            )
    return [e_metric,f_metric]
    