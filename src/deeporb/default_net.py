import os
import glob
import torch
import time
from cace.tasks import LightningTrainingTask
from deeporb.data import OrbDataset, OrbData
from deeporb.ceonet import CEONet

mean_std_dct = {
    "qh9_occ":[-0.5144,0.2229],
    "qh9_virt":[1.4346,0.9854],
    "qh9_v12":[0.5353,0.3168],
    "sto3g_occ":[-0.6605,0.2801],
    "sto3g_virt":[0.6892,0.1825],
    "tm_occ":[-0.6154,0.2862],
    "tm_virt":[0.6896,0.2229],
    "qh9_all":[1.1572,1.1341],
    "sto3g_all":[-0.0188,0.7169],
    "tm_all":[-0.0234,0.7003],
    "kl_constructed":[0,1],
    "qh9_class_occ":[0,1],
    "qh9_class_forb":[0,1],
    "qh9_balanced":[0,1],
    "sto3g_balanced":[0,1],
    "sto3g_class_occ":[-0.0188,0.7169],
    "b3lyp_occ":[-0.4527,0.2246],
    "b3lyp_virt":[0.4318,0.1652],
    "nan":[0,1],
}

data_dct = {
    "sto3g_occ":"sto3g_5000_occ.h5",
    "sto3g_virt":"sto3g_5000_virt.h5",
    "tm_occ":"tm_5000_occ.h5",
    "tm_virt":"tm_5000_virt.h5",
    "qh9_occ":"qh9_5000_occ.h5",
    "qh9_virt":"qh9_5000_virt.h5",
    "qh9_v12":"qh9_5000_v12.h5",
    "sto3g_all":"sto3g_5000_all.h5",
    "qh9_all":"qh9_5000_all.h5",
    "tm_all":"tm_5000_all.h5",
    "kl_constructed":"kl_constructed.h5",
    "qh9_class_occ":"qh9_5000_all.h5",
    "qh9_class_forb":"qh9_5000_all.h5",
    "qh9_balanced":"qh9_balanced.h5",
    "sto3g_balanced":"sto3g_balanced.h5",
    "sto3g_class_occ":"sto3g_5000_all.h5",
    "b3lyp_occ":"b3lyp_5000_occ.h5",
    "b3lyp_virt":"b3lyp_5000_virt.h5",
}

def default_net(mean_std,chkpt=None,nr=16,batch_size=1):
    CUTOFF = 7.6
    LINMAX = 2
    LOMAX = 2
    NC = 16
    LAYERS = 2
    N_RBF = nr
    N_RSAMPLES = nr
    STACKING = True
    IRREP_MIXING = False
    CHARGE_EMBEDDING = False
    if "sto3g" in mean_std:
        LINMAX = 1
    if "b3lyp" in mean_std:
        LINMAX = 1
    if "tm" in mean_std:
        LAYERS = 1
    if "kl" in mean_std:
        LAYERS = 1

    AVGE0, SIGMA = mean_std_dct[mean_std]
    
    #Training params
    DEV_RUN = False #change!
    LR = 0.001
    MAX_STEPS = 300000

    representation = CEONet(NC,cutoff=CUTOFF,n_rbf=N_RBF,n_rsamples=N_RSAMPLES,stacking=STACKING,irrep_mixing=IRREP_MIXING,
                            linmax=LINMAX,lomax=LOMAX,layers=LAYERS,charge_embedding=CHARGE_EMBEDDING)

    from cace.models import NeuralNetworkPotential
    from deeporb.atomwise import AttentionAtomwise

    if "kl" in mean_std:
        atomwise = AttentionAtomwise(
                            output_key='pred_energy',
                            n_hidden=[32,16],
                            attention_hidden_nc=128,
                            avge0=AVGE0,sigma=SIGMA,
                            bias=True
                           )
        
        atomwise_label = AttentionAtomwise(
                            output_key='pred_label_logit',
                            n_hidden=[32,16],
                            n_out=4,
                            attention_hidden_nc=128,
                            bias=True
                           )
        
        model = NeuralNetworkPotential(
            input_modules=None,
            representation=representation,
            output_modules=[atomwise,atomwise_label]
        )
    else:
        atomwise = AttentionAtomwise(
                            output_key='pred_energy',
                            n_hidden=[32,16],
                            attention_hidden_nc=128,
                            avge0=AVGE0,sigma=SIGMA,
                            bias=True
                           )
    
        model = NeuralNetworkPotential(
            input_modules=None,
            representation=representation,
            output_modules=[atomwise]
        )

    if not chkpt:
        return model
    else:
        state_dct = torch.load(chkpt,weights_only=True)
        model.load_state_dict(state_dct)
        return model

def default_net_task(mean_std,chkpt=None,nr=16,batch_size=128):
    AVGE0, SIGMA = mean_std_dct[mean_std]
    
    from cace.tasks import GetLoss
    from deeporb.metrics import Metrics
    e_loss = GetLoss(
                target_name="energy_ssh",
                predict_name='pred_energy',
                loss_fn=torch.nn.MSELoss(),
                loss_weight=1,
                )
    losses = [e_loss]
    
    e_metric = Metrics(
                target_name="energy",
                predict_name='pred_energy',
                name='e',
                metric_keys=["mae"],
                avge0=AVGE0,sigma=SIGMA,
                per_atom=False,
            )
    metrics = [e_metric]
    
    model = default_net(mean_std=mean_std,nr=nr)
    LR = 0.001
    task = LightningTrainingTask(model,losses=losses,metrics=metrics,
                                 logs_directory="lightning_logs",name="default",
                                 scheduler_args={'mode': 'min', 'factor': 0.8, 'patience': 10},
                                 optimizer_args={'lr': LR},
                                )

    data_name = data_dct[mean_std]
    
    if chkpt:
        task.load(chkpt)
    return task

# def default_data(h5fn,mean_std="qh9_5000_occ",in_memory=False,batch_size=128):
#     num_train = 51200
#     num_val = 5000
#     if "tm_all" in mean_std:
#         if batch_size == 128:
#             batch_size = 32
#     if "kl_constructed" in mean_std:
#         if batch_size == 128:
#             batch_size = 32
#         num_train = None
#         num_val = None
#     avge0, sigma = mean_std_dct[mean_std]
#     data = OrbData(h5fn,batch_size=batch_size,num_val=num_val,num_train=num_train,cutoff=7.6,in_memory=in_memory,avge0=avge0,sigma=sigma)
#     data.setup()
#     return data