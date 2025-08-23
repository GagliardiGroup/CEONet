import os
import sys
import glob
import torch
import time
from cace.tasks import LightningTrainingTask
from deeporb.data import OrbDataset, OrbData
from deeporb.ceonet import CEONet
from cace.models import NeuralNetworkPotential
from deeporb.atomwise import AttentionAtomwise
from cace.tasks import GetLoss
from deeporb.metrics import Metrics
from deeporb.model_params import get_params

def get_data(d1,d2,root="../data",train_size=1,val_size=5000,test_size=5000):
    params = get_params(d1,d2)
    
    torch.multiprocessing.set_sharing_strategy('file_system')
    time_start = time.perf_counter()
    data = OrbData(data_path=f"{root}/{d1}_110k_{d2}.h5",train_split=train_size,val_split=val_size,test_split=test_size,
                   batch_size=params["BATCH_SIZE"],cutoff=params["CUTOFF"],avge0=params["AVGE0"],sigma=params["SIGMA"])
    data.setup()
    time_stop = time.perf_counter()
    print("Time elapsed in data production:",time_stop-time_start)
    return data

def get_model(d1,d2,n,root="../model_eval"):
    params = get_params(d1,d2)
    
    representation = CEONet(params["NC"],cutoff=params["CUTOFF"],n_rbf=params["N_RBF"],n_rsamples=params["N_RSAMPLES"],
                            stacking=params["STACKING"],irrep_mixing=params["IRREP_MIXING"],
                            linmax=params["LINMAX"],lomax=params["LOMAX"],layers=params["LAYERS"],
                            charge_embedding=params["CHARGE_EMBEDDING"])
    
    atomwise = AttentionAtomwise(
                        output_key='pred_energy',
                        n_hidden=[32,16],
                        attention_hidden_nc=128,
                        avge0=params["AVGE0"],sigma=params["SIGMA"],
                        bias=True
                       )
    
    model = NeuralNetworkPotential(
        input_modules=None,
        representation=representation,
        output_modules=[atomwise]
    )
    state_dict = torch.load(f"{root}/orbital_energy_results/{d1}_{d2}_{n}/best_model_state.pth")
    model.load_state_dict(state_dict)
    return model