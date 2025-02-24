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

#Write here
LOGS_NAME = "sto3g_occ_100000"
DATA_NAME = "sto3g_occ.pt"
SUBSET_SIZE = "100000"

#Model params
CUTOFF = 7.6
LINMAX = 1
LOMAX = 2
NC = 16
LAYERS = 2
N_RBF = 16
N_RSAMPLES = 16
STACKING = True
IRREP_MIXING = False
CHARGE_EMBEDDING = False

#Data params
BATCH_SIZE = 128
#NUM_TRAIN = 5120
#NUM_VAL = 10000
IN_MEMORY = True
AVGE0 = -0.6605
SIGMA = 0.2801

#Training params
#Set DEV_RUN to true to test if a single val batch works, etc.
DEV_RUN = False
LR = 0.001
MAX_STEPS = 600000

#Note: you will need to adjust this section to path it to the correct .h5
on_cluster = False
#import os
# if 'SLURM_JOB_CPUS_PER_NODE' in os.environ.keys():
#     on_cluster = True
# if on_cluster:
#     root = f"/global/scratch/users/king1305/data/{DATA_NAME}"
# else:
#     root = f"../data/{DATA_NAME}"

#root = f"../data/{DATA_NAME.split('_')[0]}/{DATA_NAME}"
def main():

    root = f"/eagle/DeepOrb/{DATA_NAME.split('_')[0]}/subset/{DATA_NAME.split('.')[0]}_{SUBSET_SIZE}.pt"

    print(root)

    #in_memory = True if on_cluster else False
    #if on_cluster:
    #    in_memory = IN_MEMORY
    #if not in_memory:

    torch.multiprocessing.set_sharing_strategy('file_system')
    in_memory = True

    print("Making dataset...")
    time_start = time.perf_counter()
    data = OrbData(root=root,batch_size=BATCH_SIZE,cutoff=CUTOFF,in_memory=in_memory,avge0=AVGE0,sigma=SIGMA)
    #data = OrbData(root=root,batch_size=BATCH_SIZE,num_val=NUM_VAL,num_train=NUM_TRAIN,cutoff=CUTOFF,in_memory=in_memory,avge0=AVGE0,sigma=SIGMA)
    time_stop = time.perf_counter()
    print("Time elapsed:",time_stop-time_start)

    representation = CEONet(NC,cutoff=CUTOFF,n_rbf=N_RBF,n_rsamples=N_RSAMPLES,stacking=STACKING,irrep_mixing=IRREP_MIXING,
                            linmax=LINMAX,lomax=LOMAX,layers=LAYERS,charge_embedding=CHARGE_EMBEDDING)

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

    #Init lazy layers
    for batch in data.train_dataloader():
        exdatabatch = batch
        break
    model.cuda()
    model(exdatabatch.cuda())

    #Check for checkpoint and restart if found:
    chkpt = None
    dev_run = DEV_RUN
    if os.path.isdir(f"/eagle/DeepOrb/deeporb/lightning_logs/{LOGS_NAME}"):
        latest_version = None
        num = 0
        while os.path.isdir(f"lightning_logs/{LOGS_NAME}/version_{num}"):
            latest_version = f"lightning_logs/{LOGS_NAME}/version_{num}"
            num += 1
        if latest_version:
            chkpt_list = glob.glob(f"{latest_version}/checkpoints/*.ckpt")
            if len(chkpt_list)>0:
                chkpt = chkpt_list[0]
    if chkpt:
        print("Checkpoint found!",chkpt)
        print("Restarting...")
        dev_run = False

    progress_bar = True
    #if on_cluster:
    #    torch.set_float32_matmul_precision('medium')
    #    progress_bar = False

    torch.set_float32_matmul_precision('medium')

    #task = LightningTrainingTask(model,losses=losses,metrics=metrics,metric_typ="mae",
    #                             logs_directory="lightning_logs",name=LOGS_NAME,
    #                             scheduler_args={'mode': 'min', 'factor': 0.8, 'patience': 10},
    #                             optimizer_args={'lr': LR},
    #                            )

    task = LightningTrainingTask(model,losses=losses,metrics=metrics,
                                 logs_directory="lightning_logs",name=LOGS_NAME,
                                 scheduler_args={'mode': 'min', 'factor': 0.8, 'patience': 10},
                                 optimizer_args={'lr': LR},
                                )

    task.fit(data,dev_run=dev_run,max_steps=MAX_STEPS,chkpt=chkpt,progress_bar=progress_bar)

if __name__ == "__main__":
    main()
