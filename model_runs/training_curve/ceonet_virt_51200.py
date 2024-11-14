import os
import glob
import torch
import time
from cace.cace.tasks import LightningTrainingTask
from deeporb.data import OrbDataset, OrbData
from deeporb.ceonet import CEONet

#This is for running on the berkeley cluster

#Important argument : in_memory
#If True will load all data into memory at startup time, takes ~15 min of startup time with 5/130 of the training data
#But then training is much faster
num_train = 51200
valid_p = 0.2
orb_type = "virt"
logs_name = f"ceonet_{orb_type}_{num_train}"
cutoff = 4.0

on_cluster = False
import os
if 'SLURM_JOB_CPUS_PER_NODE' in os.environ.keys():
    on_cluster = True
if on_cluster:
    root = f"/global/scratch/users/king1305/data/aocart_{orb_type}.h5"
else:
    root = f"../data/aocart_{orb_type}.h5"
if orb_type == "occ":
    #For occ = 2 orbitals
    avge0 = -0.6637
    sigma = 0.2863
elif orb_type == "virt":
    #For occ = 0 orbitals
    avge0 = 0.6872
    sigma = 0.1880

in_memory = True if on_cluster else False
if not in_memory:
    torch.multiprocessing.set_sharing_strategy('file_system')
print("Making dataset...")
time_start = time.perf_counter()
batch_size = 128
data = OrbData(root=root,batch_size=batch_size,valid_p=valid_p,num_train=num_train,cutoff=cutoff,in_memory=in_memory,avge0=avge0,sigma=sigma)
time_stop = time.perf_counter()
print("Time elapsed:",time_stop-time_start)

representation = CEONet(64,cutoff=cutoff,n_rbf=8,n_rsamples=8,lomax=2,layers=4)

from cace.cace.models import NeuralNetworkPotential
from deeporb.atomwise import Atomwise, AttentionAtomwise
atomwise = AttentionAtomwise(
                    output_key='pred_energy',
                    n_hidden=[32,16],
                    attention_hidden_nc=128,
                    avge0=avge0,sigma=sigma,
                    bias=True
                   )

model = NeuralNetworkPotential(
    input_modules=None,
    representation=representation,
    output_modules=[atomwise]
)

from cace.cace.tasks import GetLoss
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
            avge0=avge0,sigma=sigma,
            per_atom=False,
        )
metrics = [e_metric]

#Init lazy layers
for batch in data.train_dataloader():
    exdatabatch = batch
    break
model(exdatabatch)

#Check for checkpoint and restart if found:
chkpt = None
dev_run = False
if os.path.isdir(f"lightning_logs/{logs_name}"):
    latest_version = None
    num = 0
    while os.path.isdir(f"lightning_logs/{logs_name}/version_{num}"):
        latest_version = f"lightning_logs/{logs_name}/version_{num}"
        num += 1
    if latest_version:
        chkpt = glob.glob(f"{latest_version}/checkpoints/*.ckpt")[0]
if chkpt:
    print("Checkpoint found!",chkpt)
    print("Restarting...")
    dev_run = False

progress_bar = True
if on_cluster:
    torch.set_float32_matmul_precision('medium')
    progress_bar = False

max_steps = 300000
b_size = min(len(data.train),batch_size)
steps_per_epoch = len(data.train)/b_size
check_val_every_n_epoch = max(int(1000/steps_per_epoch),1)
task = LightningTrainingTask(model,losses=losses,metrics=metrics,log_rmse=False,
                             scheduler_args={'mode': 'min', 'factor': 0.8, 'patience': 10},
                             optimizer_args={'lr': 0.001},
                             lr_frequency = check_val_every_n_epoch,
                            )
task.fit(data,dev_run=dev_run,max_steps=max_steps,check_val_every_n_epoch=check_val_every_n_epoch,
         chkpt=chkpt,name=logs_name,progress_bar=progress_bar)
