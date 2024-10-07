from cace.cace.tasks.lightning import LightningData

def WaterData(batch_size=4,cutoff=5.5):
    root = "/home/king1305/Apps/cacefit/fit-water/water.xyz"
    avge0 = {1: -187.6043857100553, 8: -93.80219285502734}
    data = LightningData(root,batch_size=batch_size,cutoff=cutoff,atomic_energies=avge0)
    return data

def EthanolData(batch_size=10,cutoff=4):
    root = "/home/king1305/Apps/cacefit/fit-ethanol/train-n1000.xyz"
    avge0 = {1: -14223.773967801664, 6: -4741.257989267255, 8: -2370.6289946336274}
    data = LightningData(root,batch_size=batch_size,cutoff=cutoff,atomic_energies=avge0)
    return data
    
# edata = EnergyData(batch_size=1)
# for batch in edata.train_dataloader():
#     exdatabatch = batch
#     break
# exdatabatch