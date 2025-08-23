#Note: everything is in BOHR (7.6 bohr cutoff)
default_params = {
    "CUTOFF":7.6,
    "LOMAX":2,
    "NC":16,
    "LAYERS":2,
    "N_RBF":16,
    "N_RSAMPLES":16,
    "BATCH_SIZE":128,
    "IN_MEMORY":True,
    "LR":0.001,
    "MAX_STEPS":300000,
    "STACKING" : True,
    "IRREP_MIXING" : False,
    "CHARGE_EMBEDDING" : False,
}

#These are good estimates of the mean and std
mean_std = {
    "qh9_occ":[-0.5144,0.2229],
    "qh9_virt":[1.4346,0.9854],
    "sto3g_occ":[-0.6605,0.2801],
    "sto3g_virt":[0.6892,0.1825],
    "tm_occ":[-0.6154,0.2862],
    "tm_virt":[0.6896,0.2229],
}

#Fill in dict below to generate scripts
script_gen_dct = {
    "sto3g_occ":{
        "LINMAX":1,
    },
    "sto3g_virt":{
        "LINMAX":1,
    },
    "tm_occ":{
        "LINMAX":2,
        "LAYERS":1,
        "BATCH_SIZE":32,
    },
    "tm_virt":{
        "DATA_NAME":"tm_5000_virt.h5",
        "LINMAX":2,
        "LAYERS":1,
        "BATCH_SIZE":32,
    },
    "qh9_occ":{
        "LINMAX":2,
    },
    "qh9_virt":{
        "LINMAX":2,
    },
}

def get_params(d1,d2):
    datak = f"{d1}_{d2}"
    dct = {k:v for k,v in default_params.items()}
    mean, std = mean_std[datak]
    for k, v in script_gen_dct[datak].items():
        dct[k] = v
    dct["AVGE0"] = mean
    dct["SIGMA"] = std
    return dct


        
        