import torch
import torch.nn as nn
from cace.cace.modules.atomwise import Atomwise
from cace.cace.modules.forces import Forces
from cace.cace.models import AtomisticModel
from typing import Dict, Optional, List, Tuple

class NetworkPotential(AtomisticModel):
    def __init__(
        self,
        representation: nn.Module = None,
        n_out : int = 1,
    )->None:
        super().__init__() #calls lightning super
        self.representation = representation
        
        #Default readout: forces + atomistic
        #Will generate "pred_energy" and "pred_forces" in output
        self.energy_key = "pred_energy"
        self.forces_key = "pred_force"
        atomwise = Atomwise(n_layers=3,
                            output_key=self.energy_key,
                            n_hidden=[32,16],
                            n_out=n_out,
                            use_batchnorm=False,
                            add_linear_nn=True)
        
        forces = Forces(energy_key=self.energy_key,
                        forces_key=self.forces_key)
        output_modules = [atomwise,forces]
        
        self.output_modules = nn.ModuleList(output_modules)

        #These are in 'atomwise' and 'forces'
        self.collect_derivatives() #looks for 'required derivatives' in all modules
        self.collect_outputs() #looks for 'model outputs' in all modules

    def forward(self, 
                data: Dict[str, torch.Tensor], 
                training: bool = True, #needs to be true by default for plugin to work 
                compute_stress: bool = False, 
                compute_virials: bool = False
                ) -> Dict[str, torch.Tensor]:
        # initialize derivatives for response properties
        data = self.initialize_derivatives(data)

        if 'stress' in self.model_outputs or 'CACE_stress' in self.model_outputs:
            compute_stress = True
        # for m in self.input_modules:
        #     data = m(data, compute_stress=compute_stress, compute_virials=compute_virials)

        data = self.representation(data)
        
        for m in self.output_modules:
            data = m(data, training=training)

        results = self.extract_outputs(data)

        return results