from sbi.neural_nets.net_builders import build_nsf
from copy import deepcopy
from typing import Callable

import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch import eye, ones
from torch.optim import Adam, AdamW
from torch.utils import data

from sbi.analysis import pairplot
from sbi.utils.user_input_checks import (
    process_prior,
    process_simulator,
)



from mach3sbitools.sbi.sbi_mach3_interface import MaCh3SBIInterface, set_inference, set_inference_embedding

class MaCh3Inference:
    def __init__(self, prior, device):
        self._prior = prior
        self._device = device
        self.
        
        
    def append_simulations(self, x, theta):
        
    
    def train(self):
        