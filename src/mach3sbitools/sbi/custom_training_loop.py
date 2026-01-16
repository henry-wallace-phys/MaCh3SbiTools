from sbi.neural_nets.net_builders import build_nsf
from copy import deepcopy
from typing import Callable

import torch
from torch import eye, ones
from torch.optim import Adam, AdamW
from torch.utils import data

from sbi.analysis import pairplot

from mach3sbitools.sbi.sbi_mach3_interface import MaCh3SBIInterface, set_inference, set_inference_embedding

class MaCh3i