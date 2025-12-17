from mach3sbitools.mach3_interface.mach3_interface import MaCh3Interface
from mach3sbitools.sbi.mach3_prior import MaCh3TorchPrior
from mach3sbitools.utils.device_handler import TorchDeviceHander

from sbi.utils import BoxUniform

from pathlib import Path
import pickle
import torch
from abc import ABC, abstractmethod
from tqdm.autonotebook import tqdm

class MaCh3SBIInterface(ABC):
    device_handler = TorchDeviceHander()
    def __init__(self, handler: MaCh3Interface, n_rounds: int, samples_per_round: int):
        self._simulator = handler
        self._n_rounds = n_rounds
        # self._prior = MaCh3TorchPrior(handler, self.device_handler)
        
        self._config_file = self._simulator.get_config_file()
        
        low, high = handler.get_bounds()
                
        self._prior = BoxUniform(self.device_handler.to_tensor(low),
                                 self.device_handler.to_tensor(high))
        
        self._posterior = self._proposal = self._prior
        self._samples_per_round = samples_per_round

    @property
    def posterior(self):
        return self._posterior

    @property
    def x0(self):
        return self.device_handler.to_tensor(self._simulator.get_data_bins())
    
    def sample(self, n_samples: int, **kwargs):
        return self._posterior.sample((n_samples, ), **kwargs)

    def get_x_vals(self, theta: torch.Tensor):
        # Convert to numpy array on  CPU
        theta_cpu = theta.cpu().numpy()
        valid_theta = []
        valid_x = []
        for t in theta_cpu:
            try:
                sims = self._simulator.simulate(t)
                valid_x.append(sims)
                valid_theta.append(t)
            except Exception:
                continue
            
        if not len(valid_theta):
            raise Exception("Proposal has failed, no valid values found!")
            
        return self.device_handler.to_tensor(valid_x), self.device_handler.to_tensor(valid_theta)
    
    def simulate(self, **kwargs):
        theta = self._proposal.sample((self._samples_per_round, ), **kwargs)
        x, theta = self.get_x_vals(theta)
        return x, theta
    
    def train(self, sampling_settings, training_settings):
        self._proposal = self._prior
        for _ in tqdm(range(self._n_rounds)):
            self.training_iter(sampling_settings, training_settings)
            
    @abstractmethod
    def training_iter(self, sampling_settings, training_settings):
        ...
        
    # Allows us to pickle the file!
    def __getstate__(self):
        state = self.__dict__.copy()
        # remove unpickleable C++ handler
        state["_simulator"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # recreate C++ handler, singleton so it's the same across ALL processes!
        self._simulator = MaCh3Interface(self._config_file)

    def save(self, output: Path):
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, 'wb') as handle:
            pickle.dump(self, handle)
    
    @staticmethod
    def load_from_file(input: Path)->'MaCh3SBIInterface':
        if not input.exists():
            raise FileNotFoundError(f"Cannot find file: {input}")
        
        with open(input, 'rb') as handle:
            input_file: MaCh3SBIInterface = pickle.load(handle)
        
        return input_file