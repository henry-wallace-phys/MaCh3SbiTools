from pathlib import Path
from typing import List, Optional
import torch
import pickle as pkl
import fnmatch
import numpy as np

from sbi.inference import NPE
from sbi.neural_nets import posterior_nn

from mach3sbitools.mach3_interface.mach3_simulator import MaCh3Simulator
from mach3sbitools.data_loaders.paraket_dataloader import ParaketDataset
from mach3sbitools.utils.device_handler import TorchDeviceHander

class MaCh3SBIInterface:
    def __init__(self, mach3_name: str, config_file: Path, nuisance_pars: Optional[List[str]]=None):
        self.simulator = MaCh3Simulator(mach3_name, config_file, nuisance_pars)
        
        self.nuisance_pars = nuisance_pars
        self.mach3_name = mach3_name
        
        self.dataset = None
        self.inference = None
        self.posterior = None
        
        self._density_estimator = None
        self.device_handler = TorchDeviceHander()
        
        self._curr_x = None
        self._curr_theta = None
        
    def set_dataset(self, data_folder: Path) -> None:
        self.dataset = ParaketDataset(data_folder)
    
    def create_posterior(self, hidden_features: int = 50, num_transforms: int = 2, dropout_probability=0.05, num_blocks=3) -> None:
        """
        Creates an SBI posterior using Neural Posterior Estimation (NPE).

        Args:
            hidden_features (int): Number of hidden features in the neural network.
            num_transforms (int): Number of transforms in the neural network.
        Returns:
            NPE: The SBI posterior object.
        """
        neural_net = posterior_nn(
            model="maf",
            hidden_features=hidden_features,
            num_transforms=num_transforms,
            dropout_probability=dropout_probability,
            num_blocks=num_blocks
        )

        self.inference = NPE(prior=self.simulator.prior, density_estimator=neural_net, device=self.device_handler.device)

    def append_data(self, idx: int, data_device='cpu') -> None:
        if self.dataset is None:
            raise ValueError("Dataset not set. Please set the dataset using set_dataset method before appending data.")

        if self.inference is None:
            raise ValueError("Posterior not created. Please create the posterior using create_posterior method before appending data.")
    
        theta, x = self.dataset[idx]
        parameter_names = self.simulator.mach3_wrapper.get_parameter_names()
        if self.nuisance_pars is not None:
            mask = [not any(fnmatch.fnmatch(name, n)for n in self.nuisance_pars) for name in parameter_names]
            theta = theta[:, mask]

        if self._curr_x is None:
            self._curr_x = x.to(data_device)
            self._curr_theta = theta.to(data_device)

        else:
            self._curr_x = torch.cat([self._curr_x, x.to(data_device)], dim=0)
            self._curr_theta = torch.cat([self._curr_theta, theta.to(data_device)], dim=0)

    def train_posterior(self, save_file: Path | None = None, **kwargs) -> None:
        if self.dataset is None:
            raise ValueError("Dataset not set. Please set the dataset using set_dataset method before training.")
        if self.inference is None:
            raise ValueError("Inference not created. Please create the inference using create_posterior method before training.")
        
        if save_file is None:
            save_file = Path(f"{self.mach3_name}_sbi_inference.pkl")
        
        if self._curr_x is None or self._curr_theta is None:
            raise ValueError("No data appended. Please append data using append_data method before training.")
        
        self.inference.append_simulations(self._curr_theta, self._curr_x, data_device=self._curr_theta.device)
        self._curr_x = self._curr_theta = None  # Clear current data after appending
                
        d = self.inference.train(**kwargs)
        self._density_estimator = d
        self.save_inference(save_file)
        print(f"Training complete after {self.inference.epoch} epochs. Inference saved to {save_file}.")
    
    def build_posterior(self) -> None:        
        self.posterior = self.inference.build_posterior(self._density_estimator)
    
    def sample_posterior(self, num_samples: int = 1000, x: List[float] | None = None, **kwargs) -> torch.Tensor:
        print(f"Generating {num_samples} from posterior")
        
        self.posterior = self.inference.build_posterior(self._density_estimator)
        
        if x is None:
            x = self.simulator.mach3_wrapper.get_data_bins()
        
        x_tensor = torch.tensor([x], dtype=torch.float32, device=self.device_handler.device)
        samples = self.posterior.sample((num_samples,), x=x_tensor, **kwargs)
        return samples
    
    def save_inference(self, file_path: Path) -> None:
        if self.inference is None:
            raise ValueError("Inference not built. Please build the inference§ using build_posterior method before saving.")
        with open(file_path, 'wb') as f:
            pkl.dump(self.inference, f)
            
    def load_inference(self, file_path: Path) -> None:
        with open(file_path, 'rb') as f:
            self.inference = pkl.load(f)