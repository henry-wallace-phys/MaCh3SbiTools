from pathlib import Path
from typing import List
import torch
import pickle as pkl

from sbi.inference import NPE
from sbi.neural_nets import posterior_nn

from mach3sbitools.mach3_interface.mach3_simulator import MaCh3Simulator
from mach3sbitools.data_loaders.paraket_dataloader import ParaketDataset
from mach3sbitools.utils.device_handler import TorchDeviceHander

class MaCh3SBIInterface:
    def __init__(self, mach3_name: str, config_file: Path):
        self.simulator = MaCh3Simulator(mach3_name, config_file)
        self.mach3_name = mach3_name
        
        self.dataset = None
        self.inference = None
        self.posterior = None
        
        self._density_estimator = None
        self.device_handler = TorchDeviceHander()
        
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

    def append_data(self, idx: int, nuisance_vars: List[str]) -> None:
        if self.dataset is None:
            raise ValueError("Dataset not set. Please set the dataset using set_dataset method before appending data.")

        if self.inference is None:
            raise ValueError("Posterior not created. Please create the posterior using create_posterior method before appending data.")
    
        theta, x = self.dataset[idx]
        if nuisance_vars is not None:
            # Nuisance vars filter by substring i.e. if "xsec_" is in the param name, it is a nuisance var
            theta = [
                t for t, name in zip(theta, self.simulator.mach3_wrapper.get_parameter_names())
                if not any(nuisance in name for nuisance in nuisance_vars)
            ]

        theta = torch.tensor([theta], dtype=torch.float32, device='cpu')
        x = torch.tensor([x], dtype=torch.float32, device='cpu')

        self.inference.append_simulations(theta, x)

    def train_posterior(self, save_file: Path | None = None, checkpoint_interval: int = 100,
                        lr_decay: float = 0.1, min_lr: float = 1e-6, **kwargs) -> None:
        if self.dataset is None:
            raise ValueError("Dataset not set. Please set the dataset using set_dataset method before training.")
        if self.inference is None:
            raise ValueError("Inference not created. Please create the inference using create_posterior method before training.")
        
        if save_file is None:
            save_file = Path(f"{self.mach3_name}_sbi_inference.pkl")
        
        
        max_num_epochs = kwargs.get('max_num_epochs', 100)
        learning_rate = kwargs.get('learning_rate', 1e-4)
        
        while True:
            d = self.inference.train(**kwargs)
            self.save_inference(save_file)
            
            # We're still going
            if (self.inference.epoch-1) == max_num_epochs :
                max_num_epochs += checkpoint_interval
                kwargs['max_num_epochs'] = max_num_epochs
                continue
            
            if learning_rate >= min_lr:
                learning_rate *= lr_decay
                kwargs['learning_rate'] = learning_rate
                kwargs['max_num_epochs'] += checkpoint_interval
                continue
 
            break
        
        print(f"Training complete after {self.inference.epoch} epochs. Inference saved to {save_file}.")
        self._density_estimator = d
    
    
    def build_posterior(self) -> None:
        if self._density_estimator is None:
            raise ValueError("Density estimator not trained. Please train the density estimator using train_posterior method before building the posterior.")
        
        self.posterior = self.inference.build_posterior(self._density_estimator)
    
    def sample_posterior(self, num_samples: int = 1000, x: List[float] | None = None, **kwargs) -> torch.Tensor:
        if self.posterior is None:
            raise ValueError("Posterior not built. Please build the posterior using build_posterior method before sampling.")
        
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