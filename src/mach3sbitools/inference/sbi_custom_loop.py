from pathlib import Path
from typing import List

import torch
import numpy as np

from sbi.neural_nets.net_builders import build_maf



import torch
from torch.optim import AdamW
from torch.utils import data
from torch.utils.tensorboard.writer import SummaryWriter

from sbi.analysis import pairplot
from sbi.utils import BoxUniform


from mach3sbitools.mach3_interface.mach3_simulator import MaCh3Simulator
from mach3sbitools.mach3_interface.mach3_prior import create_mach3_prior
from mach3sbitools.data_loaders.paraket_dataloader import ParaketDataset
from mach3sbitools.utils.device_handler import TorchDeviceHander

class NPETrainer:
    def __init__(
        self,
        mach3: object,
        data_folder: Path,
        data_batch_size: int|None = None,
        summary: SummaryWriter |None = None,
    ):
        self._device_handler = TorchDeviceHander()
        self._prior = create_mach3_prior(mach3, device=self._device_handler.device)
        self._mach3 = mach3
        self._dataset = ParaketDataset(data_folder)
        self._dataloader = data.DataLoader(self._dataset, batch_size=data_batch_size, shuffle=True)
    
        self._losses = []
    
        if summary is None:
            self._summary = SummaryWriter()
        else:
            self._summary = summary
    
    def build_network(
        self,
        num_transforms: int = 10,
        hidden_features: int = 100,
        num_blocks: int = 2,
    ) -> torch.nn.Module:
        dummy_data = data.DataLoader(self._dataset, batch_size=1)
        dummy_theta, dummy_x = next(iter(dummy_data))

        maf_estimator = build_maf(
            dummy_theta,
            dummy_x,
            num_transforms=num_transforms,
            hidden_features=hidden_features,
            num_blocks=num_blocks,
            device=self._device_handler.device
        )
        
        return maf_estimator
    
    
    def train(
        self,
        neural_net: torch.nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 50,
    ) -> List[float]:
        optimizer = AdamW(neural_net.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Now split data loader into test/train
        num_train_samples = int(0.8 * len(self._dataset))
        num_test_samples = len(self._dataset) - num_train_samples
        train_dataset, test_dataset = data.random_split(self._dataset, [num_train_samples, num_test_samples])
        train_dataloader = data.DataLoader(train_dataset, batch_size=self._dataloader.batch_size, shuffle=True)
        test_dataloader = data.DataLoader(test_dataset, batch_size=self._dataloader.batch_size, shuffle=False)
        
        best_loss = float('inf')
        epoch = 0
        patience_counter = 0
        
        while True:
        
            epoch_losses = []
            for theta_batch, x_batch in train_dataloader:
                theta_batch = theta_batch.to(self._device_handler.device)
                x_batch = x_batch.to(self._device_handler.device)
                
                optimizer.zero_grad()
                loss = -neural_net.log_prob(theta_batch, x_batch).mean()
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
            
            # Get validation loss
            with torch.no_grad():
                val_losses = []
                for theta_batch, x_batch in test_dataloader:
                    theta_batch = theta_batch.to(self._device_handler.device)
                    x_batch = x_batch.to(self._device_handler.device)
                    
                    val_loss = -neural_net.log_prob(theta_batch, x_batch).mean()
                    val_losses.append(val_loss.item())
                
                avg_val_loss = np.mean(val_losses)
                self._summary.add_scalar("Validation Loss", avg_val_loss, epoch)
            
            
            avg_epoch_loss = np.mean(epoch_losses)
            self._losses.append(avg_epoch_loss)
            self._summary.add_scalar("Loss", avg_epoch_loss, epoch)
            print(f"Epoch {epoch + 1}/{epoch}, Loss: {avg_epoch_loss:.4f}")

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
            epoch += 1
                
        return self._losses
    
    def sample_posterior(
        self,
        neural_net: torch.nn.Module,
        num_samples: int = 1000,
    ) -> torch.Tensor:
        posterior = neural_net.build_posterior(self._prior)
        samples = posterior.sample((num_samples,), x=None)
        return samples
    
    def save_model(
        self,
        neural_net: torch.nn.Module,
        save_path: Path,
    ):
        torch.save(neural_net.state_dict(), save_path)
        