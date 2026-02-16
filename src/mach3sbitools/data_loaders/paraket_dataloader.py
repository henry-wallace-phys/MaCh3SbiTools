from torch.utils.data import Dataset, TensorDataset
from pathlib import Path
from pyarrow import feather
import numpy as np
import torch
from tqdm import tqdm

class ParaketDataset(Dataset):
    """File-level dataset — one __getitem__ = one feather file.
    Use .to_tensor_dataset() before training."""

    def __init__(self, data_folder: Path):
        self.data_folder = data_folder
        self.files = sorted(data_folder.glob("*.feather"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        table = feather.read_feather(str(self.files[idx]))
        theta = np.array(table['theta'].to_list(), dtype=np.float32)
        x = np.array(table['x'].to_list(), dtype=np.float32)
        return torch.from_numpy(theta), torch.from_numpy(x)

    def to_tensor_dataset(
        self,
        device: str = 'cpu',
        nuisance_mask: torch.BoolTensor | None = None,
    ) -> TensorDataset:
        """
        Load all feather files into RAM once and return a flat TensorDataset.
        Optionally mask out nuisance parameters from theta.
        """
        all_theta, all_x = [], []

        for idx in tqdm(range(len(self)), desc="Pre-loading dataset"):
            theta, x = self[idx]
            if nuisance_mask is not None:
                theta = theta[:, nuisance_mask]
            all_theta.append(theta)
            all_x.append(x)

        theta_tensor = torch.cat(all_theta, dim=0).to(device)
        x_tensor = torch.cat(all_x, dim=0).to(device)

        print(f"Loaded {theta_tensor.shape[0]:,} simulations | "
              f"θ: {theta_tensor.shape[1]}D  x: {x_tensor.shape[1]}D | "
              f"RAM: {(theta_tensor.nbytes + x_tensor.nbytes) / 1e9:.2f} GB")

        return TensorDataset(theta_tensor, x_tensor)