from torch.utils.data import Dataset
from pathlib import Path
from pyarrow import feather
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from tqdm import tqdm
import torch

class ParaketDataset(Dataset):
    def __init__(self, data_folder: Path):
        self.data_folder = data_folder
        self.files = list(data_folder.glob("*.feather"))
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        table = feather.read_feather(str(file_path))
        
        # Cannot directly to_numpy without error
        theta = np.array(table['theta'].to_list(), dtype=np.float32)
        x = np.array(table['x'].to_list(), dtype=np.float32)
        return torch.tensor(theta), torch.tensor(x)
    
    def make_data_plot(self):
        # We concatenate all data for plotting then make histograms for each dimension and save to a PDF
        all_theta = []
        all_x = []
        for idx in tqdm(range(len(self)), desc="loading dataset"):
            theta, x = self[idx]
            all_theta.extend(theta)
            all_x.extend(x)
        
        print("Making into np array")
        x = np.array(all_x)
        theta = np.array(all_theta)
        
        with PdfPages(self.data_folder / "model_distribution.pdf") as pdf:
            # We first plot x distribution
            for dim in tqdm(range(x.shape[1]), desc="plotting x distributions"):
                plt.figure(figsize=(10, 5))
                plt.hist(x[:, dim], bins=50, color='green', histtype='step')
                plt.title(f'X Dimension {dim} Distribution')
                plt.xlabel('X')
                plt.ylabel('Frequency')
                pdf.savefig()
                plt.close()
            
            # We now do the same for theta
            for dim in tqdm(range(theta.shape[1]), desc="plotting theta distributions"):
                plt.figure(figsize=(10, 5))
                plt.hist(theta[:, dim], bins=50, color='blue', histtype='step')
                plt.title(f'Theta Dimension {dim} Distribution')
                plt.xlabel('Theta')
                plt.ylabel('Frequency')
                pdf.savefig()
                plt.close()