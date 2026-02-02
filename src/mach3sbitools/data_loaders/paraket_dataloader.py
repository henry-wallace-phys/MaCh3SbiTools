from torch.utils.data import Dataset
from pathlib import Path
import pyarrow as pa

class ParaketDataset(Dataset):
    def __init__(self, data_folder: Path):
        self.data_folder = data_folder
        self.files = list(data_folder.glob("*.feather"))
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        table = pa.feather.read_feather(str(file_path))
        theta = table['theta'].to_pylist()
        x = table['x'].to_pylist()
        return theta, x
    
    