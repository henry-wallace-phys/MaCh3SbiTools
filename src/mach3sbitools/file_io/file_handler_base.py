'''
Generic file handler concept. 
'''

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List
import pandas as pd
import numpy as np

from mach3sbitools.utils.device_handler import TorchDeviceHander

class FileHandlerBase(ABC):
    def __init__(self, input_file_path: Path, data_labels: Optional[List[str]] = None, theta_labels: Optional[List[str]] = None, batch_mode: bool=False):
        
        self.device_handler= TorchDeviceHander()
        
        if not input_file_path.exists():
            raise FileNotFoundError(f"Error cannot open {input_file_path}")
        
        self._input_file = self.open_file(input_file_path)

        self._data_labels: List[str] = []
        if data_labels:
            self.add_data_labels(data_labels)
        
        self._theta_labels: List[str] = []
        if theta_labels:
            self.add_theta_labels(theta_labels)
        
        self._x_loaded = None
        self._theta_loaded = None
        
        self._batch_mode = batch_mode
    
        self.set_batch_mode(batch_mode)
    
    
    def can_be_batched(self):
        return False
    
    def set_batch_mode(self, mode: bool):
        if not self.can_be_batched():
            raise ValueError("Set batch mode but cannot batch")
        self._batch_mode = mode

    
    @property
    def x(self):
        return self._x_loaded
    
    @property
    def theta(self):
        return self._theta_loaded
        
    @abstractmethod
    def open_file(self, input_path):
        ...
    
    @abstractmethod
    def get_dim()->int:
        ...
        
    @abstractmethod
    def get_x_dim()->int:
        ...

    @abstractmethod
    def get_theta_dim()->int:
        ...

    
    @abstractmethod
    def label_in_file(self, lab: str)->bool:
        ...
        
    def check_labels_in_file(self, labels: List[str]):
        not_in = []
        
        for label in labels:
            if self.label_in_file(label):
                continue
            not_in.append(label)
        
        if not_in:
            raise ValueError(f"Tried to add {not_in} as labels but these are not defined!")
        
        
    def add_theta_labels(self, theta_labels: List[str]):
        self.check_labels_in_file(theta_labels)
        self._theta_labels.extend(theta_labels)
    
    def add_data_labels(self, data_labels: List[str]):
        self.check_labels_in_file(data_labels)
        self._data_labels.extend(data_labels)
    
    @abstractmethod
    def load_x_theta(self, **kwargs):
        ...
