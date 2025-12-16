from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, List
import fnmatch

import uproot as ur
import numpy as np
import pandas as pd

from mach3sbitools.file_io.file_handler_base import FileHandlerBase


class RootFileHandler(FileHandlerBase):
    _input_file: ur.TTree
    
    def __init__(self, input_file_path: Path, ttree_name: str="posteriors", data_labels: Optional[List[str]] = None, theta_labels: Optional[List[str]] = None):
        self._ttree_name = ttree_name
        super().__init__(input_file_path, data_labels, theta_labels)
        
    def open_file(self, input_file_path: Path):
        return ur.open(f"{input_file_path}:{self._ttree_name}")
    
    def can_be_batched(self):
        return True
    
    def label_in_file(self, lab: str) -> bool:
        return any(fnmatch.fnmatch(s, lab) for s in self._input_file.keys())

    def get_all_branches(self):
        branches = []
        for b in self._theta_labels + self._data_labels:
            branches += [s for s in self._input_file.keys() if fnmatch.fnmatch(s, b)]
        
        return list(set(branches))

    def get_dim(self):
        return len(self.get_all_branches())
    
    def get_theta_labels(self):
        return list({s for l in self._theta_labels for s in self._input_file.keys() if fnmatch.fnmatch(s, l)})

    def get_x_labels(self):
        return list({s for l in self._data_labels for s in self._input_file.keys() if fnmatch.fnmatch(s, l)})

    def get_x_dim(self):
        return len(self.get_x_labels())
    
    def get_theta_dim(self):
        return len(self.get_theta_labels())

    def load_x_theta(self, **kwargs):
        with ThreadPoolExecutor() as executor:
            # Make sure we have loads of memory available!
            # Ensures we don't run into funny behaviour when uncompressing            
            total_memory_needed = self._input_file.uncompressed_bytes #in bytes
            
            all_branches = self.get_all_branches()
            verbose =  kwargs.get('verbose', False)
            
            if self._batch_mode:
                batch_num = kwargs.get('batch_num', None)
                if batch_num is None:
                    raise ValueError("Set up for batch mode but did not pass 'batch_num!")

                n_batches = kwargs.get('n_batches', 10)
                steps_per_batch = self._input_file.num_entries//n_batches
                
                # Loading WAYYYYY less than non-batched!
                total_memory_needed /= 0.5*n_batches
                
                kwargs['entry_start'] = steps_per_batch*batch_num 
                kwargs['entry_stop'] = steps_per_batch*(batch_num+1)
                
            if verbose:
                print(f"Using {executor._max_workers} threads and requiring {np.round(self._input_file.uncompressed_bytes*1e-9,3)} Gb memory")
                print("Using the following branches: ")
                for i in all_branches:
                    print(f"  -> {i}")
            
            if len(all_branches)==0:
                df_array: pd.DataFrame = self._input_file.arrays(self._input_file.keys(), cut=kwargs.get('cuts', []),
                                                            entry_start = kwargs.get('entry_start', 0), entry_stop = kwargs.get('entry_stop', self._input_file.num_entries),
                                                            library='pd', decompression_executor=executor, interpretation_executor=executor) # Load in ROOT TTree
            else:
                df_array: pd.DataFrame = self._input_file.arrays(all_branches, cut=kwargs.get('cuts', []),
                                                            entry_start = kwargs.get('entry_start', 0), entry_stop = kwargs.get('entry_stop', self._input_file.num_entries),
                                                            library='pd', array_cache=f"{total_memory_needed} b", decompression_executor=executor, interpretation_executor=executor) # Load in ROOT TTree

            if verbose:
                print(f"Converted TTree to pandas dataframe with {len(df_array)} elements")

        self._theta_labels = self.get_theta_labels()
        self._data_labels = self.get_x_labels()

        # Want to drop duplicate PARAMETERS since something fishy has happened!
        df_array.drop_duplicates(subset=self._theta_labels, inplace=True)
        
        # Now we split
        self._theta_loaded = self.device_handler.to_tensor(df_array[self._theta_labels])
        self._x_loaded = self.device_handler.to_tensor(df_array[self._data_labels])
        
        # Free our memory up        
        del df_array
        
        if(verbose):
            print("Done")