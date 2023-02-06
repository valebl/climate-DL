import numpy as np
import pickle
import sys

import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_convert

from torch_geometric.data import Data

class Dataset_pr(Dataset):

    def __init__(self, args, pad=2, lat_dim=43, lon_dim=49, shift=2): #path, input_file, target_file, data_file, idx_file, net_type, get_key=False, mask_file=None, weights_file=None, **kwargs):
        super().__init__()
        self.pad = pad
        self.lat_dim = lat_dim # number of points in the GRIPHO rectangle (0.25 grid)
        self.lon_dim = lon_dim
        self.space_idxs_dim = self.lat_dim * self.lon_dim
        self.shift = shift # relative shift between GRIPHO and ERA5 (idx=0 in ERA5 corresponds to 2 in GRIPHO)
        self.args = args
        self.length = None

    def _load_data_into_memory(self): # path, input_file, target_file, data_file, idx_file, net_type, mask_file, weights_file):
        raise NotImplementedError
    
    def __len__(self):
        return self.length


class Dataset_pr_ae(Dataset_pr):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input, self.idx_to_key = self._load_data_into_memory(self.args)
    
    def _load_data_into_memory(self):
        with open(self.args.input_path + self.args.input_file, 'rb') as f:
            input = pickle.load(f) 
        with open(self.args.input_path + self.args.idx_file,'rb') as f:
            idx_to_key = pickle.load(f)
        self.length = len(idx_to_key)
        return input, idx_to_key

    def __getitem__(self, idx, lat_shift=29, lon_shift=0):
        k = self.idx_to_key[idx]   
        time_idx = k // self.space_idxs_dim
        space_idx = k % self.space_idxs_dim
        lat_idx = space_idx // self.lon_dim - lat_shift
        lon_idx = space_idx % self.lon_dim - lon_shift
        #-- derive input
        input = self.input[time_idx - 24 : time_idx+1, :, :, lat_idx - self.pad + 2 : lat_idx + self.pad + 4, lon_idx - self.pad + 2 : lon_idx + self.pad + 4]
        return input


class Dataset_pr_reg(Dataset_pr):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input, self.idx_to_key, self.target, self,data, self.mask = self._load_data_into_memory()
    
    def _load_data_into_memory(self):
        with open(self.args.input_path + self.args.input_file, 'rb') as f:
            input = pickle.load(f)
        with open(self.args.input_path + self.args.idx_file,'rb') as f:
            idx_to_key = pickle.load(f)
        with open(self.args.input_path + self.args.target_file, 'rb') as f:
            target = pickle.load(f)        
        with open(self.args.input_path + self.args.data_file, 'rb') as f:
            data = pickle.load(f)
        with open(self.args.input_path + self.args.mask_file, 'rb') as f:
            mask = pickle.load(f)
        self.length = len(idx_to_key)
        return input, idx_to_key, target, data, mask
    
    def __getitem__(self, idx, lat_shift=29, lon_shift=0):
        k = self.idx_to_key[idx]   
        time_idx = k // self.space_idxs_dim
        space_idx = k % self.space_idxs_dim
        lat_idx = space_idx // self.lon_dim - lat_shift
        lon_idx = space_idx % self.lon_dim - lon_shift
        #-- derive input
        input = self.input[time_idx - 24 : time_idx+1, :, :, lat_idx - self.pad + 2 : lat_idx + self.pad + 4, lon_idx - self.pad + 2 : lon_idx + self.pad + 4]
        #-- derive gnn data
        y = torch.tensor(self.target[k])
        edge_index = torch.tensor(self.data[space_idx]['edge_index'])
        x = torch.tensor(self.data[space_idx]['x'])
        mask = torch.tensor(self.mask[k].astype(bool)) 
        data = Data(x=x, edge_index=edge_index, y=y, mask=mask)
        return input, data

class Dataset_pr_cl(Dataset_pr):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input, self.idx_to_key, self.target, self.data = self._load_data_into_memory()
    
    def _load_data_into_memory(self):
        with open(self.args.input_path + self.args.input_file, 'rb') as f:
            input = pickle.load(f)
        with open(self.args.input_path + self.args.idx_file,'rb') as f:
            idx_to_key = pickle.load(f)
        with open(self.args.input_path + self.args.target_file, 'rb') as f:
            target = pickle.load(f)        
        with open(self.args.input_path + self.args.data_file, 'rb') as f:
            data = pickle.load(f)
        self.length = len(idx_to_key)
        return input, idx_to_key, target, data
 
    def __getitem__(self, idx, lat_shift=29, lon_shift=0):
        k = self.idx_to_key[idx]   
        time_idx = k // self.space_idxs_dim
        space_idx = k % self.space_idxs_dim
        lat_idx = space_idx // self.lon_dim - lat_shift
        lon_idx = space_idx % self.lon_dim - lon_shift
        #-- derive input
        input = self.input[time_idx - 24 : time_idx+1, :, :, lat_idx - self.pad + 2 : lat_idx + self.pad + 4, lon_idx - self.pad + 2 : lon_idx + self.pad + 4]
        #-- derive gnn data
        y = torch.tensor(self.target[k])
        edge_index = torch.tensor(self.data[space_idx]['edge_index'])
        x = torch.tensor(self.data[space_idx]['x'])
        data = Data(x=x, edge_index=edge_index, y=y)
        return input, data

def custom_collate_fn_ae(batch):
    input = np.array(batch)
    input = default_convert(input)
    return input

def custom_collate_fn_gnn(batch):
    input = np.array([item[0] for item in batch])
    data = [item[1] for item in batch]
    input = default_convert(input)
    return input, data
    
