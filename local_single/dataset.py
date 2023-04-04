import pickle
import sys

import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_convert

import time

class Dataset_pr(Dataset):

    def __init__(self, args, pad=2, lat_dim=16, lon_dim=31):
        super().__init__()
        self.pad = pad
        self.lat_low_res_dim = lat_dim # number of points in the GRIPHO rectangle (0.25 grid)
        self.lon_low_res_dim = lon_dim
        self.space_low_res_dim = self.lat_low_res_dim * self.lon_low_res_dim
        self.args = args
        self.length = None

    def _load_data_into_memory(self):
        raise NotImplementedError
    
    def __len__(self):
        return self.length

class Dataset_pr_ae(Dataset_pr):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input, self.idx_to_key = self._load_data_into_memory()
    
    def _load_data_into_memory(self):
        with open(self.args.input_path + self.args.input_file, 'rb') as f:
            input = pickle.load(f) 
        with open(self.args.input_path + self.args.idx_file,'rb') as f:
            idx_to_key = pickle.load(f)
        self.length = len(idx_to_key)
        return input, idx_to_key

    def __getitem__(self, idx):
        k = self.idx_to_key[idx]   
        time_idx = k // self.space_low_res_dim
        space_idx = k % self.space_low_res_dim
        lat_idx = space_idx // self.lon_low_res_dim
        lon_idx = space_idx % self.lon_low_res_dim
        input = torch.zeros((25, 5, 5, 6, 6))
        input[:] = self.input[time_idx - 24 : time_idx+1, :, :, lat_idx - self.pad + 2 : lat_idx + self.pad + 4, lon_idx - self.pad + 2 : lon_idx + self.pad + 4]
        return input

class Dataset_pr_gnn(Dataset_pr):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input, self.idx_to_key, self.target, self.graph, self.mask_target, self.subgraphs = self._load_data_into_memory()

    def _load_data_into_memory(self):
        with open(self.args.input_path + self.args.input_file, 'rb') as f:
            input = pickle.load(f)
        with open(self.args.input_path + self.args.idx_file,'rb') as f:
            idx_to_key = pickle.load(f)
        with open(self.args.input_path + self.args.target_file, 'rb') as f:
            target = pickle.load(f)        
        with open(self.args.input_path + self.args.graph_file, 'rb') as f:
            graph = pickle.load(f)
        with open(self.args.input_path + self.args.mask_target_file, 'rb') as f:
            mask_target = pickle.load(f)
        with open(self.args.input_path + self.args.subgraphs_file, 'rb') as f:
            subgraphs = pickle.load(f)
        self.length = len(idx_to_key)
        self.low_res_abs = abs(graph.low_res)
        return input, idx_to_key, target, graph, mask_target, subgraphs
    
    def __getitem__(self, idx):
        k = self.idx_to_key[idx]   
        time_idx = k // self.space_low_res_dim
        space_idx = k % self.space_low_res_dim
        lat_idx = space_idx // self.lon_low_res_dim
        lon_idx = space_idx % self.lon_low_res_dim
        #-- derive input
        input = torch.zeros((25, 5, 5, 6, 6))
        input = self.input[time_idx - 24 : time_idx+1, :, :, lat_idx - self.pad + 2 : lat_idx + self.pad + 4, lon_idx - self.pad + 2 : lon_idx + self.pad + 4]
        #-- derive gnn data
        subgraph = self.subgraphs[space_idx].cuda()
        y = self.target[subgraph.mask_subgraph, time_idx] # shape = (n_nodes_subgraph,)
        subgraph["y"] = y.cuda()
        return input, subgraph

def custom_collate_fn_ae(batch):
    input = np.array(batch)
    input = default_convert(input)
    return input

def custom_collate_fn_gnn(batch):
    input = np.array([item[0] for item in batch])
    data = [item[1] for item in batch]
    input = default_convert(input)
    return input, data
    

def custom_collate_fn_ae(batch):
    input = torch.stack(batch)
    input = default_convert(input)
    return input

def custom_collate_fn_gnn(batch):
    input = torch.stack([item[0] for item in batch]) # shape = (batch_size, 25, 5, 5, 6, 6)
    data = [item[1] for item in batch]
    input = default_convert(input)
    return input, data