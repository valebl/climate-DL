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

class Dataset_e(Dataset_pr_ae):
    
    def __getitem__(self, idx):
        k = self.idx_to_key[idx]
        time_idx = k[1]
        space_idx = k[0]
        lat_idx = space_idx // self.lon_low_res_dim
        lon_idx = space_idx % self.lon_low_res_dim
        input = torch.zeros((25, 5, 5, 6, 6))
        input[:] = self.input[time_idx - 24 : time_idx+1, :, :, lat_idx - self.pad + 2 : lat_idx + self.pad + 4, lon_idx - self.pad + 2 : lon_idx + self.pad + 4]
        return input, k 

class Dataset_pr_gnn(Dataset_pr):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input, self.idx_to_key, self.target, self.graph, self.mask_target, self.subgraphs = self._load_data_into_memory()
        #self.t_input=0
        #self.t_gnn=0

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
        #t0 = time.time()
        k = self.idx_to_key[idx]   
        time_idx = k // self.space_low_res_dim
        space_idx = k % self.space_low_res_dim
        lat_idx = space_idx // self.lon_low_res_dim
        lon_idx = space_idx % self.lon_low_res_dim
        #-- derive input
        input = torch.zeros((9, 25, 5, 5, 6, 6))
        lat_lon_idx_list = torch.tensor([[ii, jj] for ii in range(lat_idx-1,lat_idx+2) for jj in range(lon_idx-1,lon_idx+2)])
        for i, idx in enumerate(lat_lon_idx_list):
            input[i, :] = self.input[time_idx - 24 : time_idx+1, :, :, idx[0] - self.pad + 2 : idx[0] + self.pad + 4, idx[1] - self.pad + 2 : idx[1] + self.pad + 4]
        #t1 = time.time()
        #self.t_input += (t1 - t0)
        #-- derive gnn data
        subgraph = self.subgraphs[space_idx].clone()
        mask_y_nodes = subgraph.mask_1_cell * self.mask_target[:,time_idx] # shape = (n_nodes,)
        subgraph["train_mask"] = mask_y_nodes[subgraph.mask_9_cells]
        y = self.target[subgraph.mask_9_cells, time_idx] # shape = (n_nodes_subgraph,)
        subgraph["y"] = y
        #self.t_gnn += (time.time() - t1)
        return input, subgraph
    
class Dataset_pr_test(Dataset_pr):

    def __init__(self, time_min, time_max, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_min = time_min
        self.time_max = time_max
        self.input, self.idx_to_key, self.graph, self.subgraphs = self._load_data_into_memory()

    def _load_data_into_memory(self):
        with open(self.args.input_path + self.args.input_file, 'rb') as f:
            input = pickle.load(f)
        with open(self.args.input_path + self.args.idx_file,'rb') as f:
            idx_to_key = pickle.load(f)   
        with open(self.args.input_path + self.args.graph_file, 'rb') as f:
            graph = pickle.load(f)
        with open(self.args.input_path + self.args.subgraphs, 'rb') as f:
            subgraphs = pickle.load(f)
        self.length = len(idx_to_key)
        self.low_res_abs = abs(graph.low_res)
        return input, idx_to_key, graph, subgraphs

    def __getitem__(self, idx):
        k = self.idx_to_key[idx]   
        time_idx = k // self.space_low_res_dim
        space_idx = k % self.space_low_res_dim
        lat_idx = space_idx // self.lon_low_res_dim
        lon_idx = space_idx % self.lon_low_res_dim
        #-- derive input
        lon_lat_idx_list = torch.tensor([[ii, jj] for ii in range(lat_idx-1,lat_idx+2) for jj in range(lon_idx-1,lon_idx+2)])
        input = torch.zeros((9, 25, 5, 5, 6, 6))
        for i, idx in enumerate(lon_lat_idx_list):
            input[i, :] = self.input[time_idx - 24 : time_idx+1, :, :, idx[0] - self.pad + 2 : idx[0] + self.pad + 4, idx[1] - self.pad + 2 : idx[1] + self.pad + 4]
        ##-- derive gnn data
        subgraph = self.subgraphs[space_idx].clone()
        cell_idx_list = torch.tensor([ii * self.lon_low_res_dim + jj for ii in range(lat_idx-1,lat_idx+2) for jj in range(lon_idx-1,lon_idx+2)])
        subgraph["idx_list"] = cell_idx_list
        subgraph["time_idx"] = time_idx - self.time_min
        subgraph["test_mask"] = subgraph.mask_1_cell[subgraph.mask_9_cells]
        return input, subgraph

class Dataset_pr_ft_gnn(Dataset_pr_gnn):

    def __getitem__(self, idx):
        k = self.idx_to_key[idx]   
        time_idx = k // self.space_low_res_dim
        space_idx = k % self.space_low_res_dim
        lat_idx = space_idx // self.lon_low_res_dim
        lon_idx = space_idx % self.lon_low_res_dim
        #-- derive input
        encoding = torch.zeros((9, 128))
        cell_idx_list = torch.tensor([ii * self.lon_low_res_dim + jj for ii in range(lat_idx-1,lat_idx+2) for jj in range(lon_idx-1,lon_idx+2)])
        for i, s in enumerate(cell_idx_list):
            encoding[i, :] = self.input[s, time_idx, :]
        
        #-- derive gnn data
        mask_subgraph = self.mask_9_cells[space_idx] # shape = (n_nodes,)
        #print(mask_subgraph)
        subgraph = self.graph.subgraph(subset=mask_subgraph)
        mask_y_nodes = self.mask_1_cell[space_idx] * self.mask_target[:,time_idx] # shape = (n_nodes,)
        subgraph["train_mask"] = mask_y_nodes[mask_subgraph]
        y = self.target[mask_subgraph, time_idx] # shape = (n_nodes_subgraph,)
        subgraph["y"] = y
        subgraph["idx_list"] = cell_idx_list
        return encoding, subgraph


def custom_collate_fn_ae(batch):
    input = torch.stack(batch)
    input = default_convert(input)
    return input

def custom_collate_fn_e(batch):
    input = torch.stack([item[0] for item in batch])
    idxs = [item[1] for item in batch] 
    input = default_convert(input)
    idxs = default_convert(idxs)
    idxs = torch.stack(idxs)
    return input, idxs

def custom_collate_fn_gnn(batch):
    input = torch.stack([item[0] for item in batch]) # shape = (batch_size, 9, 25, 5, 5, 6, 6)
    data = [item[1] for item in batch]
    input = default_convert(input)
    return input, data
    
