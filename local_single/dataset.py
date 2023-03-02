import numpy as np
import pickle
import sys

import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_convert

from torch_geometric.data import Data

class Dataset_pr(Dataset):

    def __init__(self, args, pad=2, lat_dim=15, lon_dim=31, shift=2): #path, input_file, target_file, data_file, idx_file, net_type, get_key=False, mask_file=None, weights_file=None, **kwargs):
        super().__init__()
        self.pad = pad
        self.lat_low_res_dim = lat_dim # number of points in the GRIPHO rectangle (0.25 grid)
        self.lon_low_res_dim = lon_dim
        self.space_low_res_dim = self.lat_low_res_dim * self.lon_low_res_dim
        self.shift = shift # relative shift between GRIPHO and ERA5 (idx=0 in ERA5 corresponds to 2 in GRIPHO)
        self.args = args
        self.length = None
        self.low_res_abs = None

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

    def __getitem__(self, idx):
        k = self.idx_to_key[idx]   
        time_idx = k // self.space_low_res_dim
        space_idx = k % self.space_low_res_dim
        lat_idx = space_idx // self.lon_low_res_dim
        lon_idx = space_idx % self.lon_low_res_dim
        #-- derive input
        input = self.input[time_idx - 24 : time_idx+1, :, :, lat_idx - self.pad + 2 : lat_idx + self.pad + 4, lon_idx - self.pad + 2 : lon_idx + self.pad + 4]
        return input


class Dataset_pr_gnn(Dataset_pr):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input, self.idx_to_key, self.target, self.graph, self.mask_target,
            self.mask_1_cell, self.mask_9_cells = self._load_data_into_memory()
    
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
        with open(self.args.input_path + self.args.mask_1_cell_file, 'rb') as f:
            mask_1_cell = pickle.load(f)
        with open(self.args.input_path + self.args.mask_9_cells_file, 'rb') as f:
            mask_9_cells = pickle.load(f)
        self.length = len(idx_to_key)
        self.low_res_abs = abs(graph.low_res)
        return input, idx_to_key, target, graph, mask_target, mask_1_cell, mask_9_cells

    def __getitem__(self, idx):
        k = self.idx_to_key[idx]   
        time_idx = k // self.space_low_res_dim
        space_idx = k % self.space_low_res_dim
        lat_idx = space_idx // self.lon_low_res_dim
        lon_idx = space_idx % self.lon_low_res_dim
        #-- derive input
        input = self.input[time_idx - 24 : time_idx+1, :, :, lat_idx - self.pad + 2 : lat_idx + self.pad + 4, lon_idx - self.pad + 2 : lon_idx + self.pad + 4]
        #-- derive gnn data
        mask_subgraph = self.mask_9_cells[space_idx] # shape = (n_nodes,)
        subgraph = self.graph.subgraph(subset=mask_subgraph)
        mask_y_nodes = self.mask_1_cell[space_idx] * self.mask_target[:,time_idx] # shape = (n_nodes,)
        subgraph["train_mask"] = mask_y_nodes[mask_subgraph]
        y = self.target[mask_y_nodes, time_idx] # shape = (n_nodes_train,)
        subgraph["y"] = y
        return input, subgraph
        #selected_cells = torch.tensor([ii * self.lon_low_res_dim + jj for ii in range(lat_idx-1,lat_idx+2) for jj in range(lon_idx-1,lon_idx+2)])
        #mask_subgraph = torch.isin(self.low_res_abs, selected_cells)        # abs to consider all nodes (regarldess nan)
        #subgraph = self.graph.subgraph(subset=mask_subgraph)
        #subgraph["y"] = self.target[mask_subgraph][:,time_idx]                           # target is for a single time step
        #not_nan = ~torch.isnan(subgraph.y])                                 # retrieve mask of not nan target
        #mask_cell = torch.isin(subgraph.low_res, space_idx)                 # retrieve mask of central cell
        #subgraph["train_mask"] = torch.logical_and(not_nan, mask_cell)      # loss computed on nodes in central cell with not nan target


'''
class Dataset_pr_cl(Dataset_pr):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input, self.idx_to_key, self.target, self.graph = self._load_data_into_memory()
    
    def _load_data_into_memory(self):
        with open(self.args.input_path + self.args.input_file, 'rb') as f:
            input = pickle.load(f)
        with open(self.args.input_path + self.args.idx_file,'rb') as f:
            idx_to_key = pickle.load(f)
        with open(self.args.input_path + self.args.target_file, 'rb') as f:
            target = pickle.load(f)        
        with open(self.args.input_path + self.args.graph_file, 'rb') as f:
            graph = pickle.load(f)
        #graph["y"] = target
        self.length = len(idx_to_key)
        return input, idx_to_key, target, graph
 
    def __getitem__(self, idx):
        k = self.idx_to_key[idx]   
        time_idx = k // self.space_low_res_dim
        space_idx = k % self.space_low_res_dim
        lat_idx = space_idx // self.lon_low_res_dim
        lon_idx = space_idx % self.lon_low_res_dim
        #-- derive input
        input = self.input[time_idx - 24 : time_idx+1, :, :, lat_idx - self.pad + 2 : lat_idx + self.pad + 4, lon_idx - self.pad + 2 : lon_idx + self.pad + 4]
        #-- derive gnn data
        selected_cells = torch.tensor([ii * self.lon_low_res_dim + jj for ii in range(lat_idx-1,lat_idx+2) for jj in range(lon_idx-1,lon_idx+2)])
        mask_subgraph = torch.isin(self.low_res_abs, selected_cells)        # abs to consider all nodes (regarldess nan)
        subgraph = self.graph.subgraph(subset=mask_subgraph)
        subgraph.y = subgraph.y[:,time_idx]                           # target is for a single time step
        not_nan = ~torch.isnan(subgraph.y)                      # retrieve mask of not nan target
        mask_cell = torch.isin(subgraph.low_res, space_idx)                 # retrieve mask of central cell
        subgraph["train_mask"] = torch.logical_and(not_nan, mask_cell)      # loss computed on nodes in central cell with not nan target
        subgraph.y[train_mask] = torch.where(subgraph.y[train_mask])
        return input, subgraph
'''

def custom_collate_fn_ae(batch):
    input = np.array(batch)
    input = default_convert(input)
    return input

def custom_collate_fn_gnn(batch):
    input = np.array([item[0] for item in batch])
    data = [item[1] for item in batch]
    input = default_convert(input)
    return input, data
    
