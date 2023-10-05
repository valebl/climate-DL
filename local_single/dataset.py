import pickle
import sys
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_convert

import time

from torch_geometric.data import Data, Batch

class Dataset_pr(Dataset):

    def __init__(self, args, lat_dim, lon_dim, pad=2):
        super().__init__()
        self.pad = pad
        self.lon_low_res_dim = lon_dim
        self.lat_low_res_dim = lat_dim
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
        self.input, self.idx_to_key, self.target, self.graph, self.subgraphs, self.mask_target, self.weights, self.idx_to_key_time = self._load_data_into_memory()
        self.__set_length__()

    def __set_length__(self):
        self.length = len(self.idx_to_key)
    
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
        if self.args.weights_file is not None:
            with open(self.args.input_path + self.args.weights_file, 'rb') as f:
                weights = pickle.load(f)
        else:
            weights = None
        if self.args.idx_time_file is not None:
            with open(self.args.input_path + self.args.idx_time_file,'rb') as f:
                idx_to_key_time = pickle.load(f)
        else:
            idx_to_key_time = None
        self.low_res_abs = abs(graph.low_res)
        return input, idx_to_key, target, graph, subgraphs, mask_target, weights, idx_to_key_time

    def __getitem__(self, idx):
        k = self.idx_to_key[idx]   
        time_idx = k // self.space_low_res_dim
        space_idx = k % self.space_low_res_dim
        lat_idx = space_idx // self.lon_low_res_dim
        lon_idx = space_idx % self.lon_low_res_dim
        #-- derive input
        input = torch.zeros((25, 5, 5, 6, 6))                               # (time, var, lev, lat, lon)
        input[:, :] = self.input[time_idx - 24 : time_idx+1, :, :, lat_idx - self.pad + 2 : lat_idx + self.pad + 4, lon_idx - self.pad + 2 : lon_idx + self.pad + 4]
        #-- derive graphs and target
        subgraph = self.subgraphs[space_idx].clone()
        train_mask = self.mask_target[:,time_idx][subgraph.mask_1_cell]     # train_mask.shape = (self.graph.n_nodes, )
        subgraph["train_mask"] = train_mask
        y = self.target[subgraph.mask_1_cell, time_idx][train_mask]         # y.shape = (subgraph.n_nodes,)
        subgraph["y"] = y 
        if self.weights is not None:
            w = self.weights[subgraph.mask_1_cell, time_idx][train_mask]         # y.shape = (subgraph.n_nodes,)
            subgraph["w"] = w
        return input, subgraph

class Dataset_pr_gnn_large(Dataset_pr_gnn):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.node_dim=1
        self.encoding_dim=512

    def __set_length__(self):
        self.length = len(self.idx_to_key_time)

    def __getitem__(self, idx):
        time_idx = self.idx_to_key_time[idx]
        graph = self.graph.clone()
        graph["input"] = torch.zeros((self.graph.valid_examples_space_gnn.shape[0], 25, 5, 5, 6, 6))
        graph["train_mask"] = torch.zeros((self.graph.num_nodes), dtype=bool)
        graph["y"] = torch.zeros((graph.num_nodes),dtype=torch.float32)
        graph["w"] = torch.zeros((graph.num_nodes), dtype=torch.float32)
        graph["features"] = torch.zeros((graph.num_nodes), self.node_dim + self.encoding_dim)
        for i, space_idx in enumerate(self.graph.valid_examples_space_gnn): # loop over coloured and grey low-res cells
            lat_idx = space_idx // self.lon_low_res_dim
            lon_idx = space_idx % self.lon_low_res_dim
            mask_1_cell = graph.mask_1_cell_subgraphs[space_idx, :]
            #-- derive input
            #input = torch.zeros((25, 5, 5, 6, 6))                               # (time, var, lev, lat, lon)
            graph["input"][i, :, :, :, :, :] = self.input[time_idx - 24 : time_idx+1, :, :, lat_idx - self.pad + 2 : lat_idx + self.pad + 4, lon_idx - self.pad + 2 : lon_idx + self.pad + 4]
            #-- derive graphs and target
            #subgraph = self.subgraphs[space_idx]#.clone()
            graph["train_mask"][mask_1_cell] = self.mask_target[:,time_idx][mask_1_cell]     # train_mask.shape = (self.graph.n_nodes, )
            #subgraph["train_mask"] = train_mask
            graph["y"][mask_1_cell] = self.target[mask_1_cell, time_idx]#[train_mask]         # y.shape = (subgraph.n_nodes,)
                        #subgraph["time_idx"] = time_idx - self.time_min
            #y = self.test_graph.y[subgraph.mask_1_cell, time_idx - self.time_min]
            #subgraph["y"] = y
            if self.weights is not None:
                graph["w"][mask_1_cell] = self.weights[mask_1_cell, time_idx]#[train_mask]         # y.shape = (subgraph.n_nodes,)
            #    subgraph["w"] = w
            #batch.append([input, subgraph])
            ## Fill graph 
        return graph

class Dataset_pr_test(Dataset_pr):

    def __init__(self, idx_to_key_time, idx_to_key, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.idx_to_key_time = idx_to_key_time
        self.idx_to_key = idx_to_key
        self.time_min = min(idx_to_key_time)
        self.input, self.subgraphs, self.test_graph = self._load_data_into_memory()
        self.__set_length__()

    def __set_length__(self):
        self.length = len(self.idx_to_key)
    
    def _load_data_into_memory(self):
        with open(self.args.input_path + self.args.input_file, 'rb') as f:
            input = pickle.load(f)
        with open(self.args.input_path + self.args.subgraphs, 'rb') as f:
            subgraphs = pickle.load(f)
        with open(self.args.input_path + self.args.graph_file_test, 'rb') as f:
            test_graph = pickle.load(f)
        return input, subgraphs, test_graph

    def __getitem__(self, idx):
        k = self.idx_to_key[idx]   
        time_idx = k // self.space_low_res_dim
        space_idx = k % self.space_low_res_dim
        lat_idx = space_idx // self.lon_low_res_dim
        lon_idx = space_idx % self.lon_low_res_dim
        #-- derive input
        input = torch.zeros((25, 5, 5, 6, 6))                               # (time, var, lev, lat, lon)
        input[:, :] = self.input[time_idx - 24 : time_idx+1, :, :, lat_idx - self.pad + 2 : lat_idx + self.pad + 4, lon_idx - self.pad + 2 : lon_idx + self.pad + 4]
        #-- derive graphs and target
        subgraph = self.subgraphs[space_idx].clone()
        subgraph["time_idx"] = time_idx - self.time_min
        y = self.test_graph.y[subgraph.mask_1_cell, time_idx - self.time_min]
        subgraph["y"] = y
        return input, subgraph

class Dataset_pr_test_large(Dataset_pr_test):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __set_length__(self):
        self.length = len(self.idx_to_key_time)

    def __getitem__(self, idx):
        time_idx = self.idx_to_key_time[idx]
        batch = []
        for i, space_idx in enumerate(self.test_graph.valid_examples_space):
            lat_idx = space_idx // self.lon_low_res_dim
            lon_idx = space_idx % self.lon_low_res_dim
            #-- derive input
            input = torch.zeros((25, 5, 5, 6, 6))                               # (time, var, lev, lat, lon)
            input[:, :] = self.input[time_idx - 24 : time_idx+1, :, :, lat_idx - self.pad + 2 : lat_idx + self.pad + 4, lon_idx - self.pad + 2 : lon_idx + self.pad + 4]
            #-- derive graphs and target
            subgraph = self.subgraphs[space_idx].clone()
            subgraph["time_idx"] = time_idx - self.time_min
            y = self.test_graph.y[subgraph.mask_1_cell, time_idx - self.time_min]
            subgraph["y"] = y
            batch.append([input, subgraph])
        return batch

def custom_collate_fn_ae(batch):
    input = torch.stack(batch)
    input = default_convert(input)
    return input

def custom_collate_fn_gnn(batch):
    input = torch.stack([item[0] for item in batch])                        # shape = (batch_size, 25, 5, 5, 6, 6)
    data = [item[1] for item in batch]
    input = default_convert(input)
    return input, data

def custom_collate_fn_gnn_large(batch):
    #batch = Batch.from_data_list(graphs_list)
    return batch

def custom_collate_fn_gnn_test_large(batch):
    input = torch.stack([item[0] for item in batch[0]])
    data = [item[1] for item in batch[0]]
    #batch = batch[0]
    #input = torch.stack([item for item in batch[0]])                        # shape = (batch_size, 25, 5, 5, 6, 6)
    #data = batch[1]
    #input = default_convert(input)
    return input, data

