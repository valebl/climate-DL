import numpy as np
import pickle
import sys

import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_convert

from torch_geometric.data import Data

if __name__ == "__main__":

    #input_path = "/home/vblasone/DATA/graph/"
    input_path = "/m100_work/ICT23_ESP_C/vblasone/DATA/graph/" 
    space_idxs_file = "valid_examples_space.pkl"
    graph_file = "G_north_italy_train.pkl"
    mask_1_cell_file = "mask_1_cell_subgraphs.pkl"
    mask_9_cells_file = "mask_9_cells_subgraphs.pkl"
    #output_path = "/home/vblasone/DATA/graph/"
    output_path = "/m100_work/ICT23_ESP_C/vblasone/DATA/graph/"
    subgraphs_file = "subgraphs_s.pkl"

    lat_dim=16
    lon_dim=31
    lat_lon_dim = lat_dim * lon_dim 

    with open(input_path + space_idxs_file, 'rb') as f:
        space_idxs = pickle.load(f)

    with open(input_path + graph_file, 'rb') as f:
        graph = pickle.load(f)
    
    with open(input_path + mask_1_cell_file, 'rb') as f:
        mask_1_cell = pickle.load(f)

    with open(input_path + mask_9_cells_file, 'rb') as f:
        mask_9_cells = pickle.load(f)
    
    subgraphs = [[] for i in range(max(space_idxs)+1)]

    for space_idx in space_idxs:
        lat_idx = space_idx // lon_dim
        lon_idx = space_idx % lon_dim
        mask_subgraph = mask_9_cells[space_idx] # shape = (n_nodes,)
        #print(mask_subgraph)
        subgraph = graph.subgraph(subset=mask_subgraph)
        #mask_y_nodes = mask_1_cell[space_idx] * mask_target[:,time_idx] # shape = (n_nodes,)
        #subgraph["train_mask"] = mask_y_nodes[mask_subgraph]
        subgraph["mask_1_cell"] = mask_1_cell[space_idx].cpu()
        subgraph["mask_subgraph"] = mask_subgraph.cpu()
        #y = self.target[mask_subgraph, time_idx] # shape = (n_nodes_subgraph,)
        #subgraph["y"] = y
        cell_idx_list = torch.tensor([ii * lon_dim + jj for ii in range(lat_idx-1,lat_idx+2) for jj in range(lon_idx-1,lon_idx+2)])
        subgraph["idx_list"] = cell_idx_list
        subgraphs[space_idx] = subgraph
        if space_idx % 10 == 0:
            print(f"Done until {space_idx}.")
    
    with open(output_path + subgraphs_file, 'wb') as f:
        pickle.dump(subgraphs, f)
