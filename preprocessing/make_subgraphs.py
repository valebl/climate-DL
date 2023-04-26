import numpy as np
import pickle
import sys

import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_convert

from torch_geometric.data import Data

import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# paths and files
parser.add_argument('--input_path', type=str, default="/m100_work/ICT23_ESP_C/vblasone/DATA/graph/")
parser.add_argument('--output_path', type=str, default="/m100_work/ICT23_ESP_C/vblasone/DATA/graph/")
parser.add_argument('--graph_file', type=str, default="G_north_italy_train_all.pkl")
parser.add_argument('--subgraphs_file', type=str, default="subgraphs_s.pkl")
parser.add_argument('--space_idxs_file', type=str, default="valid_examples_space.pkl")
parser.add_argument('--mask_1_cell_file', type=str, default="mask_1_cell_subgraphs.pkl")
parser.add_argument('--mask_9_cells_file', type=str, default="mask_9_cells_subgraphs_all.pkl")

# other
parser.add_argument('--lat_dim', type=int, default=16)
parser.add_argument('--lon_dim', type=int, default=31)

if __name__ == "__main__":

    args = parser.parse_args()

    lat_lon_dim = args.lat_dim * args.lon_dim 

    with open(args.input_path + args.space_idxs_file, 'rb') as f:
        space_idxs = pickle.load(f)

    with open(args.input_path + args.graph_file, 'rb') as f:
        graph = pickle.load(f)
    
    with open(args.input_path + args.mask_9_cells_file, 'rb') as f:
        mask_9_cells = pickle.load(f)
    
    with open(args.input_path + args.mask_1_cell_file, 'rb') as f:
        mask_1_cell = pickle.load(f)
    
    subgraphs = [[] for i in range(max(space_idxs)+1)]

    for space_idx in space_idxs:
        lat_idx = space_idx // args.lon_dim
        lon_idx = space_idx % args.lon_dim
        mask_subgraph = mask_9_cells[space_idx] # shape = (n_nodes,)
        subgraph = graph.subgraph(subset=mask_subgraph)
        cell_idx_list = torch.tensor([ii * args.lon_dim + jj for ii in range(lat_idx-1,lat_idx+2) for jj in range(lon_idx-1,lon_idx+2)])
        idx_list_mapped = torch.sum(torch.stack([(subgraph.low_res==idx)* j for j, idx in enumerate(cell_idx_list)]), dim=0)
        subgraph["mask_1_cell"] = mask_1_cell[space_idx].cpu()  # (n_nodes)
        subgraph["mask_9_cells"] = mask_subgraph.cpu()         # (n_nodes)
        subgraph["idx_list"] = cell_idx_list
        subgraph["idx_list_mapped"] = idx_list_mapped
        subgraphs[space_idx] = subgraph
        if space_idx % 10 == 0:
            print(f"Done until {space_idx}.")
    
    with open(args.output_path + args.subgraphs_file, 'wb') as f:
        pickle.dump(subgraphs, f)
