import numpy as np
import pickle
import sys

import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_convert

from torch_geometric.data import Data

class Clima_dataset(Dataset):

    def _load_data_into_memory(self, args):
        path = args.input_path
        input_file, target_file, data_file, idx_file = args.input_file, args.target_file, args.data_file, args.idx_file
        mask_file, weights_file = args.mask_file, args.weights_file
        with open(path + input_file, 'rb') as f:
            self.input = pickle.load(f)  
        with open(path + idx_file,'rb') as f:
            self.idx_to_key = pickle.load(f)
        if args.net_type != "ae":
            with open(path + target_file, 'rb') as f:
                self.target = pickle.load(f)
        if args.net_type == "gnn":
            with open(path + data_file, 'rb') as f:
                self.data = pickle.load(f)
            if mask_file is not None:
                with open(path + mask_file, 'rb') as f:
                    self.mask = pickle.load(f)
            else:
                self.mask = None
            if weights_file is not None:
                with open(path + weights_file, 'rb') as f:
                    self.weights = pickle.load(f)
            else:
                self.weights = None
        else:
            self.data, self.mask, self.weights = None, None, None
        return

    def _get_lat_lon_lists(self, lon_min=6.50, lon_max=14.00, lat_min=43.75, lat_max=47.25):
        LON_MIN = 6.5
        LON_MAX = 18.75
        LAT_MIN = 36.5
        LAT_MAX =  47.25
        INTERVAL = 0.25
        idx_lat_start = int((lat_min - LAT_MIN) / INTERVAL) #int(np.where(lat_era5_list == lat_min_sel)[0])
        idx_lat_end = int((lat_max - LAT_MIN) / INTERVAL) #int(np.where(lat_era5_list == lat_max_sel)[0])
        idx_lat_list = np.arange(idx_lat_start, idx_lat_end, 1)
        idx_lon_start = int((lon_min - LON_MIN) / INTERVAL) # int(np.where(lon_era5_list == lon_min_sel)[0])
        idx_lon_end = int((lon_max - LON_MIN) / INTERVAL) # int(np.where(lon_era5_list == lon_max_sel)[0])
        idx_lon_list = np.arange(idx_lon_start, idx_lon_end, 1)
        return idx_lat_list, idx_lon_list

    def __init__(self, args, **kwargs):
        super().__init__()
        self.PAD = 2
        self.LAT_DIM = 43 # number of points in the GRIPHO rectangle (0.25 grid)
        self.LON_DIM = 49
        self.SPACE_IDXS_DIM = self.LAT_DIM * self.LON_DIM
        self.SHIFT = 2 # relative shift between GRIPHO and ERA5 (idx=0 in ERA5 corresponds to 2 in GRIPHO)
        self.net_type = args.net_type
        self.idx_lat_list, self.idx_lon_list = self._get_lat_lon_lists()
        _ = self._load_data_into_memory(args)
        
        if self.net_type != 'ae':
            self.length = len(self.target)
        else:
            self.length = len(self.input)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        if self.net_type == "ae":
            k = self.idx_to_key[idx]
            idx_time = k // self.SPACE_IDXS_DIM
            idx_space = k % self.SPACE_IDXS_DIM
            idx_lat = idx_space // self.LON_DIM
            idx_lon = idx_space % self.LON_DIM
            return self.input[idx_time - 24 : idx_time+1, :, :, idx_lat - self.PAD + 2 : idx_lat + self.PAD + 4, idx_lon - self.PAD + 2 : idx_lon + self.PAD + 4]
        else:
            idx_time = self.idx_to_key[idx]
            input_ae = []
            for idx_lat in self.idx_lat_list:
                for idx_lon in self.idx_lon_list:
                    input_ae.append(np.array(self.input[idx_time - 24 : idx_time+1, :, :, idx_lat - self.PAD + 2 : idx_lat + self.PAD + 4, idx_lon - self.PAD + 2 : idx_lon + self.PAD + 4]))
            #-- derive gnn data
            y = torch.tensor(self.target[idx_time])
            edge_index = torch.tensor(self.data['edge_index'])
            x = torch.tensor(self.data['x'])
            if self.mask is not None:
                mask = torch.tensor(self.mask[idx_time].astype(bool)) 
            else:
                mask = None
            if self.weights is not None:
                weights = torch.tensor(self.weights[idx_time])
            else:
                weights = None
            data = Data(x=x, edge_index=edge_index, y=y, mask=mask, weights=weights)
            input_ae = np.array(input_ae)
            return input_ae, data
        
def custom_collate_fn_ae(batch):
    input = np.array(batch)
    input = default_convert(input)
    return input

def custom_collate_fn_gnn(batch):    
    input = np.array([item[0] for item in batch])
    data = [item[1] for item in batch]
    input = default_convert(input)
    return input, data
    

