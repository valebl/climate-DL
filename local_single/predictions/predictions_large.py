import numpy as np
import pickle
import pandas as pd
import torch

from dataset import Clima_dataset as Dataset
from dataset import custom_collate_fn_ae, custom_collate_fn_gnn
from models_large import Encoder as Encoder
#from models import Regressor_test as Regressor_GNN
#from models import Classifier_test as Classifier_GNN
from models import Regressor_test_large as Model_reg
from models import Classifier_test_large as Model_cl

from torch_geometric.data import Data, Batch

import sys
import copy

from itertools import chain

def getitem_encoding(dataset, idx_time, idx_lat, idx_lon):
    input = dataset.input[idx_time - 24 : idx_time+1, :, :, idx_lat - dataset.PAD + 2 : idx_lat + dataset.PAD + 4, idx_lon - dataset.PAD + 2 : idx_lon + dataset.PAD + 4]
    return input
    
if __name__ == '__main__':

    TIME_DIM = 140256
    SPATIAL_POINTS_DIM = 2107
    LON_MIN = 6.5
    LON_MAX = 18.75
    LAT_MIN = 36.5
    LAT_MAX =  47.25
    INTERVAL = 0.25
    LON_DIM = (LON_MAX - LON_MIN) // INTERVAL # 49
    LAT_DIM = (LAT_MAX - LAT_MIN) // INTERVAL # 43
    SPATIAL_POINTS_DIM = LAT_DIM * LON_DIM

    lon_era5_list = np.arange(LON_MIN, LON_MAX, INTERVAL)
    lat_era5_list = np.arange(LAT_MIN, LAT_MAX, INTERVAL)

    lon_min = 6.50
    lon_max = 13.75
    lat_min = 43.75
    lat_max = 47.00
    idx_lat_start = int((lat_min - LAT_MIN) / INTERVAL) #int(np.where(lat_era5_list == lat_min_sel)[0])
    idx_lat_end = int((lat_max - LAT_MIN) / INTERVAL) #int(np.where(lat_era5_list == lat_max_sel)[0])
    idx_lat_list = np.arange(idx_lat_start, idx_lat_end, 1)

    idx_lon_start = int((lon_min - LON_MIN) / INTERVAL) # int(np.where(lon_era5_list == lon_min_sel)[0])
    idx_lon_end = int((lon_max - LON_MIN) / INTERVAL) # int(np.where(lon_era5_list == lon_max_sel)[0])
    idx_lon_list = np.arange(idx_lon_start, idx_lon_end, 1)

    idx_space_sel = np.array([[i * LON_DIM + j for j in idx_lon_list] for i in idx_lat_list])
    idx_space_sel = idx_space_sel.flatten()

    #print(idx_space_sel.shape, idx_lat_list.shape, idx_lon_list.shape)

    #sys.exit()

    year = 2016
    device = 'cuda:0'

    #work_dir = "/work_dir/220927/regression-gnn-north/ctd-20-30-0001/"
    work_dir = "/work_dir/"
    data_dir = "/data/north_01-15/"
    #work_dir = "/home/vblasone/precipitation-maps/"
    #data_dir = "/home/vblasone/DATA/"

    log_file = f"get_results_large/prova_{year}.txt"

    checkpoint_cl = "/work_dir/checkpoints/cl_small/checkpoint_3.pth"
    checkpoint_reg = "/work_dir/checkpoints/reg_small/checkpoint_39.pth"

    input_path = "/data/"
    input_file = "input_standard.pkl"

    data_file = "gnn_data_standard.pkl"
    target_file = "gnn_target_test.pkl"
    idx_to_key_file = "idx_to_key_test.pkl"

    ## CREATE MODELS AND LOAD PARAMETERS

    model_e_cl = Encoder()
    model_e_reg = Encoder()
    model_gnn_cl = Model_cl()
    model_gnn_reg = Model_reg()

    model_e_cl.eval()
    model_e_reg.eval()
    model_gnn_cl.eval()
    model_gnn_reg.eval()
    
    checkpoint_cl = torch.load(checkpoint_cl)
    checkpoint_reg = torch.load(checkpoint_reg)
    
    for name, param in checkpoint_cl["parameters"].items():
        param = param.data
        if name.startswith("module."):                
            name = name.partition("module.")[2]
        if "encoder" in name or "gru" in name or "dense" in name:
            model_e_cl.state_dict()[name].copy_(param)
        elif "gnn" in name:
            model_gnn_cl.state_dict()[name].copy_(param)

    for name, param in checkpoint_reg["parameters"].items():
        param = param.data
        if name.startswith("module."):                
            name = name.partition("module.")[2]
        if "encoder" in name or "gru" in name or "dense" in name:
            model_e_reg.state_dict()[name].copy_(param)
        elif "gnn" in name:
            model_gnn_reg.state_dict()[name].copy_(param)

    # move models to device

    model_e_cl = model_e_cl.to(device)
    model_e_reg = model_e_reg.to(device)
    model_gnn_cl = model_gnn_cl.to(device)
    model_gnn_reg = model_gnn_reg.to(device)

    ## START

    with open(work_dir+log_file, 'w') as f:
        f.write("Starting...")

    with open(work_dir+log_file, 'a') as f:
        f.write("\nbuilding the dataset...")

    net_type = 'ae'
    dataset_encoding = Dataset(path="", input_file=input_path+input_file, data_file=data_dir+data_file, target_file=data_dir+target_file, idx_file=data_dir+idx_to_key_file, net_type=net_type, get_key=False)

    with open(work_dir+log_file, 'a') as f:
        f.write("\ndone!")

    with open('/data/north/north_italy_graph.pkl', 'rb') as f:
        spatial_graph = pickle.load(f)

    era5_to_gripho_idxs = dict()
    space_idxs_list = []
    era5_to_gripho_list = []

    for idx_lat in idx_lat_list:
        lat_north = lat_era5_list[idx_lat]
        for idx_lon in idx_lon_list:
            lon_north = lon_era5_list[idx_lon]
            space_idx = int(idx_lat * SPATIAL_POINTS_DIM + idx_lon)
            bool_lat = np.logical_and(spatial_graph['x'][:,1] >= lat_north, spatial_graph['x'][:,1] <= lat_north + INTERVAL)
            bool_lon = np.logical_and(spatial_graph['x'][:,0] >= lon_north, spatial_graph['x'][:,0] <= lon_north + INTERVAL)
            bool_both = np.logical_and(bool_lon, bool_lat)
            era5_to_gripho_idxs[space_idx] = bool_both
            space_idxs_list.append(space_idx)
            era5_to_gripho_list.append(bool_both)
            #print(lat_north, lon_north, space_idx, sum(bool_both))

    with open('/data/north/north_italy_graph_standard.pkl', 'rb') as f:
        spatial_graph = pickle.load(f)

    #with open(work_dir+f"era5_to_gripho_list.pkl", 'wb') as f:
    #    pickle.dump(era5_to_gripho_list, f)
    
    #with open(work_dir+f"space_idx_prova.pkl", 'wb') as f:
    #    pickle.dump(space_idxs_list, f)

    #with open(work_dir+f"era5_to_gripho_idxs.pkl", 'wb') as f:
    #    pickle.dump(era5_to_gripho_idxs, f)

    with open(work_dir+"get_results_large/idx_years_list_complete.pkl", 'rb') as f:
        idx_list_time = pickle.load(f)

    with open(work_dir+log_file, 'a') as f:
        f.write("\nstarting the loop...")

    len_idx_list_time = len(idx_list_time[year-2000-1])
    y_pred_dict = dict()

    idx_list = chain(idx_list_time[year-2000-1][-31*24:],idx_list_time[year-2000-1])

    # cicle over time indexes
    for j, idx_time in enumerate(idx_list):
        # cicle over space indexes
        batch = []
        for idx_lat in idx_lat_list:
            for idx_lon in idx_lon_list:
                batch.append(getitem_encoding(dataset_encoding, idx_time, idx_lat, idx_lon))
                                    
        X = custom_collate_fn_ae(batch)
        X = X.to(device)

        # classifier
        e_cl = model_e_cl(X)
        e_cl = e_cl.detach()

        # regresssor
        e_reg = model_e_reg(X)
        e_reg = e_reg.detach()

        features_cl = torch.zeros((spatial_graph['x'].shape[0], 3 + e_cl.shape[1])).to(device)
        features_cl[:,:3] = torch.tensor(spatial_graph['x'][:,:])

        features_reg = torch.zeros((spatial_graph['x'].shape[0], 3 + e_reg.shape[1])).to(device)
        features_reg[:,:3] = torch.tensor(spatial_graph['x'][:,:])

        for s, space in enumerate(space_idxs_list):
            features_cl[era5_to_gripho_idxs[space], 3:] = e_cl[s] 
            features_reg[era5_to_gripho_idxs[space], 3:] = e_reg[s]

        data_cl = Data(x=features_cl, edge_index=torch.tensor(spatial_graph['edge_index']).to(device))
        y_pred_cl = model_gnn_cl(data_cl).detach().cpu().to(torch.float32)

        data_reg = Data(x=features_reg, edge_index=torch.tensor(spatial_graph['edge_index']).to(device))
        y_pred_reg = model_gnn_reg(data_reg).detach().cpu().to(torch.float32)

        #y_pred_dict[idx_time] = torch.argmax(y_pred_cl, dim=-1)
        #y_pred_dict[idx_time] = y_pred_reg
        y_pred_dict[idx_time] = y_pred_reg * y_pred_cl

        #print(y_pred_dict[idx_time].shape)
        #sys.exit()

        if j % 1000 == 0: 
            with open(work_dir+log_file, 'a') as f:
                f.write(f"\n{j / len_idx_list_time * 100} % done")
    
    with open(work_dir+log_file, 'a') as f:
        f.write("\nwriting the files...")

    with open(work_dir+f"get_results_large/y_pred_{year}_small.pkl", 'wb') as f:
        pickle.dump(y_pred_dict, f)

    with open(work_dir+log_file, 'a') as f:
        f.write("\ndone! :)")

    sys.exit()

    




















    for i, data in enumerate(data_batch):
        data = data.to(device)
        features = torch.zeros((data.num_nodes, 3 + encoding.shape[1])).to(device)
        features[:,:3] = data.x[:,:3]
        features[:,3:] = encoding[i,:]
        data.__setitem__('x', features)

    #X, data = custom_collate_fn_gnn(batch)
    #y_pred, y, batch_idxs = model(X, data, device)
    #for ii, ki in enumerate(keys):
    #    idxi = torch.where(batch_idxs == ii)
    #    y_pred_dict[ki] = y_pred[idxi].detach().cpu().numpy()
    #    y_dict[ki] = y[idxi].detach().cpu().numpy()
    #with open(work_dir+log_file, 'a') as f:
    #    f.write("\nbatch done")

    with open(work_dir+log_file, 'a') as f:
        f.write("\nwriting the files...")

    with open(work_dir+f"y_pred_regression_{year}_total.pkl", 'wb') as f:
        pickle.dump(y_pred_dict, f)

    #with open(work_dir+f"y_regression_{year}.pkl", 'wb') as f:
    #    pickle.dump(y_dict, f)

    #with open(work_dir+f"batch_idxs_{year}.pkl", 'wb') as f:
    #    pickle.dump(batch_idxs, f)

    #with open(work_dir+f"mask_{year}.pkl", 'wb') as f:
    #    pickle.dump(mask, f)

    with open(work_dir+log_file, 'a') as f:
        f.write("\ndone! :)")


