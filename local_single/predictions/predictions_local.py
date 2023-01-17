import numpy as np
import pickle
import pandas as pd
import torch

from dataset import Clima_dataset as Dataset
from dataset import custom_collate_fn_ae, custom_collate_fn_get_key, custom_collate_fn_gnn
from models import Regressor_test as Model
from torch_geometric.data import Data, Batch
from models import Classifier_test as Model_classifier

import sys

#from torchmetrics.classification import BinaryConfusionMatrix

def getitem(dataset, k):
    time_idx = k // dataset.SPACE_IDXS_DIM
    space_idx = k % dataset.SPACE_IDXS_DIM
    lat_idx = space_idx // dataset.LON_DIM
    lon_idx = space_idx % dataset.LON_DIM
    #-- derive input
    #if dataset.get_key:
    #    return k
    #if dataset.net_type == "gru" or "gnn":
    #    input = dataset.input[time_idx - 24 : time_idx+1, :, :, lat_idx - dataset.PAD + 2 : lat_idx + dataset.PAD + 4, lon_idx - dataset.PAD + 2 : lon_idx + dataset.PAD + 4]
    #else:
    input = dataset.input[time_idx - 24 : time_idx+1, :, :, lat_idx - dataset.PAD + 2 : lat_idx + dataset.PAD + 4, lon_idx - dataset.PAD + 2 : lon_idx + dataset.PAD + 4]
    #-- derive gnn data
    #if dataset.net_type == "cnn" or dataset.net_type == "gnn" or dataset.net_type == "gru":
        #y = torch.tensor(dataset.target[k])
        #if dataset.net_type == "cnn" or dataset.net_type == "gru":
        #    return input
        #else:
    edge_index = torch.tensor(dataset.data[space_idx]['edge_index'])
    x = torch.tensor(dataset.data[space_idx]['x'])
            #if dataset.mask is not None:
            #    mask = torch.tensor(dataset.mask[k].astype(bool))  #torch.where(y==0, False, True)
            #    data = Data(x=x, edge_index=edge_index, mask=mask)
            #else:
    #try:
    y = torch.tensor(dataset.target[k])
    #    #print(y.shape)
    #    #sys.exit()
    #except:
    #    y = torch.full((x.shape[0],1), torch.nan)
    data = Data(x=x, edge_index=edge_index, y=y)
    return input, data
    #else:
    #    return input


if __name__ == '__main__':

    year = 2016
    device = 'cuda:0'

    #work_dir = "/m100_work/ICT22_ESP_0/vblasone/precipitation-maps/results_2016/"
    #data_dir = "/m100_work/ICT22_ESP_0/vblasone/DATA/north_01-15/"
    work_dir = "/work_dir/get_results/"
    data_dir = "/data/north_01-15/"

    log_file = f"log_{year}.txt"

    with open(work_dir+log_file, 'w') as f:
        f.write("Starting...")

    LAT_DIM = 43 # number of points in the GRIPHO rectangle (0.25 grid)
    LON_DIM = 49
    SPACE_IDXS_DIM = LAT_DIM * LON_DIM
    TIME_DIM = 140256
    SPATIAL_POINTS_DIM = LAT_DIM * LON_DIM

    #checkpoint_cl = "/m100_work/ICT22_ESP_0/vblasone/precipitation-maps/221118/cl/ctd-4/checkpoint.pth"
    #checkpoint_reg = "/m100_work/ICT22_ESP_0/vblasone/precipitation-maps/221118/reg/ctd-3/checkpoint.pth"
    checkpoint_cl = "/work_dir/checkpoints/cl_small/checkpoint_3.pth"
    checkpoint_reg = "/work_dir/checkpoints/reg_small/checkpoint_39.pth"

    #input_path = "/m100_work/ICT22_ESP_0/vblasone/DATA/"
    input_path = "/data/"
    input_file = "input_standard.pkl"
    idx_to_key_file = "idx_to_key_test.pkl"
    data_file = "gnn_data_standard.pkl"
    target_file = "gnn_target_test.pkl"
    #mask_file = "north/gnn_target_classes_north.pkl"
    # checkpoint_file = "/home/vblasone/precipitation-maps/220920/ae-north/checkpoint_ae_north_e3.pth"
    
    with open(work_dir+log_file, 'a') as f:
        f.write("\nbuilding the dataset...")

    net_type = 'gnn'
    dataset = Dataset(path="", input_file=input_path+input_file, data_file=input_path+data_file, target_file=data_dir+target_file,
        idx_file=data_dir+idx_to_key_file, net_type=net_type, get_key=False)

    with open(work_dir+log_file, 'a') as f:
        f.write("\ndone!")

    with open(data_dir+idx_to_key_file, 'rb') as f:
        test_keys = pickle.load(f)

    model = Model()
    model.eval()

    checkpoint = torch.load(checkpoint_reg)

    try:
        model.load_state_dict(checkpoint["parameters"])
    except:
        for name, param in checkpoint["parameters"].items():
            param = param.data
            if name.startswith("module."):
                name = name.partition("module.")[2]
            model.state_dict()[name].copy_(param)
    
    model = model.to(device)

    model_cl = Model_classifier()
    model_cl.eval()

    checkpoint_cl = torch.load(checkpoint_cl)

    try:
        model_cl.load_state_dict(checkpoint_cl["parameters"])
    except:
        for name, param in checkpoint_cl["parameters"].items():
            param = param.data
            if name.startswith("module."):
                name = name.partition("module.")[2]
            model_cl.state_dict()[name].copy_(param)

    model_cl = model_cl.to(device)

#    with open("/work_dir/get_results/idx_list_all.pkl", 'rb') as f:
#        idx_list_all = pickle.load(f)

    y_pred_dict = dict()
    y_pred_reg_dict = dict()
    y_pred_cl_dict = dict()
    
    y_dict = dict()

    i = 0
    batch = []
    keys = []

    with open(work_dir+log_file, 'a') as f:
        f.write("\nstarting the loop...")

    #for k in idx_list_all[year-2000-1]:

    #test_keys = idx_list_test[year-2000-2] + idx_list_test[year-2000-1]

    for j, k in enumerate(test_keys):
        if i < 512:
            batch.append(getitem(dataset, k))
            keys.append(k)
            i += 1
        else:
            X, data = custom_collate_fn_gnn(batch)
            y_pred, y, batch_idxs = model(X, data, device)
            y_pred_class, _, batch_idxs_class = model_cl(X, data, device)
            for ii, ki in enumerate(keys):
                idxi = torch.where(batch_idxs == ii)
                idxi_class = torch.where(batch_idxs_class == ii)
                y_pred_dict[ki] = (y_pred[idxi] * y_pred_class[idxi_class]).detach().cpu().numpy()
                y_pred_reg_dict[ki] = y_pred[idxi].detach().cpu().numpy()
                y_pred_cl_dict[ki] = y_pred_class[idxi_class].detach().cpu().numpy()
                y_dict[ki] = y[idxi].detach().cpu().numpy()
            if (j % 512) % 100 == 0:
                with open(work_dir+log_file, 'a') as f:
                    f.write(f"\nbatch {j // 512} done")
            i = 0
            batch = []
            keys = []
            batch.append(getitem(dataset, k))
            keys.append(k)
    
    with open(work_dir+log_file, 'a') as f:
        f.write(f"\nProcessing the last batch...")

    if i > 0:
        X, data = custom_collate_fn_gnn(batch)
        y_pred, y, batch_idxs = model(X, data, device)
        y_pred_class, y_class, batch_idxs_class = model_cl(X, data, device)
        for ii, ki in enumerate(keys):
            idxi = torch.where(batch_idxs == ii)
            idxi_class = torch.where(batch_idxs_class == ii)
            y_pred_dict[ki] = (y_pred[idxi] * y_pred_class[idxi_class]).detach().cpu().numpy()
            y_pred_reg_dict[ki] = y_pred[idxi].detach().cpu().numpy()
            y_pred_cl_dict[ki] = y_pred_class[idxi_class].detach().cpu().numpy()
            y_dict[ki] = y[idxi].detach().cpu().numpy()
        with open(work_dir+log_file, 'a') as f:
            f.write(f"\nbatch {j // 512 + 1} done")

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

    with open(work_dir+f"y_pred_{year}_small.pkl", 'wb') as f:
        pickle.dump(y_pred_dict, f)

    #with open(work_dir+f"y_pred_{year}_reg.pkl", 'wb') as f:
    #    pickle.dump(y_pred_reg_dict, f)

    #with open(work_dir+f"y_pred_{year}_cl.pkl", 'wb') as f:
    #    pickle.dump(y_pred_cl_dict, f)

    #with open(work_dir+f"y_{year}.pkl", 'wb') as f:
    #    pickle.dump(y_dict, f)

    #metric = BinaryConfusionMatrix()
    #metric(preds, target)

    #with open(work_dir+f"batch_idxs_{year}.pkl", 'wb') as f:
    #    pickle.dump(batch_idxs, f)

    #with open(work_dir+f"mask_{year}.pkl", 'wb') as f:
    #    pickle.dump(mask, f)

    with open(work_dir+log_file, 'a') as f:
        f.write("\ndone! :)")


