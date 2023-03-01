import numpy as np
import xarray as xr
import pickle
import matplotlib.pyplot as plt
import time
import argparse
import sys
import torch
import os

from torch_geometric.data import Data

#from numba import jit

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#-- paths
parser.add_argument('--output_path', type=str, default='/m100_work/ICT23_ESP_C/vblasone/climate-DL/preprocessing/')
parser.add_argument('--log_file', type=str, default='log.txt')
parser.add_argument('--target_path_file', type=str, default='/m100_work/ICT23_ESP_C/vblasone/GRIPHO/gripho-v1_1h_TSmin30pct_2001-2016_cut.nc')
parser.add_argument('--topo_path_file', type=str, default='/m100_work/ICT23_ESP_C/vblasone/TOPO/GMTED_DEM_30s_remapdis_GRIPHO.nc')
parser.add_argument('--input_path_file', type=str, default='/m100_work/ICT23_ESP_C/vblasone/SLICED/q_sliced.nc')
parser.add_argument('--idx_file', type=str, default='/work_dir/preprocessing/idx_time_2001-2016.pkl')

# lat lon grid values
parser.add_argument('--lon_min', type=float, default=6.50)
parser.add_argument('--lon_max', type=float, default=14.25)
parser.add_argument('--lat_min', type=float, default=43.50)
parser.add_argument('--lat_max', type=float, default=47.50)
parser.add_argument('--interval', type=float, default=0.25)
parser.add_argument('--time_dim', type=float, default=140256)

def cut_window(lon_min, lon_max, lat_min, lat_max, lon, lat, z, pr, time_dim):
    bool_lon = np.logical_and(lon >= lon_min, lon <= lon_max)
    bool_lat = np.logical_and(lat >= lat_min, lat <= lat_max)
    bool_both = np.logical_and(bool_lon, bool_lat)
    lon_sel = lon[bool_both]
    lat_sel = lat[bool_both]
    z_sel = z[bool_both]
    pr_sel = np.array(pr[:,bool_both])
    return lon_sel, lat_sel, z_sel, pr_sel

def select_nodes(lon_centre, lat_centre, lon, lat, pr, cell_idx, cell_idx_array, offset, time_dim):
    bool_lon = np.logical_and(lon >= lon_centre, lon <= lon_centre+offset)
    bool_lat = np.logical_and(lat >= lat_centre, lat <= lat_centre+offset)
    bool_both = np.logical_and(bool_lon, bool_lat)
    cell_idx_array[bool_both] = cell_idx
    flag_valid_example = False
    for i in np.argwhere(bool_both):
        if np.all(np.isnan(pr[:,i])):
            cell_idx_array[i] *= -1
        else:
            flag_valid_example = True
    return cell_idx_array, flag_valid_example

def write_log(s, args, mode='a'):
    with open(args.output_path + args.log_file, mode) as f:
        f.write(s)

def subdivide_train_test_time_indexes(idx_time_years, first_test_year=2016):
    idx_time_train = []
    idx_time_test = []
    for idx_time_y in idx_time_years[:first_test_year-2000-3]:
        idx_time_train += idx_time_y
    idx_time_train += idx_time_years[first_test_year-2000-2][:-31*24]
    idx_time_test = idx_time_years[first_test_year-2000-2][-31*24:] + idx_time_years[-1]
    idx_time_train.sort(); idx_time_test.sort()
    return idx_time_train, idx_time_test

'''
@jit
def derive_idxs_lists(valid_examples_space, idx_time_train, pr_sel_train_cl, pr_sel_train_reg, args):
    idx_train_cl = np.empty((0), dtype=np.int32)
    idx_train_reg = np.empty((0), dtype=np.int32)
    for i in range(len(valid_examples_space)):
        s = valid_examples_space[i]
        mask = np.in1d(cell_idx_array, s)
        pr_cl = pr_sel_train_cl[mask]
        pr_reg = pr_sel_train_reg[mask]
        for j in range(len(idx_time_train)):
            t = idx_time_train[j]
            if not np.isnan(pr_cl[:,t]).all():
                k = t * space_low_res_dim + s
                idx_train_cl = np.append(idx_train_cl, k)
                if not np.isnan(pr_reg[:,t]).all():
                    idx_train_reg = np.append(idx_train_reg, k)
        if s % 10 == 0:
            f = open(args.output_path + args.log_file, 'a')
            f.write(f"\nSpace idx {s} done.")
            f.close()
            #total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
            #write_log(f"\nRAM memory {round((used_memory/total_memory) * 100, 2)} %", args)
    #idx_train_cl = np.stack(idx_train_cl)  
    #idx_train_reg = np.stack(idx_train_reg)
    return idx_train_cl, idx_train_reg
'''

if __name__ == '__main__':

    args = parser.parse_args()
    
    with open(args.idx_file, 'rb') as f:
        idx_time_years = pickle.load(f)

    idx_time_train, idx_time_test = subdivide_train_test_time_indexes(idx_time_years)
    
    write_log("\nStart!", args, 'w')

    write_log(f"\nTrain idxs from {min(idx_time_train)} to {max(idx_time_train)}. Test idxs from {min(idx_time_test)} to {max(idx_time_test)}.", args, 'w')
    
    LON_DIFF_MAX = 0.25 / 8 * 2
    LAT_DIFF_MAX = 0.25 / 10 * 2

    gripho = xr.open_dataset(args.target_path_file)
    topo = xr.open_dataset(args.topo_path_file)

    lon = gripho.lon.to_numpy()
    lat = gripho.lat.to_numpy()
    pr = gripho.pr.to_numpy()
    z = topo.z.to_numpy()

    write_log("\nCutting the window...", args)

    # cut gripho and topo to the desired window
    lon_sel, lat_sel, z_sel, pr_sel = cut_window(args.lon_min, args.lon_max, args.lat_min, args.lat_max, lon, lat, z, pr, args.time_dim)

    write_log("\nDone!", args)

    cell_idx_array = np.zeros(pr_sel.shape[1])
    lon_low_res_array = np.arange(args.lon_min, args.lon_max, args.interval)
    lat_low_res_array = np.arange(args.lat_min, args.lat_max, args.interval)
    lon_low_res_dim = lon_low_res_array.shape[0]
    lat_low_res_dim = lat_low_res_array.shape[0]
    space_low_res_dim = lon_low_res_dim * lat_low_res_dim

    # start the preprocessing
    write_log(f"\nStarting the preprocessing.", args)
    start = time.time()
    
    valid_examples_space = [ii * lon_low_res_dim + jj for ii in range(1,lat_low_res_dim-1) for jj in range(1,lon_low_res_dim-1)]
    graph_cells_space = []

    for i, lat_low_res in enumerate(lat_low_res_array):
        for j, lon_low_res in enumerate(lon_low_res_array):
            cell_idx = i * lon_low_res_dim + j
            cell_idx_array, flag_valid_example = select_nodes(lon_low_res, lat_low_res, lon_sel, lat_sel, pr_sel, cell_idx,
                    cell_idx_array, args.interval, args.time_dim)         
            if cell_idx in valid_examples_space:
                if flag_valid_example:
                    idx_list = np.array([ii * lon_low_res_dim + jj for ii in range(i-1,i+2) for jj in range(j-1,j+2)])
                    _ = [graph_cells_space.append(abs(d)) for d in idx_list]
                else:
                    valid_examples_space.remove(cell_idx)

    graph_cells_space = list(set(graph_cells_space))
    graph_cells_space.sort()
    valid_examples_space.sort()

    end = time.time()
    write_log(f'\nLoop took {end - start} s', args)

    with open(args.output_path + 'cell_idx_array.pkl', 'wb') as f:         # array that assigns to each high res node the corresponding low res cell index
        pickle.dump(cell_idx_array, f)

    with open(args.output_path + 'valid_examples_space.pkl', 'wb') as f:   # low res cells indexes valid as examples for the training
        pickle.dump(valid_examples_space, f)

    with open(args.output_path + 'graph_cells_space.pkl', 'wb') as f:      # all low res cells that are used (examples + surroundings)
        pickle.dump(graph_cells_space, f)

    # keep only the graph cells space idxs
    mask_graph_cells_space = np.in1d(abs(cell_idx_array), graph_cells_space)
    lon_sel = lon_sel[mask_graph_cells_space]
    lat_sel = lat_sel[mask_graph_cells_space]
    z_sel = z_sel[mask_graph_cells_space]
    pr_sel = pr_sel[:, mask_graph_cells_space]
    cell_idx_array = cell_idx_array[mask_graph_cells_space]

    threshold = 0.1 # mm
    pr_sel = pr_sel.swapaxes(0,1) # (num_nodes, time)
    pr_sel_train = pr_sel[:,min(idx_time_train):max(idx_time_train)+1]
    pr_sel_train_cl = np.array([np.where(pr >= threshold, 1, 0) for pr in pr_sel_train], dtype=np.float32)
    pr_sel_train_cl[np.isnan(pr_sel_train)] = np.nan
    pr_sel_train_reg = np.array([np.where(pr >= threshold, np.log1p(pr), np.nan) for pr in pr_sel_train], dtype=np.float32)
    pr_sel_test = pr_sel[:,min(idx_time_test):max(idx_time_test)+1]

    z_sel_s = (z_sel - z_sel.mean()) / z_sel.std()
    
    pos = np.column_stack((lon_sel,lat_sel))

    edge_index = np.empty((2,0), dtype=int)
    edge_attr = np.empty((2,0), dtype=float)

    for ii, xi in enumerate(pos):
        bool_lon = abs(pos[:,0] - xi[0]) < LON_DIFF_MAX
        bool_lat = abs(pos[:,1] - xi[1]) < LAT_DIFF_MAX
        bool_both = np.logical_and(bool_lon, bool_lat)
        jj_list = np.flatnonzero(bool_both)
        xj_list = pos[bool_both, :]
        for jj, xj in zip(jj_list, xj_list):
            if not np.array_equal(xi, xj):
                edge_index = np.concatenate((edge_index, np.array([[ii], [jj]])), axis=-1, dtype=int)
                edge_attr = np.concatenate((edge_attr, np.array([[xj[0] - xi[0]], [xj[1] - xi[1]]])), axis=-1, dtype=float)
        #write_log(f"\nStart node: {xi} - done. Node has {n_neighbours} neighbours.", args)

    edge_attr = edge_attr.swapaxes(0,1)
    edge_attr[:,0] = edge_attr[:,0] / edge_attr[:,0].max() 
    edge_attr[:,1] = edge_attr[:,1] / edge_attr[:,1].max()
    
    # convert to torch tensors
    '''
    pr_sel_train_cl = torch.tensor(pr_sel_train_cl)
    pr_sel_train_reg = torch.tensor(pr_sel_train_reg)
    pr_sel_test = torch.tensor(pr_sel_test)
    cell_idx_array = torch.tensor(cell_idx_array)
    pos = torch.tensor(pos)
    edge_index = torch.tensor(edge_index)
    edge_attr = torch.tensor(edge_attr)
    z_sel_s = torch.tensor(z_sel_s)
    '''

    # create the graph objects
    G_test = Data(pos=torch.tensor(pos), pr=torch.tensor(pr_sel_test), low_res=torch.tensor(abs(cell_idx_array)))
    G_train = Data(x=torch.tensor(z_sel_s), edge_index=torch.tensor(edge_index), edge_attr=torch.tensor(edge_attr),
            low_res=torch.tensor(abs(cell_idx_array)))
    #G_train_reg = Data(x=z_sel_s, edge_index=edge_index, edge_attr=edge_attr, low_res=cell_idx_array, y=pr_sel_train_reg)

    with open(args.output_path + 'G_north_italy_test.pkl', 'wb') as f:
        pickle.dump(G_test, f)
    
    with open(args.output_path + 'G_north_italy_train.pkl', 'wb') as f:
        pickle.dump(G_train, f)
    
    with open(args.output_path + 'target_train_cl.pkl', 'wb') as f:
        pickle.dump(pr_sel_train_cl, f)    
    
    with open(args.output_path + 'target_train_reg.pkl', 'wb') as f:
        pickle.dump(pr_sel_train_reg, f)    
    
    write_log(f"\nIn total, preprocessing took {time.time() - start} seconds", args)    

    # create the indexes list for the dataloader
    write_log("\nLet's now create the list of indexes for the training.", args)
    start = time.time()

    idx_train_cl = []
    idx_train_reg = []
    mask_train_cl = []
    mask_train_reg = []
    mask_subgraphs = torch.tensor(np.ones((space_low_res_dim, pr_sel.shape[0])) * np.nan)

    cell_idx_array = torch.tensor(cell_idx_array).cuda()
    valid_examples_space = torch.tensor(valid_examples_space).cuda()
    pr_sel_train_cl = torch.tensor(pr_sel_train_cl)
    pr_sel_train_reg = torch.tensor(pr_sel_train_reg)
    
    '''
    def derive_idxs_lists(valid_examples_space, idx_time_train):
        masks_list = [torch.isin(cell_idx_array, s) for s in valid_examples_space]
        pr_cl_reg_list = [[pr_sel_train_cl[mask], pr_sel_train_reg[mask]] for mask in masks_list]
        for t in idx_time_train:
            valid_s_cl_list = [torch.isnan(pr_cl_reg_list[i][:,t]).all()
    '''
    
    with torch.no_grad():
        for s in valid_examples_space:
            mask = torch.isin(cell_idx_array, s)
            mask_subgraphs[s,:] = mask.cpu()
            pr_cl = pr_sel_train_cl[mask].cuda()
            pr_reg = pr_sel_train_reg[mask].cuda()
            for t in idx_time_train:
                nan_cl = torch.isnan(pr_cl[:,t])
                if not nan_cl.all():
                    k = t * space_low_res_dim + s
                    mask_cl = torch.clone(mask)
                    mask_cl[mask==1] = ~nan_cl
                    idx_train_cl.append(k)
                    mask_cl = mask_cl.cpu()
                    mask_train_cl.append(mask_cl)
                    nan_reg = torch.isnan(pr_reg[:,t])
                    if not nan_reg.all():
                        mask_reg = torch.clone(mask)
                        mask_reg[mask==1] = ~nan_reg
                        idx_train_reg.append(k)
                        mask_reg = mask_reg.cpu()
                        mask_train_reg.append(mask_reg)
            torch.cuda.empty_cache()
            if s % 10 == 0:
                write_log(f"\nSpace idx {s} done.", args)
                total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
                write_log(f"\nRAM memory {round((used_memory/total_memory) * 100, 2)} %", args)
            #del mask; del pr_cl; del pr_reg; del nan_cl; del nan_reg; del k;
        
    #idx_train_cl = torch.stack(idx_train_cl)
    #idx_train_cl = idx_train_cl.cpu().numpy()

    #idx_train_reg = torch.stack(idx_train_reg)
    #idx_train_reg = idx_train_reg.cpu().numpy()

    idx_train_cl = np.array(idx_train_cl)
    idx_train_reg = np.array(idx_train_reg)

    mask_train_cl = torch.stack(mask_train_cl)
    mask_train_cl = mask_train_cl.numpy()

    mask_train_reg = torch.stack(mask_train_reg)
    mask_train_reg = mask_train_reg.numpy()

    with open(args.output_path + 'idx_train_cl.pkl', 'wb') as f:
        pickle.dump(idx_train_cl, f)

    with open(args.output_path + 'idx_train_reg.pkl', 'wb') as f:
        pickle.dump(idx_train_reg, f)
    
    with open(args.output_path + 'mask_train_cl.pkl', 'wb') as f:
        pickle.dump(mask_train_cl, f)

    with open(args.output_path + 'mask_train_reg.pkl', 'wb') as f:
        pickle.dump(mask_train_reg, f)
    
    write_log(f"\nCreating the idx array took {time.time() - start} seconds", args)    

    write_log("\nDone!", args)


