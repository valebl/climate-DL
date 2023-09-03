import numpy as np
import xarray as xr
import pickle
import time
import argparse
import sys
import torch
import os
import matplotlib.pyplot as plt

from torch_geometric.data import Data
from utils_preprocessing import create_zones, plot_italy

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#-- paths
parser.add_argument('--input_path', type=str, default='/leonardo_work/ICT23_ESP_0/SHARED/preprocessed/north_italy/')
parser.add_argument('--output_path', type=str, default='/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/preprocessing/')
parser.add_argument('--plot_name', type=str, default='graph.png')

#-- lat lon grid values
parser.add_argument('--lon_min', type=float, default=6.75)
parser.add_argument('--lon_max', type=float, default=14.00)
parser.add_argument('--lat_min', type=float, default=43.75)
parser.add_argument('--lat_max', type=float, default=47.00)
parser.add_argument('--interval', type=float, default=0.25)
parser.add_argument('--time_dim', type=float, default=140256)
parser.add_argument('--make_plots', action='store_true', default=True)

if __name__ == '__main__':

    args = parser.parse_args()
    
    ## derive arrays corresponding to the lon/lat low resolution grid points
    lon_low_res_array = np.arange(args.lon_min-args.interval, args.lon_max+args.interval, args.interval)
    lat_low_res_array = np.arange(args.lat_min-args.interval, args.lat_max+args.interval, args.interval)
    lon_low_res_dim = lon_low_res_array.shape[0]
    lat_low_res_dim = lat_low_res_array.shape[0]
    space_low_res_dim = lon_low_res_dim * lat_low_res_dim
    
    lon_high_res_array = np.arange(args.lon_min, args.lon_max, args.interval)
    lat_high_res_array = np.arange(args.lat_min, args.lat_max, args.interval)
    lon_high_res_dim = lon_high_res_array.shape[0] - 1
    lat_high_res_dim = lat_high_res_array.shape[0] - 1

    lon_input_points_array = np.arange(args.lon_min-args.interval*3, args.lon_max+args.interval*4, args.interval)
    lat_input_points_array = np.arange(args.lat_min-args.interval*3, args.lat_max+args.interval*4, args.interval)
    lon_input_points_dim = lon_input_points_array.shape[0]
    lat_input_points_dim = lat_input_points_array.shape[0]
    space_input_points_dim = lon_input_points_dim * lat_input_points_dim

    args.lon_min = args.lon_min - args.interval
    args.lon_max = args.lon_max + args.interval
    args.lat_min = args.lat_min - args.interval
    args.lat_max = args.lat_max + args.interval

    with open(args.input_path + 'G_test.pkl', 'rb') as f:
        G = pickle.load(f)
    
    with open(args.input_path + 'subgraphs.pkl', 'rb') as f:
        subgraphs = pickle.load(f)

    with open(args.input_path + 'cell_idx_array.pkl', 'rb') as f:
        cell_idx_array = pickle.load(f)
    
    pos = G.pos.numpy()

    #-----------------------------------------------------
    #----------------------- PLOTS -----------------------
    #-----------------------------------------------------

    if args.make_plots:

        x_size = lon_input_points_dim
        y_size = lat_input_points_dim
        
        y_size_fig = 8.27
        x_size_fig = x_size * y_size_fig / y_size

        font_size = 16
        font_size_text = 6 * 41 / (x_size+5)
        linewidth_grids = 0.5
        linewidth_italy = 1
        area=4.0 * 41 / (x_size+5)

        print(x_size_fig, y_size_fig, font_size, font_size_text, linewidth_grids, linewidth_italy)

        plt.rcParams.update({'font.size': font_size})

        fig, ax = plt.subplots(figsize=(x_size_fig, y_size_fig))

        zones = create_zones()
        
        _ = ax.scatter(pos[:,0], pos[:,1], c='grey',alpha=0.2,marker="s", s=area)
        c = torch.tensor(abs(cell_idx_array)).int()

        for s in subgraphs:
            if s != []:
                alpha = np.ones(s.mask_1_cell.sum())*0.2
                _ = ax.scatter(pos[:,0][s.mask_1_cell],pos[:,1][s.mask_1_cell],
                        c=c[s.mask_1_cell], marker="s", s=area, vmin = 0,
                        vmax = space_low_res_dim, cmap='turbo', alpha=alpha)

        for l in lon_input_points_array:
            _ = ax.plot([l, l], [lat_input_points_array.min(), lat_input_points_array.max()], 'b', alpha=0.2, linewidth=linewidth_grids)

        for l in lat_input_points_array:
            _ = ax.plot([lon_input_points_array.min(), lon_input_points_array.max()], [l,l], 'b', alpha=0.2, linewidth=linewidth_grids)

        lon_plot_array = np.arange(lon_low_res_array.min(), lon_low_res_array.max()+args.interval*2, args.interval)
        lat_plot_array = np.arange(lat_low_res_array.min(), lat_low_res_array.max()+args.interval*2, args.interval)

        for l in lon_plot_array:
            _ = ax.plot([l, l], [lat_plot_array.min(), lat_plot_array.max()], 'k', alpha=0.4, linewidth=linewidth_grids)

        for l in lat_plot_array:
            _ = ax.plot([lon_plot_array.min(), lon_plot_array.max()], [l,l], 'k', alpha=0.4, linewidth=linewidth_grids)

        for i, lat_i in enumerate(lat_low_res_array):
            for j, lon_j in enumerate(lon_low_res_array):
                k = i * lon_low_res_dim + j
                _ = ax.text(lon_j+0.05, lat_i+0.1, k, fontsize=font_size_text)
                                                                                                        
        plot_italy(zones, color='black', ax=ax, alpha_fill=0, linewidth=linewidth_italy)
        plt.xlim([lon_input_points_array.min() - 0.25, lon_input_points_array.max() + 0.25])
        plt.ylim([lat_input_points_array.min() - 0.25, lat_input_points_array.max() + 0.25])
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.savefig(args.output_path + args.plot_name, dpi=400, bbox_inches='tight', pad_inches=0.0)

        print("Done!")
