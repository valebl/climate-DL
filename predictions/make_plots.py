import numpy as np
import pickle
import torch
import argparse
import time
import os
import matplotlib.pyplot as plt
import sys
sys.path.append("/m100_work/ICT23_ESP_C/vblasone/climate-DL/local_single")
#sys.path.append("/home/vblasone/climate-DL/local_multiple")

import models
import dataset
from utils import load_encoder_checkpoint as load_checkpoint, Tester
from utils_predictions import create_zones, plot_maps

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#-- paths
parser.add_argument('--input_path', type=str, help='path to input directory')
parser.add_argument('--output_path', type=str, help='path to output directory')

#-- input files
parser.add_argument('--graph_file_test', type=str) 

#-- output files
parser.add_argument('--log_file', type=str, default='log.txt', help='log file')

#-- other
parser.add_argument('--test_year', type=int)
parser.add_argument('--lon_dim', type=int)
parser.add_argument('--lat_dim', type=int)
parser.add_argument('--idx_min', type=int, default=130728)
parser.add_argument('--img_extension', type=str, default='pdf')



if __name__ == '__main__':

    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    TIME_DIM = 140256
    spatial_points_dim = args.lat_dim * args.lon_dim

    with open(args.graph_file_test, 'rb') as f:
        G_test = pickle.load(f)

#-----------------------------------------------------
#----------------------- PLOTS -----------------------
#-----------------------------------------------------

    x_size = args.lon_dim
    y_size = args.lat_dim
    font_size = int(12 / 7 * x_size)

    with open(args.output_path + args.log_file, 'w') as f:
        f.write(f"\n\nMaking some plots.")
    with open(args.input_path+"valid_examples_space.pkl", 'rb') as f:
        valid_examples_space = pickle.load(f)
    zones = create_zones()
    mask = np.logical_and(np.array([~torch.isnan(G_test.y[i,:]).all().numpy() for i in range(G_test.pr.shape[0])]), np.in1d(G_test.low_res, valid_examples_space))
    # Classifier
    y_cl = torch.where(G_test.y[mask,:] >= 0.1, 1.0, 0.0)
    corrects = (G_test.pr_cl.numpy()[mask,:].flatten() == y_cl.numpy().flatten())
    acc = corrects.sum() / len(y_cl.flatten()) * 100
    acc_0 = corrects[y_cl.flatten()==0].sum() / len(y_cl.flatten()[y_cl.flatten()==0]) * 100
    acc_1 = corrects[y_cl.flatten()==1].sum() / len(y_cl.flatten()[y_cl.flatten()==1]) * 100            
    with open(args.output_path + args.log_file, 'a') as f:
        f.write(f"\nClassifier\nAccuracy: {acc:.2f}\nAccuracy on class 0: {acc_0:.2f}\nAccuracy on class 1: {acc_1:.2f}")
    
    plot_maps(G_test.pos[mask,:], G_test.pr_cl.numpy()[mask,:], y_cl.numpy(), pr_min=0.1, pr_max=2000, aggr=np.nansum,
        title=f"Classifier - Number of hours with pr>=0.1mm for the year {args.test_year}", idx_start=0, idx_end=-1, legend_title="hours",
        save_path=args.output_path, save_file_name=f"maps_cl_{args.test_year}.{args.img_extension}", zones=zones, x_size=x_size, y_size=y_size, font_size=font_size)
    # Regressor
    pr_mask = G_test.y.numpy()[mask,:] >= 0.1
    plot_maps(G_test.pos[mask,:], G_test.pr_reg.numpy()[mask,:] * pr_mask, G_test.y.numpy()[mask,:] * pr_mask, pr_min=0.1, pr_max=2500, aggr=np.nansum,
        title=f"Regressor - Cumulative precipitation when pr>=0.1 in observations for the year {args.test_year}", idx_start=0, idx_end=-1,
        save_path=args.output_path, save_file_name=f"maps_reg_{args.test_year}.{args.img_extension}", zones=zones, x_size=x_size, y_size=y_size, font_size=font_size)
    # Combines results
    plot_maps(G_test.pos[mask,:], G_test.pr.numpy()[mask,:], G_test.y.numpy()[mask,:], pr_min=0.1, pr_max=2500, aggr=np.nansum,
        title=f"Cumulative precipitation for the year {args.test_year}", idx_start=0, idx_end=-1,
        save_path=args.output_path, save_file_name=f"maps_cumulative_{args.test_year}.{args.img_extension}", zones=zones, x_size=x_size, y_size=y_size, font_size=font_size)
    plot_maps(G_test.pos[mask,:], G_test.pr.numpy()[mask,:], G_test.y.numpy()[mask,:], pr_min=0.0, pr_max=0.25, aggr=np.nanmean,
        title=f"Mean precipitation for the year {args.test_year}", idx_start=0, idx_end=-1,
        save_path=args.output_path, save_file_name=f"maps_mean_{args.test_year}.{args.img_extension}", zones=zones, x_size=x_size, y_size=y_size, font_size=font_size)
    # Histogram
    plt.rcParams.update({'font.size': 16})
    y = G_test.y.numpy()[mask,:].flatten()
    pr = G_test.pr.numpy()[mask,:].flatten()
    pr_reg = G_test.pr_reg[mask,:].numpy().flatten()
    binwidth = 1
    fig, ax = plt.subplots(figsize=(16,8))
    _ = plt.hist(y, bins=range(int(min(y)), int(max(y)+binwidth), binwidth), facecolor='blue', alpha=0.2, density=True, label='observations', edgecolor='k') 
    _ = plt.hist(pr, bins=range(int(pr.min()), int(pr.max()+binwidth), binwidth), facecolor='orange', alpha=0.2, density=True, label='predictions', edgecolor='k')  # arguments are passed to np.histogram
    # _ = plt.hist(pr_reg, bins=range(int(pr_reg.min()), int(pr_reg.max()+binwidth), binwidth), facecolor='green', alpha=0.2, density=True, label='regression', edgecolor='k') 
    plt.title('Histogram', fontsize=18)
    ax.set_yscale('log')
    ax.set_xlabel('precipitation [mm]')
    ax.set_ylabel('count (normalised)')
    plt.legend()
    plt.savefig(f'{args.output_path}histogram.{args.img_extension}', dpi=400, bbox_inches='tight', pad_inches=0.0) 

    with open(args.output_path + args.log_file, 'a') as f:
        f.write(f"\n\nDone.")


