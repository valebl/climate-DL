import numpy as np
import pickle
import torch
import argparse
import time
import os
import matplotlib.pyplot as plt
import sys

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#-- paths
parser.add_argument('--input_path', type=str, help='path to input directory')
parser.add_argument('--output_path', type=str, help='path to output directory')
parser.add_argument('--base_path', type=str, help='path to climate-DL')

#-- input files
parser.add_argument('--input_file', type=str, default="input_standard.pkl")
parser.add_argument('--test_graph_file', type=str) 
parser.add_argument('--subgraphs_file', type=str) 
parser.add_argument('--checkpoint_cl', type=str)
parser.add_argument('--checkpoint_reg', type=str)
parser.add_argument('--output_file', type=str, default="G_predictions.pkl")

#-- output files
parser.add_argument('--log_file', type=str, default='log.txt', help='log file')

#-- boolean
parser.add_argument('--use_accelerate',  action='store_true')
parser.add_argument('--no-use_accelerate', dest='use_accelerate', action='store_false')
parser.add_argument('--make_plots',  action='store_true', default=False)
parser.add_argument('--large_graph',  action='store_true')
parser.add_argument('--no-large_graph', dest='large_graph', action='store_false')

#-- other
#parser.add_argument('--test_year', type=int, default=2016)
parser.add_argument('--batch_size', type=int, default=128, help='batch size (global)')
parser.add_argument('--model_name_cl', type=str, default='Classifier_old_test')
parser.add_argument('--model_name_reg', type=str, default='Regressor_old_test')
#parser.add_argument('--idx_min', type=int, default=130728)
parser.add_argument('--img_extension', type=str, default='pdf')

parser.add_argument('--first_year', type=int, default=None)
parser.add_argument('--year_start', type=int, default=None)
parser.add_argument('--month_start', type=int, default=None)
parser.add_argument('--day_start', type=int, default=None)
parser.add_argument('--year_end', type=int, default=None)
parser.add_argument('--month_end', type=int, default=None)
parser.add_argument('--day_end', type=int, default=None)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--cmap', type=str, default='turbo')

parser.add_argument('--lon_min', type=float)
parser.add_argument('--lon_max', type=float)
parser.add_argument('--lat_min', type=float)
parser.add_argument('--lat_max', type=float)
parser.add_argument('--interval', type=float, default=0.25)
parser.add_argument('--lon_dim', type=int, default=None)
parser.add_argument('--lat_dim', type=int, default=None)

#from torchmetrics.classification import BinaryConfusionMatrix

args = parser.parse_args()
sys.path.append(args.base_path+"/climate-DL/local_single")

import models
import dataset
from utils import load_encoder_checkpoint as load_checkpoint, Tester
from utils_predictions import create_zones, plot_maps, date_to_day, extremes_cmap


if __name__ == '__main__':

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    with open(args.output_path + args.log_file, 'w') as f:
        f.write(f"Starting!\nBatch size is {args.batch_size}.")

    ## derive arrays corresponding to the lon/lat low resolution grid points
    lon_low_res_array = np.arange(args.lon_min-args.interval, args.lon_max+args.interval, args.interval)
    lat_low_res_array = np.arange(args.lat_min-args.interval, args.lat_max+args.interval, args.interval)
    args.lon_dim = lon_low_res_array.shape[0]
    args.lat_dim = lat_low_res_array.shape[0]
    spatial_points_dim = args.lat_dim * args.lon_dim

    with open(args.input_path + args.test_graph_file, 'rb') as f:
        G_test = pickle.load(f)

    start_idx, end_idx = date_to_day(args.year_start, args.month_start, args.day_start, args.year_end, args.month_end, args.day_end, first_year=args.first_year)
    start_idx = max(start_idx,25)
    idx_to_key_time = list(range(start_idx, end_idx))

    if not args.large_graph:
        idx_to_key = []
        for time_idx in idx_to_key_time:
            for space_idx in G_test.valid_examples_space:
                k = time_idx * spatial_points_dim + space_idx
                idx_to_key.append(k)
    else:
        idx_to_key=None

    with open(args.output_path + args.log_file, 'a') as f:
        f.write(f"\nTest idxs from {start_idx} to {end_idx}")

    G_test["y"] = G_test["y"][:,torch.tensor(idx_to_key_time)]
    G_test["pr_cl"] = torch.zeros(G_test["y"].shape)
    G_test["pr_reg"] = torch.zeros(G_test["y"].shape)

#-----------------------------------------------------
#----------------- DATASET AND MODELS ----------------
#-----------------------------------------------------

    if args.large_graph:
        dataset_type = 'Dataset_pr_test_large'
        custom_collate_type = 'custom_collate_fn_gnn_large'
    else:
        dataset_type = 'Dataset_pr_test'
        custom_collate_type = 'custom_collate_fn_gnn'
    
    with open(args.output_path + args.log_file, 'a') as f:
        f.write(f"\nUsing {dataset_type} and {custom_collate_type}.")

    Dataset = getattr(dataset, dataset_type)
    custom_collate_fn = getattr(dataset, custom_collate_type)
    
    with open(args.output_path + args.log_file, 'a') as f:
        f.write("\nBuilding the dataset and the dataloader.")

    dataset = Dataset(args=args, lon_dim=args.lon_dim, lat_dim=args.lat_dim, idx_to_key=idx_to_key, idx_to_key_time=idx_to_key_time, time_min=min(idx_to_key_time))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)

    with open(args.output_path + args.log_file, 'a') as f:
        f.write("\nDone!")
        f.write(f"\nUsing {args.model_name_cl} and {args.model_name_reg}")
        f.write("\nInstantiate models and load checkpoints.\n")

    Model_cl = getattr(models, args.model_name_cl)
    Model_reg = getattr(models, args.model_name_reg)

    model_cl = Model_cl()
    model_reg = Model_reg()

    with open(args.output_path + args.log_file, 'a') as f:
        f.write("\nClassifier:")

    checkpoint_cl = load_checkpoint(model_cl, args.checkpoint_cl, args.output_path, args.log_file, None, 
            net_names=["encoder.", "gru.", "dense.", "gnn."], fine_tuning=False, device=args.device)

    with open(args.output_path + args.log_file, 'a') as f:
        f.write("\nRegressor:")

    checkpoint_reg = load_checkpoint(model_reg, args.checkpoint_reg, args.output_path, args.log_file, None,
            net_names=["encoder.", "gru.", "dense.", "gnn."], fine_tuning=False, device=args.device)

    model_cl = model_cl.to(args.device)
    model_reg = model_reg.to(args.device)

    with open(args.output_path + args.log_file, 'a') as f:
        f.write("\n\nDone!")

#-----------------------------------------------------
#-------------------- PREDICTIONS --------------------
#-----------------------------------------------------

    with open(args.output_path + args.log_file, 'a') as f:
        f.write(f"\nStarting the test, from {args.day_start}/{args.month_start}/{args.year_start} to {args.day_end}/{args.month_end}/{args.year_end}.")

    tester = Tester()
    
    start = time.time()
    tester.test(model_cl, model_reg, dataloader, G_test=G_test, args=args)
    end = time.time()

    with open(args.output_path + args.log_file, 'a') as f:
        f.write(f"\nDone. Testing concluded in {end-start} seconds.")
        f.write("\nWrite the files.")

    with open(args.output_path + args.output_file, 'wb') as f:
        pickle.dump(G_test, f)

#-----------------------------------------------------
#----------------------- PLOTS -----------------------
#-----------------------------------------------------

    if args.cmap == 'extremes':
        cmap = extremes_cmap()
    else:
        cmap = 'turbo'

    x_size = args.lon_dim
    y_size = args.lat_dim
    font_size = 42 #int(12 / 7 * x_size)
    font_size_title = 60 #int(20 / 7 * x_size)

    if args.make_plots:        
        with open(args.output_path + args.log_file, 'a') as f:
            f.write(f"\n\nMaking some plots.")
        with open(args.input_path+"valid_examples_space.pkl", 'rb') as f:
            valid_examples_space = pickle.load(f)
        zones = create_zones(zones_file=args.base_path+"climate-DL/preprocessing/Italia.txt")
        mask = np.logical_and(np.array([~torch.isnan(G_test.y[i,:]).all().numpy() for i in range(G_test.pr.shape[0])]), np.in1d(G_test.low_res, valid_examples_space))
        y_target = G_test.y.numpy()[mask,:]
        # Classifier
        y_target_cl = torch.where(torch.tensor(y_target) >= 0.1, 1.0, 0.0)
        corrects = (G_test.pr_cl.numpy()[mask,:].flatten() == y_target_cl.numpy().flatten())
        acc = corrects.sum() / len(y_target_cl.flatten()) * 100
        acc_0 = corrects[y_target_cl.flatten()==0].sum() / len(y_target_cl.flatten()[y_target_cl.flatten()==0]) * 100
        acc_1 = corrects[y_target_cl.flatten()==1].sum() / len(y_target_cl.flatten()[y_target_cl.flatten()==1]) * 100            
        with open(args.output_path + args.log_file, 'a') as f:
            f.write(f"\nClassifier\nAccuracy: {acc:.2f}\nAccuracy on class 0: {acc_0:.2f}\nAccuracy on class 1: {acc_1:.2f}")
        
        plot_maps(G_test.pos[mask,:], G_test.pr_cl.numpy()[mask,:], y_target_cl, pr_min=0.1, aggr=np.nansum, cmap=cmap, pr_max=75,
            title=f"Classifier - Number of hours with pr>=0.1mm (from {args.day_start}/{args.month_start}/{args.year_start} to {args.day_end}/{args.month_end}/{args.year_end})",
            idx_start=0, idx_end=-1, legend_title="hours", save_path=args.output_path, save_file_name=f"maps_cl.{args.img_extension}", zones=zones, x_size=x_size, y_size=y_size,
            font_size=font_size, font_size_title=font_size_title)
        # Regressor
        pr_mask = G_test.y.numpy()[mask,:] >= 0.1
        plot_maps(G_test.pos[mask,:], G_test.pr_reg.numpy()[mask,:] * pr_mask, y_target * pr_mask, pr_min=0.1, aggr=np.nansum, cmap=cmap, pr_max=350,
            title=f"Regressor - Cumulative precipitation when pr>=0.1 in observations (from {args.day_start}/{args.month_start}/{args.year_start} to {args.day_end}/{args.month_end}/{args.year_end})",
            idx_start=0, idx_end=-1, save_path=args.output_path, save_file_name=f"maps_reg.{args.img_extension}", zones=zones, x_size=x_size, y_size=y_size,
            font_size=font_size, font_size_title=font_size_title)
        # Combines results
        plot_maps(G_test.pos[mask,:], G_test.pr.numpy()[mask,:], y_target, pr_min=0.1, aggr=np.nansum, cmap=cmap, pr_max=350,
            title=f"Cumulative precipitation (from {args.day_start}/{args.month_start}/{args.year_start} to {args.day_end}/{args.month_end}/{args.year_end})", idx_start=0, idx_end=-1,
            save_path=args.output_path, save_file_name=f"maps_cumulative.{args.img_extension}", zones=zones, x_size=x_size, y_size=y_size, font_size=font_size, font_size_title=font_size_title)
        plot_maps(G_test.pos[mask,:], G_test.pr.numpy()[mask,:], y_target, pr_min=0.0, aggr=np.nanmean, cmap=cmap, pr_max=5,
            title=f"Mean precipitation (from {args.day_start}/{args.month_start}/{args.year_start} to {args.day_end}/{args.month_end}/{args.year_end})", idx_start=0, idx_end=-1,
            save_path=args.output_path, save_file_name=f"maps_mean.{args.img_extension}", zones=zones, x_size=x_size, y_size=y_size, font_size=font_size, font_size_title=font_size_title)
        # Histogram
        plt.rcParams.update({'font.size': 16})
        y = G_test.y.numpy()[mask,:].flatten()
        pr = G_test.pr.numpy()[mask,:].flatten()
        pr_reg = y_target.flatten()
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

            
