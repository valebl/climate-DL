import numpy as np
import xarray as xr
import torch
import pickle
import argparse
import os
import sys

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--input_path', type=str, help='path to input directory', default='/m100_work/ICT23_ESP_C/vblasone/NORTH_ITALY/')
parser.add_argument('--output_path', type=str, help='path to output directory', default='/m100_work/ICT23_ESP_C/vblasone/NORTH_ITALY/')
parser.add_argument('--input_files_prefix', type=str, help='prefix for the input files (convenction: {prefix}{parameter}.nc)', default='sliced_')
parser.add_argument('--log_file', type=str, help='log file name', default='log.txt')
parser.add_argument('--output_file', type=str, help='path to output directory', default='input_ds_standard.pkl')
parser.add_argument('--n_levels', type=int, help='number of pressure levels considered', default=5)
parser.add_argument('--stats_path', type=str, default='/m100_work/ICT23_ESP_C/vblasone/NORTH_ITALY/north_italy/')
parser.add_argument('--means_file', type=str, default='means.pkl')
parser.add_argument('--stds_file', type=str, default='stds.pkl')
parser.add_argument('--mean_std_over_variable', action='store_true')
parser.add_argument('--mean_std_over_variable_and_level', dest='mean_std_over_variable', action='store_false')
parser.add_argument('--load_stats', action='store_true',help='load means and stds from files')


if __name__ == '__main__':

    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    params = ['q', 't', 'u', 'v', 'z']
    n_params = len(params)
 
    #-----------------------------------------------------
    #-------------- INPUT TENSOR FROM FILES --------------
    #-----------------------------------------------------
    
    with open(args.output_path + args.log_file, 'w') as f:
        f.write(f'\nStarting to create the input dataset from files.')

    for p_idx, p in enumerate(params):
        with open(args.output_path + args.log_file, 'a') as f:
            f.write(f'\nPreprocessing {args.input_files_prefix}{p}.nc ...')
        with xr.open_dataset(f'{args.input_path}/{args.input_files_prefix}{p}.nc') as f:
            data = f[p].values
            if p_idx == 0: # first parameter being processed -> get dimensions and initialize the input dataset
                lat_dim = len(f.latitude)
                lon_dim = len(f.longitude)
                time_dim = len(f.time)
                input_ds = np.zeros((time_dim, n_params, args.n_levels, lat_dim, lon_dim), dtype=np.float32) # variables, levels, time, lat, lon
        input_ds[:, p_idx,:,:,:] = data

    #-----------------------------------------------------
    #-------------- POST-PROCESSING OF INPUT--------------
    #-----------------------------------------------------
    
    # flip the dataset
    input_ds = np.flip(input_ds, 3) # the origin in the input files is in the top left corner, while we use the bottom left corner

    # standardizing the dataset
    with open(args.output_path + args.log_file, 'a') as f:
        f.write(f'\nStandardizing the dataset.')
    
    input_ds_standard = np.zeros((input_ds.shape), dtype=np.float32)
    
    if args.load_stats:
        with open(args.stats_path+args.means_file, 'rb') as f:
            means = pickle.load(f)
        with open(args.stats_path+args.stds_file, 'rb') as f:
            stds = pickle.load(f)

    if not args.mean_std_over_variable:
        if not args.load_stats:
            means = np.zeros((5))
            stds = np.zeros((5))
            for var in range(5):
                m = np.mean(input_ds[:,var,:,:,:])
                s = np.std(input_ds[:,var,:,:,:])
                input_ds_standard[:,var,:,:,:] = (input_ds[:,var,:,:,:]-m)/s
                means[var] = m
                stds[var] = s
        else:
            for var in range(5):
                input_ds_standard[:,var,:,:,:] = (input_ds[:,var,:,:,:]-means[var])/stds[var]    
    else:
        if not args.load_stats:
            means = np.zeros((5,5))
            stds = np.zeros((5,5))
            for var in range(5):
                for lev in range(5):
                    m = np.mean(input_ds[:,var,lev,:,:])
                    s = np.std(input_ds[:,var,lev,:,:])
                    input_ds_standard[:,var,lev,:,:] = (input_ds[:,var,lev,:,:]-m)/s
                    means[var, lev] = m
                    stds[var, lev] = s
        else:
            for var in range(5):
                for lev in range(5):
                    input_ds_standard[:,var,lev,:,:] = (input_ds[:,var,lev,:,:]-means[var, lev])/stds[var, lev]

    if not args.load_stats:
        with open(args.output_path + "means.pkl", 'wb') as f:
            pickle.dump(means, f)
        with open(args.output_path + "stds.pkl", 'wb') as f:
            pickle.dump(stds, f)
    
    input_ds_standard = torch.tensor(input_ds_standard)

    # write the input datasets to files
    with open(args.output_path + args.log_file, 'a') as f:
        f.write(f'\nStarting to write the output file.')
    
    #with open(args.output_path + args.output_file, 'wb') as f:
    #    pickle.dump(input_ds, f)
    
    with open(args.output_path + args.output_file, 'wb') as f:
        pickle.dump(input_ds_standard, f)
    
    with open(args.output_path + args.log_file, 'a') as f:
        f.write(f'\nOutput file written.\nPreprocessing finished.')

