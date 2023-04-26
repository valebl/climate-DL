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
parser.add_argument('--input_files_suffix', type=str, help='suffix for the input files (convenction: {parameter}{suffix}.nc)', default='_sliced')
parser.add_argument('--log_file', type=str, help='log file name', default='log.txt')
parser.add_argument('--output_file', type=str, help='path to output directory', default='input_ds_standard.pkl')
parser.add_argument('--n_levels', type=int, help='number of pressure levels considered', default=5)
parser.add_argument('--statistics_path', type=str, default='/m100_work/ICT23_ESP_C/vblasone/NORTH_ITALY/north_italy/')
parser.add_argument('--means_file', type=str, default='means.pkl')
parser.add_argument('--stds_file', type=str, default='stds.pkl')


if __name__ == '__main__':

    load_statistics = False

    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    params = ['q', 't', 'u', 'v', 'z']
    n_params = len(params)
 
    ## create the input dataset from files
    
    with open(args.output_path + args.log_file, 'w') as f:
        f.write(f'\nStarting to create the input dataset from files.')

    for p_idx, p in enumerate(params):
        with open(args.output_path + args.log_file, 'a') as f:
            f.write(f'\nPreprocessing {p}{args.input_files_suffix}.nc ...')
        with xr.open_dataset(f'{args.input_path}/{p}{args.input_files_suffix}.nc') as f:
            data = f[p].values
            if p_idx == 0: # first parameter being processed -> get dimensions and initialize the input dataset
                lat_dim = len(f.latitude)
                lon_dim = len(f.longitude)
                time_dim = len(f.time)
                input_ds = np.zeros((time_dim, n_params, args.n_levels, lat_dim, lon_dim), dtype=np.float32) # variables, levels, time, lat, lon
        input_ds[:, p_idx,:,:,:] = data

    ## post-processing of the input dataset
    
    # flip the dataset
    input_ds = np.flip(input_ds, 3) # the origin in the input files is in the top left corner, while we use the bottom left corner

    # standardizing the dataset
    with open(args.output_path + args.log_file, 'a') as f:
        f.write(f'\nStandardizing the dataset.')
    
    input_ds_standard = np.zeros((input_ds.shape), dtype=np.float32)
    
    if not load_statistics:
        means = np.zeros((5))
        stds = np.zeros((5))
        for var in range(5):
            m = np.mean(input_ds[:,var,:,:,:])
            s = np.std(input_ds[:,var,:,:,:])
            input_ds_standard[:,var,:,:,:] = (input_ds[:,var,:,:,:]-m)/s
            means[var] = m
            stds[var] = s
        with open(args.output_path + "means.pkl", 'wb') as f:
            pickle.dump(means, f)
        with open(args.output_path + "stds.pkl", 'wb') as f:
            pickle.dump(stds, f)
    else:
        with open(args.statistics_path+args.means_file, 'rb') as f:
            means = pickle.load(f)
        with open(args.statistics_path+args.stds_file, 'rb') as f:
            stds = pickle.load(f)
        for var in range(5):
            input_ds_standard[:,var,:,:,:] = (input_ds[:,var,:,:,:]-means[var])/stds[var]    

    #if not load_statistics:
    #    means = np.zeros((5,5))
    #    stds = np.zeros((5,5))
    #
    #    for var in range(5):
    #        for lev in range(5):
    #            m = np.mean(input_ds[:,var,lev,:,:])
    #            s = np.std(input_ds[:,var,lev,:,:])
    #            input_ds_standard[:,var,lev,:,:] = (input_ds[:,var,lev,:,:]-m)/s
    #            means[var, lev] = m
    #            stds[var, lev] = s
    #    with open(args.output_path + "means.pkl", 'wb') as f:
    #        pickle.dump(means, f)
    #    with open(args.output_path + "stds.pkl", 'wb') as f:
    #        pickle.dump(stds, f)
    #else:
    #    with open(args.statistics_path+args.means_file, 'rb') as f:
    #        means = pickle.load(f)
    #    with open(args.statistics_path+args.stds_file, 'rb') as f:
    #        stds = pickle.load(f)
    #    for var in range(5):
    #        for lev in range(5):
    #            input_ds_standard[:,var,lev,:,:] = (input_ds[:,var,lev,:,:]-means[var, lev])/stds[var, lev]

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

