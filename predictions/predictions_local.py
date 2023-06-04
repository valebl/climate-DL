import numpy as np
import pickle
import torch
import argparse
import time
import os
import sys
sys.path.append("/m100_work/ICT23_ESP_C/vblasone/climate-DL/local_single")
#sys.path.append("/home/vblasone/climate-DL/local_multiple")

import models
import dataset
from utils import load_encoder_checkpoint as load_checkpoint, Tester

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#-- paths
parser.add_argument('--input_path', type=str, help='path to input directory')
parser.add_argument('--output_path', type=str, help='path to output directory')

#-- input files
parser.add_argument('--input_file', type=str, default="input_standard.pkl")
parser.add_argument('--idx_file', type=str, default="idx_test.pkl")
parser.add_argument('--idx_time_test', type=str, default="idx_time_test.pkl")
parser.add_argument('--graph_file_test', type=str) 
parser.add_argument('--subgraphs', type=str) 
parser.add_argument('--checkpoint_cl', type=str)
parser.add_argument('--checkpoint_reg', type=str)

#-- output files
parser.add_argument('--log_file', type=str, default='log.txt', help='log file')

#-- boolean
parser.add_argument('--use_accelerate',  action='store_true')
parser.add_argument('--no-use_accelerate', dest='use_accelerate', action='store_false')

#-- other
parser.add_argument('--test_year', type=int, default=2016)
parser.add_argument('--batch_size', type=int, default=128, help='batch size (global)')
parser.add_argument('--lon_dim', type=int, default=7)
parser.add_argument('--lat_dim', type=int, default=7)
parser.add_argument('--model_name_cl', type=str, default='Classifier_old_test')
parser.add_argument('--model_name_reg', type=str, default='Regressor_old_test')
parser.add_argument('--idx_min', type=int, default=130728)

#from torchmetrics.classification import BinaryConfusionMatrix


if __name__ == '__main__':

    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    with open(args.output_path + args.log_file, 'w') as f:
        f.write("Starting.")

    #LAT_DIM = 7 # number of points in the GRIPHO rectangle (0.25 grid)
    #LON_DIM = 7
    TIME_DIM = 140256
    spatial_points_dim = args.lat_dim * args.lon_dim

    #with open(args.input_path + args.idx_file, 'rb') as f:
    #    test_keys = pickle.load(f)
    
    #with open(args.input_path + args.idx_time_test, 'rb') as f:
    #    idx_time_test = pickle.load(f)
    
    with open(args.input_path + args.graph_file_test, 'rb') as f:
        G_test = pickle.load(f)

#-----------------------------------------------------
#----------------- DATASET AND MODELS ----------------
#-----------------------------------------------------

    Dataset = getattr(dataset, 'Dataset_pr_test')
    custom_collate_fn = getattr(dataset, 'custom_collate_fn_gnn')
    
    with open(args.output_path + args.log_file, 'a') as f:
        f.write("\nBuilding the dataset and the dataloader.")

    dataset = Dataset(args=args, lon_dim=args.lon_dim, lat_dim=args.lat_dim, time_min=args.idx_min, time_max=140255) #time_min=113951, time_max=140255)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)

    with open(args.output_path + args.log_file, 'a') as f:
        f.write("\nDone!")
        f.write("\nInstantiate models and load checkpoints.\n")

    Model_cl = getattr(models, args.model_name_cl)
    Model_reg = getattr(models, args.model_name_reg)

    model_cl = Model_cl()
    model_reg = Model_reg()

    with open(args.output_path + args.log_file, 'a') as f:
        f.write("\nClassifier:")

    checkpoint_cl = load_checkpoint(model_cl, args.checkpoint_cl, args.output_path, args.log_file, None, net_names=["encoder.", "gru.", "dense.", "gnn."], fine_tuning=False)

    with open(args.output_path + args.log_file, 'a') as f:
        f.write("\nRegressor:")

    checkpoint_reg = load_checkpoint(model_reg, args.checkpoint_reg, args.output_path, args.log_file, None, net_names=["encoder.", "gru.", "dense.", "gnn."], fine_tuning=False)
    
    model_cl = model_cl.cuda()
    model_reg = model_reg.cuda()

    with open(args.output_path + args.log_file, 'a') as f:
        f.write("\n\nDone!")

#-----------------------------------------------------
#-------------------- PREDICTIONS --------------------
#-----------------------------------------------------

    with open(args.output_path + args.log_file, 'a') as f:
        f.write("\nStarting the test.")

    tester = Tester()
    
    start = time.time()
    tester.test(model_cl, model_reg, dataloader, G_test=G_test, args=args)
    end = time.time()

    with open(args.output_path + args.log_file, 'a') as f:
        f.write(f"\nDone. Testing concluded in {end-start} seconds.")
        f.write("\nWrite the files.")

    with open(args.output_path + "G_predictions_2016.pkl", 'wb') as f:
        pickle.dump(G_test, f)

    with open(args.output_path + args.log_file, 'a') as f:
        f.write(f"\nDone.")


