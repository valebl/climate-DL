import numpy as np
import pickle
import torch
import argparse
import time

import models, dataset
from utils import load_encoder_checkpoint as load_checkpoint, Tester
# from torch_geometric.data import Data, Batch

import sys

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#-- paths
parser.add_argument('--input_path', type=str, help='path to input directory', default="/m100_work/ICT23_ESP_C/vblasone/DATA/graph/")
parser.add_argument('--output_path', type=str, help='path to output directory', default="/m100_work/ICT23_ESP_C/vblasone/climate-DL/predictions/")

#-- input files
parser.add_argument('--input_file', type=str, default="input_standard.pkl")
parser.add_argument('--idx_file', type=str, default="idx_test.pkl")
parser.add_argument('--graph_file', type=str, default="G_north_italy_test.pkl") 
parser.add_argument('--mask_1_cell_file', type=str, default="mask_1_cell_subgraphs.pkl")
parser.add_argument('--mask_9_cells_file', type=str, default="mask_9_cells_subgraphs.pkl") 

#-- output files
parser.add_argument('--log_file', type=str, default='log.txt', help='log file')

#-- boolean
parser.add_argument('--use_accelerate',  action='store_true')
parser.add_argument('--no-use_accelerate', dest='use_accelerate', action='store_false')

#-- other
parser.add_argument('--test_year', type=int, default=2016)
parser.add_argument('--batch_size', type=int, default=128, help='batch size (global)')

#from torchmetrics.classification import BinaryConfusionMatrix


if __name__ == '__main__':

    args = parser.parse_args()

    log_file = f"log_{args.test_year}.txt"

    with open(args.output_path + args.log_file, 'w') as f:
        f.write("Starting.")

    LAT_DIM = 16 # number of points in the GRIPHO rectangle (0.25 grid)
    LON_DIM = 31
    SPACE_IDXS_DIM = LAT_DIM * LON_DIM
    TIME_DIM = 140256
    SPATIAL_POINTS_DIM = LAT_DIM * LON_DIM

    checkpoint_cl = "/m100_work/ICT23_ESP_C/vblasone/climate-DL/local_multiple/cl-230324-12/checkpoint_tmp_0.pth"
    checkpoint_reg = "/m100_work/ICT23_ESP_C/vblasone/climate-DL/local_multiple/reg-230324-12/checkpoint_1.pth"
    model_name_cl = "Classifier_test"
    model_name_reg = "Regressor_test"

    with open(args.input_path + args.idx_file, 'rb') as f:
        test_keys = pickle.load(f)
    
    with open(args.input_path + args.graph_file, 'rb') as f:
        G_test = pickle.load(f)

#-----------------------------------------------------
#----------------- DATASET AND MODELS ----------------
#-----------------------------------------------------

    Dataset = getattr(dataset, 'Dataset_gnn_test')
    custom_collate_fn = getattr(dataset, 'custom_collate_fn_gnn')
    
    with open(args.output_path + args.log_file, 'a') as f:
        f.write("\nBuilding the dataset and the dataloader.")

    dataset = Dataset(args)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)

    with open(args.output_path + args.log_file, 'a') as f:
        f.write("\nDone!")
        f.write("\nInstantiate models and load checkpoints.")

    Model_cl = getattr(models, model_name_cl)
    Model_reg = getattr(models, model_name_reg)

    model_cl = Model_cl()
    model_reg = Model_reg()

    checkpoint_cl = load_checkpoint(model_cl, checkpoint_cl, args.output_path, args.log_file, None, net_names=["encoder.", "gru.", "gnn."], fine_tuning=False)
    checkpoint_reg = load_checkpoint(model_reg, checkpoint_reg, args.output_path, args.log_file, None, net_names=["encoder.", "gru.", "gnn."], fine_tuning=False)
    
    model_cl = model_cl.cuda()
    model_reg = model_reg.cuda()

    with open(args.output_path + args.log_file, 'a') as f:
        f.write("\nDone!")

#-----------------------------------------------------
#-------------------- PREDICTIONS --------------------
#-----------------------------------------------------

    with open(args.output_path + args.log_file, 'a') as f:
        f.write("\nStarting the test.")

    tester = Tester()
    
    start = time.time()
    y_pred_cl, y_pred_reg, y_pred = tester.test(model_cl, model_reg, dataloader, G_test.pr.shape)
    end = time.time()

    G_test["y_pred_cl"] = y_pred_cl
    G_test["y_pred_reg"] = y_pred_reg
    G_test["y_pred"] = y_pred

    with open(args.output_path + args.log_file, 'a') as f:
        f.write(f"\nDone. Testing concluded in {end-start} seconds.")
        f.write("\nWrite the files.")

    with open(args.output_path + "G_predictions.pkl", 'wb'):
        pickle.dump(G_test, f)

    with open(args.output_path + args.log_file, 'a') as f:
        f.write(f"\nDone.")


