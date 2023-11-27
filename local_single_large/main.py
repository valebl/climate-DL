import numpy as np
import os
import sys
import time
import argparse
import pickle

import torch
from torch import nn
import torchvision.ops.focal_loss

import models
import utils
import dataset

from utils import load_encoder_checkpoint, check_freezed_layers
from utils import Trainer

from accelerate import Accelerator

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#-- paths
parser.add_argument('--input_path', type=str, help='path to input directory')
parser.add_argument('--output_path', type=str, help='path to output directory')

#-- input files
parser.add_argument('--input_file', type=str, default=None)
parser.add_argument('--target_file', type=str, default=None)
parser.add_argument('--idx_file', type=str)
parser.add_argument('--checkpoint_file', type=str, default=None)
parser.add_argument('--graph_file', type=str, default=None) 
parser.add_argument('--mask_target_file', type=str, default=None)
parser.add_argument('--subgraphs_file', type=str, default=None)
parser.add_argument('--weights_file', type=str, default=None)
parser.add_argument('--encodings_file', type=str, default=None) 

#-- output files
parser.add_argument('--log_file', type=str, default='log.txt', help='log file')
parser.add_argument('--out_checkpoint_file', type=str, default="checkpoint.pth")
parser.add_argument('--out_loss_file', type=str, default="loss.csv")

#-- training hyperparameters
parser.add_argument('--pct_trainset', type=float, default=1.0, help='percentage of dataset in trainset')
parser.add_argument('--epochs', type=int, default=15, help='number of total training epochs')
parser.add_argument('--batch_size', type=int, default=64, help='batch size (global)')
parser.add_argument('--step_size', type=int, default=10, help='scheduler step size (global)')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay (wd)')
parser.add_argument('--fine_tuning',  action='store_true')
parser.add_argument('--no-fine_tuning', dest='fine_tuning', action='store_false')
parser.add_argument('--load_checkpoint',  action='store_true')
parser.add_argument('--no-load_checkpoint', dest='load_checkpoint', action='store_false')

#-- boolean
parser.add_argument('--checkpoint_ctd', type=str, help='checkpoint to load to continue')
parser.add_argument('--ctd_training',  action='store_true')
parser.add_argument('--no-ctd_training', dest='ctd_training', action='store_false')
parser.add_argument('--use_accelerate',  action='store_true')
parser.add_argument('--no-use_accelerate', dest='use_accelerate', action='store_false')
parser.add_argument('--test_model',  action='store_true')
parser.add_argument('--no-test_model', dest='test_model', action='store_false')

#-- other
parser.add_argument('--model_name', type=str)
parser.add_argument('--dataset_name', type=str)
parser.add_argument('--collate_name', type=str)
parser.add_argument('--loss_fn', type=str, default="mse_loss")
parser.add_argument('--model_type', type=str)
parser.add_argument('--performance', type=str, default=None)
parser.add_argument('--wandb_project_name', type=str)
parser.add_argument('--mode', type=str, default='train', help='train / get_encoding / test')
parser.add_argument('--lon_min', type=float)
parser.add_argument('--lon_max', type=float)
parser.add_argument('--lat_min', type=float)
parser.add_argument('--lat_max', type=float)
parser.add_argument('--interval', type=float, default=0.25)
parser.add_argument('--lon_dim', type=int, default=None)
parser.add_argument('--lat_dim', type=int, default=None)


if __name__ == '__main__':

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    #if args.model_type == 'cl' or args.model_type == 'reg':
    #    #dataset_type = 'gnn_large'
    #    collate_type = 'gnn_large'
    #elif args.model_type == 'ae':
    #    dataset_type = 'ae'
    #    collate_type = 'ae'

#-----------------------------------------------------
#--------------- WANDB and ACCELERATE ----------------
#-----------------------------------------------------

    if args.use_accelerate is True:
        accelerator = Accelerator(log_with="wandb")
    else:
        accelerator = None
    
    if args.mode == 'train':
        os.environ['WANDB_API_KEY'] = 'b3abf8b44e8d01ae09185d7f9adb518fc44730dd'
        os.environ['WANDB_USERNAME'] = 'valebl'
        os.environ['WANDB_MODE'] = 'offline'
        os.environ['WANDB_CONFIG_DIR']='./wandb/'

        accelerator.init_trackers(
            project_name=args.wandb_project_name
            )

    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path+args.log_file, 'w') as f:
            f.write("Starting the training...")
            f.write(f"Cuda is available: {torch.cuda.is_available()}. There are {torch.cuda.device_count()} available GPUs.")

    ## derive arrays corresponding to the lon/lat low resolution grid points
    lon_low_res_array = np.arange(args.lon_min-args.interval, args.lon_max+args.interval, args.interval)
    lat_low_res_array = np.arange(args.lat_min-args.interval, args.lat_max+args.interval, args.interval)
    args.lon_dim = lon_low_res_array.shape[0]
    args.lat_dim = lat_low_res_array.shape[0]
    #space_low_res_dim = lon_low_res_dim * lat_low_res_dim

#-----------------------------------------------------
#--------------- MODEL, LOSS, OPTIMIZER --------------
#-----------------------------------------------------

    Model = getattr(models, args.model_name)
    model = Model()

    if args.mode == 'train':

        if args.loss_fn == 'sigmoid_focal_loss':
            loss_fn = getattr(torchvision.ops.focal_loss, args.loss_fn)
        elif args.loss_fn == 'weighted_cross_entropy_loss':
            loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.1,1]))
        elif args.loss_fn == 'weighted_mse_loss':
            loss_fn = getattr(utils, 'weighted_mse_loss')
        elif args.loss_fn == 'quantile_loss':
            loss_fn = getattr(utils, 'quantile_loss')
        else:
            loss_fn = getattr(nn.functional, args.loss_fn)    
        
        if accelerator is None or accelerator.is_main_process:
            with open(args.output_path+args.log_file, 'a') as f:
                f.write(f"\nStarting with pct_trainset={args.pct_trainset}, lr={args.lr}, "+
                    f"weight decay = {args.weight_decay} and epochs={args.epochs}.")
                if accelerator is None:
                    f.write(f"\nModel = {args.model_name}, batch size = {args.batch_size}")
                else:
                    f.write(f"\nModel = {args.model_name}, batch size = {args.batch_size*torch.cuda.device_count()}")

        #-- define the optimizer and trainable parameters
        if not args.fine_tuning:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                
#-----------------------------------------------------
#-------------- DATASET AND DATALOADER ---------------
#----------------------------------------------------

    Dataset = getattr(dataset, args.dataset_name)
    
    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path+args.log_file, 'a') as f:
            f.write(f"\nLoading graph...")
    
    with open(args.input_path+args.graph_file, 'rb') as f:
        graph = pickle.load(f)
    
    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path+args.log_file, 'a') as f:
            f.write(f" Done!")
    
    encodings = None
    
    with open(args.input_path+"input_standard.pkl", 'rb') as f:
        input_data = pickle.load(f)

#    if accelerator is None or accelerator.is_main_process:
#        with open(args.output_path+args.log_file, 'a') as f:
#            f.write(f"\nLoading encodings...")
#    
#    with open(args.input_path+args.encodings_file, 'rb') as f:
#        encodings = pickle.load(f)
#
#    if accelerator is None or accelerator.is_main_process:
#        with open(args.output_path+args.log_file, 'a') as f:
#            f.write(f" Done!")

    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path+args.log_file, 'a') as f:
            f.write(f"\nLoading target..")
    
    with open(args.input_path+args.target_file, 'rb') as f:
        target = pickle.load(f)
    
    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path+args.log_file, 'a') as f:
            f.write(f" Done!")
    
    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path+args.log_file, 'a') as f:
            f.write(f"\nLoading mask..")

    with open(args.input_path+args.mask_target_file, 'rb') as f:
        mask_target = pickle.load(f)

    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path+args.log_file, 'a') as f:
            f.write(f" Done!")
    
    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path+args.log_file, 'a') as f:
            f.write(f"\nCreating the dataloader!")
    
    if args.model_type == "reg":
        with open(args.input_path+args.weights_file, 'rb') as f:
            weights_reg_train = pickle.load(f)
        print(torch.swapaxes(weights_reg_train,0,1).numpy().shape)
        dataset_graph = Dataset(edge_index=graph.edge_index,
            features=None, #x[:130728,:,:].numpy(),
            targets=torch.swapaxes(target,0,1).numpy(),
            edge_weight=None,
            z=graph.x,
            low_res=graph.low_res,
            encodings=encodings,
            train_mask=torch.swapaxes(mask_target,0,1).numpy(),
            w=torch.swapaxes(weights_reg_train,0,1).numpy(),
            input_data=input_data,
            lon_low_res_dim=args.lon_dim)
    elif args.model_type=="cl":
        dataset_graph = Dataset(edge_index=graph.edge_index,
            features=None, #x[:130728,:,:].numpy(),
            targets=torch.swapaxes(target,0,1).numpy(),
            edge_weight=None,
            z=graph.x,
            low_res=graph.low_res,
            encodings=encodings,
            train_mask=torch.swapaxes(mask_target,0,1).numpy(),
            input_data=input_data,
            lon_low_res_dim=args.lon_dim)

    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path+args.log_file, 'a') as f:
            f.write(f"\nDone!")

    # Create the dataloader

    custom_collate_fn = getattr(dataset, args.collate_name)
    
    sampler_graph = dataset.Iterable_StaticGraphTemporalSignal(static_graph_temporal_signal=dataset_graph, shuffle=True)
    
    dataloader = torch.utils.data.DataLoader(dataset_graph, batch_size=args.batch_size, num_workers=0,
                                             sampler=sampler_graph, collate_fn=custom_collate_fn)

    if accelerator is None or accelerator.is_main_process:
        total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
        with open(args.output_path+args.log_file, 'a') as f:
            f.write(f"\nRAM memory {round((used_memory/total_memory) * 100, 2)} %")
    
#-----------------------------------------------------
#------------------ LOAD PARAMETERS ------------------
#-----------------------------------------------------
    
    epoch_start = 0
    net_names = ["encoder.", "gru.", "gnn.", "dense."]
    #elif args.mode == 'get_encoding':
    #    net_names = ["encoder.", "gru."]
    
    #-- either load the model checkpoint or load the parameters for the encoder
    if args.load_checkpoint is True and args.ctd_training is False:
        model = load_encoder_checkpoint(model, args.checkpoint_file, args.output_path, args.log_file, accelerator=accelerator,
                fine_tuning=args.fine_tuning, net_names=net_names)
    elif args.load_checkpoint is True and args.ctd_training is True:
        if accelerator is None or accelerator.is_main_process:
            with open(args.output_path+args.log_file, 'a') as f:
                f.write("\nLoading the checkpoint to continue the training.")
        checkpoint = torch.load(args.checkpoint_file)
        #try:
        #    model.load_state_dict(checkpoint["parameters"])
        #except:
        state_dict = checkpoint["parameters"]
        for name, param in state_dict.items():
            for net_name in net_names:
                if net_name in name and "edge.weight" not in name:
                    if accelerator is None or accelerator.is_main_process:
                        with open(args.output_path+args.log_file, 'a') as f:
                            f.write(f"\nLoading parameters '{name}'")
                    param = param.data
                    if name.startswith("module"):
                        name = name.partition("module.")[2]
                        try:
                            model.state_dict()[name].copy_(param)
                        except:
                            if accelerator is None or accelerator.is_main_process:
                                with open(args.output_path+args.log_file, 'a') as f:
                                    f.write(f"\nParam {name} was not loaded..")
        #optimizer.load_state_dict(checkpoint["optimizer"])
        epoch_start = checkpoint["epoch"] + 1

    check_freezed_layers(model, args.output_path, args.log_file, accelerator)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if accelerator is None or accelerator.is_main_process: 
        with open(args.output_path+args.log_file, 'a') as f:
            f.write(f"\nTotal number of trainable parameters: {total_params}.")

    if accelerator is not None:
        if args.mode == 'train':
            model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    else:
        model = model.cuda()
    
#-----------------------------------------------------
#----------------------- TRAIN -----------------------
#-----------------------------------------------------


    start = time.time()

    if args.mode == 'train':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.5)
        trainer = Trainer()
        trainer.train(model, dataloader, optimizer, loss_fn, lr_scheduler, accelerator, args, epoch_start=epoch_start)
        
    end = time.time()

    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path+args.log_file, 'a') as f:
            f.write(f"\nCompleted in {end - start} seconds.")
            f.write(f"\nDONE!")
    
    if args.mode == 'train':
        accelerator.end_training()

