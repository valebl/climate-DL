import numpy as np
import os
import sys
import time
import argparse

import torch
from torch import nn

import models
import utils
import dataset
import pickle

from utils import load_checkpoint, check_freezed_layers, write_log
from utils import train_model
from dataset import Clima_dataset as Dataset

from accelerate import Accelerator

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#-- paths
parser.add_argument('--input_path', type=str, help='path to input directory')
parser.add_argument('--output_path', type=str, help='path to output directory')

#-- input files
parser.add_argument('--input_file', type=str, default="input_standard.pkl")
parser.add_argument('--data_file', type=str, default=None)
parser.add_argument('--target_file', type=str, default=None)
parser.add_argument('--mask_file', type=str, default=None)
parser.add_argument('--idx_file', type=str)
parser.add_argument('--checkpoint_ae_file', type=str)
parser.add_argument('--weights_file', type=str, default=None)

#-- output files
parser.add_argument('--out_log_file', type=str, default='log.txt', help='log file')
parser.add_argument('--out_checkpoint_file', type=str, default="checkpoint.pth")
parser.add_argument('--out_loss_file', type=str, default="loss.csv")

#-- training hyperparameters
parser.add_argument('--pct_trainset', type=float, default=0.8, help='percentage of dataset in trainset')
parser.add_argument('--epochs', type=int, default=15, help='number of total training epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size (global)')
parser.add_argument('--step_size', type=int, default=10, help='scheduler step size (global)')
parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay (wd)')
parser.add_argument('--fine_tuning',  action='store_true')
parser.add_argument('--no-fine_tuning', dest='fine_tuning', action='store_false')
parser.add_argument('--load_ae_checkpoint',  action='store_true')
parser.add_argument('--no-load_ae_checkpoint', dest='load_ae_checkpoint', action='store_false')

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
parser.add_argument('--loss_fn', type=str, default="mse_loss")
parser.add_argument('--net_type', type=str)
parser.add_argument('--performance', type=str, default=None)

if __name__ == '__main__':


    device = "cpu"
    torch.backends.cudnn.benchmark = True

    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    if args.use_accelerate is True:
        accelerator = Accelerator()
    else:
        accelerator = None

    Model = getattr(models, args.model_name)
    train_epoch = getattr(utils, "train_epoch_"+args.net_type)
    test_model = getattr(utils, "test_model_"+args.net_type)
    custom_collate_fn = getattr(dataset, "custom_collate_fn_"+args.net_type)
    validate_model = getattr(utils, "validate_"+args.net_type)

    if args.loss_fn == 'weighted_mse_loss' or args.loss_fn == 'mse_loss_mod':
        loss_fn = getattr(utils, args.loss_fn)
    elif args.loss_fn == 'weighted_cross_entropy_loss':
        if accelerator is None:
            loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.1,1]).cuda())
        else:
            loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.1,1]).to(accelerator.device))
    else:
        loss_fn = getattr(nn.functional, args.loss_fn)
    
    if args.use_accelerate is True:
        accelerator = Accelerator()
    else:
        accelerator = None

    write_log(f"Cuda is available: {torch.cuda.is_available()}.\nStarting with pct_trainset={args.pct_trainset}, lr={args.lr}, "+
                f"weight decay = {args.weight_decay} and epochs={args.epochs}."+
                f"\nThere are {torch.cuda.device_count()} available GPUs.", accelerator, args, mode='w')
    
    if accelerator is None:
        write_log(f"\nModel = {args.model_name}, batch size = {args.batch_size}", accelerator, args)
    else:
        write_log(f"\nModel = {args.model_name}, batch size = {args.batch_size*torch.cuda.device_count()}", accelerator, args)


    with open("/data/large/era5_to_gripho_list.pkl", 'rb') as f:
        era5_to_gripho_list = pickle.load(f)

    #-- create the dataset
    dataset = Dataset(args)

    #-- split into trainset and testset
    generator=torch.Generator().manual_seed(42)
    len_trainset = int(len(dataset) * args.pct_trainset)
    len_testset = len(dataset) - len_trainset
    trainset, testset = torch.utils.data.random_split(dataset, lengths=(len_trainset, len_testset), generator=generator)
    
    # split testset into validationset and testset
    #len_testset = int(len(testset) * 0.999)
    #len_validationset = len(testset) - len_testset
    #testset, validationset = torch.utils.data.random_split(testset, lengths=(len_testset, len_validationset), generator=generator)

    write_log(f"\nTrainset size = {len_trainset}", accelerator, args) #, testset size = {len_testset}, validationset size = {len_validationset}.', accelerator, args)

    #-- construct the dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate_fn)
    #testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
    #validationloader = torch.utils.data.DataLoader(validationset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)

    if accelerator is None or accelerator.is_main_process:
        total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
    
    write_log(f"\nRAM memory {round((used_memory/total_memory) * 100, 2)} %", accelerator, args)
    
    model = Model()
    net_names = ["encoder", "gru", "dense"]

    #-- either load the model checkpoint or load the parameters for the encoder
    if args.load_ae_checkpoint is True and args.ctd_training is False:
        model = load_checkpoint(model, args.checkpoint_ae_file, accelerator, net_names, args, fine_tuning=args.fine_tuning)
    elif args.load_ae_checkpoint is True and args.ctd_training is True:
        raise RuntimeError("Either load the ae parameters or continue the training.")

    #-- define the optimizer and trainable parameters
    if not args.fine_tuning:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.5)

    if accelerator is not None:
        model, optimizer, trainloader, testloader, validationloader = accelerator.prepare(model, optimizer, trainloader, testloader, validationloader)
    else:
        model = model.to(device)
    
    epoch_start = 0

    if args.ctd_training:
        write_log("\nLoading the checkpoint to continue the training.", accelerator, args)
        checkpoint = torch.load(args.checkpoint_ctd)
        try:
            model.load_state_dict(checkpoint["parameters"])
        except:
            for name, param in checkpoint["parameters"].items():
                param = param.data
                if name.startswith("module."):
                    name = name.partition("module.")[2]
                model.state_dict()[name].copy_(param)
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch_start = checkpoint["epoch"] + 1

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    write_log(f"\nTotal number of trainable parameters: {total_params}.", accelerator, args)

    check_freezed_layers(model, accelerator, args)

    start = time.time()

    total_loss, loss_list = train_model(model=model, dataloader=trainloader, loss_fn=loss_fn, optimizer=optimizer,
        train_epoch=train_epoch, validate_model=validate_model, validationloader=None, accelerator=accelerator, 
        lr_scheduler=scheduler, checkpoint_name=args.output_path+args.out_checkpoint_file, epoch_start=epoch_start, args=args,
        era5_to_gripho_list=era5_to_gripho_list, device=device)

    end = time.time()

    write_log(f"\nTraining _cnn_gnn completed in {end - start} seconds.", accelerator, args)

    #-- test the model
    if args.test_model:
        test_loss_total, test_loss_avg = test_model(model, testloader, args.output_path, args.out_log_file, accelerator,
                loss_fn=nn.functional.mse_loss, performance=args.performance) 

    write_log(f"\nDONE!", accelerator, args)

