import wandb
import numpy as np
import os
import sys
import time
import argparse

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
parser.add_argument('--input_file', type=str, default="input_standard.pkl")
#parser.add_argument('--data_file', type=str, default=None)
parser.add_argument('--target_file', type=str, default=None)
#parser.add_argument('--mask_file', type=str, default=None)
parser.add_argument('--idx_file', type=str)
parser.add_argument('--checkpoint_file', type=str)
#parser.add_argument('--weights_file', type=str, default=None)
parser.add_argument('--graph_file', type=str, default=None)
parser.add_argument('--mask_target_file', type=str, default=None)
parser.add_argument('--subgraphs_file', type=str, default="subgraphs_local.pkl")

#-- output files
parser.add_argument('--log_file', type=str, default='log.txt', help='log file')
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
parser.add_argument('--load_checkpoint',  action='store_true')
parser.add_argument('--no-load_checkpoint', dest='load_ae_checkpoint', action='store_false')

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
parser.add_argument('--model_type', type=str)
parser.add_argument('--performance', type=str, default=None)
parser.add_argument('--mode', type=str, default='train', help='train / get_encoding / test')

if __name__ == '__main__':

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    if args.use_accelerate is True:
        accelerator = Accelerator(log_with="wandb")
    else:
        accelerator = None
    
    # wand
    if args.mode == 'train':
        os.environ['WANDB_API_KEY'] = 'b3abf8b44e8d01ae09185d7f9adb518fc44730dd'
        os.environ['WANDB_USERNAME'] = 'valebl'
        os.environ['WANDB_MODE'] = 'offline'

    if accelerator.init_trackers(
            project_name=args.wandb_project_name
            )        

    if args.model_type == 'cl' or args.model_type == 'reg':
        dataset_type = 'gnn'
        collate_type = 'gnn'
    elif args.model_type == 'ae':
        dataset_type = 'ae'
        collate_type = 'ae'
    elif args.model_type == 'e':
        dataset_type = 'e'
        collate_type = 'e'
    elif args.model_type == 'reg-ft-gnn':
        dataset_type = 'ft_gnn'
        collate_type = 'gnn'
    
    Model = getattr(models, args.model_name)
    Dataset = getattr(dataset, 'Dataset_'+dataset_type)
    custom_collate_fn = getattr(dataset, 'custom_collate_fn_'+collate_type)
    
    model = Model()
    epoch_start = 0
    
    
    if args.loss_fn == 'sigmoid_focal_loss':
        loss_fn = getattr(torchvision.ops.focal_loss, args.loss_fn)
    elif args.loss_fn == 'weighted_cross_entropy_loss':
        if accelerator is None:
            loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.1,1]).cuda())
        else:
            loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.1,1]).to(accelerator.device))
    else:
        loss_fn = getattr(nn.functional, args.loss_fn)
    
    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path+args.log_file, 'w') as f:
            f.write(f"Cuda is available: {torch.cuda.is_available()}.\nStarting with pct_trainset={args.pct_trainset}, lr={args.lr}, "+
                f"weight decay = {args.weight_decay} and epochs={args.epochs}."+
                f"\nThere are {torch.cuda.device_count()} available GPUs.")
            if accelerator is None:
                f.write(f"\nModel = {args.model_name}, batch size = {args.batch_size}")
            else:
                f.write(f"\nModel = {args.model_name}, batch size = {args.batch_size*torch.cuda.device_count()}")


    #-- create the dataset
    dataset = Dataset(args)

    #-- split into trainset and testset
    generator=torch.Generator().manual_seed(42)
    len_trainset = int(len(dataset) * args.pct_trainset)
    len_testset = len(dataset) - len_trainset
    trainset, testset = torch.utils.data.random_split(dataset, lengths=(len_trainset, len_testset), generator=generator)
    
    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path+args.log_file, 'a') as f:
            f.write(f'\nTrainset size = {len_trainset}.')

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate_fn)

    if accelerator is None or accelerator.is_main_process:
        total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
        with open(args.output_path+args.log_file, 'a') as f:
            f.write(f"\nRAM memory {round((used_memory/total_memory) * 100, 2)} %")
    
    model = Model()
    net_names = ["encoder.", "gru.", "linear."]

    #-- either load the model checkpoint or load the parameters for the encoder
    if args.load_ae_checkpoint is True and args.ctd_training is False:
        model = load_encoder_checkpoint(model, args.checkpoint_ae_file, args.output_path, args.log_file, accelerator,
                fine_tuning=args.fine_tuning, net_names=net_names)
    elif args.load_ae_checkpoint is True and args.ctd_training is True:
        raise RuntimeError("Either load the ae parameters or continue the training.")

    #-- define the optimizer and trainable parameters
    if not args.fine_tuning:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.5)

    if accelerator is not None:
        model, optimizer, trainloader = accelerator.prepare(model, optimizer, trainloader)
        #model, optimizer, trainloader, testloader, validationloader = accelerator.prepare(model, optimizer, trainloader, testloader, validationloader)
    else:
        model = model.cuda()
    
    epoch_start = 0

    if args.ctd_training:
        if accelerator is None or accelerator.is_main_process:
            with open(args.output_path+args.log_file, 'a') as f:
                f.write("\nLoading the checkpoint to continue the training.")
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
    if accelerator is None or accelerator.is_main_process: 
        with open(args.output_path+args.log_file, 'a') as f:
            f.write(f"\nTotal number of trainable parameters: {total_params}.")

    check_freezed_layers(model, args.output_path, args.log_file, accelerator)

    start = time.time()

    trainer = Trainer()
    trainer.train(model, trainloader, optimizer, loss_fn, lr_scheduler, accelerator, args)

    #total_loss, loss_list = train_model(model=model, dataloader=trainloader, loss_fn=loss_fn, optimizer=optimizer,
    #    num_epochs=args.epochs, log_path=args.output_path, log_file=args.out_log_file, train_epoch=train_epoch,
    #    accelerator=accelerator, lr_scheduler=scheduler, checkpoint_name=args.output_path+args.out_checkpoint_file,
    #    performance=args.performance, epoch_start=epoch_start)

    end = time.time()

    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path+args.log_file, 'a') as f:
            f.write(f"\nTraining completed in {end - start} seconds.")
            f.write(f"\nDONE!")

    wandb.finish()
