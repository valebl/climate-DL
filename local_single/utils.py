import wandb
import numpy as np
import time
import sys
import pickle

import torch

#------Some useful utilities------

class AverageMeter(object):
    '''
    a generic class to keep track of performance metrics during training or testing of models
    (from the Deep Learning tutorials of DSSC)
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def use_gpu_if_possible():
    return "cuda:0" if torch.cuda.is_available() else "cpu"

def accuracy_binary_one(prediction, target):
    prediction_class = torch.where(prediction > 0.5, 1.0, 0.0) 
    correct_items = (prediction_class == target)
    acc = correct_items.sum().item() / prediction.shape[0]  
    return acc

def accuracy_binary_two(prediction, target):
    prediction_class = torch.argmax(prediction, dim=-1).squeeze()
    correct_items = (prediction_class == target)
    acc = correct_items.sum().item() / prediction.shape[0]  
    return acc

def weighted_mse_loss(input_batch, target_batch, weights):
    return (weights * (input_batch - target_batch) ** 2).sum() / weights.sum()

def load_encoder_checkpoint(model, checkpoint, log_path, log_file, accelerator, fine_tuning=True, net_names=['encoder', 'gru', 'linear']):
    if accelerator is None or accelerator.is_main_process:
        with open(log_path+log_file, 'a') as f:
            f.write("\nLoading encoder parameters.") 
    checkpoint = torch.load(checkpoint)
    state_dict = checkpoint["parameters"]
    for name, param in state_dict.items():
        for net_name in net_names:
            if net_name in name:
                if accelerator is None or accelerator.is_main_process:
                    with open(log_path+log_file, 'a') as f:
                        f.write(f"\nLoading parameters '{name}'")
                param = param.data
                if name.startswith("module"):
                    name = name.partition("module.")[2]
                try:
                    model.state_dict()[name].copy_(param)
                except:
                     if accelerator is None or accelerator.is_main_process:
                        with open(log_path+log_file, 'a') as f:
                            f.write(f"\nParam {name} was not loaded..")
    if not fine_tuning:
        for net_name in net_names:
            [param.requires_grad_(False) for name, param in model.named_parameters() if net_name in name]
    return model


def check_freezed_layers(model, log_path, log_file, accelerator):
    for name, param in model.named_parameters():
        n_param = param.numel()
        if accelerator is None or accelerator.is_main_process:
            with open(log_path+log_file, 'a') as f:
                f.write(f"\nLayer {name} requires_grad = {param.requires_grad} and has {n_param} parameters") 


class Trainer(object):

    def _train_epoch_ae(self, epoch, model, dataloader, optimizer, loss_fn, accelerator, args):
        loss_meter = AverageMeter()     
        start = time.time()
        for X in dataloader:
            optimizer.zero_grad()
            X_pred = model(X)
            loss = loss_fn(X_pred, X)
            accelerator.backward(loss)
            optimizer.step()
            loss_meter.update(val=loss.item(), n=X.shape[0])
            wandb.log({'loss iteration': loss_meter.val, 'loss avg': loss_meter.avg})
        end = time.time()
        wandb.log({'epoch':epoch, 'loss epoch': loss_meter.avg})
        if accelerator.is_main_process:
            with open(args.output_path+args.log_file, 'a') as f:
                f.write(f"\nEpoch {epoch+1} completed in {end - start:.4f} seconds. Loss - total: {loss_meter.sum:.4f} - average: {loss_meter.avg:.10f}. ")

    def _train_epoch_cl(self, epoch, model, dataloader, optimizer, loss_fn, accelerator, args, alpha=0.95, gamma=2):
        loss_meter = AverageMeter()
        performance_meter = AverageMeter()
        start = time.time()
        for X, data in dataloader:
            optimizer.zero_grad()
            y_pred, y, _  = model(X, data, accelerator.device)
            loss = loss_fn(y_pred, y, alpha, gamma, reduction='mean')
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            optimizer.step()
            loss_meter.update(val=loss.item(), n=X.shape[0])    
            performance = accuracy_binary_one(y_pred, y)
            performance_meter.update(val=performance, n=X.shape[0])
            wandb.log({'loss iteration': loss_meter.val, 'accuracy iteration': performance_meter.val, 'loss avg': loss_meter.avg, 'accuracy avg': performance_meter.avg})
        end = time.time()
        wandb.log({'epoch': epoch, 'loss epoch': loss_meter.avg, 'accuracy epoch': performance_meter.avg})
        if accelerator.is_main_process:
            with open(args.output_path+args.log_file, 'a') as f:
                f.write(f"\nEpoch {epoch+1} completed in {end - start:.4f} seconds. Loss - total: {loss_meter.sum:.4f} - average: {loss_meter.avg:.10f}; "+
                    f"performance: {performance_meter.avg:.4f}.")

    def _train_epoch_reg(self, epoch, model, dataloader, optimizer, loss_fn, accelerator, args):
        loss_meter = AverageMeter()
        start = time.time()
        for X, data in dataloader:
            optimizer.zero_grad()
            y_pred, y, _  = model(X, data, accelerator.device)
            loss = loss_fn(y_pred, y)
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            optimizer.step()
            loss_meter.update(val=loss.item(), n=X.shape[0])    
            wandb.log({'loss iteration': loss_meter.val, 'loss avg': loss_meter.avg})
        end = time.time()
        wandb.log({'epoch': epoch, 'loss epoch': loss_meter.avg})
        if accelerator.is_main_process:
            with open(args.output_path+args.log_file, 'a') as f:
                f.write(f"\nEpoch {epoch+1} completed in {end - start:.4f} seconds. Loss - total: {loss_meter.sum:.4f} - average: {loss_meter.avg:.10f}. ")

    def train(self, model, dataloader, optimizer, loss_fn, lr_scheduler, accelerator, args, epoch_start=0):
        train_epoch = getattr(self, f"_train_epoch_{args.model_type}")
        model.train()
        for epoch in range(epoch_start, epoch_start+args.epochs):
            if accelerator.is_main_process:
                with open(args.output_path+args.log_file, 'a') as f:
                    f.write(f"\nEpoch {epoch+1} --- learning rate {optimizer.param_groups[0]['lr']:.8f}")
            train_epoch(epoch, model, dataloader, optimizer, loss_fn, accelerator, args)
            #if lr_scheduler is not None and lr_scheduler.get_last_lr()[0] > 0.000001:
            #    lr_scheduler.step()
            if accelerator.is_main_process:
                checkpoint_dict = {
                    "parameters": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    }
                torch.save(checkpoint_dict, args.output_path+f"checkpoint_{epoch}.pth")

'''
#----- VALIDATION ------

def validate_ae(model, dataloader, accelerator, loss_fn, val_loss_meter, val_performance_meter):

    model.eval()
    with torch.no_grad():
        for X in dataloader:
            if accelerator is None:
                X = X.cuda()
            X_pred = model(X)
            loss = loss_fn(X_pred, X)
            val_loss_meter.update(loss.item(), X.shape[0])
            if val_performance_meter is not None:
                perf = accuracy(X_pred, X)
                val_performance_meter.update(val=perf, n=X.shape[0])
        val_loss_meter.add_iter_loss()
        if val_performance_meter is not None:
            val_performance_meter.add_iter_loss()
    model.train()
    return


def validate_gnn(model, dataloader, accelerator, loss_fn, val_loss_meter, val_performance_meter):

    model.eval()
    with torch.no_grad():
        for X, data in dataloader:
            device = 'cuda' if accelerator is None else accelerator.device
            y_pred, y, _ = model(X, data, device)
            if val_performance_meter is not None:
                perf = accuracy(y_pred, y)
                val_performance_meter.update(val=perf, n=X.shape[0])
            loss = loss_fn(y_pred, y)
            val_loss_meter.update(loss.item(), X.shape[0])
        val_loss_meter.add_iter_loss()
        if val_performance_meter is not None:
            val_performance_meter.add_iter_loss()
    model.train()
    return


#------ TEST ------  

def test_model_ae(model, dataloader, log_path, log_file, accelerator, loss_fn=None, performance=None):
    
    if loss_fn is not None:
        loss_meter = AverageMeter()

    i = 0
    model.eval()
    with torch.no_grad():
        for X in dataloader:
            if accelerator is None:
                X = X.cuda()
            X_pred = model(X)
            loss = loss_fn(X_pred, X) if loss_fn is not None else None
            if loss_fn is not None:
                loss_meter.update(loss.item(), X.shape[0])
            if i == 0:
                X_pred = X_pred.detach().cpu().numpy()
                X = X.detach().cpu().numpy()
                with open(log_path+"X_pred.pkl", 'wb') as f:
                    pickle.dump(X_pred, f)
                with open(log_path+"X.pkl", 'wb') as f:
                    pickle.dump(X, f)
            i += 1

    fin_loss_total = loss_meter.sum if loss_fn is not None else None
    fin_loss_avg = loss_meter.avg if loss_fn is not None else None
    if accelerator is None or accelerator.is_main_process:
        with open(log_path+log_file, 'a') as f:
            f.write(f"\nTESTING - loss total = {fin_loss_total if fin_loss_total is not None else '--'},"
                    +f"loss avg = {fin_loss_avg if fin_loss_avg is not None else '--'}")
    return fin_loss_total, fin_loss_avg


def test_model_gnn(model, dataloader, log_path, log_file, accelerator, loss_fn=None, performance=None):
    
    if loss_fn is not None:
        loss_meter = AverageMeter()
    if performance is not None:
        perf_meter = AverageMeter()

    y_pred_list = []
    y_list = []
    model.eval()
    with torch.no_grad():
        for X, data in dataloader:
            device = 'cuda' if accelerator is None else accelerator.device
            y_pred, y, _ = model(X, data, device)
            if loss_fn is not None:
                loss = loss_fn(y_pred, y)
                loss_meter.update(loss.item(), X.shape[0])
            else:
                loss = None
            if performance is not None:
                perf = accuracy(y_pred, y)
                perf_meter.update(perf, X.shape[0])
                _ = [y_pred_list.append(yi) for yi in torch.argmax(y_pred, dim=-1).detach().cpu().numpy()]
            else:
                _ = [y_pred_list.append(yi) for yi in y_pred.detach().cpu().numpy()]
            
            _ = [y_list.append(yi) for yi in y.detach().cpu().numpy()]
        y_list = np.array(y_list)
        y_pred_list = np.array(y_pred_list)
        with open(log_path+"y_pred.pkl", 'wb') as f:
            pickle.dump(y_pred_list, f)
        with open(log_path+"y.pkl", 'wb') as f:                  
            pickle.dump(y_list, f)

    fin_loss_total = loss_meter.sum if loss_fn is not None else None
    fin_loss_avg = loss_meter.avg if loss_fn is not None else None
    fin_perf_avg = perf_meter.avg if performance is not None else None

    if accelerator is None or accelerator.is_main_process:
        with open(log_path+log_file, 'a') as f:
            f.write(f"\nTESTING - loss total = {fin_loss_total if fin_loss_total is not None else '--'},"
                    +f"loss avg = {fin_loss_avg if fin_loss_avg is not None else '--'}. Performance = {fin_perf_avg}.")

    return fin_loss_total, fin_loss_avg
'''

