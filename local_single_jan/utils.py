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
        self.avg_list = []
        self.avg_iter_list = []

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
    
    def add_loss(self):
        self.avg_list.append(self.avg)
    
    def add_iter_loss(self):
        self.avg_iter_list.append(self.avg)


def use_gpu_if_possible():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def accuracy(prediction, target):
    if prediction.shape == target.shape:
        prediction_class = torch.where(prediction > 0.5, 1.0, 0.0) 
    else:
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
    return


def countour_italy_list(path='/home/vblasone/precipitation-maps/', file_name='Italia.txt'):
    zones = []
    with open('/home/vblasone/precipitation-maps/Italia.txt') as f:
        lines = f.read()
        for zone in lines.split(';'):
            zones.append(zone)
    for i in range(len(zones)):
        zones[i] = zones[i].split('\n')
        for j in range(len(zones[i])):
            zones[i][j] = zones[i][j].split(',')
        if [''] in zones[i]:
            zones[i].remove([''])
    for i in range(len(zones)):
        for j in range(len(zones[i])):
            if '' in zones[i][j]:
                zones[i][j].remove('')
            if zones[i][j] == []:
                del zones[i][j]
                continue
            for k in range(len(zones[i][j])):
                zones[i][j][k] = float(zones[i][j][k])
    return zones


def draw_rectangle(x_min, x_max, y_min, y_max, color, ax, fill=False, fill_color=None, alpha=0.5):
    y_grid = [y_min, y_min, y_max, y_max, y_min]
    x_grid = [x_min, x_max, x_max, x_min, x_min]
    ax.plot(x_grid, y_grid, color=color)
    if fill:
        if fill_color==None:
            fill_color = color
        ax.fill(x_grid, y_grid, color=fill_color, alpha=alpha)
    return


def plot_countour_italy(zones, ax, color='k', alpha_fill=0.1, linewidth=1):
    j = 0
    for zone in zones:
        x_zone = [zone[i][0] for i in range(len(zone)) if i > 0]
        y_zone = [zone[i][1] for i in range(len(zone)) if i > 0]
        ax.fill(x_zone, y_zone, color, alpha=alpha_fill)
        ax.plot(x_zone, y_zone, color, alpha=1, linewidth=1)
        j += 1
    return


#------Training utilities------

#------EPOCH LOOPS------  

def train_epoch_ae(model, dataloader, loss_fn, optimizer, lr_scheduler, loss_meter, performance_meter, val_loss_meter,
        val_performance_meter, log_path, log_file, validationloader, validate_model, accelerator, intermediate=False, epoch=0):
    
    loss_meter.reset()
    val_loss_meter.reset()
    if performance_meter is not None:
        performance_meter.reset()
        val_performance_meter.reset()

    i = 0
    for X in dataloader:
        if accelerator is None:
            X = X.cuda()
        optimizer.zero_grad()
        X_pred = model(X)
        loss = loss_fn(X_pred, X)
        if accelerator is None:
            loss.backward()
        else:
            accelerator.backward(loss)
        optimizer.step()
        loss_meter.update(val=loss.item(), n=X.shape[0])
        loss_meter.add_iter_loss()
 
        if performance_meter is not None:
            perf = accuracy(X_pred, X)
            performance_meter.update(val=perf, n=X.shape[0])
            performance_meter.add_iter_loss()

        if i % 5000 == 0:
            validate_model(model, validationloader, accelerator, loss_fn, val_loss_meter, val_performance_meter)
            if intermediate:
                with open(log_path+log_file, 'a') as f:
                    if val_performance_meter is not None:
                        f.write(f"\nValidation loss at iteration {i}, tot = {val_loss_meter.sum}, avg = {val_loss_meter.avg}, val perf avg = {val_performance_meter.avg}.")
                    else:
                        f.write(f"\nValidation loss at iteration {i}, tot = {val_loss_meter.sum}, avg = {val_loss_meter.avg}")
            np.savetxt(log_path+"val_loss_iter.csv", val_loss_meter.avg_iter_list)
            np.savetxt(log_path+"train_loss_iter.csv", loss_meter.avg_iter_list)

            if performance_meter is not None:
                np.savetxt(log_path+"train_accuracy_iter.csv", performance_meter.avg_iter_list)
                np.savetxt(log_path+"val_accuracy_iter.csv", val_performance_meter.avg_iter_list)
        i += 1

    validate_model(model, validationloader, accelerator, loss_fn, val_loss_meter, val_performance_meter)


def train_epoch_gnn(model, dataloader, loss_fn, optimizer, lr_scheduler, loss_meter, performance_meter, val_loss_meter,
            val_performance_meter, log_path, log_file, validationloader, validate_model, accelerator, intermediate=False, epoch=0):

    loss_meter.reset()
    val_loss_meter.reset()
    if performance_meter is not None:
        performance_meter.reset()
        val_performance_meter.reset()

    model.train()
    i = 0
    for X, data in dataloader:
        device = 'cuda' if accelerator is None else accelerator.device
        optimizer.zero_grad()
        y_pred, y = model(X, data, device)
        loss = loss_fn(y_pred, y, alpha=0.9, gamma=2, reduction='mean')
        #loss = loss_fn(y_pred, y)
        if accelerator is None:
            loss.backward()
        else:
            accelerator.backward(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(),5)
        optimizer.step()
        loss_meter.update(val=loss.item(), n=X.shape[0])    
        loss_meter.add_iter_loss()    

        if performance_meter is not None:
            perf = accuracy(y_pred, y)
            performance_meter.update(val=perf, n=X.shape[0])
            performance_meter.add_iter_loss()
        
        #print("OK!")
        #sys.exit()

        if i % 5000 == 0:
            #validate_model(model, validationloader, accelerator, loss_fn, val_loss_meter, val_performance_meter)
            #if intermediate:
            #    with open(log_path+log_file, 'a') as f:
            #        if val_performance_meter is not None:
            #            f.write(f"\nValidation loss at iteration {i}, tot = {val_loss_meter.sum}, avg = {val_loss_meter.avg}, val perf avg = {val_performance_meter.avg}.")
            #        else:
            #            f.write(f"\nValidation loss at iteration {i}, tot = {val_loss_meter.sum}, avg = {val_loss_meter.avg}")
        
            if accelerator is None or accelerator.is_main_process:
                np.savetxt(log_path+"train_loss_iter.csv", loss_meter.avg_iter_list)
                np.savetxt(log_path+"val_loss_iter.csv", val_loss_meter.avg_iter_list)
                if performance_meter is not None:
                    np.savetxt(log_path+"train_perf_iter.csv", performance_meter.avg_iter_list)
                    np.savetxt(log_path+"val_perf_iter.csv", val_performance_meter.avg_iter_list)
                checkpoint_dict = {
                    "parameters": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch
                    }
                torch.save(checkpoint_dict, log_path+f"checkpoint_{epoch}_tmp.pth")

        i += 1

    #validate_model(model, validationloader, accelerator, loss_fn, val_loss_meter, val_performance_meter)


#------ TRAIN ------  

def train_model(model, dataloader, loss_fn, optimizer, num_epochs,
        log_path, log_file, train_epoch, validate_model, validationloader, accelerator,
        lr_scheduler=None, checkpoint_name="checkpoint.pth", performance=None, epoch_start=0):
    
    model.train()

    # define average meter objects
    loss_meter = AverageMeter()
    val_loss_meter = AverageMeter()

    if performance is not None:
        performance_meter = AverageMeter()
        val_performance_meter = AverageMeter()
    else:
        performance_meter = None
        val_performance_meter = None

    # epoch loop
    for epoch in range(epoch_start, epoch_start + num_epochs):
        
        if accelerator is None or accelerator.is_main_process:
            with open(log_path+log_file, 'a') as f:
                f.write(f"\nEpoch {epoch+1} --- learning rate {optimizer.param_groups[0]['lr']:.8f}")
        
        start_time = time.time()
        
        train_epoch(model, dataloader, loss_fn, optimizer, lr_scheduler, loss_meter, performance_meter, val_loss_meter,
            val_performance_meter, log_path, log_file, validationloader, validate_model, accelerator, epoch=epoch)
        
        end_time = time.time()
        
        loss_meter.add_loss()
        val_loss_meter.add_loss()
        if performance is not None:
            performance_meter.add_loss()
            val_performance_meter.add_loss()

        if lr_scheduler is not None and lr_scheduler.get_last_lr()[0] > 0.000001:
            lr_scheduler.step()

        if accelerator is None or accelerator.is_main_process:
            with open(log_path+log_file, 'a') as f:
                if performance_meter is None:
                    f.write(f"\nEpoch {epoch+1} completed in {end_time - start_time:.4f} seconds. Loss - total: {loss_meter.sum:.4f} - average: {loss_meter.avg:.10f}. "+
                            f"Validation loss avg = {val_loss_meter.avg:.4f}")
                else:
                    f.write(f"\nEpoch {epoch+1} completed in {end_time - start_time:.4f} seconds. Loss - total: {loss_meter.sum:.4f} - average: {loss_meter.avg:.10f}; "+
                            f"performance: {performance_meter.avg:.4f}. Validation loss avg = {val_loss_meter.avg:.4f}; performance: {val_performance_meter.avg:.4f}")

            np.savetxt(log_path+"train_loss.csv", loss_meter.avg_list)
            np.savetxt(log_path+"val_loss.csv", val_loss_meter.avg_list)
            np.savetxt(log_path+"train_loss_iter.csv", loss_meter.avg_iter_list)
            np.savetxt(log_path+"val_loss_iter.csv", val_loss_meter.avg_iter_list)
            if performance is not None:
                np.savetxt(log_path+"train_perf.csv", performance_meter.avg_list)
                np.savetxt(log_path+"val_perf.csv", val_performance_meter.avg_list)
                np.savetxt(log_path+"train_perf_iter.csv", performance_meter.avg_iter_list)
                np.savetxt(log_path+"val_perf_iter.csv", val_performance_meter.avg_iter_list)
            checkpoint_dict = {
                "parameters": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                }
            torch.save(checkpoint_dict, log_path+f"checkpoint_{epoch}.pth")

