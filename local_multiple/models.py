import numpy as np
import torch
from torch import nn
from torch_geometric import nn as geometric_nn
from torch_geometric.nn import GATConv, GATv2Conv
from torch_geometric.data import Batch
import sys
import time
import copy

class Autoencoder(nn.Module):
    def __init__(self, input_size=5, gru_hidden_dim=24, cnn_output_dim=256, n_layers=2):
        super().__init__() 
        self.cnn_output_dim = cnn_output_dim
        self.gru_hidden_dim = gru_hidden_dim

        self.encoder = nn.Sequential(
            nn.Conv3d(input_size, 64, kernel_size=3, padding=(1,1,1), stride=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=(1,1,1), stride=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=(1,1,1), stride=2),
            nn.Conv3d(64, 256, kernel_size=3, padding=(1,1,1), stride=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=(1,0,0), stride=2),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, cnn_output_dim),
            nn.BatchNorm1d(cnn_output_dim),
            nn.ReLU()
            )

        # define the decoder modules
        self.gru = nn.Sequential(
            nn.GRU(cnn_output_dim, gru_hidden_dim, n_layers, batch_first=True),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.gru_hidden_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 2048),
            nn.Unflatten(-1,(256, 2, 2, 2)),
            nn.Upsample(size=(3,4,4)),
            nn.ReLU(),
            nn.ConvTranspose3d(256, 64, kernel_size=3, padding=(1,1,1), stride=1),
            nn.ReLU(),
            nn.Upsample(size=(5,6,6)),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 64, kernel_size=3, padding=(1,1,1), stride=1),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 5, kernel_size=3, padding=(1,1,1), stride=1),
            )

    def forward(self, X):
        s = X.shape
        X = X.reshape(s[0]*s[1], s[2], s[3], s[4], s[5])        # (batch_dim*25, 5, 5, 6, 6)
        X = self.encoder(X)                                     # (batch_dim*25, cnn_output_dim)
        X = X.reshape(s[0], s[1], self.cnn_output_dim)          # (batch_dim, 25, cnn_output_dim)
        X, _ = self.gru(X) # out, h                             # (batch_dim, 25, gru_hidden_dim
        X = X.reshape(s[0]*25, self.gru_hidden_dim)             # (batch_dim*25, gru_hidden_dim)
        X = self.decoder(X)                                     # (batch_dim*25, gru_hidden_dim)
        X = X.reshape(s[0], s[1], s[2], s[3], s[4], s[5])       # (batch_dim, 25, 5, 5, 6, 6)
        return X

class Encoder(nn.Module):
    def __init__(self, input_size=5, gru_hidden_dim=24, cnn_output_dim=256, n_layers=2):
        super().__init__() 
        self.cnn_output_dim = cnn_output_dim
        self.gru_hidden_dim = gru_hidden_dim

        self.encoder = nn.Sequential(
            nn.Conv3d(input_size, 64, kernel_size=3, padding=(1,1,1), stride=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=(1,1,1), stride=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=(1,1,1), stride=2),
            nn.Conv3d(64, 256, kernel_size=3, padding=(1,1,1), stride=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=(1,0,0), stride=2),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, cnn_output_dim),
            nn.BatchNorm1d(cnn_output_dim),
            nn.ReLU()
            )

        # define the decoder modules
        self.gru = nn.Sequential(
            nn.GRU(cnn_output_dim, gru_hidden_dim, n_layers, batch_first=True),
        )

    def forward(self, X):
        s = X.shape
        X = X.reshape(s[0]*s[1], s[2], s[3], s[4], s[5])        # (batch_dim*25, 5, 5, 6, 6)
        encoding = self.encoder(X)                                     # (batch_dim*25, cnn_output_dim)
        encoding = encoding.reshape(s[0], s[1], self.cnn_output_dim)          # (batch_dim, 25, cnn_output_dim)
        encoding, _ = self.gru(encoding) # out, h                             # (batch_dim, 25, gru_hidden_dim
        return encoding # (batch_size, encoding_dim)


class Classifier(nn.Module):
    def __init__(self, input_size=5, gru_hidden_dim=12, cnn_output_dim=256, n_layers=2, num_node_features=1):
        super().__init__()
        self.cnn_output_dim = cnn_output_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.num_node_features = num_node_features

        self.encoder = nn.Sequential(
            nn.Conv3d(input_size, 64, kernel_size=3, padding=(1,1,1), stride=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=(1,1,1), stride=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=(1,1,1), stride=2),
            nn.Conv3d(64, 256, kernel_size=3, padding=(1,1,1), stride=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=(1,0,0), stride=2),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, cnn_output_dim),
            nn.BatchNorm1d(cnn_output_dim),
            nn.ReLU()
            )

        # define the decoder modules
        self.gru = nn.Sequential(
            nn.GRU(cnn_output_dim, gru_hidden_dim, n_layers, batch_first=True),
        )

        #gnn
        self.gnn = geometric_nn.Sequential('x, edge_index, edge_attr', [
            (geometric_nn.BatchNorm(num_node_features+self.gru_hidden_dim*25), 'x -> x'),
            (GATv2Conv(num_node_features+self.gru_hidden_dim*25, 128, heads=2, aggr='mean', dropout=0.5, edge_dim=2),  'x, edge_index, edge_attr -> x'), 
            (geometric_nn.BatchNorm(256), 'x -> x'),
            nn.ReLU(),                                                     
            (GATv2Conv(256, 128, aggr='mean', edge_dim=2), 'x, edge_index, edge_attr -> x'),
            (geometric_nn.BatchNorm(128), 'x -> x'),
            nn.ReLU(),
            #(GATv2Conv(128, 2, aggr='mean'), 'x, edge_index -> x'), # weighted cross entropy
            #nn.Softmax(dim=-1)                                      # weighted cross entropy
            (GATv2Conv(128, 1, aggr='mean', edge_dim=2), 'x, edge_index, edge_attr -> x'), # focal loss
            nn.Sigmoid()                                            # focal loss
            ])
        
    def forward(self, X_batch, data_list):
        s = X_batch.shape
        X_batch = X_batch.reshape(s[0]*s[1]*s[2], s[3], s[4], s[5], s[6])   # (batch_dim*9*25, 5, 5, 6, 6)
        X_batch = self.encoder(X_batch)                                     # (batch_dim*9*25, cnn_output_dim)
        X_batch = X_batch.reshape(s[0]*s[1], s[2], self.cnn_output_dim)     # (batch_dim*9, 25, cnn_output_dim)
        encoding, _ = self.gru(X_batch)                                     # (batch_dim*9, 25, gru_hidden_dim)
        encoding = encoding.reshape(s[0], s[1], s[2]*self.gru_hidden_dim)   # (batch_dim, 9, 25*gru_hidden_dim)

        for i, data in enumerate(data_list):
            data['x'] = torch.cat((data.z, encoding[i,data.idx_list_mapped,:]),dim=-1)
        
        data_batch = Batch.from_data_list(data_list)                                                    
        
        y_pred = self.gnn(data_batch.x, data_batch.edge_index, data_batch.edge_attr.float())
        train_mask = data_batch.train_mask

        return y_pred[train_mask].squeeze(), data_batch.y[train_mask]              
            #return y_pred, data_batch.y.squeeze().to(torch.long), data_batch.batch  # weighted cross entropy loss


class Regressor(nn.Module):
    def __init__(self, input_size=5, gru_hidden_dim=12, cnn_output_dim=256, n_layers=2, num_node_features=1):
        super().__init__()
        self.cnn_output_dim = cnn_output_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.num_node_features = num_node_features

        self.encoder = nn.Sequential(
            nn.Conv3d(input_size, 64, kernel_size=3, padding=(1,1,1), stride=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=(1,1,1), stride=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=(1,1,1), stride=2),
            nn.Conv3d(64, 256, kernel_size=3, padding=(1,1,1), stride=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=(1,0,0), stride=2),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, cnn_output_dim),
            nn.BatchNorm1d(cnn_output_dim),
            nn.ReLU()
            )

        self.gru = nn.Sequential(
            nn.GRU(cnn_output_dim, gru_hidden_dim, n_layers, batch_first=True),
        )

        self.gnn = geometric_nn.Sequential('x, edge_index, edge_attr', [
            (geometric_nn.BatchNorm(num_node_features+self.gru_hidden_dim*25), 'x -> x'),
            (GATv2Conv(num_node_features+self.gru_hidden_dim*25, 128, heads=2, aggr='mean', dropout=0.5, edge_dim=2),  'x, edge_index, edge_attr -> x'), 
            (geometric_nn.BatchNorm(256), 'x -> x'),
            nn.ReLU(),                                                     
            (GATv2Conv(256, 128, aggr='mean', edge_dim=2), 'x, edge_index, edge_attr -> x'),
            (geometric_nn.BatchNorm(128), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(128, 1, aggr='mean', edge_dim=2), 'x, edge_index, edge_attr -> x'),
            ])
        
    def forward(self, X_batch, data_list):
        s = X_batch.shape
        X_batch = X_batch.reshape(s[0]*s[1]*s[2], s[3], s[4], s[5], s[6])   # (batch_dim*9*25, 5, 5, 6, 6)
        X_batch = self.encoder(X_batch)                                     # (batch_dim*9*25, cnn_output_dim)
        X_batch = X_batch.reshape(s[0]*s[1], s[2], self.cnn_output_dim)     # (batch_dim*9, 25, cnn_output_dim)
        encoding, _ = self.gru(X_batch)                                     # (batch_dim*9, 25, gru_hidden_dim)
        encoding = encoding.reshape(s[0], s[1], s[2]*self.gru_hidden_dim)   # (batch_dim, 9, 25*gru_hidden_dim)
        
        for i, data in enumerate(data_list):
            data['x'] = torch.cat((data.z, encoding[i,data.idx_list_mapped,:]),dim=-1)
        
        data_batch = Batch.from_data_list(data_list, exclude_keys=["z", "low_res", "mask_1_cell", "mask_subgraph", "idx_list", "idx_list_mapped"]) 
        
        y_pred = self.gnn(data_batch.x, data_batch.edge_index, data_batch.edge_attr.float())
        train_mask = data_batch.train_mask

        return y_pred[train_mask].squeeze(), data_batch.y[train_mask]     

class Regressor_GNN(nn.Module):
    def __init__(self, encoding_dim=128, num_node_features=1):
        super().__init__()
        self.encoding_dim = encoding_dim

        self.gnn = geometric_nn.Sequential('x, edge_index, edge_attr', [
            (geometric_nn.BatchNorm(num_node_features+self.encoding_dim), 'x -> x'),
            (GATv2Conv(num_node_features+encoding_dim, 64, heads=2, aggr='mean', dropout=0.5, edge_dim=2),  'x, edge_index, edge_attr -> x'), 
            (geometric_nn.BatchNorm(128), 'x -> x'),
            nn.ReLU(),                                                     
            (GATv2Conv(128, 64, aggr='mean', edge_dim=2), 'x, edge_index, edge_attr -> x'),
            (geometric_nn.BatchNorm(64), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(64, 1, aggr='mean', edge_dim=2), 'x, edge_index, edge_attr -> x'),
            ])

    def forward(self, encoding_batch, data_batch, num_node_features=1): # encoding_batch.shape = (batch_dim, space_dim, time_dim, 128)
        for i, data in enumerate(data_batch):
            features = torch.zeros((data.num_nodes, num_node_features + self.encoding_dim)).cuda()
            features[:,0] = data.x
            for j, idx in enumerate(data.idx_list):
                #print(features[data.low_res==idx,num_node_features:].shape, encoding_batch.shape, encoding_batch[i,j,:,:].shape)
                #sys.exit()
                features[data.low_res==idx,num_node_features:] = encoding_batch[i,j,:]
            data.__setitem__('x', features)
        data_batch = Batch.from_data_list(data_batch)
        y_pred = self.gnn(data_batch.x, data_batch.edge_index, data_batch.edge_attr.float())
        train_mask = data_batch.train_mask
        return y_pred[train_mask].squeeze(), data_batch.y[train_mask]              
        
class Classifier_test(Classifier):

    def __init__(self):
        super().__init__()
    
    def forward(self, X_batch, data_list, G_test):
        s = X_batch.shape
        X_batch = X_batch.reshape(s[0]*s[1]*s[2], s[3], s[4], s[5], s[6])   # (batch_dim*9*25, 5, 5, 6, 6)
        X_batch = self.encoder(X_batch)                                     # (batch_dim*9*25, cnn_output_dim)
        X_batch = X_batch.reshape(s[0]*s[1], s[2], self.cnn_output_dim)     # (batch_dim*9, 25, cnn_output_dim)
        encoding, _ = self.gru(X_batch)                                     # (batch_dim*9, 25, gru_hidden_dim)
        encoding = encoding.reshape(s[0], s[1], s[2]*self.gru_hidden_dim)   # (batch_dim, 9, 25*gru_hidden_dim)

        for i, data in enumerate(data_list):
            data['x'] = torch.cat((data.z, encoding[i,data.idx_list_mapped,:]),dim=-1)
        
        data_batch = Batch.from_data_list(data_list, exclude_keys=["z", "low_res", "idx_list", "idx_list_mapped"]) 
        y_pred = self.gnn(data_batch.x, data_batch.edge_index, data_batch.edge_attr.float())
        for idx in range(s[0]):
            data = data_batch.get_example(idx)
            #print(y_pred.shape, G_test['pr_reg'].shape, data.mask_1_cell.shape, G_test['pr_reg'][data.mask_1_cell, data.time_idx].shape)
            y_pred_i = torch.where(y_pred[data_batch.batch == idx][data.test_mask].squeeze() > 0.5, 1.0, 0.0).cpu()
            G_test['pr_cl'][data.mask_1_cell, data.time_idx] = y_pred_i
            return
    #return y_pred, data_batch.y.squeeze().to(torch.long), data_batch.batch  # weighted cross entropy loss


class Regressor_test(Regressor):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, X_batch, data_list, G_test):
        s = X_batch.shape
        X_batch = X_batch.reshape(s[0]*s[1]*s[2], s[3], s[4], s[5], s[6])   # (batch_dim*9*25, 5, 5, 6, 6)
        X_batch = self.encoder(X_batch)                                     # (batch_dim*9*25, cnn_output_dim)
        X_batch = X_batch.reshape(s[0]*s[1], s[2], self.cnn_output_dim)     # (batch_dim*9, 25, cnn_output_dim)
        encoding, _ = self.gru(X_batch)                                     # (batch_dim*9, 25, gru_hidden_dim)
        encoding = encoding.reshape(s[0], s[1], s[2]*self.gru_hidden_dim)   # (batch_dim, 9, 25*gru_hidden_dim)

        for i, data in enumerate(data_list):
            data['x'] = torch.cat((data.z, encoding[i,data.idx_list_mapped,:]),dim=-1)
        
        data_batch = Batch.from_data_list(data_list, exclude_keys=["z", "low_res", "idx_list", "idx_list_mapped"]) 
        y_pred = self.gnn(data_batch.x, data_batch.edge_index, data_batch.edge_attr.float())
        for idx in range(s[0]):
            data = data_batch.get_example(idx)
            y_pred_i = y_pred[data_batch.batch == idx][data.test_mask].squeeze().cpu()
            G_test['pr_reg'][data.mask_1_cell, data.time_idx] = y_pred_i
        return     


if __name__ =='__main__':

    model = Regressor()
    batch_dim = 64
    input_batch = torch.rand((batch_dim, 25, 5, 5, 6, 6))

    start = time.time()
    X = model(input_batch)
    print(f"{time.time()-start} s\n")
    print(X.shape) 
