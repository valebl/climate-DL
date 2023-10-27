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
    def __init__(self, input_size=5, cnn_output_dim=256, gru_input_dim=256, gru_hidden_dim=256, encoding_dim=512, n_layers=2):
        super().__init__() 
        self.cnn_output_dim = cnn_output_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.encoding_dim = encoding_dim

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
            nn.GRU(gru_input_dim, gru_hidden_dim, n_layers, batch_first=True),
        )

        self.dense = nn.Sequential(
            nn.Linear(gru_hidden_dim*25, encoding_dim),
            nn.ReLU()
        )

        self.decoder_dense = nn.Sequential(
            nn.Linear(encoding_dim, encoding_dim*25),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 2048),
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
        X = X.reshape(s[0]*s[1], s[2], s[3], s[4], s[5])                # (batch_dim*25, 5, 5, 6, 6)
        X = self.encoder(X)                                             # (batch_dim*25, cnn_output_dim)
        X = X.reshape(s[0], s[1], self.cnn_output_dim)                  # (batch_dim, 25, cnn_output_dim)
        encoding, _ = self.gru(X) # out, h                              # (batch_dim, 25, gru_hidden_dim
        encoding = encoding.reshape(s[0], s[1]*self.gru_hidden_dim)     # (batch_dim, 25*gru_hidden_dim)
        encoding = self.dense(encoding)                                 # (batch_dim, encoding_dim)
        out = self.decoder_dense(encoding)                              # (batch_dim, encoding_dim*25)
        out = out.reshape(s[0]*25, self.encoding_dim)                   # (batch_dim*25, encoding_dim)
        out = self.decoder(out)                                         # (batch_dim,*25, 5*5*6*6)
        out = out.reshape(s[0], s[1], s[2], s[3], s[4], s[5])           # (batch_dim, 25, 5, 5, 6, 6)
        return out

class Autoencoder_space(nn.Module):
    def __init__(self, input_size=5, encoding_dim=128):
        super().__init__()
        self.encoding_dim = encoding_dim

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
            #nn.Linear(2048, 512),
            #nn.BatchNorm1d(512),
            #nn.ReLU(),
            nn.Linear(2048, encoding_dim),
            #nn.BatchNorm1d(encoding_dim),
            nn.ReLU()
            )

        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 2048),
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
        x = self.encoder(X)                                         # (batch_dim, encoding_dim)
        x = self.decoder(x)                                         # (batch_dim, 5*5*6*6)
        return x


class Encoder_space(nn.Module):
    def __init__(self, input_size=5, encoding_dim=128):
        super().__init__()
        self.encoding_dim = encoding_dim

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
            nn.Linear(2048, encoding_dim),
            nn.ReLU()
            )

    
    def forward(self, X):
        x = self.encoder(X)                                         # (batch_dim, encoding_dim)
        return x


#----------------------------------------------
#-------- no lat lon in node features ---------
#----------------------------------------------

class Classifier_z_only(nn.Module):
    def __init__(self, input_size=5, gru_hidden_dim=12, cnn_output_dim=256, n_layers=2, input_dim=256, hidden_dim=256, node_dim=1):
        super().__init__()
        self.cnn_output_dim = cnn_output_dim
        self.node_dim = node_dim
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
            nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True),
            )
        self.dense = nn.Sequential(
            nn.Linear(hidden_dim*25, 512),
            nn.ReLU()
            )
        self.gnn = geometric_nn.Sequential('x, edge_index', [
            (geometric_nn.BatchNorm(node_dim+512), 'x -> x'),
            (GATv2Conv(node_dim+512, 128, heads=2, aggr='mean', dropout=0.5),  'x, edge_index -> x'),
            (geometric_nn.BatchNorm(256), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(256, 128, aggr='mean'), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(128), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(128, 1, aggr='mean'), 'x, edge_index -> x'),
            nn.Sigmoid()
            ])
    
    def forward(self, X_batch, data_batch, device):
        s = X_batch.shape
        X_batch = X_batch.reshape(s[0]*s[1], s[2], s[3], s[4], s[5])        # (batch_dim*25, 5, 5, 6, 6)
        X_batch = self.encoder(X_batch)                                     # (batch_dim*25, cnn_output_dim)
        X_batch = X_batch.reshape(s[0], s[1], self.cnn_output_dim)          # (batch_dim, 25, cnn_output_dim)
        encoding, _ = self.gru(X_batch)                                     # (batch_dim, 25, gru_hidden_dim)
        encoding = encoding.reshape(s[0], s[1]*self.cnn_output_dim)         # (batch_dim, 25*gru_hidden_dim)
        encoding = self.dense(encoding)

        for i, data in enumerate(data_batch):
            data = data.to(device)
            features = torch.zeros((data.num_nodes, self.node_dim + encoding.shape[1])).to(device)
            features[:,:self.node_dim] = data.x[:,:self.node_dim]
            features[:,self.node_dim:] = encoding[i,:]
            data.__setitem__('x', features)
            
        data_batch = Batch.from_data_list(data_batch, exclude_keys=["low_res", "mask_1_cell", "mask_subgraph", "idx_list", "idx_list_mapped"]) 
        y_pred = self.gnn(data_batch.x, data_batch.edge_index) 
        train_mask = data_batch.train_mask
        return y_pred.squeeze()[train_mask], data_batch.y.squeeze()     


class Regressor_z_only(nn.Module):
    def __init__(self, input_size=5, gru_hidden_dim=12, cnn_output_dim=256, n_layers=2, input_dim=256, hidden_dim=256, node_dim=1):
        super().__init__()
        self.cnn_output_dim = cnn_output_dim
        self.node_dim = node_dim
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
            nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True),
            )
        self.dense = nn.Sequential(
            nn.Linear(hidden_dim*25, 512),
            nn.ReLU()
            )
        self.gnn = geometric_nn.Sequential('x, edge_index', [
            (geometric_nn.BatchNorm(node_dim+512), 'x -> x'),
            (GATv2Conv(node_dim+512, 128, heads=2, aggr='mean', dropout=0.5),  'x, edge_index -> x'),
            (geometric_nn.BatchNorm(256), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(256, 128, aggr='mean'), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(128), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(128, 1, aggr='mean'), 'x, edge_index -> x'),
            ])

    def forward(self, X_batch, data_batch, device):
        s = X_batch.shape
        X_batch = X_batch.reshape(s[0]*s[1], s[2], s[3], s[4], s[5])        # (batch_dim*25, 5, 5, 6, 6)
        X_batch = self.encoder(X_batch)                                     # (batch_dim*25, cnn_output_dim)
        X_batch = X_batch.reshape(s[0], s[1], self.cnn_output_dim)          # (batch_dim, 25, cnn_output_dim)
        encoding, _ = self.gru(X_batch)                                     # (batch_dim, 25, gru_hidden_dim)
        encoding = encoding.reshape(s[0], s[1]*self.cnn_output_dim)         # (batch_dim, 25*gru_hidden_dim)
        encoding = self.dense(encoding)

        for i, data in enumerate(data_batch):
            data = data.to(device)
            features = torch.zeros((data.num_nodes, self.node_dim + encoding.shape[1])).to(device)
            features[:,:self.node_dim] = data.x[:,:self.node_dim]
            features[:,self.node_dim:] = encoding[i,:]
            data.__setitem__('x', features)
            
        data_batch = Batch.from_data_list(data_batch, exclude_keys=["low_res", "mask_1_cell", "mask_subgraph", "idx_list", "idx_list_mapped"]) 
        y_pred = self.gnn(data_batch.x, data_batch.edge_index)
        train_mask = data_batch.train_mask
        return y_pred.squeeze()[train_mask], data_batch.y.squeeze(), data_batch.w.squeeze()


#----------------------------------------------
#-------------- edge attributes ---------------
#----------------------------------------------

class Classifier_edges(nn.Module):
    def __init__(self, input_size=5, cnn_output_dim=256, n_layers=2, input_dim=256, hidden_dim=256, node_dim=1, edge_attr_dim=1):
        super().__init__()
        self.cnn_output_dim = cnn_output_dim
        self.node_dim = node_dim
        self.edge_attr_dim = edge_attr_dim
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
            nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True),
            )
        self.dense = nn.Sequential(
            nn.Linear(hidden_dim*25, 512),
            nn.ReLU()
            ) 
        self.gnn = geometric_nn.Sequential('x, edge_index, edge_attr', [
            (geometric_nn.BatchNorm(node_dim+512), 'x -> x'),
            (GATv2Conv(node_dim+512, 128, heads=2, aggr='mean', dropout=0.5, edge_dim=edge_attr_dim),  'x, edge_index, edge_attr -> x'),
            (geometric_nn.BatchNorm(256), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(256, 128, aggr='mean', edge_dim=edge_attr_dim), 'x, edge_index, edge_attr -> x'),
            (geometric_nn.BatchNorm(128), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(128, 1, aggr='mean', edge_dim=edge_attr_dim), 'x, edge_index, edge_attr -> x'),
            nn.Sigmoid()
            ])
    
    def forward(self, X_batch, data_batch, device):
        s = X_batch.shape
        X_batch = X_batch.reshape(s[0]*s[1], s[2], s[3], s[4], s[5])        # (batch_dim*25, 5, 5, 6, 6)
        X_batch = self.encoder(X_batch)                                     # (batch_dim*25, cnn_output_dim)
        X_batch = X_batch.reshape(s[0], s[1], self.cnn_output_dim)          # (batch_dim, 25, cnn_output_dim)
        encoding, _ = self.gru(X_batch)                                     # (batch_dim, 25, gru_hidden_dim)
        encoding = encoding.reshape(s[0], s[1]*self.cnn_output_dim)         # (batch_dim, 25*gru_hidden_dim)
        encoding = self.dense(encoding)

        for i, data in enumerate(data_batch):
            data = data.to(device)
            features = torch.zeros((data.num_nodes, self.node_dim + encoding.shape[1])).to(device)
            features[:,:self.node_dim] = data.x[:,:self.node_dim]
            features[:,self.node_dim:] = encoding[i,:]
            #features[:,:1] = data.x[:,:1]
            #features[:,1:3] = data.laplacian_eigenvector_pe[:,:]
            #features[:,3:] = encoding[i,:]
            data.__setitem__('x', features)
            
        data_batch = Batch.from_data_list(data_batch, exclude_keys=["low_res", "mask_1_cell", "mask_subgraph", "idx_list", "idx_list_mapped", "laplacian_eigenvector_pe"]) 
        y_pred = self.gnn(data_batch.x, data_batch.edge_index, data_batch.edge_attr.float()) 
        train_mask = data_batch.train_mask
        return y_pred.squeeze()[train_mask], data_batch.y.squeeze()     


class Regressor_edges(nn.Module):
    def __init__(self, input_size=5, gru_hidden_dim=12, cnn_output_dim=256, n_layers=2, input_dim=256, hidden_dim=256, node_dim=1, edge_attr_dim=1):
        super().__init__()
        self.cnn_output_dim = cnn_output_dim
        self.node_dim = node_dim
        self.edge_attr_dim = edge_attr_dim
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
            nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True),
            )
        self.dense = nn.Sequential(
            nn.Linear(hidden_dim*25, 512),
            nn.ReLU()
            )
        self.gnn = geometric_nn.Sequential('x, edge_index, edge_attr', [
            (geometric_nn.BatchNorm(node_dim+512), 'x -> x'),
            (GATv2Conv(node_dim+512, 128, heads=2, aggr='mean', dropout=0.5, edge_dim=edge_attr_dim),  'x, edge_index, edge_attr -> x'),
            (geometric_nn.BatchNorm(256), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(256, 128, aggr='mean', edge_dim=edge_attr_dim), 'x, edge_index, edge_attr -> x'),
            (geometric_nn.BatchNorm(128), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(128, 1, aggr='mean', edge_dim=edge_attr_dim), 'x, edge_index, edge_attr -> x'),
            ])

    def forward(self, X_batch, data_batch, device):
        s = X_batch.shape
        X_batch = X_batch.reshape(s[0]*s[1], s[2], s[3], s[4], s[5])        # (batch_dim*25, 5, 5, 6, 6)
        X_batch = self.encoder(X_batch)                                     # (batch_dim*25, cnn_output_dim)
        X_batch = X_batch.reshape(s[0], s[1], self.cnn_output_dim)          # (batch_dim, 25, cnn_output_dim)
        encoding, _ = self.gru(X_batch)                                     # (batch_dim, 25, gru_hidden_dim)
        encoding = encoding.reshape(s[0], s[1]*self.cnn_output_dim)         # (batch_dim, 25*gru_hidden_dim)
        encoding = self.dense(encoding)

        for i, data in enumerate(data_batch):
            data = data.to(device)
            features = torch.zeros((data.num_nodes, self.node_dim + encoding.shape[1])).to(device)
            features[:,:self.node_dim] = data.x[:,:self.node_dim]
            features[:,self.node_dim:] = encoding[i,:]
            # add positional encodings
            #features[:,:1] = data.x[:,:1]
            #features[:,1:3] = data.laplacian_eigenvector_pe[:,:]
            #features[:,3:] = encoding[i,:]
            data.__setitem__('x', features)
            
        data_batch = Batch.from_data_list(data_batch, exclude_keys=["low_res", "mask_1_cell", "mask_subgraph", "idx_list", "idx_list_mapped", "laplacian_eigenvector_pe"]) 
        y_pred = self.gnn(data_batch.x, data_batch.edge_index, data_batch.edge_attr.float())
        train_mask = data_batch.train_mask
        return y_pred.squeeze()[train_mask], data_batch.y.squeeze(), data_batch.w.squeeze()


#----------------------------------------------
#---------------- test models -----------------
#----------------------------------------------

## z only

class Classifier_z_only_test(Classifier_z_only):

    def __init__(self):
        super().__init__()

    def forward(self, X_batch, data_list, G_test, device):
        s = X_batch.shape
        X_batch = X_batch.reshape(s[0]*s[1], s[2], s[3], s[4], s[5])        # (batch_dim*25, 5, 5, 6, 6)
        X_batch = self.encoder(X_batch)                                     # (batch_dim*25, cnn_output_dim)
        X_batch = X_batch.reshape(s[0], s[1], self.cnn_output_dim)          # (batch_dim, 25, cnn_output_dim)
        encoding, _ = self.gru(X_batch)                                     # (batch_dim, 25, gru_hidden_dim)
        encoding = encoding.reshape(s[0], s[1]*self.cnn_output_dim)         # (batch_dim, 25*gru_hidden_dim)
        encoding = self.dense(encoding)

        for i, data in enumerate(data_list):
            data = data.to(device)
            features = torch.zeros((data.num_nodes, self.node_dim + encoding.shape[1])).to(device)
            features[:,:self.node_dim] = data.x[:,:self.node_dim]
            features[:,self.node_dim:] = encoding[i,:]
            data.__setitem__('x', features)
            
        data_batch = Batch.from_data_list(data_list, exclude_keys=["low_res", "mask_subgraph", "idx_list", "idx_list_mapped"]).to(device) 
        data_batch['x'] = self.gnn(data_batch.x, data_batch.edge_index)
        
        data_list = data_batch.to_data_list()       

        for data in data_list:
            y_pred_i = data.x.squeeze()
            G_test['pr_cl'][data.mask_1_cell, data.time_idx] = torch.where(y_pred_i > 0.5, 1.0, 0.0).cpu()        
        return

class Regressor_z_only_test(Regressor_z_only):

    def __init__(self):
        super().__init__()
    
    def forward(self, X_batch, data_list, G_test, device):
        s = X_batch.shape
        X_batch = X_batch.reshape(s[0]*s[1], s[2], s[3], s[4], s[5])        # (batch_dim*25, 5, 5, 6, 6)
        X_batch = self.encoder(X_batch)                                     # (batch_dim*25, cnn_output_dim)
        X_batch = X_batch.reshape(s[0], s[1], self.cnn_output_dim)          # (batch_dim, 25, cnn_output_dim)
        encoding, _ = self.gru(X_batch)                                     # (batch_dim, 25, gru_hidden_dim)
        encoding = encoding.reshape(s[0], s[1]*self.cnn_output_dim)         # (batch_dim, 25*gru_hidden_dim)
        encoding = self.dense(encoding)

        for i, data in enumerate(data_list):
            data = data.to(device)
            features = torch.zeros((data.num_nodes, self.node_dim + encoding.shape[1])).to(device)
            features[:,:self.node_dim] = data.x[:,:self.node_dim]
            features[:,self.node_dim:] = encoding[i,:]
            data.__setitem__('x', features)
            
        data_batch = Batch.from_data_list(data_list, exclude_keys=["low_res", "mask_subgraph", "idx_list", "idx_list_mapped"]).to(device) 
        data_batch['x'] = self.gnn(data_batch.x, data_batch.edge_index)
        data_batch.x = torch.expm1(data_batch.x)

        data_list = data_batch.to_data_list()        
        
        for data in data_list:
            y_pred_i = data.x.squeeze().cpu()
            G_test['pr_reg'][data.mask_1_cell, data.time_idx] = torch.where(y_pred_i >= 0.1, y_pred_i, torch.tensor(0.0, dtype=y_pred_i.dtype)) 
        return



## edges

class Classifier_edges_test(Classifier_edges):

    def __init__(self):
        super().__init__()

    def forward(self, X_batch, data_list, G_test, device):
        encoding = self._forward_encoder(X_batch)
        G_test = self._forward_gnn(encoding, data_list, G_test, device)
        return

    def _forward_encoder(self, X_batch):
        s = X_batch.shape
        X_batch = X_batch.reshape(s[0]*s[1], s[2], s[3], s[4], s[5])        # (batch_dim*25, 5, 5, 6, 6)
        X_batch = self.encoder(X_batch)                                     # (batch_dim*25, cnn_output_dim)
        X_batch = X_batch.reshape(s[0], s[1], self.cnn_output_dim)          # (batch_dim, 25, cnn_output_dim)
        encoding, _ = self.gru(X_batch)                                     # (batch_dim, 25, gru_hidden_dim)
        encoding = encoding.reshape(s[0], s[1]*self.cnn_output_dim)         # (batch_dim, 25*gru_hidden_dim)
        encoding = self.dense(encoding)
        return encoding

    def _forward_gnn(self, encoding, data_list, G_test, device):
        for i, data in enumerate(data_list):
            data = data.to(device)
            features = torch.zeros((data.num_nodes, self.node_dim + encoding.shape[1])).to(device)
            features[:,:self.node_dim] = data.x[:,:self.node_dim]
            features[:,self.node_dim:] = encoding[i,:]
            data.__setitem__('x', features)
   
        data_batch = Batch.from_data_list(data_list, exclude_keys=["low_res", "mask_subgraph", "idx_list", "idx_list_mapped"]).to(device) 
        data_batch['x'] = self.gnn(data_batch.x, data_batch.edge_index, data_batch.edge_attr.float()) 
        data_list = data_batch.to_data_list()       

        for data in data_list:
            y_pred_i = data.x.squeeze()
            G_test['pr_cl'][data.mask_1_cell, data.time_idx] = torch.where(y_pred_i > 0.5, 1.0, 0.0).cpu()      
        return G_test


class Classifier_edges_test_large(Classifier_edges_test):

    def __init__(self):
        super().__init__()

    def forward(self, X_batch, data_list, G_test, device):
        encoding = super()._forward_encoder(X_batch)
        G_test = self._forward_gnn(encoding, data_list, G_test, device)
        return

    def _forward_gnn(self, encoding, data_list, G_test, device):
        features = torch.zeros((G_test.num_nodes), self.node_dim + encoding.shape[1]).to(device)   
        for i, data in enumerate(data_list):
            data = data.to(device)
            #features = torch.zeros((data.num_nodes, self.node_dim + encoding.shape[1])).to(device)
            features[data.mask_1_cell,:self.node_dim] = data.x[:,:self.node_dim]
            features[data.mask_1_cell,self.node_dim:] = encoding[i,:]
        y_pred = self.gnn(features, G_test.edge_index.to(device), G_test.edge_attr.float().to(device)).squeeze()
        G_test['pr_cl'][:, data_list[0].time_idx] = torch.where(y_pred > 0.5, 1.0, 0.0).cpu()
        return G_test

class Regressor_edges_test(Regressor_edges):

    def __init__(self):
        super().__init__()

    def forward(self, X_batch, data_list, G_test, device):
        encoding = self._forward_encoder(X_batch)
        G_test = self._forward_gnn(encoding, data_list, G_test, device)
        return

    def _forward_encoder(self, X_batch):
        s = X_batch.shape
        X_batch = X_batch.reshape(s[0]*s[1], s[2], s[3], s[4], s[5])        # (batch_dim*25, 5, 5, 6, 6)
        X_batch = self.encoder(X_batch)                                     # (batch_dim*25, cnn_output_dim)
        X_batch = X_batch.reshape(s[0], s[1], self.cnn_output_dim)          # (batch_dim, 25, cnn_output_dim)
        encoding, _ = self.gru(X_batch)                                     # (batch_dim, 25, gru_hidden_dim)
        encoding = encoding.reshape(s[0], s[1]*self.cnn_output_dim)         # (batch_dim, 25*gru_hidden_dim)
        encoding = self.dense(encoding)
        return encoding

    def _forward_gnn(self, encoding, data_list, G_test, device):
        for i, data in enumerate(data_list):
            data = data.to(device)
            features = torch.zeros((data.num_nodes, self.node_dim + encoding.shape[1])).to(device)
            features[:,:self.node_dim] = data.x[:,:self.node_dim]
            features[:,self.node_dim:] = encoding[i,:]
            data.__setitem__('x', features)
            
        data_batch = Batch.from_data_list(data_list, exclude_keys=["low_res", "mask_subgraph", "idx_list", "idx_list_mapped"]).to(device) 
        data_batch['x'] = self.gnn(data_batch.x, data_batch.edge_index, data_batch.edge_attr.float())
        data_batch.x = torch.expm1(data_batch.x)
        data_list = data_batch.to_data_list()        
        
        for data in data_list:
            y_pred_i = data.x.squeeze().cpu()
            G_test['pr_reg'][data.mask_1_cell, data.time_idx] = torch.where(y_pred_i >= 0.1, y_pred_i, torch.tensor(0.0, dtype=y_pred_i.dtype)) 
        return G_test


class Regressor_edges_test_large(Regressor_edges_test):

    def __init__(self):
        super().__init__()

    def forward(self, X_batch, data_list, G_test, device):
        encoding = super()._forward_encoder(X_batch)
        G_test = self._forward_gnn(encoding, data_list, G_test, device)
        return

    def _forward_gnn(self, encoding, data_list, G_test, device):
        features = torch.zeros((G_test.num_nodes), self.node_dim + encoding.shape[1]).to(device)
        for i, data in enumerate(data_list):
            data = data.to(device)
            #features = torch.zeros((data.num_nodes, self.node_dim + encoding.shape[1])).to(device)
            features[data.mask_1_cell,:self.node_dim] = data.x[:,:self.node_dim]
            features[data.mask_1_cell,self.node_dim:] = encoding[i,:]
        y_pred = self.gnn(features, G_test.edge_index.to(device), G_test.edge_attr.float().to(device)).squeeze()
        y_pred = torch.expm1(y_pred).cpu()
        G_test['pr_reg'][:, data.time_idx] = torch.where(y_pred >= 0.1, y_pred, torch.tensor(0.0, dtype=y_pred.dtype))
        return G_test

if __name__ =='__main__':

    model = Regressor()
    batch_dim = 64
    input_batch = torch.rand((batch_dim, 25, 5, 5, 6, 6))

    start = time.time()
    X = model(input_batch)
    print(f"{time.time()-start} s\n")
    print(X.shape)

