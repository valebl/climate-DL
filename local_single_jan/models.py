import numpy as np
import torch
from torch import nn
from torch_geometric import nn as geometric_nn
from torch_geometric.nn import GATConv, GATv2Conv
from torch_geometric.data import Batch
import sys

class Autoencoder(nn.Module):
    def __init__(self, input_size=5, input_dim=256, hidden_dim=256, output_dim=256, n_layers=2):
        super().__init__() 
        self.output_dim = output_dim

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
            nn.Linear(512, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()            
            )   

        self.gru = nn.Sequential(
            nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True),        
            )

        self.decoder = nn.Sequential(
            nn.Linear(output_dim, 512),
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
        X = X.reshape(s[0]*s[1], s[2], s[3], s[4], s[5])
        X = self.encoder(X)
        X = X.reshape(s[0], s[1], self.output_dim)
        out, h = self.gru(X)
        out = out.reshape(s[0]*s[1], self.output_dim)
        out = self.decoder(out)
        out = out.reshape(s[0], s[1], s[2], s[3], s[4], s[5])
        return out


class Classifier(nn.Module):
    def __init__(self, input_size=5, input_dim=256, hidden_dim=256, output_dim=256, n_layers=2):
        super().__init__()
        self.output_dim = output_dim

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
            nn.Linear(512, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
            )

        # define the decoder modules
        self.gru = nn.Sequential(
            nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True),
        )

        self.dense = nn.Sequential(
            nn.Linear(hidden_dim*25, 512),
            nn.ReLU()
        )

        #gnn
        self.gnn = geometric_nn.Sequential('x, edge_index', [
            (geometric_nn.BatchNorm(3+512), 'x -> x'),
            (GATv2Conv(3+512, 128, heads=2, aggr='mean', dropout=0.5),  'x, edge_index -> x'),
            (geometric_nn.BatchNorm(256), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(256, 128, aggr='mean'), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(128), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(128, 1, aggr='mean'), 'x, edge_index -> x'), # focal loss
            nn.Sigmoid()                                            # focal loss
            ])

    def forward(self, X_batch, data_batch, device):
        s = X_batch.shape
        X_batch = X_batch.reshape(s[0]*s[1], s[2], s[3], s[4], s[5])
        X_batch = self.encoder(X_batch.to(device))
        X_batch = X_batch.reshape(s[0], s[1], self.output_dim)
        encoding, _ = self.gru(X_batch)
        encoding = encoding.reshape(s[0], s[1]*self.output_dim)
        encoding = self.dense(encoding)
            
        for i, data in enumerate(data_batch):
            data = data.to(device)
            features = torch.zeros((data.num_nodes, 3 + encoding.shape[1])).to(device)
            features[:,:3] = data.x[:,:3]
            features[:,3:] = encoding[i,:]
            data.__setitem__('x', features)
        data_batch = Batch.from_data_list(data_batch)
        y_pred = self.gnn(data_batch.x, data_batch.edge_index)
        return y_pred.squeeze(), data_batch.y.squeeze()

class Regressor(nn.Module):
    def __init__(self, input_size=5, input_dim=256, hidden_dim=256, output_dim=256, n_layers=2):
        super().__init__()
        self.output_dim = output_dim
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
            nn.Linear(512, output_dim),
            nn.BatchNorm1d(output_dim),
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
            (geometric_nn.BatchNorm(3+512), 'x -> x'),
            (GATv2Conv(3+512, 128, heads=2, aggr='mean', dropout=0.5),  'x, edge_index -> x'), 
            (geometric_nn.BatchNorm(256), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(256, 128, aggr='mean'), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(128), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(128, 1, aggr='mean'), 'x, edge_index -> x'),
            ])

    def forward(self, X_batch, data_batch, device):
        s = X_batch.shape
        X_batch = X_batch.reshape(s[0]*s[1], s[2], s[3], s[4], s[5])
        X_batch = self.encoder(X_batch.to(device))
        X_batch = X_batch.reshape(s[0], s[1], self.output_dim)
        encoding, _ = self.gru(X_batch)
        encoding = encoding.reshape(s[0], s[1]*self.output_dim)
        encoding = self.dense(encoding)
            
        for i, data in enumerate(data_batch):
            data = data.to(device)
            features = torch.zeros((data.num_nodes, 3 + encoding.shape[1])).to(device)
            features[:,:3] = data.x[:,:3]
            features[:,3:] = encoding[i,:]
            data.__setitem__('x', features)
        data_batch = Batch.from_data_list(data_batch)
        y_pred = self.gnn(data_batch.x, data_batch.edge_index)
        mask = data_batch.mask.squeeze()
        return y_pred.squeeze()[mask], data_batch.y.squeeze()[mask]


class Classifier_test(Classifier):

    def __init__(self):
        super().__init__()

    def forward(self, X_batch, data_batch, device):
        s = X_batch.shape
        X_batch = X_batch.reshape(s[0]*s[1], s[2], s[3], s[4], s[5])
        X_batch = self.encoder(X_batch.to(device))
        X_batch = X_batch.reshape(s[0], s[1], self.output_dim)
        encoding, _ = self.gru(X_batch)
        encoding = encoding.reshape(s[0], s[1]*self.output_dim)
        encoding = self.dense(encoding)

        for i, data in enumerate(data_batch):
            data = data.to(device)
            features = torch.zeros((data.num_nodes, 3 + encoding.shape[1])).to(device)
            features[:,:3] = data.x[:,:3]
            features[:,3:] = encoding[i,:]
            data.__setitem__('x', features)
        data_batch = Batch.from_data_list(data_batch)
        y_pred = self.gnn(data_batch.x, data_batch.edge_index)
        prediction_class = torch.where(y_pred >= 0.5, 1, 0)              # focal loss 
        y = data_batch.y
        return prediction_class, y, data_batch.batch                             # focal loss


class Regressor_test(Regressor):

    def __init__(self):
        super().__init__()

    def forward(self, X_batch, data_batch, device):
        s = X_batch.shape
        X_batch = X_batch.reshape(s[0]*s[1], s[2], s[3], s[4], s[5])
        X_batch = self.encoder(X_batch.to(device))
        X_batch = X_batch.reshape(s[0], s[1], self.output_dim)
        encoding, _ = self.gru(X_batch)
        encoding = encoding.reshape(s[0], s[1]*self.output_dim)
        encoding = self.dense(encoding)

        for i, data in enumerate(data_batch):
            data = data.to(device)
            features = torch.zeros((data.num_nodes, 3 + encoding.shape[1])).to(device)
            features[:,:3] = data.x[:,:3]
            features[:,3:] = encoding[i,:]
            data.__setitem__('x', features)
        data_batch = Batch.from_data_list(data_batch)
        y_pred = self.gnn(data_batch.x, data_batch.edge_index)
        y_pred = torch.expm1(y_pred)                            # log1p
        y = data_batch.y
        return y_pred, y, data_batch.batch


class Regressor_test_large(Regressor):
    
    def __init__(self):
        super().__init__()

    def forward(self, data_batch):
        y_pred = torch.expm1(self.gnn(data_batch.x, data_batch.edge_index))
        return y_pred


class Classifier_test_large(Classifier):
    
    def __init__(self):
        super().__init__()

    def forward(self, data_batch):
        y_pred = self.gnn(data_batch.x, data_batch.edge_index)
        y_pred = torch.where(y_pred > 0.5, 1, 0) # focal loss
        return y_pred
