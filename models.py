import numpy as np
import torch
from torch import nn
from torch_geometric import nn as geometric_nn
from torch_geometric.nn import GATConv, GATv2Conv
from torch_geometric.data import Batch
import sys

class Encoder(nn.Module):
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
            nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
            )

        self.dense = nn.Sequential(
            nn.Linear(hidden_dim*25, 512),
            nn.ReLU()
            )

    def forward(self, X_batch):
        s = X_batch.shape
        X_batch = X_batch.reshape(s[0]*s[1], s[2], s[3], s[4], s[5])
        X_batch = self.encoder(X_batch)
        X_batch = X_batch.reshape(s[0], s[1], self.output_dim)
        encoding, _ = self.gru(X_batch)
        encoding = encoding.reshape(s[0], s[1]*self.output_dim)
        encoding = self.dense(encoding)

        return encoding

class Regressor(Encoder):
    def __init__(self):
        super().__init__()

        self.gnn = geometric_nn.Sequential('x, edge_index', [
            (geometric_nn.BatchNorm(3+512), 'x -> x'),
            (GATv2Conv(3+512, 128, heads=2, aggr='mean', dropout=0.5),  'x, edge_index -> x'),
            (geometric_nn.BatchNorm(256), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(256, 128, heads=2, aggr='mean'), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(256), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(256, 128, aggr='mean'), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(128), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(128, 128, aggr='mean'), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(128), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(128, 128, aggr='mean'), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(128), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(128, 128, aggr='mean'), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(128), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(128, 1, aggr='mean'), 'x, edge_index -> x'),
            ])

    def forward(self, X_batch, data_batch, era5_to_gripho_list, device):
        s = X_batch.shape # (batch_dim, 420, 25, 5, 5, 6, 6)
        X_batch = X_batch.reshape(s[0]*s[1], s[2], s[3], s[4], s[5], s[6]) 
        s1 = X_batch.shape # (batch_dim*420, 25, 5, 5, 6, 6)
        X_batch = X_batch.reshape(s1[0]*s1[1], s1[2], s1[3], s1[4], s1[5]) # (batch_dim*420*25, 5, 5, 6, 6)
        X_batch = self.encoder(X_batch)
        X_batch = X_batch.reshape(s1[0], s1[1], self.output_dim) # (batch_dim*420, 25, output_dim)
        encoding, _ = self.gru(X_batch)
        encoding = encoding.reshape(s1[0], s1[1]*self.output_dim) # (bacth_dim*420, 25*output_dim)
        encoding = self.dense(encoding)
        encoding = encoding.reshape(s[0], s[1], encoding.shape[-1])

        s = encoding.shape
        features = torch.zeros((s[0], data_batch[0].x.shape[0], 3 + s[2])).to(device)
        features[:,:,:3] = data_batch[0].x[:,:]
        
        for cell_idx, mapping_idxs in enumerate(era5_to_gripho_list):
            features[:, mapping_idxs, 3:] = encoding[:, cell_idx, :].unsqueeze(1).repeat(repeats = (1, sum(mapping_idxs), 1))

        [data.__setitem__('x', features[i, :,:]) for i, data in enumerate(data_batch)]
        data_batch = Batch.from_data_list(data_batch)
        y_pred = torch.exp(self.gnn(data_batch.x, data_batch.edge_index))
        mask = data_batch.mask.squeeze()
        return y_pred.squeeze()[mask], data_batch.y.squeeze()[mask]
        sys.exit()

class Classifier(Encoder):
    def __init__():
        super().__init__()

        self.gnn = geometric_nn.Sequential('x, edge_index', [
            (geometric_nn.BatchNorm(3+512), 'x -> x'),
            (GATv2Conv(3+512, 128, heads=2, aggr='mean', dropout=0.5),  'x, edge_index -> x'),
            (geometric_nn.BatchNorm(256), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(256, 128, heads=2, aggr='mean'), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(256), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(256, 128, aggr='mean'), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(128), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(128, 128, aggr='mean'), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(128), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(128, 128, aggr='mean'), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(128), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(128, 128, aggr='mean'), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(128), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(128, 2, aggr='mean'), 'x, edge_index -> x'),
            nn.Softmax(dim=-1)
            ])

    def forward(self, X_batch, data_batch, era5_to_gripho_list, device):
        s = X_batch.shape
        X_batch = X_batch.reshape(s[0]*s[1], s[2], s[3], s[4], s[5])
        X_batch = self.encoder(X_batch)
        X_batch = X_batch.reshape(s[0], s[1], self.output_dim)
        encoding, _ = self.gru(X_batch)
        encoding = encoding.reshape(s[0], s[1]*self.output_dim)
        encoding = self.dense(encoding)

        features = torch.zeros((len(data_batch), data_batch[0].x.shape[0], 3 + encoding.shape[1])).to(device)
        features[:,:,:3] = torch.tensor(data_batch[0].x[:,:])

        for cell_idx, mapping_idxs in enumerate(era5_to_gripho_list):
            features[mapping_idxs, 3:] = encoding[cell_idx]

        #data_batch = Batch.from_data_list(data_batch)
        y_pred = self.gnn(data_batch.x, data_batch.edge_index)
        mask = data_batch.mask.squeeze()
        return y_pred[mask], data_batch.y[mask]


class Regressor_test(Regressor):
    def __init__(self):
        super().__init__()

    def forward(self, data_batch):
        y_pred = torch.exp(self.gnn(data_batch.x, data_batch.edge_index))
        return y_pred.squeeze()


class Classifier_test(Classifier):
    def __init__(self):
        super().__init__()

    def forward(self, data_batch):
        y_pred = self.gnn(data_batch.x, data_batch.edge_index)
        return y_pred

