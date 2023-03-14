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
        self.hidden_dim = hidden_dim

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
        X = X.reshape(s[0], s[1], self.output_dim) # (64,25,256)
        out, h = self.gru(X)
        out = out.reshape(s[0]*s[1], self.output_dim) # (64,25*256) = (64,6400)
        out = self.decoder(out)
        out = out.reshape(s[0], s[1], s[2], s[3], s[4], s[5])
        return out

class Encoder(nn.Module):
    def __init__(self, input_size=5, input_dim=256, hidden_dim=256, output_dim=256, encoding_dim=128, n_layers=2):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
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
            nn.Linear(512, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
            )

        # define the decoder modules
        self.gru = nn.Sequential(
            nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        )

        self.dense = nn.Sequential(
            nn.Linear(self.hidden_dim*25, self.encoding_dim),
            nn.ReLU()
        )

    def forward(self, X):
        s = X.shape
        X = X.reshape(s[0]*s[1], s[2], s[3], s[4], s[5])
        X = self.encoder(X)
        X = X.reshape(s[0], s[1], self.output_dim) # (64,25,256)
        encoding, h = self.gru(X)
        encoding = encoding.reshape(s[0], s[1]*self.hidden_dim)
        encoding = self.dense(encoding)
        return encoding # (batch_size, encoding_dim)


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
        self.gnn = geometric_nn.Sequential('x, edge_index, edge_attr', [
            (geometric_nn.BatchNorm(1+512), 'x -> x'),
            (GATv2Conv(1+512, 128, heads=2, aggr='mean', dropout=0.5, edge_dim=2),  'x, edge_index, edge_attr -> x'),
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
        
    def forward(self, X_batch, data_batch, num_node_features=1, encoding_dim=512):
        s = X_batch.shape
        X_batch = X_batch.reshape(s[0]*s[1]*s[2], s[3], s[4], s[5], s[6])
        X_batch = self.encoder(X_batch) #.to(device))
        X_batch = X_batch.reshape(s[0]*s[1], s[2], self.output_dim)
        encoding, _ = self.gru(X_batch)
        encoding = encoding.reshape(s[0]*s[1], s[2]*self.hidden_dim)
        encoding = self.dense(encoding)
        encoding = encoding.reshape(s[0], s[1], encoding_dim)

        for i, data in enumerate(data_batch):
            features = torch.zeros((data.num_nodes, num_node_features + encoding.shape[-1])).cuda()
            features[:,0] = data.x
            for j, idx in enumerate(data.idx_list):
                features[data.low_res==idx,num_node_features:] = encoding[i,j,:]
            data.__setitem__('x', features)
        data_batch = Batch.from_data_list(data_batch)
        y_pred = self.gnn(data_batch.x, data_batch.edge_index, data_batch.edge_attr.float())
        train_mask = data_batch.train_mask
        return y_pred[train_mask].squeeze(), data_batch.y[train_mask]                           # focal loss
            #return y_pred, data_batch.y.squeeze().to(torch.long), data_batch.batch  # weighted cross entropy loss


class Regressor(nn.Module):
    def __init__(self, input_size=5, input_dim=256, hidden_dim=256, output_dim=256, encoding_dim=512, n_layers=2, num_node_features=1):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
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
            nn.Linear(512, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
            )

        self.gru = nn.Sequential(
            nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True),
        )

        self.dense = nn.Sequential(
            nn.Linear(hidden_dim*25, self.encoding_dim),
            nn.ReLU()
        )

        self.gnn = geometric_nn.Sequential('x, edge_index, edge_attr', [
            (geometric_nn.BatchNorm(num_node_features+self.encoding_dim), 'x -> x'),
            (GATv2Conv(num_node_features+encoding_dim, 128, heads=2, aggr='mean', dropout=0.5, edge_dim=2),  'x, edge_index, edge_attr -> x'), 
            (geometric_nn.BatchNorm(256), 'x -> x'),
            nn.ReLU(),                                                     
            (GATv2Conv(256, 128, aggr='mean', edge_dim=2), 'x, edge_index, edge_attr -> x'),
            (geometric_nn.BatchNorm(128), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(128, 1, aggr='mean', edge_dim=2), 'x, edge_index, edge_attr -> x'),
            ])

    def forward(self, X_batch, data_batch, num_node_features=1):
        s = X_batch.shape
        X_batch = X_batch.reshape(s[0]*s[1]*s[2], s[3], s[4], s[5], s[6])
        X_batch = self.encoder(X_batch) #.to(device))
        X_batch = X_batch.reshape(s[0]*s[1], s[2], self.output_dim)
        encoding, _ = self.gru(X_batch)
        encoding = encoding.reshape(s[0]*s[1], s[2]*self.hidden_dim)
        encoding = self.dense(encoding)
        encoding = encoding.reshape(s[0], s[1], self.encoding_dim)

        for i, data in enumerate(data_batch):
            features = torch.zeros((data.num_nodes, num_node_features + self.encoding_dim)).cuda()
            features[:,0] = data.x
            for j, idx in enumerate(data.idx_list):
                features[data.low_res==idx,num_node_features:] = encoding[i,j,:]
            data.__setitem__('x', features)
        data_batch = Batch.from_data_list(data_batch)
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
        prediction_class = torch.argmax(y_pred, dim=-1).squeeze() 
        return prediction_class, data_batch.y.squeeze().to(torch.long), data_batch.batch


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
        return y_pred.squeeze(), data_batch.y.squeeze(), data_batch.batch
