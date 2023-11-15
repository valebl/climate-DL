import numpy as np
import torch
from torch import nn
from torch_geometric import nn as geometric_nn
from torch_geometric.nn import GATConv, GATv2Conv
from torch_geometric.data import Batch
import sys
import time
import copy


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
            nn.Linear(2048, encoding_dim),
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

    
#-------------------------------------------------------------------
#------------------------ GNN only models --------------------------
#-------------------------------------------------------------------

class Classifier_GNN(nn.Module):
    def __init__(self, node_dim=1, encoding_dim=12, edge_attr_dim=1):
        super().__init__()

        self.node_dim = node_dim
        self.encoding_dim = encoding_dim
    
        self.gnn = geometric_nn.Sequential('x, edge_index, edge_attr', [
            (geometric_nn.BatchNorm(node_dim+encoding_dim), 'x -> x'),
            (GATv2Conv(node_dim+encoding_dim, 128, heads=2, aggr='mean', dropout=0.5, edge_dim=edge_attr_dim),  'x, edge_index, edge_attr -> x'),
            (geometric_nn.BatchNorm(256), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(256, 128, aggr='mean', edge_dim=edge_attr_dim), 'x, edge_index, edge_attr -> x'),
            (geometric_nn.BatchNorm(128), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(128, 1, aggr='mean', edge_dim=edge_attr_dim), 'x, edge_index, edge_attr -> x'),
            nn.Sigmoid()
            ])

    #def forward(self, graph):
    #    y_pred, y = self._forward_gnn(graph)
    #    return y_pred, y
    
    def forward(self, graph):
        y_pred = self.gnn(graph.x, graph.edge_index, graph.edge_attr)
        return y_pred.squeeze()[graph.train_mask], graph.y[graph.train_mask]


class Regressor_GNN(nn.Module):
    def __init__(self, node_dim=1, encoding_dim=128, edge_attr_dim=1):
        super().__init__()

        self.node_dim = node_dim
        self.encoding_dim = encoding_dim
        
        self.gnn = geometric_nn.Sequential('x, edge_index, edge_attr', [
            (geometric_nn.BatchNorm(node_dim+encoding_dim), 'x -> x'),
            (GATv2Conv(node_dim+encoding_dim, 128, heads=2, aggr='mean', dropout=0.5, edge_dim=edge_attr_dim),  'x, edge_index, edge_attr -> x'),
            (geometric_nn.BatchNorm(256), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(256, 128, aggr='mean', edge_dim=edge_attr_dim), 'x, edge_index, edge_attr -> x'),
            (geometric_nn.BatchNorm(128), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(128, 1, aggr='mean', edge_dim=edge_attr_dim), 'x, edge_index, edge_attr -> x'),
            ])
   
    #def forward(self, graph):
    #    y_pred, y, w = self._forward_gnn(graph)
    #    return y_pred, y, w
    
    def forward(self, graph):
        y_pred = self.gnn(graph.x, graph.edge_index, graph.edge_attr)
        return y_pred.squeeze()[graph.train_mask], graph.y[graph.train_mask], graph.w.squeeze()[graph.train_mask]


#----------------------------------------------
#---------------- test models -----------------
#----------------------------------------------


class Classifier_GNN_test(Classifier_GNN):
    def __init__(self, node_dim=1, encoding_dim=128, edge_attr_dim=1):
        super().__init__()
    
    def forward(self, graph, G_test, time_index):
        y_pred = self._forward_gnn(graph)
        G_test['pr_cl'][:,time_index] = torch.where(y_pred > 0.5, 1.0, 0.0).cpu()
        return G_test

    def _forward_gnn(self, graph):
        y_pred = self.tgnn(graph.x, graph.edge_index, graph.edge_attr_dim)
        return y_pred.squeeze()


class Regressor_GNN_test(Regressor_GNN):
    def __init__(self, node_dim=1, encoding_dim=128, edge_attr_dim=1):
        super().__init__()

    def forward(self, graph, G_test, time_index):
        y_pred = self._forward_gnn(graph)
        G_test['pr_reg'][:,time_index] = torch.where(y_pred >= 0.1, y_pred, torch.tensor(0.0, dtype=y_pred.dtype)).cpu()
        return G_test

    def _forward_gnn(self, graph):
        y_pred = self.tgnn(graph.x, graph.edge_index, graph.edge_attr_dim)
        y_pred = torch.expm1(y_pred)
        return y_pred.squeeze()

if __name__ =='__main__':

    model = Regressor_temporal()
    batch_dim = 64
    input_batch = torch.rand((25, 5, 5, 6, 6))

    start = time.time()
    X = model(input_batch)
    print(f"{time.time()-start} s\n")
    print(X.shape)

