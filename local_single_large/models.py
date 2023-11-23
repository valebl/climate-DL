import numpy as np
import torch
from torch import nn
from torch_geometric import nn as geometric_nn
from torch_geometric.nn import GATConv, GATv2Conv, GCNConv
from torch_geometric.data import Batch
import sys
import time
import copy

class Encoder(nn.Module):
    def __init__(self, input_size=5, cnn_output_dim=256, gru_input_dim=256, gru_hidden_dim=256, encoding_dim=128, n_layers=2):
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

    def forward(self, X):
        s = X.shape
        X = X.reshape(s[0]*s[1], s[2], s[3], s[4], s[5])                # (batch_dim*25, 5, 5, 6, 6)
        X = self.encoder(X)                                             # (batch_dim*25, cnn_output_dim)      
        X = X.reshape(s[0], s[1], self.cnn_output_dim)                  # (batch_dim, 25, cnn_output_dim)
        encoding, _ = self.gru(X) # out, h                              # (batch_dim, 25, gru_hidden_dim 
        encoding = encoding.reshape(s[0], s[1]*self.gru_hidden_dim)     # (batch_dim, 25*gru_hidden_dim)
        encoding = self.dense(encoding)                                 # (batch_dim, encoding_dim)
        return encoding


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
    def __init__(self, node_dim=1, encoding_dim=128):
        super().__init__()

        self.node_dim = node_dim
        self.encoding_dim = encoding_dim
    
        self.gnn = geometric_nn.Sequential('x, edge_index', [
            #(geometric_nn.BatchNorm(node_dim+encoding_dim), 'x -> x'),
            (GATv2Conv(node_dim+encoding_dim, 128, heads=2, aggr='mean', dropout=0.5),  'x, edge_index -> x'),
            #(geometric_nn.BatchNorm(256), 'x -> x'),
            #nn.ReLU(),
            (GATv2Conv(256, 128, aggr='mean'), 'x, edge_index -> x'),
            #(geometric_nn.BatchNorm(128), 'x -> x'),
            #nn.ReLU(),
            (GATv2Conv(128, 1, aggr='mean'), 'x, edge_index -> x'),
            nn.Sigmoid()
            ])
    
    def forward(self, graph):
        y_pred = self.gnn(graph.x, graph.edge_index)
        return y_pred.squeeze()[graph.train_mask], graph.y[graph.train_mask]


class Regressor_GNN(nn.Module):
    def __init__(self, node_dim=1, encoding_dim=128):
        super().__init__()

        self.node_dim = node_dim
        self.encoding_dim = encoding_dim
        
        self.gnn = geometric_nn.Sequential('x, edge_index', [
            (geometric_nn.BatchNorm(node_dim+encoding_dim), 'x -> x'),
            (GATv2Conv(node_dim+encoding_dim, 128, heads=2, aggr='mean', dropout=0.5),  'x, edge_index -> x'),
            (geometric_nn.BatchNorm(256), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(256, 128, aggr='mean'), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(128), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(128, 1, aggr='mean'), 'x, edge_index -> x'),
            ])
   
    def forward(self, graph):
        y_pred = self.gnn(graph.x, graph.edge_index)
        return y_pred.squeeze()[graph.train_mask], graph.y[graph.train_mask], graph.w.squeeze()[graph.train_mask]


#-------------------------------------------------------------------
#------------------------ GNN only models --------------------------
#-------------------------------------------------------------------

class Classifier_GNN_concat(nn.Module):
    def __init__(self, node_dim=1, encoding_dim=128):
        super().__init__()

        self.node_dim = node_dim
        self.encoding_dim = encoding_dim
    
        self.gnn = geometric_nn.Sequential('x, edge_index', [
            (geometric_nn.BatchNorm(node_dim+encoding_dim*25), 'x -> x'),
            (GATv2Conv(node_dim+encoding_dim*25, 512, heads=2, aggr='mean', dropout=0.5),  'x, edge_index -> x'),
            (geometric_nn.BatchNorm(1024), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(1024, 512, aggr='mean'), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(512), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(512, 512, aggr='mean'), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(512), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(512, 512, aggr='mean'), 'x, edge_index -> x'),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
            ])
    
    def forward(self, graph):
        y_pred = self.gnn(graph.x, graph.edge_index)
        return y_pred.squeeze()[graph.train_mask], graph.y[graph.train_mask]


class Regressor_GNN_concat(nn.Module):
    def __init__(self, node_dim=1, encoding_dim=128):
        super().__init__()

        self.node_dim = node_dim
        self.encoding_dim = encoding_dim
        
        self.gnn = geometric_nn.Sequential('x, edge_index', [
            (geometric_nn.BatchNorm(node_dim+encoding_dim*25), 'x -> x'),
            (GATv2Conv(node_dim+encoding_dim*25, 512, heads=2, aggr='mean', dropout=0.5),  'x, edge_index -> x'),
            (geometric_nn.BatchNorm(1024), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(1024, 512, aggr='mean'), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(512), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(512, 512, aggr='mean'), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(512), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(512, 512, aggr='mean'), 'x, edge_index -> x'),
            nn.ReLU(),
            nn.Linear(512, 1),
            ])
   
    def forward(self, graph):
        y_pred = self.gnn(graph.x, graph.edge_index)
        return y_pred.squeeze()[graph.train_mask], graph.y[graph.train_mask], graph.w.squeeze()[graph.train_mask]


class Regressor_GNN_concat_masked(Regressor_GNN_concat):
    def __init__(self, node_dim=1, encoding_dim=128):
        super().__init__()
    
    def forward(self, graph):
        y_pred = self.gnn(graph.x, graph.edge_index)
        return y_pred.squeeze(), graph.y, graph.w.squeeze()

class Classifier_GNN_concat_GCN(nn.Module):
    def __init__(self, node_dim=1, encoding_dim=128):
        super().__init__()

        self.node_dim = node_dim
        self.encoding_dim = encoding_dim
    
        self.gnn = geometric_nn.Sequential('x, edge_index', [
            (geometric_nn.BatchNorm(node_dim+encoding_dim*25), 'x -> x'),
            (GCNConv(node_dim+encoding_dim*25, 1024),  'x, edge_index -> x'),
            (geometric_nn.BatchNorm(1024), 'x -> x'),
            nn.ReLU(),
            (GCNConv(1024, 512), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(512), 'x -> x'),
            nn.ReLU(),
            (GCNConv(512, 512), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(512), 'x -> x'),
            nn.ReLU(),
            (GCNConv(512, 512), 'x, edge_index -> x'),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
            ])
    
    def forward(self, graph):
        y_pred = self.gnn(graph.x, graph.edge_index)
        return y_pred.squeeze()[graph.train_mask], graph.y[graph.train_mask]


class Regressor_GNN_concat_GCN(nn.Module):
    def __init__(self, node_dim=1, encoding_dim=128):
        super().__init__()

        self.node_dim = node_dim
        self.encoding_dim = encoding_dim
    
        self.gnn = geometric_nn.Sequential('x, edge_index', [
            (geometric_nn.BatchNorm(node_dim+encoding_dim*25), 'x -> x'),
            (GCNConv(node_dim+encoding_dim*25, 1024),  'x, edge_index -> x'),
            (geometric_nn.BatchNorm(1024), 'x -> x'),
            nn.ReLU(),
            (GCNConv(1024, 512), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(512), 'x -> x'),
            nn.ReLU(),
            (GCNConv(512, 512), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(512), 'x -> x'),
            nn.ReLU(),
            (GCNConv(512, 512), 'x, edge_index -> x'),
            nn.ReLU(),
            nn.Linear(512, 1),
            ])
    
    def forward(self, graph):
        y_pred = self.gnn(graph.x, graph.edge_index)
        return y_pred.squeeze()[graph.train_mask], graph.y[graph.train_mask], graph.w.squeeze()[graph.train_mask]


class Classifier_e_GNN_large(nn.Module):

    def __init__(self, input_size=5, cnn_output_dim=256, n_layers=2, input_dim=256, hidden_dim=256, node_dim=1, encoding_dim=512):
        super().__init__()
        
        self.cnn_output_dim = cnn_output_dim
        self.node_dim = node_dim
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
        self.gru = nn.Sequential(
            nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True),
            )
        self.dense = nn.Sequential(
            nn.Linear(hidden_dim*25, 512),
            nn.ReLU()
            ) 
        self.gnn = geometric_nn.Sequential('x, edge_index', [
            (geometric_nn.BatchNorm(node_dim+encoding_dim), 'x -> x'),
            (GATv2Conv(node_dim+encoding_dim, 128, heads=2, aggr='mean', dropout=0.5),  'x, edge_index -> x'),
            (geometric_nn.BatchNorm(256), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(256, 128, aggr='mean'), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(128), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(128, 1, aggr='mean'), 'x, edge_index -> x'),
            nn.Sigmoid()
            ])

    def forward(self, data_batch):
        encoding = self._forward_encoding(data_batch.input_data)
        y_pred, y = self._forward_gnn(data_batch, encoding)
        return y_pred, y

    def _forward_encoding(self, input_data):
        s = input_data.shape
        input_data = input_data.reshape(s[0]*s[1], s[2], s[3], s[4], s[5])                  # (batch_dim*25, 5, 5, 6, 6)
        input_data = self.encoder(input_data)                                               # (batch_dim*25, cnn_output_dim)
        input_data = input_data.reshape(s[0], s[1], self.cnn_output_dim)                    # (batch_dim, 25, cnn_output_dim)
        encoding, _ = self.gru(input_data)                                                  # (batch_dim*low_res_dim, 25, gru_hidden_dim)
        encoding = encoding.reshape(s[0], s[1]*self.cnn_output_dim)                         # (batch_dim* 25*gru_hidden_dim)
        encoding = self.dense(encoding)
        encoding = encoding.reshape(s[0], self.encoding_dim)                                # (batch_dim, 513)
        return encoding
    
    def _forward_gnn(self, data_batch, encoding):
        device = data_batch.input_data.device
        data_batch.x = torch.zeros((data_batch.num_nodes, self.node_dim+self.encoding_dim)).to(device)
        data_batch.x[:,:1] = data_batch.z
        for i, ki in enumerate(data_batch.k):
            space_idx = ki[0].item()
            time_idx = ki[1].item()
            mask_space = data_batch.low_res == space_idx       
            mask_time = data_batch.t_list == time_idx
            mask = mask_space*mask_time
            data_batch.x[mask,1:] = encoding[i].repeat(mask.sum(),1)

        y_pred = self.gnn(data_batch.x, data_batch.edge_index)
        mask_train_nodes = data_batch.train_mask
        return y_pred.squeeze()[mask_train_nodes], data_batch.y[mask_train_nodes]


class Regressor_e_GNN_large(nn.Module):

    def __init__(self, input_size=5, cnn_output_dim=256, n_layers=2, input_dim=256, hidden_dim=256, node_dim=1, encoding_dim=512):
        super().__init__()
        
        self.cnn_output_dim = cnn_output_dim
        self.node_dim = node_dim
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
        self.gru = nn.Sequential(
            nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True),
            )
        self.dense = nn.Sequential(
            nn.Linear(hidden_dim*25, encoding_dim),
            nn.ReLU()
            )
        self.gnn = geometric_nn.Sequential('x, edge_index', [
            (geometric_nn.BatchNorm(node_dim+encoding_dim), 'x -> x'),
            (GATv2Conv(node_dim+encoding_dim, 128, heads=2, aggr='mean', dropout=0.5),  'x, edge_index -> x'),
            (geometric_nn.BatchNorm(256), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(256, 128, aggr='mean'), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(128), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(128, 1, aggr='mean'), 'x, edge_index -> x'),
            ])

    def forward(self, data_batch):
        encoding = self._forward_encoding(data_batch.input_data)
        y_pred, y, w = self._forward_gnn(data_batch, encoding)
        return y_pred, y, w

    def _forward_encoding(self, input_data):
        s = input_data.shape
        input_data = input_data.reshape(s[0]*s[1], s[2], s[3], s[4], s[5])                  # (batch_dim*25, 5, 5, 6, 6)
        input_data = self.encoder(input_data)                                               # (batch_dim*25, cnn_output_dim)
        input_data = input_data.reshape(s[0], s[1], self.cnn_output_dim)                    # (batch_dim, 25, cnn_output_dim)
        encoding, _ = self.gru(input_data)                                                  # (batch_dim*low_res_dim, 25, gru_hidden_dim)
        encoding = encoding.reshape(s[0], s[1]*self.cnn_output_dim)                         # (batch_dim* 25*gru_hidden_dim)
        encoding = self.dense(encoding)
        encoding = encoding.reshape(s[0], self.encoding_dim)                                # (batch_dim, 513)
        return encoding
    
    def _forward_gnn(self, data_batch, encoding):
        device = data_batch.input_data.device
        data_batch.x = torch.zeros((data_batch.num_nodes, self.node_dim+self.encoding_dim)).to(device)
        data_batch.x[:,:1] = data_batch.z
        for i, ki in enumerate(data_batch.k):
            space_idx = ki[0].item()
            time_idx = ki[1].item()
            mask_space = data_batch.low_res == space_idx       
            mask_time = data_batch.t_list == time_idx
            mask = mask_space*mask_time
            data_batch.x[mask,1:] = encoding[i].repeat(mask.sum(),1)

        y_pred = self.gnn(data_batch.x, data_batch.edge_index)
        mask_train_nodes = data_batch.train_mask
        return y_pred.squeeze()[mask_train_nodes], data_batch.y[mask_train_nodes], data_batch.w[mask_train_nodes]


#----------------------------------------------
#---------------- test models -----------------
#----------------------------------------------


class Classifier_GNN_test(Classifier_GNN):
    def __init__(self, node_dim=1, encoding_dim=128):
        super().__init__()
    
    def forward(self, graph, G_test, time_index):
        y_pred = self._forward_gnn(graph)
        G_test['pr_cl'][:,time_index] = torch.where(y_pred > 0.5, 1.0, 0.0).cpu()
        return G_test

    def _forward_gnn(self, graph):
        y_pred = self.gnn(graph.x, graph.edge_index)
        return y_pred


class Regressor_GNN_test(Regressor_GNN):
    def __init__(self, node_dim=1, encoding_dim=128):
        super().__init__()

    def forward(self, graph, G_test, time_index):
        y_pred = self._forward_gnn(graph)
        G_test['pr_reg'][:,time_index] = torch.where(y_pred >= 0.1, y_pred, torch.tensor(0.0, dtype=y_pred.dtype)).cpu()
        return G_test

    def _forward_gnn(self, graph):
        y_pred = self.gnn(graph.x, graph.edge_index)
        y_pred = torch.expm1(y_pred)
        return y_pred


class Classifier_GNN_concat_test(Classifier_GNN_concat):
    def __init__(self, node_dim=1, encoding_dim=128):
        super().__init__()
    
    def forward(self, graph, G_test, time_index):
        y_pred = self._forward_gnn(graph)
        G_test['pr_cl'][:,time_index] = torch.where(y_pred > 0.5, 1.0, 0.0).cpu()
        return G_test

    def _forward_gnn(self, graph):
        y_pred = self.gnn(graph.x, graph.edge_index)
        return y_pred


class Regressor_GNN_concat_test(Regressor_GNN_concat):
    def __init__(self, node_dim=1, encoding_dim=128):
        super().__init__()

    def forward(self, graph, G_test, time_index):
        y_pred = self._forward_gnn(graph)
        G_test['pr_reg'][:,time_index] = torch.where(y_pred >= 0.1, y_pred, torch.tensor(0.0, dtype=y_pred.dtype)).cpu()
        return G_test

    def _forward_gnn(self, graph):
        y_pred = self.gnn(graph.x, graph.edge_index)
        y_pred = torch.expm1(y_pred)
        return y_pred


class Classifier_GNN_concat_GCN_test(Classifier_GNN_concat_GCN):
    def __init__(self, node_dim=1, encoding_dim=128):
        super().__init__()
    
    def forward(self, graph, G_test, time_index):
        y_pred = self._forward_gnn(graph)
        G_test['pr_cl'][:,time_index] = torch.where(y_pred > 0.5, 1.0, 0.0).cpu()
        return G_test

    def _forward_gnn(self, graph):
        y_pred = self.gnn(graph.x, graph.edge_index)
        return y_pred


class Regressor_GNN_concat_GCN_test(Regressor_GNN_concat_GCN):
    def __init__(self, node_dim=1, encoding_dim=128):
        super().__init__()

    def forward(self, graph, G_test, time_index):
        y_pred = self._forward_gnn(graph)
        G_test['pr_reg'][:,time_index] = torch.where(y_pred >= 0.1, y_pred, torch.tensor(0.0, dtype=y_pred.dtype)).cpu()
        return G_test

    def _forward_gnn(self, graph):
        y_pred = self.gnn(graph.x, graph.edge_index)
        y_pred = torch.expm1(y_pred)
        return y_pred


if __name__ =='__main__':

    model = Regressor_temporal()
    batch_dim = 64
    input_batch = torch.rand((25, 5, 5, 6, 6))

    start = time.time()
    X = model(input_batch)
    print(f"{time.time()-start} s\n")
    print(X.shape)

