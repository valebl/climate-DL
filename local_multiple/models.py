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
        
    def forward(self, X_batch, data_batch, num_node_features=1):
        s = X_batch.shape
        X_batch = X_batch.reshape(s[0]*s[1]*s[2], s[3], s[4], s[5], s[6])   # (batch_dim*9*25, 5, 5, 6, 6)
        X_batch = self.encoder(X_batch)                                     # (batch_dim*9*25, cnn_output_dim)
        X_batch = X_batch.reshape(s[0]*s[1], s[2], self.cnn_output_dim)     # (batch_dim*9, 25, cnn_output_dim)
        encoding, _ = self.gru(X_batch)                                     # (batch_dim*9, 25, gru_hidden_dim)
        encoding = encoding.reshape(s[0], s[1], s[2]*self.gru_hidden_dim)   # (batch_dim, 9, 25*gru_hidden_dim)

        for i, data in enumerate(data_batch):
            features = torch.zeros((data.num_nodes, self.num_node_features + s[2]*self.gru_hidden_dim)).cuda()
            features[:,0] = data.x
            for j, idx in enumerate(data.idx_list):
                features[data.low_res==idx,self.num_node_features:] = encoding[i,j,:]
            data.__setitem__('x', features)
        data_batch = Batch.from_data_list(data_batch)
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
        self.time_encoder = 0
        self.time_gnn = 0
        self.time_features = 0
        self.time_tot = 0

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
        
    def forward(self, X_batch, data_batch, accelerator, step):
        t0 = time.time()
        s = X_batch.shape
        X_batch = X_batch.reshape(s[0]*s[1]*s[2], s[3], s[4], s[5], s[6])
        X_batch = self.encoder(X_batch)
        X_batch = X_batch.reshape(s[0]*s[1], s[2], self.cnn_output_dim)
        encoding, _ = self.gru(X_batch)
        encoding = encoding.reshape(s[0], s[1], s[2]*self.gru_hidden_dim)
        t1 = time.time()        
        # Create a mask that maps low_res indices to node indices
        node_mask = torch.zeros((data_batch.num_nodes,))
        node_mask[data_batch.low_res] = torch.arange(data_batch.num_nodes)
        node_mask = node_mask.long().cuda()
        
        # Create a node feature tensor with num_nodes rows and num_node_features+gru_hidden_dim*25 columns
        features = torch.zeros((data_batch.num_nodes, self.num_node_features + s[2]*self.gru_hidden_dim)).cuda()
        features[:, 0] = data_batch.z
        features[:, self.num_node_features:] = encoding.view(-1, s[2]*self.gru_hidden_dim)
        
        # Use node_mask to index into features and assign the correct encoding values
        mask = data_batch.low_res.view(-1, 1) == data_batch.idx_list.view(1, -1)
        mask = mask.float()
        node_encodings = torch.matmul(mask, features[:, self.num_node_features:])
        node_indices = node_mask[data_batch.low_res]
        features[node_indices, self.num_node_features:] = node_encodings
        
        # Update the node features in the original data batch
        data_batch.x = features
        data_batch = Batch.from_data_list(data_batch)
        t2 = time.time()
        y_pred = self.gnn(data_batch.x, data_batch.edge_index, data_batch.edge_attr.float())
        train_mask = data_batch.train_mask
        self.time_encoder += (t1-t0)
        self.time_features += (t2-t1)
        self.time_gnn += (t3-t2)
        self.time_tot += (t3-t0)
        if step == 200:
            if accelerator.is_main_process:
                print(f"Time totals: Total: {self.time_tot:.3f}s, Encoder: {self.time_encoder:.3f}s, Features: {self.time_features:.3f}s, GNN: {self.time_gnn:.3f}s")
                print(f"Time percentages: Encoder: {self.time_encoder/self.time_tot*100:.3f}%, Features: {self.time_features/self.time_tot*100:.3f}%, GNN: {self.time_gnn/self.time_tot*100:.3f}%")
        return y_pred[train_mask].squeeze(), data_batch.y[train_mask]

    # def forward(self, X_batch, data_batch, accelerator, step, num_node_features=1):
    #     t0 = time.time()
    #     s = X_batch.shape
    #     X_batch = X_batch.reshape(s[0]*s[1]*s[2], s[3], s[4], s[5], s[6])   # (batch_dim*9*25, 5, 5, 6, 6)
    #     X_batch = self.encoder(X_batch)                                     # (batch_dim*9*25, cnn_output_dim)
    #     X_batch = X_batch.reshape(s[0]*s[1], s[2], self.cnn_output_dim)     # (batch_dim*9, 25, cnn_output_dim)
    #     encoding, _ = self.gru(X_batch)                                     # (batch_dim*9, 25, gru_hidden_dim)
    #     encoding = encoding.reshape(s[0], s[1], s[2]*self.gru_hidden_dim)   # (batch_dim, 9, 25*gru_hidden_dim)
    #     t1 = time.time()
    #     for i, data in enumerate(data_batch):
    #         features = torch.zeros((data.num_nodes, self.num_node_features + s[2]*self.gru_hidden_dim))
    #         #print(features[:,0].shape, data_i.x.shape)
    #         features[:,0] = data.z
    #         features = features.cuda()
    #         for j, idx in enumerate(data.idx_list):
    #             features[data.low_res==idx,self.num_node_features:] = encoding[i,j,:]
    #         data['x'] = features
    #     data_batch = Batch.from_data_list(data_batch)
    #     t2 = time.time()
    #     y_pred = self.gnn(data_batch.x, data_batch.edge_index, data_batch.edge_attr.float())
    #     t3 = time.time()
    #     train_mask = data_batch.train_mask
    #     #print(f"Data.x.shape: {data_batch.x.shape}")
    #     self.time_encoder += (t1-t0)
    #     self.time_features += (t2-t1)
    #     self.time_gnn += (t3-t2)
    #     self.time_tot += (t3-t0)
    #     if step == 200:
    #         if accelerator.is_main_process:
    #             print(f"Time totals: Total: {self.time_tot:.3f}s, Encoder: {self.time_encoder:.3f}s, Features: {self.time_features:.3f}s, GNN: {self.time_gnn:.3f}s")
    #             print(f"Time percentages: Encoder: {self.time_encoder/self.time_tot*100:.3f}%, Features: {self.time_features/self.time_tot*100:.3f}%, GNN: {self.time_gnn/self.time_tot*100:.3f}%")
    #     #if accelerator.is_main_process:
    #     #    print(f"Time totals: Encoder: {(t1-t0):.3f}s, Features: {(t2-t1):.3f}s, GNN: {(t3-t2):.3f}s")
    #     #    print(f"Time percentages: Encoder: {(t1-t0)/(t3-t0)*100:.3f}%, Features: {(t2-t1)/(t3-t0)*100:.3f}%, GNN: {(t3-t2)/(t3-t0)*100:.3f}%")
    #     return y_pred[train_mask].squeeze(), data_batch.y[train_mask]              


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
    
    def forward(self, X_batch, data_batch, num_node_features=1):
        s = X_batch.shape
        X_batch = X_batch.reshape(s[0]*s[1]*s[2], s[3], s[4], s[5], s[6])   # (batch_dim*9*25, 5, 5, 6, 6)
        X_batch = self.encoder(X_batch)                                     # (batch_dim*9*25, cnn_output_dim)
        X_batch = X_batch.reshape(s[0]*s[1], s[2], self.cnn_output_dim)     # (batch_dim*9, 25, cnn_output_dim)
        encoding, _ = self.gru(X_batch)                                     # (batch_dim*9, 25, gru_hidden_dim)
        encoding = encoding.reshape(s[0], s[1], s[2]*self.gru_hidden_dim)   # (batch_dim, 9, 25*gru_hidden_dim)

        space_idxs = []
        time_idxs = []

        for i, data in enumerate(data_batch):
            features = torch.zeros((data.num_nodes, self.num_node_features + s[2]*self.gru_hidden_dim)).cuda()
            features[:,0] = data.x
            for j, idx in enumerate(data.idx_list):
                features[data.low_res==idx,self.num_node_features:] = encoding[i,j,:]
            data.__setitem__('x', features)
            _ = [space_idxs.append(s) for s in data.space_idxs]
            _ = [time_idxs.append(data.time_idx) for s in data.space_idxs]
        data_batch = Batch.from_data_list(data_batch)
        y_pred = self.gnn(data_batch.x.cuda(), data_batch.edge_index.cuda(), data_batch.edge_attr.float().cuda())
        test_mask = data_batch.test_mask
        return y_pred[test_mask].squeeze(), np.array(space_idxs), np.array(time_idxs)
    #return y_pred, data_batch.y.squeeze().to(torch.long), data_batch.batch  # weighted cross entropy loss


class Regressor_test(Regressor):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, X_batch, data_batch, num_node_features=1):
        s = X_batch.shape
        X_batch = X_batch.reshape(s[0]*s[1]*s[2], s[3], s[4], s[5], s[6])   # (batch_dim*9*25, 5, 5, 6, 6)
        X_batch = self.encoder(X_batch)                                     # (batch_dim*9*25, cnn_output_dim)
        X_batch = X_batch.reshape(s[0]*s[1], s[2], self.cnn_output_dim)     # (batch_dim*9, 25, cnn_output_dim)
        encoding, _ = self.gru(X_batch)                                     # (batch_dim*9, 25, gru_hidden_dim)
        encoding = encoding.reshape(s[0], s[1], s[2]*self.gru_hidden_dim)   # (batch_dim, 9, 25*gru_hidden_dim)

        for i, data in enumerate(data_batch):
            features = torch.zeros((data.num_nodes, self.num_node_features + s[2]*self.gru_hidden_dim)).cuda()
            features[:,0] = data.x
            for j, idx in enumerate(data.idx_list):
                features[data.low_res==idx,self.num_node_features:] = encoding[i,j,:]
            data.__setitem__('x', features)
        data_batch = Batch.from_data_list(data_batch)
        y_pred = self.gnn(data_batch.x.cuda(), data_batch.edge_index.cuda(), data_batch.edge_attr.float().cuda())
        test_mask = data_batch.test_mask
        return y_pred[test_mask].squeeze(), data_batch.space_idxs, np.array(data_batch.time_idx)    


if __name__ =='__main__':

    model = Regressor()
    batch_dim = 64
    input_batch = torch.rand((batch_dim, 25, 5, 5, 6, 6))

    start = time.time()
    X = model(input_batch)
    print(f"{time.time()-start} s\n")
    print(X.shape) 