import numpy as np
import torch
from torch import nn
from torch_geometric import nn as geometric_nn
from torch_geometric.nn import GATConv, GATv2Conv
from torch_geometric.data import Batch
import sys
import time
import copy
from torch_geometric_temporal.nn.recurrent import A3TGCN
from torch_geometric_temporal.nn.attention import MTGNN
from torch_geometric.nn import GCNConv
#from .temporalgcn import TGCN


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

#------------------------------------    
    
class TGCN_mod(nn.Module):
    r"""An implementation of the Temporal Graph Convolutional Gated Recurrent Cell.
    For details see this paper: `"T-GCN: A Temporal Graph ConvolutionalNetwork for
    Traffic Prediction." <https://arxiv.org/abs/1811.05320>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        improved (bool): Stronger self loops. Default is False.
        cached (bool): Caching the message weights. Default is False.
        add_self_loops (bool): Adding self-loops for smoothing. Default is True.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        node_dim: int
    ):
        super(TGCN_mod, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.node_dim = node_dim
        self._create_parameters_and_layers()

    def _create_update_gate_parameters_and_layers(self):

        self.conv_z = geometric_nn.Sequential('x, edge_index', [
            (geometric_nn.BatchNorm(self.node_dim+self.in_channels), 'x -> x'),
            (GATv2Conv(self.node_dim+self.in_channels, self.out_channels, heads=2, aggr='mean', dropout=0.5),  'x, edge_index -> x'),
            (geometric_nn.BatchNorm(self.out_channels*2), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(self.out_channels*2, self.out_channels, aggr='mean'), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(self.out_channels), 'x -> x'),
            nn.ReLU()
            ])
        
        self.linear_z = nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_reset_gate_parameters_and_layers(self):

        self.conv_r = geometric_nn.Sequential('x, edge_index', [
            (geometric_nn.BatchNorm(self.node_dim+self.in_channels), 'x -> x'),
            (GATv2Conv(self.node_dim+self.in_channels, self.out_channels, heads=2, aggr='mean', dropout=0.5),  'x, edge_index -> x'),
            (geometric_nn.BatchNorm(self.out_channels*2), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(self.out_channels*2, self.out_channels, aggr='mean'), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(self.out_channels), 'x -> x'),
            nn.ReLU()
            ])

        self.linear_r = nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_candidate_state_parameters_and_layers(self):

        self.conv_h = geometric_nn.Sequential('x, edge_index', [
            (geometric_nn.BatchNorm(self.node_dim+self.in_channels), 'x -> x'),
            (GATv2Conv(self.node_dim+self.in_channels, self.out_channels, heads=2, aggr='mean', dropout=0.5),  'x, edge_index -> x'),
            (geometric_nn.BatchNorm(self.out_channels*2), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(self.out_channels*2, self.out_channels, aggr='mean'), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(self.out_channels), 'x -> x'),
            nn.ReLU()
            ])

        self.linear_h = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return H

    def _calculate_update_gate(self, X, edge_index, edge_weight, H):
        Z = torch.cat([self.conv_z(X, edge_index), H], axis=1)
        Z = self.linear_z(Z)
        Z = torch.sigmoid(Z)
        return Z

    def _calculate_reset_gate(self, X, edge_index, edge_weight, H):
        R = torch.cat([self.conv_r(X, edge_index), H], axis=1)
        R = self.linear_r(R)
        R = torch.sigmoid(R)
        return R

    def _calculate_candidate_state(self, X, edge_index, edge_weight, H, R):
        H_tilde = torch.cat([self.conv_h(X, edge_index), H * R], axis=1)
        H_tilde = self.linear_h(H_tilde)
        H_tilde = torch.tanh(H_tilde)
        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde
        return H

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.

        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
        """
        H = self._set_hidden_state(X, H)
        Z = self._calculate_update_gate(X, edge_index, edge_weight, H)
        R = self._calculate_reset_gate(X, edge_index, edge_weight, H)
        H_tilde = self._calculate_candidate_state(X, edge_index, edge_weight, H, R)
        H = self._calculate_hidden_state(Z, H, H_tilde)
        return H

#---------------------------------------------    

class A3TGCN_mod(nn.Module):
    r"""An implementation of the Attention Temporal Graph Convolutional Cell.
    For details see this paper: `"A3T-GCN: Attention Temporal Graph Convolutional
    Network for Traffic Forecasting." <https://arxiv.org/abs/2006.11583>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        periods (int): Number of time periods.
        improved (bool): Stronger self loops (default :obj:`False`).
        cached (bool): Caching the message weights (default :obj:`False`).
        add_self_loops (bool): Adding self-loops for smoothing (default :obj:`True`).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        periods: int,
        node_dim: int,
    ):
        super(A3TGCN_mod, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.periods = periods
        self.node_dim = node_dim
        self._setup_layers()

    def _setup_layers(self):
        self._base_tgcn = TGCN_mod(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            node_dim=self.node_dim
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._attention = torch.nn.Parameter(torch.empty(self.periods, device=device))
        torch.nn.init.uniform_(self._attention)

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.

        Arg types:
            * **X** (PyTorch Float Tensor): Node features for T time periods.
            * **edge_index** (PyTorch Long Tensor): Graph edge indices.
            * **edge_weight** (PyTorch Long Tensor, optional)*: Edge weight vector.
            * **H** (PyTorch Float Tensor, optional): Hidden state matrix for all nodes.

        Return types:
            * **H** (PyTorch Float Tensor): Hidden state matrix for all nodes.
        """
        H_accum = 0
        probs = torch.nn.functional.softmax(self._attention, dim=0)
        for period in range(self.periods):
            H_accum = H_accum + probs[period] * self._base_tgcn(
                X[:, :, period], edge_index, edge_weight, H
            )
        return H_accum

#----------------------------------------------
#-------------- edge attributes ---------------
#----------------------------------------------

class Classifier_temporal(nn.Module):
    def __init__(self, node_dim=1, encoding_dim=128, periods=25):
        super().__init__()

        self.node_dim = node_dim
        self.encoding_dim = encoding_dim
        
        self.tgnn = geometric_nn.Sequential('x, edge_index', [
            #(A3TGCN_mod(in_channels=self.encoding_dim,node_dim=self.node_dim, out_channels=64, periods=periods),  'x, edge_index -> x'),
            (A3TGCN(in_channels=self.encoding_dim+self.node_dim, out_channels=64, periods=periods),  'x, edge_index -> x'),
            nn.Linear(64, 1),
            nn.Sigmoid()
            ])
    
    def forward(self, graph):
        y_pred, y = self._forward_gnn(graph)
        return y_pred, y
    
    def _forward_gnn(self, graph):
        y_pred = self.tgnn(graph.x, graph.edge_index)
        return y_pred.squeeze()[graph.train_mask], graph.y[graph.train_mask]


class Regressor_temporal(nn.Module):
    def __init__(self, node_dim=1, encoding_dim=128, periods=25):
        super().__init__()

        self.node_dim = node_dim
        self.encoding_dim = encoding_dim
        
        self.tgnn = geometric_nn.Sequential('x, edge_index', [
            #(A3TGCN_mod(in_channels=self.encoding_dim, node_dim=self.node_dim, out_channels=64, periods=periods),  'x, edge_index -> x'),
            (A3TGCN(in_channels=self.encoding_dim+self.node_dim, out_channels=64, periods=periods),  'x, edge_index -> x'),
            nn.Linear(64, 1),
            ])
    
    def forward(self, graph):
        y_pred, y, w = self._forward_gnn(graph)
        return y_pred, y, w
    
    def _forward_gnn(self, graph):
        y_pred = self.tgnn(graph.x, graph.edge_index)
        return y_pred.squeeze()[graph.train_mask], graph.y[graph.train_mask], graph.w.squeeze()[graph.train_mask]

#----------------------------------------------
#----------------------------------------------
#----------------------------------------------

# MTGNN
 # """An implementation of the Multivariate Time Series Forecasting Graph Neural Networks.
 #    For details see this paper: `"Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks."
 #    <https://arxiv.org/pdf/2005.11650.pdf>`_

 #    Args:
 #    gcn_true (bool): Whether to add graph convolution layer.
 #    build_adj (bool): Whether to construct adaptive adjacency matrix.
 #    gcn_depth (int): Graph convolution depth.
 #    num_nodes (int): Number of nodes in the graph.
 #    kernel_set (list of int): List of kernel sizes.
 #    kernel_size (int): Size of kernel for convolution, to calculate receptive field size.
 #    dropout (float): Droupout rate.
 #    subgraph_size (int): Size of subgraph.
 #    node_dim (int): Dimension of nodes.
 #    dilation_exponential (int): Dilation exponential.
 #    conv_channels (int): Convolution channels.
 #    residual_channels (int): Residual channels.
 #    skip_channels (int): Skip channels.
 #    end_channels (int): End channels.
 #    seq_length (int): Length of input sequence.
 #    in_dim (int): Input dimension.
 #    out_dim (int): Output dimension.
 #    layers (int): Number of layers.
 #    propalpha (float): Prop alpha, ratio of retaining the root nodes's original states in mix-hop propagation, a value between 0 and 1.
 #    tanhalpha (float): Tanh alpha for generating adjacency matrix, alpha controls the saturation rate.
 #    layer_norm_affline (bool): Whether to do elementwise affine in Layer Normalization.
 #    xd (int, optional): Static feature dimension, default None.
 #    """

class Classifier_temporal_MTGNN(nn.Module):
    def __init__(self, static_features=1, encoding_dim=128, periods=25, num_nodes=1259, gcn_depth=3):
        super().__init__()

        self.gcn_true =True
        self.build_adj = False
        self.gcn_depth = gcn_depth
        self.num_nodes = num_nodes
        self.kernel_set = [3,3,3]
        self.kernel_size = 10
        self.dropout = 0.1
        self.subgraph_size = self.num_nodes
        self.node_dim = static_features_dim + encoding_dim
        self.dilation_exponential = 2
        self.conv_channels = 128
        self.residual_channels = 128
        self.skip_channels = 128
        self.end_channels = 128
        self.seq_length = 25
        self.in_dim = 129
        self.out_dim = 1
        self.layers = 1
        self.propalpha = 0.5
        self.tanhalpha = 1
        self.layer_norm_affline = False
        self.xd = 1

        self.tgnn = geometric_nn.Sequential('x, edge_index', [
            (MTGNN(gcn_true=self.gcn_true, build_adj=self.build_adj, gcn_depth=self.gcn_depth, num_nodes=self.num_nodes, kernel_set=self.kernel_set,
                   kernel_size=self.kernel_size, dropout=self.dropout, subgraph_size=self.subgraph_size, node_dim=self.node_dim, 
                   dilation_exponential=self.dilation_exponential, conv_channels=self.conv_channels, residual_channels=self.residual_channels,
                   skip_channels=self.skip_channels, end_channels=self.end_channels, sequ_length=self.seq_length, in_dim=self.in_dim,
                   out_dim=self.out_dim, layers=self.layers, propalpha=self.propalpha, tanhalpha=self.tanhalpha, layer_norm_affline=self.layer_norm_affline,
                   xd=self.xd),  'x, edge_index -> x'),
            nn.Linear(64, 1),
            nn.Sigmoid()
            ])
    
    def forward(self, graph):
        y_pred, y = self._forward_gnn(graph)
        return y_pred, y
    
    def _forward_gnn(self, graph):
        y_pred = self.tgnn(graph.x, graph.edge_index)
        return y_pred.squeeze()[graph.train_mask], graph.y[graph.train_mask]


class Regressor_temporal_MTGNN(nn.Module):
    def __init__(self, node_dim=1, encoding_dim=128, periods=25):
        super().__init__()

        self.node_dim = node_dim
        self.encoding_dim = encoding_dim
        
        self.tgnn = geometric_nn.Sequential('x, edge_index', [
            (MTGNN(in_channels=self.encoding_dim, node_dim=self.node_dim, out_channels=64, periods=periods),  'x, edge_index -> x'),
            nn.Linear(64, 1),
            ])
    
    def forward(self, graph):
        y_pred, y, w = self._forward_gnn(graph)
        return y_pred, y, w
    
    def _forward_gnn(self, graph):
        y_pred = self.tgnn(graph.x, graph.edge_index)
        return y_pred.squeeze()[graph.train_mask], graph.y[graph.train_mask], graph.w.squeeze()[graph.train_mask]



#----------------------------------------------
#---------------- test models -----------------
#----------------------------------------------

class Classifier_temporal_test(Classifier_temporal):
    def __init__(self, node_dim=1, encoding_dim=128, periods=25):
        super().__init__()
    
    def forward(self, graph, G_test, time_index):
        y_pred = self._forward_gnn(graph)
        G_test['pr_cl'][:,time_index] = torch.where(y_pred > 0.5, 1.0, 0.0).cpu()
        return G_test

    def _forward_gnn(self, graph):
        y_pred = self.tgnn(graph.x, graph.edge_index)
        return y_pred.squeeze()

class Regressor_temporal_test(Regressor_temporal):
    def __init__(self, node_dim=1, encoding_dim=128, periods=25):
        super().__init__()

    def forward(self, graph, G_test, time_index):
        y_pred = self._forward_gnn(graph)
        G_test['pr_reg'][:,time_index] = torch.where(y_pred >= 0.1, y_pred, torch.tensor(0.0, dtype=y_pred.dtype)).cpu()
        return G_test

    def _forward_gnn(self, graph):
        y_pred = self.tgnn(graph.x, graph.edge_index)
        return y_pred.squeeze()


if __name__ =='__main__':

    model = Regressor_temporal()
    batch_dim = 64
    input_batch = torch.rand((25, 5, 5, 6, 6))

    start = time.time()
    X = model(input_batch)
    print(f"{time.time()-start} s\n")
    print(X.shape)

