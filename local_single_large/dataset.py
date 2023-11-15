import pickle
import sys
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_convert

import time

from torch_geometric.data import Data

import torch
import numpy as np
from typing import Sequence, Union
from torch_geometric.data import Data, Batch
from torch_geometric.utils import k_hop_subgraph

Edge_Index = Union[np.ndarray, None]
Edge_Weight = Union[np.ndarray, None]
Node_Features = Sequence[Union[np.ndarray, None]]
Targets = Sequence[Union[np.ndarray, None]]
Z = Sequence[Union[torch.tensor, None]]
Encodings = Sequence[Union[list, None]]
Low_Res = Sequence[Union[torch.tensor, None]]
Additional_Features = Sequence[np.ndarray]

class Dataset_StaticGraphTemporalSignal(Dataset):
    r"""A data iterator object to contain a static graph with a dynamically
    changing constant time difference temporal feature set (multiple signals).
    The node labels (target) are also temporal. The iterator returns a single
    constant time difference temporal snapshot for a time period (e.g. day or week).
    This single temporal snapshot is a Pytorch Geometric Data object. Between two
    temporal snapshots the features and optionally passed attributes might change.
    However, the underlying graph is the same.

    Args:
        edge_index (Numpy array): Index tensor of edges.
        edge_weight (Numpy array): Edge weight tensor.
        features (Sequence of Numpy arrays): Sequence of node feature tensors.
        targets (Sequence of Numpy arrays): Sequence of node label (target) tensors.
        **kwargs (optional Sequence of Numpy arrays): Sequence of additional attributes.
    """

    def __init__(
        self,
        edge_index: Edge_Index,
        edge_weight: Edge_Weight,
        features: Node_Features,
        targets: Targets,
        encodings: Encodings,
        z: Z,
        low_res: Low_Res,
        **kwargs: Additional_Features
    ):
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.features = features
        self.targets = targets
        self.encodings = encodings
        self.z = z
        #self.z = self.z.unsqueeze(0).repeat(25,1,1)
        self.low_res = low_res
        self.additional_feature_keys = []
        for key, value in kwargs.items():
            setattr(self, key, value)
            self.additional_feature_keys.append(key)
#        self._check_temporal_consistency()
        self.length = self.__len__()
        self._set_snapshot_count()

    def __len__(self):
        #return len(self.features)
        return len(self.targets)
                                                                                                                                                            
    def _check_temporal_consistency(self):
        assert len(self.features) == len(
            self.targets
        ), "Temporal dimension inconsistency."
        for key in self.additional_feature_keys:
            assert len(self.targets) == len(
                getattr(self, key)
            ), "Temporal dimension inconsistency."

    def _set_snapshot_count(self):
        self.snapshot_count = self.length #len(self.features)

    def _get_edge_index(self):
        if self.edge_index is None:
            return self.edge_index
        else:
            return torch.LongTensor(self.edge_index)
    
    def _get_edge_weight(self):
        if self.edge_weight is None:
            return self.edge_weight
        else:
            return torch.FloatTensor(self.edge_weight)

    def _get_features(self, time_index: int):
#        if self.features[time_index] is None:      
#            return self.features[time_index]      
#        else:
        x = torch.zeros((len(self.low_res), 1 + self.encodings[-1].shape[1]))
        for idx in self.low_res.unique():
            idx_mask = self.low_res == idx
            encoding = self.encodings[idx][time_index,:].unsqueeze(0).repeat(idx_mask.sum(),1)
            x[idx_mask,1:] = encoding
        x[:,0] = self.z[:,0]
            #x = torch.FloatTensor(self.features[time_index-24:time_index+1])
        #x = x.swapaxes(0,1)
        #x = x.swapaxes(1,2)
        return x

    def _get_target(self, time_index: int):
        if self.targets[time_index] is None:
            return self.targets[time_index]
        else:
            if self.targets[time_index].dtype.kind == "i":
                return torch.LongTensor(self.targets[time_index])
            elif self.targets[time_index].dtype.kind == "f":
                return torch.FloatTensor(self.targets[time_index])

    def _get_additional_feature(self, time_index: int, feature_key: str):
        feature = getattr(self, feature_key)[time_index]
        if feature.dtype.kind == "i":
            return torch.LongTensor(feature)
        elif feature.dtype.kind == "f":
            return torch.FloatTensor(feature)
        elif feature.dtype.kind == "b":
            return torch.BoolTensor(feature)

    def _get_additional_features(self, time_index: int):
        additional_features = {
            key: self._get_additional_feature(time_index, key)
            for key in self.additional_feature_keys
        }
        return additional_features

    def __getitem__(self, time_index: int):
        x = self._get_features(time_index)
        edge_index = self._get_edge_index()
        edge_weight = self._get_edge_weight()
        y = self._get_target(time_index)
        additional_features = self._get_additional_features(time_index)

        snapshot = Data(x=x, edge_index=edge_index, edge_attr=edge_weight,
                y=y, time_index=time_index, **additional_features)

        #return snapshot

        node_idx = torch.randint(high=snapshot.num_nodes, size=(1,)).item()
        num_hops = torch.randint(low=20, high=30, size=(1,)).item()

        subset, _, _, _ = k_hop_subgraph(node_idx=node_idx, num_hops=num_hops, edge_index=snapshot.edge_index, relabel_nodes=False)
        s = snapshot.subgraph(subset=subset)

        return s


class Iterable_StaticGraphTemporalSignal(object):

    def __init__(self, static_graph_temporal_signal, shuffle):
        self.static_graph_temporal_signal = static_graph_temporal_signal
        self.shuffle = shuffle
        if self.shuffle:
            self.sampling_vector = torch.randperm(len(self)-24) + 24
        else:
            self.sampling_vector = torch.arange(24, len(self))

    def __len__(self):
        return len(self.static_graph_temporal_signal)
    
    def __next__(self):
        if self.t < len(self) - 24:
            self.idx = self.sampling_vector[self.t].item()
            self.t = self.t + 1
            return self.idx
        else:
            self.t = 0
            self.idx = 0
            raise StopIteration

    def __iter__(self):
        self.t = 0
        self.idx = 0
        return self
        

#################################################################################

class Dataset_pr(Dataset):

    def __init__(self, args, lat_dim, lon_dim, pad=2):
        super().__init__()
        self.pad = pad
        self.lon_low_res_dim = lon_dim
        self.lat_low_res_dim = lat_dim
        self.space_low_res_dim = self.lat_low_res_dim * self.lon_low_res_dim
        self.args = args
        self.length = None
    
    def _load_data_into_memory(self):
        raise NotImplementedError

    def __len__(self):
        return self.length


class Dataset_pr_ae(Dataset_pr):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input, self.idx_to_key = self._load_data_into_memory()

    def _load_data_into_memory(self):
        with open(self.args.input_path + self.args.input_file, 'rb') as f:
            input = pickle.load(f)
        with open(self.args.input_path + self.args.idx_file,'rb') as f:
            idx_to_key = pickle.load(f)
        self.length = len(idx_to_key)
        return input, idx_to_key

    def __getitem__(self, idx):
        k = self.idx_to_key[idx]
        time_idx = k // self.space_low_res_dim
        space_idx = k % self.space_low_res_dim
        lat_idx = space_idx // self.lon_low_res_dim
        lon_idx = space_idx % self.lon_low_res_dim
        input = torch.zeros((25, 5, 5, 6, 6))
        input[:] = self.input[time_idx - 24 : time_idx+1, :, :, lat_idx - self.pad + 2 : lat_idx + self.pad + 4, lon_idx - self.pad + 2 : lon_idx + self.pad + 4]
        return input


class Dataset_pr_ae_space(Dataset_pr_ae):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        k = self.idx_to_key[idx]
        time_idx = k // self.space_low_res_dim
        space_idx = k % self.space_low_res_dim
        lat_idx = space_idx // self.lon_low_res_dim
        lon_idx = space_idx % self.lon_low_res_dim
        input = torch.zeros((5, 5, 6, 6))
        input[:] = self.input[time_idx, :, :, lat_idx - self.pad + 2 : lat_idx + self.pad + 4, lon_idx - self.pad + 2 : lon_idx + self.pad + 4]
        return input


class Dataset_pr_e_space(Dataset_pr):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input = self._load_data_into_memory()    

    def _load_data_into_memory(self):
        with open(self.args.input_path + self.args.input_file, 'rb') as f:
            input = pickle.load(f)
        return input

    def __getitem__(self, idx):
        k = idx
        time_idx = k // self.space_low_res_dim
        space_idx = k % self.space_low_res_dim
        lat_idx = space_idx // self.lon_low_res_dim
        lon_idx = space_idx % self.lon_low_res_dim
        input = torch.zeros((5, 5, 6, 6))
        input[:] = self.input[time_idx, :, :, lat_idx - self.pad + 2 : lat_idx + self.pad + 4, lon_idx - self.pad + 2 : lon_idx + self.pad + 4]
        return input


#################################################################################

def custom_collate_fn_ae(batch):
    input = torch.stack(batch)
    input = default_convert(input)
    return input

def custom_collate_fn_gnn(batch):
    input = torch.stack([item[0] for item in batch])                        # shape = (batch_size, 25, 5, 5, 6, 6)
    data = [item[1] for item in batch]
    input = default_convert(input)
    return input, data

def custom_collate_fn_gnn_large(batch):
    return Batch.from_data_list(batch)

