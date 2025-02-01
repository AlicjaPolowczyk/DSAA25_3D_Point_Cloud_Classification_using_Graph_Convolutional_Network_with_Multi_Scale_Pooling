import numpy as np
import torch
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data

def create_knn_graph(points, k=5):
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    edges = []
    edge_features = []
    for i in range(len(points)):
        for j in indices[i]:
            if i != j:
                edges.append((i, j))
                distance = torch.norm(torch.tensor(points[i]) - torch.tensor(points[j]), p=2)
                edge_features.append(distance)
    return edges, edge_features

def create_graph_features(data_reduced, k=5):
    graphs = []
    features = []
    edge_attrs = []
    for points in data_reduced:
        edges, edge_features = create_knn_graph(points, k)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float32).view(-1, 1)
        
        graphs.append(edge_index)
        edge_attrs.append(edge_attr)
        features.append(torch.tensor(points, dtype=torch.float))
    return graphs, features, edge_attrs


def create_gcn_dataset(features, graphs,edge_attrs, labels, device):
    data_list = []
    labels = torch.tensor(labels, dtype=torch.long)
    
    for i in range(len(features)):
        edge_index = graphs[i]
        edge_attr = edge_attrs[i]
        x = features[i]  
        y = labels[i]
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data_list.append(data)
    return data_list
