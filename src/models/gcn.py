import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool

class GCN(nn.Module):
    def __init__(self, in_features, hidden_features=64, num_classes=10, dropout=0.2):
        super(GCN, self).__init__()

        self.conv1 = GCNConv(in_features, hidden_features)
        self.conv2 = GCNConv(hidden_features, hidden_features*4)
        self.conv3 = GCNConv(hidden_features*4, hidden_features*8)

        self.dropout = nn.Dropout(p=0.5)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_features*16, hidden_features*8),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x1 = F.relu(self.conv1(x, edge_index))
        x2 = F.relu(self.conv2(x1, edge_index))
        x3 = F.relu(self.conv3(x2, edge_index))

        x_mean = global_mean_pool(x3, data.batch)
        x_max = global_max_pool(x3, data.batch)
        x = torch.cat([x_mean, x_max], dim=1)

        x = self.mlp(x)
        return x
