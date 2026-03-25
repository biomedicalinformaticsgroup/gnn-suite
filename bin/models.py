import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, GCN2Conv, TransformerConv


class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_hidden=16, num_layers=2, dropout=0.5):
        super(GCN, self).__init__()

        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(num_features, num_hidden))
        self.dropout = dropout

        for _ in range(num_layers - 2):  # minus 2 to account for the first and last layers
            self.convs.append(GCNConv(num_hidden, num_hidden))
        self.convs.append(GCNConv(num_hidden, num_classes))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.elu(x)
            x = F.dropout(x,p=self.dropout, training=self.training)

        x = self.convs[self.num_layers - 1](x, edge_index)

        return x



class GAT(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_hidden=16, num_heads=1, num_layers=2, dropout=0.5):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()  # Batch Normalization layers
        
        self.convs.append(GATConv(num_features, num_hidden, heads=num_heads))
        self.bns.append(torch.nn.BatchNorm1d(num_heads * num_hidden))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(num_heads * num_hidden, num_hidden, heads=num_heads))
            self.bns.append(torch.nn.BatchNorm1d(num_heads * num_hidden))
        self.convs.append(GATConv(num_heads * num_hidden, num_classes, heads=1))  # Use a single head in the final layer

        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)  # Apply Batch Normalization
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[self.num_layers - 1](x, edge_index)

        return x



class HGCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_hidden=16, num_layers=2, dropout=0.5):
        super(HGCN, self).__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.linears = torch.nn.ModuleList()
        self.convs.append(GCNConv(num_features, num_hidden))
        self.linears.append(nn.Linear(num_hidden, num_hidden))
        self.dropout = dropout

        for _ in range(1, num_layers):
            self.convs.append(GCNConv(num_hidden, num_hidden))
            self.linears.append(nn.Linear(num_hidden, num_hidden))

        self.linear = nn.Linear(num_hidden * num_layers, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # hierarchical layer computations
        outputs = []
        for layer in range(self.num_layers):
            x = self.convs[layer](x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.linears[layer](x)
            outputs.append(x)
        
        # concatenate hierarchical layers
        z = torch.cat(outputs, dim=1)
        z = self.linear(z)
        return z


class PHGCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_hidden=16, num_layers=2, dropout=0.5):
        super(PHGCN, self).__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.linears = torch.nn.ModuleList()
        self.convs.append(GCNConv(num_features, num_hidden))
        self.linears.append(nn.Linear(num_hidden, num_hidden))

        self.dropout = dropout

        for _ in range(1, num_layers):
            self.convs.append(GCNConv(num_hidden, num_hidden))
            self.linears.append(nn.Linear(num_hidden, num_hidden))

        self.linear = nn.Linear(num_hidden * num_layers, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # hierarchical layer computations
        outputs = []
        for layer in range(self.num_layers):
            y = self.convs[layer](x, edge_index)
            y = F.elu(y)
            y = F.dropout(y, p=self.dropout, training=self.training)
            y = self.linears[layer](y)
            outputs.append(y)
        
        # concatenate hierarchical layers
        z = torch.cat(outputs, dim=1)
        z = self.linear(z)
        return z
 

class GraphSAGE(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_hidden=16, num_layers=2, dropout=0.5):
        super(GraphSAGE, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(num_features, num_hidden))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(num_hidden, num_hidden))
        self.convs.append(SAGEConv(num_hidden, num_classes))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[self.num_layers - 1](x, edge_index)

        return x




class GraphIsomorphismNetwork(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_hidden=16, num_layers=2, dropout=0.5):
        super(GraphIsomorphismNetwork, self).__init__()

        self.num_layers = num_layers
        self.convs = nn.ModuleList()

        nn1 = nn.Sequential(nn.Linear(num_features, num_hidden),
                       nn.BatchNorm1d(num_hidden), nn.ELU(),
                       nn.Linear(num_hidden, num_hidden), nn.ELU())
        self.convs.append(GINConv(nn1))
        self.dropout = dropout

        for _ in range(num_layers - 2):  # minus 2 to account for the first and last layers
            nn_hidden = nn.Sequential(nn.Linear(num_hidden, num_hidden),
                       nn.BatchNorm1d(num_hidden), nn.ELU(),
                       nn.Linear(num_hidden, num_hidden), nn.ELU())
            self.convs.append(GINConv(nn_hidden))

        nn2 = nn.Sequential(nn.Linear(num_hidden, num_hidden),
                       nn.BatchNorm1d(num_hidden), nn.ELU(),
                       nn.Linear(num_hidden, num_hidden), nn.ELU())  # ReLU activation
        self.convs.append(GINConv(nn2))

        self.lin1 = nn.Linear(num_hidden * num_layers, num_hidden)
        self.lin2 = nn.Linear(num_hidden, num_classes)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Node embeddings
        embeddings = []
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            embeddings.append(h)

        # Concatenate 
        h = torch.cat(embeddings, dim=1)

        # Classifier
        h = self.lin1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.lin2(h)

        return h




class GraphTransformer(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_hidden=16, num_layers=2, dropout=0.5):
        super(GraphTransformer, self).__init__()

        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.dropout = dropout

        self.convs.append(TransformerConv(num_features, num_hidden, heads=2))
        for _ in range(num_layers - 2):
            self.convs.append(TransformerConv(num_hidden*2, num_hidden, heads=2))
        self.convs.append(TransformerConv(num_hidden*2, num_hidden, heads=2))

        self.lin = nn.Linear(num_hidden*2, num_classes)  # *2 because of the multi-head attention

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index, edge_attr)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[self.num_layers - 1](x, edge_index, edge_attr)
        x = self.lin(x)

        return x
    



# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gcn2_conv.html



class GCNII(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_hidden=16, num_layers=2, alpha=0.5, theta=1.0, dropout=0.5):
        super(GCNII, self).__init__()

        self.initial_lin = nn.Linear(num_features, num_hidden)
        self.dropout = dropout
        self.layers = torch.nn.ModuleList()

        for i in range(num_layers):
            layer_idx = None if theta is None else (i + 1)
            self.layers.append(GCN2Conv(channels=num_hidden, alpha=alpha, theta=theta, layer=layer_idx))
        
        self.final_lin = nn.Linear(num_hidden, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
#       x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        # Transform the initial node features
        x = self.initial_lin(x)
        x = F.dropout(F.elu(x), p=self.dropout, training=self.training)

        # Save a copy of the initial transformed node features
        x_0 = x.clone()

        for layer in self.layers:
            # Pass the current node features and the initial node features to each GCN2Conv layer
            x = layer(x, x_0, edge_index)
            x = F.dropout(F.elu(x), p=self.dropout, training=self.training) + x

        # Apply the final linear layer to transform to the classification output
        out = self.final_lin(x)

        return out


        
