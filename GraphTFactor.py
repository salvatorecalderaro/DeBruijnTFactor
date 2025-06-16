import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch.nn import Linear, Dropout
from scipy import stats
import numpy as np
import os

MODELS_PATH = "models"
EMB_SIZE = 256
K_SIZES = [3, 5, 7, 9, 11, 13]
HIDDEN_CHANNELS = 1024
DROPOUT_RATE = .3

class GraphTFactor(nn.Module):
    def __init__(self, in_features, hidden_channels, num_classes, dropout_rate=0.5):
        super(GraphTFactor, self).__init__()
        self.conv1 = GCNConv(in_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)

        self.lin = Linear(hidden_channels, num_classes)
        self.dropout = Dropout(dropout_rate)  # Dropout layer

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)  # Apply dropout after the first convolution

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)  # Apply dropout after the second convolution

        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)  # Apply dropout after the third convolution

        # 2. Readout layer (alternative pooling options)
        x = global_mean_pool(x, batch)  # Alternative: global_max_pool, global_add_pool
        
        # 3. Apply a final linear layer (logits)
        x = self.lin(x)
        return x  # Raw logits returned



def identify_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device



def load_models(device):
    models = []
    for k in K_SIZES:
        model_path = os.path.join(MODELS_PATH, f"model_{k}.pth")
        model = GraphTFactor(EMB_SIZE, HIDDEN_CHANNELS,1, DROPOUT_RATE).to(device)
        model.load_state_dict(torch.load(model_path))
        if device == torch.device("mps"):
            model = model.to(torch.float32).to(device)
        else:
            model = model.to(device)
        models.append(model)
    return models

def predict(graphs):
    device = identify_device()
    models = load_models(device)
    preds = []
    
    for (graph,model) in zip(graphs,models):
        model.eval()
        with torch.no_grad():
            if device == torch.device("mps"):
                graph = graph.to(torch.float32).to(device)
            else:
                graph = graph.to(device)
            null_batch = torch.zeros(graph.num_nodes, dtype=torch.long).to(device)
            proba = model(graph.x, graph.edge_index, null_batch)
            proba = torch.sigmoid(proba)
            pred = torch.round(proba).item()
        preds.append(pred)
    
    
    moda_result = stats(np.array(preds))
    final_pred = moda_result[0]
    return final_pred
    