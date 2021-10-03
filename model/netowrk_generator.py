import torch
import torch.nn as nn
from graph import Graph

class Rand_Wire(nn.Module):
    def __init__(self, graph_type, channels):
        self.channels = channels
        self.graph = Graph(graph_type, channels)

    def forward(x):
        pass

    

    
