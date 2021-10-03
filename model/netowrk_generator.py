import torch
import torch.nn as nn


class Rand_Wire(nn.Module):
    def __init__(self, algo, nodes, channels):
        self.nodes = nodes
        self.channels = channels
        self.algo = algo

    

    
