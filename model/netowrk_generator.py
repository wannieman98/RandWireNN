import torch
import pickle
from node import Node
import torch.nn as nn
from graph import Graph


class Rand_Wire(nn.Module):

    def __init__(self, node_num, p, k, m, graph_mode, in_channels, out_channels, is_train, name):
        """Generator mapping the random graph onto the CNN architecture.

           Args:
              input_node(int): Node connected to all original input nodes.
              output_node(int): Node connected to all original output nodes.
              graph_mode(str): Type of random graph the object will generate.
              first_node(Module): node operation for all nodes after input node.
              is_train(bool): whether the model is in training phase.
              name(str): a unique name to identify graphs.
              graph(dict): post-processed random graph.
        """
        super(Rand_Wire, self).__init__()
        self.input_node = 0
        self.output_node = node_num + 1
        self.first_node = Node(in_channels, out_channels, 2)

        if is_train:
            self.graph = Graph(node_num, p, k, m, graph_mode)
            self.save_graph()
        else:
            self.graph = self.load_graph()

    def save_graph(self):
        pass

    def load_graph(self):
        pass

    def forward(self, x):
        node_data = {}
        input_nodes = self.graph[self.input_node]

        for in_node, out_node in self.graph.items():
            if in_node == 0:
                x = self.first_node(x)
                for node in out_node:
                    pass