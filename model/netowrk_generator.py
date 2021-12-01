import torch
import pickle
from node import Node
import torch.nn as nn
from graph import Graph
from heapq import heappush, heappop


class Rand_Wire(nn.Module):

    def __init__(self, node_num, p, k, m, graph_mode, in_channels, out_channels, is_train, name):
        """Generator mapping the random graph onto the CNN architecture.

           Args:
              params(dict): Parameters to build necessary componenets.
              input_node(int): Node connected to all original input nodes.
              output_node(int): Node connected to all original output nodes.
              in_nodes(dict): input nodes for each nodes.
              nodeOps(ModuleDict): Node operation for each nodes.
              graph(dict): Post-processed random graph.
        """
        super(Rand_Wire, self).__init__()
        self.params = {
            'node_num': node_num,
            'p': p,
            'k': k,
            'm': m,
            'graph_mode': graph_mode,
            'in_channels': in_channels,
            'out_channels': out_channels,
            'is_train': is_train,
            'name': name
        }
        self.input_node = 0
        self.output_node = node_num + 1
        self.graph = self.get_graph()
        self.in_nodes = self.get_in_nodes()
        self.nodeOps = self.get_nodeOps()

    def get_graph(self):
        if self.params['is_train']:
            nx_graph = Graph(self.params['node_num'],
                             self.params['p'],
                             self.params['k'],
                             self.params['m'],
                             self.params['graph_mode']
                             )
            graph = nx_graph.get_dag()
            self.save_graph(graph)
        else:
            graph = self.load_graph()

        return graph

    def get_in_nodes(self):
        """Returns a dict of input nodes for each node."""
        in_nodes = {}
        for in_, out_ in self.graph.items():
            for node in out_:
                if node not in in_nodes:
                    in_nodes[node] = [in_]
                else:
                    in_nodes[node].append(in_)

        return in_nodes

    def get_nodeOps(self):
        """Returns a ModuleDict of Node Operations for each node."""
        tempNodeOps = {}

        for node, inputs in self.in_nodes.items():
            node = str(node)
            if inputs[0] == 0:
                tempNodeOps[node] = Node(
                    self.params['in_channels'],
                    self.params['out_channels'],
                    len(inputs), 2
                )
            elif node == str(self.output_node):
                pass
            else:
                tempNodeOps[node] = Node(
                    self.params['out_channels'],
                    self.params['out_channels'],
                    len(inputs)
                )

        return nn.ModuleDict(tempNodeOps)

    def save_graph(self, graph):
        path = self.params["name"] + ".pkl"
        with open(path, "wb") as f:
            pickle.dump(graph, f)

    def load_graph(self):
        path = self.params["name"] + ".pkl"
        with open(path, "r") as f:
            out_graph = pickle.load(f)

        return out_graph

    def forward(self, x):
        """Forward propagate through the random graph given input tensor x.

           Use heap queue to begin from the lowest node since the graph is a
           directed acyclic graph. The input node and the output node will not
           have any computation and they are not included in the node count.
        """
        heap = [self.input_node]
        heapSet = set()
        heapSet.add(self.input_node)
        processed = {self.input_node: x}

        while heap:
            node = heappop(heap)
            heapSet.remove(node)

            if node != self.input_node:
                in_nodes = torch.stack([processed[input_node]
                                       for input_node in self.in_nodes[node]], -1)
                processed[node] = self.nodeOps[str(node)](in_nodes)

            neighbors = self.graph[node]

            for neighbor in neighbors:
                if neighbor not in heapSet and neighbor != self.output_node:
                    heappush(heap, neighbor)
                    heapSet.add(neighbor)

        original_outs = torch.stack([processed[node]
                                    for node in self.in_nodes[self.output_node]])

        return torch.mean(original_outs, 0)
