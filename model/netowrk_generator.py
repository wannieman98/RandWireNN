import torch
from node import Node
import torch.nn as nn
from graph import Graph

class Rand_Wire(nn.Module):
    def __init__(self, node_num, p, in_channels, out_channels, graph_mode, is_train, name):
        super(Rand_Wire, self).__init__()
        self.node_num = node_num
        self.p = p
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.graph_mode = graph_mode
        self.is_train = is_train
        self.name = name

        # get graph nodes and in edges
        graph_node = Graph(self.node_num, self.p, graph_mode=graph_mode)
        
        # if self.is_train is True:
        #     print("is_train: True")
        #     graph = graph_node.make_graph()
        #     self.nodes, self.in_edges = graph_node.get_graph_info(graph)
        #     graph_node.save_random_graph(graph, name)
        # else:
        #     graph = graph_node.load_random_graph(name)
        #     self.nodes, self.in_edges = graph_node.get_graph_info(graph)

        # # define input Node
        # self.module_list = nn.ModuleList([Node(self.in_channels, self.out_channels, self.in_edges[0], stride=2)])
        # # define the rest Node
        # self.module_list.extend([Node(self.in_channels, self.out_channels, self.in_edges[node]) for node in self.nodes if node > 0])

    def forward(self, x):
        memory = {}
        # start vertex
        out = self.module_list[0].forward(x)
        memory[0] = out

        # the rest vertex
        for node in range(1, len(self.nodes) - 1):
            if len(self.in_edges[node]) > 1:
                out = self.module_list[node].\
                    forward(*[memory[in_vertex]\
                        for in_vertex in self.in_edges[node]])
            else:
                out = self.module_list[node].\
                    forward(memory[self.in_edges[node][0]])
            memory[node] = out

        out = memory[self.in_edges[self.node_num + 1][0]]
        for in_vertex_index in range(1, len(self.in_edges[self.node_num + 1])):
            out += memory[self.in_edges[self.node_num + 1][in_vertex_index]]
        out = out / len(self.in_edges[self.node_num + 1])
        return out


    