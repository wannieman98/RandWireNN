import os
import yaml
import networkx as nx
import torch.nn as nn

class Graph(nn.Module):
    def __init__(self, node_num, p, k=4, m=5, graph_mode="WS"):
        self.node_num = node_num
        self.p = p
        self.k = k
        self.m = m
        self.graph_mode = graph_mode

    def make_graph(self):
        seed = 981126
        if self.graph_mode == "ER":
            graph = nx.random_graphs.erdos_renyi_graph(self.node_num, self.p, seed=seed)
        elif self.graph_mode == "WS":
            graph = nx.random_graphs.connected_watts_strogatz_graph(self.node_num, self.k, self.p, seed=seed)
        elif self.graph_mode == "BA":
            graph = nx.random_graphs.barabasi_albert_graph(self.node_num, self.m, seed=seed)

        return graph

    def get_graph_info(self, graph):
        in_edges = {}
        in_edges[0] = []
        nodes = [0]
        end = []
        for node in graph.nodes():
            neighbors = list(graph.neighbors(node))
            neighbors.sort()

            edges = []
            check = []
            for neighbor in neighbors:
                if node > neighbor:
                    edges.append(neighbor + 1)
                    check.append(neighbor)
            if not edges:
                edges.append(0)
            in_edges[node + 1] = edges
            if check == neighbors:
                end.append(node + 1)
            nodes.append(node + 1)
        in_edges[self.node_num + 1] = end
        nodes.append(self.node_num + 1)

        return nodes, in_edges

    def save_random_graph(self, graph, path):
        pass
    #     if not os.path.isdir("saved_graph"):
    #         os.mkdir("saved_graph")
    #     nx.yaml(graph, "./saved_graph/" + path)
    #     yaml.dump(graph,)

    def load_random_graph(self, path):
        pass
    #     return nx.read_yaml("./saved_graph/" + path)