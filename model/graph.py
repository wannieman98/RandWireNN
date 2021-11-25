import torch.nn as nn
import networkx as nx

class Graph(nn.Module):

    def __init__(self, node_num, p, k=4, m=5, graph_mode="WS"):
        """Random Graph constructor class to map it onto the CNN.

           Args:
              node_num(int): Total number of nodes in the graph.
              p(float): Probability that a graph generate a new edge.
              k(int): Number of nearest neighbors the graph will join.
              m(int): Number of edges to be added sequentially. 
              graph_mode(str): Type of random graph the object will generate.
        """
        self.node_num = node_num
        self.p = p
        self.k = k
        self.m = m
        self.graph_mode = graph_mode

    def get_dag(self) -> dict:
        """Returns a dict representing the post-processed random graph."""
        networkx_random_graph = self.make_graph()
        return self.make_DAG(networkx_random_graph)

    def make_graph(self) -> object:
        """Creates a networkx object based on the given graph mode."""
        if self.graph_mode == "ER":
            graph = nx.random_graphs.erdos_renyi_graph(self.node_num, self.p, seed=SEED)
        elif self.graph_mode == "WS":
            assert self.k % 2 == 0, "k must an even number."
            graph = nx.random_graphs.connected_watts_strogatz_graph(self.node_num, self.k, self.p, seed=SEED)
        elif self.graph_mode == "BA":
            graph = nx.random_graphs.barabasi_albert_graph(self.node_num, self.m, seed=SEED)

        return graph

    def make_DAG(self, graph) -> dict:
        """Given an undirected graph, constructs a directed acyclic graph
           and adds an extra node to all the initial nodes
           and adds an extra node to all the output nodes
           in order to map the graph 
           to the convolutional neural network computation.

           Args:
              Networkx generated random graph class object 
              
           Return:
              Python dictionary with input nodes as keys and output nodes as values
        """
        sets = set()
        out_graph = {}
        graph = graph.to_directed()
        for temp_edge in graph.out_edges:
          edge = (temp_edge[0]+1, temp_edge[1]+1)
          sorted_edge_tuple = tuple(sorted(edge))
          if not has_edges(sorted_edge_tuple, sets):
            insert_edge(edge, out_graph)
            sets.add(sorted_edge_tuple)
            
        post_processing(out_graph, self.node_num)
        
        return out_graph

def has_edges(edge, sets) -> bool:
  """Check whether the edge exists in the graph to avoid cycle.

     Args:
        edge(tuple): Represents an edge (input node, output node)
        sets(set): Stores inserted edges

     Return:
        Whether edge has already been inserted to the grpah
  """
  return edge in sets

def insert_edge(edge, graph) -> None:
  """In-place modifier that inserts the edge into the graph 
     with input node as key and output node as values

     Args:
        edge(tuple): Represents an edge (input node, output node)
        graph(dict): Represents the graph {input node: [output nodes]}

     Return:
        None
  """
  input_node, output_node = edge

  if input_node not in graph:
    graph[input_node] = [output_node]
  else:
    graph[input_node].append(output_node)

def post_processing(graph, node_num) -> None:
  """In-place modifier that 
      inserts an extra input node to all original input nodes
      and inserts an extra output node to to all original output nodes

      extra input node is defined as 0
      extra output node is defined as max(node_value) + 1 

      Args:
          graph(dict): Represents dag {input node: [output nodes]}
          node_num(int): Number of total nodes in the graph.

      Return:
          None - Modified in-place
  """  
  original_input_nodes, original_output_nodes = find_original_nodes(graph)
  all_output_node = node_num + 1

  graph[0] = []
  
  for input_node in original_input_nodes:
    graph[0].append(input_node)
  
  for output_node in original_output_nodes:
    graph[output_node] = [all_output_node]

def find_original_nodes(graph):
  """Returns origianl input and output nodes"""
  original_input_nodes = []
  original_output_nodes = []
  all_output_nodes = set()

  for output_nodes in graph.values():
    for output_node in output_nodes:
      all_output_nodes.add(output_node)

  for input_node, output_nodes in graph.items():
    if input_node not in all_output_nodes:
      original_input_nodes.append(input_node)

    for output_node in output_nodes:
      if output_node not in graph:
        original_output_nodes.append(output_node)

  return original_input_nodes, original_output_nodes