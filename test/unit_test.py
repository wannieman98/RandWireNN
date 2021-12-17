import unittest
import torch
from model.graph import Graph
from model.network_generator import Rand_Wire
from model.node import *


class TestMethods(unittest.TestCase):
    # test seperable convolution
    def test_sep_conv(self):
        test = [(1, 1), (3, 4), (3, 10)]
        for i in test:
            tensor = torch.randn((32, i[0], 96, 96), dtype=torch.float)
            sep_conv = SeperableConvolution(i[0], i[1])
            output = sep_conv(tensor)

            expected = torch.randn(32, i[1], 96, 96)

            self.assertEqual(output.shape, expected.shape)

    # test node operations

    def test_node_operation(self):
        for i in range(1, 10):
            with self.subTest(i=i):
                nod = Node(3, 4, i)
                tensor = torch.randn((32, 3, 96, 96, i), dtype=torch.float)
                output = nod(tensor)

                expected = torch.randn(32, 4, 96, 96)

                self.assertEqual(output.shape, expected.shape)

    # test graph functions
    def test_ws_graph(self):
        ws_graph = Graph(32, 0.1, 4, 5, 'WS')
        graph1 = ws_graph.make_graph()
        node, edges = ws_graph.get_graph_info(graph1)
        self.assertEqual(len(node), 34)
        self.assertEqual(len(edges), 34)

    def test_er_graph(self):
        er_graph = Graph(32, 0.1, 4, 5, 'ER')
        graph1 = er_graph.make_graph()
        node, edges = er_graph.get_graph_info(graph1)
        self.assertEqual(len(node), 34)
        self.assertEqual(len(edges), 34)

    def test_ba_graph(self):
        ba_graph = Graph(32, 0.1, 4, 5, 'BA')
        graph1 = ba_graph.make_graph()
        node, edges = ba_graph.get_graph_info(graph1)
        self.assertEqual(len(node), 34)
        self.assertEqual(len(edges), 34)

    # test randome graph generator
    def test_generator(self):
        generator = Rand_Wire(32, 0.1, 3, 3, 'WS', True, 'temp')
        tensor = torch.randn(32, 3, 96, 96)
        outcome = generator(tensor)
        print(outcome.shape)

    # test neural network model

    # test train function

    # test


if __name__ == "__main__":
    unittest.main()
