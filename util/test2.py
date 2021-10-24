import unittest
import torch
from node import *

class TestMethods(unittest.TestCase):
    # test seperable convolution
    def test_sep_conv(self):
        test = [(1,1), (3,4), (3,10)]
        for i in test:
            tensor = torch.randn((32, i[0], 96, 96), dtype=torch.float)
            sep_conv = SeperableConvolution(i[0], i[1])
            output = sep_conv(tensor)

            expected = torch.randn(32, i[1], 96, 96)

            self.assertEqual(output.shape, expected.shape)


    # test node operations
    def test_node_operation(self):
        for i in range(1, 10):
            with self.subTest(i = i):
                nod = Node(3, 4, i)
                tensor = torch.randn((32, 3, 96, 96, i), dtype=torch.float)
                output = nod(tensor)
                
                expected = torch.randn(32, 4, 96, 96)

                self.assertEqual(output.shape, expected.shape)

if __name__ == "__main__":
    unittest.main()