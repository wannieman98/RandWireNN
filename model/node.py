import torch
import torch.nn as nn


class Node(nn.Module):

    def __init__(self, in_channel, out_channel, degree=1, stride=1):
        """Node Operation defined by the paper.

           Node Operation:
            1. Aggregation of the input nodes.
            2. Transformation through the convolutional layers.

           Args:
              aggregate_weight(tensor): Weight to aggregate the node tensors.
              conv(Module): Convolutional layers defined by the paper.
              degree(int): The in_degree of the node.
              stride(int): The stride of the seperable conv layer.
        """
        super(Node, self).__init__()
        self.aggregate_weight = torch.randn(degree, requires_grad=True)
        self.conv = nn.Sequential(
            nn.ReLU(),
            SeperableConvolution(in_channel, out_channel, stride),
            nn.BatchNorm2d(out_channel)
        )
        self.degree = degree

    def forward(self, x):
        # x = [B, channel, W, H, degree]
        # aggregate weight = [degree, 1]
        if self.degree >= 1 and len(x.shape) > 4:
            x = torch.matmul(x, torch.sigmoid(self.aggregate_weight))

        # x = [B, channel, W, H]
        x = self.conv(x)

        return x


class SeperableConvolution(nn.Module):

    def __init__(self, in_channel, out_channel, stride=1):
        super(SeperableConvolution, self).__init__()
        """
        Seperable convolution defined by the paper.

        Combination of depthwise 3x3 convolution and 1x1 convolution would act 
        as convolution for each channels and then taking a channel-wise pooling.
        """
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3,
                      groups=in_channel, padding=1, stride=stride),
            nn.Conv2d(in_channel, out_channel, 1)
        )

    def forward(self, x):
        # x = [B, C, W, H]
        return self.conv(x)
