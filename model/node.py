import torch
import torch.nn as nn

class Node(nn.Module):
    def __init__(self, in_channel, out_channel, degree, stride=1):
        super(Node, self).__init__()
        self.aggregate_weight = torch.randn(degree, requires_grad=True)
        self.conv = nn.Sequential(
            nn.ReLU(),
            SeperableConvolution(in_channel, out_channel, stride),
            nn.BatchNorm2d(out_channel)
        )
        if isinstance(degree, int):
            self.degree = [degree]
        else:
            self.degree = degree

    def forward(self, x):
        # x = [B, channel, W, H, degree]
        # aggregate weight = [degree, 1]
        if len(self.degree) >= 1 and len(x.shape) > 4:
            x = torch.matmul(x, torch.sigmoid(self.aggregate_weight))
        # x = [B, channel, W, H]
        x = self.conv(x)

        return x


class SeperableConvolution(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(SeperableConvolution, self).__init__()
        """
        Depthwise Convolution: Type of convolution where we apply a single convolutional 
        filter for each input channels and keeps the convolutions of each channels seperate
        Convolution: Type of convolution that apply a convolutional filter of over the input 
        channels as a whole and mix the elements among the channels.
        """
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, groups=in_channel, padding=1, stride=stride),
            nn.Conv2d(in_channel, out_channel, 1) 
        )

    def forward(self, x):
        """
        The combination of depthwise convolution and 1x1 convolution would act 
        as convolution for each channels, and then taking a channel-wise pooling.
        """
        # x = [B, C, W, H]
        return self.conv(x)
