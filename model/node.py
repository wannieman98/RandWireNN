import torch
import torch.nn as nn

class Node(nn.Module):
    def __init__(self, in_channel, out_channel, degree):
        super(Node, self).__init__()
        self.aggregate_weight = torch.zeros(degree, requires_grad=True)
        self.conv = nn.Sequential(
            nn.ReLU(),
            SeperableConvolution(in_channel, out_channel),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, x):
        # x = [B, channel, W, H, degree]
        # aggregate weight = [degree, 1]
        print(x.shape, self.aggregate_weight.shape)
        x = torch.matmul(x, torch.sigmoid(self.aggregate_weight))
        # x = [B, channel, W, H]
        x = self.conv(x)

        return x


class SeperableConvolution(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SeperableConvolution, self).__init__()
        """
        Depthwise Convolution: Type of convolution where we apply a single convolutional 
        filter for each input channels and keeps the convolutions of each channels seperate
        Convolution: Type of convolution that apply a convolutional filter of over the input 
        channels as a whole and mix the elements among the channels.
        """
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, groups=in_channel, padding=1),
            nn.Conv2d(in_channel, out_channel, 1) 
        )

    def forward(self, x):
        """
        The combination of depthwise convolution and 1x1 convolution would act 
        as convolution for each channels, and then taking a channel-wise pooling.
        """
        return self.conv(x)

# if __name__ == "__main__":
#     node = Node(3, 4, 3)
#     tensor = torch.randn((32, 3, 96, 96, 3), dtype=torch.float)

#     print(node(tensor).shape)
