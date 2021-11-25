from netowrk_generator import Rand_Wire
import torch.nn as nn


class RandomlyWiredNeuralNetwork(nn.Module):

    def __init__(self, channel, input_channel, p, graph_type, classes, node_num, is_train=True, is_small_regime=True):
        """The base architecture of Randomly Wired Neural Network model.

           Args:
              channel(int): Number of initial output channels for generator.
              classes(int): Number of classes to classify.
              input_channel(int): Number of initial channels.
              node_num(int): Total number of nodes in the graph.
              conv_layers(Modules): Convolutional Neural Network Layers.
        """
        super(RandomlyWiredNeuralNetwork, self).__init__()
        self.channel = channel
        self.classes = classes
        self.input_channel = input_channel
        self.node_num = node_num

        conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=channel //
                      2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(channel//2)
        )
        if is_small_regime:
            conv2 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(input_channel=channel//2, out_channels=channel,
                          kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(channel)
            )

            conv3 = Rand_Wire(node_num, p, channel, channel,
                              graph_type, is_train, "small_regime_1")

            conv4 = Rand_Wire(node_num, p, channel, 2*channel,
                              graph_type, is_train, "small_regime_2")

            conv5 = Rand_Wire(node_num, p, 2*channel, 4*channel,
                              graph_type, is_train, "small_regime_3")

            conv6 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(4*channel, 1280, kernel_size=1),
                nn.BatchNorm2d()
            )
        else:
            conv2 = Rand_Wire(node_num, p, channel//2, channel,
                              graph_type, is_train, "regular_regime_1")

            conv3 = Rand_Wire(node_num, p, channel, 2*channel,
                              graph_type, is_train, "regular_regime_2")

            conv4 = Rand_Wire(node_num, p, 2*channel, 4*channel,
                              graph_type, is_train, "regular_regime_3")

            conv5 = Rand_Wire(node_num, p, 4*channel, 8*channel,
                              graph_type, is_train, "regular_regime_4")

            conv6 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(8*channel, 1280, kernel_size=1),
                nn.BatchNorm2d()
            )

        classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1000),
            nn.Linear(1280, classes)
        )

        self.conv_layers = nn.Sequential(
            conv1(),
            conv2(),
            conv3(),
            conv4(),
            conv5(),
            conv6(),
            classifier()
        )

    def forward(self, x):
        #               x = [B, classes, 96, 96]
        #       Small Regime        |         Regular Regime
        x = self.conv1(x)  # [B, channel/2, 48, 48]
        x = self.conv2  # [B, channel, 24, 24]       [B, channel, 24, 24]
        x = self.conv3(x)  # [B, channel, 12, 12]       [B, 2*channel, 12, 12]
        x = self.conv4(x)  # [B, 2*channel, 6, 6]       [B, 4*channel, 6, 6]
        x = self.conv5(x)  # [B, 4*channel, 3, 3]       [B, 8*channel, 3, 3]
        x = self.conv6(x)  # [B, 1280, 3, 3]
        x = self.classifier(x)  # [B, classes]

        return x
