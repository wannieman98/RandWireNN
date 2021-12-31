from model.network_generator import Rand_Wire
import torch.nn as nn


class RandomlyWiredNeuralNetwork(nn.Module):

    def __init__(self, channel, input_channel, p, k, m, graph_type, classes, node_num, path, load=False, is_small_regime=True):
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

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=channel //
                      2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(channel//2)
        )

        if is_small_regime:
            self.conv2 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(in_channels=channel//2, out_channels=channel,
                          kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(channel)
            )

            self.conv3 = Rand_Wire(
                node_num, p, k, m, graph_type, channel, channel, path, load, "small_regime_1")

            self.conv4 = Rand_Wire(
                node_num, p, k, m, graph_type, channel, 2*channel, path, load, "small_regime_2")

            self.conv5 = Rand_Wire(
                node_num, p, k, m, graph_type, 2*channel, 4*channel, path, load, "small_regime_3")

            self.conv6 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(4*channel, 1280, 1),
                nn.BatchNorm2d(1280),
                nn.AdaptiveAvgPool2d(1)
            )
        else:
            self.conv2 = Rand_Wire(
                node_num, p, k, m, graph_type, channel//2, channel, path, load, "regular_regime_1")

            self.conv3 = Rand_Wire(
                node_num, p, k, m, graph_type, channel, 2*channel, path, load, "regular_regime_2")

            self.conv4 = Rand_Wire(
                node_num, p, k, m, graph_type, 2*channel, 4*channel, path, load, "regular_regime_3")

            self.conv5 = Rand_Wire(
                node_num, p, k, m, graph_type, 4*channel, 8*channel, path, load, "regular_regime_4")

            self.conv6 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(8*channel, 1280, 1),
                nn.BatchNorm2d(1280),
                nn.AdaptiveAvgPool2d(1)
            )

        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(1280, classes)
        )

        for parameter in self.parameters():
            if parameter.dim() >= 2:
                nn.init.xavier_uniform_(parameter)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x).squeeze(-1).squeeze(-1)
        x = self.classifier(x)

        return x
