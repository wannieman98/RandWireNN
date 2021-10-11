import torch.nn as nn
from netowrk_generator import Rand_Wire


class RandomlyWiredNeuralNetwork(nn.Module):
    def __init__(self, channel, input_channel, graph_type, classes, node_num, is_small_regime=True):
        self.channel = channel
        self.classes = classes
        self.input_channel = input_channel
        self.node_num

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=channel//2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(channel//2)
            )
        if is_small_regime:
            self.conv2 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(input_channel=channel//2, out_channels=channel, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(channel)
            )

            self.conv3 = Rand_Wire(graph_type, node_num, channel)

            self.conv4 = Rand_Wire(graph_type, node_num, 2*channel)
            
            self.conv5 = Rand_Wire(graph_type, node_num, 4*channel)

            self.class_conv = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(4*channel, 1280, kernel_size=1),
                nn.BatchNorm2d()
            )
        else:
            self.conv2 = Rand_Wire(graph_type, node_num//2, channel)

            self.conv3 = Rand_Wire(graph_type, node_num, 2*channel)
            
            self.conv4 = Rand_Wire(graph_type, node_num, 4*channel)

            self.conv5 = Rand_Wire(graph_type, node_num, 8*channel)

            self.class_conv = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(8*channel, 1280, kernel_size=1),
                nn.BatchNorm2d()
            )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(),
            nn.Linear(1280, classes),
            nn.LogSoftmax()
        )

    # noinspection PyCallingNonCallable
    def forward(self, x):
                          #                     x = [B, classes, 96, 96]       
                          #             Small Regime        |         Regular Regime 
        x = self.conv1(x) #                      [B, channel/2, 48, 48]
        x = self.conv2  #         [B, channel, 24, 24]       [B, channel, 24, 24]
        x = self.conv3(x) #         [B, channel, 12, 12]       [B, 2*channel, 12, 12]
        x = self.conv4(x) #         [B, 2*channel, 6, 6]       [B, 4*channel, 6, 6]
        x = self.conv5(x) #         [B, 4*channel, 3, 3]       [B, 8*channel, 3, 3]
        x = self.class_conv(x) #                    [B, 1280, 3, 3]   
        x = self.classifier(x) #                    [B, classes]

        return x

