import torch.nn as nn
import torch.nn.functional as F

from models import DeepZMODELS


@DeepZMODELS.register_module(name="FCN")
class FCN(nn.Module):

    def __init__(self, in_channels, num_classes, num_filters=64):
        super(FCN, self).__init__()
        num_input_channels = in_channels
        num_output_classes = num_classes
        self.conv1 = nn.Conv2d(num_input_channels, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.last = nn.Conv2d(num_filters, num_output_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, inputs):
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.last(x)
        return x


@DeepZMODELS.register_module(name="SkipFCN")
class Skip_FCN(nn.Module):

    def __init__(self, in_channels, num_classes, num_filters=64):
        super(Skip_FCN, self).__init__()
        num_input_channels = in_channels
        num_output_classes = num_classes
        self.conv1 = nn.Conv2d(num_input_channels, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.last = nn.Conv2d(num_filters, num_output_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, inputs):
        x = F.relu(self.conv1(inputs))
        x1 = F.relu(self.conv2(x))
        x2 = self.conv3(x1)
        x2 = F.relu(x + x2)
        x3 = F.relu(self.conv4(x2))
        x4 = self.conv5(x3)
        x4 = F.relu(x2 + x4)
        x5 = self.last(x4)
        return x5
