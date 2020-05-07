import torch.nn as nn
import torch.nn.functional as F
import pdb


def conv_layer(in_channels, out_channels, kernel, strides, padding=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=strides, padding_mode="zeros", bias=False,
                     padding=padding)


def batch_norm(filters):
    return nn.BatchNorm2d(filters)


def relu():
    return nn.ReLU()


class Block(nn.Module):

    def __init__(self, in_channels, out_channels, stride, kernel_size, short):
        super(Block, self).__init__()

        self.short = short
        self.bn1 = batch_norm(in_channels)
        self.relu1 = relu()
        self.conv1 = conv_layer(in_channels, out_channels, 1, stride, padding=0)

        self.conv2 = conv_layer(in_channels, out_channels, kernel_size, stride)
        self.bn2 = batch_norm(out_channels)
        self.relu2 = relu()
        self.conv3 = conv_layer(out_channels, out_channels, kernel_size, 1)

    def forward(self, x):
        x = self.bn1(x)
        x = self.relu1(x)

        x_short = x
        if self.short:
            x_short = self.conv1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)

        out = x + x_short
        return out


class FeatureModelTrafficSigns(nn.Module):

    def __init__(self, in_channels, strides=[1, 2, 2, 2], filters=[32, 32, 32, 32]):
        super(FeatureModelTrafficSigns, self).__init__()

        stride_prev = strides.pop(0)
        filters_prev = filters.pop(0)

        self.conv1 = conv_layer(in_channels, filters_prev, 3, stride_prev)

        module_list = nn.ModuleList()
        for s, f in zip(strides, filters):
            module_list.append(Block(filters_prev, f, s, 3, s != 1 or f != filters_prev))

            stride_prev = s
            filters_prev = f

        self.module_list = nn.Sequential(*module_list)

        self.bn1 = batch_norm(filters_prev)
        self.relu1 = relu()
        self.pool = nn.AvgPool2d(kernel_size=(13, 13))

    def forward(self, x):
        out = self.conv1(x)
        out = self.module_list(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool(out)
        out = out.view(out.shape[0], out.shape[1])
        out = F.normalize(out, p=2, dim=-1)
        return out


class FeatureModelMNIST(nn.Module):

    def __init__(self, in_channels):
        super(FeatureModelMNIST, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7)
        self.relu1 = relu()

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.relu2 = relu()

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.relu3 = relu()

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.relu4 = relu()

        self.pool = nn.AvgPool2d(kernel_size=(38, 38))

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.relu3(out)

        out = self.conv4(out)
        out = self.relu4(out)

        out = self.pool(out)
        out = out.view(out.shape[0], out.shape[1])
        out = F.normalize(out, p=2, dim=-1)

        return out
