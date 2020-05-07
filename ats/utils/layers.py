import torch
import torch.nn as nn
import torch.nn.functional as F


class SampleSoftmax(nn.Module):
    """ Apply softmax to the whole sample not just the last dimension.
        Arguments
        ---------
        squeeze_channels: bool, if True then squeeze the channel dimension of the input
        """

    def __init__(self, squeeze_channels=False, smooth=0):
        self.squeeze_channels = squeeze_channels
        self.smooth = smooth
        super(SampleSoftmax, self).__init__()

    def forward(self, x):
        # Apply softmax to the whole x (per sample)
        s = x.shape
        x = F.softmax(x.reshape(s[0], -1), dim=-1)

        # Smooth the distribution
        if 0 < self.smooth < 1:
            x = x * (1 - self.smooth)
            x = x + self.smooth / float(x.shape[1])

        # Finally reshape to the original shape
        x = x.reshape(s)

        # Squeeze the channels dimension if set
        if self.squeeze_channels:
            x = torch.squeeze(x, 1)

        return x
