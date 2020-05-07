import torch.nn as nn


class ClassificationHead(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(ClassificationHead, self).__init__()

        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        return self.classifier(x)
