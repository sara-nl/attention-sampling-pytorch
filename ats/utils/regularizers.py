import torch
import torch.nn as nn


class MultinomialEntropy(nn.Module):
    """Increase or decrease the entropy of a multinomial distribution.
    Arguments
    ---------
    strength: A float that defines the strength and direction of the
              regularizer. A positive number increases the entropy, a
              negative number decreases the entropy.
    eps: A small float to avoid numerical errors when computing the entropy
    """

    def __init__(self, strength=1, eps=1e-6):
        super(MultinomialEntropy, self).__init__()
        if strength is None:
            self.strength = float(0)
        else:
            self.strength = float(strength)
        self.eps = float(eps)

    def forward(self, x):
        logx = torch.log(x + self.eps)
        # Formally the minus sign should be here
        return - self.strength * torch.sum(x * logx) / float(x.shape[0])
