import torch
import torch.nn as nn

from ..utils import expand_many, to_float32


class ExpectWithReplacement(torch.autograd.Function):
    """ Custom pytorch layer for calculating the expectation of the sampled patches
        with replacement.
    """
    @staticmethod
    def forward(ctx, weights, attention, features):

        axes = [-1] * (len(features.shape) - 2)
        wf = expand_many(weights, axes)

        F = torch.sum(wf * features, dim=1)

        ctx.save_for_backward(weights, attention, features, F)
        return F

    @staticmethod
    def backward(ctx, grad_output):
        weights, attention, features, F = ctx.saved_tensors
        axes = [-1] * (len(features.shape) - 2)
        wf = expand_many(weights, axes)

        grad = torch.unsqueeze(grad_output, 1)

        # Gradient wrt to the attention
        ga = grad * features
        ga = torch.sum(ga, axis=list(range(2, len(ga.shape))))
        ga = ga * weights / attention

        # Gradient wrt to the features
        gf = wf * grad

        return None, ga, gf


class ExpectWithoutReplacement(torch.autograd.Function):
    """ Custom pytorch layer for calculating the expectation of the sampled patches
        without replacement.
    """

    @staticmethod
    def forward(ctx, weights, attention, features):
        # Reshape the passed weights and attention in feature compatible shapes
        axes = [-1] * (len(features.shape) - 2)
        wf = expand_many(weights, axes)
        af = expand_many(attention, axes)

        # Compute how much of the probablity mass was available for each sample
        pm = 1 - torch.cumsum(attention, axis=1)
        pmf = expand_many(pm, axes)

        # Compute the features
        Fa = af * features
        Fpm = pmf * features
        Fa_cumsum = torch.cumsum(Fa, axis=1)
        F_estimator = Fa_cumsum + Fpm

        F = torch.sum(wf * F_estimator, axis=1)

        ctx.save_for_backward(weights, attention, features, pm, pmf, Fa, Fpm, Fa_cumsum, F_estimator)

        return F

    @staticmethod
    def backward(ctx, grad_output):
        weights, attention, features, pm, pmf, Fa, Fpm, Fa_cumsum, F_estimator = ctx.saved_tensors
        device = weights.device

        axes = [-1] * (len(features.shape) - 2)
        wf = expand_many(weights, axes)
        af = expand_many(attention, axes)

        N = attention.shape[1]
        probs = attention / pm
        probsf = expand_many(probs, axes)
        grad = torch.unsqueeze(grad_output, 1)

        # Gradient wrt to the attention
        ga1 = F_estimator / probsf
        ga2 = (
                torch.cumsum(features, axis=1) -
                expand_many(to_float32(torch.arange(0, N, device=device)), [0] + axes) * features
        )
        ga = grad * (ga1 + ga2)
        ga = torch.sum(ga, axis=list(range(2, len(ga.shape))))
        ga = ga * weights

        # Gradient wrt to the features
        gf = expand_many(to_float32(torch.arange(N-1, -1, -1, device=device)), [0] + axes)
        gf = pmf + gf * af
        gf = wf * gf
        gf = gf * grad

        return None, ga, gf


class Expectation(nn.Module):
    """ Approximate the expectation of all the features under the attention
        distribution (and its gradient) given a sampled set.

        Arguments
        ---------
        attention: Tensor of shape (B, N) containing the attention values that
                   correspond to the sampled features
        features: Tensor of shape (B, N, ...) containing the sampled features
        replace: bool describing if we sampled with or without replacement
        weights: Tensor of shape (B, N) or None to weigh the samples in case of
                 multiple samplings of the same position. If None it defaults
                 o torch.ones(B, N)
        """

    def __init__(self, replace=False):
        super(Expectation, self).__init__()
        self._replace = replace

        self.E = ExpectWithReplacement() if replace else ExpectWithoutReplacement()

    def forward(self, features, attention, weights=None):
        if weights is None:
            weights = torch.ones_like(attention) / float(attention.shape[1])

        return self.E.apply(weights, attention, features)
