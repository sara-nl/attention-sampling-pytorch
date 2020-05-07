"""Implement sampling from a multinomial distribution on a n-dimensional
tensor."""
import torch
import torch.distributions as dist


def _sample_with_replacement(logits, n_samples):
    """Sample with replacement using the pytorch categorical distribution op."""
    distribution = dist.categorical.Categorical(logits=logits)
    return distribution.sample(sample_shape=torch.Size([n_samples])).transpose(0, 1)


def _sample_without_replacement(logits, n_samples):
    """Sample without replacement using the Gumbel-max trick.
    See lips.cs.princeton.edu/the-gumbel-max-trick-for-discrete-distributions/
    """
    z = -torch.log(-torch.log(torch.rand_like(logits)))
    return torch.topk(logits+z, k=n_samples)[1]


def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return torch.stack(tuple(reversed(out)))


def sample(n_samples, attention, sample_space, replace=False,
           use_logits=False):
    """Sample from the passed in attention distribution.
    Arguments
    ---------
    n_samples: int, the number of samples per datapoint
    attention: tensor, the attention distribution per datapoint (could be logits
               or normalized)
    sample_space: This should always equal K.shape(attention)[1:]
    replace: bool, sample with replacement if set to True (defaults to False)
    use_logits: bool, assume the input is logits if set to True (defaults to False)
    """
    # Make sure we have logits and choose replacement or not
    logits = attention if use_logits else torch.log(attention)
    sampling_function = (
        _sample_with_replacement if replace
        else _sample_without_replacement
    )

    # Flatten the attention distribution and sample from it
    logits = logits.reshape(-1, sample_space[0]*sample_space[1])
    samples = sampling_function(logits, n_samples)

    # Unravel the indices into sample_space
    batch_size = attention.shape[0]
    n_dims = len(sample_space)

    # Gather the attention
    attention = attention.view(batch_size, 1, -1).expand(batch_size, n_samples, -1)
    sampled_attention = torch.gather(attention, -1, samples[:, :, None])[:, :, 0]

    samples = unravel_index(samples.reshape(-1, ), sample_space)
    samples = torch.reshape(samples.transpose(1, 0), (batch_size, n_samples, n_dims))

    return samples, sampled_attention
