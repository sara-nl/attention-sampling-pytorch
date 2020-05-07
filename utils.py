import numpy as np
import torch
from sklearn.metrics import accuracy_score


def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    elif isinstance(var, list):
        return [move_to(v, device) for v in var]
    elif isinstance(var, tuple):
        return (move_to(v, device) for v in var)
    return var.to(device)


def calc_cls_measures(probs, label):
    """Calculate multi-class classification measures (Accuracy)
    :probs: NxC numpy array storing probabilities for each case
    :label: ground truth label
    :returns: a dictionary of accuracy
    """
    label = label.reshape(-1, 1)
    n_classes = probs.shape[1]
    preds = np.argmax(probs, axis=1)
    accuracy = accuracy_score(label, preds)

    metric_collects = {'accuracy': accuracy}
    return metric_collects
