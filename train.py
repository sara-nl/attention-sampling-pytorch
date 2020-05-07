import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import pdb

from utils import calc_cls_measures, move_to


def train(model, optimizer, train_loader, criterion, entropy_loss_func, opts):
    """ Train for a single epoch """

    y_probs = np.zeros((0, len(train_loader.dataset.CLASSES)), np.float)
    y_trues = np.zeros((0), np.int)
    losses = []

    # Put model in training mode
    model.train()

    for i, (x_low, x_high, label) in enumerate(tqdm(train_loader)):
        x_low, x_high, label = move_to([x_low, x_high, label], opts.device)

        optimizer.zero_grad()
        y, attention_map, patches, x_low = model(x_low, x_high)

        entropy_loss = entropy_loss_func(attention_map)

        loss = criterion(y, label) - entropy_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opts.clipnorm)
        optimizer.step()

        loss_value = loss.item()
        losses.append(loss_value)

        y_prob = F.softmax(y, dim=1)
        y_probs = np.concatenate([y_probs, y_prob.detach().cpu().numpy()])
        y_trues = np.concatenate([y_trues, label.cpu().numpy()])

    train_loss_epoch = np.round(np.mean(losses), 4)
    metrics = calc_cls_measures(y_probs, y_trues)
    return train_loss_epoch, metrics


def evaluate(model, test_loader, criterion, entropy_loss_func, opts):
    """ Evaluate a single epoch """

    y_probs = np.zeros((0, len(test_loader.dataset.CLASSES)), np.float)
    y_trues = np.zeros((0), np.int)
    losses = []

    # Put model in eval mode
    model.eval()

    for i, (x_low, x_high, label) in enumerate(tqdm(test_loader)):

        x_low, x_high, label = move_to([x_low, x_high, label], opts.device)

        y, attention_map, patches, x_low = model(x_low, x_high)

        entropy_loss = entropy_loss_func(attention_map)
        loss = criterion(y, label) - entropy_loss

        loss_value = loss.item()
        losses.append(loss_value)

        y_prob = F.softmax(y, dim=1)
        y_probs = np.concatenate([y_probs, y_prob.detach().cpu().numpy()])
        y_trues = np.concatenate([y_trues, label.cpu().numpy()])

    test_loss_epoch = np.round(np.mean(losses), 4)
    metrics = calc_cls_measures(y_probs, y_trues)
    return test_loss_epoch, metrics
