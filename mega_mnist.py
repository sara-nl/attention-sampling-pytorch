import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import time

from models.attention_model import AttentionModelMNIST
from models.feature_model import FeatureModelMNIST
from models.classifier import ClassificationHead

from ats.core.ats_layer import ATSModel
from ats.utils.regularizers import MultinomialEntropy
from ats.utils.logging import AttentionSaverMNIST

from dataset.mega_mnist_dataset import MNIST
from train import train, evaluate


def main(opts):
    train_dataset = MNIST('dataset/mega_mnist', train=True)
    train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=1)

    test_dataset = MNIST('dataset/mega_mnist', train=False)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=opts.batch_size, num_workers=1)

    attention_model = AttentionModelMNIST(squeeze_channels=True, softmax_smoothing=1e-4)
    feature_model = FeatureModelMNIST(in_channels=1)
    classification_head = ClassificationHead(in_channels=32, num_classes=10)

    ats_model = ATSModel(attention_model, feature_model, classification_head, n_patches=opts.n_patches, patch_size=opts.patch_size)
    ats_model = ats_model.to(opts.device)
    optimizer = optim.Adam(ats_model.parameters(), lr=opts.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.decrease_lr_at, gamma=0.1)

    logger = AttentionSaverMNIST(opts.output_dir, ats_model, test_dataset, opts)

    criterion = nn.CrossEntropyLoss()
    entropy_loss_func = MultinomialEntropy(opts.regularizer_strength)

    for epoch in range(opts.epochs):
        train_loss, train_metrics = train(ats_model, optimizer, train_loader,
                                          criterion, entropy_loss_func, opts)

        with torch.no_grad():
            test_loss, test_metrics = evaluate(ats_model, test_loader, criterion,
                                               entropy_loss_func, opts)

        logger(epoch, (train_loss, test_loss), (train_metrics, test_metrics))
        scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--regularizer_strength", type=float, default=0.0001,
                        help="How strong should the regularization be for the attention")
    parser.add_argument("--softmax_smoothing", type=float, default=1e-5,
                        help="Smoothing for calculating the attention map")
    parser.add_argument("--lr", type=float, default=0.001, help="Set the optimizer's learning rate")
    parser.add_argument("--n_patches", type=int, default=10, help="How many patches to sample")
    parser.add_argument("--patch_size", type=int, default=50, help="Patch size of a square patch")
    parser.add_argument("--batch_size", type=int, default=128, help="Choose the batch size for SGD")
    parser.add_argument("--epochs", type=int, default=500, help="How many epochs to train for")
    parser.add_argument("--decrease_lr_at", type=float, default=1000, help="Decrease the learning rate in this epoch")
    parser.add_argument("--clipnorm", type=float, default=1, help="Clip the norm of the gradients")
    parser.add_argument("--output_dir", type=str, help="An output directory", default='output/mnist')
    parser.add_argument('--run_name', type=str, default='run')
    parser.add_argument('--num_workers', type=int, default=20, help='Number of workers to use for data loading')

    opts = parser.parse_args()
    opts.run_name = f"{opts.run_name}_{time.strftime('%Y%m%dT%H%M%S')}"
    opts.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    main(opts)
