import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import time

from models.attention_model import AttentionModelColonCancer
from models.feature_model import FeatureModelColonCancer
from models.classifier import ClassificationHead

from ats.core.ats_layer import ATSModel
from ats.utils.regularizers import MultinomialEntropy
from ats.utils.logging import AttentionSaverTrafficSigns

from dataset.colon_cancer_dataset import ColonCancerDataset
from train import train, evaluate


def main(opts):
    train_dataset = ColonCancerDataset('dataset/colon_cancer', train=True)
    train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers, drop_last=False)

    attention_model = AttentionModelColonCancer(squeeze_channels=True, softmax_smoothing=0)
    feature_model = FeatureModelColonCancer(in_channels=3, out_channels=500)
    classification_head = ClassificationHead(in_channels=500, num_classes=len(train_dataset.CLASSES))

    ats_model = ATSModel(attention_model, feature_model, classification_head, n_patches=opts.n_patches,
                         patch_size=opts.patch_size)
    ats_model = ats_model.to(opts.device)
    optimizer = optim.Adam(ats_model.parameters(), lr=opts.lr)

    logger = AttentionSaverTrafficSigns(opts.output_dir, ats_model, train_dataset, opts)

    criterion = nn.CrossEntropyLoss()
    entropy_loss_func = MultinomialEntropy(opts.regularizer_strength)

    for epoch in range(opts.epochs):
        train_loss, train_metrics = train(ats_model, optimizer, train_loader,
                                          criterion, entropy_loss_func, opts)

        with torch.no_grad():
            test_loss, test_metrics = evaluate(ats_model, train_loader, criterion,
                                               entropy_loss_func, opts)

        logger(epoch, (train_loss, test_loss), (train_metrics, test_metrics))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--regularizer_strength", type=float, default=0.01,
                        help="How strong should the regularization be for the attention")
    parser.add_argument("--softmax_smoothing", type=float, default=1e-4,
                        help="Smoothing for calculating the attention map")
    parser.add_argument("--lr", type=float, default=1e-3, help="Set the optimizer's learning rate")
    parser.add_argument("--n_patches", type=int, default=10, help="How many patches to sample")
    parser.add_argument("--patch_size", type=int, default=27, help="Patch size of a square patch")
    parser.add_argument("--batch_size", type=int, default=8, help="Choose the batch size for SGD")
    parser.add_argument("--epochs", type=int, default=500, help="How many epochs to train for")
    parser.add_argument("--decrease_lr_at", type=float, default=250, help="Decrease the learning rate in this epoch")
    parser.add_argument("--clipnorm", type=float, default=1, help="Clip the norm of the gradients")
    parser.add_argument("--output_dir", type=str, help="An output directory", default='output/colon_cancer')
    parser.add_argument('--run_name', type=str, default='run')
    parser.add_argument('--num_workers', type=int, default=20, help='Number of workers to use for data loading')

    opts = parser.parse_args()
    opts.run_name = f"{opts.run_name}_{time.strftime('%Y%m%dT%H%M%S')}"
    opts.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    main(opts)
