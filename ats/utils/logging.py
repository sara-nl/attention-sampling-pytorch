import os
import numpy as np
import random
import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter


class AttentionSaverTrafficSigns:
    """Save the attention maps to monitor model evolution."""

    def __init__(self, output_directory, ats_model, training_set, opts):
        self.dir = output_directory
        os.makedirs(self.dir, exist_ok=True)
        self.ats_model = ats_model
        self.opts = opts

        idxs = training_set.strided(9)
        data = [training_set[i] for i in idxs]
        self.x_low = torch.stack([d[0] for d in data]).cpu()
        self.x_high = torch.stack([d[1] for d in data]).cpu()
        self.labels = torch.LongTensor([d[2] for d in data]).numpy()

        self.writer = SummaryWriter(os.path.join(self.dir, opts.run_name), flush_secs=5)
        self.on_train_begin()

    def on_train_begin(self):
        opts = self.opts
        with torch.no_grad():
            _, _, _, x_low = self.ats_model(self.x_low.to(opts.device), self.x_high.to(opts.device))
            x_low = x_low.cpu()
            image_list = [x for x in x_low]

        grid = torchvision.utils.make_grid(image_list, nrow=3, normalize=True, scale_each=True)

        self.writer.add_image('original_images', grid, global_step=0, dataformats='CHW')
        self.__call__(-1)

    def __call__(self, epoch, losses=None, metrics=None):
        opts = self.opts
        with torch.no_grad():
            _, att, _, x_low = self.ats_model(self.x_low.to(opts.device), self.x_high.to(opts.device))
            att = att.unsqueeze(1)
            att = F.interpolate(att, size=(x_low.shape[-2], x_low.shape[-1]))
            att = att.cpu()

        grid = torchvision.utils.make_grid(att, nrow=3, normalize=True, scale_each=True, pad_value=1.)
        self.writer.add_image('attention_map', grid, epoch, dataformats='CHW')

        if metrics is not None:
            train_metrics, test_metrics = metrics
            self.writer.add_scalar('Accuracy/Train', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('Accuracy/Test', test_metrics['accuracy'], epoch)

        if losses is not None:
            train_loss, test_loss = losses
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Test', test_loss, epoch)

    @staticmethod
    def imsave(filepath, x):
        if x.shape[-1] == 3:
            plt.imshow(x)
            plt.savefig(filepath)
        else:
            plt.imshow(x, cmap='viridis')
            plt.savefig(filepath)

    @staticmethod
    def reverse_transform(inp):
        """ Do a reverse transformation. inp should be a torch tensor of shape [3, H, W] """
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        inp = (inp * 255).astype(np.uint8)

        return inp

    @staticmethod
    def reverse_transform_torch(inp):
        """ Do a reverse transformation. inp should be a torch tensor of shape [3, H, W] """
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        inp = torch.from_numpy(inp).permute(2, 0, 1)

        return inp


class AttentionSaverMNIST:
    def __init__(self, output_directory, ats_model, dataset, opts):
        self.dir = output_directory
        os.makedirs(self.dir, exist_ok=True)
        self.ats_model = ats_model
        self.opts = opts

        idxs = [random.randrange(0, len(dataset)-1) for _ in range(9)]
        data = [dataset[i] for i in idxs]
        self.x_low = torch.stack([d[0] for d in data]).cpu()
        self.x_high = torch.stack([d[1] for d in data]).cpu()
        self.label = torch.LongTensor([d[2] for d in data]).numpy()

        self.writer = SummaryWriter(os.path.join(self.dir, opts.run_name), flush_secs=2)
        self.__call__(-1)

    def __call__(self, epoch, losses=None, metrics=None):
        opts = self.opts
        with torch.no_grad():
            _, att, patches, x_low = self.ats_model(self.x_low.to(opts.device), self.x_high.to(opts.device))
            att = att.unsqueeze(1)
            att = F.interpolate(att, size=(x_low.shape[-2], x_low.shape[-1]))
            att = att.cpu()

        grid = torchvision.utils.make_grid(att, nrow=3, normalize=True, scale_each=True, pad_value=1.)
        self.writer.add_image('attention_map', grid, epoch, dataformats='CHW')

        if metrics is not None:
            train_metrics, test_metrics = metrics
            self.writer.add_scalar('Accuracy/Train', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('Accuracy/Test', test_metrics['accuracy'], epoch)

        if losses is not None:
            train_loss, test_loss = losses
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Test', test_loss, epoch)
