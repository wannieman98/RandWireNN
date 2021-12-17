
import os
import time
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from util import epoch_time
import torch.optim as optim
from model.neural_network import RandomlyWiredNeuralNetwork
from data.data_util import fetch_dataloader, test_voc, test_imagenet

SEED = 981126

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


class Trainer:

    def __init__(self, num_epoch, lr,
                 batch_size, num_node,
                 p, k, m, channel,
                 in_channels, path,
                 graph_mode, dataset,
                 is_small_regime,
                 checkpoint_path, load):
        super(Trainer, self).__init__()

        self.params = {'num_epoch': num_epoch,
                       'batch_size': batch_size,
                       'lr': lr,
                       'node_num': num_node,
                       'p': p,
                       'k': k,
                       'm': m,
                       'in_channels': in_channels,
                       'channel': channel,
                       'classes': 21 if dataset == 'voc' else 1000,
                       'graph_mode': graph_mode,
                       'load': load,
                       'path': path,
                       'dataset': dataset,
                       'is_small_regime': is_small_regime,
                       'checkpoint_path': checkpoint_path
                       }

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.train_data, self.val_data, self.test_data = fetch_dataloader(
            self.params['dataset'],
            self.params['path'],
            self.params['batch_size'])

        self.rwnn = RandomlyWiredNeuralNetwork(
            self.params['channel'],
            self.params['in_channels'],
            self.params['p'],
            self.params['k'],
            self.params['m'],
            self.params['graph_mode'],
            self.params['classes'],
            self.params['node_num'],
            self.params['checkpoint_path'],
            self.params['load'],
            self.params['is_small_regime']
        ).to(self.device)

        self.best_loss = float('inf')

        if load:
            checkpoint = torch.load(self.param['checkpoint_path'])
            self.rwnn.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer = checkpoint.load_state_dict(
                checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']
            self.best_loss = checkpoint['best_loss']
            self.scheduler = checkpoint['lr_scheduler']
        else:
            # TODO: simulate the learning curve to that of the paper
            self.optimizer = optim.Adam(self.rwnn.parameters(), lr)
            self.epoch = 0
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=20, gamma=0.85)

        self.criterion = nn.CrossEntropyLoss()

        pytorch_total_params = sum(p.numel() for p in self.rwnn.parameters())
        print(f"Number of parameters {pytorch_total_params}")

    def train(self):
        print("\nbegin training...")

        for epoch in range(self.epoch, self.params['num_epoch']):
            print("\nEpoch: {} out of {}".format(
                epoch+1, self.params['num_epoch']))
            start_time = time.perf_counter()

            epoch_loss = train_loop(
                self.train_data, self.rwnn, self.optimizer, self.criterion, self.device)

            val_loss = val_loop(self.val_data, self.rwnn,
                                self.criterion, self.device)

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                torch.save(
                    self.rwnn,
                    os.path.join(self.params['checkpoint_path'], 'best.tar'))

            self.scheduler.step()

            end_time = time.perf_counter()

            minutes, seconds, time_left_min, time_left_sec = epoch_time(
                end_time-start_time, epoch, self.params['num_epoch'])

            print(
                f"Train_loss: {round(epoch_loss, 3)} - Val_loss: {round(val_loss, 3)}")

            if (epoch + 1) % 5 == 0:
                if self.params['dataset'] == "voc":
                    test_voc(self.test_data, self.rwnn,
                             self.criterion, self.device)
                elif self.params['dataset'] == "imagenet":
                    test_imagenet(self.test_data, self.rwnn,
                                  self.criterion, self.device)

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.rwnn.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_loss': self.best_loss,
                    'lr_scheduler': self.lr_scheduler
                }, os.path.join(self.params['checkpoint_path'], 'train.tar'))

            print(
                f"Epoch time: {minutes}m {seconds}s - Time left for training: {time_left_min}m {time_left_sec}s")


def train_loop(train_iter, model, optimizer, criterion, device):
    epoch_loss = 0
    model.train()

    print("Training...")
    for src, tgt in tqdm(train_iter):
        src = src.to(device)
        tgt = tgt.to(device)

        optimizer.zero_grad()

        logits = model(src)

        loss = criterion(logits, tgt)
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(train_iter)


def val_loop(val_iter, model, criterion, device):
    model.eval()
    val_loss = 0

    with torch.no_grad():

        print("Validating...")
        for src, tgt in tqdm(val_iter):
            src = src.to(device)
            tgt = tgt.to(device)

            logits = model(src)

            loss = criterion(logits, tgt)

            val_loss += loss.item()

    return val_loss / len(val_iter)
