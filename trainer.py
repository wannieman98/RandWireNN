
import time
import tqdm
import torch
import random
import numpy as np
import torch.nn as nn
from util import epoch_time
import torch.optim as optim
from data.data_util import fetch_dataloader, test_voc
from model.neural_network import RandomlyWiredNeuralNetwork

SEED = 981126

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


class Trainer:

    def __init__(self, num_epoch, lr, batch_size,
                 num_node, p, k, m, classes,
                 in_channels, channel,
                 graph_mode, dataset, path,
                 is_train, is_small_regime):
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
                       'classes': classes,
                       'graph_mode': graph_mode,
                       'is_train': is_train,
                       'path': path,
                       'dataset': dataset,
                       'is_small_regime': is_small_regime
                       }
                       
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.rwnn = RandomlyWiredNeuralNetwork(
            self.params['channel'],
            self.params['in_channels'],
            self.params['p'],
            self.params['k'],
            self.params['m'],
            self.params['graph_mode'],
            self.params['classes'],
            self.params['node_num'],
            self.params['is_train'],
            self.params['is_small_regime']
        ).to(self.device)

        self.train, self.val, self.test = fetch_dataloader(
            self.params['dataset'], 
            self.params['path'], 
            self.params['batch_size'])

        if not self.params['is_train']:
            pass

        pytorch_total_params = sum(p.numel() for p in self.rwnn.parameters())
        print(f"Number of parameters {pytorch_total_params}")

        # self.optimizer = optim.SGD(self.rwnn.parameters(), lr, 0.9, 5e-5)
        self.optimizer = optim.Adam(self.rwnn.parameters(), lr)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=30, gamma=0.5)

        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        print("\nbegin training...")

        for epoch in range(self.params['num_epoch']):
            print("\nEpoch: {} out of {}".format(
                epoch+1, self.params['num_epoch']))
            start_time = time.perf_counter()

            epoch_loss = train_loop(
                self.train, self.rwnn, self.optimizer, self.criterion, self.device)
            val_loss = val_loop(self.val, self.rwnn, self.criterion, self.device)

            self.scheduler.step()

            end_time = time.perf_counter()

            minutes, seconds, time_left_min, time_left_sec = epoch_time(
                end_time-start_time, epoch, self.params['num_epoch'])

            print(
                f"Train_loss: {round(epoch_loss, 3)} - Val_loss: {round(val_loss, 3)}")

            if (epoch + 1) % 5 == 0:
                test_voc(self.test, self.rwnn, self.criterion, self.device)

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