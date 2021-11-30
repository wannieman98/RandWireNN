
import time
import tqdm
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from util import epoch_time
from data.voc.voc_dataset import VOC_CLASSES
from data.voc.voc_dataloader import get_dataloader
from sklearn.metrics import average_precision_score
from torchvision.datasets import VOCSegmentation as VOC
from model.neural_network import RandomlyWiredNeuralNetwork

SEED = 981126

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


class Trainer:
    def __init__(self, num_epoch, lr, batch_size,
                 num_node, p, k, m,
                 in_channels, channel,
                 graph_mode, is_train, name):
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
                       'classes': 21,
                       'graph_mode': graph_mode,
                       'is_train': is_train,
                       'name': name
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
            True,
        ).to(self.device)

        pytorch_total_params = sum(p.numel() for p in self.rwnn.parameters())
        print(f"Number of parameters {pytorch_total_params}")

        self.optimizer = optim.Adam(self.rwnn.parameters(), lr)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=30, gamma=0.1)
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        print("\nbegin training...")
        path = '/content/'  # filepath to download voc data

        VOC(path, "2012", "train", True)

        train = get_dataloader('train', self.params['batch_size'])
        val = get_dataloader('val', self.params['batch_size'])
        test_set = get_dataloader('test', self.params['batch_size'])

        for epoch in range(self.params['num_epoch']):
            print("\nEpoch: {} out of {}".format(
                epoch+1, self.params['num_epoch']))
            start_time = time.perf_counter()

            epoch_loss = train_loop(
                train, self.rwnn, self.optimizer, self.criterion, self.device)
            val_loss = val_loop(val, self.rwnn, self.criterion, self.device)

            self.scheduler.step()

            end_time = time.perf_counter()

            minutes, seconds, time_left_min, time_left_sec = epoch_time(
                end_time-start_time, epoch, self.params['num_epoch'])

            print(
                f"Train_loss: {round(epoch_loss, 3)} - Val_loss: {round(val_loss, 3)}")

            if (epoch + 1) % 10 == 0:
                test(test_set, self.rwnn, self.criterion, self.device)

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


def test(test_iter, model, criterion, device):
    model.eval()
    test_loss = 0

    print("Testing...")
    with torch.no_grad():
        y_true = np.zeros((0, 21))
        y_score = np.zeros((0, 21))

        for src, tgt in tqdm(test_iter):
            src = src.to(device)
            tgt = tgt.to(device)

            logits = model(src)

            y_true = np.concatenate((y_true, tgt.cpu().numpy()), axis=0)
            y_score = np.concatenate((y_score, logits.cpu().numpy()), axis=0)
            loss = criterion(logits, tgt)
            test_loss += loss.item()
        aps = []
        for i in range(1, y_true.shape[1]):
            ap = average_precision_score(y_true[:, i], y_score[:, i])
            print(
                '-------  Class: {:<12}     AP: {:>8.4f}  -------'.format(VOC_CLASSES[i], ap))
            aps.append(ap)
        mAP = np.mean(aps)

        test_loss /= len(test_iter)

        print(
            f"Test Loss: {round(test_loss, 3)}, Test Accuracy: {round(mAP,3)}")
