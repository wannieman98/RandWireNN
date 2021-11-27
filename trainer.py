import time
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from util import epoch_time
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
                       'classes': 10,
                       'graph_mode': graph_mode,
                       'is_train': is_train,
                       'name': name
                       }
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.rwnn = RandomlyWiredNeuralNetwork(
            self.params['channel'], 
            self.params['in_channels'], 
            self.params['p'],
            self.params['k'],
            self.params['m'],
            self.params['graph_mode'],
            self.params['classes'], 
            self.params['node_num']
        ).to(self.device)

        pytorch_total_params = sum(p.numel() for p in self.rwnn.parameters())
        print(pytorch_total_params)

        self.optimizer = optim.Adam(self.rwnn.parameters(), betas=(0.9, 0.98), eps=self.params['lr'])

        self.criterion = nn.CrossEntropyLoss()

    def train(self):
          print("\nbegin training...")
          normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std= [0.229, 0.224, 0.225])
          
          train_transform = transforms.Compose([
              transforms.Resize(32),
              transforms.CenterCrop(32),
              transforms.RandomHorizontalFlip(),
              transforms.RandomResizedCrop(32),
              transforms.ToTensor(),
              normalize
          ])

          test_transform = transforms.Compose([
              transforms.Resize(32),
              transforms.CenterCrop(32),
              transforms.ToTensor(),
              normalize
          ])
          cifar = CIFAR10('/content/', True, train_transform, download=True)
          cifar = CIFAR10('/content/', False, train_transform, download=True)
          train = DataLoader(cifar, 64, True)
          val = DataLoader(cifar, 64, False)
          test_set = DataLoader(cifar, 64, False)


          for epoch in range(self.params['num_epoch']):
              start_time = time.time()

              epoch_loss = train_loop(train, self.rwnn, self.optimizer, self.criterion, self.device)
              val_loss = val_loop(val, self.rwnn, self.criterion, self.device)

              end_time = time.time()

              if (epoch + 1) % 5 == 0:
                  test(test_set, self.rwnn, self.criterion, self.device)

              minutes, seconds, time_left_min, time_left_sec = epoch_time(end_time-start_time, epoch, self.params['num_epoch'])
          
              print("Epoch: {} out of {}".format(epoch+1, self.params['num_epoch']))
              print("Train_loss: {} - Val_loss: {} - Epoch time: {}m {}s - Time left for training: {}m {}s"\
              .format(round(epoch_loss, 3), round(val_loss, 3), minutes, seconds, time_left_min, time_left_sec))

def train_loop(train_iter, model, optimizer, criterion, device):
    epoch_loss = 0
    model.train()
    for src, tgt in train_iter:
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

    for src, tgt in val_iter:
        src = src.to(device)
        tgt = tgt.to(device)

        logits = model(src)
        
        loss = criterion(logits, tgt) 
        
        val_loss += loss.item()

    return val_loss / len(val_iter)

def test(test_iter, model, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total_count = 0

    for src, tgt in test_iter:
        src = src.to(device)
        tgt = tgt.to(device)

        logits = model(src)
        _, predictions = torch.max(logits, 1)
        correct += (predictions == tgt).float().sum()
        total_count += 1
        loss = criterion(logits, tgt) 
          
        test_loss += loss.item()
    test_loss /= len(test_iter)
    accuracy = 100 * (correct / total_count)

    print(f"Test Loss: {test_loss}, Test Accuracy: {accuracy}")