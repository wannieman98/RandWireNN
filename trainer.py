import time
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from util import epoch_time
from model.neural_network import Rand_Wire

SEED = 981126

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class Trainer:
    def __init__(self, num_epoch, lr,
                batch_size, num_node, p,
                in_channels, out_channels,
                graph_mode, is_train, name):
        super(Trainer, self).__init__()

        self.params = {'num_epoch': num_epoch,
                       'batch_size': batch_size,
                       'lr': lr,
                       'num_node': num_node,
                       'p': p,
                       'in_channels': in_channels,
                       'out_channels': out_channels,
                       'graph_mode': graph_mode,
                       'is_train': is_train,
                       'name': name
                       }

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.rwnn = Rand_Wire(num_node, p, in_channels, out_channels, graph_mode, is_train, name) 

        self.optimizer = optim.Adam(self.transformer.parameters(), betas=(0.9, 0.98), eps=self.params['lr'])

        self.criterion = nn.CrossEntropyLoss()

        def train(self):
            print("\nbegin training...")

            for epoch in range(self.params['num_epoch']):
                start_time = time.time()

                epoch_loss = train_loop(self.data.train_iter, self.rwnn, self.optimizer, self.criterion, self.device)
                val_loss = val_loop(self.data.val_iter, self.rwnn, self.criterion, self.device)

                end_time = time.time()

                if (epoch + 1) % 2 == 0:
                    test(self.data.test_iter, self.transformer, self.criterion, self.device)

                minutes, seconds, time_left_min, time_left_sec = epoch_time(end_time-start_time, epoch, self.params['num_epoch'])
            
                print("Epoch: {} out of {}".format(epoch+1, self.params['num_epoch']))
                print("Train_loss: {} - Val_loss: {} - Epoch time: {}m {}s - Time left for training: {}m {}s"\
                .format(round(epoch_loss, 3), round(val_loss, 3), minutes, seconds, time_left_min, time_left_sec))

            torch.save(self.transformer.state_dict(), 'data/checkpoints/checkpoint.pth')
            torch.save(self.transformer, 'data/checkpoints/checkpoint.pt')

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

    for src, tgt in test_iter:
        src = src.to(device)
        tgt = tgt.to(device)

        logits = model(src)
        
        loss = criterion(logits, tgt) 
          
        test_loss += loss.item()
    test_loss /= len(test_iter)

    print("Test Loss: {}".format(round(test_loss, 3)))
