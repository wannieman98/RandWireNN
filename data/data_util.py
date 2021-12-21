import torch
import numpy as np
from tqdm import tqdm
from data.imagenet.move import move_file
from torchvision import datasets, transforms
from data.voc.voc_dataset import VOC_CLASSES
from sklearn.metrics import average_precision_score
from data.voc.voc_dataloader import get_voc_dataloader
from torchvision.datasets import VOCSegmentation as VOC


def fetch_dataloader(dataset, path, batch_size):
    if dataset == "voc":
        VOC(path, "2012", "train", True)

        train = get_voc_dataloader('train', batch_size)
        val = get_voc_dataloader('val', batch_size)
        test = get_voc_dataloader('test', batch_size)

    elif dataset == "ImageNet":
        move_file()
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        train = torch.utils.data.DataLoader(
            datasets.ImageNet(
                root=path,
                split="train",
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])),
            batch_size=batch_size,
            num_workers=4,
            shuffle=True, pin_memory=True, drop_last=True)

        val = torch.utils.data.DataLoader(
            datasets.ImageNet(
                root=path,
                split="val",
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])),
            batch_size=batch_size,
            num_workers=4,
            shuffle=True, pin_memory=True, drop_last=False)

        test = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                root=path,
                split="val",
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])),
            batch_size=batch_size,
            num_workers=4,
            shuffle=False, pin_memory=True, drop_last=False)
    else:
        raise NameError("This dataset is not supported.")

    return train, val, test


def test_imagenet(test_iter, model, criterion, device):
    model.eval()
    test_loss = 0

    with torch.no_grad():
        y_true = np.zeros((0, 1000))
        y_score = np.zeros((0, 1000))

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
            aps.append(ap)
        mAP = np.mean(aps)

        test_loss /= len(test_iter)

        print(
            f"Test Loss: {round(test_loss, 3)}, Test Accuracy: {round(mAP,3)}")


def test_voc(test_iter, model, device):
    model.eval()

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
        aps = []
        for i in range(1, y_true.shape[1]):
            ap = average_precision_score(y_true[:, i], y_score[:, i])
            print(
                '-------  Class: {:<12}     AP: {:>8.4f}  -------'.format(VOC_CLASSES[i], ap))
            aps.append(ap)
        mAP = np.mean(aps)

        print(
            f"Test Accuracy: {round(mAP,3)}")
