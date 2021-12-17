from data.voc.voc_dataset import VocDataset
from torchvision import transforms
import torch

def get_voc_dataloader(split='train', batch_size=32):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if split == "train":
        train_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            normalize
        ])

        ds_train = VocDataset("/content/VOCdevkit/VOC2012",
                              'train', train_transform)

        return torch.utils.data.DataLoader(dataset=ds_train,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=1)
    elif split == "val":
        test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        ds_val = VocDataset("/content/VOCdevkit/VOC2012",
                            'val', test_transform)

        return torch.utils.data.DataLoader(dataset=ds_val,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=1)
    else:
        test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        ds_val = VocDataset("/content/VOCdevkit/VOC2012",
                            'val', test_transform)

        return torch.utils.data.DataLoader(dataset=ds_val,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           num_workers=1)
