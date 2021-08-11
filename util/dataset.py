import os
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transform

def get_dataset(dataset_name: str = "cifar10", root: str = "data", train: bool = True, batch_size: int = 64):
    data_dir = os.path.join(root, dataset_name)
    download = False

    if not os.path.exists(data_dir):
        download = True
        os.mkdir(data_dir)

    if dataset_name == "cifar10":
        
        normalize = transform.Normalize(mean=[0.5,],
                                        std=[0.1])

        transforms = transform.Compose([
            transform.ToTensor(),
            normalize
        ])

        dataset = datasets.CIFAR10(root=data_dir, download=download, transform=transforms, train=train)
        if train:
            return DataLoader(dataset, batch_size=batch_size, shuffle=True)
        else:
            return DataLoader(dataset, batch_size=batch_size, shuffle=False)
            
    elif dataset_name == "VOC":
        normalize = transform.Normalize(mean=[0.485, 0.456, 0.406],
                                        std= [0.229, 0.224, 0.225])

        if train:
            transforms = transform.Compose([
                transform.Resize(224),
                transform.CenterCrop(224),
                transform.RandomHorizontalFlip(),
                transform.RandomResizedCrop(224),
                transform.ToTensor(),
                normalize
            ])
        else:
            transforms = transform.Compose([
                transform.Resize(224),
                transform.CenterCrop(224),
                transform.ToTensor(),
                normalize,
            ])

        dataset = datasets.VOCSegmentation(root=data_dir, download=download, year="2012", image_set="train", transform=transforms)
        if train:
            return DataLoader(dataset, batch_size=batch_size, shuffle=True)
        else:
            return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    