import os
import torchvision as tv
import torchvision.transforms as tr
from torch.utils.data import DataLoader, Dataset
import numpy as np


class DataSubset(Dataset):
    def __init__(self, base_dataset, inds=None, size=-1):
        self.base_dataset = base_dataset
        if inds is None:
            inds = np.random.choice(list(range(len(base_dataset))), size, replace=False)
        self.inds = inds

    def __getitem__(self, index):
        base_ind = self.inds[index]
        return self.base_dataset[base_ind]

    def __len__(self):
        return len(self.inds)


def cycle(loader):
    while True:
        for data in loader:
            yield data


def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """

    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })


def dataset_fn(args, train, transform):
    if args.dataset == "cifar10":
        args.n_classes = 10
        cls = dataset_with_indices(tv.datasets.CIFAR10)
        return cls(root=args.data_root, transform=transform, download=True, train=train)
    elif args.dataset == "cifar100":
        args.n_classes = 100
        cls = dataset_with_indices(tv.datasets.CIFAR100)
        return cls(root=args.data_root, transform=transform, download=True, train=train)
    elif args.dataset == 'tinyimagenet':
        args.n_classes = 200
        cls = dataset_with_indices(tv.datasets.ImageFolder)
        return cls(root=os.path.join(args.data_root, 'train' if train else 'val'), transform=transform)
    elif 'img' in args.dataset:
        args.n_classes = 1000
        cls = dataset_with_indices(tv.datasets.ImageFolder)
        return cls(root=os.path.join(args.data_root, 'train' if train else 'val'), transform=transform)
    else:
        args.n_classes = 10
        cls = dataset_with_indices(tv.datasets.SVHN)
        return cls(root=args.data_root, transform=transform, download=True, split="train" if train else "test")


def get_data(args):
    mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    if args.dataset == "svhn":
        transform_train = tr.Compose(
            [tr.Pad(4, padding_mode="reflect"),
             tr.RandomCrop(32),
             tr.ToTensor(),
             tr.Normalize(mean, std),
             ]
        )
    else:
        transform_train = tr.Compose(
            [tr.Pad(4, padding_mode="reflect"),
             tr.RandomCrop(32),
             tr.RandomHorizontalFlip(),
             tr.ToTensor(),
             tr.Normalize(mean, std),
             ]
        )
    transform_test = tr.Compose(
        [tr.ToTensor(),
         tr.Normalize(mean, std),
         ]
    )
    if args.dataset == 'img256' or args.dataset == 'imagenet':
        transform_train = tr.Compose([
            tr.RandomResizedCrop(224),
            tr.RandomHorizontalFlip(),
            tr.ToTensor(),
            tr.Normalize(mean, std),
        ])
        transform_px = tr.Compose(
            [
                tr.Resize(256),
                tr.CenterCrop(224),
                tr.ToTensor(),
                tr.Normalize(mean, std),
            ]
        )
    elif 'img' in args.dataset:
        transform_train = tr.Compose([
            tr.RandomHorizontalFlip(),
            tr.ToTensor(),
            tr.Normalize(mean, std),
        ])
        transform_px = tr.Compose(
            [
                tr.Resize(256),
                tr.CenterCrop(224),
                tr.ToTensor(),
                tr.Normalize(mean, std),
            ]
        )
    else:
        transform_px = tr.Compose(
            [
                tr.ToTensor(),
                tr.Normalize(mean, std),
            ]
        )

    # get all training inds
    full_train = dataset_fn(args, True, transform_train)
    all_inds = list(range(len(full_train)))
    # set seed
    np.random.seed(args.seed)
    # shuffle
    np.random.shuffle(all_inds)
    # seperate out validation set
    if args.n_valid > args.n_classes:
        valid_inds, train_inds = all_inds[:args.n_valid], all_inds[args.n_valid:]
    else:
        valid_inds, train_inds = [], all_inds
    train_inds = np.array(train_inds)
    train_labeled_inds = train_inds

    if 'argment' in vars(args) and args.augment is False:
        transform_w = transform_px  # if args.dataset == 'cifar10' else transform_train
    else:
        transform_w = transform_train

    dset_train = DataSubset(dataset_fn(args, True, transform_px), inds=train_inds)
    dset_train_labeled = DataSubset(dataset_fn(args, True, transform_w), inds=train_labeled_inds)
    dset_valid = DataSubset(dataset_fn(args, True, transform_test), inds=valid_inds)

    num_workers = 0 if args.debug else 4
    dload_train = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    dload_train_labeled = DataLoader(dset_train_labeled, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    dload_train = cycle(dload_train)
    dset_test = dataset_fn(args, False, transform_test)
    dload_valid = DataLoader(dset_valid, batch_size=100, shuffle=False, num_workers=num_workers, drop_last=False)
    dload_test = DataLoader(dset_test, batch_size=100, shuffle=False, num_workers=num_workers, drop_last=False)
    return dload_train, dload_train_labeled, dload_valid, dload_test
