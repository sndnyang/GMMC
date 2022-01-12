# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import logging
import torch
import torch as t
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as tr
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
import numpy as np

from ExpUtils import AverageMeter


def sqrt(x):
    return int(t.sqrt(t.Tensor([x])))


def plot(p, x):
    return tv.utils.save_image(t.clamp(x, -1, 1), p, normalize=True, nrow=sqrt(x.size(0)))


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def save_checkpoint(state, save, epoch):
    if not os.path.exists(save):
        os.makedirs(save)
    filename = os.path.join(save, 'checkpt-%04d.pth' % epoch)
    torch.save(state, filename)


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


def grad_norm(m):
    total_norm = 0
    for p in m.parameters():
        param_grad = p.grad
        if param_grad is not None:
            param_norm = param_grad.data.norm(2) ** 2
            total_norm += param_norm
    total_norm = total_norm ** (1. / 2)
    return total_norm.item()


def grad_vals(m):
    ps = []
    for p in m.parameters():
        if p.grad is not None:
            ps.append(p.grad.data.view(-1))
    ps = t.cat(ps)
    return ps.mean().item(), ps.std(), ps.abs().mean(), ps.abs().std(), ps.abs().min(), ps.abs().max()


def init_random(args, bs, im_sz=32, n_ch=3):
    return t.FloatTensor(bs, n_ch, im_sz, im_sz).uniform_(-1, 1)


def init_random_x_y(args, bs, im_sz=32, n_ch=3):
    return [t.FloatTensor(bs, n_ch, im_sz, im_sz).uniform_(-1, 1), t.LongTensor(np.random.randint(args.n_classes, size=bs))]


def get_data(args):
    if args.dataset == "svhn":
        transform_train = tr.Compose(
            [tr.Pad(4, padding_mode="reflect"),
             tr.RandomCrop(32),
             tr.ToTensor(),
             tr.Normalize((.5, .5, .5), (.5, .5, .5)),
             lambda x: x + args.sigma * t.randn_like(x)]
        )
    else:
        transform_train = tr.Compose(
            [tr.Pad(4, padding_mode="reflect"),
             tr.RandomCrop(32),
             tr.RandomHorizontalFlip(),
             tr.ToTensor(),
             tr.Normalize((.5, .5, .5), (.5, .5, .5)),
             lambda x: x + args.sigma * t.randn_like(x)]
        )
    transform_test = tr.Compose(
        [tr.ToTensor(),
         tr.Normalize((.5, .5, .5), (.5, .5, .5))]
        #lambda x: x + args.sigma * t.randn_like(x)]  # this should be removed
    )

    def dataset_fn(train, transform):
        if args.dataset == "cifar10":
            return tv.datasets.CIFAR10(root=args.data_root, transform=transform, download=True, train=train)
        elif args.dataset == "cifar100":
            return tv.datasets.CIFAR100(root=args.data_root, transform=transform, download=True, train=train)
        else:
            return tv.datasets.SVHN(root=args.data_root, transform=transform, download=True, split="train" if train else "test")

    # get all training inds
    full_train = dataset_fn(True, transform_train)
    all_inds = list(range(len(full_train)))
    # set seed
    np.random.seed(1234)
    # shuffle
    np.random.shuffle(all_inds)
    # seperate out validation set
    if args.n_valid is not None:
        valid_inds, train_inds = all_inds[:args.n_valid], all_inds[args.n_valid:]
    else:
        valid_inds, train_inds = [], all_inds
    train_inds = np.array(train_inds)
    train_labeled_inds = []
    other_inds = []
    if args.labels_per_class > 0:
        train_labels = np.array([full_train[ind][1] for ind in train_inds])  # to speed up
        for i in range(args.n_classes):
            print(i)
            train_labeled_inds.extend(train_inds[train_labels == i][:args.labels_per_class])
            other_inds.extend(train_inds[train_labels == i][args.labels_per_class:])
    else:
        train_labeled_inds = train_inds

    dset_train = DataSubset(dataset_fn(True, transform_train), inds=train_inds)
    dset_train_labeled = DataSubset(dataset_fn(True, transform_train), inds=train_labeled_inds)
    dset_valid = DataSubset(dataset_fn(True, transform_test), inds=valid_inds)

    num_workers = 0 if args.debug else 4
    dload_train = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    dload_train_labeled = DataLoader(dset_train_labeled, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    dload_train_labeled = cycle(dload_train_labeled)
    dset_test = dataset_fn(False, transform_test)
    dload_valid = DataLoader(dset_valid, batch_size=100, shuffle=False, num_workers=num_workers, drop_last=False)
    dload_test = DataLoader(dset_test, batch_size=100, shuffle=False, num_workers=num_workers, drop_last=False)
    return dload_train, dload_train_labeled, dload_valid, dload_test


def lda_eval_classification(f, data_loader, lgm_loss, set_name, epoch, args=None, wlog=None):
    """
    support top1 and top5 accuracy
    """
    corrects, losses = [], []
    device = args.device
    if lgm_loss.num_classes >= 200:
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

    for data in data_loader:
        x, y = data[:2]
        x, y = x.to(device), y.to(device)
        try:
            features = f.feature(x)
        except AttributeError:
            features = f(x)
        logits = lgm_loss(features)
        loss = (-t.gather(logits, 1, t.unsqueeze(y, 1))).mean().item()
        losses.append(loss)
        if lgm_loss.num_classes >= 200:
            acc1, acc5 = accuracy(logits, y, topk=(1, 5))
            top1.update(acc1[0], x.size(0))
            top5.update(acc5[0], x.size(0))
        else:
            correct = (logits.max(1)[1] == y).float().cpu().numpy()
            corrects.extend(correct)
    loss = np.mean(losses)
    if wlog:
        my_print = wlog
    else:
        my_print = print
    if lgm_loss.num_classes >= 200:
        correct = top1.avg
        my_print("Epoch %d, %s loss %.5f, top1 acc %.4f, top5 acc %.4f" % (epoch, set_name, loss, top1.avg, top5.avg))
    else:
        correct = np.mean(corrects)
        my_print("Epoch %d, %s loss %.5f, acc %.4f" % (epoch, set_name, loss, correct))
    if args.vis:
        args.writer.add_scalar('%s/Loss' % set_name, loss, epoch)
        if lgm_loss.num_classes >= 200:
            args.writer.add_scalar('%s/Acc_1' % set_name, top1.avg, epoch)
            args.writer.add_scalar('%s/Acc_5' % set_name, top5.avg, epoch)
        else:
            args.writer.add_scalar('%s/Accuracy' % set_name, correct, epoch)
    return correct, loss


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def eval_classification(f, dload, lgm_loss, device):
    corrects, losses = [], []
    for x_p_d, y_p_d in dload:
        x_p_d, y_p_d = x_p_d.to(device), y_p_d.to(device)
        # logits = f.classify(x_p_d)
        try:
            feats = f.feature(x_p_d)
        except AttributeError:
            feats = f(x_p_d)
        logits, mlogits, likelihood = lgm_loss(feats, y_p_d)
        loss = nn.CrossEntropyLoss(reduction='none')(mlogits, y_p_d).cpu().numpy()
        losses.extend(loss)
        correct = (logits.max(1)[1] == y_p_d).float().cpu().numpy()
        corrects.extend(correct)
    loss = np.mean(losses)
    correct = np.mean(corrects)
    return correct, loss


def checkpoint(f, buffer, component, tag, args, device):
    f.cpu()
    ckpt_dict = {
        "model_state_dict": f.state_dict(),
        "replay_buffer": buffer,
        # remove the warning: "margin": component.state_dict()
        "margin": component
    }
    t.save(ckpt_dict, os.path.join(args.save_dir, tag))
    f.to(device)


def isnan(tensor):
    return (tensor != tensor)


def tens2numpy(tens):
    if tens.is_cuda:
        tens = tens.cpu()
    if tens.requires_grad:
        tens = tens.detach()
    return tens.numpy()


def t2n(tens):
    if isinstance(tens, np.ndarray):
        return tens
    elif isinstance(tens, list):
        return np.array(tens)
    elif isinstance(tens, float) or isinstance(tens, int):
        return np.array([tens])
    else:
        return tens2numpy(tens)


def n2t(tens):
    return torch.from_numpy(tens)


def confidence_softmax(x, const=0, dim=1):
    #x -= x.max(dim=1, keepdim=True)[0] # why not using stable softmax?
    x = torch.exp(x)
    n_classes = x.shape[1]
    # return x
    norms = torch.sum(x, dim=dim, keepdim=True)
    return (x + const) / (norms + const * n_classes)


def update_distal_adv(a, a_up, grads, opti):
    a_up.data = torch.from_numpy(a)
    opti.zero_grad()
    a_up.grad = grads
    opti.step()
    a_up.data.clamp_(0, 1)
    a = a_up.data.numpy()
    return a


def compute_sigma_module(module):
    try:
        w = module.weight.detach()
    except AttributeError:
        w = module.module.weight.detach()
    u, s, v = torch.svd(w.reshape(w.shape[0], -1))
    sigma = torch.max(s)
    return sigma
