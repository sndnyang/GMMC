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

import argparse
import os
import torch as t
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as tr
from torch.utils.data import DataLoader, Dataset
# Sampling
from tqdm import tqdm

from utils import init_random_x_y
from ExpUtils import *
from models.mmc_models import F

t.backends.cudnn.benchmark = True
t.backends.cudnn.enabled = True
seed = 1
im_sz = 32
n_ch = 3
n_classes = 10
correct = 0


def sample_p_0(device, replay_buffer, bs, y=None):
    if len(replay_buffer) == 0:
        return init_random_x_y(args, bs), []
    buffer_size = len(replay_buffer[0]) if y is None else len(replay_buffer[0]) // args.n_classes
    inds = t.randint(0, buffer_size, (bs,))
    # if cond, convert inds to class conditional inds
    if y is not None:
        inds = y.cpu() * buffer_size + inds
        assert not args.uncond, "Can't drawn conditional samples without giving me y"
    buffer_x = replay_buffer[0][inds]
    buffer_y = replay_buffer[1][inds]
    random_samples = init_random_x_y(args, bs)
    choose_random_y = (t.rand(bs) < args.reinit_freq)
    choose_random_x = choose_random_y.float()[:, None, None, None]
    samples = choose_random_x * random_samples[0] + (1 - choose_random_x) * buffer_x
    if y is None:
        y = choose_random_y * random_samples[1] + (~ choose_random_y) * buffer_y
    return samples.to(device), y, inds


def sample_q(args, device, f, replay_buffer, lgm, y=None):
    """this func takes in replay_buffer now so we have the option to sample from
    scratch (i.e. replay_buffer==[]).  See test_wrn_ebm.py for example.
    """
    # eval and train
    n_steps = args.n_steps
    f.eval()
    lgm.eval()
    # get batch size
    bs = args.batch_size if y is None else y.size(0)
    # generate initial samples and buffer inds of those samples (if buffer is used)
    init_sample, y, buffer_inds = sample_p_0(device, replay_buffer, bs=bs, y=y)
    x_k = t.autograd.Variable(init_sample, requires_grad=True)
    # sgld
    # hist = []
    y = y.to(device)
    u_y = t.index_select(lgm.centers, 0, y)
    if not args.inject:
        z_x = u_y + args.gamma * t.randn(bs, lgm.centers.shape[1]).to(device)
    for it in range(n_steps):
        if args.inject:
            z_x = u_y + args.gamma * t.randn(bs, lgm.centers.shape[1]).to(device)
        features = f.feature(x_k)
        e_x = lgm.P_X_y(features, z_x)
        # hist.append(e_x.item())
        g_t = t.autograd.grad(e_x, [x_k], retain_graph=True)[0]
        x_k.data += -args.sgld_lr * g_t
        x_k = t.clamp(x_k, -1., 1.)
    f.train()
    lgm.train()

    final_samples = x_k.detach()
    # update replay buffer
    if len(replay_buffer) > 0:
        replay_buffer[0][buffer_inds] = final_samples.cpu()
        replay_buffer[1][buffer_inds] = y.cpu()
    return final_samples, y


def uncond_samples(f, lgm_loss, args, device, save=True):
    sqrt = lambda x: int(t.sqrt(t.Tensor([x])))
    plot = lambda p, x: tv.utils.save_image(t.clamp(x, -1, 1), p, normalize=True, nrow=sqrt(x.size(0)))

    replay_buffer = [t.FloatTensor(args.buffer_size, 3, 32, 32).uniform_(-1, 1),
                     t.LongTensor(np.random.randint(args.n_classes, size=args.buffer_size))]
    for i in range(args.n_sample_steps):
        samples = sample_q(args, device, f, replay_buffer, lgm_loss)
        if i % args.print_every == 0 and save:
            plot('{}/samples_{}.png'.format(args.save_dir, i), samples[0])
        print(i)
    return replay_buffer


def cond_samples(f, replay_buffer, lgm_loss, args, device, fresh=False):
    sqrt = lambda x: int(t.sqrt(t.Tensor([x])))
    plot = lambda p, x: tv.utils.save_image(t.clamp(x, -1, 1), p, normalize=True, nrow=sqrt(x.size(0)))

    if fresh:
        replay_buffer = uncond_samples(f, lgm_loss, args, device, save=False)
    n_it = replay_buffer[0].size(0) // 100
    all_y = []
    for i in range(n_it):
        x = replay_buffer[0][i * 100: (i + 1) * 100].to(device)
        feats = f.feature(x)
        logits = lgm_loss(feats, None)
        y = logits.max(1)[1]
        all_y.append(y)

    all_y = t.cat(all_y, 0)
    each_class = [replay_buffer[0][all_y == l] for l in range(10)]
    print([len(c) for c in each_class])
    start, end = 0, 10
    if args.n_classes == 100:
        start, end = 10, 20
    for i in range(100):
        this_im = []
        for l in range(start, end):
            this_l = each_class[l][i * 10: (i + 1) * 10]
            this_im.append(this_l)
        this_im = t.cat(this_im, 0)
        if this_im.size(0) > 0:
            plot('{}/samples_{}.png'.format(args.save_dir, i), this_im)
        print(i)


def test_clf(f, lgm_loss, args, device):
    transform_test = tr.Compose(
        [tr.ToTensor(),
         tr.Normalize((.5, .5, .5), (.5, .5, .5)),
         lambda x: x + t.randn_like(x) * args.sigma]
    )

    def sample(x, n_steps=args.n_steps):
        x_k = t.autograd.Variable(x.clone(), requires_grad=True)
        # sgld
        for k in range(n_steps):
            f_prime = t.autograd.grad(f(x_k).sum(), [x_k], retain_graph=True)[0]
            x_k.data += f_prime + 1e-2 * t.randn_like(x_k)
        final_samples = x_k.detach()
        return final_samples

    if args.dataset == "cifar_train":
        dset = tv.datasets.CIFAR10(root="../data", transform=transform_test, download=True, train=True)
    elif args.dataset == "cifar_test":
        dset = tv.datasets.CIFAR10(root="../data", transform=transform_test, download=True, train=False)
    elif args.dataset == "cifar100_train":
        dset = tv.datasets.CIFAR100(root="../data", transform=transform_test, download=True, train=True)
    elif args.dataset == "cifar100_test":
        dset = tv.datasets.CIFAR100(root="../data", transform=transform_test, download=True, train=False)
    elif args.dataset == "svhn_train":
        dset = tv.datasets.SVHN(root="../data", transform=transform_test, download=True, split="train")
    elif args.dataset == "svhn_test":
        dset = tv.datasets.SVHN(root="../data", transform=transform_test, download=True, split="test")
    else:  # args.dataset == "svhn_test":
        dset = tv.datasets.CIFAR10(root="../data", transform=transform_test, download=True, train=False)

    num_workers = 0 if args.debug else 4
    dload = DataLoader(dset, batch_size=100, shuffle=False, num_workers=num_workers, drop_last=False)

    corrects, losses, pys, preds = [], [], [], []
    for x_p_d, y_p_d in tqdm(dload):
        x_p_d, y_p_d = x_p_d.to(device), y_p_d.to(device)
        feats = f.feature(x_p_d)
        logits = lgm_loss(feats)
        py = nn.Softmax(dim=1)(logits).max(1)[0].detach().cpu().numpy()
        loss = (-t.gather(logits, 1, t.unsqueeze(y_p_d, 1))).detach().cpu().numpy()
        losses.extend(loss)
        correct = (logits.max(1)[1] == y_p_d).float().cpu().numpy()
        corrects.extend(correct)
        pys.extend(py)
        preds.extend(logits.max(1)[1].cpu().numpy())

    loss = np.mean(losses)
    correct = np.mean(corrects)
    gamma2 = loss * 2 / lgm_loss.feat_dim
    print('loss %.5g,  accuracy: %g%%, gamma^2 = %.4g, gamma=%.4g' % (loss, correct * 100, gamma2, np.sqrt(gamma2)))
    return correct, gamma2


def eval_fid(f, lgm_loss, replay_buffer, args, device):
    from Task.eval_buffer import eval_fid
    n_it = replay_buffer[0].size(0) // 100
    all_y = []
    probs = []
    with t.no_grad():
        for i in tqdm(range(n_it)):
            x = replay_buffer[0][i * 100: (i + 1) * 100].to(device)
            feats = f.feature(x)
            logits = lgm_loss(feats, None)
            y = logits.max(1)[1]
            prob = nn.Softmax(dim=1)(logits).max(1)[0]
            all_y.append(y)
            probs.append(prob)

    all_y = t.cat(all_y, 0)
    probs = t.cat(probs, 0)
    each_class = [replay_buffer[0][all_y == l] for l in range(10)]
    each_class_probs = [probs[all_y == l] for l in range(10)]
    print([len(c) for c in each_class])

    new_buffer = []
    ratio = abs(args.ratio)
    for c in range(10):
        each_probs = each_class_probs[c]

        if ratio < 1:
            topk = int(len(each_probs) * ratio)
        else:
            topk = int(ratio)
        topk = min(topk, len(each_probs))
        topks = t.topk(each_probs, topk, largest=args.ratio > 0)
        index_list = topks[1]
        images = each_class[c][index_list]
        new_buffer.append(images)

    replay_buffer = t.cat(new_buffer, 0)
    print("Total number of samples to eval fid:  %d" % len(replay_buffer))
    fid = eval_fid(replay_buffer, args)
    print("FID of score {}".format(fid))


def main(args):
    global correct
    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    args.device = device

    model_cls = F
    f = model_cls(args.depth, args.width, args.norm)
    print(f"loading model from {args.load_path}")

    # load em up
    ckpt_dict = t.load(args.load_path)
    f.load_state_dict(ckpt_dict["model_state_dict"])
    replay_buffer = ckpt_dict["replay_buffer"]
    lgm_loss = ckpt_dict["margin"]

    f = f.to(device)
    lgm_loss = lgm_loss.to(device)
    f.eval()
    lgm_loss.eval()

    if args.eval == "test_clf":
        test_clf(f, lgm_loss, args, device)

    if args.eval == "fid":
        eval_fid(f, lgm_loss, replay_buffer, args, device)

    if args.eval == "cond_samples":
        cond_samples(f, replay_buffer, lgm_loss, args, device, args.fresh_samples)

    if args.eval == "uncond_samples":
        uncond_samples(f, lgm_loss, args, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("GMMC")
    parser.add_argument("--eval", default="test_clf", type=str, choices=["uncond_samples", "cond_samples", "test_clf", "fid"])
    parser.add_argument("--dataset", default="cifar_test", type=str, choices=["cifar_train", "cifar_test", "svhn_test", "svhn_train", "cifar100_train", "cifar100_test"],
                        help="Dataset to use when running test_clf for classification accuracy")
    # optimization
    parser.add_argument("--batch_size", type=int, default=64)
    # regularization
    parser.add_argument("--sigma", type=float, default=0)
    # network
    parser.add_argument("--norm", type=str, default=None, choices=[None, "batch", "instance", "layer", "act"])
    # EBM specific
    parser.add_argument("--n_steps", type=int, default=0)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--depth", type=int, default=28)
    parser.add_argument("--uncond", action="store_true")
    parser.add_argument("--buffer_size", type=int, default=0)
    parser.add_argument("--reinit_freq", type=float, default=.0)
    parser.add_argument("--sgld_lr", type=float, default=1.0)
    parser.add_argument("--inject", action="store_true", help="If set, then Noise Injected Sampling, otherwise staged sampling")

    # logging + evaluation
    parser.add_argument("--save_dir", type=str, default='lda_eval')
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--n_sample_steps", type=int, default=1000)
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--fresh_samples", action="store_true", help="If set, then we generate a new replay buffer from scratch for conditional sampling, Will be much slower.")

    parser.add_argument("--ratio", type=float, default=100000, help="if ratio < 1, use the percentile for each category, if ratio > 1, choose int(ratio) or all from each category ")
    parser.add_argument("--gamma", type=float, default=0.001)
    parser.add_argument("--gpu-id", type=str, default="")

    args = parser.parse_args()
    auto_select_gpu(args)
    init_debug(args)
    run_time = time.strftime('%m%d%H%M%S', time.localtime(time.time()))
    if args.save_dir == 'lda_eval':
        # by default to eval the model
        args.dir_path = 'eval_run/' + args.load_path.split('/')[-1] + "_eval_%s_%s" % (args.eval, run_time)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    args.n_classes = 100 if "cifar100" in args.dataset else 10
    set_file_logger(logger, args)
    args.save_dir = args.dir_path
    print = wlog
    print(args.save_dir)
    main(args)
    print(args.save_dir)
