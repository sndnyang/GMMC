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

import torch as t
import torchvision as tv
import os
import argparse
import numpy as np
import wandb
from tqdm import tqdm
from ExpUtils import *
from utils import *
from Task.eval_buffer import  eval_fid
from models.mmc_models import *
from Utils.mmc_utils import cal_center
from LDALayer import LDALayer
from get_data import get_data
t.backends.cudnn.benchmark = True
t.backends.cudnn.enabled = True
seed = 1
conditionals = []


def get_model_and_buffer(args, device):
    model_cls = F
    f = model_cls(args.depth, args.width, args.norm, dropout_rate=args.dropout_rate)
    if not args.uncond:
        assert args.buffer_size % args.n_classes == 0, "Buffer size must be divisible by args.n_classes"
    lgm_loss = LDALayer(args.n_classes, f.f.last_dim, 1.0, args).to(device)
    if args.load_path is None:
        # make replay buffer
        replay_buffer = init_random_x_y(args, args.buffer_size)
    else:
        print(f"loading model from {args.load_path}")
        ckpt_dict = t.load(args.load_path)
        f.load_state_dict(ckpt_dict["model_state_dict"])
        replay_buffer = ckpt_dict["replay_buffer"]
        lgm_loss = ckpt_dict["margin"]

    f = f.to(device)
    return f, replay_buffer, lgm_loss


def init_random(arg, bs):
    global conditionals
    n_ch = 3
    if arg.dataset == 'svhn':
        size = [3, 32, 32]
        im_sz = 32
    else:
        size = [3, 32, 32]
        im_sz = 32
    new = t.zeros(bs, n_ch, im_sz, im_sz)
    y = np.zeros(bs)
    for i in range(bs):
        index = np.random.randint(arg.n_classes)
        dist = conditionals[index]
        new[i] = dist.sample().view(size)
        y[i] = index
    return t.clamp(new, -1, 1).cpu(), t.LongTensor(y)


def get_sample_q(args, device):
    def sample_p_0(replay_buffer, bs, y=None):
        if len(replay_buffer) == 0:
            return init_random(args, bs), []
        buffer_size = len(replay_buffer[0]) if y is None else len(replay_buffer[0]) // args.n_classes
        inds = t.randint(0, buffer_size, (bs,))
        if y is not None:
            # if cond, convert inds to class conditional inds
            inds = y.cpu() * buffer_size + inds
        buffer_x = replay_buffer[0][inds]
        buffer_y = replay_buffer[1][inds]
        random_samples = init_random(args, bs)
        choose_random_y = (t.rand(bs) < args.reinit_freq)
        choose_random_x = choose_random_y.float()[:, None, None, None]
        samples = choose_random_x * random_samples[0] + (1 - choose_random_x) * buffer_x
        if y is None:
            y = choose_random_y * random_samples[1] + (~ choose_random_y) * buffer_y
        return samples.to(device), y, inds

    def sample_q(f, replay_buffer, lgm, y=None, n_steps=args.n_steps):
        """this func takes in replay_buffer now so we have the option to sample from
        scratch (i.e. replay_buffer==[]).  See test_wrn_ebm.py for example.
        """
        # eval and train
        # f.eval()
        lgm.eval()
        # get batch size
        bs = args.batch_size if y is None else y.size(0)
        # generate initial samples and buffer inds of those samples (if buffer is used)
        init_sample, y, buffer_inds = sample_p_0(replay_buffer, bs=bs, y=y)
        x_k = t.autograd.Variable(init_sample, requires_grad=True)
        # sgld
        y = y.to(device)
        u_y = t.index_select(lgm.centers, 0, y)
        if not args.inject:
            z_x = u_y + args.gamma * t.randn(bs, lgm.centers.shape[1]).to(device)

        for it in range(n_steps):
            if args.inject:
                z_x = u_y + args.gamma * t.randn(bs, lgm.centers.shape[1]).to(device)
            features = f.feature(x_k)
            e_x = lgm.P_X_y(features, z_x)
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
    return sample_q


def init_from_centers(arg):
    global conditionals
    from torch.distributions.multivariate_normal import MultivariateNormal
    bs = arg.buffer_size
    if arg.dataset == 'svhn':
        size = [3, 32, 32]
    else:
        size = [3, 32, 32]
    centers = t.load('../%s_mean.pt' % arg.dataset)
    covs = t.load('../%s_cov.pt' % arg.dataset)

    buffer = []
    y = np.zeros(bs)
    for i in range(arg.n_classes):
        mean = centers[i].to(arg.device)
        cov = covs[i].to(arg.device)
        dist = MultivariateNormal(mean, covariance_matrix=cov + 1e-4 * t.eye(int(np.prod(size))).to(arg.device))
        buffer.append(dist.sample((bs // arg.n_classes,)).view([bs // arg.n_classes] + size).cpu())
        conditionals.append(dist)
        y[i * bs // args.n_classes:(i + 1) * bs // args.n_classes] = i
    return t.clamp(t.cat(buffer), -1, 1), t.LongTensor(y)


def main(arg):

    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    arg.device = device

    sample_q = get_sample_q(arg, device)

    f, replay_buffer, lgm_loss = get_model_and_buffer(arg, device)
    if not arg.load_path and arg.method in ['mmc', 'dmmc']:
        print('initialized with MMC')
        mu = cal_center(n_classes=lgm_loss.num_classes, dim=lgm_loss.feat_dim, c=arg.mu_c)
        lgm_loss.centers.data = mu.to(device)

    if arg.gen:
        buffer = init_from_centers(arg)
        if args.load_path is None:
            replay_buffer = buffer

    fid_init = {'img32': 415.6, 'cifar10': 220.3, 'cifar100': 208.9}
    # if arg.dataset in fid_init and args.load_path is None:
    if arg.dataset in fid_init:
        fid = fid_init[arg.dataset]
    else:
        fid = eval_fid(f, replay_buffer, arg, device, ratio=0.9, eval='fid')

    prev_fid = fid
    inc_score = 0
    print('IS {}, fid {}'.format(0, fid))
    # optimizer
    params = f.parameters()
    if "adam" in arg.optimizer:
        print('adam')
        optim = t.optim.Adam(params, lr=arg.lr, betas=[.9, .999], weight_decay=arg.weight_decay)
    else:
        print('sgd')
        optim = t.optim.SGD(params, lr=arg.lr, momentum=.9, weight_decay=arg.weight_decay, nesterov=True)

    # datasets
    dload_train, dload_train_labeled, dload_valid, dload_test = get_data(arg)

    best_valid_acc = 0.0
    cur_iter = 0
    # trace learning rate
    new_lr = arg.lr
    best_fid = 10000
    for epoch in range(arg.n_epochs):
        if epoch in arg.decay_epochs:
            for param_group in optim.param_groups:
                new_lr = param_group['lr'] * arg.decay_rate
                param_group['lr'] = new_lr
            print("Decaying lr to {}".format(new_lr))

        for i, (x_lab, y_lab, idx) in tqdm(enumerate(dload_train_labeled)):
            if cur_iter <= arg.warmup_iters:
                lr = arg.lr * cur_iter / float(arg.warmup_iters)
                for param_group in optim.param_groups:
                    param_group['lr'] = lr

            L = 0.
            if arg.gen:
                x_p_d, y_p_d, x_idx = dload_train.__next__()
                x_p_d = x_p_d.to(device)
                x_p_d += arg.sigma * t.randn_like(x_p_d)

                # generative training
                x_p_d = x_p_d.to(device)
                y_p_d = y_p_d.to(device)

                fp_all = f.feature(x_p_d)
                fp = lgm_loss.P_X_y(fp_all, y=y_p_d)
                if arg.class_cond_p_x_sample:
                    x_q, y_q = sample_q(f, replay_buffer, lgm_loss, y=y_p_d)
                else:
                    x_q, y_q = sample_q(f, replay_buffer, lgm_loss)

                fq_all = f.feature(x_q)
                fq = lgm_loss.P_X_y(fq_all, y=y_q)
                l_p_x = (fp - arg.beta * fq)
                # trace the loss curve
                if arg.vis:
                    arg.writer.add_scalar('Train/fp', fp, cur_iter)
                    arg.writer.add_scalar('Train/fq', fq, cur_iter)
                    arg.writer.add_scalar('Train/Loss', l_p_x, cur_iter)
                if cur_iter % arg.print_every == 0:
                    print('{} P(x) | {}:{:>d} f(x_p_d)={:>6.4f} f(x_q)={:>6.4f} d={:>6.4f}'.format(arg.pid, epoch, i, fp, fq, l_p_x))

                L += l_p_x

            if arg.cls:  # maximize log p(x, y)
                # discriminative training
                x_lab, y_lab = x_lab.to(device), y_lab.to(device)

                features = f.feature(x_lab)
                logits = lgm_loss(features)

                l_p_x_given_y = (-t.gather(logits, 1, t.unsqueeze(y_lab, 1))).mean()
                if cur_iter % arg.print_every == 0:
                    acc = (logits.max(1)[1] == y_lab).float().mean()
                    print('P(y|x) {}:{:>d} loss={:>6.4f}, acc={:>6.4f}'.format(epoch, cur_iter, l_p_x_given_y.item(), acc.item()))
                L += l_p_x_given_y

            # break if the loss diverged...easier for poppa to run experiments this way
            if L.abs().item() > 1e5:
                print("BAD BOIIIIIIIIII")
                # 1 / 0
                # just return
                return

            optim.zero_grad()
            L.backward()
            optim.step()
            cur_iter += 1

            if cur_iter % 100 == 0 and arg.gen:
                if arg.plot_uncond:
                    if not arg.uncond:
                        # for sampling new images,
                        y_q = t.randint(0, arg.n_classes, (arg.batch_size,)).to(device)
                        x_q, y_q = sample_q(f, replay_buffer, lgm_loss, y=y_q)
                    else:
                        x_q, y_q = sample_q(f, replay_buffer, lgm_loss)

                    plot('{}/samples/x_q_{}_{:>06d}.png'.format(arg.save_dir, epoch, i), x_q)
                if arg.plot_cond:  # generate class-conditional samples
                    y = t.arange(0, arg.n_classes)[None].repeat(arg.n_classes, 1).transpose(1, 0).contiguous().view(-1).to(device)
                    x_q_y, _ = sample_q(f, replay_buffer, lgm_loss, y=y)
                    plot('{}/samples/x_q_y{}_{:>06d}.png'.format(arg.save_dir, epoch, i), x_q_y)

        if epoch % arg.ckpt_every == 0 and arg.gen and epoch >= 30:
            # save less
            checkpoint(f, replay_buffer, lgm_loss, f'ckpt_{epoch}.pt', arg, device)

        if epoch % arg.eval_every == 0:
            f.eval()
            lgm_loss.eval()
            with t.no_grad():
                # validation set
                correct, loss = lda_eval_classification(f, dload_valid, lgm_loss, "Valid", epoch, arg, wlog)
                t_c = lda_eval_classification(f, dload_test, lgm_loss, "Test", epoch, arg, wlog)
                metrics = {'Acc/Val': t_c[0], 'Loss/Val': t_c[1]}
                if correct > best_valid_acc:
                    best_valid_acc = correct
                    print("Best Valid!: {}".format(correct))
                    checkpoint(f, replay_buffer, lgm_loss, "best_valid_ckpt.pt", arg, device)

            if arg.gen and epoch % 5 == 0:
                fid = eval_fid(replay_buffer, arg)
                if fid > 0:
                    prev_fid = fid
                else:
                    fid = prev_fid
                print('IS {}, fid {}'.format(inc_score, fid))
                arg.writer.add_scalar('Gen/IS', inc_score, epoch)
                arg.writer.add_scalar('Gen/FID', fid, epoch)
                metrics['Gen/FID'] = fid
                if fid < best_fid:
                    best_fid = fid
                    checkpoint(f, replay_buffer, lgm_loss, "best_fid_ckpt.pt", arg, device)
            f.train()
            lgm_loss.train()

            if not arg.debug and not arg.no_wandb:
                init_wandb(arg)
                wandb.log(metrics)
            if not arg.no_wandb:
                wandb.log(metrics)

        checkpoint(f, replay_buffer, lgm_loss, "last_ckpt.pt", arg, device)

    corrects, losses = [], []
    for x_p_d, y_p_d in tqdm(dload_train):
        x_p_d, y_p_d = x_p_d.to(device), y_p_d.to(device)
        feats = f.feature(x_p_d)
        logits = lgm_loss(feats)
        loss = (-t.gather(logits, 1, t.unsqueeze(y_p_d, 1))).detach().cpu().numpy()
        losses.extend(loss)
        correct = (logits.max(1)[1] == y_p_d).float().cpu().numpy()
        corrects.extend(correct)

    loss = np.mean(losses)
    correct = np.mean(corrects)
    gamma2 = loss * 2 / lgm_loss.feat_dim
    print('final train loss %.5g,  accuracy: %g%%, gamma^2 = %.4g, gamma=%.4g' % (loss, correct * 100, gamma2, np.sqrt(gamma2)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("GMMC")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "svhn", "cifar100"])
    parser.add_argument("--data_root", type=str, default="../data")
    # optimization
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--decay_epochs", nargs="+", type=int, default=[60, 100, 120, 135], help="decay learning rate by decay_rate at these epochs")
    parser.add_argument("--decay_rate", type=float, default=.2, help="learning rate decay multiplier")
    parser.add_argument("--labels_per_class", type=int, default=-1, help="number of labeled examples per class, if zero then use all labels")
    parser.add_argument("--optimizer", type=str, default="adam", help="adam, sgd, adam_one, sgd_one")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=150)
    parser.add_argument("--warmup_iters", type=int, default=-1, help="number of iters to linearly increase learning rate, if -1 then no warmmup")

    # regularization
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--sigma", type=float, default=0, help="stddev of gaussian noise to add to input, .03 works but .1 is more stable")
    parser.add_argument("--weight_decay", type=float, default=4e-4)
    # network
    parser.add_argument("--norm", type=str, default=None, choices=[None, "batch", "instance", "layer", "act"], help="norm to add to weights, none works fine")
    # EBM specific
    parser.add_argument("--n_steps", type=int, default=20, help="number of steps of SGLD per iteration, 20 works for PCD")
    parser.add_argument("--width", type=int, default=10, help="WRN width parameter")
    parser.add_argument("--depth", type=int, default=28, help="WRN depth parameter")
    parser.add_argument("--uncond", action="store_true", help="If set, then the EBM is unconditional")
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--reinit_freq", type=float, default=.05)
    parser.add_argument("--sgld_lr", type=float, default=1.0)

    # logging + evaluation
    parser.add_argument("--save_dir", type=str, default='./experiment')
    parser.add_argument("--dir_path", type=str, default='./experiment')
    parser.add_argument("--log_dir", type=str, default='./runs')
    parser.add_argument("--log_arg", type=str, default='method-sgld_lr-n_steps')
    parser.add_argument("--ckpt_every", type=int, default=10, help="Epochs between checkpoint save")
    parser.add_argument("--eval_every", type=int, default=1, help="Epochs between evaluation")
    parser.add_argument("--print_every", type=int, default=100, help="Iterations between print")
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--plot_cond", action="store_true", help="If set, save class-conditional samples")
    parser.add_argument("--plot_uncond", action="store_true", help="If set, save unconditional samples")
    parser.add_argument("--n_valid", type=int, default=5000, help="number of validation samples")

    # new or different parameters
    parser.add_argument("--gen", action="store_true", help="If set, we use generative loss")
    parser.add_argument("--cls", action="store_true", help="If set, we use discriminative loss")
    parser.add_argument("--beta", type=float, default=0.5, help="beta * sampled - real_data")
    parser.add_argument("--class_cond_p_x_sample", action="store_true", help="If set, we keep")
    parser.add_argument("--inject", action="store_true", help="If set, then use Noise Injected Sampling, otherwise use Staged Sampling")
    parser.add_argument("--gamma", type=float, default=0, help="gamma for sampling")
    # MMC
    parser.add_argument("--method", type=str, default="mmc", help="use mmc(fixed mu), but we can also implement lda, dmmc, emmc")
    parser.add_argument("--mu_c", type=float, default=10, help="the constant C for generating means of centers in MMC")

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--exp_name", type=str, default="GMMCPP", help="exp name, for description")
    parser.add_argument("--no_fid", action="store_true", help="If set, evaluate FID/Inception Score")
    parser.add_argument("--no_wandb", action="store_true", help="If set, evaluate FID/Inception Score")
    parser.add_argument("--novis", action="store_true", help="")
    parser.add_argument("--debug", action="store_true", help="")
    parser.add_argument("--gpu-id", type=str, default="0")
    parser.add_argument("--note", type=str, default="")

    args = parser.parse_args()
    assert args.cls or args.gen
    init_env(args, logger)
    args.save_dir = args.dir_path
    os.makedirs('{}/samples'.format(args.dir_path))
    print = wlog
    print(' '.join(sys.argv))
    print(args.dir_path)
    main(args)
    print(args.dir_path)
