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
from tqdm import tqdm
from ExpUtils import *
from utils import *
from models.mmc_models import *
from Utils.mmc_utils import cal_center
from LDALayer import LDALayer
t.backends.cudnn.benchmark = True
t.backends.cudnn.enabled = True
seed = 1


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


def get_sample_q(args, device):
    def sample_p_0(replay_buffer, bs, y=None):
        if len(replay_buffer) == 0:
            return init_random_x_y(args, bs), []
        buffer_size = len(replay_buffer[0]) if y is None else len(replay_buffer[0]) // args.n_classes
        inds = t.randint(0, buffer_size, (bs,))
        if y is not None:
            # if cond, convert inds to class conditional inds
            inds = y.cpu() * buffer_size + inds
        buffer_x = replay_buffer[0][inds]
        buffer_y = replay_buffer[1][inds]
        random_samples = init_random_x_y(args, bs)
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
        f.eval()
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


def main(args):

    # discriminative training
    # generative training
    # joint training
    assert not(args.generative and args.start_generative > args.n_epochs)

    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    args.device = device

    sample_q = get_sample_q(args, device)

    f, replay_buffer, lgm_loss = get_model_and_buffer(args, device)
    if not args.load_path and args.method in ['mmc', 'dmmc']:
        print('initialized with MMC')
        mu = cal_center(n_classes=lgm_loss.num_classes, dim=lgm_loss.feat_dim, c=args.mu_c)
        lgm_loss.centers.data = mu.to(device)

    def sqrt(x): return int(t.sqrt(t.Tensor([x])))
    def plot(p, x): return tv.utils.save_image(t.clamp(x, -1, 1), p, normalize=True, nrow=sqrt(x.size(0)))

    # optimizer
    params = f.parameters()
    if "adam" in args.optimizer:
        print('adam')
        optim = t.optim.Adam(params, lr=args.lr, betas=[.9, .999], weight_decay=args.weight_decay)
    else:
        print('sgd')
        optim = t.optim.SGD(params, lr=args.lr, momentum=.9, weight_decay=args.weight_decay, nesterov=True)

    # datasets
    dload_train, dload_train_labeled, dload_valid, dload_test = get_data(args)

    best_valid_acc = 0.0
    cur_iter = 0
    # trace learning rate
    new_lr = args.lr
    for epoch in range(args.n_epochs):
        if epoch in args.decay_epochs:
            for param_group in optim.param_groups:
                new_lr = param_group['lr'] * args.decay_rate
                param_group['lr'] = new_lr
            print("Decaying lr to {}".format(new_lr))

        for i, (x_p_d, y_p_d) in tqdm(enumerate(dload_train)):
            if cur_iter <= args.warmup_iters:
                lr = args.lr * cur_iter / float(args.warmup_iters)
                for param_group in optim.param_groups:
                    param_group['lr'] = lr

            L = 0.
            if args.generative and epoch >= args.start_generative:
                # generative training
                x_p_d = x_p_d.to(device)
                y_p_d = y_p_d.to(device)

                fp_all = f.feature(x_p_d)
                fp = lgm_loss.P_X_y(fp_all, y=y_p_d)
                if args.class_cond_p_x_sample:
                    x_q, y_q = sample_q(f, replay_buffer, lgm_loss, y=y_p_d)
                else:
                    x_q, y_q = sample_q(f, replay_buffer, lgm_loss)

                fq_all = f.feature(x_q)
                fq = lgm_loss.P_X_y(fq_all, y=y_q)
                l_p_x = (fp - args.beta * fq)
                # trace the loss curve
                if args.vis:
                    args.writer.add_scalar('Train/fp', fp, cur_iter)
                    args.writer.add_scalar('Train/fq', fq, cur_iter)
                    args.writer.add_scalar('Train/Loss', l_p_x, cur_iter)
                if cur_iter % args.print_every == 0:
                    print('{} P(x) | {}:{:>d} f(x_p_d)={:>14.9f} f(x_q)={:>14.9f} d={:>14.9f}'.format(args.pid, epoch, i, fp, fq, l_p_x))

                L += l_p_x

            if not args.generative or (args.generative and epoch < args.start_generative):  # maximize log p(x | y)
                # discriminative training
                x_lab, y_lab = dload_train_labeled.__next__()
                x_lab, y_lab = x_lab.to(device), y_lab.to(device)

                features = f.feature(x_lab)
                logits = lgm_loss(features)

                l_p_x_given_y = (-t.gather(logits, 1, t.unsqueeze(y_lab, 1))).mean()
                if cur_iter % args.print_every == 0:
                    acc = (logits.max(1)[1] == y_lab).float().mean()
                    print('P(y|x) {}:{:>d} loss={:>14.9f}, acc={:>14.9f}'.format(epoch, cur_iter, l_p_x_given_y.item(), acc.item()))
                L += l_p_x_given_y

            # break if the loss diverged...easier for poppa to run experiments this way
            if L.abs().item() > 1e8:
                print("BAD BOIIIIIIIIII")
                # 1 / 0
                # just return
                return

            optim.zero_grad()
            L.backward()
            optim.step()
            cur_iter += 1

            if cur_iter % 100 == 0 and args.generative and epoch >= args.start_generative:
                if args.plot_uncond:
                    if not args.uncond:
                        # for sampling new images,
                        y_q = t.randint(0, args.n_classes, (args.batch_size,)).to(device)
                        x_q, y_q = sample_q(f, replay_buffer, lgm_loss, y=y_q)
                    else:
                        x_q, y_q = sample_q(f, replay_buffer, lgm_loss)

                    plot('{}/x_q_{}_{:>06d}.png'.format(args.save_dir, epoch, i), x_q)
                if args.plot_cond:  # generate class-conditional samples
                    y = t.arange(0, args.n_classes)[None].repeat(args.n_classes, 1).transpose(1, 0).contiguous().view(-1).to(device)
                    x_q_y, _ = sample_q(f, replay_buffer, lgm_loss, y=y)
                    plot('{}/x_q_y{}_{:>06d}.png'.format(args.save_dir, epoch, i), x_q_y)

        if epoch % args.ckpt_every == 0 and args.generative and epoch >= args.start_generative - 15:
            # save less
            checkpoint(f, replay_buffer, lgm_loss, f'ckpt_{epoch}.pt', args, device)

        if epoch % args.eval_every == 0:
            f.eval()
            lgm_loss.eval()
            with t.no_grad():
                # validation set
                correct, loss = lda_eval_classification(f, dload_valid, lgm_loss, "Valid", epoch, args, wlog)
                t_c = lda_eval_classification(f, dload_test, lgm_loss, "Test", epoch, args, wlog)
                if correct > best_valid_acc:
                    best_valid_acc = correct
                    print("Best Valid!: {}".format(correct))
                    checkpoint(f, replay_buffer, lgm_loss, "best_valid_ckpt.pt", args, device)
            f.train()
            lgm_loss.train()
        checkpoint(f, replay_buffer, lgm_loss, "last_ckpt.pt", args, device)

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
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decay_epochs", nargs="+", type=int, default=[50, 60, 100], help="decay learning rate by decay_rate at these epochs")
    parser.add_argument("--decay_rate", type=float, default=.3, help="learning rate decay multiplier")
    parser.add_argument("--labels_per_class", type=int, default=-1, help="number of labeled examples per class, if zero then use all labels")
    parser.add_argument("--optimizer", type=str, default="adam", help="adam, sgd, adam_one, sgd_one")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=150)
    parser.add_argument("--warmup_iters", type=int, default=-1, help="number of iters to linearly increase learning rate, if -1 then no warmmup")

    # regularization
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--sigma", type=float, default=3e-2, help="stddev of gaussian noise to add to input, .03 works but .1 is more stable")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    # network
    parser.add_argument("--norm", type=str, default=None, choices=[None, "batch", "instance", "layer", "act"], help="norm to add to weights, none works fine")
    # EBM specific
    parser.add_argument("--n_steps", type=int, default=20, help="number of steps of SGLD per iteration, 20 works for PCD")
    parser.add_argument("--width", type=int, default=10, help="WRN width parameter")
    parser.add_argument("--depth", type=int, default=28, help="WRN depth parameter")
    parser.add_argument("--uncond", action="store_true", help="If set, then the EBM is unconditional")
    parser.add_argument("--buffer_size", type=int, default=100000)
    parser.add_argument("--reinit_freq", type=float, default=.025)
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
    parser.add_argument("--vis", action="store_true", help="use tensorboard to trace the log")
    parser.add_argument("--gpu-id", type=str, default="0")

    # new or different parameters
    parser.add_argument("--generative", action="store_true", help="If set, we use generative training after start_generative")
    parser.add_argument("--start_generative", type=int, default=0, help="use joint training: transfer to generative training")
    parser.add_argument("--beta", type=float, default=0.5, help="beta * sampled - real_data")
    parser.add_argument("--class_cond_p_x_sample", action="store_true", help="If set, we keep")
    parser.add_argument("--inject", action="store_true", help="If set, then use Noise Injected Sampling, otherwise use Staged Sampling")
    parser.add_argument("--gamma", type=float, default=0, help="gamma for sampling")
    # MMC
    parser.add_argument("--method", type=str, default="mmc", help="use mmc(fixed mu), but we can also implement lda, dmmc, emmc")
    parser.add_argument("--mu_c", type=float, default=10, help="the constant C for generating means of centers in MMC")

    args = parser.parse_args()
    args.n_classes = 100 if args.dataset == "cifar100" else 10
    args.pid = os.getpid()
    # set environment
    auto_select_gpu(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    init_debug(args)
    # init log directory
    args.dir_path = form_dir_path('GMMC', args)
    args.method = args.method.lower()
    set_file_logger(logger, args)
    args.save_dir = args.dir_path
    init_logger_board(args)
    os.makedirs('{}/energy'.format(args.dir_path))
    print = wlog
    print(args.dir_path)
    main(args)
    print(args.dir_path)
