import os
import sys
import json
import time
import shutil
import signal
import logging
from functools import partial

import torch
import numpy as np
import tensorboardX as tbX

logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(filename)s[%(lineno)d]: %(message)s", datefmt="%m-%d %H:%M:%S")
logger = logging.getLogger()
logger.setLevel(logging.INFO)
wlog = logger.info


def init_wandb(args):
    import wandb
    wandb.init(project='RefineEBM')
    name = args.note
    if name:
        wandb.run.name = args.note + str(os.getpid())
        wandb.run.save()
    args.pid = os.getpid()
    args.node = os.uname().nodename.split('.')[0]
    wandb.config.update(args)


def init_env(args, exp_logger):
    # 1. debug -> num_workers
    init_debug(args)
    args.vis = not args.novis

    # 2. select gpu
    auto_select_gpu(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    args.dir_path = form_dir_path(args.exp_name, args)
    set_file_logger(exp_logger, args)
    init_logger_board(args)

    args.n_classes = 10
    if args.dataset == "cifar100":
        args.n_classes = 100
    if args.dataset == "tinyimagenet":
        args.n_classes = 200


def init_debug(args):
    # verify the debug mode. pytorch loader has a parameter num_workers, in debug mode, it should be 0  so set args.debug
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace is None:
        print('No sys.gettrace')
        args.debug = False
    elif gettrace():
        print('Hmm, Big Debugger is watching me')
        args.debug = True
    else:
        args.debug = False


def auto_select_gpu(args):
    if args.gpu_id:
        return
    try:
        import GPUtil
    except ImportError:
        wlog("please install GPUtil for automatically selecting GPU")
        args.gpu_id = '1'
        return

    if len(GPUtil.getGPUs()) == 0:
        return
    id_list = GPUtil.getAvailable(order="load", maxLoad=0.7, maxMemory=0.9, limit=8)
    if len(id_list) == 0:
        print("GPU memory is not enough for predicted usage")
        raise NotImplementedError
    args.gpu_id = str(id_list[0])


def init_logger_board(args):
    if 'vis' in vars(args) and args.vis:
        args.writer = tbX.SummaryWriter(log_dir=args.dir_path)


def vlog(writer, cur_iter, set_name, wlog=None, verbose=True, **kwargs):
    for k in kwargs:
        v = kwargs[k]
        writer.add_scalar('%s/%s' % (set_name, k.capitalize()), v, cur_iter)
    if wlog:
        my_print = wlog
    else:
        my_print = print
    if not verbose:
        prompt = "%d " % cur_iter
        prompt += ','.join("%s: %.4f" % (k, kwargs[k]) for k in ['loss', 'acc', 'acc1', 'acc5'] if k in kwargs)
        my_print(prompt)


def set_file_logger(exp_logger, args):
    # Just use "logger" above
    # use tensorboard + this function to substitute ExpSaver
    args_dict = vars(args)
    dir_path = args.dir_path
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    with open(os.path.join(dir_path, "para.json"), "w") as fp:
        json.dump(args_dict, fp, indent=4, sort_keys=True)
    logfile = os.path.join(dir_path, "exp.log")
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d]: %(message)s")
    fh.setFormatter(formatter)
    exp_logger.addHandler(fh)
    copy_script_to_folder(sys.argv[0], args.dir_path)


def list_args(args):
    for e in sorted(vars(args).items()):
        print("args.%s = %s" % (e[0], e[1] if not isinstance(e[1], str) else '"%s"' % e[1]))


def form_dir_path(task, args):
    """
    Params:
        task: the name of your experiment/research
        args: the namespace of argparse
            requires:
                --dataset: always need a dataset.
                --log-arg: the details shown in the name of your directory where logs are.
                --log-dir: the directory to save logs, default is ~/projecct/runs.
    """
    args.pid = os.getpid()
    args_dict = vars(args)
    if "log_dir" not in args_dict:
        args.log_dir = ""
    if "log_arg" not in args_dict:
        args.log_arg = ""

    run_time = time.strftime('%m%d%H%M%S', time.localtime(time.time()))
    log_arg_list = []
    if args.debug:
        task += '-debug'
    for e in args.log_arg.split("-"):
        v = args_dict.get(e, None)
        if v is None:
            log_arg_list.append(str(e))
        elif isinstance(v, str):
            log_arg_list.append(str(v))
        else:
            log_arg_list.append("%s=%s" % (e, str(v)))
    args.exp_marker = exp_marker = "-".join(log_arg_list)
    exp_marker = "%s/%s/%s@%s@%d" % (args.dataset, task, run_time, exp_marker, os.getpid())
    base_dir = os.path.join(os.environ['HOME'], 'project/runs') if not args.log_dir else args.log_dir
    dir_path = os.path.join(base_dir, exp_marker)
    return dir_path


def summary(data):
    assert isinstance(data, np.ndarray) or isinstance(data, torch.Tensor)
    wlog("shape: %s, num of points: %d, pixels: %d" % (str(data.shape), data.shape[0], np.prod(data.shape[1:])))
    wlog("max: %g, min %g" % (data.max(), data.min()))
    wlog("mean: %g" % data.mean())
    wlog("mean of abs: %g" % np.abs(data).mean())
    wlog("mean of square sum: %g" % (data ** 2).mean())


def remove_outliers(x, outlier_constant=1.5):
    a = np.array(x)
    upper_quartile = np.percentile(a, 75)
    lower_quartile = np.percentile(a, 25)
    iqr = (upper_quartile - lower_quartile) * outlier_constant
    quartile_set = (lower_quartile - iqr, upper_quartile + iqr)

    result = a[np.where((a >= quartile_set[0]) & (a <= quartile_set[1]))]

    return result


def vis_step(writer, step, dicts):
    """
    Add several curves.
    """
    for k in dicts:
        writer.add_scalar(k, dicts[k], step)


def copy_script_to_folder(caller_path, folder):
    '''copy script'''
    script_filename = caller_path.split('/')[-1]
    script_relative_path = os.path.join(folder, script_filename)
    shutil.copy(caller_path, script_relative_path)
    for file in ['utils.py']:
        shutil.copy(file, folder)


def time_string():
    '''convert time format'''
    ISOTIMEFORMAT='%Y-%m-%d %X'
    string = '[{}]'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
    return string


def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600*need_hour) / 60)
    need_secs = int(epoch_time - 3600*need_hour - 60*need_mins)
    return need_hour, need_mins, need_secs


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name='', fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
