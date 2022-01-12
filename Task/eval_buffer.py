import os
import torch as t
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm


def norm_ip(img, min, max):
    temp = t.clamp(img, min=min, max=max)
    temp = (temp + -min) / (max - min + 1e-5)
    return temp


def eval_fid(replay_buffer, args):
    from Task.fid import get_fid_score
    if isinstance(replay_buffer, list):
        images = replay_buffer[0]
    elif isinstance(replay_buffer, tuple):
        images = replay_buffer[0]
    else:
        images = replay_buffer

    feed_imgs = []
    for i, img in enumerate(images):
        n_img = norm_ip(img, -1, 1)
        new_img = n_img.cpu().numpy().transpose(1, 2, 0) * 255
        feed_imgs.append(new_img)
    feed_imgs = np.stack(feed_imgs)

    from Task.data import Cifar10
    test_dataset = Cifar10(args, full=True, noise=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0, shuffle=True, drop_last=False)

    test_ims = []

    def rescale_im(im):
        return np.clip(im * 256, 0, 255).astype(np.uint8)

    for data_corrupt, data, label_gt in test_dataloader:
        data = data.numpy()
        test_ims.extend(list(rescale_im(data)))

    # FID score
    print("Computing FID")
    fid = get_fid_score(feed_imgs, test_ims)
    print("FID of score {}".format(fid))
    return fid
