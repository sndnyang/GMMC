# GMMC

Pytorch Implementation for the paper ["Generative Max-Mahalanobis Classifiers for Image Classification, Generation and More"](https://arxiv.org/abs/2101.00122). The implementation is based on [JEM](https://github.com/wgrathwohl/JEM) of Grathwohl et al. (2020).



<!--## Overview

The pipeline of GMMC:

![pipeline](figs/GMMC.png)

<img src="figs/tSNE.png" width="70%" alt="latent space" style="zoom:50%;" /> -->



# Installation

```markdown
pip install -r requirements.txt
```

# Usage

## Training

To train a model on CIFAR10 as in the paper, please refer to scripts/cifar10_dis.sh, cifar10_gen.sh, cifar10_joint.sh

```markdown
python train_gmmc.py --lr=.0001 --dataset=cifar10 --optimizer=adam \
  --sigma=.03 --width=10 --depth=28 --plot_uncond \
  --print_every=100 --n_epochs=100 \
  --warmup_iters 1000 \
  --log_arg=MMC_gen-sgld_lr-mu_c-n_steps-buffer_size  \
  --sgld_lr=1 --log_dir=./run \
  --gpu-id=0 \
  --method=mmc \
  --mu_c=10 \
  --gamma=0.001 \
  --n_steps=20 \
  --buffer_size=100000 \
  --reinit_freq=0.025 \
  --class_cond_p_x_sample \
  --vis \
  --decay_epochs 30 50 80 \
  --beta=0.5 \
  --generative \
  --start_generative=0
```

Two pretrained models on CIFAR10 can be found at

https://mygsu-my.sharepoint.com/:f:/g/personal/xyang22_gsu_edu/EsA3KMmmK7hOqZST2GufHYUBLxAoXPtKDCMzJv1x20W6HQ?e=N7785i

A: Accuracy 94.08, FID 37.0
B: Accuracy 92.51, FID 35.96

## Evaluation

To evaluate the classifier:
```markdown
python eval_gmmc.py --load_path /PATH/TO/YOUR/MODEL.pt --eval test_clf --dataset cifar_test --gpu-id 0
```

To evaluate FID of the samples in replay buffer
```shell script
python eval_gmmc.py --load_path /PATH/TO/YOUR/MODEL.pt --eval fid --ratio 9000 --gpu-id 0

  ratio: if ratio < 1, use the percentile for each category; if ratio > 1, choose int(ratio) or all from each category
  ex. if buffer size is 100k, we select the top 9000 samples from each category as there is a small percentage of init noisy samples in the buffer.
```

To generate conditional samples from a saved replay buffer

```shell script
python eval_gmmc.py --load_path /PATH/TO/YOUR/MODEL.pt --eval cond_samples  --gpu-id 0
```

To generate new unconditional samples
```shell script
python eval_gmmc.py --load_path /PATH/TO/YOUR/MODEL.pt --eval uncond_samples --n_sample_steps 100 --buffer_size 100 --n_steps 40 --print_every 10 --gpu-id 0
```

To generate new conditional samples

```shell script
python eval_gmmc.py --load_path /PATH/TO/YOUR/MODEL.pt --eval cond_samples --n_sample_steps 100 --buffer_size 100 --n_steps 40 --print_every 10 --fresh_samples --gpu-id 0
```



<!-- ## Generated Samples

| ![a](figs/topk_0.png) | ![a](figs/topk_1.png) | ![a](figs/topk_2.png) | ![a](figs/topk_3.png) | ![a](figs/topk_4.png) |
| --------------------- | --------------------- | --------------------- | --------------------- | --------------------- |
| ![a](figs/topk_5.png) | ![a](figs/topk_6.png) | ![a](figs/topk_7.png) | ![a](figs/topk_8.png) | ![a](figs/topk_9.png) |
-->


# Citation

If you found this work useful and used it on your own research, please concider citing this paper.
```
@article{yang2021gmmc,
    title={Generative Max-Mahalanobis Classifiers for Image Classification, Generation and More},
    author={Xiulong Yang and Hui Ye and Yang Ye and Xiang Li and Shihao Ji},
    journal={The European Conference on Machine Learning (ECML)},
    year={2021}
}
```
