# GMMC

Pytorch Implementation for the paper "Generative Max-Mahalanobis Classifiers for Image Classification, Generation and More".


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
  --print_every=100 --n_epochs=150 \
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
  --decay_epochs 40 80 120 \
  --beta=0.5 \
  --generative \
  --start_generative=0
```

A pretrained model on CIFAR10 can be found at

https://drive.google.com/file/d/1n13S_Ni_xg8tMBjmYNRnk97ej4UlMTuy/view?usp=sharing (1.3GB)

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
python eval_gmmc.py --load_path /PATH/TO/YOUR/MODEL.pt --eval uncond_samples --n_sample_steps {THE_MORE_THE_BETTER (10000 minimum)} --buffer_size 100000 --n_steps 40 --print_every 100 --gpu-id 0
```

To generate new conditional samples

```shell script
python eval_gmmc.py --load_path /PATH/TO/YOUR/MODEL.pt --eval cond_samples --n_sample_steps {THE_MORE_THE_BETTER (10000 minimum)} --buffer_size 100000 --n_steps 40 --print_every 100 --fresh_samples --gpu-id 0
```

To generate new unconditional samples
```shell script
python eval_gmmc.py --load_path model1/best_valid_ckpt.pt --eval uncond_samples --n_sample_steps 200 --buffer_size 100 --n_steps 40 --print_every 10 --sgld_lr 1 --gamma 5 --inject --gpu-id 7
```

