
#

python train_gmmc.py --lr=.0001 --dataset=cifar10 --optimizer=adam \
  --sigma=.03 --width=10 --depth=28 --plot_uncond --warmup_iters=1000 \
  --print_every=100 --n_epochs=150 --decay_epochs 30 50 80 100 120 \
  --log_arg=MMC_dis-mu_c  \
  --sgld_lr=1 --log_dir=./run \
  --gpu-id=1 \
  --method=mmc \
  --mu_c=10 \
  --buffer_size=1000 \
  --vis \
  --start_generative=150

#  --inject
