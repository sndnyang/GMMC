
#

python train_gmmcpp.py --lr=.03 --dataset=cifar10 --optimizer=sgd \
  --sigma=.0 --width=10 --depth=28 --plot_uncond --warmup_iters=1000 \
  --print_every=100 --n_epochs=150 --decay_epochs 50 80 100 120 \
  --log_arg=Gen-beta-mu_c  \
  --sgld_lr=1 --log_dir=./run \
  --method=mmc \
  --beta 0.5 \
  --mu_c=10 \
  --norm=batch \
  --buffer_size=1000 \
  --n_steps 5 \
  --no_wandb \
  --cls \
  --gen \
  --no_wandb \
  --gpu-id=6

#  --inject
