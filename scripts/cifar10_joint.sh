
#

python train_gmmc.py --lr=.0001 --dataset=cifar10 --optimizer=adam \
  --sigma=.03 --width=10 --depth=28 --plot_uncond --warmup_iters=1000 \
  --print_every=100 --n_epochs=150 --decay_epochs 30 50 80 100 120 \
  --log_arg=MMC_joint-sgld_lr-mu_c-n_steps-start_generative \
  --sgld_lr=1 --log_dir=./run \
  --gpu-id=1 \
  --method=mmc \
  --mu_c=10 \
  --gamma=0.001 \
  --n_steps=50 \
  --buffer_size=50000 \
  --reinit_freq=0.05 \
  --class_cond_p_x_sample \
  --vis \
  --beta=0.5 \
  --generative \
  --start_generative=50

#  --inject
