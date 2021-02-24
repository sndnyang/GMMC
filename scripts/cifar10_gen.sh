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

#  --inject
