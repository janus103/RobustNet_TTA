#!/usr/bin/env bash
    # Example on Cityscapes
     python train.py \
        --dataset cityscapes \
        --covstat_val_dataset cityscapes \
        --val_dataset cityscapes \
        --arch network.deepv3.DeepR50V3PlusD_DWT \
        --city_mode 'train' \
        --lr_schedule poly \
        --lr 0.01 \
        --poly_exp 0.9 \
        --max_cu_epoch 10000 \
        --class_uniform_pct 0.5 \
        --class_uniform_tile 1024 \
        --crop_size 768 \
        --scale_min 0.5 \
        --scale_max 2.0 \
        --rrotate 0 \
        --max_iter 160000 \
        --bs_mult 2 \
        --gblur \
        --color_aug 0.5 \
        --wt_reg_weight 0.6 \
        --relax_denom 0.0 \
        --clusters 2 \
        --cov_stat_epoch 5 \
        --trials 10 \
        --wt_layer 0 0 2 2 2 0 0 \
        --date 0101 \
        --exp r50os16_city_isw \
        --ckpt ./logs/ \
        --tb_path ./logs/
