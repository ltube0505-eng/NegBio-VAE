#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python dis_main.py --dataset MNIST --model_type negbio --reparam_type gumbel --kl mc  --max_epochs 200
