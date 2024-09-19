#!/bin/sh

dataset='yelp'
model='exbert'
seed=1111
CUDA_VISIBLE_DEVICES=0 python main.py --model-name ${model} --dataset ${dataset} --seed ${seed} --fold 0 --cuda --log-to-file
