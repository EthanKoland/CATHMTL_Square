#!/bin/bash

source ${miniconda_or_ananconda_root_path}/bin/activate
conda activate cathmtl

export CUDA_VISIBLE_DEVICES=${NUM_GPU}
export OMP_NUM_THREADS=${NUM_THREADS}

arch='odsnet'  # check /models/networks and /models/trainer.py
dataset='XRay' # [Duke, UCL, XRay]
weigths=${pretrained_weight}  # e.g. 'weights/snapshot.pth'
data_root='data'

python test.py \
    --arch ${arch} \
    --dataset ${dataset} \
    --data_root ${data_root} \
    --weights ${weigths} \
    --multitask
