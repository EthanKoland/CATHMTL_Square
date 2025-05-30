#!/bin/bash

source ${miniconda_or_ananconda_root_path}/bin/activate
conda activate cathmtl

export CUDA_VISIBLE_DEVICES=${NUM_GPU}
export OMP_NUM_THREADS=${NUM_THREADS}

arch='odsnet'  # check /models/networks and /models/trainer.py
dataset='XRay' # [Duke, UCL, XRay]
# load_network='weights/snapshot.pth'
data_root='data'
batch_size=${BATCH_SIZE}

echo "Stage 2: Dataset: ${dataset}"
# echo "Load pretrained model from ${load_network}"
python train.py \
    --arch ${arch} \
    --dataset ${dataset} \
    --data_root ${data_root} \
    --batch_size ${batch_size} \
    --multitask
# --load_network ${load_network}
