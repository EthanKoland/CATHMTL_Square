#!/bin/bash
#SBATCH --mail-type=NONE                 # Mail events (NONE,BEGIN,END,FAIL,ALL)
#SBATCH --nodes=1                        # limit to one node
#SBATCH -p gpu-rtx6000-2                 # Where queue to use
#SBATCH --gres=gpu:1                     # number of GPUs per node
#SBATCH --qos=gpu-rtx                    # gpu or gpu for non rtx6000 nodes
#SBATCH --mem=80G
#SBATCH -c 12                            # memory
#SBATCH --time=7-00:00                   # time (DD-HH:MM)
#SBATCH --job-name=gpu-will_job          # Job name
#SBATCH -o gpu-test-%j.out               # Standard output log
#SBATCH -e gpu-test-%j.err               # Standard error log
module add python/anaconda/2023.07/3.11.4
module add cuda/11.8
nvidia-smi

source activate torch

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1

arch='odsnet'  # check /models/networks and /models/trainer.py
dataset='XRay' # [Duke, UCL, XRay]
load_network='weights/odsnet_UCL_3000.pth'
data_root='data'
batch_size=64

echo "Stage 2: Dataset: ${dataset}"
echo "Load pretrained model from ${load_network}"
# --iterations 3000 for X-ray
# --finetune 1000 \
# --iterations 4000 for pre-train
# --finetune 0 \
python train.py \
    --arch ${arch} \
    --dataset ${dataset} \
    --data_root ${data_root} \
    --batch_size ${batch_size} \
    --iterations 3000 \
    --finetune 1000 \
    --save_network_interval 1000 \
    --save_checkpoint_interval 2000 \
    --multitask
# --load_network ${load_network}
 
  

