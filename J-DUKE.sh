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
module add python/anaconda/2024.06/
module add cuda/11.8
nvidia-smi

conda init
conda activate torch_lts

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1

arch='odsnet'  # check /models/networks and /models/trainer.py
dataset='Duke' # [Duke, UCL, XRay]
# load_network='weights/snapshot.pth'
data_root='data'
batch_size=128

echo "Stage 1: Dataset: ${dataset}"
# echo "Load pretrained model from ${load_network}"
python train.py \
    --arch ${arch} \
    --dataset ${dataset} \
    --data_root ${data_root} \
    --batch_size ${batch_size} #\
    #--multitask
# --load_network ${load_network}
 
  


