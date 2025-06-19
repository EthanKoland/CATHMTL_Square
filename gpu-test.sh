#!/bin/bash
#SBATCH --mail-type=NONE                 # Mail events (NONE,BEGIN,END,FAIL,ALL)
#SBATCH --nodes=1                        # limit to one node
#SBATCH -p gpu-rtx6000-2                 # Where queue to use
#SBATCH --gres=gpu:1                     # number of GPUs per node
#SBATCH --qos=gpu-rtx                    # gpu or gpu for non rtx6000 nodes
#SBATCH --mem=80G
#SBATCH -c 12                            # memory
#SBATCH --time=0-01:00                   # time (DD-HH:MM)
#SBATCH --job-name=CATHMTL_EnvTest         # Job name
#SBATCH --output=R-out-%x-%j.out          #Standard output log
#SBATCH --error=R-err-%x-%j.err          #Standard error log
module add python/anaconda/2024.06/
module add cuda/11.8
nvidia-smi

source .venv/bin/activate