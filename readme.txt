conda activate torch

//For training Duke and UCL data, disable multitask option in train.sh or rtx6000.sh

//UEA HPC
squeue -u ksh22kfu
//see how much GPU memory is used, use your jobid
srun --jobid=20278914  -n1 nvidia-smi


//view tensorboard
tensorboard --logdir=./
use browser to view result http://localhost:6006/


////////////////////////Stage 1 Duke/////////////////
////////////////////////Stage 1 UCL use pre-train model:odsnet_Duke_3000.pth/////////////////
arch='odsnet'  # check /models/networks and /models/trainer.py
dataset='Duke' # [Duke, UCL, XRay]
# load_network='weights/odsnet_Duke_3000.pth'
data_root='data'
batch_size=128
# batch_size=64 for UCL dataset

echo "Stage 1: Dataset: ${dataset}"
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
    --iterations 4000 \
    --finetune 0 \
    --save_network_interval 1000 \
    --save_checkpoint_interval 2000

///////////// Visualize the result ///////////////
////Install NMS command
cd ./external
pip install Cython
Python setup.py build_ext --inplace


/////////////ENV setup//////////////////////////////
python3.12 -m venv .venv
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install ml_collections matplotlib opencv-python pandas pyyaml tensorboard scipy