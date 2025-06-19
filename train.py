import os
import math
import random
import datetime

import numpy as np

import torch
from torch.utils.data import DataLoader

from utils.logger import Logger
from models.trainer import Trainer
from utils.hyper_para import HyperParameters
from datasets.oct_dataset import OCTDataset
from datasets.xray_dataset import RandomXRayDataset, XRayCathUCLDataset
from datasets.square_dataset import Sqaure_Dataset
"""
Initial setup
"""
# Init running environment
print(f'CUDA Device count: {torch.cuda.device_count()}')

# Parse command line arguments
config = HyperParameters()
config.parse()

torch.backends.cudnn.deterministic = True
if config['benchmark']:
    torch.backends.cudnn.benchmark = True

torch.manual_seed(14159265)
torch.cuda.manual_seed_all(14159265)
np.random.seed(14159265)
random.seed(14159265)

network_in_memory = None

if config['exp_id'] == 'NULL':
    config['exp_id'] = 'Debug'
id = '%s_%s_%s_%s' % (datetime.datetime.now().strftime('%b%d_%H.%M.%S'), config['exp_id'], config['arch'], config['dataset'])

os.makedirs(os.path.join(config['log_dirs'], id), exist_ok=True)
"""
Model related
"""
logger = Logger(config, id)
logger.log_string('hyperpara', str(config))
model = Trainer(config, logger=logger, save_path=os.path.join(config['log_dirs'], id, id))

if config['load_checkpoint'] is not None:
    total_iter = model.load_checkpoint(config['load_checkpoint'])
    config['load_checkpoint'] = None
    print('Previously trained model loaded!')
else:
    total_iter = 0

if network_in_memory is not None:
    print('I am loading network from the previous stage')
    model.load_network_in_memory(network_in_memory)
    network_in_memory = None
elif config['load_network'] is not None:
    print('I am loading network from a disk, as listed in configuration')
    model.load_network(config['load_network'])
    config['load_network'] = None
"""
Dataloader related
"""
if config['dataset'] == 'Duke':
    config['data_root'] = os.path.join(config['data_root'], 'DukeData')
    train_dataset = OCTDataset(data_root=config['data_root'], split='train', size=config['size'])
    print(f'Train dataset size: {len(train_dataset)}')
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dataset = OCTDataset(data_root=config['data_root'], split='val', size=config['size'])
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
elif config['dataset'] == 'UCL':
    config['data_root'] = os.path.join(config['data_root'], 'CatheterSegmentationData')
    train_dataset = XRayCathUCLDataset(data_root=config['data_root'], split='train', size=config['size'])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dataset = XRayCathUCLDataset(data_root=config['data_root'], split='val', size=config['size'])
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
elif config['dataset'] == 'XRay':
    config['data_root'] = os.path.join(config['data_root'], 'XRayCath')
    config['split'] = 'train'
    train_dataset = RandomXRayDataset(config=config)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    config['split'] = 'val'
    val_dataset = RandomXRayDataset(config=config)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
elif config['dataset'] == 'XRayNew':
    config['data_root'] = os.path.join(config['data_root'], 'XRayCathNew')
    config['split'] = 'train'
    train_dataset = RandomXRayDataset(config=config)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    config['split'] = 'val'
    val_dataset = RandomXRayDataset(config=config)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
elif config['dataset'] == 'square':
    config['data_root'] = os.path.join(config['data_root'], 'square')
    train_dataset = OCTDataset(data_root=config['data_root'], split='train', size=config['size'])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dataset = OCTDataset(data_root=config['data_root'], split='val', size=config['size'])
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
else:
    raise NotImplementedError(f'{config["dataset"]} dataset not implemented')
"""
Determine max epoch
"""
total_epoch = math.ceil(config['iterations'] / len(train_loader))
current_epoch = total_iter // len(train_loader)
print(f'We approximately use {total_epoch} epochs.')
"""
Starts training
"""
finetuning = False
# Need this to select random bases in different workers
# np.random.seed(np.random.randint(2**30-1) + local_rank*100)
try:
    while total_iter < config['iterations'] + config['finetune']:
        # Crucial for randomness!
        current_epoch += 1
        print(f'Current epoch: {current_epoch}')

        # Train loop
        model.train()
        for data in train_loader:

            # fine-tune means fewer augmentations to train the sensory memory
            if config['finetune'] > 0 and not finetuning and total_iter >= config['iterations']:
                finetuning = True
                model.save_network_interval = 1000
                break

            model.do_pass(data, total_iter)
            total_iter += 1

            if total_iter >= config['iterations'] + config['finetune']:
                break

finally:
    if model.logger is not None and total_iter > 5000:
        model.save_network(total_iter)
        model.save_checkpoint(total_iter)

network_in_memory = model.model.state_dict()
