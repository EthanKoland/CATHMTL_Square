import os

import torch
from torch.utils.data import DataLoader

from models.inference import Inference
from utils.hyper_para import HyperParametersTest
from datasets.xray_dataset import RandomXRayDataset

# Parse command line arguments
config = HyperParametersTest()
config.parse()

if config['output'] is None:
    config['output'] = f'./outputs/{config["arch"]}_{config["dataset"]}_{config["split"]}'
    print(f'Output path not provided. Defaulting to {config["output"]}')
os.makedirs(config['output'], exist_ok=True)

if config['dataset'] == 'Duke':
    pass
elif config['dataset'] == 'UCL':
    pass
elif config['dataset'] == 'XRay':
    config['data_root'] = os.path.join(config['data_root'], 'XRayCath')
    test_dataset = RandomXRayDataset(config)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
else:
    raise NotImplementedError(f'{config["dataset"]} dataset not implemented')

torch.autograd.set_grad_enabled(False)

processor = Inference(config)

total_process_time = 0
total_frames = 0
total_iter = len(test_loader)

for it, data in enumerate(test_loader):
    img = data['img'].cuda()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(stream=torch.cuda.current_stream())

    segs, dets = processor.step(img)

    end.record(stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    total_process_time += (start.elapsed_time(end) / 1000)
    total_frames += 1

print(f'Total processing time: {total_process_time}')
print(f'Total processed frames: {total_frames}')
print(f'FPS: {total_frames / total_process_time}')
print(f'Max allocated memory (MB): {torch.cuda.max_memory_allocated() / (2**20)}')
