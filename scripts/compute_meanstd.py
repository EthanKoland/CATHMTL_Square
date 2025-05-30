import os
import glob

import torch
import numpy as np
from PIL import Image

data_dir = 'data/XRay'

mean_list = []
std_list = []

img_list = sorted(glob.glob(os.path.join(data_dir, 'train', 'images', '*.jpg')))
for idx, img_name in enumerate(img_list):
    img = np.array(Image.open(img_name)).flatten()
    mean_list.append(np.mean(img))
    std_list.append(np.std(img))

img_list = sorted(glob.glob(os.path.join(data_dir, 'val', 'images', '*.jpg')))
for idx, img_name in enumerate(img_list):
    img = np.array(Image.open(img_name)).flatten()
    mean_list.append(np.mean(img))
    std_list.append(np.std(img))

mean = np.mean(mean_list)
# std =
print()
