import os

import numpy as np
import pandas as pd
from PIL import Image

from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from datasets.utils import reseed, draw_umich_gaussian, draw_dense_reg


class Sqaure_Dataset(Dataset):

    def __init__(self, config):
        super(Sqaure_Dataset, self).__init__()

        #The dictionary containing the configuration parameters defined with the argPareser script
        #in utils/hyper_para.py
        self.config = config

        #The base directory where the dataset is stored defined in the configuration file
        self.data_root = config['data_root']
        
        
        self.split = config['split']

        #2x1 size of the images in the dataset the default is 384x384
        #The dataset images are 512, 512 native resoltion
        self.size = config['size']

        #Not sure what down ratio is, but it is staticallya defined as 4
        self.down_ratio = config['down_ratio']
        self.multitask = config['multitask']
        self.output_size = [self.size[0] // self.down_ratio, self.size[1] // self.down_ratio]
        self.max_objs = 50  # TODO: need to modify by computing the dataset parameters
        self.radius_base = 10

        #TODO: Rework this to work with the CSV file
        csvPath = os.path.join(self.data_root, "train.csv" if self.split == 'train' else "val.csv")

        self.dataEntries = []
        with open(csvPath, 'r') as csvFile:
            csvLines = csvFile.readlines()
            self.dataEntries = [line.strip().split(',') for line in csvLines[1:]]  # Skip header
        
        self.img_dual_transform = transforms.Compose(
            [
                transforms.RandomAffine(degrees=15, shear=10, interpolation=InterpolationMode.BILINEAR, fill=int(128)),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(self.size, interpolation=InterpolationMode.BILINEAR)
            ]
        )

        self.gt_dual_transform = transforms.Compose(
            [
                transforms.RandomAffine(degrees=15, shear=10, interpolation=InterpolationMode.NEAREST, fill=int(255)),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(self.size, interpolation=InterpolationMode.NEAREST)
            ]
        )
        self.val_im_dual_transform = transforms.Compose(
            [transforms.Resize(self.size, interpolation=InterpolationMode.BILINEAR)]
        )

        self.val_gt_dual_transform = transforms.Compose([transforms.Resize(self.size, interpolation=InterpolationMode.NEAREST)])

        self.im_normalize_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0.5, std=0.5)])

        self.gt_normalize_transform = transforms.Compose([transforms.ToTensor()])



    def __len__(self):
        return len(self.dataEntries)

    def __getitem__(self, idx):
        CT_Scan, X_Angle, DRR, Segmentation_DRR, Centerline = self.dataEntries[idx]

        # Load the images and masks
        # ODSnet takes in a 3 channel RGB image
        img = Image.open(os.path.join(self.data_root, DRR)).convert('RGB')
        mask = Image.open(os.path.join(self.data_root, Segmentation_DRR))
        centerLine = Image.open(os.path.join(self.data_root, Centerline))

        
        if self.multitask:
            return {
                'img': img,
                'mask': mask.long(),
                'heatmap_keypoint': centerLine[2, :, :],
                'heatmap_endpoints': centerLine[1, :, :],
                'centerLine': centerLine[0, :, :],
            }
        else:
            return {'img': img, 'mask': mask.long()}


class XRayCathUCLDataset(Dataset):

    def __init__(self, data_root, split='train', size=[256, 256]):
        super(XRayCathUCLDataset, self).__init__()

        self.data_root = data_root
        self.split = split

        with open(os.path.join(data_root, 'Phantom', 'ImageSets', split + '.txt')) as f:
            name_list = f.readlines()

        # data transform
        self.img_dual_transform = transforms.Compose(
            [
                transforms.RandomAffine(degrees=15, shear=10, interpolation=InterpolationMode.BILINEAR, fill=int(128)),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(size, interpolation=InterpolationMode.BILINEAR)
            ]
        )

        self.gt_dual_transform = transforms.Compose(
            [
                transforms.RandomAffine(degrees=15, shear=10, interpolation=InterpolationMode.NEAREST, fill=int(0)),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(size, interpolation=InterpolationMode.NEAREST)
            ]
        )

        self.val_im_dual_transform = transforms.Compose([transforms.Resize(size, interpolation=InterpolationMode.BILINEAR)])

        self.val_gt_dual_transform = transforms.Compose([transforms.Resize(size, interpolation=InterpolationMode.NEAREST)])

        self.im_normalize_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0.5, std=0.5)])

        self.gt_normalize_transform = transforms.Compose([transforms.ToTensor()])

        self.name_list = name_list

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.data_root, 'Phantom', 'JPEGImages', self.name_list[idx].strip() + '.jpg'))
        mask = Image.open(os.path.join(self.data_root, 'Phantom', 'Annotations', self.name_list[idx].strip() + '.png'))

        if self.split == 'train':
            pairwise_seed = np.random.randint(2147483647)
            reseed(pairwise_seed)
            img = self.img_dual_transform(img)
            reseed(pairwise_seed)
            mask = self.gt_dual_transform(mask)
        else:
            img = self.val_im_dual_transform(img)
            mask = self.val_gt_dual_transform(mask)

        img = self.im_normalize_transform(img)
        mask = self.gt_normalize_transform(mask)
        mask[mask != 0] = 1  # type: ignore

        return {'img': img, 'mask': mask.long()}
