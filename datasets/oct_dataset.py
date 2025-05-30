import os
import glob

import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from datasets.utils import reseed


class OCTDataset(Dataset):

    def __init__(self, data_root, split='train', size=[128, 128]):
        super().__init__()

        self.split = split

        self.img_list = sorted(glob.glob(os.path.join(data_root, split, 'images', '*.npy')))
        self.mask_list = sorted(glob.glob(os.path.join(data_root, split, 'masks', '*.npy')))

        assert len(self.img_list) == len(self.mask_list)

        # data transform
        self.im_dual_transform = transforms.Compose(
            [
                transforms.RandomAffine(degrees=15, shear=10, interpolation=InterpolationMode.BILINEAR, fill=int(128)),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(size, interpolation=InterpolationMode.BILINEAR)
            ]
        )

        self.gt_dual_transform = transforms.Compose(
            [
                transforms.RandomAffine(degrees=15, shear=10, interpolation=InterpolationMode.NEAREST, fill=0),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(size, interpolation=InterpolationMode.NEAREST)
            ]
        )

        self.val_im_dual_transform = transforms.Compose([transforms.Resize(size, interpolation=InterpolationMode.BILINEAR)])

        self.val_gt_dual_transform = transforms.Compose([transforms.Resize(size, interpolation=InterpolationMode.NEAREST)])

        self.im_normalize_transform = transforms.Compose([transforms.Normalize(128, 128)])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):

        img = np.load(self.img_list[idx])
        mask = np.load(self.mask_list[idx])

        img = img.squeeze()
        mask = mask.squeeze()

        img = torch.Tensor(img).reshape(1, 1, *img.shape)
        mask = torch.Tensor(mask).reshape(1, 1, *mask.shape)
        mask[mask == 8] = 0
        mask[mask == 9] = 8
        mask[mask != 0] = 1

        if self.split == 'train':
            pairwise_seed = np.random.randint(2147483647)
            reseed(pairwise_seed)
            img = self.im_dual_transform(img)
            reseed(pairwise_seed)
            mask = self.gt_dual_transform(mask)
        else:
            img = self.val_im_dual_transform(img)
            mask = self.val_gt_dual_transform(mask)

        img = self.im_normalize_transform(img)

        img = img.squeeze()
        img = img.reshape(1, *img.shape)
        mask = mask.squeeze(1).long()

        return {'img': img, 'mask': mask}
