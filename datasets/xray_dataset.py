import os

import numpy as np
import pandas as pd
from PIL import Image

from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from datasets.utils import reseed, draw_umich_gaussian, draw_dense_reg


class RandomXRayDataset(Dataset):

    def __init__(self, config):
        super(RandomXRayDataset, self).__init__()
        self.config = config
        self.data_root = config['data_root']
        self.split = config['split']
        self.size = config['size']
        self.down_ratio = config['down_ratio']
        self.multitask = config['multitask']
        self.output_size = [self.size[0] // self.down_ratio, self.size[1] // self.down_ratio]
        self.max_objs = 50  # TODO: need to modify by computing the dataset parameters
        self.radius_base = 10

        with open(os.path.join(self.data_root, 'trainval', 'imagesets', self.split + '.txt')) as f:
            name_list = f.readlines()

        # data transform
        self.img_dual_transform = transforms.Compose([transforms.Resize(self.size, interpolation=InterpolationMode.BILINEAR)])

        self.gt_dual_transform = transforms.Compose([transforms.Resize(self.size, interpolation=InterpolationMode.NEAREST)])

        self.val_im_dual_transform = transforms.Compose(
            [transforms.Resize(self.size, interpolation=InterpolationMode.BILINEAR)]
        )

        self.val_gt_dual_transform = transforms.Compose([transforms.Resize(self.size, interpolation=InterpolationMode.NEAREST)])

        self.im_normalize_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0.5, std=0.5)])

        self.gt_normalize_transform = transforms.Compose([transforms.ToTensor()])

        self.name_list = name_list

        if self.multitask:
            bdf = pd.read_csv(os.path.join(self.data_root, 'trainval', 'csv', 'blob.csv'), delimiter=',')
            self.bdf = bdf

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.data_root, 'trainval', 'images',
                                      self.name_list[idx].strip() + '.jpg')).convert('RGB')
        mask = Image.open(os.path.join(self.data_root, 'trainval', 'masks', self.name_list[idx].strip() + '.png'))
        height, width = img.size
        output_h, output_w = self.output_size
        c = np.array([height / 2., width / 2.], dtype=np.float32)
        if self.multitask:
            blob_df = self.bdf.loc[self.bdf['imageId'] == f"{(idx):04d}.jpg"]
            num_objs = len(blob_df)
            hm = np.zeros((self.config['heads']['hm'], self.output_size[0], self.output_size[1]), dtype=np.float32)
            bs = np.zeros((self.max_objs, self.config['heads']['dense_bs']), dtype=np.float32)
            dense_bs = np.zeros((self.config['heads']['dense_bs'], self.output_size[0], self.output_size[1]), dtype=np.float32)
            ind = np.zeros((self.max_objs), dtype=np.int64)
            of = np.zeros((self.max_objs, self.config['heads']['of']), dtype=np.float32)
            of_mask = np.zeros((self.max_objs), dtype=np.uint8)

            for k in range(num_objs):
                tx = blob_df["Center X"].iloc[k]
                ty = blob_df["Center Y"].iloc[k]
                radius = blob_df["Strength"].iloc[k]
                cls_id = 0 if radius < 0.5 else 2 if radius >= 0.7 else 1
                tx = tx * output_w
                ty = ty * output_h
                radius = int(radius * self.radius_base)
                ct = np.array([tx, ty], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_umich_gaussian(hm[cls_id], ct_int, radius)
                bs[k] = 1. * radius
                ind[k] = ct_int[1] * output_w + ct_int[0]
                of[k] = ct - ct_int
                of_mask[k] = 1
                draw_dense_reg(dense_bs, hm.max(axis=0), ct_int, bs[k], radius)
                if dense_bs is not None:
                    dense_bs_mask = hm.max(axis=0, keepdims=True)

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
        mask[mask < 0.5] = 0
        mask[mask != 0] = 1
        if self.multitask:
            return {
                'img': img,
                'mask': mask.long(),
                'hm': hm,
                'bs': bs,
                'dense_bs': dense_bs,
                'dense_bs_mask': dense_bs_mask,
                'ind': ind,
                'of': of,
                'of_mask': of_mask
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
