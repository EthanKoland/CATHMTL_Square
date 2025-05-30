import cv2
import numpy as np
import torch

from external.nms import soft_nms
from models.network import ODSNet
from models.networks import *
from models.utils import det_decode, det_post_process


class Inference:

    def __init__(self, config):
        self.config = config
        self.names = class_name
        self.num_classes = len(self.names)
        self.max_per_image = 100
        self.imgs = {}
        self.ipynb = False
        if not self.ipynb:
            import matplotlib.pyplot as plt
            self.plt = plt
        self.pause = True

        self.theme = 'black'
        colors = [(color_list[_]).astype(np.uint8) \
            for _ in range(len(color_list))]
        self.colors = np.array(colors, dtype=np.uint8).reshape(len(colors), 1, 1, 3)
        if self.theme == 'white':
            self.colors = self.colors.reshape(-1)[::-1].reshape(len(colors), 1, 1, 3)
            self.colors = np.clip(self.colors, 0., 0.6 * 255).astype(np.uint8)

        self.model = self.get_model().cuda().eval()
        if config['weights'] is not None:
            src_dict = torch.load(config['weights'], map_location=lambda storage, loc: storage, weights_only=True)
            self.model.load_state_dict(src_dict)
            print(f'Network weight loaded from {config["weights"]}')
        else:
            print('No model loaded.')

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.num_classes + 1):
            results[j] = np.concatenate([detection[j] for detection in detections], axis=0).astype(np.float32)
            soft_nms(results[j], Nt=0.5, method=2)
        scores = np.hstack([results[j][:, 4] for j in range(1, self.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    def step(self, img):
        dets = []
        show_txt = True
        img_id = 'default'
        out = self.model(img)
        hm = out['hm']
        bs = out['dense_bs']
        if bs.shape[1] == 1:
            bs = bs.expand(-1, 2, -1, -1)
        of = out['of']
        det = det_decode(hm, bs, of, K=50)
        det = det.detach().cpu().numpy()
        det = det.reshape(1, -1, det.shape[2])
        c = np.array([self.config['size'][0] / 2., self.config['size'][1] / 2.], dtype=np.float32)
        s = np.array([self.config['size'][0], self.config['size'][1]], dtype=np.float32)
        o_h = self.config['size'][0] / self.config['down_ratio']
        o_w = self.config['size'][0] / self.config['down_ratio']
        scale = 1
        det = det_post_process(det.copy(), [c], [s], o_h, o_w, self.num_classes)
        for j in range(1, self.num_classes + 1):
            det[0][j] = np.array(det[0][j], dtype=np.float32).reshape(-1, 5)
            det[0][j][:, :4] /= scale

        dets.append(det[0])

        dets = self.merge_outputs(dets)
        segs = out['seg_prob']
        return segs, dets

    def get_model(self):
        if self.config['arch'] == 'unet':
            return UNet(in_channels=1, out_channels=self.config['num_classes'], multitask=self.config['multitask'])
        elif self.config['arch'] == 'unet1':
            return U_Net(in_channels=1, out_channels=self.config['num_classes'], multitask=self.config['multitask'])
        elif self.config['arch'] == 'unetplus':
            return ResNet34UnetPlus(num_channels=1, num_class=self.config['num_classes'])
        elif self.config['arch'] == 'attunet':
            return AttU_Net(img_ch=1, output_ch=self.config['num_classes'], multitask=self.config['multitask'])
        elif self.config['arch'] == 'cmunet_v1':
            return CMUNet(img_ch=1, output_ch=self.config['num_classes'], multitask=self.config['multitask'])
        elif self.config['arch'] == 'cmunet_v2':
            return CMUNetv2_CM(img_ch=1, output_ch=self.config['num_classes'], multitask=self.config['multitask'])
        elif self.config['arch'] == 'cmunext':
            return CMUNeXt(input_channel=1, num_classes=self.config['num_classes'], multitask=self.config['multitask'])
        elif self.config['arch'] == 'transunet':
            return TransUnet(img_ch=1, output_ch=self.config['num_classes'])
        elif self.config['arch'] == 'odsnet':
            return ODSNet(in_channels=3, heads=self.config['heads'])
        else:
            raise NotImplementedError


class_name = ['AAA', 'BBB', 'CCC']

color_list = np.array(
    [
        1.000, 1.000, 1.000, 0.850, 0.325, 0.098, 0.929, 0.694, 0.125, 0.494, 0.184, 0.556, 0.466, 0.674, 0.188, 0.301, 0.745,
        0.933, 0.635, 0.078, 0.184, 0.300, 0.300, 0.300, 0.600, 0.600, 0.600, 1.000, 0.000, 0.000, 1.000, 0.500, 0.000, 0.749,
        0.749, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 1.000, 0.667, 0.000, 1.000, 0.333, 0.333, 0.000, 0.333, 0.667, 0.000,
        0.333, 1.000, 0.000, 0.667, 0.333, 0.000, 0.667, 0.667, 0.000, 0.667, 1.000, 0.000, 1.000, 0.333, 0.000, 1.000, 0.667,
        0.000, 1.000, 1.000, 0.000, 0.000, 0.333, 0.500, 0.000, 0.667, 0.500, 0.000, 1.000, 0.500, 0.333, 0.000, 0.500, 0.333,
        0.333, 0.500, 0.333, 0.667, 0.500, 0.333, 1.000, 0.500, 0.667, 0.000, 0.500, 0.667, 0.333, 0.500, 0.667, 0.667, 0.500,
        0.667, 1.000, 0.500, 1.000, 0.000, 0.500, 1.000, 0.333, 0.500, 1.000, 0.667, 0.500, 1.000, 1.000, 0.500, 0.000, 0.333,
        1.000, 0.000, 0.667, 1.000, 0.000, 1.000, 1.000, 0.333, 0.000, 1.000, 0.333, 0.333, 1.000, 0.333, 0.667, 1.000, 0.333,
        1.000, 1.000, 0.667, 0.000, 1.000, 0.667, 0.333, 1.000, 0.667, 0.667, 1.000, 0.667, 1.000, 1.000, 1.000, 0.000, 1.000,
        1.000, 0.333, 1.000, 1.000, 0.667, 1.000, 0.167, 0.000, 0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000,
        0.000, 0.833, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.167, 0.000, 0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000,
        0.667, 0.000, 0.000, 0.833, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.167, 0.000, 0.000, 0.333, 0.000, 0.000, 0.500,
        0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.143, 0.143, 0.143, 0.286, 0.286,
        0.286, 0.429, 0.429, 0.429, 0.571, 0.571, 0.571, 0.714, 0.714, 0.714, 0.857, 0.857, 0.857, 0.000, 0.447, 0.741, 0.50,
        0.5, 0
    ]
).astype(np.float32)
color_list = color_list.reshape((-1, 3)) * 255
