import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from models.cbam import CBAM
from models.utils import _sigmoid


def fill_head_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def aggregate(prob, dim, return_logits=False):
    new_prob = torch.cat([torch.prod(1 - prob, dim=dim, keepdim=True), prob], dim).clamp(1e-7, 1 - 1e-7)
    logits = torch.log((new_prob / (1 - new_prob)))
    prob = F.softmax(logits, dim=dim)

    if return_logits:
        return logits, prob
    else:
        return prob


class ResBlock(nn.Module):

    def __init__(self, indim, outdim=None):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)

        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)

    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))

        if self.downsample is not None:
            x = self.downsample(x)

        return x + r


class UpsampleBlock(nn.Module):

    def __init__(self, skip_c, up_c, out_c, scale_factor=2):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_c, up_c, kernel_size=3, padding=1)
        self.out_conv = ResBlock(up_c, out_c)
        self.scale_factor = scale_factor

    def forward(self, skip_f, up_f):
        x = self.skip_conv(skip_f)
        x = x + F.interpolate(up_f, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        x = self.out_conv(x)
        return x


class Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        # resnet = models.resnet50(pretrained=True)
        resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1  # 1/4, 256
        self.layer2 = resnet.layer2  # 1/8, 512
        self.layer3 = resnet.layer3  # 1/16, 1024

    def forward(self, f):
        x = self.conv1(f)
        x = self.bn1(x)
        x = self.relu(x)  # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        f4 = self.res2(x)  # 1/4, 256
        f8 = self.layer2(f4)  # 1/8, 512
        f16 = self.layer3(f8)  # 1/16, 1024

        return f16, f8, f4


class Decoder(nn.Module):

    def __init__(self, heads):
        super().__init__()
        self.heads = heads

        self.attention = CBAM(1024)
        self.compress = ResBlock(1024, 512)
        self.up_16_8 = UpsampleBlock(512, 512, 256)  # 1/16 -> 1/8
        self.up_8_4 = UpsampleBlock(256, 256, 256)  # 1/8 -> 1/4

        for head in self.heads:
            classes = self.heads[head]
            if 'seg' in head:
                pred_head = nn.Conv2d(256, classes, kernel_size=3, padding=1)
            else:
                pred_head = nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                    nn.Conv2d(256, classes, kernel_size=1)
                )
                if 'hm' in head:
                    pred_head[-1].bias.data.fill_(-2.19)
                else:
                    fill_head_weights(pred_head)
            self.__setattr__(head, pred_head)

    def forward(self, f16, f8, f4):
        x = self.compress(self.attention(f16))
        x = self.up_16_8(f8, x)
        x = self.up_8_4(f4, x)

        out = {}
        for head in self.heads:
            if 'seg' in head:
                logits = F.interpolate(self.__getattr__(head)(F.relu(x)), scale_factor=4, mode='bilinear', align_corners=False)
                logits, prob = aggregate(torch.sigmoid(logits), dim=1, return_logits=True)
                # strip away the background
                prob = prob[:, 1:]
                out[f'{head}_logits'] = logits
                out[f'{head}_prob'] = prob
            # elif 'hm' in head:
            #     out[head] = F.sigmoid(self.__getattr__(head)(x))
            #     # out[head] = _sigmoid(self.__getattr__(head)(x))
            else:
                out[head] = self.__getattr__(head)(x)

        return out
